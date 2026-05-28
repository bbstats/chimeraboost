"""ChimeraBoost benchmark harness.

Runs ChimeraBoost against whatever competitors are installed (scikit-learn
HistGradientBoosting is always available; CatBoost, XGBoost and LightGBM are
auto-detected and skipped if absent) across a fixed suite of regression and
classification tasks, including categorical-heavy ones.

Every task is run over multiple seeds and reported as mean +/- std so that a
real improvement can be told apart from noise. This is the tool we use to
decide whether any future change (ordered boosting, feature combinations, ...)
actually helps before it goes in.

Usage:
    python benchmarks/run_benchmarks.py                 # default scale
    python benchmarks/run_benchmarks.py --scale 3       # ~3x bigger datasets
    python benchmarks/run_benchmarks.py --seeds 10      # more seeds
    python benchmarks/run_benchmarks.py --only classification
    python benchmarks/run_benchmarks.py --threads 8     # ChimeraBoost threads
"""

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score, log_loss, f1_score,
)
import json as _json
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


# --------------------------------------------------------------------------
# Optional competitors: detected once, skipped silently if not installed.
# --------------------------------------------------------------------------
def _detect():
    have = {}
    try:
        import catboost  # noqa
        have["catboost"] = True
    except Exception:
        have["catboost"] = False
    try:
        import xgboost  # noqa
        have["xgboost"] = True
    except Exception:
        have["xgboost"] = False
    try:
        import lightgbm  # noqa
        have["lightgbm"] = True
    except Exception:
        have["lightgbm"] = False
    return have


HAVE = _detect()


# --------------------------------------------------------------------------
# Dataset builders. Each returns (X, y, cat_features, task).
# `scale` multiplies the synthetic sample counts.
# --------------------------------------------------------------------------
def _ds_diabetes(scale, rng):
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    return X, y, None, "regression"


def _ds_friedman(scale, rng):
    from sklearn.datasets import make_friedman1
    n = int(2000 * scale)
    X, y = make_friedman1(n_samples=n, noise=1.0, random_state=int(rng.integers(1e9)))
    return X, y, None, "regression"


def _ds_synthetic_reg(scale, rng):
    from sklearn.datasets import make_regression
    n = int(8000 * scale)
    X, y = make_regression(n_samples=n, n_features=30, n_informative=20,
                           noise=20.0, random_state=int(rng.integers(1e9)))
    return X, y, None, "regression"


def _ds_breast_cancer(scale, rng):
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, None, "binary"


def _ds_wine(scale, rng):
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    return X, y, None, "multiclass"


def _ds_categorical_binary(scale, rng):
    """High-cardinality + low-cardinality categoricals driving a binary target."""
    n = int(6000 * scale)
    hi = rng.integers(0, 150, n)              # high-card categorical
    lo = rng.integers(0, 5, n)                # low-card categorical
    num = rng.normal(size=(n, 3))
    hi_eff = rng.normal(0, 1.5, 150)[hi]
    lo_eff = np.array([-1.0, -0.3, 0.2, 0.8, 1.5])[lo]
    logit = hi_eff + lo_eff + 0.6 * num[:, 0] - 0.4 * num[:, 1] + rng.normal(0, 1, n)
    y = (logit > np.median(logit)).astype(int)
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = np.array([f"h{c}" for c in hi], dtype=object)
    X[:, 1] = np.array([f"l{c}" for c in lo], dtype=object)
    X[:, 2:] = num
    return X, y, [0, 1], "binary"


def _ds_categorical_multiclass(scale, rng):
    n = int(5000 * scale)
    region = rng.choice(["N", "S", "E", "W"], n)
    tier = rng.choice(["a", "b", "c"], n)
    num = rng.normal(size=(n, 3))
    score = (np.select([region == "N", region == "S", region == "E"],
                       [1.5, -1.0, 0.3], 0.0)
             + np.select([tier == "a", tier == "b"], [1.0, -0.5], 0.0)
             + 0.5 * num[:, 0] + rng.normal(0, 0.5, n))
    y = np.digitize(score, [-0.5, 1.0])
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = region
    X[:, 1] = tier
    X[:, 2:] = num
    return X, y, [0, 1], "multiclass"


DATASETS = {
    "diabetes": _ds_diabetes,
    "friedman1": _ds_friedman,
    "synthetic_reg": _ds_synthetic_reg,
    "breast_cancer": _ds_breast_cancer,
    "wine": _ds_wine,
    "cat_binary": _ds_categorical_binary,
    "cat_multiclass": _ds_categorical_multiclass,
}


# --------------------------------------------------------------------------
# Real external datasets via OpenML (the standard tabular-ML benchmark repo).
# These are fetched on demand with --openml and cached by sklearn. They exist
# so that decisions about defaults rest on many real datasets, not a handful of
# synthetic ones hand-picked here. Each entry: (openml_name_or_id, task,
# cat_feature_indices_or_None). Categoricals are auto-detected from dtype when
# the index list is "auto".
#
# This list is intentionally broad and editable -- add the datasets you care
# about. IDs are OpenML dataset IDs (stable); names can drift, so IDs preferred.
# --------------------------------------------------------------------------
OPENML_SUITE = {
    # classification (binary)
    "credit-g":      dict(data_id=31,    task="binary",     cats="auto"),
    "adult":         dict(data_id=1590,  task="binary",     cats="auto"),
    "bank-marketing":dict(data_id=1461,  task="binary",     cats="auto"),
    "kc1":           dict(data_id=1067,  task="binary",     cats=None),
    "phoneme":       dict(data_id=1489,  task="binary",     cats=None),
    "electricity":   dict(data_id=151,   task="binary",     cats=None),
    "magic":         dict(data_id=1120,  task="binary",     cats=None),
    "spambase":      dict(data_id=44,    task="binary",     cats=None),
    "kc2":           dict(data_id=1063,  task="binary",     cats=None),
    "sick":          dict(data_id=38,    task="binary",     cats="auto"),
    # classification (multiclass)
    "vehicle":       dict(data_id=54,    task="multiclass", cats=None),
    "segment":       dict(data_id=40984, task="multiclass", cats=None),
    "optdigits":     dict(data_id=28,    task="multiclass", cats=None),
    "car":           dict(data_id=40975, task="multiclass", cats="auto"),
    # regression
    "cpu_act":       dict(data_id=197,   task="regression", cats=None),
    "wine_quality":  dict(data_id=287,   task="regression", cats=None),
    "boston":        dict(data_id=531,   task="regression", cats=None),
    "elevators":     dict(data_id=216,   task="regression", cats=None),
    "ailerons":      dict(data_id=296,   task="regression", cats=None),
    "abalone":       dict(data_id=183,   task="regression", cats="auto"),
}


def _make_openml_builder(spec):
    """Build a dataset-builder closure for one OpenML spec."""
    def builder(scale, rng):
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=spec["data_id"], as_frame=True)
        df = ds.frame
        target_col = ds.target.name
        y = ds.target
        X_df = df.drop(columns=[target_col])

        # Detect categoricals from dtype if requested.
        if spec["cats"] == "auto":
            def _is_cat(dtype):
                s = str(dtype).lower()
                return s in ("category", "object") or s.startswith("string")
            cat_idx = [i for i, c in enumerate(X_df.columns) if _is_cat(X_df[c].dtype)]
        else:
            cat_idx = spec["cats"]

        task = spec["task"]
        # Encode target for classification; coerce to float for regression.
        if task == "regression":
            y = y.astype(float).to_numpy()
        else:
            y = y.astype("category").cat.codes.to_numpy()

        if cat_idx:
            # Categorical columns: NaN -> "__nan__" string. CatBoost rejects
            # float NaN in cat_features ("must be integer or string"), and
            # ChimeraBoost already maps the "__nan__" label to its missing bucket
            # in factorize(), so both see missing the same way.
            cat_cols = set(cat_idx)
            cols = []
            for i, c in enumerate(X_df.columns):
                s = X_df[c]
                if i in cat_cols:
                    cols.append(s.astype(object).where(s.notna(), "__nan__"))
                else:
                    cols.append(s.astype(float))   # keep numerics numeric
            import pandas as pd
            X = pd.concat(cols, axis=1).to_numpy(dtype=object)
        else:
            X = X_df.to_numpy(dtype=float)
        return X, y, (cat_idx or None), task
    return builder


def _add_openml_datasets():
    for name, spec in OPENML_SUITE.items():
        DATASETS[f"oml:{name}"] = _make_openml_builder(spec)


# --------------------------------------------------------------------------
# Model runners. Each returns (metrics_dict, fit_seconds, best_iter). The
# metrics dict always includes "primary" (higher=better; -RMSE for regression,
# F1-macro for classification) which the summary/sign-test logic uses. For
# classification it also includes "log_loss" so table cuts can report both.
# Returns None if the model can't run the task (e.g. competitor without
# native categorical support we skip).
# --------------------------------------------------------------------------
def _compute_metrics(task, y_true, model, X_test):
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, model.predict(X_test))))
        return {"primary": -rmse, "rmse": rmse}
    f1 = float(f1_score(y_true, model.predict(X_test), average="macro"))
    proba = model.predict_proba(X_test)
    # log_loss needs labels= for safety when a class is missing from y_true.
    ll = float(log_loss(y_true, proba, labels=np.unique(np.concatenate([y_true]))))
    return {"primary": f1, "f1_macro": f1, "log_loss": ll}


def _val_split(Xtr, ytr, task, seed):
    """Carve an internal validation set from training data for early stopping.
    Never touches the test set."""
    strat = ytr if task != "regression" else None
    return train_test_split(Xtr, ytr, test_size=0.2, random_state=seed,
                            stratify=strat)


# Shared early-stopping budget for every model, so the comparison is fair.
MAX_ITERS = 2000
PATIENCE = 50


def _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads, lr=None,
                 ordered_boosting=True, depth=6, subsample=1.0, mcw=1.0,
                 cat_combinations=False):
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    m = Est(iterations=MAX_ITERS, early_stopping_rounds=PATIENCE,
            learning_rate=lr, depth=depth, ordered_boosting=ordered_boosting,
            subsample=subsample, min_child_weight=mcw,
            cat_combinations=cat_combinations,
            thread_count=threads, random_state=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return _compute_metrics(task, yte, m, Xte), time.time() - t, m.best_iteration_


def _run_sklearn(task, Xtr, ytr, Xte, yte, cat, threads):
    """sklearn HGB with native categorical support.

    HGB requires integer-encoded categoricals; we ordinal-encode them here so
    the comparison is fair (same information given to all models).
    """
    from sklearn.preprocessing import OrdinalEncoder
    t = time.time()
    if cat is not None:
        cat_idx = list(cat)
        Xtr = np.array(Xtr, dtype=object)
        Xte = np.array(Xte, dtype=object)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[:, cat_idx])
        Xtr[:, cat_idx] = enc.transform(Xtr[:, cat_idx])
        Xte[:, cat_idx] = enc.transform(Xte[:, cat_idx])
        Xtr = Xtr.astype(float)
        Xte = Xte.astype(float)
    else:
        cat_idx = None
    # HGB has built-in early stopping via a validation fraction.
    common = dict(max_iter=MAX_ITERS, early_stopping=True,
                  validation_fraction=0.2, n_iter_no_change=PATIENCE,
                  categorical_features=cat_idx,
                  random_state=0)
    Est = (HistGradientBoostingRegressor if task == "regression"
           else HistGradientBoostingClassifier)
    m = Est(**common)
    m.fit(Xtr, ytr)
    return _compute_metrics(task, yte, m, Xte), time.time() - t, m.n_iter_


def _run_catboost(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["catboost"]:
        return None
    from catboost import CatBoostRegressor, CatBoostClassifier
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(iterations=MAX_ITERS, early_stopping_rounds=PATIENCE,
                  thread_count=threads or -1, verbose=False, random_seed=0)
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = Est(**common)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return _compute_metrics(task, yte, m, Xte), time.time() - t, m.best_iteration_


def _xgb_dataframes(Xtr, Xval, Xte, cat_idx):
    """Build three pandas DataFrames sharing the same category sets per cat
    column. Categories absent from training become NaN at predict-time, which
    XGBoost handles via its default missing direction. This avoids the
    'category not in the training set' error on unseen values."""
    import pandas as pd
    cat_set = set(cat_idx)

    def _to_df(X):
        df = pd.DataFrame(X)
        for i in range(df.shape[1]):
            if i not in cat_set:
                df[i] = pd.to_numeric(df[i], errors="coerce")
        return df

    df_tr = _to_df(Xtr)
    df_va = _to_df(Xval)
    df_te = _to_df(Xte)
    for i in cat_idx:
        df_tr[i] = df_tr[i].astype("category")
        cats = df_tr[i].cat.categories
        df_va[i] = pd.Categorical(df_va[i], categories=cats)
        df_te[i] = pd.Categorical(df_te[i], categories=cats)
    return df_tr, df_va, df_te


def _run_xgboost(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["xgboost"]:
        return None
    import xgboost as xgb
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
                  n_jobs=threads or -1, random_state=0, verbosity=0,
                  tree_method="hist")
    if cat is not None:
        common["enable_categorical"] = True
        Xf_in, Xv_in, Xte_in = _xgb_dataframes(Xf, Xv, Xte, list(cat))
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte
    Est = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    m = Est(**common)
    m.fit(Xf_in, yf, eval_set=[(Xv_in, yv)], verbose=False)
    best = getattr(m, "best_iteration", None)
    return _compute_metrics(task, yte, m, Xte_in), time.time() - t, best


def _lgb_prepare(Xtr, Xval, Xte, cat_idx):
    """Ordinal-encode the cat columns to ints using a single encoder fit on
    training. Validation and test reuse it; unseen categories become -1.
    LightGBM expects integer-coded categoricals when categorical_feature is set.
    """
    from sklearn.preprocessing import OrdinalEncoder
    Xtr = np.array(Xtr, dtype=object)
    Xval = np.array(Xval, dtype=object)
    Xte = np.array(Xte, dtype=object)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[:, cat_idx])
    Xtr[:, cat_idx] = enc.transform(Xtr[:, cat_idx])
    Xval[:, cat_idx] = enc.transform(Xval[:, cat_idx])
    Xte[:, cat_idx] = enc.transform(Xte[:, cat_idx])
    return Xtr.astype(float), Xval.astype(float), Xte.astype(float)


def _run_lightgbm(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["lightgbm"]:
        return None
    import lightgbm as lgb
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=MAX_ITERS, n_jobs=threads or -1,
                  random_state=0, verbosity=-1)
    fit_kw = dict(callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    if cat is not None:
        Xf_in, Xv_in, Xte_in = _lgb_prepare(Xf, Xv, Xte, list(cat))
        fit_kw["categorical_feature"] = list(cat)
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte
    fit_kw["eval_set"] = [(Xv_in, yv)]
    Est = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    m = Est(**common)
    m.fit(Xf_in, yf, **fit_kw)
    return _compute_metrics(task, yte, m, Xte_in), time.time() - t, m.best_iteration_


RUNNERS = {
    "ChimeraBoost": _run_chimera,
    "sklearn_HGB": _run_sklearn,
    "CatBoost": _run_catboost,
    "XGBoost": _run_xgboost,
    "LightGBM": _run_lightgbm,
}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def _rel_gap(ours, theirs, task):
    """Relative gap of ChimeraBoost vs a competitor, as a signed percentage
    where POSITIVE means ChimeraBoost is better.

    Regression score is -RMSE (higher=better), classification is F1 macro
    (higher=better), so in both cases higher is better and the formula is the
    same once we work in the 'higher=better' space.
    """
    # convert to higher-is-better magnitude
    if task == "regression":
        o, t = -ours, -theirs          # RMSE magnitudes (lower better)
        # improvement = how much smaller our RMSE is
        return 100.0 * (t - o) / t
    else:
        return 100.0 * (ours - theirs) / theirs


def main():
    global PATIENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier for synthetic dataset sizes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None,
                    help="ChimeraBoost thread_count (None = all cores)")
    ap.add_argument("--only", choices=["regression", "classification"],
                    default=None)
    ap.add_argument("--openml", action="store_true",
                    help="include real OpenML benchmark datasets (downloads + caches)")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="run ONLY the OpenML datasets (implies --openml)")
    ap.add_argument("--models", nargs="+", default=None,
                    metavar="MODEL",
                    help=("limit to specific runners, e.g. "
                          "--models ChimeraBoost CatBoost sklearn_HGB. "
                          f"Available: {list(RUNNERS)}"))
    ap.add_argument("--lr", type=float, default=None,
                    help=("override ChimeraBoost learning rate (default: auto=0.1 "
                          "with early stopping). Try --lr 0.05 to test whether "
                          "smaller steps + more trees improve accuracy on OpenML "
                          "before promoting to a new default."))
    ap.add_argument("--chimera-depth", type=int, default=6,
                    help=("override ChimeraBoost tree depth (default: 6). Use "
                          "--chimera-depth 8 to A/B whether deeper trees close "
                          "the bias gap on numeric-heavy datasets."))
    ap.add_argument("--patience", type=int, default=None,
                    help=("override early stopping patience rounds for ALL models "
                          "(default: PATIENCE=%d). Higher values let more trees "
                          "accumulate before stopping." % PATIENCE))
    ap.add_argument("--no-ordered-boosting", dest="ordered_boosting",
                    action="store_false", default=True,
                    help=("disable LOO leaf correction in ChimeraBoost "
                          "(default: on). Use to A/B test the improvement."))
    ap.add_argument("--chimera-subsample", type=float, default=1.0,
                    dest="chimera_subsample",
                    help=("ChimeraBoost row subsample fraction (default: 1.0=off). "
                          "Uses MVS (gradient-weighted) sampling when < 1. "
                          "Try 0.8 to test whether stochastic boosting helps."))
    ap.add_argument("--chimera-mcw", type=float, default=1.0,
                    dest="chimera_mcw",
                    help=("ChimeraBoost min_child_weight (default: 1.0). "
                          "Try 0.5 or 0.0 to see whether relaxing the hessian "
                          "constraint allows better splits on small datasets."))
    ap.add_argument("--chimera-cat-combinations", action="store_true",
                    default=False, dest="cat_combinations",
                    help=("Enable 2-way categorical feature combinations in "
                          "ChimeraBoost (default: off). Generates all C(n_cat,2) "
                          "pairwise combo columns before target encoding, mirroring "
                          "CatBoost's feature combination step."))
    ap.add_argument("--datasets", nargs="+", default=None,
                    metavar="DS",
                    help=("run only these datasets, e.g. --datasets diabetes "
                          "oml:phoneme boston. Names must match keys in DATASETS "
                          "(after --openml datasets are added)."))
    ap.add_argument("--save", nargs="?", const="auto", default=None,
                    metavar="PATH",
                    help=("Also write the full benchmark output to a file. "
                          "Pass a path, or no argument for a timestamped file "
                          "under benchmarks/results/."))
    args = ap.parse_args()

    # Optional tee: mirror stdout to a results file so runs are inspectable
    # later. Default location is benchmarks/results/YYYYMMDD-HHMMSS.txt.
    tee = None
    if args.save is not None:
        import os, sys, datetime
        if args.save == "auto":
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(results_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(results_dir, f"{stamp}.txt")
        else:
            save_path = args.save
        tee_file = open(save_path, "w", encoding="utf-8")
        real_stdout = sys.stdout
        class _Tee:
            def write(self, s):
                real_stdout.write(s); tee_file.write(s); tee_file.flush()
            def flush(self):
                real_stdout.flush(); tee_file.flush()
        sys.stdout = _Tee()
        tee = (tee_file, save_path)
        print(f"# Benchmark results will be saved to: {save_path}")

    if args.openml or args.no_synthetic or (
            args.datasets and any(d.startswith("oml:") for d in (args.datasets or []))):
        _add_openml_datasets()
    if args.no_synthetic:
        for k in [k for k in DATASETS if not k.startswith("oml:")]:
            del DATASETS[k]

    # Apply patience override globally before building runner dicts.
    if args.patience is not None:
        PATIENCE = args.patience

    # Build the runner dict; ChimeraBoost gets the lr override if provided.
    import functools
    active_runners = dict(RUNNERS)
    active_runners["ChimeraBoost"] = functools.partial(
        _run_chimera, lr=args.lr, ordered_boosting=args.ordered_boosting,
        depth=args.chimera_depth, subsample=args.chimera_subsample,
        mcw=args.chimera_mcw, cat_combinations=args.cat_combinations,
    )
    if args.models:
        unknown = set(args.models) - set(active_runners)
        if unknown:
            ap.error(f"Unknown models: {unknown}. Available: {list(active_runners)}")
        active_runners = {k: v for k, v in active_runners.items()
                         if k in args.models}

    print("Detected competitors:",
          ", ".join(k for k, v in HAVE.items() if v) or "none (sklearn only)")
    print(f"scale={args.scale}  seeds={args.seeds}  "
          f"threads={args.threads or 'all'}  "
          f"early stopping: max_iter={MAX_ITERS}, patience={PATIENCE}"
          + (f"  chimera_lr={args.lr}" if args.lr else "")
          + (f"  ordered_boosting={args.ordered_boosting}")
          + (f"  subsample={args.chimera_subsample}" if args.chimera_subsample < 1.0 else "")
          + (f"  cat_combinations=True" if args.cat_combinations else "")
          + (f"  models={args.models}" if args.models else "")
          + "\n")

    metric_name = {"regression": "RMSE (lower better)",
                   "binary": "F1 macro (higher better)",
                   "multiclass": "F1 macro (higher better)"}

    # accumulate per-competitor relative gaps + speed ratios across datasets
    gap_acc = {r: [] for r in active_runners if r != "ChimeraBoost"}
    speed_acc = {r: [] for r in active_runners if r != "ChimeraBoost"}

    # Raw per-(dataset, model, seed) records for downstream table generation.
    # Captures every metric (incl. log_loss for classification) and per-seed
    # fit time, so the table script can slice by task / cats / size buckets.
    raw_records = []
    dataset_meta = {}

    for ds_name, builder in DATASETS.items():
        if args.datasets and ds_name not in args.datasets:
            continue
        _, _, _, task = builder(args.scale, np.random.default_rng(0))
        if args.only == "regression" and task != "regression":
            continue
        if args.only == "classification" and task == "regression":
            continue

        results = {r: [] for r in active_runners}     # primary score per seed
        metrics_all = {r: [] for r in active_runners} # full dict per seed
        times = {r: [] for r in active_runners}
        iters = {r: [] for r in active_runners}
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            X, y, cat, task = builder(args.scale, rng)
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=s, stratify=strat
            )
            if ds_name not in dataset_meta:
                dataset_meta[ds_name] = {
                    "task": task,
                    "n_train": int(Xtr.shape[0]),
                    "n_total": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "has_cats": bool(cat),
                }
            for rname, runner in active_runners.items():
                out = runner(task, Xtr, ytr, Xte, yte, cat, args.threads)
                if out is not None:
                    metrics, secs, best_it = out
                    results[rname].append(metrics["primary"])
                    metrics_all[rname].append(metrics)
                    times[rname].append(secs)
                    if best_it is not None:
                        iters[rname].append(best_it)
                    raw_records.append({
                        "dataset": ds_name, "model": rname, "seed": s,
                        "metrics": metrics, "fit_time": secs,
                        "best_iter": int(best_it) if best_it is not None else None,
                    })

        print(f"### {ds_name}  [{task}]  metric={metric_name[task]}")
        for rname in active_runners:
            if not results[rname]:
                continue
            sc = np.array(results[rname])
            tm = np.array(times[rname])
            disp = (-sc if task == "regression" else sc)
            it_str = (f"  trees~{int(np.mean(iters[rname]))}"
                      if iters[rname] else "")
            star = " <-- ours" if rname == "ChimeraBoost" else ""
            print(f"  {rname:14s} {disp.mean():8.4f} +/- {disp.std():.4f}"
                  f"   fit {tm.mean():6.2f}s{it_str}{star}")

        # record relative gaps for the summary
        if results["ChimeraBoost"]:
            our_score = np.mean(results["ChimeraBoost"])
            our_time = np.mean(times["ChimeraBoost"])
            for rname in gap_acc:
                if results[rname]:
                    gap_acc[rname].append(
                        _rel_gap(our_score, np.mean(results[rname]), task))
                    speed_acc[rname].append(
                        np.mean(times[rname]) / max(our_time, 1e-9))
        print()

    # ---- summary verdict ----
    print("=" * 64)
    print("SUMMARY (averaged over datasets; + = ChimeraBoost better)")
    print("=" * 64)
    for rname in gap_acc:
        if not gap_acc[rname]:
            continue
        g = np.array(gap_acc[rname])
        sp = np.array(speed_acc[rname])
        # speed ratio >1 means ChimeraBoost is faster
        wins = int(np.sum(g > 0))
        verdict = _verdict(rname, g.mean())
        print(f"  vs {rname:12s}  F1 macro {g.mean():+6.2f}% "
              f"(wins {wins}/{len(g)})   speed x{sp.mean():.2f}   -> {verdict}")
    print()
    if tee is not None:
        import sys
        # Sidecar JSON: every metric for every (dataset, model, seed), plus
        # dataset metadata (task, size, has_cats). Used by make_tables.py.
        json_path = tee[1].rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            _json.dump({
                "config": {
                    "seeds": args.seeds, "max_iters": MAX_ITERS,
                    "patience": PATIENCE,
                },
                "datasets": dataset_meta,
                "records": raw_records,
            }, jf, indent=2)
        print(f"# Saved results to: {tee[1]}")
        print(f"# Saved raw data to: {json_path}")
        sys.stdout = real_stdout
        tee[0].close()


def _verdict(competitor, mean_gap):
    if competitor == "sklearn_HGB":
        return "PASS: beats sklearn" if mean_gap > 0 else "FAIL: must beat sklearn"
    if competitor == "CatBoost":
        if mean_gap >= 0:
            return "PASS: matches/beats CatBoost"
        if mean_gap > -3.0:
            return "PASS: within 3% of CatBoost (close, on average)"
        return f"GAP: {-mean_gap:.1f}% behind CatBoost on average"
    return "better" if mean_gap > 0 else "behind"


if __name__ == "__main__":
    main()