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
import json as _json
import os
import time
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, f1_score
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

# Task type per synthetic dataset, so selection/filtering needn't build them.
SYNTH_TASKS = {
    "diabetes": "regression", "friedman1": "regression",
    "synthetic_reg": "regression", "breast_cancer": "binary",
    "wine": "multiclass", "cat_binary": "binary", "cat_multiclass": "multiclass",
}


def _task_of(ds_name):
    """Task type of a dataset by name, without building it."""
    if ds_name.startswith("oml:"):
        return OPENML_SUITE[ds_name[4:]]["task"]
    if ds_name.startswith("gr:"):
        return GRINSZTAJN_TASKS[ds_name]
    return SYNTH_TASKS[ds_name]


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
    "mushroom":      dict(data_id=24,    task="binary",     cats="auto"),
    "kr-vs-kp":      dict(data_id=3,     task="binary",     cats="auto"),
    # classification (multiclass)
    "vehicle":       dict(data_id=54,    task="multiclass", cats=None),
    "segment":       dict(data_id=40984, task="multiclass", cats=None),
    "optdigits":     dict(data_id=28,    task="multiclass", cats=None),
    "car":           dict(data_id=40975, task="multiclass", cats="auto"),
    "splice":        dict(data_id=46,    task="multiclass", cats="auto"),
    "satimage":      dict(data_id=182,   task="multiclass", cats=None),
    "pendigits":     dict(data_id=32,    task="multiclass", cats=None),
    "letter":        dict(data_id=6,     task="multiclass", cats=None),
    # regression
    "cpu_act":       dict(data_id=197,   task="regression", cats=None),
    "wine_quality":  dict(data_id=287,   task="regression", cats=None),
    "boston":        dict(data_id=531,   task="regression", cats=None),
    "elevators":     dict(data_id=216,   task="regression", cats=None),
    "ailerons":      dict(data_id=296,   task="regression", cats=None),
    "abalone":       dict(data_id=183,   task="regression", cats="auto"),
    "house_16H":     dict(data_id=574,   task="regression", cats=None),
}


def _frame_to_dataset(X_df, y, cats, task):
    """Turn a (features DataFrame, target Series) into (X, y, cat_idx, task).
    Shared by the OpenML and Grinsztajn/HuggingFace builders.

    `cats` is "auto" (detect object/category/string columns), an explicit index
    list, or None. Categorical NaNs become the "__nan__" string: CatBoost rejects
    float NaN in cat_features, and ChimeraBoost maps "__nan__" to its missing
    bucket, so both see missing the same way. Numerics stay float.
    """
    if cats == "auto":
        def _is_cat(dtype):
            s = str(dtype).lower()
            return s in ("category", "object") or s.startswith("string")
        cat_idx = [i for i, c in enumerate(X_df.columns) if _is_cat(X_df[c].dtype)]
    else:
        cat_idx = cats

    if task == "regression":
        y = y.astype(float).to_numpy()
    else:
        y = y.astype("category").cat.codes.to_numpy()

    if cat_idx:
        import pandas as pd
        cat_cols = set(cat_idx)
        cols = []
        for i, c in enumerate(X_df.columns):
            s = X_df[c]
            if i in cat_cols:
                cols.append(s.astype(object).where(s.notna(), "__nan__"))
            else:
                cols.append(s.astype(float))
        X = pd.concat(cols, axis=1).to_numpy(dtype=object)
    else:
        X = X_df.to_numpy(dtype=float)
    return X, y, (cat_idx or None), task


def _make_openml_builder(spec):
    """Build a dataset-builder closure for one OpenML spec (fetched by data_id)."""
    def builder(scale, rng):
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=spec["data_id"], as_frame=True)
        X_df = ds.frame.drop(columns=[ds.target.name])
        return _frame_to_dataset(X_df, ds.target, spec["cats"], spec["task"])
    return builder


def _add_openml_datasets():
    for name, spec in OPENML_SUITE.items():
        DATASETS[f"oml:{name}"] = _make_openml_builder(spec)


# The Grinsztajn et al. 2022 tabular benchmark ("Why do tree-based models still
# outperform deep learning on tabular data?"), the standard reference for this
# question. Loaded from the official inria-soda HuggingFace mirror (the exact
# transformed CSVs from the paper) rather than OpenML's flaky `study` API, so the
# dataset membership is hardcoded below and only the working CSV download is
# needed. Binary-classification + regression only (no multiclass). The target is
# always the last column. Folder -> (task, has-categoricals).
GRINSZTAJN_HF = ("https://huggingface.co/datasets/inria-soda/tabular-benchmark/"
                 "resolve/main")
GRINSZTAJN_FOLDERS = {
    "clf_num": ("binary", False),
    "clf_cat": ("binary", True),
    "reg_num": ("regression", False),
    "reg_cat": ("regression", True),
}
# Dataset membership per folder (from the HuggingFace mirror's file tree). Names
# repeat across folders (e.g. electricity is in both clf_num and clf_cat, with
# different feature sets), so the folder is part of the DATASETS key.
GRINSZTAJN_DATASETS = {
    "clf_num": ["Bioresponse", "Diabetes130US", "Higgs", "MagicTelescope",
                "MiniBooNE", "bank-marketing", "california", "covertype",
                "credit", "default-of-credit-card-clients", "electricity",
                "eye_movements", "heloc", "house_16H", "jannis", "pol"],
    "clf_cat": ["albert", "compas-two-years", "covertype",
                "default-of-credit-card-clients", "electricity",
                "eye_movements", "road-safety"],
    "reg_num": ["Ailerons", "Bike_Sharing_Demand", "Brazilian_houses",
                "MiamiHousing2016", "abalone", "cpu_act",
                "delays_zurich_transport", "diamonds", "elevators", "house_16H",
                "house_sales", "houses", "medical_charges",
                "nyc-taxi-green-dec-2016", "pol", "sulfur", "superconduct",
                "wine_quality", "yprop_4_1"],
    "reg_cat": ["Airlines_DepDelay_1M", "Allstate_Claims_Severity",
                "Bike_Sharing_Demand", "Brazilian_houses",
                "Mercedes_Benz_Greener_Manufacturing",
                "SGEMM_GPU_kernel_performance", "abalone", "analcatdata_supreme",
                "delays_zurich_transport", "diamonds", "house_sales",
                "medical_charges", "nyc-taxi-green-dec-2016",
                "particulate-matter-ukair-2017", "seattlecrime6", "topo_2_1",
                "visualizing_soil"],
}
GRINSZTAJN_TASKS = {}   # "gr:<folder>/<name>" -> task, filled at registration
# Cap rows so the largest datasets (Higgs, nyc-taxi, ...) stay tractable, in the
# spirit of the paper's size caps. Seeded subsample for reproducibility.
_GRINSZTAJN_MAX_ROWS = 50000


def _make_grinsztajn_builder(folder, name, task, has_cats):
    def builder(scale, rng):
        import pandas as pd
        df = pd.read_csv(f"{GRINSZTAJN_HF}/{folder}/{name}.csv")
        if len(df) > _GRINSZTAJN_MAX_ROWS:
            df = df.sample(_GRINSZTAJN_MAX_ROWS, random_state=0).reset_index(drop=True)
        return _frame_to_dataset(df.iloc[:, :-1], df.iloc[:, -1],
                                 "auto" if has_cats else None, task)
    return builder


def _add_grinsztajn_datasets():
    """Register the Grinsztajn benchmark (HuggingFace mirror) into DATASETS as
    gr:<folder>/<name>. Idempotent so workers can call it once cheaply."""
    if any(k.startswith("gr:") for k in DATASETS):
        return
    for folder, (task, has_cats) in GRINSZTAJN_FOLDERS.items():
        for name in GRINSZTAJN_DATASETS[folder]:
            key = f"gr:{folder}/{name}"
            DATASETS[key] = _make_grinsztajn_builder(folder, name, task, has_cats)
            GRINSZTAJN_TASKS[key] = task


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
    classes = getattr(model, "classes_", np.unique(y_true))
    # log_loss needs labels= for safety when a class is missing from y_true.
    ll = float(log_loss(y_true, proba, labels=classes))
    # Multiclass Brier: mean over samples of sum_k (p_k - onehot_k)^2. Bounded,
    # outlier-robust, and a proper scoring rule like log loss, but it aggregates
    # far more stably across datasets (no unbounded tail). Used for binary too
    # (the K=2 sum form), so the two tasks share one definition.
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    # Miscalibration (MCB), the CORP / Dimitriadis-Gneiting-Jordan calibration
    # measure: how much a *monotone* recalibration improves the (per-class) Brier
    # score. Refit each class's probabilities to its outcomes with ascending,
    # clipped isotonic regression (the optimal calibration map) and take the gap
    # MCB_k = Brier_k(p_k) - Brier_k(isotonic(p_k)); average over classes. 0 means
    # already perfectly calibrated, higher = worse. In-sample isotonic on the test
    # fold is the standard CORP diagnostic. Binary -> just the class-1 curve.
    from sklearn.isotonic import IsotonicRegression
    mcb_k = []
    for k in range(proba.shape[1]):
        p = proba[:, k]
        yk = onehot[:, k]
        recal = IsotonicRegression(increasing=True, out_of_bounds="clip"
                                   ).fit_transform(p, yk)
        mcb_k.append(np.mean((p - yk) ** 2) - np.mean((recal - yk) ** 2))
    mcb = float(np.mean(mcb_k))
    return {"primary": f1, "f1_macro": f1, "log_loss": ll, "brier": brier,
            "calibration_mcb": mcb}


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
                 ordered_boosting=None, depth=6, subsample=1.0, mcw=None,
                 cat_combinations=False, leaf_estimation_iterations=None,
                 linear_leaves=False, linear_lambda=1.0):
    t = time.time()
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    # None = use the class default. For ordered_boosting that's False (Reg) /
    # False (Clf); for min_child_weight it's the classifier's size-adaptive auto.
    # An explicit value overrides (e.g. --no-ordered-boosting, --chimera-mcw 0).
    kw = {} if ordered_boosting is None else {"ordered_boosting": ordered_boosting}
    if leaf_estimation_iterations is not None:
        kw["leaf_estimation_iterations"] = leaf_estimation_iterations
    if mcw is not None:
        kw["min_child_weight"] = mcw
    if linear_leaves:
        # multiclass doesn't support linear leaves yet; fall back to constant
        # leaves there so a full-suite run doesn't crash on multiclass tasks.
        is_multiclass = task != "regression" and len(np.unique(ytr)) > 2
        if not is_multiclass:
            kw["linear_leaves"] = True
            kw["linear_lambda"] = linear_lambda
    # IMPORTANT: this measures OUT-OF-BOX DEFAULT behavior. We call fit(Xtr, ytr)
    # with NO explicit eval_set, so ChimeraBoost performs its own internal
    # early-stopping split (early_stopping=True, validation_fraction default) —
    # exactly what a user gets from `ChimeraBoostX().fit(X, y)`. n_estimators and
    # early_stopping_rounds come from the shared harness budget, which equals the
    # class defaults for the headline run, so the benchmark == the default.
    m = Est(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
            learning_rate=lr, depth=depth,
            subsample=subsample,
            cat_combinations=cat_combinations,
            thread_count=threads, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    return _compute_metrics(task, yte, m, Xte), time.time() - t, m.best_iteration_


# Bagged ChimeraBoost: train N members on bootstrap resamples, each early-stopping
# on its own bootstrap, and average. ensemble_n_jobs=1 (members sequential, each
# using the job's thread budget) so we don't nest a joblib pool inside the
# harness's --jobs ProcessPool. For a faster dedicated sweep, run with --jobs 1.
ENSEMBLE_N = 10


def _chimera_ens(n, task, Xtr, ytr, Xte, yte, cat, threads):
    """Shared implementation for all bagged-ChimeraBoost runners."""
    t = time.time()
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    m = Est(n_estimators=MAX_ITERS, early_stopping=True, early_stopping_rounds=PATIENCE,
            n_ensembles=n, ensemble_n_jobs=1,
            thread_count=threads, random_state=0)
    m.fit(Xtr, ytr, cat_features=cat)
    return _compute_metrics(task, yte, m, Xte), time.time() - t, m.best_iteration_


def _run_chimera_ensemble(task, Xtr, ytr, Xte, yte, cat, threads):
    return _chimera_ens(ENSEMBLE_N, task, Xtr, ytr, Xte, yte, cat, threads)


def _run_chimera_ensemble_2(task, Xtr, ytr, Xte, yte, cat, threads):
    return _chimera_ens(2, task, Xtr, ytr, Xte, yte, cat, threads)


def _run_chimera_ensemble_5(task, Xtr, ytr, Xte, yte, cat, threads):
    return _chimera_ens(5, task, Xtr, ytr, Xte, yte, cat, threads)


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
    # HGB has built-in early stopping via a validation fraction. NOTE: HGB has
    # NO n_jobs constructor arg (it parallelises via OpenMP, set through
    # OMP_NUM_THREADS / threadpoolctl), so we must not pass `threads` here —
    # doing so raised TypeError and silently dropped the sklearn column.
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
    common = dict(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
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
    "ChimeraBoostEns2": _run_chimera_ensemble_2,
    "ChimeraBoostEns5": _run_chimera_ensemble_5,
    "ChimeraBoostEns10": _run_chimera_ensemble,
    "sklearn_HGB": _run_sklearn,
    "CatBoost": _run_catboost,
    "XGBoost": _run_xgboost,
    "LightGBM": _run_lightgbm,
}

# Always available (hard deps); the rest are gated on _detect(). Ensemble variants
# are also dep-free but N× slower, so they're selectable via --models but off by
# default (like XGBoost).
_ALWAYS = ("ChimeraBoost", "sklearn_HGB")
_OFF_BY_DEFAULT = ("XGBoost", "ChimeraBoostEns2", "ChimeraBoostEns5", "ChimeraBoostEns10")
_OPTIONAL = ("CatBoost", "XGBoost", "LightGBM")


def _make_runners(model_names, chimera_cfg):
    """Build the runner dict for `model_names`, wiring ChimeraBoost's CLI knobs."""
    import functools
    runners = dict(RUNNERS)
    runners["ChimeraBoost"] = functools.partial(_run_chimera, **chimera_cfg)
    return {name: runners[name] for name in model_names}


def _run_seed_task(task):
    """Fit every requested model on one (dataset, seed) draw. Top-level and
    picklable so it can run in a worker process. Returns
    (ds_name, seed, meta, {model: (metrics, secs, best_iter) or None})."""
    global PATIENCE, ENSEMBLE_N
    (ds_name, seed, scale, threads, model_names, chimera_cfg, patience,
     ensemble_n, need_openml, need_grinsztajn) = task
    PATIENCE = patience
    ENSEMBLE_N = ensemble_n
    if need_openml:
        _add_openml_datasets()
    if need_grinsztajn:
        _add_grinsztajn_datasets()

    rng = np.random.default_rng(1000 + seed)
    X, y, cat, ttype = DATASETS[ds_name](scale, rng)
    strat = y if ttype != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat)
    meta = {"task": ttype, "n_train": int(Xtr.shape[0]),
            "n_total": int(X.shape[0]), "n_features": int(X.shape[1]),
            "has_cats": bool(cat)}
    # Target scale, so the table layer can flag "near-solved" regression datasets
    # (best NRMSE = best_RMSE / y_std below a threshold), where the "% vs best"
    # RMSE ratio explodes a negligible absolute gap. See summarize.NEAR_SOLVED_NRMSE.
    if ttype == "regression":
        meta["y_std"] = float(np.std(y))

    out = {}
    for name, runner in _make_runners(model_names, chimera_cfg).items():
        out[name] = runner(ttype, Xtr, ytr, Xte, yte, cat, threads)
    return ds_name, seed, meta, out


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


class _Progress:
    """Writes a JSON sidecar tracking run progress so another process (the
    /bench status command) can poll it. One file per run, located next to the
    saved results when --save is given, else results/_running.progress.

    Schema: {status, completed, total, pct, started, updated, elapsed_s,
             eta_s, models, config, results_json}. `status` is "running" while
    tasks complete and "done" once finished. Best-effort: any write error is
    swallowed so progress tracking never breaks a benchmark run.
    """

    def __init__(self, results_path, total, model_names, args):
        import datetime
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        if results_path:
            self.path = results_path.rsplit(".", 1)[0] + ".progress"
            self.results_json = results_path.rsplit(".", 1)[0] + ".json"
        else:
            self.path = os.path.join(results_dir, "_running.progress")
            self.results_json = None
        self.total = total
        self.start = time.time()
        lei = args.leaf_estimation_iterations
        self.config = (f"seeds={args.seeds} jobs={args.jobs} "
                       f"grinsztajn={args.grinsztajn} "
                       f"depth={args.chimera_depth} "
                       f"lei={'default' if lei is None else lei} "
                       f"ob={'off' if not args.ordered_boosting else 'on'}")
        self.models = list(model_names)
        self.started_iso = datetime.datetime.now().isoformat(timespec="seconds")
        self._write("running", 0)

    def _write(self, status, completed):
        try:
            elapsed = time.time() - self.start
            rate = completed / elapsed if elapsed > 0 and completed else 0
            remaining = (self.total - completed) / rate if rate > 0 else None
            payload = {
                "status": status, "completed": completed, "total": self.total,
                "pct": round(100.0 * completed / self.total, 1) if self.total else 0,
                "started": self.started_iso,
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(remaining, 1) if remaining is not None else None,
                "models": self.models, "config": self.config,
                "results_json": self.results_json,
            }
            with open(self.path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
        except Exception:
            pass

    def update(self, completed):
        self._write("running", completed)

    def finish(self):
        self._write("done", self.total)


def main():
    global PATIENCE, ENSEMBLE_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier for synthetic dataset sizes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None,
                    help="total thread budget across all parallel jobs "
                         "(None = all cores).")
    ap.add_argument("--jobs", type=int, default=5,
                    help="(dataset, seed) tasks to run in parallel processes; "
                         "each gets threads/jobs threads. GBDT thread scaling is "
                         "sublinear, so spreading seeds beats piling threads on "
                         "one fit (default: 5). Use 1 to run inline.")
    ap.add_argument("--with-xgboost", action="store_true",
                    help="include XGBoost (off by default; it tracks LightGBM "
                         "closely and roughly doubles competitor runtime).")
    ap.add_argument("--only", choices=["regression", "classification"],
                    default=None)
    ap.add_argument("--openml", action="store_true",
                    help="include real OpenML benchmark datasets (downloads + caches)")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="run ONLY the OpenML datasets (implies --openml)")
    ap.add_argument("--grinsztajn", action="store_true",
                    help="run the Grinsztajn et al. 2022 tabular benchmark "
                         "(binary + regression), loaded from the HuggingFace mirror.")
    ap.add_argument("--models", nargs="+", default=None,
                    metavar="MODEL",
                    help=("limit to specific runners, e.g. "
                          "--models ChimeraBoost CatBoost sklearn_HGB. "
                          f"Available: {list(RUNNERS)}"))
    ap.add_argument("--lr", type=float, default=None,
                    help="ChimeraBoost learning rate (default: auto).")
    ap.add_argument("--chimera-depth", type=int, default=6,
                    help="ChimeraBoost tree depth (default: 6).")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience for ALL models "
                         "(default: %d)." % PATIENCE)
    ap.add_argument("--ensemble-n", type=int, default=None, dest="ensemble_n",
                    help="number of members for the ChimeraBoostEns10 bagged "
                         "runner (default: %d)." % ENSEMBLE_N)
    ap.add_argument("--no-ordered-boosting", dest="ordered_boosting",
                    action="store_false", default=True,
                    help="disable ChimeraBoost's LOO leaf correction.")
    ap.add_argument("--chimera-subsample", type=float, default=1.0,
                    dest="chimera_subsample",
                    help="ChimeraBoost row subsample fraction; "
                         "MVS sampling when < 1.0 (default: 1.0 = off).")
    ap.add_argument("--chimera-mcw", type=float, default=None,
                    dest="chimera_mcw",
                    help="ChimeraBoost min_child_weight (default: None = the "
                         "classifier's size-adaptive auto rule).")
    ap.add_argument("--chimera-cat-combinations", action="store_true",
                    default=False, dest="cat_combinations",
                    help="enable 2-way categorical feature combinations "
                         "(default: off).")
    ap.add_argument("--chimera-lei", type=int, default=None,
                    dest="leaf_estimation_iterations",
                    help="leaf estimation iterations: additional Newton steps to "
                         "refine leaf values after tree structure is fixed. "
                         "None = use each class default (Regressor=1, Classifier=3). "
                         "Only applies on the non-LOO path.")
    ap.add_argument("--chimera-linear-leaves", action="store_true",
                    dest="linear_leaves",
                    help="enable per-leaf linear models (regression + binary; "
                         "multiclass falls back to constant leaves). Default off.")
    ap.add_argument("--chimera-linear-lambda", type=float, default=1.0,
                    dest="linear_lambda",
                    help="ridge penalty on per-leaf linear slopes (default 1.0).")
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
        import sys, datetime
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

    if args.patience is not None:
        PATIENCE = args.patience
    if args.ensemble_n is not None:
        ENSEMBLE_N = args.ensemble_n

    need_openml = (args.openml or args.no_synthetic or bool(
        args.datasets and any(d.startswith("oml:") for d in args.datasets)))
    if need_openml:
        _add_openml_datasets()
    if args.no_synthetic:
        for k in [k for k in DATASETS if not k.startswith("oml:")]:
            del DATASETS[k]

    # Grinsztajn suites: register them, then (unless specific --datasets were
    # named) run ONLY them, since they are the "serious" recognized benchmark.
    need_grinsztajn = args.grinsztajn or bool(
        args.datasets and any(d.startswith("gr:") for d in args.datasets))
    if need_grinsztajn:
        _add_grinsztajn_datasets()
        if args.grinsztajn and not args.datasets:
            for k in [k for k in DATASETS if not k.startswith("gr:")]:
                del DATASETS[k]

    # Resolve the model set. Competitors are gated on install; XGBoost is off
    # by default (it tracks LightGBM). --models overrides everything.
    available = (list(_ALWAYS)
                 + [m for m in _OFF_BY_DEFAULT if not HAVE.get(m.lower(), False)]
                 + [m for m in _OPTIONAL if HAVE[m.lower()]])
    if args.models:
        unknown = set(args.models) - set(RUNNERS)
        if unknown:
            ap.error(f"Unknown models: {unknown}. Available: {list(RUNNERS)}")
        model_names = [m for m in args.models if m in available]
    else:
        model_names = [m for m in available if m not in _OFF_BY_DEFAULT
                       or (m == "XGBoost" and args.with_xgboost)]
    if "ChimeraBoost" not in model_names:
        ap.error("ChimeraBoost must be one of the models (it is the baseline).")

    # None = use each class's default (Regressor=False, Classifier=True).
    # --no-ordered-boosting forces False for both.
    ob_override = None if args.ordered_boosting else False
    chimera_cfg = dict(lr=args.lr, ordered_boosting=ob_override,
                       depth=args.chimera_depth, subsample=args.chimera_subsample,
                       mcw=args.chimera_mcw, cat_combinations=args.cat_combinations,
                       leaf_estimation_iterations=args.leaf_estimation_iterations,
                       linear_leaves=args.linear_leaves,
                       linear_lambda=args.linear_lambda)

    # Split the thread budget across parallel jobs: GBDT thread scaling is
    # sublinear, so running J seeds at threads/J each beats one fit at all cores.
    total_threads = args.threads or os.cpu_count() or 1
    jobs = max(1, args.jobs)
    threads_per = max(1, total_threads // jobs)

    selected = [ds for ds in DATASETS
                if not (args.datasets and ds not in args.datasets)
                and not (args.only == "regression" and _task_of(ds) != "regression")
                and not (args.only == "classification" and _task_of(ds) == "regression")]

    print("Detected competitors:",
          ", ".join(k for k, v in HAVE.items() if v) or "none (sklearn only)")
    print(f"scale={args.scale}  seeds={args.seeds}  jobs={jobs}  "
          f"threads/job={threads_per}  max_iter={MAX_ITERS}  patience={PATIENCE}  "
          f"models={model_names}"
          + (f"  chimera_lr={args.lr}" if args.lr else "")
          + ("  ordered_boosting=off" if not args.ordered_boosting else "")
          + (f"  subsample={args.chimera_subsample}" if args.chimera_subsample < 1.0 else "")
          + ("  cat_combinations=on" if args.cat_combinations else "")
          + "\n")

    # Run every (dataset, seed) draw, in parallel processes unless jobs == 1.
    tasks = [(ds, s, args.scale, threads_per, model_names, chimera_cfg,
              PATIENCE, ENSEMBLE_N, need_openml, need_grinsztajn)
             for ds in selected for s in range(args.seeds)]
    total_tasks = len(tasks)

    # Live-progress sidecar so `bench_status.py` (the /bench command) can report
    # how far along a run is from another process. Written next to the results
    # file when --save is on; otherwise to results/_running.progress. It records
    # completed/total task counts (one task = one dataset-seed draw) and the
    # config line, and is marked done at the end / removed on a clean exit.
    prog = _Progress(tee[1] if tee else None, total_tasks, model_names, args)

    collected = defaultdict(dict)   # collected[ds][seed] = (meta, out)
    done = 0
    if jobs == 1:
        for t in tasks:
            ds, seed, meta, out = _run_seed_task(t)
            collected[ds][seed] = (meta, out)
            done += 1
            prog.update(done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_run_seed_task, t) for t in tasks]
            for fut in as_completed(futs):
                ds, seed, meta, out = fut.result()
                collected[ds][seed] = (meta, out)
                done += 1
                prog.update(done)
    prog.finish()

    metric_name = {"regression": "RMSE (lower better)",
                   "binary": "F1 macro (higher better)",
                   "multiclass": "F1 macro (higher better)"}
    gap_acc = {m: [] for m in model_names if m != "ChimeraBoost"}
    speed_acc = {m: [] for m in model_names if m != "ChimeraBoost"}
    raw_records = []      # one row per (dataset, model, seed); feeds make_tables
    dataset_meta = {}

    for ds_name in selected:
        seed_map = collected.get(ds_name)
        if not seed_map:
            continue
        dataset_meta[ds_name] = seed_map[next(iter(seed_map))][0]
        task = dataset_meta[ds_name]["task"]

        results = {m: [] for m in model_names}
        times = {m: [] for m in model_names}
        iters = {m: [] for m in model_names}
        for s in range(args.seeds):
            if s not in seed_map:
                continue
            for name, res in seed_map[s][1].items():
                if res is None:
                    continue
                metrics, secs, best_it = res
                results[name].append(metrics["primary"])
                times[name].append(secs)
                if best_it is not None:
                    iters[name].append(best_it)
                raw_records.append({
                    "dataset": ds_name, "model": name, "seed": s,
                    "metrics": metrics, "fit_time": secs,
                    "best_iter": int(best_it) if best_it is not None else None,
                })

        print(f"### {ds_name}  [{task}]  metric={metric_name[task]}")
        for name in model_names:
            if not results[name]:
                continue
            sc = np.array(results[name])
            tm = np.array(times[name])
            disp = (-sc if task == "regression" else sc)
            it_str = f"  trees~{int(np.mean(iters[name]))}" if iters[name] else ""
            star = " <-- ours" if name == "ChimeraBoost" else ""
            print(f"  {name:14s} {disp.mean():8.4f} +/- {disp.std():.4f}"
                  f"   fit {tm.mean():6.2f}s{it_str}{star}")

        if results["ChimeraBoost"]:
            our_score = np.mean(results["ChimeraBoost"])
            our_time = np.mean(times["ChimeraBoost"])
            for name in gap_acc:
                if results[name]:
                    gap_acc[name].append(_rel_gap(our_score, np.mean(results[name]), task))
                    speed_acc[name].append(np.mean(times[name]) / max(our_time, 1e-9))
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
                    "patience": PATIENCE, "ensemble_n": ENSEMBLE_N,
                    "threads_per_model": threads_per,
                    "total_threads": total_threads,
                    "jobs": jobs,
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