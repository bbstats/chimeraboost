"""benchmarks/fuzz_inputs.py — deep robustness sweep ("toggle every case").

Crosses {parameter configs} x {input pathologies} x {regression, binary,
multiclass} and, for each combination, fits + predicts and classifies the
outcome into exactly one of:

  PASS         ran cleanly; output has the right shape, is all-finite, and (for
               classifiers) predict_proba rows sum to 1.
  OK-REJECTED  raised a CLEAN, named ValueError/TypeError (or NotFittedError)
               with a non-empty message -- the right response to invalid params
               or unsupported input.
  CRASH        anything else: an ugly exception (numba TypingError, IndexError,
               AssertionError, ZeroDivisionError, OverflowError, ...), a non-finite
               / wrong-shape output, proba not summing to 1, OR a bad input that
               was silently ACCEPTED when it should have been rejected.

Exits nonzero if any CRASH, so it can gate a change. This is the on-demand,
exhaustive sibling of tests/test_robustness.py (which runs a fast curated subset
every test cycle). Report-only on accuracy/timing -- those have their own guards.

Usage:
    python benchmarks/fuzz_inputs.py                 # full sweep
    python benchmarks/fuzz_inputs.py --only inputs   # input pathologies only
    python benchmarks/fuzz_inputs.py --only params   # parameter boundaries only
    python benchmarks/fuzz_inputs.py -v              # print every case, not just CRASH
"""
import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import pandas as pd
except ImportError:  # pandas is a hard dep for the cat path; bail loudly.
    print("pandas is required for this sweep", file=sys.stderr)
    raise

# Exceptions that count as a CLEAN rejection. Anything else raised is a CRASH:
# a user passing bad data should get one of these with an actionable message,
# never a bare numba/numpy internal error.
_CLEAN_ERRORS = (ValueError, TypeError)
try:
    from sklearn.exceptions import NotFittedError
    _CLEAN_ERRORS = _CLEAN_ERRORS + (NotFittedError,)
except Exception:
    pass

# Small + fast: every case fits in well under a second.
N = 300
COMMON = dict(n_estimators=15, random_state=0, thread_count=1)


# ---------------------------------------------------------------------------
# Clean baseline data per task.
# ---------------------------------------------------------------------------
def base_numeric(task, n=N, n_features=5, seed=0):
    """A clean all-numeric (X, y) for the given task."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # Signal from up to the first three columns (robust to n_features < 3).
    coefs = np.array([2.0, -1.5, 0.5])[:n_features]
    signal = X[:, :len(coefs)] @ coefs
    if task == "regression":
        y = signal + rng.normal(scale=0.3, size=n)
    elif task == "binary":
        y = (signal > np.median(signal)).astype(int)
    else:  # multiclass
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    return X, y


def base_mixed(task, n=N, seed=0):
    """A clean mixed numeric/categorical DataFrame (X, y, cat_features)."""
    rng = np.random.default_rng(seed)
    lo = rng.integers(0, 4, n)
    hi = rng.integers(0, 40, n)
    num = rng.normal(size=(n, 2))
    eff = np.array([-1.0, -0.2, 0.5, 1.2])[lo] + 0.6 * num[:, 0]
    if task == "regression":
        y = eff + rng.normal(scale=0.3, size=n)
    elif task == "binary":
        y = (eff > np.median(eff)).astype(int)
    else:
        y = np.digitize(eff, np.quantile(eff, [1 / 3, 2 / 3]))
    X = pd.DataFrame({
        "lo": [f"l{c}" for c in lo],
        "hi": [f"h{c}" for c in hi],
        "n0": num[:, 0],
        "n1": num[:, 1],
    })
    return X, y, ["lo", "hi"]


def _classifier_ok(y):
    """A classifier needs >=2 classes; squash a 1-class probe target if needed."""
    return y


# ---------------------------------------------------------------------------
# Outcome classification.
# ---------------------------------------------------------------------------
class Outcome:
    PASS = "PASS"
    OK_REJECTED = "OK-REJECTED"
    CRASH = "CRASH"


def _check_output(task, est, Xpred, n_pred):
    """Validate prediction output; return an error string or None if fine."""
    pred = est.predict(Xpred)
    pred = np.asarray(pred)
    if pred.shape[0] != n_pred:
        return f"predict shape {pred.shape} != n_pred {n_pred}"
    if not np.isfinite(np.asarray(pred, dtype=float)).all():
        return "predict produced non-finite values"
    if task != "regression":
        proba = np.asarray(est.predict_proba(Xpred), dtype=float)
        if proba.shape[0] != n_pred:
            return f"predict_proba shape {proba.shape} != n_pred {n_pred}"
        if not np.isfinite(proba).all():
            return "predict_proba produced non-finite values"
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-4):
            return f"predict_proba rows do not sum to 1 (max dev "\
                   f"{np.max(np.abs(proba.sum(axis=1) - 1.0)):.2e})"
    return None


def evaluate(task, params, X, y, cat_features, Xpred, expect, fit_kw=None):
    """Run one case and return (Outcome, detail)."""
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    fit_kw = dict(fit_kw or {})
    if cat_features is not None:
        fit_kw.setdefault("cat_features", cat_features)
    n_pred = np.asarray(Xpred).shape[0] if hasattr(Xpred, "shape") \
        else len(Xpred)
    try:
        est = Est(**{**COMMON, **params})
        est.fit(X, y, **fit_kw)
        err = _check_output(task, est, Xpred, n_pred)
    except _CLEAN_ERRORS as e:
        msg = str(e).strip()
        if not msg:
            return Outcome.CRASH, f"{type(e).__name__} with EMPTY message"
        if expect == "reject":
            return Outcome.OK_REJECTED, f"{type(e).__name__}: {msg[:90]}"
        return Outcome.CRASH, f"expected PASS but raised {type(e).__name__}: {msg[:90]}"
    except Exception as e:  # noqa: BLE001 — any non-clean error is a CRASH
        return Outcome.CRASH, f"UGLY {type(e).__name__}: {str(e)[:90]}"
    # No exception was raised.
    if expect == "reject":
        return Outcome.CRASH, "bad input/param SILENTLY ACCEPTED (expected rejection)"
    if err is not None:
        return Outcome.CRASH, err
    return Outcome.PASS, ""


# ---------------------------------------------------------------------------
# Input-pathology cases. Each builder returns (X, y, cat_features, Xpred, expect).
# ---------------------------------------------------------------------------
def input_cases(task):
    """Yield (name, builder) for input pathologies on the given task."""
    cases = []

    def add(name, fn):
        cases.append((name, fn))

    # ---- dtypes -----------------------------------------------------------
    def dtype_case(dtype):
        def build():
            X, y = base_numeric(task)
            df = pd.DataFrame(X).astype(dtype)
            return df, y, None, df, "pass"
        return build
    for dt in ["float64", "float32", "int64"]:
        add(f"dtype:{dt}", dtype_case(dt))

    def nullable_case(dtype, with_na):
        def build():
            X, y = base_numeric(task)
            col = pd.array(np.round(X[:, 0] * 3).astype("int64"), dtype="Int64") \
                if dtype == "Int64" else (
                pd.array(X[:, 0], dtype="Float64") if dtype == "Float64"
                else pd.array(X[:, 0] > 0, dtype="boolean"))
            if with_na:
                col[:10] = pd.NA
            df = pd.DataFrame(X)
            df[0] = col
            return df, y, None, df, "pass"
        return build
    for dt in ["Int64", "Float64", "boolean"]:
        add(f"nullable:{dt}+NA", nullable_case(dt, True))
    add("nullable:all-NA", lambda: _all_na_nullable(task))

    add("dtype:object-numeric", lambda: _object_numeric(task))
    add("dtype:bool", lambda: _bool_cols(task))

    # categorical-flavored
    add("cat:pandas-category", lambda: _cat_category(task))
    add("cat:pandas-string", lambda: _cat_string(task))
    add("cat:datetime", lambda: _cat_datetime(task))
    add("cat:high-cardinality-unique", lambda: _cat_high_card(task))
    add("cat:empty-string", lambda: _cat_empty_string(task))
    add("cat:unseen-at-predict", lambda: _cat_unseen(task))
    add("cat:category-no-cat_features", lambda: _category_no_catfeatures(task))

    # ---- degenerate shapes ------------------------------------------------
    add("shape:1-row", lambda: _n_rows(task, 1))
    add("shape:2-rows", lambda: _n_rows(task, 2))
    add("shape:1-feature", lambda: _one_feature(task))
    add("shape:wide-n<<p", lambda: _wide(task))

    # ---- degenerate content ----------------------------------------------
    add("content:all-nan-column", lambda: _all_nan_col(task))
    add("content:all-constant-column", lambda: _const_col(task))
    add("content:all-nan-row", lambda: _all_nan_row(task))
    add("content:duplicate-columns", lambda: _dup_cols(task))
    add("content:huge-magnitude-1e300", lambda: _huge(task))
    add("content:tiny-magnitude-1e-300", lambda: _tiny(task))
    add("content:+inf", lambda: _inf(task, +1))
    add("content:-inf", lambda: _inf(task, -1))
    add("content:nan-only-at-predict", lambda: _nan_predict(task))

    # ---- predict-time -----------------------------------------------------
    add("predict:column-reorder", lambda: _reorder(task))
    add("predict:wrong-feature-count", lambda: _wrong_ncols(task))

    if task == "regression":
        # Huge finite targets: RMSE squares residuals, so an overflow here would
        # surface as a non-finite gradient -> NaN model. Must stay finite.
        add("content:huge-target-1e300", lambda: _huge_target())
    if task != "regression":
        add("target:single-class", lambda: _single_class(task))

    return cases


def _huge_target():
    X, y = base_numeric("regression")
    return X, y * 1e300, None, X, "pass"


# ---- individual builders --------------------------------------------------
def _all_na_nullable(task):
    X, y = base_numeric(task)
    df = pd.DataFrame(X)
    df[0] = pd.array([pd.NA] * len(df), dtype="Int64")
    return df, y, None, df, "pass"


def _object_numeric(task):
    X, y = base_numeric(task)
    df = pd.DataFrame(X)
    df[0] = df[0].astype(object)
    return df, y, None, df, "pass"


def _bool_cols(task):
    X, y = base_numeric(task)
    df = pd.DataFrame(X)
    df[0] = df[0] > 0
    return df, y, None, df, "pass"


def _cat_category(task):
    X, y, _ = base_mixed(task)
    X = X.copy()
    X["lo"] = X["lo"].astype("category")
    return X, y, ["lo", "hi"], X, "pass"


def _cat_string(task):
    X, y, _ = base_mixed(task)
    X = X.copy()
    X["lo"] = X["lo"].astype("string")
    return X, y, ["lo", "hi"], X, "pass"


def _cat_datetime(task):
    X, y, _ = base_mixed(task)
    X = X.copy()
    X["lo"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        np.arange(len(X)) % 7, unit="D")
    return X, y, ["lo", "hi"], X, "pass"


def _cat_high_card(task):
    X, y, _ = base_mixed(task)
    X = X.copy()
    X["hi"] = [f"u{i}" for i in range(len(X))]   # every value unique
    return X, y, ["lo", "hi"], X, "pass"


def _cat_empty_string(task):
    X, y, _ = base_mixed(task)
    X = X.copy()
    vals = X["lo"].to_numpy().astype(object)
    vals[:20] = ""
    X["lo"] = vals
    return X, y, ["lo", "hi"], X, "pass"


def _cat_unseen(task):
    X, y, _ = base_mixed(task)
    Xp = X.iloc[:10].copy()
    Xp["lo"] = "BRAND_NEW_LEVEL"
    Xp["hi"] = "ALSO_NEW"
    return X, y, ["lo", "hi"], Xp, "pass"


def _category_no_catfeatures(task):
    # A category column without cat_features should clean-error (non-numeric).
    X, y, _ = base_mixed(task)
    X = X.copy()
    X["lo"] = X["lo"].astype("category")
    return X, y, None, X, "reject"


def _n_rows(task, n):
    X, y = base_numeric(task, n=max(n, 1))
    X, y = X[:n], y[:n]
    if task != "regression" and len(np.unique(y)) < 2:
        # too few rows to host 2 classes -> clean error is acceptable
        return X, y, None, X, "reject"
    return X, y, None, X, "pass"


def _one_feature(task):
    X, y = base_numeric(task, n_features=1)
    return X, y, None, X, "pass"


def _wide(task):
    X, y = base_numeric(task, n=20, n_features=200)
    return X, y, None, X, "pass"


def _all_nan_col(task):
    X, y = base_numeric(task)
    X = X.copy(); X[:, 2] = np.nan
    return X, y, None, X, "pass"


def _const_col(task):
    X, y = base_numeric(task)
    X = X.copy(); X[:, 1] = 7.0
    return X, y, None, X, "pass"


def _all_nan_row(task):
    X, y = base_numeric(task)
    X = X.copy(); X[0, :] = np.nan
    return X, y, None, X, "pass"


def _dup_cols(task):
    X, y = base_numeric(task)
    df = pd.DataFrame(X, columns=["a", "b", "a", "c", "d"])  # duplicate "a"
    return df, y, None, df, "pass"


def _huge(task):
    X, y = base_numeric(task)
    X = X.copy(); X[:, 0] = X[:, 0] * 1e300
    return X, y, None, X, "pass"


def _tiny(task):
    X, y = base_numeric(task)
    X = X.copy(); X[:, 0] = X[:, 0] * 1e-300
    return X, y, None, X, "pass"


def _inf(task, sign):
    X, y = base_numeric(task)
    X = X.copy(); X[0, 0] = sign * np.inf
    return X, y, None, X, "reject"


def _nan_predict(task):
    X, y = base_numeric(task)
    Xp = X[:10].copy(); Xp[0, 0] = np.nan
    return X, y, None, Xp, "pass"


def _reorder(task):
    # The booster consumes columns positionally and (like sklearn's
    # feature_names_in_ check) REJECTS a reordered DataFrame rather than silently
    # realigning by name -- reordering without rejection would give wrong preds.
    X, y = base_numeric(task)
    df = pd.DataFrame(X, columns=list("abcde"))
    Xp = df.iloc[:10][["c", "a", "e", "b", "d"]]   # reordered columns, same names
    return df, y, None, Xp, "reject"


def _wrong_ncols(task):
    X, y = base_numeric(task)
    Xp = X[:10, :3]                                 # fewer features than fit
    return X, y, None, Xp, "reject"


def _single_class(task):
    X, y = base_numeric(task)
    y = np.zeros(len(y), dtype=int)
    return X, y, None, X, "reject"


# ---------------------------------------------------------------------------
# Parameter-boundary cases. (name, params, expect, task_filter)
# ---------------------------------------------------------------------------
def param_cases():
    P = []

    def add(name, params, expect, tasks=("regression", "binary", "multiclass")):
        P.append((name, params, expect, tasks))

    # depth
    add("depth=1", dict(depth=1), "pass")
    add("depth=16", dict(depth=16), "pass")
    add("depth=0", dict(depth=0), "reject")
    add("depth=17", dict(depth=17), "reject")
    add("depth=-1", dict(depth=-1), "reject")
    # max_bins
    add("max_bins=2", dict(max_bins=2), "pass")
    add("max_bins=65534", dict(max_bins=65534), "pass")
    add("max_bins=1", dict(max_bins=1), "reject")
    add("max_bins=65535", dict(max_bins=65535), "reject")
    # n_estimators
    add("n_estimators=1", dict(n_estimators=1, early_stopping=False), "pass")
    add("n_estimators=0", dict(n_estimators=0), "reject")
    # learning_rate
    add("learning_rate=1e-4", dict(learning_rate=1e-4), "pass")
    add("learning_rate=5.0", dict(learning_rate=5.0), "pass")
    add("learning_rate=0", dict(learning_rate=0.0), "reject")
    add("learning_rate=-0.1", dict(learning_rate=-0.1), "reject")
    # subsample / colsample
    add("subsample=0.01", dict(subsample=0.01), "pass")
    add("subsample=0", dict(subsample=0.0), "reject")
    add("subsample=1.5", dict(subsample=1.5), "reject")
    add("colsample=0.01", dict(colsample=0.01), "pass")
    add("colsample=0", dict(colsample=0.0), "reject")
    add("colsample=1.5", dict(colsample=1.5), "reject")
    # l2 / mcw
    add("l2_leaf_reg=0", dict(l2_leaf_reg=0.0), "pass")
    add("l2_leaf_reg=1e6", dict(l2_leaf_reg=1e6), "pass")
    add("l2_leaf_reg=-1", dict(l2_leaf_reg=-1.0), "reject")
    add("min_child_weight=0", dict(min_child_weight=0.0), "pass")
    add("min_child_weight=1e6", dict(min_child_weight=1e6), "pass")
    add("min_child_weight=-1", dict(min_child_weight=-1.0), "reject")
    # cat knobs (use mixed data; handled in runner)
    add("cat_smoothing=1e-6", dict(cat_smoothing=1e-6), "pass")
    add("cat_smoothing=1e3", dict(cat_smoothing=1e3), "pass")
    add("cat_smoothing=0", dict(cat_smoothing=0.0), "reject")
    add("cat_n_permutations=1", dict(cat_n_permutations=1), "pass")
    add("cat_n_permutations=0", dict(cat_n_permutations=0), "reject")
    # leaf estimation
    add("leaf_estimation_iterations=1", dict(leaf_estimation_iterations=1), "pass")
    add("leaf_estimation_iterations=10", dict(leaf_estimation_iterations=10), "pass")
    add("leaf_estimation_iterations=0", dict(leaf_estimation_iterations=0), "reject")
    # validation_fraction / early_stopping_rounds
    add("validation_fraction=0.01", dict(validation_fraction=0.01), "pass")
    add("validation_fraction=0", dict(validation_fraction=0.0), "reject")
    add("validation_fraction=1.0", dict(validation_fraction=1.0), "reject")
    add("early_stopping_rounds=1", dict(early_stopping_rounds=1), "pass")
    add("early_stopping_rounds=0", dict(early_stopping_rounds=0), "reject")
    # toggles
    add("ordered_boosting=True", dict(ordered_boosting=True), "pass")
    add("linear_leaves=True", dict(linear_leaves=True), "pass",
        tasks=("regression", "binary"))
    add("cat_combinations=True", dict(cat_combinations=True), "pass")
    add("cat_combinations=False", dict(cat_combinations=False), "pass")
    add("n_ensembles=3", dict(n_ensembles=3), "pass")
    # regressor-only loss configs handled separately in run_params

    return P


# Regressor-only loss configurations (need the regressor + an alpha for Quantile).
def loss_cases():
    return [
        ("loss=MAE", dict(loss="MAE"), "pass"),
        ("loss=Quantile,alpha=0.5", dict(loss="Quantile", alpha=0.5), "pass"),
        ("loss=Quantile,alpha=0.01", dict(loss="Quantile", alpha=0.01), "pass"),
        ("loss=Quantile,alpha=0.99", dict(loss="Quantile", alpha=0.99), "pass"),
        ("loss=Quantile,alpha=0", dict(loss="Quantile", alpha=0.0), "reject"),
        ("loss=Quantile,alpha=1", dict(loss="Quantile", alpha=1.0), "reject"),
        ("loss=bogus", dict(loss="NOPE"), "reject"),
    ]


# ---------------------------------------------------------------------------
# Fit-option matrix (toggled on clean mixed data, one task each is enough).
# ---------------------------------------------------------------------------
def fit_option_cases(task):
    rng = np.random.default_rng(1)
    X, y, cat = base_mixed(task)
    n = len(y)
    opts = []
    opts.append(("opt:sample_weight-uniform",
                 dict(sample_weight=np.ones(n)), "pass"))
    opts.append(("opt:sample_weight-zero-heavy",
                 dict(sample_weight=(rng.random(n) > 0.5).astype(float) + 1e-9),
                 "pass"))
    opts.append(("opt:sample_weight-all-zero",
                 dict(sample_weight=np.zeros(n)), "reject"))
    opts.append(("opt:sample_weight-negative",
                 dict(sample_weight=-np.ones(n)), "reject"))
    opts.append(("opt:eval_set",
                 dict(eval_set=(X.iloc[:50], y[:50])), "pass"))
    opts.append(("opt:groups",
                 dict(groups=rng.integers(0, 5, n)), "pass"))
    return X, y, cat, opts


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Case generators. Each yields fully-resolved dicts so both this script and
# tests/test_robustness.py consume one source of truth. A case is:
#   dict(category, task, name, params, X, y, cat, Xpred, expect, fit_kw)
# Builder failures surface as a sentinel "builder_error" key (a harness bug).
# ---------------------------------------------------------------------------
def iter_input_cases():
    for task in ("regression", "binary", "multiclass"):
        for name, build in input_cases(task):
            try:
                X, y, cat, Xpred, expect = build()
            except Exception as e:  # builder itself failed -> harness bug
                yield dict(category="input", task=task, name=name,
                           builder_error=f"{type(e).__name__}: {e}")
                continue
            yield dict(category="input", task=task, name=name, params={},
                       X=X, y=y, cat=cat, Xpred=Xpred, expect=expect, fit_kw=None)


def iter_param_cases():
    for name, params, expect, tasks in param_cases():
        for task in tasks:
            # cat knobs / cat_combinations need categorical data to exercise.
            if any(k.startswith("cat_") for k in params):
                X, y, cat = base_mixed(task)
                Xpred = X
            else:
                X, y = base_numeric(task)
                cat, Xpred = None, X
            yield dict(category="param", task=task, name=name, params=params,
                       X=X, y=y, cat=cat, Xpred=Xpred, expect=expect, fit_kw=None)
    for name, params, expect in loss_cases():    # regressor-only loss configs
        X, y = base_numeric("regression")
        yield dict(category="param", task="regression", name=name, params=params,
                   X=X, y=y, cat=None, Xpred=X, expect=expect, fit_kw=None)


def iter_fit_option_cases():
    for task in ("regression", "binary", "multiclass"):
        X, y, cat, opts = fit_option_cases(task)
        for name, fit_kw, expect in opts:
            yield dict(category="fit-opt", task=task, name=name, params={},
                       X=X, y=y, cat=cat, Xpred=X, expect=expect, fit_kw=fit_kw)


def run_case(case):
    """Resolve one generated case to (Outcome, detail)."""
    if "builder_error" in case:
        return Outcome.CRASH, f"builder error {case['builder_error']}"
    return evaluate(case["task"], case["params"], case["X"], case["y"],
                    case["cat"], case["Xpred"], case["expect"],
                    fit_kw=case["fit_kw"])


def run_inputs(results, verbose):
    for case in iter_input_cases():
        outcome, detail = run_case(case)
        _record(results, "input", case["task"], case["name"], outcome, detail, verbose)


def run_params(results, verbose):
    for case in iter_param_cases():
        outcome, detail = run_case(case)
        _record(results, "param", case["task"], case["name"], outcome, detail, verbose)


def run_fit_options(results, verbose):
    for case in iter_fit_option_cases():
        outcome, detail = run_case(case)
        _record(results, "fit-opt", case["task"], case["name"], outcome, detail, verbose)


def _record(results, category, task, name, outcome, detail, verbose):
    results.append((category, task, name, outcome, detail))
    if outcome == Outcome.CRASH:
        print(f"  CRASH  [{category}/{task}] {name}\n         -> {detail}", flush=True)
    elif verbose:
        print(f"  {outcome:<11} [{category}/{task}] {name}"
              + (f"  ({detail})" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["inputs", "params", "fit-opts"], default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    results = []
    if args.only in (None, "inputs"):
        print("== input pathologies ==", flush=True)
        run_inputs(results, args.verbose)
    if args.only in (None, "params"):
        print("== parameter boundaries ==", flush=True)
        run_params(results, args.verbose)
    if args.only in (None, "fit-opts"):
        print("== fit-option matrix ==", flush=True)
        run_fit_options(results, args.verbose)

    counts = {o: sum(r[3] == o for r in results)
              for o in (Outcome.PASS, Outcome.OK_REJECTED, Outcome.CRASH)}
    print("\n== summary ==")
    print(f"  total cases : {len(results)}")
    print(f"  PASS        : {counts[Outcome.PASS]}")
    print(f"  OK-REJECTED : {counts[Outcome.OK_REJECTED]}")
    print(f"  CRASH       : {counts[Outcome.CRASH]}")
    if counts[Outcome.CRASH]:
        print("\nFAIL: crashes above must be triaged (real bug, missing clean "
              "rejection, or harness false-positive).")
        return 1
    print("\nOK: every case either worked or was cleanly rejected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
