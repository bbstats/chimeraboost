"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import warnings

import numpy as np
from .booster import (GradientBoosting, LINEAR_LEAVES_MIN_SAMPLES,
                      MulticlassBoosting)
from .preprocessing import as_model_array
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _fit_temperature(raw, y, multiclass):
    """Learn the scalar T > 0 minimizing validation log loss of sigmoid(raw/T)
    (binary) or softmax(raw/T) (multiclass). Dividing logits by T is monotonic,
    so predictions are unchanged â€” only their probabilities are recalibrated.
    `y` is the 0/1 label (binary) or the class index (multiclass)."""
    from scipy.optimize import minimize_scalar

    raw = np.asarray(raw, dtype=np.float64)
    if multiclass:
        rows = np.arange(raw.shape[0])

        def loss(T):
            logits = raw / T
            mx = logits.max(axis=1, keepdims=True)
            log_z = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            return float(np.mean(log_z - logits[rows, y]))
    else:
        def loss(T):
            z = raw / T
            # Stable binary cross-entropy: softplus(z) - y*z.
            return float(np.mean(np.log1p(np.exp(-np.abs(z)))
                                 + np.maximum(z, 0.0) - y * z))

    res = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x) if res.success else 1.0


# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({"early_stopping", "validation_fraction",
                           "n_ensembles", "ensemble_n_jobs", "cat_features",
                           "cross_features", "selection_rounds"})


def _validate_hyperparams(estimator):
    """Reject malformed constructor parameters with clear, named errors.

    Called at the start of ``fit`` (sklearn's recommended place for parameter
    validation -- never in ``__init__``). Without this, bad values either fail
    cryptically deep in numba (e.g. ``depth=-1`` -> "negative shift count"),
    silently produce a broken model (``learning_rate=-0.1`` diverges to garbage;
    ``n_estimators=0`` builds an empty model), or OOM (``depth=30`` allocates a
    2**30-leaf histogram). ``None`` is left to the documented per-parameter
    default resolution and is not rejected here.
    """
    p = estimator.get_params()

    def _pos_int(name, lo=1):
        v = p[name]
        if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                and v >= lo):
            raise ValueError(f"{name} must be an integer >= {lo}; got {v!r}.")

    def _in_range(name, lo, hi, *, lo_incl=True, hi_incl=True, allow_none=False):
        v = p[name]
        if v is None and allow_none:
            return
        ok = isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
        if ok:
            ok = (v >= lo if lo_incl else v > lo) and \
                 (v <= hi if hi_incl else v < hi)
        if not ok:
            lb = "[" if lo_incl else "("
            rb = "]" if hi_incl else ")"
            raise ValueError(
                f"{name} must be in {lb}{lo}, {hi}{rb}; got {v!r}.")

    _pos_int("n_estimators")
    _pos_int("cat_n_permutations")
    _pos_int("leaf_estimation_iterations")
    # depth: a depth-d tree allocates 2**d leaves in the histogram buffer, so an
    # unbounded depth OOMs. 16 matches CatBoost's documented maximum. None is the
    # regressor's loss-adaptive default, resolved at fit.
    v = p["depth"]
    if v is not None and not (isinstance(v, (int, np.integer))
                              and not isinstance(v, bool) and 1 <= v <= 16):
        raise ValueError(f"depth must be an integer in [1, 16] or None; got {v!r}.")
    _in_range("max_bins", 2, 65534)
    _in_range("learning_rate", 0.0, np.inf, lo_incl=False, allow_none=True)
    _in_range("l2_leaf_reg", 0.0, np.inf)
    _in_range("subsample", 0.0, 1.0, lo_incl=False)
    _in_range("colsample", 0.0, 1.0, lo_incl=False, allow_none=True)
    # cat_smoothing is a Bayesian pseudocount in the ordered-TS denominator
    # (count + a); a=0 makes the first occurrence of every category divide 0/0.
    _in_range("cat_smoothing", 0.0, np.inf, lo_incl=False)
    _in_range("linear_lambda", 0.0, np.inf)
    _in_range("min_child_weight", 0.0, np.inf, allow_none=True)
    _in_range("validation_fraction", 0.0, 1.0, lo_incl=False, hi_incl=False)
    _in_range("early_stopping_rounds", 1, np.inf, allow_none=True)
    _in_range("selection_rounds", 1, np.inf, allow_none=True)
    if p.get("n_ensembles") is not None:
        _pos_int("n_ensembles")
    # Regressor-only loss / alpha (the classifier picks its loss automatically).
    if "loss" in p:
        if p["loss"] not in ("RMSE", "MAE", "Quantile"):
            raise ValueError(
                f"loss must be one of 'RMSE', 'MAE', 'Quantile'; got {p['loss']!r}.")
        if p["loss"] == "Quantile":
            _in_range("alpha", 0.0, 1.0, lo_incl=False, hi_incl=False)


def _resolve_cat_features(estimator, cat_features):
    """Resolve the effective cat_features: the ``fit`` argument when given,
    otherwise the ``cat_features`` constructor argument. The fit argument wins so
    a one-off call can override, while the constructor form lets sklearn meta-
    estimators (GridSearchCV/Pipeline) carry it -- a fit-only kwarg cannot. Never
    mutates ``estimator.cat_features`` (sklearn forbids fit changing init params)."""
    if cat_features is not None:
        return cat_features
    return getattr(estimator, "cat_features", None)


def _resolve_cat_feature_names(cat_features, X):
    """Map any column *names* in ``cat_features`` to integer positions using X's
    column metadata, leaving integer indices untouched.

    Lets a user mark categoricals the same way LightGBM/CatBoost do -- either by
    position (``cat_features=[0, 2]``) or by name (``cat_features=["city",
    "brand"]``), or a mix. Names are resolved against the DataFrame columns at
    fit time, so order changes are handled by the existing predict-time feature-
    name check. Returns ``None`` unchanged; returns the original object when it
    holds no strings (the downstream integer validation then applies)."""
    if cat_features is None:
        return None
    try:
        items = list(cat_features)
    except TypeError:
        return cat_features  # not iterable; let downstream validation report it
    if not any(isinstance(c, str) for c in items):
        return cat_features
    names = _extract_feature_names(X)
    if names is None:
        raise ValueError(
            "cat_features contains column names (strings), but X has no column "
            "names to resolve them against; pass integer indices instead, or "
            "fit on a DataFrame.")
    name_to_idx = {n: i for i, n in enumerate(names)}
    resolved = []
    for c in items:
        if isinstance(c, str):
            if c not in name_to_idx:
                raise ValueError(
                    f"cat_features name {c!r} is not a column of X; columns are "
                    f"{list(names)}.")
            resolved.append(name_to_idx[c])
        else:
            resolved.append(c)
    return resolved


def _check_eval_set(eval_set, n_features):
    """Validate a user-passed ``eval_set`` up front with a named error instead of
    a cryptic IndexError/broadcast failure deep in the booster."""
    if not (isinstance(eval_set, (tuple, list)) and len(eval_set) == 2):
        raise ValueError("eval_set must be a (X_val, y_val) tuple.")
    Xv, yv = eval_set
    shape = getattr(Xv, "shape", None)
    if shape is None or len(shape) != 2:
        shape = np.asarray(Xv, dtype=object).shape
    nfv = shape[1] if len(shape) == 2 else None
    if nfv != n_features:
        raise ValueError(
            f"eval_set X has {nfv} features, but the training data has "
            f"{n_features}; they must match.")
    if len(yv) != shape[0]:
        raise ValueError(
            f"eval_set X and y have inconsistent lengths: {shape[0]} vs "
            f"{len(yv)}.")


def _is_numeric_dtype(dt):
    """True if a column dtype is numeric, across numpy / pandas / polars."""
    try:
        return bool(np.issubdtype(np.dtype(dt), np.number))
    except TypeError:
        pass  # not a numpy-castable dtype (e.g. a polars DataType object)
    is_num = getattr(dt, "is_numeric", None)  # polars DataType
    if callable(is_num):
        try:
            return bool(is_num())
        except Exception:
            pass
    s = str(dt).lower()
    return (any(k in s for k in ("int", "float", "uint", "double", "decimal"))
            and "object" not in s)


def _describe_nonnumeric_columns(X):
    """Name the non-numeric columns of a DataFrame-like X (pandas/polars) so a
    user who forgot ``cat_features`` gets "column 'city' (index 2)" instead of
    a bare ``could not convert string to float: 'NYC'``. Returns [] for inputs
    without column metadata (plain ndarrays)."""
    cols = getattr(X, "columns", None)
    dtypes = getattr(X, "dtypes", None)
    if cols is None or dtypes is None:
        return []
    try:
        col_list, dtype_list = list(cols), list(dtypes)
    except TypeError:
        return []
    return [f"'{c}' (index {i})"
            for i, (c, dt) in enumerate(zip(col_list, dtype_list))
            if not _is_numeric_dtype(dt)]


def _fit_bagged(estimator, X, y, cat_features, eval_set, groups, sample_weight):
    """Train ``estimator.n_ensembles`` bootstrap clones and return them as a list.

    Each member is a clone of ``estimator`` with bagging switched off
    (``n_ensembles=None``) and its own seed, fit on a bootstrap resample (drawn
    with replacement, same size as the training set). Because a member is the
    same estimator class, all per-model machinery â€” binary/multiclass dispatch,
    ``cat_features``, the early-stopping auto-split, temperature scaling â€” is
    reused unchanged, and ``cat_features``/``sample_weight``/``groups`` forward
    naturally (which a ``sklearn.ensemble.Bagging`` wrapper would not do).

    Members are independent, so they fit across ``ensemble_n_jobs`` worker
    processes (default -1: as many workers as the thread budget supports,
    capped at K). The thread budget â€” ``thread_count`` if set, else numba's
    thread count â€” is divided across the workers, so a bagged fit uses the
    same cores a single fit would (numba's sublinear thread scaling is what
    makes K members at budget/K threads faster than K sequential full-budget
    fits: 1.2-2.0x wall-clock on the BAGGING_PLAN.md B4 panel, identical
    models by construction). ``ensemble_n_jobs=1`` restores sequential fits.
    """
    from sklearn.base import clone
    from joblib import Parallel, delayed

    X = as_model_array(X, bool(cat_features))
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    n = X.shape[0]
    K = int(estimator.n_ensembles)
    n_jobs = int(estimator.ensemble_n_jobs)

    if n_jobs == 1:
        n_workers, member_threads = 1, estimator.thread_count
    else:
        budget = estimator.thread_count
        if budget is None:
            import numba
            budget = numba.config.NUMBA_NUM_THREADS
        n_workers = max(1, min(K, budget if n_jobs < 0 else n_jobs))
        member_threads = max(1, int(budget) // n_workers)

    seeds = np.random.default_rng(estimator.random_state).integers(
        0, 2**31 - 1, size=K)

    # Bagged-mode member defaults (benchmarks/BAGGING_PLAN.md B3): averaging
    # tolerates coarser, cheaper members, so params the user left on auto
    # resolve to the tuned member values instead of the single-model ones
    # (PMLB-tuned, holdout-confirmed, decision-suite validated: 54W-17L
    # +0.28% pooled vs the previous bagged defaults at par fit cost).
    # Explicit user values always win. Announced once per fit -- an opt-in
    # bagged fit should never silently train members on different defaults.
    member_defaults = {}
    if estimator.learning_rate is None:
        member_defaults["learning_rate"] = 0.15
    if estimator.colsample is None:
        member_defaults["colsample"] = 0.85
    estimator.member_params_ = dict(member_defaults)
    if member_defaults:
        print("ChimeraBoost bagged mode: member defaults "
              + ", ".join(f"{k}={v}" for k, v in member_defaults.items())
              + " (pass explicit values to override; see docs).", flush=True)

    def _fit_one(seed):
        member = clone(estimator).set_params(
            n_ensembles=None, random_state=int(seed), thread_count=member_threads,
            **member_defaults)
        # SCREEN VARIANT B-samp (BAGGING_PLAN.md): subagging — draw
        # max_samples*n rows WITHOUT replacement instead of a full-size
        # bootstrap. A bootstrap's duplicates are just integer weights
        # (effective sample size ~n/2), so 0.632n unique rows match its
        # information at ~2/3 the training rows. OOB machinery unchanged.
        _MAX_SAMPLES = 0.8
        m = max(1, int(round(_MAX_SAMPLES * n)))
        idx = np.random.default_rng(seed).choice(n, size=m, replace=False)
        wb = None if sample_weight is None else np.asarray(sample_weight)[idx]
        gb = None if groups is None else groups[idx]
        # Use OOB rows as the early-stopping eval set when no explicit eval_set
        # was provided. The alternative (auto-splitting the bootstrap) contaminates
        # the validation set: ~57% of auto-split val rows are duplicates of
        # training rows, so val loss is optimistically low, early stopping fires
        # late, and each member builds ~38% more trees than it should.
        # OOB rows are guaranteed unseen by the member, giving a clean signal.
        if eval_set is None:
            oob_mask = np.ones(n, dtype=np.bool_)
            oob_mask[idx] = False
            oob_idx = np.where(oob_mask)[0]
            # Degenerate case: every row drawn (possible for tiny n). Fall back
            # to letting the member auto-split rather than training with no eval.
            member_eval = (X[oob_idx], y[oob_idx]) if len(oob_idx) > 0 else None
        else:
            member_eval = eval_set
        member.fit(X[idx], y[idx], cat_features=cat_features, eval_set=member_eval,
                   groups=gb, sample_weight=wb)
        return member

    return Parallel(n_jobs=n_workers)(delayed(_fit_one)(s) for s in seeds)


def _make_eval_split(X, y, validation_fraction, random_state,
                     groups=None, stratify=None):
    """Return (train_idx, val_idx) for automatic early-stopping splits.

    Parameters
    ----------
    stratify : array-like or None
        Class labels for stratified splitting (pass for classification tasks).
    groups : array-like or None
        Group membership array (e.g. ``df['subject_id']``).  When supplied,
        groups are kept intact across the split boundary.  For classification,
        ``StratifiedGroupKFold`` is used so class proportions are preserved;
        for regression ``GroupShuffleSplit`` is used.

    Returns ``None`` when the data is too small to carve a valid validation set
    (e.g. tiny ``n``, or a class with too few members for a stratified split).
    The caller treats ``None`` as "train on all rows, early stopping disabled"
    rather than crashing on a degenerate split.
    """
    from sklearn.model_selection import (
        ShuffleSplit,
        StratifiedShuffleSplit,
        GroupShuffleSplit,
        StratifiedGroupKFold,
    )

    # Cheap size precheck: each side of the split needs at least one row per
    # class (or >=2 rows for regression) for the holdout to be usable.
    n = len(y)
    min_per_side = len(np.unique(stratify)) if stratify is not None else 2
    n_val = int(round(n * validation_fraction))
    if n_val < min_per_side or (n - n_val) < min_per_side:
        return None

    try:
        if groups is not None:
            groups = np.asarray(groups)
            if stratify is not None:
                # StratifiedGroupKFold approximates the desired val fraction via
                # n_splits = round(1 / validation_fraction).
                n_splits = max(2, round(1.0 / validation_fraction))
                splitter = StratifiedGroupKFold(n_splits=n_splits)
                train_idx, val_idx = next(
                    splitter.split(X, stratify, groups=groups)
                )
            else:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=validation_fraction,
                    random_state=random_state,
                )
                train_idx, val_idx = next(splitter.split(X, y, groups=groups))
        elif stratify is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X, stratify))
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X))
    except ValueError:
        # Degenerate stratified split (e.g. a class with a single member).
        return None

    return train_idx, val_idx


def _extract_feature_names(X):
    """Return X's column names as a 1-D object array, or None.

    Handles the trap that ``pyarrow.Table.columns`` is the column *data* (a list
    of arrays), not names -- which would otherwise pollute ``feature_names_in_``
    with the data itself. Prefer ``.column_names`` (pyarrow) over ``.columns``
    (pandas/polars), and reject anything that isn't a flat sequence of scalar
    names (e.g. arrays, or pandas MultiIndex tuples)."""
    names = getattr(X, "column_names", None)        # pyarrow.Table
    if names is None:
        names = getattr(X, "columns", None)          # pandas / polars
    if names is None:
        return None
    try:
        arr = np.asarray(list(names), dtype=object)
    except Exception:
        return None
    if arr.ndim != 1 or any(not isinstance(v, str) and hasattr(v, "__len__")
                            for v in arr):
        return None                                  # data masquerading as names
    return arr


def _reject_masked(X, where):
    """Masked arrays silently drop the mask under ``np.asarray`` (the hidden
    values are used), inverting the user's "these are missing" intent. Reject
    with guidance instead of misbehaving silently."""
    if np.ma.isMaskedArray(X):
        raise TypeError(
            f"Masked arrays are not supported ({where}). Convert with "
            "X.filled(np.nan) -- NaN is treated as missing.")


def _validate_fit_input(estimator, X, y, cat_features, sample_weight, *,
                        classification):
    """Shared fit-time input validation + feature-metadata capture.

    Returns the (possibly raveled) ``y`` and sets ``n_features_in_`` (and
    ``feature_names_in_`` for DataFrame input) on ``estimator``. Raises clear
    errors for the common malformed inputs rather than letting them fail
    cryptically deep in numpy/numba. NaN in X is intentionally allowed (treated
    as missing, routed to its own bin); inf, complex, multi-output y, and
    scipy.sparse input are not -- see the README "scikit-learn compatibility" note.
    """
    import scipy.sparse as sp
    from sklearn.exceptions import DataConversionWarning
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None.")
    if sp.issparse(X):
        raise TypeError("Sparse input is not supported; pass a dense array.")
    _reject_masked(X, "fit")
    feature_names = _extract_feature_names(X)
    shape = getattr(X, "shape", None)
    Xc = None
    if shape is None or len(shape) != 2:
        Xc = as_model_array(X, bool(cat_features))
        shape = Xc.shape
    if len(shape) != 2:
        raise ValueError(
            f"Expected a 2D array for X; got {len(shape)}D. Reshape your data, "
            "e.g. X.reshape(-1, 1) for a single feature.")
    n, nf = int(shape[0]), int(shape[1])
    if nf == 0:
        raise ValueError(
            f"X has 0 feature(s) (shape=({n}, 0)) while a minimum of 1 is required.")
    if n == 0:
        raise ValueError(
            f"X has 0 sample(s) (shape=(0, {nf})) while a minimum of 1 is required.")
    if cat_features:
        ci = np.asarray(list(cat_features))
        if ci.size:
            if not np.issubdtype(ci.dtype, np.integer):
                raise ValueError(
                    "cat_features must be integer column indices.")
            if ci.min() < 0 or ci.max() >= nf:
                raise ValueError(
                    f"cat_features index out of range for X with {nf} "
                    f"column(s): {sorted(set(ci.tolist()))}.")
            if len(set(ci.tolist())) != ci.size:
                raise ValueError("cat_features contains duplicate indices.")
    if not cat_features:
        # Check complex BEFORE the float64 cast (which would raise its own
        # TypeError on complex input instead of our clear ValueError).
        Xraw = Xc if Xc is not None else np.asarray(X)
        if np.iscomplexobj(Xraw):
            raise ValueError("Complex data not supported.")
        try:
            # Convert from the original X (not the object-dtype Xraw) so a pandas
            # DataFrame's nullable NA is mapped to np.nan rather than crashing the
            # float cast as an NAType object.
            Xc = as_model_array(X if Xc is None else Xc, want_object=False)
        except (ValueError, TypeError) as e:
            # A non-numeric column (string/category/datetime) in a DataFrame with
            # no cat_features: name the offending columns and point at
            # cat_features. (pandas nullable NA no longer lands here -- it maps to
            # np.nan in as_model_array.) For bare arrays (no column metadata) keep
            # the original numpy error -- some sklearn estimator checks rely on
            # its exact type/message.
            bad = _describe_nonnumeric_columns(X)
            if bad:
                raise ValueError(
                    f"X could not be converted to numeric: column(s) "
                    f"{', '.join(bad)} are non-numeric. Pass their integer "
                    f"positions in cat_features=[...], or encode them first."
                ) from e
            raise
        if np.isinf(Xc).any():
            raise ValueError(
                "X contains infinity. NaN is accepted (treated as missing), but "
                "inf is not -- clip or clean it first.")
    else:
        # cat_features present: cat columns are decoded as strings, but the
        # remaining numeric columns must still be finite. Without this, inf in a
        # numeric column slips silently to the missing bin (binning treats inf as
        # NaN), contradicting the no-cat path's explicit rejection. Check only the
        # numeric columns; the cat columns are not float-castable.
        cat_set = set(int(c) for c in cat_features)
        num_idx = [i for i in range(nf) if i not in cat_set]
        num_block = _numeric_block(Xc if Xc is not None else X, num_idx)
        if num_block is not None and np.isinf(num_block).any():
            raise ValueError(
                "X contains infinity. NaN is accepted (treated as missing), "
                "but inf is not -- clip or clean it first.")
    y = np.asarray(y)
    if y.shape[0] != n:
        raise ValueError(
            f"X and y have inconsistent lengths: X has {n} samples, "
            f"y has {y.shape[0]}.")
    # Ravel a column-vector y (n, 1) with a warning, like sklearn estimators;
    # reject genuine multi-output y.
    if y.ndim == 2:
        if y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,).",
                DataConversionWarning, stacklevel=2)
            y = y.ravel()
        else:
            raise ValueError(
                "Multi-output y is not supported; pass a 1D y of shape "
                "(n_samples,).")
    if classification:
        from sklearn.utils.multiclass import type_of_target
        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(
                "Unknown label type: classification requires discrete class "
                "labels, but y looks continuous (use a regressor instead).")
        if y.dtype.kind in "fc" and \
                not np.isfinite(np.asarray(y, np.float64)).all():
            raise ValueError("y contains NaN or infinity.")
    elif not np.isfinite(np.asarray(y, np.float64)).all():
        raise ValueError("y contains NaN or infinity; targets must be finite.")
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.ndim != 1 or sw.shape[0] != n:
            raise ValueError(
                f"sample_weight must be 1D of length {n}; got shape {sw.shape}.")
        # Non-finite or negative weights, or an all-zero vector, otherwise fit
        # without error and silently yield an all-NaN model (mean-1 weight
        # normalization divides by the weight sum).
        if not np.isfinite(sw).all():
            raise ValueError("sample_weight contains NaN or infinity.")
        if (sw < 0).any():
            raise ValueError("sample_weight must be non-negative.")
        if sw.sum() <= 0:
            raise ValueError("sample_weight sums to zero; at least one weight "
                             "must be positive.")
    estimator.n_features_in_ = nf
    if feature_names is not None:
        estimator.feature_names_in_ = feature_names
    return y


def _check_feature_names_match(estimator, X):
    """Enforce that predict-time feature names agree with fit (name and order).

    A DataFrame whose columns are renamed or *reordered* relative to training
    otherwise yields silently-wrong predictions, since the booster consumes
    columns positionally. Mirrors sklearn: warn when names are present on only
    one side, raise when they disagree. Uses the same ``X.columns`` extraction
    as fit-time capture so the two are directly comparable (pandas/polars)."""
    train_names = getattr(estimator, "feature_names_in_", None)
    x_names = _extract_feature_names(X)
    if train_names is None and x_names is None:
        return
    if train_names is None:
        warnings.warn("X has feature names, but this estimator was fitted "
                      "without feature names.", UserWarning, stacklevel=3)
        return
    if x_names is None:
        warnings.warn("This estimator was fitted with feature names, but X was "
                      "passed without feature names.", UserWarning, stacklevel=3)
        return
    if not np.array_equal(np.asarray(train_names, dtype=object), x_names):
        raise ValueError(
            "The feature names of X do not match those seen during fit. "
            f"Fitted on {list(train_names)}, got {list(x_names)}. Columns must "
            "match in name and order (no automatic reordering is performed).")


def _assume_finite():
    """Honor scikit-learn's global ``assume_finite`` config. When a user sets
    ``sklearn.set_config(assume_finite=True)`` (or uses ``config_context``), the
    O(n) predict-time finiteness scan is skipped for maximum inference
    throughput -- the same escape hatch sklearn's own ``check_array`` offers."""
    try:
        from sklearn import get_config
        return bool(get_config().get("assume_finite", False))
    except Exception:
        return False


def _numeric_block(X, num_idx):
    """The numeric columns of X (positions ``num_idx``) as a float64 array, or
    None if they aren't float-castable. Used for the inf check when categoricals
    are present: it selects *only* the numeric columns so the (often string-
    heavy) categorical columns aren't dragged through an expensive object
    conversion. Maps pandas nullable NA to np.nan like the model's own path."""
    if not num_idx:
        return None
    try:
        iloc = getattr(X, "iloc", None)
        if iloc is not None and hasattr(X, "dtypes"):     # pandas DataFrame
            sub = iloc[:, num_idx]
            try:
                return sub.to_numpy(dtype=np.float64, na_value=np.nan)
            except TypeError:
                return np.asarray(sub, dtype=np.float64)
        return np.asarray(np.asarray(X)[:, num_idx], dtype=np.float64)
    except (ValueError, TypeError):
        return None  # a "numeric" column holds strings; surfaced downstream


def _fitted_prep(estimator):
    """The fitted FeaturePreprocessor of a model (or the first bagged member),
    or None if not available."""
    m = getattr(estimator, "model_", None)
    if m is None:
        members = getattr(estimator, "estimators_", None)
        m = members[0].model_ if members else None
    return getattr(m, "prep_", None)


def _was_fit_with_cats(estimator):
    """True if the fitted model used categorical features (so X is the object
    path and a whole-matrix numeric finiteness check does not apply)."""
    return bool(getattr(_fitted_prep(estimator), "cat_features_", None))


def _check_predict_input(estimator, X):
    """Raise NotFittedError if unfitted, then validate X is 2D with the same
    number of features as training -- preventing silently-wrong predictions on
    mismatched input. Messages match scikit-learn's wording for compatibility."""
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(estimator)
    # Enforce feature-name agreement with fit (reuse sklearn's logic): a
    # DataFrame whose columns are renamed or *reordered* relative to training
    # otherwise produces silently-wrong predictions. Warns when names are
    # present on only one side, raises when they disagree -- like every sklearn
    # estimator.
    _check_feature_names_match(estimator, X)
    import scipy.sparse as sp
    if sp.issparse(X):
        raise TypeError("Sparse input is not supported; pass a dense array.")
    _reject_masked(X, "predict")
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        shape = np.asarray(X, dtype=object).shape
    if len(shape) != 2:
        raise ValueError(
            f"Expected a 2D array for X; got {len(shape)}D. Reshape your data, "
            "e.g. X.reshape(1, -1) for a single sample.")
    if shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {shape[1]} features, but {type(estimator).__name__} is "
            f"expecting {estimator.n_features_in_} features as input.")
    # Reject inf at predict for the numeric path, mirroring fit (which rejects
    # it). Without this, an inf serving value is silently routed to the missing
    # bin and returns the "missing" prediction with no error. This is the only
    # O(n) check on the hot predict path, so it is skippable via sklearn's
    # ``assume_finite`` config for latency-critical serving.
    if not _assume_finite():
        if not _was_fit_with_cats(estimator):
            try:
                Xf = as_model_array(X, want_object=False)
            except (ValueError, TypeError):
                Xf = None
        else:
            # Categorical fit: only the numeric columns need to be finite (the
            # cat columns are strings). Pull their positions from the fitted
            # preprocessor and check just those, mirroring the fit-time check.
            num_idx = getattr(_fitted_prep(estimator), "num_features_", None)
            Xf = _numeric_block(X, num_idx)
        if Xf is not None and np.isinf(Xf).any():
            raise ValueError(
                "X contains infinity. NaN is accepted (treated as missing), but "
                "inf is not -- clip or clean it first.")


def _auto_min_child_weight(n_train):
    """Size-adaptive ``min_child_weight`` used when the classifier leaves it None.

    Oblivious trees UNDERFIT large data at the historical mcw=1: the shared-split
    veto amplifies the min-leaf constraint (one sparse leaf among 2**depth vetoes
    the whole level), so they want a lower min-leaf than leaf-wise trees -- which
    is why CatBoost uses min_data_in_leaf=1. But mcw~0 OVERFITS small data,
    because (unlike CatBoost) we run plain boosting without ordered-boosting
    regularization. So fade the veto by training size: keep the full veto below
    ~500 rows, drop it above ~2000, linear between. The midpoint (~1250 rows ->
    ~20 samples/leaf at depth 6) lines up with the field-standard
    min_data_in_leaf=20.
    """
    return float(np.clip((2000.0 - n_train) / 1500.0, 0.0, 1.0))


# Pairwise categorical combinations help when the target depends on categorical
# INTERACTIONS, but on mixed data the synthetic combo columns crowd out the
# numeric features that want to split (sign-tested: all-categorical car/kr-vs-kp
# gain +60%+, mixed sets regress). So the auto-default enables them ONLY when the
# data is entirely categorical -- the precise condition under which they help
# without a downside. The two caps below are resource guards (a wide all-cat
# dataset generates C(n_cat, 2) combo columns, each target-encoded over every
# row), NOT accuracy knobs: above them the user can still opt in explicitly.
_AUTO_CAT_COMBO_MAX_PAIRS = 1000        # ceiling on C(n_cat, 2) combo columns
_AUTO_CAT_COMBO_MAX_CELLS = 5e7         # ceiling on pairs * n_samples (memory)


def _auto_cat_combinations(cat_features, n_features, n_samples):
    """Resolve ``cat_combinations=None``: True only for (tractable) all-categorical
    data. ``cat_features`` is the resolved integer-index list (or None)."""
    if cat_features is None or len(cat_features) == 0:
        return False
    n_cat = len(cat_features)
    if n_cat < 2 or n_cat != n_features:
        return False
    n_pairs = n_cat * (n_cat - 1) // 2
    if n_pairs > _AUTO_CAT_COMBO_MAX_PAIRS:
        return False
    if n_pairs * n_samples > _AUTO_CAT_COMBO_MAX_CELLS:
        return False
    return True


# Numeric cross features: pair the CROSS_TOP_M most important numeric columns
# of the base fit; each pair contributes a difference and a product column.
# Selection (base vs augmented, on the ES validation split) needs enough rows
# for the val signal to be trustworthy -- below CROSS_MIN_SAMPLES the val set
# is too small to referee and small data overfits extra columns first.
CROSS_TOP_M = 6
CROSS_MIN_SAMPLES = 2000


def _cross_candidate_pairs(importances, cat_features, n_features):
    """Candidate (i, j, op) cross features from base-fit importances.

    Oblivious trees approximate numeric interactions with a depth-limited
    staircase (one shared split per level); a difference column makes the
    ``x_i < x_j`` boundary one split and a product column captures
    multiplicative structure. Pairs are the C(m, 2) combinations of the top-m
    numeric features by split-gain importance -- interactions among features
    the trees already use are the plausible ones, and irrelevant crosses cost
    only fit time (split search ignores them)."""
    cat = set(cat_features or [])
    num_idx = [i for i in range(n_features) if i not in cat]
    if len(num_idx) < 2:
        return []
    imp = np.asarray(importances, dtype=np.float64)
    key = np.zeros(n_features)
    key[:imp.shape[0]] = imp
    top = sorted(num_idx, key=lambda i: -key[i])[:CROSS_TOP_M]
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            i, j = top[a], top[b]
            pairs.append((i, j, "diff"))
            pairs.append((i, j, "prod"))
    return pairs


def _best_val(booster):
    """Best validation loss a fitted booster reached (inf when no history)."""
    return min(booster.valid_history_) if booster.valid_history_ else np.inf


def _stop_after(k):
    """Fit callback halting boosting after k rounds (selection auditions)."""
    def cb(iteration, train_loss, val_loss, model):
        return iteration + 1 >= k
    return cb


def _stop_if_behind(k, target_best):
    """Fit callback killing a challenger at round k unless its best validation
    loss has beaten ``target_best`` by then (the raced-selection rule: the
    winner at the shared budget continues, the loser stops). A best-so-far
    only improves, so a challenger ahead at k is never stopped later."""
    state = {"best": np.inf}

    def cb(iteration, train_loss, val_loss, model):
        if val_loss is not None and val_loss < state["best"]:
            state["best"] = val_loss
        return iteration + 1 >= k and not state["best"] < target_best
    return cb


def _add_callback(callbacks, extra):
    """Compose the user callbacks argument (None, a callable, or a sequence)
    with one internal callback; extra=None returns callbacks unchanged."""
    if extra is None:
        return callbacks
    base = ([] if callbacks is None else list(callbacks)
            if isinstance(callbacks, (list, tuple)) else [callbacks])
    return base + [extra]


class ChimeraBoostRegressor(RegressorMixin, BaseEstimator):
    """Gradient boosted oblivious trees for regression.

    A scikit-learn compatible regressor supporting squared-error, absolute-error,
    and quantile losses, native categorical features, sample weights, bagging, and
    exact SHAP attributions.

    Parameters
    ----------
    n_estimators : int, default 2000
        Maximum number of boosting rounds (trees). With ``early_stopping`` on,
        this is an upper bound and the best round is selected automatically.
    learning_rate : float or None, default None
        Shrinkage applied to each tree. ``None`` resolves to 0.1 when early
        stopping is active.
    depth : int or None, default None
        Depth of each oblivious tree; a depth-d tree makes d splits. ``None``
        resolves to 6 for squared-error/absolute-error losses, and to 4 for
        ``loss="Quantile"`` -- estimating an extreme conditional quantile from a
        leaf needs more samples per leaf than estimating a mean, so deep trees
        overfit the tails and the predicted quantiles collapse toward the median.
        Raise to 8-10 for large, interaction-heavy problems; set it explicitly to
        override the per-loss default.
    l2_leaf_reg : float, default 1.0
        L2 regularization on leaf values.
    max_bins : int, default 128
        Histogram bins per numeric feature.
    subsample : float, default 1.0
        Row subsampling fraction per tree. Below 1.0, rows are drawn by Minimum
        Variance Sampling (gradient-weighted, unbiased) rather than uniformly.
    colsample : float or None, default None
        Fraction of features eligible for each tree. ``None`` resolves to
        1.0 for a single model and to the bagged-member default 0.85 inside
        ``n_ensembles > 1`` fits (see ``member_params_``).
    cat_smoothing : float, default 1.0
        Prior strength for ordered target statistics; higher shrinks rare
        categories harder toward the global mean. Must be > 0 -- it is the
        Bayesian pseudocount in the encoder denominator, so 0 is undefined.
    cat_n_permutations : int, default 4
        Number of random orderings averaged by the ordered target encoder.
    early_stopping_rounds : int or None, default None
        Rounds without validation improvement before stopping. ``None`` becomes 50
        when early stopping is active.
    loss : {"RMSE", "MAE", "Quantile"}, default "RMSE"
        Training objective. Set the level with ``alpha`` for ``"Quantile"``.
    alpha : float, default 0.5
        Quantile level for ``loss="Quantile"`` (e.g. 0.9 for the 90th percentile).
    min_child_weight : float, default 1.0
        Minimum total hessian required on each side of a split.
    thread_count : int or None, default None
        numba thread count. ``None`` or -1 uses all detected cores.
    random_state : int or None, default None
        Seed for reproducibility (deterministic for a fixed ``thread_count``).
    verbose : bool, default False
        Print per-round train and validation metrics.
    ordered_boosting : bool, default False
        Use the leave-one-out leaf training step instead of plain Newton updates.
    cat_combinations : bool or None, default None
        Add all pairwise categorical-by-categorical features. ``None`` enables
        them automatically only when the data is entirely categorical (where the
        interaction columns help without crowding out numeric splits); set
        ``True``/``False`` to force it on/off.
    leaf_estimation_iterations : int, default 1
        Newton refinement steps per leaf.
    linear_leaves : bool or None, default None
        Fit a ridge linear model per leaf over the numeric split features instead
        of a constant value, adding local slope where step leaves underfit. Leaves
        with too few rows fall back to a constant. Not available with MAE or
        quantile loss. ``None`` (the default) = validation-selected: both
        variants are fit and the one with the lower validation loss is kept
        (~2x fit time; requires an early-stopping split or ``eval_set``, RMSE
        loss, and >= 1000 rows â€” otherwise constant leaves are used). Set
        ``True``/``False`` to force one variant and skip the double fit.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    cross_features : bool or None, default None
        Numeric interaction columns. ``None`` (the default) and ``True`` refit
        with difference and product columns for the pairs of the top numeric
        features of the base fit and keep whichever model reaches the lower
        validation loss (``cross_features_selected_`` records the outcome,
        ``cross_pairs_`` the columns kept); applies to RMSE loss with >= 2000
        rows and >= 2 numeric features, and is skipped otherwise. ``False``
        turns it off. Oblivious trees can only staircase a numeric interaction
        such as ``x_i < x_j``; a cross column makes it a single split. Costs
        up to ~2x fit time when the refit runs.
    selection_rounds : int or None, default 100
        Round budget for the internal selection fits. The constant/linear-leaf
        variants and the pre-cross base fit run at most this many rounds
        (auditions, judged on their best validation loss within the budget);
        the winning candidate continues to full early stopping, and the
        audition winner is refit in full only when the cross-augmented model
        loses or cross features do not apply. An audition that early-stops
        before the budget is the full fit already (no extra cost). ``None``
        runs every variant to full early stopping instead (the pre-0.15
        behavior, ~1.5x slower fits); an audition can occasionally pick a
        different variant than full fits would.
    early_stopping : bool, default True
        Hold out a validation split and stop when its score stops improving.
    validation_fraction : float, default 0.2
        Validation fraction used when ``early_stopping`` is on and no ``eval_set``
        is passed to ``fit``.
    n_ensembles : int or None, default None
        Number of bagged members. ``None`` or 1 trains a single model; >= 2
        averages independent members fit on bootstrap resamples.
    ensemble_n_jobs : int, default -1
        Worker processes fitting ensemble members concurrently, each on an
        equal share of the thread budget (same total cores as a single fit;
        models are identical either way, wall-clock 1.2-2x faster). -1 sizes
        the pool from the budget, capped at ``n_ensembles``; 1 fits members
        sequentially, each with the full budget.
    cat_features : list of int or str, or None, default None
        Default categorical columns, given as integer positions and/or column
        names (names resolved against the DataFrame at fit). Used when ``fit`` is
        called without its own ``cat_features`` (the fit argument overrides).
        Provided as a constructor argument so ``GridSearchCV``/``Pipeline`` can
        carry it.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Split-gain importance per input feature, normalized to sum to 1.
    best_iteration_ : int
        Number of trees retained after early stopping.
    expected_value_ : float
        SHAP baseline (mean prediction over the background); set after calling
        ``shap_values``.
    estimators_ : list or None
        Fitted members when ``n_ensembles > 1``, otherwise ``None``.
    member_params_ : dict
        Bagged-mode member defaults that were auto-applied (params the user
        left on auto resolve to tuned member values inside a bag; explicit
        values always win). Set only when ``n_ensembles > 1``.
    quantile_offset_ : float
        Split-conformal correction added to every prediction when
        ``loss="Quantile"`` and a validation split was available: the conformal
        order statistic of the validation residuals, restoring the nominal
        coverage that learning-rate shrinkage of the per-leaf quantile steps
        otherwise starves. 0.0 for other losses or without a validation set.
    linear_leaves_selected_ : bool or None
        With ``linear_leaves=None``, whether the linear-leaf variant won the
        validation selection. ``None`` when no selection took place.
    """

    def __init__(self, n_estimators=2000, learning_rate=None, depth=None,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0, colsample=None,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 loss="RMSE", alpha=0.5, min_child_weight=1.0, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting=False,
                 cat_combinations=None, leaf_estimation_iterations=1,
                 linear_leaves=None, linear_lambda=1.0, cross_features=None,
                 selection_rounds=100,
                 early_stopping=True, validation_fraction=0.2,
                 n_ensembles=None, ensemble_n_jobs=-1, cat_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.cat_n_permutations = cat_n_permutations
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        self.loss = loss
        self.alpha = alpha
        self.min_child_weight = min_child_weight
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.cat_combinations = cat_combinations
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.cross_features = cross_features
        self.selection_rounds = selection_rounds
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, callbacks=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or str, or None
            Columns to treat as categoricals, given as integer positions and/or
            column names (names resolved against the DataFrame). Falls back to the
            ``cat_features`` constructor argument when not given here; passing it
            here overrides the constructor value. (The constructor form lets
            ``GridSearchCV``/``Pipeline`` carry it, which a fit-only kwarg can't.)
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set.  When provided, automatic splitting is
            skipped regardless of the *early_stopping* setting.
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples (e.g. ``df['subject_id']``).  When
            supplied and *early_stopping* triggers an automatic split, groups
            are kept intact across the train/validation boundary using
            ``GroupShuffleSplit``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.  Only applied
            to the training set; the validation eval metric is always unweighted.
        callbacks : callable or list of callable, or None
            Per-round fit hooks ``cb(iteration, train_loss, val_loss, model)``;
            a callback returning True requests an early stop. Used for live
            validation-curve capture and instrumentation. Not supported with
            ``n_ensembles > 1`` (members fit in parallel worker processes).
        """
        cat_features = _resolve_cat_features(self, cat_features)
        cat_features = _resolve_cat_feature_names(cat_features, X)
        _validate_hyperparams(self)
        y = _validate_fit_input(self, X, y, cat_features, sample_weight,
                                classification=False)
        if eval_set is not None:
            _check_eval_set(eval_set, self.n_features_in_)
        if self.n_ensembles and self.n_ensembles > 1:
            if callbacks is not None:
                raise ValueError(
                    "callbacks are not supported with n_ensembles > 1.")
            self.estimators_ = _fit_bagged(self, X, y, cat_features, eval_set,
                                           groups, sample_weight)
            return self
        self.estimators_ = None
        return self._fit_single(X, y, cat_features, eval_set, groups,
                                sample_weight, callbacks)

    def __sklearn_is_fitted__(self):
        return (hasattr(self, "model_")
                or getattr(self, "estimators_", None) is not None)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN routed to a missing bin
        tags.input_tags.sparse = False
        return tags

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight,
                    callbacks=None):
        """Fit one (non-bagged) model on the data as given."""
        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        # linear_leaves is silently dropped by the booster for MAE/Quantile
        # (their leaf values are the residual median/quantile, not a Newton step
        # a ridge slope could refine). Warn so it isn't mistaken for active.
        if self.linear_leaves and self.loss in ("MAE", "Quantile"):
            warnings.warn(
                f"linear_leaves is not supported with loss={self.loss!r} and "
                "will be ignored.", UserWarning, stacklevel=2)

        es_active = bool(self.early_stopping)
        if es_active and eval_set is None:
            split = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=None,
            )
            if split is None:
                es_active = False  # data too small to hold out a val set
            else:
                train_idx, val_idx = split
                eval_set = (X[val_idx], y[val_idx])
                X, y = X[train_idx], y[train_idx]
                if sample_weight is not None:
                    sample_weight = sample_weight[train_idx]

        # If early stopping is active but patience not explicitly set, use 50.
        # 50 beats 10 on 25/34 benchmark datasets (lr=0.1 keeps improving past a
        # 10-round plateau); see benchmarks/investigate_early_stopping.py.
        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50

        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        # Resolve the loss-adaptive depth default (see the `depth` docstring):
        # 6 for RMSE/MAE (unchanged), 4 for Quantile, where deep leaves overfit
        # the tail quantile and predictions collapse toward the median.
        if kw.get("depth") is None:
            kw["depth"] = 4 if self.loss == "Quantile" else 6
        # min_child_weight is a no-op for regression in [0, 1] (a non-empty child
        # always holds >=1 sample = hess >= 1); resolve an explicit None to 1.0.
        if kw.get("min_child_weight") is None:
            kw["min_child_weight"] = 1.0
        # colsample None = auto: full columns for a single model (the bagged
        # path resolves members to 0.85 before this runs; see _fit_bagged).
        if kw.get("colsample") is None:
            kw["colsample"] = 1.0
        # Auto-resolve cat_combinations: on only for tractable all-categorical data.
        if kw.get("cat_combinations") is None:
            kw["cat_combinations"] = _auto_cat_combinations(
                cat_features, self.n_features_in_, len(X))
        # linear_leaves=None -> validation-selected: fit constant-leaf and
        # linear-leaf variants and keep whichever reaches the lower validation
        # loss. Full-Grinsztajn breadth showed fixed linear leaves are a wash
        # for regression (16W/12L) with real casualties; per-dataset selection
        # on the already-held-out ES split banks the wins without them (the
        # same post-fit-decision pattern as temperature scaling / conformal
        # quantiles). Selection needs a validation set, RMSE (MAE/Quantile
        # override leaf values), and enough rows for linear leaves to engage.
        ll = kw.pop("linear_leaves")
        select_ll = (ll is None and self.loss == "RMSE" and eval_set is not None
                     and len(X) >= LINEAR_LEAVES_MIN_SAMPLES)
        # Cross-features applicability, decidable before any fit: pair
        # candidates exist iff there are >= 2 numeric columns.
        n_cats = len(cat_features) if cat_features else 0
        cross_ok = (self.cross_features is not False and self.loss == "RMSE"
                    and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES
                    and X.shape[1] - n_cats >= 2)

        def _fit_booster(linear, cross_pairs=None, stop=None):
            b = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                 linear_leaves=linear, cross_pairs=cross_pairs,
                                 **kw)
            b.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                  sample_weight=sample_weight,
                  callbacks=_add_callback(callbacks, stop))
            return b

        self.linear_leaves_selected_ = None
        self.cross_features_selected_ = None
        self.cross_pairs_ = None
        if self.selection_rounds is not None and (select_ll or cross_ok):
            # Cheap selection (benchmarks/PARETO_PLAN.md step 2, fallback
            # design): every selection fit runs as a short audition; the
            # cross-augmented candidate -- which wins the full selection on
            # the vast majority of datasets -- gets the one full fit, and the
            # audition winner is refit in full only when the augmented model
            # loses or cross features do not apply.
            stop = _stop_after(self.selection_rounds)
            if select_ll:
                const = _fit_booster(False, stop=stop)
                lin = _fit_booster(True, stop=stop)
                # Tie goes to constant leaves, as in the full selection.
                self.linear_leaves_selected_ = _best_val(lin) < _best_val(const)
                audition = lin if self.linear_leaves_selected_ else const
                base_linear = self.linear_leaves_selected_
            else:
                base_linear = bool(ll)
                audition = _fit_booster(base_linear, stop=stop)
            # An audition that early-stopped on its own BEFORE the cap already
            # IS the full fit (same config/seed/curve) -- refitting it would
            # only re-pay for the identical model. Refit only when truncated.
            capped = len(audition.valid_history_) >= self.selection_rounds
            pairs = (_cross_candidate_pairs(audition.feature_importances_,
                                            cat_features, X.shape[1])
                     if cross_ok else [])
            if pairs:
                # Symmetric race at the shared budget (the rule the step-0
                # race sim validated): both candidates are judged on their
                # best val loss within the first selection_rounds; a trailing
                # augmented fit is killed at the budget, a leading one
                # continues to its own full early stop. Comparing the
                # augmented fit's FULL best against a capped audition would
                # bias the selection toward it (its extra rounds are not
                # evidence the audition couldn't have matched).
                base_best = _best_val(audition)
                aug = _fit_booster(base_linear, cross_pairs=pairs,
                                   stop=_stop_if_behind(self.selection_rounds,
                                                        base_best))
                self.cross_features_selected_ = (
                    min(aug.valid_history_[:self.selection_rounds])
                    < base_best) if aug.valid_history_ else False
                if self.cross_features_selected_:
                    self.model_ = aug
                    self.cross_pairs_ = pairs
                else:
                    self.model_ = (_fit_booster(base_linear) if capped
                                   else audition)
            else:
                self.model_ = (_fit_booster(base_linear) if capped
                               else audition)
        elif select_ll:
            const = _fit_booster(False)
            lin = _fit_booster(True)
            # Each variant early-stops itself; compare the best validation loss
            # reached. Tie goes to constant leaves (cheaper predictions).
            best_const = min(const.valid_history_) if const.valid_history_ else np.inf
            best_lin = min(lin.valid_history_) if lin.valid_history_ else np.inf
            self.model_ = lin if best_lin < best_const else const
            self.linear_leaves_selected_ = self.model_ is lin
        else:
            self.model_ = _fit_booster(bool(ll))

        # Numeric cross features (default-on auto): refit with
        # difference/product columns for the top numeric feature pairs of the
        # base fit and keep whichever model reaches the lower validation loss
        # -- the same selection-on-the-ES-split pattern as linear_leaves
        # above. Evidence and rationale: oblivious trees staircase numeric
        # interactions (benchmarks/probe_cross_features.py; Grinsztajn A/B
        # 51W/8L, mean +1.5%); selection dodges the variance cases. RMSE-only
        # (the probed loss). None (auto) and True behave the same here.
        # (Already handled above when a selection_rounds audition ran.)
        if (self.selection_rounds is None and cross_ok):
            pairs = _cross_candidate_pairs(
                self.model_.feature_importances_, cat_features, X.shape[1])
            if pairs:
                base_linear = (self.linear_leaves_selected_
                               if select_ll else bool(ll))
                aug = _fit_booster(base_linear, cross_pairs=pairs)
                self.cross_features_selected_ = _best_val(aug) < _best_val(self.model_)
                if self.cross_features_selected_:
                    self.model_ = aug
                    self.cross_pairs_ = pairs

        # Conformal quantile correction on the validation split -- the
        # regression analog of the classifier's temperature scaling. Boosting
        # under-disperses quantiles: each round's per-leaf quantile step is
        # shrunk by the learning rate, so the additive model converges to the
        # tail slowly and early stopping cuts it short (predictions collapse
        # toward the median). The fix is the split-conformal step: shift every
        # prediction by the k-th order statistic of the validation residuals,
        # k = ceil((n+1) * alpha) -- the standard conformal rank, which also
        # minimizes pinball loss over all constant shifts, so accuracy and
        # coverage improve together. Distribution-free marginal coverage on
        # exchangeable data (Romano, Patterson & Candes 2019).
        self.quantile_offset_ = 0.0
        if self.loss == "Quantile" and eval_set is not None:
            resid = np.sort(
                np.asarray(eval_set[1], dtype=np.float64)
                - self.model_.predict_raw(eval_set[0]))
            k = min(int(np.ceil((resid.shape[0] + 1) * self.alpha)),
                    resid.shape[0])
            if k >= 1:
                self.quantile_offset_ = float(resid[k - 1])
        return self

    def predict(self, X):
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            # Members apply their own conformal offsets inside m.predict.
            return np.mean([m.predict(X) for m in self.estimators_], axis=0)
        return self.model_.predict_raw(X) + self.quantile_offset_

    def staged_predict(self, X):
        """Yield the prediction after each successive tree (the conformal
        quantile offset, a post-fit constant, is included in every stage so the
        final stage equals ``predict``)."""
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            raise NotImplementedError("staged_predict is not defined for a "
                                      "bagged ensemble (n_ensembles > 1).")
        for staged in self.model_.staged_predict_raw(X):
            yield staged + self.quantile_offset_

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

    @property
    def validation_history_(self):
        """Per-round validation loss recorded during ``fit`` (RMSE-space loss for
        regression), as a list whose length is the number of rounds run. Empty
        when no ``eval_set`` / early-stopping split was available; for a bagged
        model (``n_ensembles > 1``) a list of the members' histories."""
        if self.estimators_ is not None:
            return [m.model_.valid_history_ for m in self.estimators_]
        return self.model_.valid_history_

    @property
    def feature_importances_(self):
        if self.estimators_ is not None:
            return np.mean([m.feature_importances_ for m in self.estimators_],
                           axis=0)
        return self.model_.feature_importances_

    def shap_values(self, X, X_background=None):
        """Exact interventional TreeSHAP contributions to the predicted target.

        Returns an array of shape ``(n_samples, n_features)`` whose rows sum to
        ``predict(X) - expected_value_``, where ``expected_value_`` (set as an
        attribute by this call) is the mean prediction over the background. Each
        entry is a feature's signed additive contribution to the prediction;
        linear-leaf slopes are included exactly. Averaged across the bag when
        ``n_ensembles > 1`` (the bag prediction is the members' mean, so the
        averaged attribution stays exact). ``X_background`` overrides the
        reference distribution (default: a sample of the training data)."""
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            out = [m.model_.shap_values(X, background=X_background)
                   for m in self.estimators_]
            # Fold each member's conformal quantile offset into the baseline so
            # rows still sum to predict(X) - expected_value_.
            self.expected_value_ = float(np.mean(
                [b + m.quantile_offset_ for m, (_, b) in zip(self.estimators_, out)]))
            return np.mean([p for p, _ in out], axis=0)
        phi, base = self.model_.shap_values(X, background=X_background)
        # The conformal quantile offset is a constant shift; it belongs to the
        # baseline, keeping rows summing to predict(X) - expected_value_.
        self.expected_value_ = base + self.quantile_offset_
        return phi


class ChimeraBoostClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosted oblivious trees for classification.

    A scikit-learn compatible classifier. Uses binary logloss for 2 classes and
    softmax for 3 or more, chosen automatically. ``predict_proba`` is temperature
    scaled on the validation split for calibrated probabilities.

    Parameters
    ----------
    n_estimators : int, default 2000
        Maximum number of boosting rounds (trees). With ``early_stopping`` on,
        this is an upper bound and the best round is selected automatically.
    learning_rate : float or None, default None
        Shrinkage applied to each tree. ``None`` resolves to 0.1 when early
        stopping is active.
    depth : int, default 6
        Depth of each oblivious tree; a depth-d tree makes d splits.
    l2_leaf_reg : float, default 1.0
        L2 regularization on leaf values.
    max_bins : int, default 128
        Histogram bins per numeric feature.
    subsample : float, default 1.0
        Row subsampling fraction per tree (Minimum Variance Sampling below 1.0).
    colsample : float or None, default None
        Fraction of features eligible for each tree. ``None`` resolves to
        1.0 for a single model and to the bagged-member default 0.85 inside
        ``n_ensembles > 1`` fits (see ``member_params_``).
    cat_smoothing : float, default 1.0
        Prior strength for ordered target statistics. Must be > 0 (a Bayesian
        pseudocount in the encoder denominator; 0 is undefined).
    cat_n_permutations : int, default 4
        Number of random orderings averaged by the ordered target encoder.
    early_stopping_rounds : int or None, default None
        Rounds without validation improvement before stopping. ``None`` becomes 50
        when early stopping is active.
    min_child_weight : float or None, default None
        Minimum total hessian on each side of a split. ``None`` resolves to a
        size-adaptive value: a full veto below ~500 rows, off above ~2000.
    thread_count : int or None, default None
        numba thread count. ``None`` or -1 uses all detected cores.
    random_state : int or None, default None
        Seed for reproducibility (deterministic for a fixed ``thread_count``).
    verbose : bool, default False
        Print per-round train and validation metrics.
    ordered_boosting : bool, default False
        Use the leave-one-out leaf training step instead of plain Newton updates.
    cat_combinations : bool or None, default None
        Add all pairwise categorical-by-categorical features. ``None`` enables
        them automatically only when the data is entirely categorical (where the
        interaction columns help without crowding out numeric splits); set
        ``True``/``False`` to force it on/off.
    leaf_estimation_iterations : int, default 3
        Newton refinement steps per leaf.
    linear_leaves : bool or None, default None
        Fit a ridge linear model per leaf over the numeric split features instead
        of a constant. ``None`` enables it for binary classification and disables
        it for multiclass (where it is unsupported). Below ~1000 rows it falls
        back to constant leaves.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    cross_features : bool or None, default None
        Numeric interaction columns. ``None`` (the default) refits binary
        models with difference and product columns for the pairs of the top
        numeric features of the base fit and keeps whichever model reaches
        the lower validation loss (``cross_features_selected_`` records the
        outcome, ``cross_pairs_`` the columns kept); needs >= 2000 rows and
        >= 2 numeric features, and silently skips multiclass (unsupported).
        ``False`` turns it off; explicit ``True`` raises for multiclass.
        Costs up to ~2x fit time when the refit runs.
    selection_rounds : int or None, default 100
        Round budget for the pre-cross base fit when the cross-features refit
        will run (binary only). The base fit is an audition capped at this
        many rounds; the candidates are judged on their best validation loss
        within the budget, the winner continues to full early stopping, and
        the base is refit in full only if the augmented model loses after
        being truncated by the cap. ``None`` runs the base fit to full early
        stopping instead (the pre-0.15 behavior). Multiclass fits are
        unaffected (no selection exists there yet).
    early_stopping : bool, default True
        Hold out a stratified validation split and stop when it stops improving.
        ``StratifiedGroupKFold`` is used when ``groups`` is passed to ``fit``.
    validation_fraction : float, default 0.2
        Validation fraction used when ``early_stopping`` is on and no ``eval_set``
        is passed to ``fit``.
    n_ensembles : int or None, default None
        Number of bagged members. ``None`` or 1 trains a single model; >= 2
        soft-votes the calibrated probabilities of members fit on bootstraps.
    ensemble_n_jobs : int, default -1
        Worker processes fitting ensemble members concurrently, each on an
        equal share of the thread budget (same total cores as a single fit;
        models are identical either way, wall-clock 1.2-2x faster). -1 sizes
        the pool from the budget, capped at ``n_ensembles``; 1 fits members
        sequentially, each with the full budget.
    cat_features : list of int or str, or None, default None
        Default categorical columns, given as integer positions and/or column
        names (names resolved against the DataFrame at fit). Used when ``fit`` is
        called without its own ``cat_features`` (the fit argument overrides).
        Provided as a constructor argument so ``GridSearchCV``/``Pipeline`` can
        carry it.

    Attributes
    ----------
    classes_ : ndarray
        Class labels, in the column order of ``predict_proba``.
    feature_importances_ : ndarray of shape (n_features,)
        Split-gain importance per input feature, normalized to sum to 1.
    best_iteration_ : int
        Number of trees retained after early stopping.
    temperature_ : float
        Fitted calibration temperature; > 1 means raw scores were over-confident.
    expected_value_ : float
        SHAP baseline (binary only); set after calling ``shap_values``.
    estimators_ : list or None
        Fitted members when ``n_ensembles > 1``, otherwise ``None``.
    member_params_ : dict
        Bagged-mode member defaults that were auto-applied (params the user
        left on auto resolve to tuned member values inside a bag; explicit
        values always win). Set only when ``n_ensembles > 1``.
    """

    def __init__(self, n_estimators=2000, learning_rate=None, depth=6,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0, colsample=None,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 min_child_weight=None, thread_count=None, random_state=None,
                 verbose=False, ordered_boosting=False,
                 cat_combinations=None, leaf_estimation_iterations=3,
                 linear_leaves=None, linear_lambda=1.0, cross_features=None,
                 selection_rounds=100,
                 early_stopping=True, validation_fraction=0.2,
                 n_ensembles=None, ensemble_n_jobs=-1, cat_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.cat_n_permutations = cat_n_permutations
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        self.min_child_weight = min_child_weight
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.cat_combinations = cat_combinations
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.cross_features = cross_features
        self.selection_rounds = selection_rounds
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, callbacks=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or str, or None
            Columns to treat as categoricals, given as integer positions and/or
            column names (names resolved against the DataFrame). Falls back to the
            ``cat_features`` constructor argument when not given here; passing it
            here overrides the constructor value. (The constructor form lets
            ``GridSearchCV``/``Pipeline`` carry it, which a fit-only kwarg can't.)
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set with original class labels.  When provided,
            automatic splitting is skipped.
        groups : array-like of shape (n_samples,) or None
            Group labels (e.g. ``df['subject_id']``).  When supplied and early
            stopping triggers an automatic split, ``StratifiedGroupKFold`` keeps
            groups intact and class proportions balanced across the split.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.  Only applied
            to the training set; the validation eval metric is always unweighted.
        callbacks : callable or list of callable, or None
            Per-round fit hooks ``cb(iteration, train_loss, val_loss, model)``;
            a callback returning True requests an early stop. Used for live
            validation-curve capture and instrumentation. Not supported with
            ``n_ensembles > 1`` (members fit in parallel worker processes).
        """
        cat_features = _resolve_cat_features(self, cat_features)
        cat_features = _resolve_cat_feature_names(cat_features, X)
        _validate_hyperparams(self)
        y = _validate_fit_input(self, X, y, cat_features, sample_weight,
                                classification=True)
        if eval_set is not None:
            _check_eval_set(eval_set, self.n_features_in_)
        if self.n_ensembles and self.n_ensembles > 1:
            if callbacks is not None:
                raise ValueError(
                    "callbacks are not supported with n_ensembles > 1.")
            # Fix the global class set up front: a member's bootstrap may miss a
            # rare class, and predict_proba aligns each member's columns to this.
            yarr = np.asarray(y)
            self.classes_ = np.unique(yarr)
            self.n_classes_ = self.classes_.size
            if self.n_classes_ < 2:
                raise ValueError(
                    f"Need at least 2 classes; got {self.n_classes_} class(es).")
            self._multiclass = self.n_classes_ > 2
            self.estimators_ = _fit_bagged(self, X, yarr, cat_features, eval_set,
                                           groups, sample_weight)
            return self
        self.estimators_ = None
        return self._fit_single(X, y, cat_features, eval_set, groups,
                                sample_weight, callbacks)

    def __sklearn_is_fitted__(self):
        return (hasattr(self, "model_")
                or getattr(self, "estimators_", None) is not None)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN routed to a missing bin
        tags.input_tags.sparse = False
        return tags

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight,
                    callbacks=None):
        """Fit one (non-bagged) classifier on the data as given."""
        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ < 2:
            raise ValueError(
                f"Need at least 2 classes; got {self.n_classes_} class(es).")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        es_active = bool(self.early_stopping)
        if es_active and eval_set is None:
            split = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=y,  # always stratify for classification
            )
            if split is None:
                es_active = False  # data too small to hold out a val set
            else:
                train_idx, val_idx = split
                eval_set = (X[val_idx], y[val_idx])
                X, y = X[train_idx], y[train_idx]
                if sample_weight is not None:
                    sample_weight = sample_weight[train_idx]
                self.classes_ = np.unique(y)
                self.n_classes_ = self.classes_.size

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50   # see GradientBoosting/Regressor note above

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        # Size-adaptive min_child_weight (see _auto_min_child_weight): resolved
        # on the FINAL training set (post early-stopping split).
        if kw.get("min_child_weight") is None:
            kw["min_child_weight"] = _auto_min_child_weight(len(X))
        # colsample None = auto: full columns for a single model (the bagged
        # path resolves members to 0.85 before this runs; see _fit_bagged).
        if kw.get("colsample") is None:
            kw["colsample"] = 1.0
        # Auto-resolve cat_combinations: on only for tractable all-categorical data
        # (targets the all-categorical multiclass gap, e.g. car).
        if kw.get("cat_combinations") is None:
            kw["cat_combinations"] = _auto_cat_combinations(
                cat_features, self.n_features_in_, len(X))

        self._multiclass = self.n_classes_ > 2
        # Resolve the linear_leaves auto-default: ON for binary (a clean broad
        # Brier win that survives bagging), OFF for multiclass (unsupported).
        # An explicit True on multiclass is a user error -> raise; explicit
        # False is honored everywhere.
        if self.linear_leaves is None:
            kw["linear_leaves"] = not self._multiclass
        elif self.linear_leaves and self._multiclass:
            raise NotImplementedError(
                "linear_leaves is not supported for multiclass classification "
                "yet; use it on regression or binary classification.")
        # cross_features: None (auto default) silently skips multiclass, like
        # linear_leaves; only an explicit True is a user error there.
        if self.cross_features is True and self._multiclass:
            raise NotImplementedError(
                "cross_features is not supported for multiclass classification "
                "yet; use it on regression or binary classification.")
        cal_Xv = cal_y = None   # validation set used to calibrate temperature
        if self._multiclass:
            self.model_ = MulticlassBoosting(**kw)
            self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight, callbacks=callbacks)
            self.classes_ = self.model_.classes_
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_y = np.searchsorted(self.classes_, np.asarray(eval_set[1]))
        else:
            y01 = (y == self.classes_[1]).astype(np.float64)
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_y = (np.asarray(eval_set[1]) == self.classes_[1]).astype(np.float64)
                eval_set = (cal_Xv, cal_y)
            # Cheap selection (benchmarks/PARETO_PLAN.md step 2): when
            # selection_rounds is set and the cross refit will run (pair
            # candidates exist iff >= 2 numeric columns), the base fit is only
            # an audition -- cap it; it is refit in full below only if the
            # augmented model loses the selection.
            n_cats = len(cat_features) if cat_features else 0
            fast = (self.selection_rounds is not None
                    and self.cross_features is not False
                    and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES
                    and X.shape[1] - n_cats >= 2)
            stop = _stop_after(self.selection_rounds) if fast else None
            self.model_ = GradientBoosting(loss="Logloss", **kw)
            self.model_.fit(X, y01, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight,
                            callbacks=_add_callback(callbacks, stop))

        # Numeric cross features (default-on auto, binary only): refit with
        # difference/product columns for the top numeric feature pairs of the
        # base fit and keep the lower-validation-loss model (the regressor's
        # selection pattern; see _cross_candidate_pairs). None (auto) and
        # True behave the same on binary.
        self.cross_features_selected_ = None
        self.cross_pairs_ = None
        if (self.cross_features is not False and not self._multiclass
                and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES):
            pairs = _cross_candidate_pairs(
                self.model_.feature_importances_, cat_features, X.shape[1])
            if pairs:
                aug = GradientBoosting(loss="Logloss", cross_pairs=pairs, **kw)
                if fast:
                    # Symmetric race at the shared budget (see the regressor):
                    # judge both candidates on their first selection_rounds;
                    # kill a trailing augmented fit at the budget.
                    base_best = _best_val(self.model_)
                    aug.fit(X, y01, cat_features=cat_features,
                            eval_set=eval_set, sample_weight=sample_weight,
                            callbacks=_add_callback(
                                callbacks, _stop_if_behind(
                                    self.selection_rounds, base_best)))
                    self.cross_features_selected_ = (
                        min(aug.valid_history_[:self.selection_rounds])
                        < base_best) if aug.valid_history_ else False
                else:
                    aug.fit(X, y01, cat_features=cat_features,
                            eval_set=eval_set, sample_weight=sample_weight,
                            callbacks=callbacks)
                    self.cross_features_selected_ = \
                        _best_val(aug) < _best_val(self.model_)
                if self.cross_features_selected_:
                    self.model_ = aug
                    self.cross_pairs_ = pairs
                elif fast and len(self.model_.valid_history_) >= self.selection_rounds:
                    # The incumbent audition was actually truncated by the cap
                    # (an audition that early-stopped on its own already IS the
                    # full fit); give the winning base variant its full fit.
                    self.model_ = GradientBoosting(loss="Logloss", **kw)
                    self.model_.fit(X, y01, cat_features=cat_features,
                                    eval_set=eval_set,
                                    sample_weight=sample_weight,
                                    callbacks=callbacks)

        # Temperature scaling on the validation set: dividing raw scores by T > 0
        # is monotonic, so predict() is unchanged while predict_proba() becomes
        # better calibrated (lower log loss).
        self.temperature_ = 1.0
        if cal_Xv is not None:
            raw = self.model_.predict_raw(cal_Xv)
            self.temperature_ = _fit_temperature(raw, cal_y, self._multiclass)
        return self

    def predict_proba(self, X):
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            # Soft-vote: average members' calibrated probabilities, aligning each
            # member's class columns to the global class set (a member whose
            # bootstrap missed a class simply contributes 0 to that column).
            probas = [m.predict_proba(X) for m in self.estimators_]
            acc = np.zeros((probas[0].shape[0], self.n_classes_))
            for m, p in zip(self.estimators_, probas):
                cols = np.searchsorted(self.classes_, m.classes_)
                acc[:, cols] += p
            return acc / len(self.estimators_)
        raw = self.model_.predict_raw(X) / self.temperature_
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

    @property
    def validation_history_(self):
        """Per-round validation loss recorded during ``fit`` (binary or softmax
        log loss), as a list whose length is the number of rounds run. Empty when
        no ``eval_set`` / early-stopping split was available; for a bagged model
        (``n_ensembles > 1``) a list of the members' histories."""
        if self.estimators_ is not None:
            return [m.model_.valid_history_ for m in self.estimators_]
        return self.model_.valid_history_

    @property
    def feature_importances_(self):
        if self.estimators_ is not None:
            return np.mean([m.feature_importances_ for m in self.estimators_],
                           axis=0)
        return self.model_.feature_importances_

    def shap_values(self, X, X_background=None):
        """Exact interventional TreeSHAP contributions in LOG-ODDS (margin) space.

        Binary only. Returns an array of shape ``(n_samples, n_features)`` whose
        rows sum to ``raw_log_odds(X) - expected_value_`` (pre-temperature), with
        ``expected_value_`` set as an attribute. Each entry is a feature's signed
        contribution to the log-odds of the positive class; linear-leaf slopes are
        included exactly. Averaged across the bag when ``n_ensembles > 1`` (an
        additive surrogate for the soft-voted probability). Multiclass is not
        supported yet. ``X_background`` overrides the reference distribution."""
        _check_predict_input(self, X)
        members = self.estimators_ if self.estimators_ is not None else None
        if (members is not None and getattr(members[0], "_multiclass", False)) \
                or (members is None and self._multiclass):
            raise NotImplementedError(
                "shap_values is not supported for multiclass classification yet.")
        if members is not None:
            out = [m.model_.shap_values(X, background=X_background)
                   for m in members]
            self.expected_value_ = float(np.mean([b for _, b in out]))
            return np.mean([p for p, _ in out], axis=0)
        phi, base = self.model_.shap_values(X, background=X_background)
        self.expected_value_ = base
        return phi