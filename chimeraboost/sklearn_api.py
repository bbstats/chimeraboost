"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import warnings

import numpy as np
from .booster import GradientBoosting, MulticlassBoosting
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _fit_temperature(raw, y, multiclass):
    """Learn the scalar T > 0 minimizing validation log loss of sigmoid(raw/T)
    (binary) or softmax(raw/T) (multiclass). Dividing logits by T is monotonic,
    so predictions are unchanged — only their probabilities are recalibrated.
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
                           "n_ensembles", "ensemble_n_jobs", "cat_features"})


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
    _in_range("colsample", 0.0, 1.0, lo_incl=False)
    # cat_smoothing is a Bayesian pseudocount in the ordered-TS denominator
    # (count + a); a=0 makes the first occurrence of every category divide 0/0.
    _in_range("cat_smoothing", 0.0, np.inf, lo_incl=False)
    _in_range("hs_lambda", 0.0, np.inf)
    _in_range("linear_lambda", 0.0, np.inf)
    _in_range("min_child_weight", 0.0, np.inf, allow_none=True)
    _in_range("validation_fraction", 0.0, 1.0, lo_incl=False, hi_incl=False)
    _in_range("early_stopping_rounds", 1, np.inf, allow_none=True)
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
    same estimator class, all per-model machinery — binary/multiclass dispatch,
    ``cat_features``, the early-stopping auto-split, temperature scaling — is
    reused unchanged, and ``cat_features``/``sample_weight``/``groups`` forward
    naturally (which a ``sklearn.ensemble.Bagging`` wrapper would not do).

    Members are independent, so they fit across ``ensemble_n_jobs`` processes.
    When that is >1 and ``thread_count`` is unset, numba threads are divided
    among the workers so the members don't oversubscribe the cores.
    """
    from sklearn.base import clone
    from joblib import Parallel, delayed

    X = (np.asarray(X, dtype=object) if cat_features
         else np.asarray(X, dtype=np.float64))
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    n = X.shape[0]
    K = int(estimator.n_ensembles)
    n_jobs = int(estimator.ensemble_n_jobs)

    member_threads = estimator.thread_count
    if n_jobs != 1 and member_threads is None:
        import numba
        member_threads = max(1, numba.config.NUMBA_NUM_THREADS // abs(n_jobs))

    seeds = np.random.default_rng(estimator.random_state).integers(
        0, 2**31 - 1, size=K)

    def _fit_one(seed):
        member = clone(estimator).set_params(
            n_ensembles=None, random_state=int(seed), thread_count=member_threads)
        idx = np.random.default_rng(seed).integers(0, n, size=n)  # bootstrap
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

    return Parallel(n_jobs=n_jobs)(delayed(_fit_one)(s) for s in seeds)


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
        Xc = np.asarray(X, dtype=object if cat_features else np.float64)
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
            Xc = np.asarray(Xraw, dtype=np.float64)
        except (ValueError, TypeError) as e:
            # A non-numeric column (string/category/datetime/pandas-NA) in a
            # DataFrame with no cat_features: name the offending columns and
            # point at cat_features. For bare arrays (no column metadata) keep
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


def _was_fit_with_cats(estimator):
    """True if the fitted model used categorical features (so X is the object
    path and a numeric finiteness check does not apply)."""
    m = getattr(estimator, "model_", None)
    if m is None:
        members = getattr(estimator, "estimators_", None)
        m = members[0].model_ if members else None
    return bool(getattr(getattr(m, "prep_", None), "cat_features_", None))


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
    if not _was_fit_with_cats(estimator) and not _assume_finite():
        try:
            Xf = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError):
            Xf = None
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
    colsample : float, default 1.0
        Fraction of features eligible for each tree.
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
    cat_combinations : bool, default False
        Add all pairwise categorical-by-categorical features.
    leaf_estimation_iterations : int, default 1
        Newton refinement steps per leaf.
    hs_lambda : float, default 0.0
        Hierarchical-shrinkage strength. Above 0, leaf values are recursively
        shrunk toward their ancestors, hardest for deep or low-mass leaves. Adds
        no inference cost.
    linear_leaves : bool, default False
        Fit a ridge linear model per leaf over the numeric split features instead
        of a constant value, adding local slope where step leaves underfit. Leaves
        with too few rows fall back to a constant. Not available with MAE or
        quantile loss.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    early_stopping : bool, default True
        Hold out a validation split and stop when its score stops improving.
    validation_fraction : float, default 0.2
        Validation fraction used when ``early_stopping`` is on and no ``eval_set``
        is passed to ``fit``.
    n_ensembles : int or None, default None
        Number of bagged members. ``None`` or 1 trains a single model; >= 2
        averages independent members fit on bootstrap resamples.
    ensemble_n_jobs : int, default 1
        Processes used to fit ensemble members; -1 uses all cores.
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
    """

    def __init__(self, n_estimators=2000, learning_rate=None, depth=None,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 loss="RMSE", alpha=0.5, min_child_weight=1.0, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting=False,
                 cat_combinations=False, leaf_estimation_iterations=1,
                 hs_lambda=0.0, linear_leaves=False, linear_lambda=1.0,
                 early_stopping=True, validation_fraction=0.2,
                 n_ensembles=None, ensemble_n_jobs=1, cat_features=None):
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
        self.hs_lambda = hs_lambda
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None):
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
        """
        cat_features = _resolve_cat_features(self, cat_features)
        cat_features = _resolve_cat_feature_names(cat_features, X)
        _validate_hyperparams(self)
        y = _validate_fit_input(self, X, y, cat_features, sample_weight,
                                classification=False)
        if eval_set is not None:
            _check_eval_set(eval_set, self.n_features_in_)
        if self.n_ensembles and self.n_ensembles > 1:
            self.estimators_ = _fit_bagged(self, X, y, cat_features, eval_set,
                                           groups, sample_weight)
            return self
        self.estimators_ = None
        return self._fit_single(X, y, cat_features, eval_set, groups,
                                sample_weight)

    def __sklearn_is_fitted__(self):
        return (hasattr(self, "model_")
                or getattr(self, "estimators_", None) is not None)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN routed to a missing bin
        tags.input_tags.sparse = False
        return tags

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight):
        """Fit one (non-bagged) model on the data as given."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
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
        self.model_ = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                       **kw)
        self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                        sample_weight=sample_weight)
        return self

    def predict(self, X):
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            return np.mean([m.predict(X) for m in self.estimators_], axis=0)
        return self.model_.predict_raw(X)

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        _check_predict_input(self, X)
        if self.estimators_ is not None:
            raise NotImplementedError("staged_predict is not defined for a "
                                      "bagged ensemble (n_ensembles > 1).")
        yield from self.model_.staged_predict_raw(X)

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

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
            self.expected_value_ = float(np.mean([b for _, b in out]))
            return np.mean([p for p, _ in out], axis=0)
        phi, base = self.model_.shap_values(X, background=X_background)
        self.expected_value_ = base
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
    colsample : float, default 1.0
        Fraction of features eligible for each tree.
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
    cat_combinations : bool, default False
        Add all pairwise categorical-by-categorical features.
    leaf_estimation_iterations : int, default 3
        Newton refinement steps per leaf.
    hs_lambda : float, default 0.0
        Hierarchical-shrinkage strength. Above 0, leaf values are recursively
        shrunk toward their ancestors, hardest for deep or low-mass leaves.
    linear_leaves : bool or None, default None
        Fit a ridge linear model per leaf over the numeric split features instead
        of a constant. ``None`` enables it for binary classification and disables
        it for multiclass (where it is unsupported). Below ~1000 rows it falls
        back to constant leaves.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    early_stopping : bool, default True
        Hold out a stratified validation split and stop when it stops improving.
        ``StratifiedGroupKFold`` is used when ``groups`` is passed to ``fit``.
    validation_fraction : float, default 0.2
        Validation fraction used when ``early_stopping`` is on and no ``eval_set``
        is passed to ``fit``.
    n_ensembles : int or None, default None
        Number of bagged members. ``None`` or 1 trains a single model; >= 2
        soft-votes the calibrated probabilities of members fit on bootstraps.
    ensemble_n_jobs : int, default 1
        Processes used to fit ensemble members; -1 uses all cores.
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
    """

    def __init__(self, n_estimators=2000, learning_rate=None, depth=6,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 min_child_weight=None, thread_count=None, random_state=None,
                 verbose=False, ordered_boosting=False,
                 cat_combinations=False, leaf_estimation_iterations=3,
                 hs_lambda=0.0, linear_leaves=None, linear_lambda=1.0,
                 early_stopping=True, validation_fraction=0.2,
                 n_ensembles=None, ensemble_n_jobs=1, cat_features=None):
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
        self.hs_lambda = hs_lambda
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None):
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
        """
        cat_features = _resolve_cat_features(self, cat_features)
        cat_features = _resolve_cat_feature_names(cat_features, X)
        _validate_hyperparams(self)
        y = _validate_fit_input(self, X, y, cat_features, sample_weight,
                                classification=True)
        if eval_set is not None:
            _check_eval_set(eval_set, self.n_features_in_)
        if self.n_ensembles and self.n_ensembles > 1:
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
                                sample_weight)

    def __sklearn_is_fitted__(self):
        return (hasattr(self, "model_")
                or getattr(self, "estimators_", None) is not None)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN routed to a missing bin
        tags.input_tags.sparse = False
        return tags

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight):
        """Fit one (non-bagged) classifier on the data as given."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
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
        cal_Xv = cal_y = None   # validation set used to calibrate temperature
        if self._multiclass:
            self.model_ = MulticlassBoosting(**kw)
            self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight)
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
            self.model_ = GradientBoosting(loss="Logloss", **kw)
            self.model_.fit(X, y01, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight)

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