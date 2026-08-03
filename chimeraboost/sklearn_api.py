"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import contextlib
import inspect
import warnings

import numpy as np
from .booster import (GradientBoosting, LINEAR_LEAVES_MIN_SAMPLES,
                      MulticlassBoosting, _thread_limit)
from .preprocessing import CatTransformCache, as_model_array
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _fit_temperature(raw, y, multiclass, sample_weight=None):
    """Learn the scalar T > 0 that minimizes validation log loss.

    The calibrated score is sigmoid(raw/T) for binary, softmax(raw/T) for
    multiclass. Dividing by T is monotonic, so predictions are unchanged -- only
    the probabilities are recalibrated. ``y`` is the 0/1 label (binary) or the
    class index (multiclass). ``sample_weight`` weights the validation log loss,
    so a zero-weight holdout row cannot steer the temperature (the sample_weight
    contract).
    """
    from scipy.optimize import minimize_scalar

    raw = np.asarray(raw, dtype=np.float64)
    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        if not np.any(w > 0):
            return 1.0

    if multiclass:
        rows = np.arange(raw.shape[0])

        def loss(T):
            logits = raw / T
            mx = logits.max(axis=1, keepdims=True)
            log_z = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            return float(np.average(log_z - logits[rows, y], weights=w))
    else:
        def loss(T):
            z = raw / T
            # Stable binary cross-entropy: softplus(z) - y*z.
            return float(np.average(np.log1p(np.exp(-np.abs(z)))
                                    + np.maximum(z, 0.0) - y * z, weights=w))

    res = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x) if res.success else 1.0


# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({"early_stopping", "validation_fraction",
                           "n_ensembles", "ensemble_n_jobs", "max_samples",
                           "cat_features", "cross_features",
                           "selection_rounds", "refit_full", "refit_members",
                           "quality"})

# --- quality: named operating points on the strength/slowdown Pareto --------
# Evidence: benchmarks/SELECT_PLAN.md. Every recipe only pins parameters that
# already exist -- nothing new is computed, and quality=None (the default)
# leaves every code path byte-identical to before this parameter existed.
QUALITY_NAMES = {1: "fast", 2: "balanced", 3: "accurate",
                 4: "ensemble", 5: "max"}


def _quality_overrides(estimator, level):
    """The parameters ``quality=level`` pins on this estimator.

    Rung 1 buys out the model-selection search: one booster fit instead of the
    default's two-to-four, and no full-data refit on top. Only the regressor pins
    linear_leaves, where None means "audition const vs linear" -- exactly the
    search this rung declines to pay for. On the classifier None is already a
    free auto rule (on for binary, off for multiclass, where an explicit True
    raises), so it is left alone.

    Rungs 4 and 5 sit on top of the plain defaults, NOT on top of rung 3:
    refit_full is a deliberate no-op inside bag members (their out-of-bag rows
    are already an eval set), so the rungs do not stack. See REFIT_PLAN.md.
    """
    if level == 1:
        ov = {"cross_features": False, "refit_full": False}
        if estimator._QUALITY_PINS_LINEAR_LEAVES:
            ov["linear_leaves"] = True
        return ov

    if level == 2:
        # The search, without the full-data refit the default now performs.
        return {"refit_full": False}

    if level == 3:
        # The default. "replay" rather than True: same accuracy for about two
        # thirds of the fit (benchmarks/REPLAY_PLAN.md). Pass refit_full=True
        # for the from-scratch refit.
        return {"refit_full": "replay"}

    if level == 4:
        return {"n_ensembles": 5}

    if level == 5:
        return {"n_ensembles": 8}

    return {}


def _ctor_default(estimator, name):
    return inspect.signature(type(estimator).__init__).parameters[name].default


@contextlib.contextmanager
def _quality_applied(estimator):
    """Apply the ``quality`` recipe for the duration of a fit.

    Constructor parameters are restored on the way out so ``get_params``
    keeps returning what the user passed (sklearn requires ``fit`` not to
    rewrite them). The recipe wins over an explicitly-set parameter it
    controls, with a warning naming the override.
    """
    level = getattr(estimator, "quality", None)
    if level is None:
        yield
        return

    overrides = _quality_overrides(estimator, int(level))
    clashes = {k: getattr(estimator, k) for k, v in overrides.items()
               if getattr(estimator, k) != v
               and getattr(estimator, k) != _ctor_default(estimator, k)}
    if clashes:
        warnings.warn(
            f"quality={int(level)} ({QUALITY_NAMES[int(level)]}) sets "
            + ", ".join(f"{k}={overrides[k]}" for k in clashes)
            + "; the explicit "
            + ", ".join(f"{k}={v}" for k, v in clashes.items())
            + " you passed is ignored. Drop quality to set these yourself.",
            UserWarning, stacklevel=3)

    saved = {k: getattr(estimator, k) for k in overrides}
    for k, v in overrides.items():
        setattr(estimator, k, v)

    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(estimator, k, v)


def _validate_hyperparams(estimator):
    """Reject malformed constructor parameters with clear, named errors.

    Called at the start of ``fit`` -- sklearn's recommended place for parameter
    validation, never ``__init__``.

    Without it, bad values fail cryptically deep in numba (``depth=-1`` ->
    "negative shift count"), silently produce a broken model
    (``learning_rate=-0.1`` diverges to garbage, ``n_estimators=0`` builds an
    empty model), or OOM (``depth=30`` allocates a 2**30-leaf histogram).
    ``None`` is left to the documented per-parameter default resolution.
    """
    p = estimator.get_params()

    # Estimators expose different parameter sets (the quantile head has no
    # loss/alpha family, no bagging, no linear leaves), so every check below
    # is a no-op for a parameter this estimator does not have.
    def _pos_int(name, lo=1, allow_none=False):
        if name not in p:
            return
        v = p[name]
        if v is None and allow_none:
            return
        if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                and v >= lo):
            raise ValueError(f"{name} must be an integer >= {lo}; got {v!r}.")

    def _in_range(name, lo, hi, *, lo_incl=True, hi_incl=True, allow_none=False):
        if name not in p:
            return
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

    # None = the classifier's auto default (resolved to 3 at fit); the regressor
    # passes a concrete int.
    _pos_int("leaf_estimation_iterations", allow_none=True)

    # A depth-d tree allocates 2**d leaves in the histogram buffer, so an
    # unbounded depth OOMs. 16 matches CatBoost's documented maximum. None is the
    # regressor's loss-adaptive default, resolved at fit.
    v = p.get("depth")
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
    _in_range("max_samples", 0.0, 1.0, lo_incl=False)

    v = p.get("refit_full")
    if v is not None and v != "replay" and not isinstance(v, (bool, np.bool_)):
        raise ValueError(
            f'refit_full must be True, False, "replay" or None; got {v!r}.')

    v = p.get("quality")
    if v is not None:
        if isinstance(v, (bool, np.bool_)) or v not in QUALITY_NAMES:
            raise ValueError(
                "quality must be None or one of "
                + ", ".join(f"{k} ({n})" for k, n in QUALITY_NAMES.items())
                + f"; got {v!r}.")

    # Regressor-only loss / alpha (the classifier picks its loss automatically).
    if "loss" in p:
        loss = p["loss"]
        if isinstance(loss, str):
            known = ("RMSE", "MAE", "Quantile", "Huber", "Poisson", "Gamma",
                     "Tweedie")
            if loss not in known:
                raise ValueError(
                    f"loss must be one of {known} or a custom objective "
                    f"instance; got {loss!r}.")

            if loss == "Quantile":
                _in_range("alpha", 0.0, 1.0, lo_incl=False, hi_incl=False)
            if loss == "Huber":
                _in_range("delta", 0.0, np.inf, lo_incl=False)
            if loss == "Tweedie":
                _in_range("tweedie_variance_power", 1.0, 2.0,
                          lo_incl=False, hi_incl=False)
        else:
            # Custom objective: an instance implementing the losses.py protocol
            # (subclass chimeraboost.CustomObjective for the optional-method
            # defaults).
            missing = [a for a in ("init", "grad_hess", "eval")
                       if not callable(getattr(loss, a, None))]
            if missing:
                raise ValueError(
                    "A custom loss must be an instance with callable "
                    f"init/grad_hess/eval (missing: {missing}); subclass "
                    "chimeraboost.CustomObjective. Got "
                    f"{loss!r}.")

    if p.get("eval_metric") is not None and not callable(p["eval_metric"]):
        raise ValueError(
            "eval_metric must be a callable metric(y_true, y_pred"
            "[, sample_weight]) -> float, or None; got "
            f"{p['eval_metric']!r}.")


def _resolve_cat_features(estimator, cat_features):
    """The effective cat_features: the ``fit`` argument if given, else the
    constructor argument.

    The fit argument wins so a one-off call can override. The constructor form
    exists so sklearn meta-estimators (GridSearchCV/Pipeline) can carry it, which
    a fit-only kwarg cannot. Never mutates ``estimator.cat_features`` -- sklearn
    forbids ``fit`` changing init params.
    """
    if cat_features is not None:
        return cat_features
    return getattr(estimator, "cat_features", None)


def _resolve_cat_feature_names(cat_features, X):
    """Map column *names* in ``cat_features`` to integer positions using X's
    column metadata, leaving integer indices untouched.

    Categoricals can be marked the way LightGBM/CatBoost allow: by position
    (``[0, 2]``), by name (``["city", "brand"]``), or a mix. Names resolve
    against the DataFrame columns at fit time, so a later column reordering is
    caught by the predict-time feature-name check. ``None`` passes through
    unchanged; a name-free sequence comes back as a plain list for downstream
    integer validation.
    """
    if cat_features is None:
        return None

    try:
        items = list(cat_features)
    except TypeError:
        return cat_features  # not iterable; let downstream validation report it

    if not any(isinstance(c, str) for c in items):
        # A plain list, not the original object: a numpy array here crashes
        # every downstream ``if cat_features:`` check with "ambiguous truth
        # value" before validation can name the problem.
        return items

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


def _check_eval_set(eval_set, n_features, classification=False):
    """Validate a user-passed ``eval_set`` up front with a named error instead of
    a cryptic IndexError or broadcast failure deep in the booster."""
    # A 2-tuple is the documented form. The optional 3rd element is per-row
    # validation weights, used internally by the weight-aware auto-split and by
    # bagged OOB early stopping so zero-weight rows don't score the metric.
    if not (isinstance(eval_set, (tuple, list)) and len(eval_set) in (2, 3)):
        raise ValueError(
            "eval_set must be a (X_val, y_val) tuple "
            "(optionally (X_val, y_val, sample_weight_val)).")

    Xv, yv = eval_set[0], eval_set[1]
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

    if len(eval_set) == 3 and eval_set[2] is not None \
            and len(eval_set[2]) != shape[0]:
        raise ValueError(
            f"eval_set sample_weight has length {len(eval_set[2])}, but "
            f"eval_set X has {shape[0]} rows; they must match.")

    if not classification:
        # Past bug: training y was rejected on NaN/inf but eval y was not, so a
        # non-finite eval y turned every validation score into NaN, early
        # stopping kept round 0, and the model shipped with one tree. Classifier
        # labels go to _check_eval_labels instead -- strings don't cast to float.
        yv_arr = np.asarray(yv)
        if yv_arr.dtype.kind in "fc" and not np.isfinite(
                yv_arr.astype(np.float64, copy=False)).all():
            raise ValueError(
                "eval_set y contains NaN or infinity; validation targets "
                "must be finite.")


def _check_eval_labels(eval_set, y):
    """Reject classifier eval_set labels that never appear in the training y.

    Without this the mapping is silently wrong: binary treats any non-positive
    label as the negative class, and multiclass ``searchsorted`` counts an
    unseen label as the next class up (or IndexErrors past the top) -- so the
    early-stopping signal is computed against labels the user never provided.
    """
    classes = np.unique(np.asarray(y))
    yv = np.unique(np.asarray(eval_set[1]))

    try:
        unseen = np.setdiff1d(yv, classes)
    except TypeError:   # mixed un-orderable label types
        seen = set(classes.tolist())
        unseen = np.array([v for v in yv.tolist() if v not in seen],
                          dtype=object)

    if unseen.size:
        raise ValueError(
            f"eval_set contains label(s) not present in y: {unseen.tolist()}. "
            "Early stopping must be evaluated on the training label set.")


def _is_numeric_dtype(dt):
    """True if a column dtype is float-castable, across numpy / pandas / polars.

    Bool counts: it casts to 0/1 cleanly, so a bool column must not be named in
    the "add these to cat_features" error guidance.
    """
    try:
        npdt = np.dtype(dt)
        return bool(np.issubdtype(npdt, np.number) or npdt == np.bool_)
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
            and "object" not in s) or s in ("bool", "boolean")


def _describe_nonnumeric_columns(X):
    """Name the non-numeric columns of a DataFrame-like X (pandas/polars).

    A user who forgot ``cat_features`` then gets "column 'city' (index 2)"
    instead of a bare ``could not convert string to float: 'NYC'``. Returns []
    for input without column metadata (a plain ndarray).
    """
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


def _member_oob_eval_indices(idx, n, groups):
    """Out-of-bag row indices usable as a bag member's early-stopping eval set.

    Without ``groups`` that is every row outside the member's sample. With
    ``groups`` it is further restricted to rows whose group never appears in the
    sample -- otherwise the member's stopping signal comes from rows correlated
    with its training rows, exactly the leakage ``groups`` promises to prevent.

    Grouped members draw whole groups (``_member_sample_indices``), so the rows
    excluded for straddling are normally just the rare-class donor's group.
    Before group draws existed, a typical 80% row draw touched essentially every
    group, this returned zero rows on real data, and OOB early stopping was
    silently disabled.

    May still return an empty array (say a cluster bootstrap that drew every
    group); the caller then falls back to the member's own auto-split, which is
    group-aware.
    """
    oob_mask = np.ones(n, dtype=np.bool_)
    oob_mask[idx] = False
    if groups is not None:
        oob_mask &= ~np.isin(groups, np.unique(groups[idx]))
    return np.where(oob_mask)[0]


def _member_sample_indices(n, ms, seed, groups):
    """Row indices for one bag member's training sample.

    Without ``groups``: ``ms`` of the ``n`` rows drawn without replacement
    ("subagging"), or a classic full-size with-replacement bootstrap at
    ``ms >= 1.0``. Byte-identical to the draws made before this helper existed
    (same rng construction, same single call).

    With ``groups``: the same recipe over whole groups -- ``ms`` of the groups
    without replacement (each contributing all of its rows), or a full-size
    cluster bootstrap of groups at ``ms >= 1.0``. Drawing rows instead would
    leave essentially every group straddling the sample, which leaks group
    structure into the member AND empties the group-disjoint OOB set
    ``_member_oob_eval_indices`` builds. The draw is capped at ``n_groups - 1``,
    so at least one group is always held out and the OOB eval set is non-empty
    by construction. ``ms`` counts groups, not rows, so a member's row count
    varies with the sizes of the groups it drew. A single all-rows group falls
    back to the row draw: there is nothing to hold out.
    """
    rng = np.random.default_rng(seed)

    if groups is not None:
        ug = np.unique(groups)
        if ug.size >= 2:
            if ms >= 1.0:
                # Cluster bootstrap: n_groups draws with replacement. A group
                # drawn twice appears twice -- duplicates are integer weights,
                # exactly like the row bootstrap.
                drawn = rng.integers(0, ug.size, size=ug.size)
                order = np.argsort(groups, kind="stable")
                sg = groups[order]
                starts = np.searchsorted(sg, ug, side="left")
                ends = np.searchsorted(sg, ug, side="right")
                return np.concatenate(
                    [order[starts[g]:ends[g]] for g in drawn])

            m = max(1, min(ug.size - 1, int(round(ms * ug.size))))
            drawn = rng.choice(ug.size, size=m, replace=False)
            return np.where(np.isin(groups, ug[drawn]))[0]

    if ms >= 1.0:
        return rng.integers(0, n, size=n)
    m = max(1, int(round(ms * n)))
    return rng.choice(n, size=m, replace=False)


def _fit_bagged(estimator, X, y, cat_features, eval_set, groups, sample_weight):
    """Train ``estimator.n_ensembles`` member clones and return them as a list.

    Each member is a clone of ``estimator`` with bagging switched off
    (``n_ensembles=None``) and its own seed, fit on its own random row sample:
    ``max_samples`` (default 0.8) of the rows drawn WITHOUT replacement
    ("subagging"; ``max_samples=1.0`` restores the classic full-size
    with-replacement bootstrap). With ``groups`` the same recipe draws whole
    groups instead of rows, so the held-out groups form each member's
    group-disjoint OOB eval set (see ``_member_sample_indices``).

    A member is the same estimator class, so every per-model machine --
    binary/multiclass dispatch, ``cat_features``, the early-stopping auto-split,
    temperature scaling -- is reused unchanged, and
    ``cat_features``/``sample_weight``/``groups`` forward naturally. A
    ``sklearn.ensemble.Bagging`` wrapper would not do that.

    Members are independent, so they fit across ``ensemble_n_jobs`` worker
    processes (default -1: as many workers as the thread budget supports, capped
    at K). The thread budget -- ``thread_count`` if set, else numba's thread
    count -- is divided across the workers, so a bagged fit uses the same cores a
    single fit would. numba's sublinear thread scaling is what makes K members at
    budget/K threads faster than K sequential full-budget fits: 1.2-2.0x
    wall-clock on the BAGGING_PLAN.md B4 panel, identical models by construction.
    ``ensemble_n_jobs=1`` restores sequential fits.
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
    # tolerates coarser, cheaper members, so params left on auto resolve to the
    # tuned member values rather than the single-model ones. PMLB-tuned,
    # holdout-confirmed, decision-suite validated: 54W-17L, +0.28% pooled vs the
    # previous bagged defaults at par fit cost.
    #
    # Explicit user values always win, and the substitution is announced once
    # per fit -- an opt-in bagged fit should never silently train members on
    # different defaults.
    member_defaults = {}
    if estimator.learning_rate is None:
        member_defaults["learning_rate"] = 0.15
    if estimator.colsample is None:
        member_defaults["colsample"] = 0.85
    estimator.member_params_ = dict(member_defaults)
    if member_defaults:
        warnings.warn(
            "ChimeraBoost bagged mode: member defaults "
            + ", ".join(f"{k}={v}" for k, v in member_defaults.items())
            + " (pass explicit values to override; see docs).",
            UserWarning, stacklevel=3)

    def _fit_one(seed):
        # quality=None on members: the recipe is already resolved on the bag
        # itself, and rungs 4/5 set n_ensembles -- leaving quality on a member
        # would re-apply it and recurse into another bag.
        member = clone(estimator).set_params(
            n_ensembles=None, quality=None, random_state=int(seed),
            thread_count=member_threads, **member_defaults)

        # Members receive internally-constructed inputs (ndarray rows, OOB eval
        # sets), so the strict user-facing checks that assume "the caller chose
        # this" are relaxed for them -- see _fit_single.
        member._is_bag_member = True

        # Member sample (benchmarks/BAGGING_PLAN.md B-samp). Why subagging beats
        # the classic bootstrap: a full-size bootstrap leaves a member only
        # ~0.632n unique rows at n rows of compute (duplicates are just integer
        # weights), while 0.8n unique rows cost 0.8n compute -- more effective
        # data and less work. Measured stronger and faster on both decision
        # suites (gr 54W-5L +0.94%, Brier 23W-0L, fit 0.87x; hc Brier 8W-0L,
        # fit 0.73x).
        ms = float(estimator.max_samples)
        idx = _member_sample_indices(n, ms, seed, groups)

        # The draw can miss a rare class entirely, and a member cannot fit on a
        # single class ("Need at least 2 classes" would crash the whole bag on
        # data whose y plainly has two). Inject one row of the most frequent
        # missing class. Draws that already hold two classes -- every
        # non-crashing fit before this guard -- are byte-identical.
        classes = getattr(estimator, "classes_", None)
        if classes is not None and np.unique(y[idx]).size < 2:
            patch_rng = np.random.default_rng([int(seed), 1])
            present = y[idx[0]]
            missing = [c for c in classes if c != present]
            counts = {c: int(np.sum(y == c)) for c in missing}
            donor_class = max(missing, key=lambda c: counts[c])
            donor = patch_rng.choice(np.where(y == donor_class)[0])
            if idx.size > 1:
                idx[patch_rng.integers(0, idx.size)] = donor
            else:
                # A one-row draw (tiny n, small max_samples) has no row to
                # spare: overwriting it just swaps which single class the
                # member sees, and it crashes again. Grow to two rows instead.
                idx = np.append(idx, donor)

        wb = None if sample_weight is None else np.asarray(sample_weight)[idx]
        gb = None if groups is None else groups[idx]

        # OOB rows are the member's early-stopping eval set when the caller gave
        # no explicit one. Auto-splitting the sample instead contaminates the
        # validation set: ~57% of auto-split val rows are duplicates of training
        # rows, so val loss reads optimistically low, early stopping fires late,
        # and each member builds ~38% more trees than it should. OOB rows are
        # guaranteed unseen by the member.
        if eval_set is None:
            oob_idx = _member_oob_eval_indices(idx, n, groups)

            # Carry the OOB rows' weights so zero-weight rows don't score the
            # member's early-stopping metric (H3). If every row was drawn
            # (possible for tiny n) let the member auto-split rather than train
            # with no eval set at all.
            sw_oob = (np.asarray(sample_weight)[oob_idx]
                      if sample_weight is not None else None)
            member_eval = ((X[oob_idx], y[oob_idx], sw_oob)
                           if len(oob_idx) > 0 else None)
        else:
            member_eval = eval_set

        # Member-level full-data refit (benchmarks/BREAKTHROUGH_PLAN.md C1):
        # once early stopping has used the OOB rows, replay this member's own
        # structure against gradients from EVERY row, so its leaf values stop
        # being estimated from max_samples*n. Only the leaf values move; the
        # splits stay exactly as this member's sample grew them, which is the
        # diversity the bag trades on. Off by default -- with refit_members
        # False nothing here fires and the member is byte-identical.
        if getattr(estimator, "refit_members", False) and eval_set is None:
            member._bag_refit_rows_ = (X, y, sample_weight, ms)

        member.fit(X[idx], y[idx], cat_features=cat_features, eval_set=member_eval,
                   groups=gb, sample_weight=wb)
        member._bag_refit_rows_ = None

        # Members train on bare ndarray rows, so without this they warn "fitted
        # without feature names" on every DataFrame predict. The parent's
        # captured names also make the column-order guard apply per member.
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            member.feature_names_in_ = names

        # member_threads (the fit-time per-worker share of the thread budget)
        # must not outlive the fit: members predict sequentially, so a member
        # left capped at budget/K threads would walk its forest on a sliver of
        # the machine. Restore the parent's setting for everything after fit.
        # Per-row tree walks don't depend on thread count, so predictions are
        # unchanged.
        member.thread_count = estimator.thread_count
        member.model_.thread_count = estimator.thread_count
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
        Group membership array (e.g. ``df['subject_id']``). When supplied,
        groups are kept intact across the split boundary. Classification uses
        ``StratifiedGroupKFold`` so class proportions are preserved; regression
        uses ``GroupShuffleSplit``.

    Returns ``None`` when the data is too small to carve a valid validation set
    -- a tiny ``n``, or a class with too few members for a stratified split. The
    caller treats ``None`` as "train on all rows, early stopping disabled"
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
                # StratifiedGroupKFold approximates the wanted val fraction with
                # n_splits = round(1 / validation_fraction). Shuffle only when a
                # random_state was given, so the seed picks the fold the way
                # every other branch here honors it. Unshuffled, the holdout is
                # always the same first fold and random_state is inert;
                # random_state=None keeps that historical behaviour.
                n_splits = max(2, round(1.0 / validation_fraction))
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=random_state is not None,
                    random_state=random_state,
                )
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


def _auto_es_split(est, X, y, sample_weight, eval_set, groups, stratify):
    """Resolve ``early_stopping=True`` into an explicit ``eval_set``, holding out
    ``validation_fraction`` of the rows.

    Shared by the regressor (``stratify=None``) and the classifier
    (``stratify=y``); the classifier's lost-class check runs at its call site.
    Returns ``(es_active, auto_split, X, y, sample_weight, eval_set)``. The
    caller keeps the pre-split arrays for the optional full-data refit.
    """
    es_active = bool(est.early_stopping)
    auto_split = False

    if es_active and eval_set is None:
        split = _make_eval_split(
            X, y, est.validation_fraction, est.random_state,
            groups=groups, stratify=stratify,
        )
        if split is None:
            es_active = False  # data too small to hold out a val set
        else:
            auto_split = True
            train_idx, val_idx = split

            if est.verbose and not getattr(est, "_is_bag_member", False):
                print(f"early_stopping=True: holding out {len(val_idx)} "
                      f"of {len(X)} rows as a validation set (pass "
                      "eval_set to choose it, or early_stopping=False "
                      "to train on all rows)")

            # Carry the val rows' weights so zero-weight rows split off into
            # the auto holdout don't score the early-stopping metric (H3).
            sw_val = (sample_weight[val_idx]
                      if sample_weight is not None else None)
            eval_set = (X[val_idx], y[val_idx], sw_val)
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]

    return es_active, auto_split, X, y, sample_weight, eval_set


def _extract_feature_names(X):
    """Return X's column names as a 1-D object array, or None.

    The trap: ``pyarrow.Table.columns`` is the column *data* (a list of arrays),
    not the names, and would pollute ``feature_names_in_`` with the data itself.
    So prefer ``.column_names`` (pyarrow) over ``.columns`` (pandas/polars), and
    reject anything that isn't a flat sequence of scalar names -- arrays, or
    pandas MultiIndex tuples.
    """
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
    """Reject masked arrays with guidance.

    ``np.asarray`` silently drops the mask and uses the hidden values, inverting
    the user's "these are missing" intent.
    """
    if np.ma.isMaskedArray(X):
        raise TypeError(
            f"Masked arrays are not supported ({where}). Convert with "
            "X.filled(np.nan) -- NaN is treated as missing.")


def _validate_fit_input(estimator, X, y, cat_features, sample_weight, *,
                        classification):
    """Shared fit-time input validation and feature-metadata capture.

    Returns the (possibly raveled) ``y`` and sets ``n_features_in_`` (and
    ``feature_names_in_`` for DataFrame input) on ``estimator``. Raises clear
    errors for the common malformed inputs rather than letting them fail
    cryptically deep in numpy/numba.

    NaN in X is deliberately allowed -- it is treated as missing and routed to
    its own bin. inf, complex, multi-output y and scipy.sparse input are not; see
    the README "scikit-learn compatibility" note.
    """
    import scipy.sparse as sp
    from sklearn.exceptions import DataConversionWarning

    if not getattr(estimator, "_is_bag_member", False):
        # Members fit in worker processes; without this a cold bagged fit
        # would print the notice once per member.
        from .warmup import _maybe_notice_cold_compile
        _maybe_notice_cold_compile()

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
                msg = "cat_features must be integer column indices or column names."
                if np.issubdtype(ci.dtype, np.floating):
                    # The classic slip: fit(X, y, w) binds the weights to
                    # cat_features (the third positional argument). Name the
                    # real mistake instead of a generic type complaint.
                    msg += (" Got an array of floats -- if these are per-sample "
                            "weights, pass them by keyword: "
                            "fit(X, y, sample_weight=w).")
                raise ValueError(msg)
            if ci.min() < 0 or ci.max() >= nf:
                raise ValueError(
                    f"cat_features index out of range for X with {nf} "
                    f"column(s): {sorted(set(ci.tolist()))}.")
            if len(set(ci.tolist())) != ci.size:
                raise ValueError("cat_features contains duplicate indices.")

    if not cat_features:
        # Check complex BEFORE the float64 cast, which would raise its own
        # TypeError on complex input instead of our clear ValueError.
        Xraw = Xc if Xc is not None else np.asarray(X)
        if np.iscomplexobj(Xraw):
            raise ValueError("Complex data not supported.")

        try:
            # Convert from the original X, not the object-dtype Xraw, so a
            # pandas DataFrame's nullable NA maps to np.nan instead of crashing
            # the float cast as an NAType object.
            Xc = as_model_array(X if Xc is None else Xc, want_object=False)
        except (ValueError, TypeError) as e:
            # A non-numeric column (string/category/datetime) in a DataFrame
            # with no cat_features: name the offending columns and point at
            # cat_features. pandas nullable NA no longer lands here -- it maps
            # to np.nan in as_model_array.
            #
            # For bare arrays (no column metadata) keep the original numpy
            # error: some sklearn estimator checks rely on its exact
            # type/message.
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
        # Categorical columns are decoded as strings, but the remaining numeric
        # ones must still be finite. Without this check, inf in a numeric column
        # slips silently into the missing bin (binning treats inf as NaN),
        # contradicting the no-cat path's explicit rejection. Only the numeric
        # columns are checked -- the categorical ones are not float-castable.
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
        # without error and silently yield an all-NaN model: mean-1 weight
        # normalization divides by the weight sum.
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
    elif hasattr(estimator, "feature_names_in_"):
        # A refit on name-less input must not keep the previous fit's names. A
        # stale set makes the column-order guard raise on the new fit's valid
        # input -- and pass on stale-matching input, the exact silent
        # wrong-prediction case it exists to stop. sklearn's
        # _check_feature_names deletes the attribute on such refits too.
        del estimator.feature_names_in_

    return y


def _check_feature_names_match(estimator, X):
    """Enforce that predict-time feature names agree with fit (name and order).

    A DataFrame whose columns are renamed or *reordered* relative to training
    otherwise yields silently-wrong predictions, since the booster consumes
    columns positionally. Mirrors sklearn: warn when names are present on only
    one side, raise when they disagree. Uses the same ``X.columns`` extraction
    as the fit-time capture, so the two are directly comparable (pandas/polars).
    """
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
    """Honor scikit-learn's global ``assume_finite`` config.

    Under ``sklearn.set_config(assume_finite=True)`` (or ``config_context``) the
    O(n) predict-time finiteness scan is skipped for maximum inference
    throughput -- the same escape hatch sklearn's own ``check_array`` offers.
    """
    try:
        from sklearn import get_config
        return bool(get_config().get("assume_finite", False))
    except Exception:
        return False


def _numeric_block(X, num_idx):
    """The numeric columns of X (positions ``num_idx``) as float64, or None if
    they aren't float-castable.

    Used for the inf check when categoricals are present. It selects *only* the
    numeric columns, so the often string-heavy categorical ones aren't dragged
    through an expensive object conversion. Maps pandas nullable NA to np.nan
    like the model's own path.
    """
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


def _bag_predict_context(estimator, X):
    """Per-call predict state shared across the members of a bagged ensemble.

    The raw-matrix conversion and a categorical-factorization cache, computed
    once instead of once per member. The caller has already validated ``X`` via
    ``_check_predict_input``, so members skip re-validation entirely. Each
    member's prediction is unchanged -- only the redundant per-member
    conversion, hashing and input checks are gone.
    """
    Xc = as_model_array(X, _was_fit_with_cats(estimator))
    return Xc, CatTransformCache()


def _check_predict_input(estimator, X):
    """Validate X at predict time; return the converted model array, or None.

    Raises NotFittedError if unfitted, then checks that X is 2D with the same
    number of features as training -- mismatched input would otherwise give
    silently-wrong predictions. Messages match scikit-learn's wording.

    The returned array is the model-array conversion the inf check already had
    to perform. Callers thread it through so a DataFrame is materialized once
    per predict instead of twice. It is None when no full conversion happened:
    under ``assume_finite``, or after a conversion failure whose descriptive
    error the booster raises.
    """
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(estimator)

    # A process that only unpickles a model and serves it never calls fit, so
    # the predict path needs its own notice.
    from .warmup import _maybe_notice_cold_compile
    _maybe_notice_cold_compile()

    # A DataFrame whose columns are renamed or *reordered* relative to training
    # otherwise produces silently-wrong predictions.
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

    # Reject inf at predict for the numeric path, mirroring fit. Without it an
    # inf serving value is silently routed to the missing bin and returns the
    # "missing" prediction with no error. This is the only O(n) check on the hot
    # predict path, hence the assume_finite escape hatch for latency-critical
    # serving.
    if _assume_finite():
        return None

    if not _was_fit_with_cats(estimator):
        try:
            Xc = as_model_array(X, want_object=False)
        except (ValueError, TypeError):
            return None
    else:
        # Categorical fit: convert to the object array the booster consumes,
        # then check finiteness of the numeric columns only (the cat columns
        # are strings), mirroring the fit-time check.
        Xc = as_model_array(X, want_object=True)
        num_idx = getattr(_fitted_prep(estimator), "num_features_", None)
        Xf = _numeric_block(Xc, num_idx)
        if Xf is not None and np.isinf(Xf).any():
            raise ValueError(
                "X contains infinity. NaN is accepted (treated as missing), but "
                "inf is not -- clip or clean it first.")
        return Xc

    if np.isinf(Xc).any():
        raise ValueError(
            "X contains infinity. NaN is accepted (treated as missing), but "
            "inf is not -- clip or clean it first.")
    return Xc


def _auto_min_child_weight(n_train):
    """Size-adaptive ``min_child_weight`` used when the classifier leaves it None.

    Oblivious trees UNDERFIT large data at the historical mcw=1: the shared split
    amplifies the min-leaf constraint, since one sparse leaf among 2**depth
    vetoes the whole level. They want a lower min-leaf than leaf-wise trees --
    which is why CatBoost uses min_data_in_leaf=1. But mcw~0 OVERFITS small data,
    because unlike CatBoost we run plain boosting with no ordered-boosting
    regularization.

    So the veto fades with training size: full below ~500 rows, gone above ~2000,
    linear between. The midpoint (~1250 rows, about 20 samples per leaf at depth
    6) lines up with the field-standard min_data_in_leaf=20.
    """
    return float(np.clip((2000.0 - n_train) / 1500.0, 0.0, 1.0))


# Pairwise categorical combinations help when the target depends on categorical
# INTERACTIONS. On mixed data the synthetic combo columns crowd out the numeric
# features that want to split: sign-tested, the all-categorical car and kr-vs-kp
# gain +60% or more while mixed sets regress. So the auto-default turns them on
# ONLY for entirely categorical data -- the one condition where they help with
# no downside.
#
# The two caps below are resource guards, not accuracy knobs: a wide all-cat
# dataset generates C(n_cat, 2) combo columns, each target-encoded over every
# row. Above them the user can still opt in explicitly.
_AUTO_CAT_COMBO_MAX_PAIRS = 1000        # ceiling on C(n_cat, 2) combo columns
_AUTO_CAT_COMBO_MAX_CELLS = 5e7         # ceiling on pairs * n_samples (memory)


def _auto_cat_combinations(cat_features, n_features, n_samples):
    """Resolve ``cat_combinations=None``: True only for tractable all-categorical
    data. ``cat_features`` is the resolved integer-index list, or None."""
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


# Numeric cross features: pair the CROSS_TOP_M most important numeric columns of
# the base fit; each pair contributes a difference and a product column.
# Selection (base vs augmented, on the ES validation split) needs enough rows for
# the validation signal to be trustworthy -- below CROSS_MIN_SAMPLES the val set
# is too small to referee, and small data overfits extra columns first.
CROSS_TOP_M = 6
CROSS_MIN_SAMPLES = 2000

# Group-centered (gdiff) candidates: top numerics x top categoricals. A gdiff
# column x_i - mean(x_i | c_j) makes "above this row's own category's baseline"
# one split -- the num x cat analog of the diff/prod staircase fix.
CROSS_GDIFF_TOP_NUM = 4
CROSS_GDIFF_TOP_CAT = 3


def _cross_candidate_pairs(importances, cat_features, n_features):
    """Candidate (i, j, op) cross features from base-fit importances.

    Oblivious trees can only approximate an interaction with a depth-limited
    staircase, one shared split per level. A difference column makes the
    ``x_i < x_j`` boundary a single split, a product column captures
    multiplicative structure, and a group-centered column makes a within-category
    deviation a single split.

    Numeric pairs are the C(m, 2) combinations of the top-m numeric features by
    split-gain importance; gdiff pairs cross the top numerics with the top
    categoricals. Interactions among features the trees already use are the
    plausible ones, and an irrelevant cross costs only fit time -- the split
    search ignores it, and the validation race referees the whole block.
    """
    cat = set(cat_features or [])
    num_idx = [i for i in range(n_features) if i not in cat]
    if not num_idx:
        return []

    imp = np.asarray(importances, dtype=np.float64)
    key = np.zeros(n_features)
    key[:imp.shape[0]] = imp

    pairs = []
    if len(num_idx) >= 2:
        top = sorted(num_idx, key=lambda i: -key[i])[:CROSS_TOP_M]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                i, j = top[a], top[b]
                pairs.append((i, j, "diff"))
                pairs.append((i, j, "prod"))

    if cat:
        gnum = sorted(num_idx, key=lambda i: -key[i])[:CROSS_GDIFF_TOP_NUM]
        gcat = sorted(cat, key=lambda i: -key[i])[:CROSS_GDIFF_TOP_CAT]
        pairs.extend((i, j, "gdiff") for i in gnum for j in gcat)

    return pairs


def _best_val(booster):
    """Best validation loss a fitted booster reached (inf when no history)."""
    return min(booster.valid_history_) if booster.valid_history_ else np.inf


def _stop_after(k):
    """Fit callback halting boosting after k rounds (selection auditions)."""
    def cb(iteration, train_loss, val_loss, model):
        return iteration + 1 >= k
    cb._cb_needs_train_loss = False
    return cb


def _stop_if_behind(k, target_best):
    """Fit callback killing a challenger at round k unless its best validation
    loss has beaten ``target_best`` by then.

    The raced-selection rule: the winner at the shared budget continues, the
    loser stops. A best-so-far only improves, so a challenger ahead at k is never
    stopped later.
    """
    state = {"best": np.inf}

    def cb(iteration, train_loss, val_loss, model):
        if val_loss is not None and val_loss < state["best"]:
            state["best"] = val_loss
        return iteration + 1 >= k and not state["best"] < target_best
    cb._cb_needs_train_loss = False
    return cb


def _bag_refit_rows(est):
    """The full row set a BAG MEMBER should re-estimate its leaf values on, or
    None when this is not a member of a refitting bag.

    A member trains on ``max_samples`` of the rows and early-stops on the
    out-of-bag complement, so ``auto_split`` is False and the ordinary full-data
    refit never fires: every member's leaf values come from 0.8n rows and nothing
    reclaims the rest. REFIT_PLAN's "bag members have no data tax" is true of the
    ENSEMBLE -- every row is in some member's bag -- but not of any individual
    member.

    Replaying the member's own structure against all-row gradients is safe for
    the ensemble because bag lift is STRUCTURAL diversity, measured in the LRE
    post-mortem ("leaf-only diversity is dead, structural diversity is the
    value"). The structures stay exactly as each member's own bag grew them, and
    only the leaf values change. See benchmarks/BREAKTHROUGH_PLAN.md.
    """
    return getattr(est, "_bag_refit_rows_", None)


def _refit_on_full(est, winner, X_full, y_full, sw_full, cat_features, kw,
                   loss_kwargs=None, replay=False, train_frac=None):
    """Retrain ``winner``'s configuration on all rows (benchmarks/REFIT_PLAN.md).

    Rounds scale by the train-size ratio and the resolved learning rate is
    pinned, so the early-stopped budget keeps its meaning; the selected
    linear-leaf variant and cross pairs carry over. The winner's early-stopping
    curves are copied so ``validation_history_`` still reports the curve that
    chose the budget. Size-adaptive autos (the classifier's min_child_weight,
    cat_combinations) re-resolve at the full row count. User callbacks already
    observed the early-stopping fits and are not re-run.

    ``train_frac`` is the fraction of ``X_full`` the winner actually trained on,
    which sets how far the round budget scales up. It defaults to the
    auto-split's ``1 - validation_fraction``; a bag member passes its
    ``max_samples`` instead, since what its leaf values never saw is the
    out-of-bag complement rather than a validation holdout.
    """
    t_star = len(winner.trees_)
    frac = (1.0 - est.validation_fraction) if train_frac is None \
        else float(train_frac)
    rounds = min(int(np.ceil(t_star / max(frac, 1e-9))),
                 int(est.n_estimators))

    rkw = dict(kw)
    rkw["n_estimators"] = rounds
    rkw["early_stopping_rounds"] = None
    rkw["learning_rate"] = float(winner.lr_)

    # The classifier's auto min_child_weight is size-adaptive (the regressor
    # resolves None to a flat 1.0 before kw is built).
    if est.min_child_weight is None and loss_kwargs is None:
        rkw["min_child_weight"] = _auto_min_child_weight(len(X_full))
    if est.cat_combinations is None:
        rkw["cat_combinations"] = _auto_cat_combinations(
            cat_features, est.n_features_in_, len(X_full))
    rkw.pop("linear_leaves", None)

    # Structure-transfer refit: replay the winner's splits against full-data
    # gradients instead of re-growing them. Scalar booster only -- the multiclass
    # round grows one vector-leaf tree through a separate loop, so it keeps the
    # from-scratch refit.
    multiclass = getattr(est, "_multiclass", False)
    donor = ((winner.trees_, winner.prep_)
             if (replay and not multiclass and winner.trees_) else None)
    if donor is not None:
        # The donor's preprocessor is reused verbatim, so the cross columns and
        # the encoder come with it; passing cross_pairs again would rebuild them.
        rkw["replay_donor"] = donor

    if loss_kwargs is not None:      # scalar regressor path
        b = GradientBoosting(loss=est.loss, loss_kwargs=loss_kwargs,
                             linear_leaves=winner.linear_leaves,
                             cross_pairs=winner.cross_pairs, **rkw)
    elif multiclass:
        b = MulticlassBoosting(linear_leaves=winner.linear_leaves,
                               cross_pairs=winner.cross_pairs, **rkw)
    else:
        b = GradientBoosting(loss="Logloss",
                             linear_leaves=winner.linear_leaves,
                             cross_pairs=winner.cross_pairs, **rkw)

    if est.verbose:
        print(f"refit_full: retraining on all {len(X_full)} rows for "
              f"{rounds} rounds (early stopping chose {t_star})")

    b.fit(X_full, y_full, cat_features=cat_features, sample_weight=sw_full)
    b.train_history_ = winner.train_history_
    b.valid_history_ = winner.valid_history_
    return b


def _add_callback(callbacks, extra):
    """Compose the user callbacks argument (None, a callable, or a sequence)
    with one internal callback. ``extra=None`` returns callbacks unchanged."""
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

    Read more in the [User Guide](https://bbstats.github.io/chimeraboost/recipes/).

    Parameters
    ----------
    n_estimators : int, default 2000
        Maximum number of boosting rounds (trees). With ``early_stopping`` on,
        this is an upper bound and the best round is selected automatically.
    learning_rate : float or None, default None
        Shrinkage applied to each tree. When early stopping is active, ``None``
        resolves to 0.1 on data of about 15,000 training rows or more and fades
        to 0.07 at 5,000 or fewer; see ``adaptive_learning_rate``.
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
    loss : str or object, default "RMSE"
        Training objective. Built in: ``"RMSE"``, ``"MAE"``, ``"Quantile"``
        (level set by ``alpha``), ``"Huber"`` (transition set by ``delta``),
        and the log-link losses ``"Poisson"``, ``"Gamma"``, ``"Tweedie"``
        (power set by ``tweedie_variance_power``), whose predictions are
        ``exp(raw score) > 0``. Alternatively a custom objective *instance*:
        subclass ``chimeraboost.CustomObjective`` and implement
        ``grad_hess(y, raw)`` and ``eval(y, raw, sample_weight=None)``.
    alpha : float, default 0.5
        Quantile level for ``loss="Quantile"`` (e.g. 0.9 for the 90th percentile).
    delta : float, default 1.0
        Huber transition point for ``loss="Huber"``, in y units: quadratic
        within ``delta`` of the target, linear beyond. Fixed, not
        quantile-adaptive -- scale it to the data.
    tweedie_variance_power : float, default 1.5
        Variance power for ``loss="Tweedie"``, strictly between 1 (Poisson)
        and 2 (Gamma).
    eval_metric : callable or None, default None
        Custom validation metric ``metric(y_true, y_pred[, sample_weight]) ->
        float`` scored on the validation set each round and used for early
        stopping and the internal model selections instead of the training
        loss. ``y_pred`` is the prediction (after the loss link). Lower is
        better, unless the callable carries a ``greater_is_better = True``
        attribute -- ``validation_history_`` then records negated values so
        the internal lower-is-better machinery is unchanged. The training
        loss still drives the gradients; the verbose per-round "train" column
        stays in training-loss units.
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
        loss, and >= 1000 rows -- otherwise constant leaves are used). Set
        ``True``/``False`` to force one variant and skip the double fit.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    quantize_gradients : bool, default True
        Run the split search on quantized gradients/hessians packed into
        integer histograms (LightGBM-style quantized training, ~15-bit):
        ~20-25% faster fits at benchmark-flat accuracy. Leaf values always
        use the exact float gradients; the rounding noise touches only
        split selection and is deterministic for a fixed ``random_state``.
        ``False`` restores exact float64 histograms.
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
        averages independent members, each fit on its own random row sample
        (``max_samples``, without replacement by default).
    ensemble_n_jobs : int, default -1
        Worker processes fitting ensemble members concurrently, each on an
        equal share of the thread budget (same total cores as a single fit;
        models are identical either way, wall-clock 1.2-2x faster). -1 sizes
        the pool from the budget, capped at ``n_ensembles``; 1 fits members
        sequentially, each with the full budget.
    max_samples : float, default 0.8
        Fraction of rows each ensemble member trains on, drawn WITHOUT
        replacement ("subagging"). The default 0.8 beats the classic
        bootstrap on strength and fit time (a full-size bootstrap holds
        only ~0.63n unique rows at n rows of compute). 1.0 restores the
        classic full-size with-replacement bootstrap. Unsampled rows are
        each member's early-stopping eval set either way. When ``groups``
        is passed to ``fit``, the draw is over whole groups instead of
        rows (a cluster bootstrap at 1.0), so each member's eval set is
        made of groups it never trained on.
    refit_full : "replay", bool, default "replay"
        After the automatic early-stopping split has chosen the tree budget
        (and model selection / calibration have used it), retrain the winning
        configuration on 100% of the rows -- rounds scaled by the train-size
        ratio, learning rate pinned -- so the final model does not pay the
        holdout data tax. Only affects fits that used the automatic split
        (an explicit ``eval_set`` or ``early_stopping=False`` is unchanged);
        ``loss="Quantile"`` ignores it to keep its conformal holdout honest.
        ``validation_history_`` keeps the early-stopped fit's curve. Costs one
        extra refit; on by default since 0.25.0, as the strongest single-model
        setting measured (benchmarks/SELECT_PLAN.md). Set ``False``, or
        ``quality=2``, for the faster pre-0.25 behaviour.

        **The default is** ``"replay"``, which gets the same thing for about
        two thirds of the cost. Growing trees is 83-85% of a fit and is a
        SEARCH; ``"replay"`` reuses the winner's tree structures and refits
        only the leaf values against full-data gradients, so the held-out rows
        still shape every leaf value while the split search is not paid for
        twice.

        Measured against ``True`` at 3 seeds: on Grinsztajn accuracy was flat
        (27W-32L over 59 datasets, mean +0.005%) and fit time fell 34% on all
        59; on high-cardinality categorical data it ran slightly behind (mean
        -0.256%) for 17% less fit time. Pass ``True`` for the from-scratch
        refit -- marginally stronger on high-card data, and the setting
        benchmarked in REFIT_PLAN.md. Multiclass ignores ``"replay"`` and
        always uses the from-scratch refit (benchmarks/REPLAY_PLAN.md).
    refit_members : bool, default False
        The bagged analogue of ``refit_full``, and off by default. A bag member
        trains on ``max_samples`` of the rows and early-stops on its
        out-of-bag complement, so its leaf values never see the rows it stopped
        on -- the full-data refit that helps a single model has never fired for
        it. With ``True`` each member replays its own tree structure against
        gradients from every row once early stopping is done. Only the leaf
        values move; the splits stay exactly as that member's own sample grew
        them, which is where a bag's diversity actually lives.

        Measured on the decision suites, an 8-member bag improves in every
        stratum, with perfect sweeps on the small-data ones (Grinsztajn at a
        quarter of the rows 12W-0L, +1.206%), for about 10-17% more fit time.
        Because each member is individually stronger you can also spend the
        gain on fewer members: 5 refit members beat a plain 8-member bag on
        accuracy while fitting about 20% faster. Ignored unless the fit is
        bagged (``n_ensembles >= 2``), and ignored for multiclass, where a
        member would need a full refit rather than a cheap structure replay.
    adaptive_learning_rate : bool, default True
        Let the auto ``learning_rate`` depend on how much data it has, instead
        of being size-blind: a linear fade from 0.07 at 5,000 training rows or
        fewer up to the historical flat 0.1 at 15,000 or more. Small data is
        where a lower rate pays, and it is also where the extra trees it needs
        are cheap. Default-on since 0.30.0; above the upper threshold it is a
        no-op and the model is byte-identical to earlier versions, so only
        small-data fits move. Set ``False`` for the flat 0.1 everywhere.

        Measured on the decision suites, the mean is positive in six of seven
        strata, with sign-test passes at a quarter of the rows on both
        Grinsztajn (9W-3L) and high-card (3W-0L) and no losses at all on
        high-card at full size (6W-0L). Gains are individually small (medians
        of +0.13% to +0.31%) and cost 1.09x to 1.31x fit time on the sizes it
        touches. Only consulted when ``learning_rate`` is ``None`` and early
        stopping is on -- without early stopping the rate already scales with
        the round budget. Bagged fits (``n_ensembles >= 2``) are unaffected too,
        since their members already carry an explicit member learning rate.
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

    # quality=1 pins linear leaves here: on the regressor, linear_leaves=None
    # means "audition const vs linear", the search the fast rung declines.
    _QUALITY_PINS_LINEAR_LEAVES = True

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
                 n_ensembles=None, ensemble_n_jobs=-1, max_samples=0.8,
                 cat_features=None, quantize_gradients=True,
                 eval_metric=None, delta=1.0, tweedie_variance_power=1.5,
                 refit_full="replay", refit_members=False, quality=None,
                 adaptive_learning_rate=True):
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
        self.delta = delta
        self.tweedie_variance_power = tweedie_variance_power
        self.eval_metric = eval_metric
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
        self.max_samples = max_samples
        self.quantize_gradients = quantize_gradients
        self.refit_full = refit_full
        self.refit_members = refit_members
        self.quality = quality
        # Size fade for the auto learning rate, default-on since 0.30.0; only
        # consulted when learning_rate is None. False == the historical flat 0.1.
        self.adaptive_learning_rate = adaptive_learning_rate

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
            Explicit validation set. When provided, automatic splitting is
            skipped regardless of the *early_stopping* setting.
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples (e.g. ``df['subject_id']``). When
            supplied and *early_stopping* triggers an automatic split, groups
            are kept intact across the train/validation boundary using
            ``GroupShuffleSplit``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights, normalized to mean 1 internally. Applied
            throughout: the gradient/leaf fit, the categorical target encoder,
            the quantile bin borders, and the early-stopping metric on an
            automatically split (or bagged out-of-bag) validation set, so a
            zero-weight row never influences the model. An explicitly passed
            ``eval_set`` carries no weights and is scored unweighted.
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

            # A reordered or renamed eval DataFrame is consumed positionally and
            # silently wrecks early stopping, and everything calibrated on the
            # holdout. Bag members skip this: their eval sets are built
            # internally from already-validated parent input.
            if not getattr(self, "_is_bag_member", False):
                _check_feature_names_match(self, eval_set[0])

        with _quality_applied(self):
            if self.n_ensembles and self.n_ensembles > 1:
                if callbacks is not None:
                    raise ValueError(
                        "callbacks are not supported with n_ensembles > 1.")
                self.estimators_ = _fit_bagged(self, X, y, cat_features,
                                               eval_set, groups, sample_weight)
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

        # The booster silently drops linear_leaves for MAE/Quantile: their leaf
        # values are the residual median or quantile, not a Newton step a ridge
        # slope could refine. Warn so it isn't mistaken for active.
        if self.linear_leaves and self.loss in ("MAE", "Quantile"):
            warnings.warn(
                f"linear_leaves is not supported with loss={self.loss!r} and "
                "will be ignored.", UserWarning, stacklevel=2)

        # Same honesty for the other knobs MAE/Quantile shadow: they re-estimate
        # leaf values directly (the booster's adjusts_leaves path), which owns
        # the training step.
        if self.loss in ("MAE", "Quantile"):
            if self.ordered_boosting:
                warnings.warn(
                    f"ordered_boosting is ignored with loss={self.loss!r} and "
                    "will have no effect.", UserWarning, stacklevel=2)
            if self.leaf_estimation_iterations > 1:
                warnings.warn(
                    f"leaf_estimation_iterations is ignored with "
                    f"loss={self.loss!r} and will have no effect.",
                    UserWarning, stacklevel=2)

        # Kept for the optional full-data refit below: the auto split
        # reassigns X/y, but the refit retrains on every row.
        X_full, y_full, sw_full = X, y, sample_weight
        es_active, auto_split, X, y, sample_weight, eval_set = _auto_es_split(
            self, X, y, sample_weight, eval_set, groups, stratify=None)

        # Mirror the classifier's inert-setting warnings: an explicit
        # linear_leaves=True shadows ordered boosting and leaf refinement once
        # the booster activates it, at >= LINEAR_LEAVES_MIN_SAMPLES train rows.
        # The None auto-audition is exempt -- it decides on validation loss.
        ll_shadows = (self.linear_leaves is True
                      and self.loss not in ("MAE", "Quantile")
                      and len(X) >= LINEAR_LEAVES_MIN_SAMPLES)
        if ll_shadows and self.ordered_boosting:
            warnings.warn(
                "ordered_boosting is ignored while linear leaves are active "
                "(the per-leaf linear model owns the training step); set "
                "linear_leaves=False to use it.", UserWarning, stacklevel=2)
        if ll_shadows and self.leaf_estimation_iterations > 1:
            warnings.warn(
                "leaf_estimation_iterations is ignored while linear leaves "
                "are active; set linear_leaves=False to use it.",
                UserWarning, stacklevel=2)

        # Default patience when early stopping is on: 50 beat 10 on 25 of 34
        # benchmark datasets when measured, because lr=0.1 keeps improving past
        # a 10-round plateau.
        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50

        loss_kwargs = {}
        if self.loss == "Quantile":
            loss_kwargs["alpha"] = self.alpha
        elif self.loss == "Huber":
            loss_kwargs["delta"] = self.delta
        elif self.loss == "Tweedie":
            loss_kwargs["power"] = self.tweedie_variance_power

        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha", "delta",
                           "tweedie_variance_power"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds

        # Loss-adaptive depth default (see the `depth` docstring): 6 for
        # RMSE/MAE, 4 for Quantile, where deep leaves overfit the tail quantile
        # and predictions collapse toward the median.
        if kw.get("depth") is None:
            kw["depth"] = 4 if self.loss == "Quantile" else 6

        # min_child_weight is a no-op for regression in [0, 1] -- a non-empty
        # child always holds >= 1 sample, so hess >= 1. Resolve None to 1.0.
        if kw.get("min_child_weight") is None:
            kw["min_child_weight"] = 1.0

        # The regressor default is a concrete 1, but the shared validator also
        # accepts None (the classifier's auto default); resolve that to 1.
        if kw.get("leaf_estimation_iterations") is None:
            kw["leaf_estimation_iterations"] = 1

        # colsample None = auto: full columns for a single model (the bagged
        # path resolves members to 0.85 before this runs; see _fit_bagged).
        if kw.get("colsample") is None:
            kw["colsample"] = 1.0

        # cat_combinations auto: on only for tractable all-categorical data.
        if kw.get("cat_combinations") is None:
            kw["cat_combinations"] = _auto_cat_combinations(
                cat_features, self.n_features_in_, len(X))

        # linear_leaves=None means validation-selected: fit the constant-leaf and
        # linear-leaf variants and keep whichever reaches the lower validation
        # loss. Full-Grinsztajn breadth showed fixed linear leaves are a wash for
        # regression (16W/12L) with real casualties; per-dataset selection on the
        # already-held-out early-stopping split banks the wins without them --
        # the same post-fit-decision pattern as temperature scaling and conformal
        # quantiles. Selection needs a validation set, RMSE (MAE and Quantile
        # override leaf values), and enough rows for linear leaves to engage.
        ll = kw.pop("linear_leaves")
        select_ll = (ll is None and self.loss == "RMSE" and eval_set is not None
                     and len(X) >= LINEAR_LEAVES_MIN_SAMPLES)

        # Cross-features applicability, decidable before any fit: pair candidates
        # exist iff there are >= 2 numeric columns (diff/prod), or >= 1 numeric
        # and >= 1 categorical (gdiff group-centering).
        n_cats = len(cat_features) if cat_features else 0
        n_nums = X.shape[1] - n_cats
        cross_ok = (self.cross_features is not False and self.loss == "RMSE"
                    and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES
                    and (n_nums >= 2 or (n_nums >= 1 and n_cats >= 1)))

        # One prep cache for every booster fit below: the auditions, the
        # cross-augmented candidate, and the winner refit all see identical
        # (X, y, cat_features, eval_set), so their preprocessing is computed
        # once and reused bit-identically (see GradientBoosting._prep_matrices).
        prep_cache = {}

        def _fit_booster(linear, cross_pairs=None, stop=None):
            b = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                 linear_leaves=linear, cross_pairs=cross_pairs,
                                 **kw)
            b.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                  sample_weight=sample_weight,
                  callbacks=_add_callback(callbacks, stop),
                  prep_cache=prep_cache)
            return b

        self.linear_leaves_selected_ = None
        self.cross_features_selected_ = None
        self.cross_pairs_ = None

        if self.selection_rounds is not None and (select_ll or cross_ok):
            # Cheap selection (benchmarks/PARETO_PLAN.md step 2, fallback
            # design). Every selection fit runs as a short audition. The
            # cross-augmented candidate wins the full selection on the vast
            # majority of datasets, so it gets the one full fit; the audition
            # winner is refit in full only when the augmented model loses or
            # cross features do not apply.
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
            # IS the full fit -- same config, seed and curve -- so refitting it
            # would only re-pay for an identical model. Refit only if truncated.
            capped = len(audition.valid_history_) >= self.selection_rounds

            pairs = (_cross_candidate_pairs(audition.feature_importances_,
                                            cat_features, X.shape[1])
                     if cross_ok else [])
            if pairs:
                # Symmetric race at the shared budget, the rule the step-0 race
                # simulation validated: both candidates are judged on their best
                # validation loss within the first selection_rounds, a trailing
                # augmented fit is killed at the budget, and a leading one
                # continues to its own full early stop. Comparing the augmented
                # fit's FULL best against a capped audition would bias selection
                # toward it -- its extra rounds are not evidence the audition
                # couldn't have matched.
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

        # Numeric cross features (default-on auto): refit with difference and
        # product columns for the top numeric feature pairs of the base fit, and
        # keep whichever model reaches the lower validation loss -- the same
        # selection-on-the-validation-split pattern as linear_leaves above.
        # Oblivious trees can only staircase a numeric interaction
        # (benchmarks/probe_cross_features.py; Grinsztajn A/B 51W/8L, mean
        # +1.5%), and selection dodges the variance cases. RMSE only, the probed
        # loss; None (auto) and True behave the same. A selection_rounds audition
        # already handled this above.
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

        # The winner is chosen; free the cached binned matrices now rather than
        # holding them through the conformal step below.
        prep_cache.clear()

        # Conformal quantile correction on the validation split -- the
        # regression analog of the classifier's temperature scaling.
        #
        # Boosting under-disperses quantiles: each round's per-leaf quantile step
        # is shrunk by the learning rate, so the additive model converges to the
        # tail slowly and early stopping cuts it short, collapsing predictions
        # toward the median. The split-conformal fix shifts every prediction by
        # the k-th order statistic of the validation residuals,
        # k = ceil((n+1) * alpha). That is the standard conformal rank, and it
        # also minimizes pinball loss over all constant shifts, so accuracy and
        # coverage improve together. Distribution-free marginal coverage on
        # exchangeable data (Romano, Patterson & Candes 2019).
        self.quantile_offset_ = 0.0
        if self.loss == "Quantile" and eval_set is not None:
            resid = (np.asarray(eval_set[1], dtype=np.float64)
                     - self.model_.predict_raw(eval_set[0]))
            w_val = (np.asarray(eval_set[2], dtype=np.float64)
                     if len(eval_set) > 2 and eval_set[2] is not None
                     else None)

            if w_val is None:
                resid = np.sort(resid)
                k = min(int(np.ceil((resid.shape[0] + 1) * self.alpha)),
                        resid.shape[0])
                if k >= 1:
                    self.quantile_offset_ = float(resid[k - 1])
            else:
                # Weighted conformal rank: each row contributes its weight of
                # mass, so a zero-weight holdout row cannot set the offset (the
                # sample_weight contract). Under uniform weights the threshold is
                # alpha*(n+1) over unit masses -- exactly the unweighted k-th
                # order statistic above.
                keep = w_val > 0
                resid, w_val = resid[keep], w_val[keep]
                if resid.shape[0] >= 1:
                    order = np.argsort(resid)
                    resid, w_val = resid[order], w_val[order]
                    total = float(w_val.sum())
                    n_eff = resid.shape[0]
                    thr = self.alpha * total * (n_eff + 1) / n_eff
                    j = int(np.searchsorted(np.cumsum(w_val), thr,
                                            side="left"))
                    self.quantile_offset_ = float(resid[min(j, n_eff - 1)])

        # Full-data refit (benchmarks/REFIT_PLAN.md). Once early stopping,
        # selection and calibration have consumed the auto-split holdout it is a
        # pure data tax, so retrain the winning configuration on 100% of the rows
        # at the selected budget: rounds scaled by the train-size ratio, the
        # resolved learning rate pinned so the budget keeps its meaning. Only the
        # auto-split path ever paid the tax -- an explicit eval_set or
        # early_stopping=False is untouched -- and Quantile keeps its genuine
        # conformal holdout instead.
        if (self.refit_full and auto_split and self.loss != "Quantile"
                and self.model_.trees_):
            self.model_ = _refit_on_full(
                self, self.model_, X_full, y_full, sw_full, cat_features, kw,
                loss_kwargs=loss_kwargs, replay=self.refit_full == "replay")
        elif _bag_refit_rows(self) is not None and self.loss != "Quantile" \
                and self.model_.trees_:
            bx, byy, bsw, bfrac = _bag_refit_rows(self)
            self.model_ = _refit_on_full(
                self, self.model_, bx, byy, bsw, cat_features, kw,
                loss_kwargs=loss_kwargs, replay=True, train_frac=bfrac)

        return self

    def _transform_raw(self, raw):
        """Map raw additive scores to predictions through the loss link.

        Identity for RMSE/MAE/Quantile/Huber -- ``raw`` is returned unchanged, so
        those paths stay bit-identical. ``exp`` for Poisson/Gamma/Tweedie, and a
        custom loss's ``transform`` when it defines one.
        """
        tf = getattr(self.model_.loss_, "transform", None)
        return raw if tf is None else tf(raw)

    def predict(self, X):
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        if self.estimators_ is not None:
            # One shared conversion and factorization cache for the whole bag,
            # and one thread-limit switch (members' re-entries are no-ops). Each
            # member applies its own link and conformal offset; the bag
            # prediction is the mean of the member predictions.
            with _thread_limit(self.thread_count):
                Xc, ctx = _bag_predict_context(self, X)
                return np.mean(
                    [m._transform_raw(m.model_.predict_raw(Xc, ctx))
                     + m.quantile_offset_
                     for m in self.estimators_], axis=0)
        return self._transform_raw(self.model_.predict_raw(X)) \
            + self.quantile_offset_

    def staged_predict(self, X):
        """Yield the prediction after each successive tree.

        The conformal quantile offset is a post-fit constant and is included in
        every stage, so the final stage equals ``predict``.
        """
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        if self.estimators_ is not None:
            raise NotImplementedError("staged_predict is not defined for a "
                                      "bagged ensemble (n_ensembles > 1).")

        for staged in self.model_.staged_predict_raw(X):
            yield self._transform_raw(staged) + self.quantile_offset_

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

    @property
    def validation_history_(self):
        """Per-round validation score recorded during ``fit``.

        The training loss (RMSE space for regression), or the custom
        ``eval_metric`` when one was set -- negated if the metric declares
        ``greater_is_better = True``, so lower is always better here.

        A list as long as the number of rounds run. Empty when no ``eval_set``
        or early-stopping split was available; a list of the members' histories
        for a bagged model (``n_ensembles > 1``).
        """
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
        ``n_ensembles > 1`` -- the bag prediction is the members' mean, so the
        averaged attribution stays exact. ``X_background`` overrides the
        reference distribution (default: a sample of the training data).

        For the log-link losses (Poisson/Gamma/Tweedie) and custom losses with a
        non-identity ``transform``, attributions are in raw (link) space: rows
        sum to the log of the prediction, the usual GBDT margin-space convention.
        """
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        if X_background is not None:
            # The background matrix is consumed positionally too; a reordered
            # DataFrame would silently skew every baseline.
            bg = _check_predict_input(self, X_background)
            X_background = X_background if bg is None else bg

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

    Read more in the [User Guide](https://bbstats.github.io/chimeraboost/recipes/).

    Parameters
    ----------
    n_estimators : int, default 2000
        Maximum number of boosting rounds (trees). With ``early_stopping`` on,
        this is an upper bound and the best round is selected automatically.
    learning_rate : float or None, default None
        Shrinkage applied to each tree. When early stopping is active, ``None``
        resolves to 0.1 on data of about 15,000 training rows or more and fades
        to 0.07 at 5,000 or fewer; see ``adaptive_learning_rate``.
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
    leaf_estimation_iterations : int or None, default None
        Extra Newton refinement steps per leaf. ``None`` is the auto default and
        resolves to 3, which helps small and categorical-heavy binary fits.
        Refinement only applies to the plain constant-leaf path: it is inert
        while ``linear_leaves`` is active (default-on for binary at >= ~1000
        rows, where the per-leaf ridge already fits the second-order-optimal
        leaf) and is not implemented for multiclass. An explicitly-set value
        that will be ignored on the path about to run warns.
    linear_leaves : bool or None, default None
        Fit a ridge linear model per leaf over the numeric split features instead
        of a constant. ``None`` enables it for binary classification and disables
        it for multiclass (where it is unsupported). Below ~1000 rows it falls
        back to constant leaves.
    linear_lambda : float, default 1.0
        Ridge penalty on per-leaf linear slopes; larger is closer to a constant.
    quantize_gradients : bool, default True
        Run the split search on quantized gradients/hessians packed into
        integer histograms (LightGBM-style quantized training, ~15-bit):
        ~20-25% faster fits at benchmark-flat accuracy. Leaf values always
        use the exact float gradients; the rounding noise touches only
        split selection and is deterministic for a fixed ``random_state``.
        ``False`` restores exact float64 histograms.
    eval_metric : callable or None, default None
        Custom validation metric ``metric(y_true, y_pred[, sample_weight]) ->
        float`` scored on the validation set each round and used for early
        stopping and the internal model selections instead of log loss.
        Binary: ``y_true`` is the 0/1-encoded target and ``y_pred`` the
        positive-class probability; multiclass: ``y_true`` is one-hot
        ``(n, K)`` and ``y_pred`` the probability matrix. Lower is better,
        unless the callable carries a ``greater_is_better = True`` attribute
        -- ``validation_history_`` then records negated values so the
        internal lower-is-better machinery is unchanged. Temperature scaling
        still calibrates on log loss.
    cross_features : bool or None, default None
        Numeric interaction columns. ``None`` (the default) refits the model
        with difference and product columns for the pairs of the top numeric
        features of the base fit and keeps whichever model reaches the lower
        validation loss (``cross_features_selected_`` records the outcome,
        ``cross_pairs_`` the columns kept); needs >= 2000 rows and >= 2
        numeric features. Binary judges on binary log loss, multiclass on
        softmax log loss. ``False`` turns it off. Costs up to ~2x fit time
        when the refit runs.
    selection_rounds : int or None, default 100
        Round budget for the pre-cross base fit when the cross-features refit
        will run. The base fit is an audition capped at this many rounds;
        the candidates are judged on their best validation loss within the
        budget, the winner continues to full early stopping, and the base is
        refit in full only if the augmented model loses after being
        truncated by the cap. ``None`` runs the base fit to full early
        stopping instead (the pre-0.15 behavior).
    early_stopping : bool, default True
        Hold out a stratified validation split and stop when it stops improving.
        ``StratifiedGroupKFold`` is used when ``groups`` is passed to ``fit``.
    validation_fraction : float, default 0.2
        Validation fraction used when ``early_stopping`` is on and no ``eval_set``
        is passed to ``fit``.
    n_ensembles : int or None, default None
        Number of bagged members. ``None`` or 1 trains a single model; >= 2
        soft-votes the calibrated probabilities of members, each fit on its
        own random row sample (``max_samples``, without replacement by
        default).
    ensemble_n_jobs : int, default -1
        Worker processes fitting ensemble members concurrently, each on an
        equal share of the thread budget (same total cores as a single fit;
        models are identical either way, wall-clock 1.2-2x faster). -1 sizes
        the pool from the budget, capped at ``n_ensembles``; 1 fits members
        sequentially, each with the full budget.
    max_samples : float, default 0.8
        Fraction of rows each ensemble member trains on, drawn WITHOUT
        replacement ("subagging"). The default 0.8 beats the classic
        bootstrap on strength and fit time (a full-size bootstrap holds
        only ~0.63n unique rows at n rows of compute). 1.0 restores the
        classic full-size with-replacement bootstrap. Unsampled rows are
        each member's early-stopping eval set either way. When ``groups``
        is passed to ``fit``, the draw is over whole groups instead of
        rows (a cluster bootstrap at 1.0), so each member's eval set is
        made of groups it never trained on.
    refit_full : "replay", bool, default "replay"
        After the automatic early-stopping split has chosen the tree budget
        (and model selection / temperature scaling have used it), retrain the
        winning configuration on 100% of the rows -- rounds scaled by the
        train-size ratio, learning rate pinned -- so the final model does not
        pay the holdout data tax. Only affects fits that used the automatic
        split (an explicit ``eval_set`` or ``early_stopping=False`` is
        unchanged); the calibrated temperature transfers to the refit model.
        ``validation_history_`` keeps the early-stopped fit's curve. Costs one
        extra refit; on by default since 0.25.0, as the strongest single-model
        setting measured (benchmarks/SELECT_PLAN.md). Set ``False``, or
        ``quality=2``, for the faster pre-0.25 behaviour.

        **The default is** ``"replay"``, which gets the same thing for about
        two thirds of the cost. Growing trees is 83-85% of a fit and is a
        SEARCH; ``"replay"`` reuses the winner's tree structures and refits
        only the leaf values against full-data gradients, so the held-out rows
        still shape every leaf value while the split search is not paid for
        twice.

        Measured against ``True`` at 3 seeds: on Grinsztajn accuracy was flat
        (27W-32L over 59 datasets, mean +0.005%) and fit time fell 34% on all
        59; on high-cardinality categorical data it ran slightly behind (mean
        -0.256%) for 17% less fit time. Pass ``True`` for the from-scratch
        refit -- marginally stronger on high-card data, and the setting
        benchmarked in REFIT_PLAN.md. Multiclass ignores ``"replay"`` and
        always uses the from-scratch refit (benchmarks/REPLAY_PLAN.md).
    refit_members : bool, default False
        The bagged analogue of ``refit_full``, and off by default. A bag member
        trains on ``max_samples`` of the rows and early-stops on its
        out-of-bag complement, so its leaf values never see the rows it stopped
        on -- the full-data refit that helps a single model has never fired for
        it. With ``True`` each member replays its own tree structure against
        gradients from every row once early stopping is done. Only the leaf
        values move; the splits stay exactly as that member's own sample grew
        them, which is where a bag's diversity actually lives.

        Measured on the decision suites, an 8-member bag improves in every
        stratum, with perfect sweeps on the small-data ones (Grinsztajn at a
        quarter of the rows 12W-0L, +1.206%), for about 10-17% more fit time.
        Because each member is individually stronger you can also spend the
        gain on fewer members: 5 refit members beat a plain 8-member bag on
        accuracy while fitting about 20% faster. Ignored unless the fit is
        bagged (``n_ensembles >= 2``), and ignored for multiclass, where a
        member would need a full refit rather than a cheap structure replay.
    adaptive_learning_rate : bool, default True
        Let the auto ``learning_rate`` depend on how much data it has, instead
        of being size-blind: a linear fade from 0.07 at 5,000 training rows or
        fewer up to the historical flat 0.1 at 15,000 or more. Small data is
        where a lower rate pays, and it is also where the extra trees it needs
        are cheap. Default-on since 0.30.0; above the upper threshold it is a
        no-op and the model is byte-identical to earlier versions, so only
        small-data fits move. Set ``False`` for the flat 0.1 everywhere.

        Measured on the decision suites, the mean is positive in six of seven
        strata, with sign-test passes at a quarter of the rows on both
        Grinsztajn (9W-3L) and high-card (3W-0L) and no losses at all on
        high-card at full size (6W-0L). Gains are individually small (medians
        of +0.13% to +0.31%) and cost 1.09x to 1.31x fit time on the sizes it
        touches. Only consulted when ``learning_rate`` is ``None`` and early
        stopping is on -- without early stopping the rate already scales with
        the round budget. Bagged fits (``n_ensembles >= 2``) are unaffected too,
        since their members already carry an explicit member learning rate.
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

    # Not pinned on the classifier: linear_leaves=None is already an auto rule
    # (on for binary, off for multiclass -- where an explicit True raises) and
    # costs no extra fit, so quality=1 leaves it alone.
    _QUALITY_PINS_LINEAR_LEAVES = False

    def __init__(self, n_estimators=2000, learning_rate=None, depth=6,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0, colsample=None,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 min_child_weight=None, thread_count=None, random_state=None,
                 verbose=False, ordered_boosting=False,
                 cat_combinations=None, leaf_estimation_iterations=None,
                 linear_leaves=None, linear_lambda=1.0, cross_features=None,
                 selection_rounds=100,
                 early_stopping=True, validation_fraction=0.2,
                 n_ensembles=None, ensemble_n_jobs=-1, max_samples=0.8,
                 cat_features=None, quantize_gradients=True,
                 eval_metric=None, refit_full="replay", refit_members=False,
                 quality=None, adaptive_learning_rate=True):
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
        self.eval_metric = eval_metric
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
        self.max_samples = max_samples
        self.quantize_gradients = quantize_gradients
        self.refit_full = refit_full
        self.refit_members = refit_members
        self.quality = quality
        # Size fade for the auto learning rate, default-on since 0.30.0; only
        # consulted when learning_rate is None. False == the historical flat 0.1.
        self.adaptive_learning_rate = adaptive_learning_rate

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
            Explicit validation set with original class labels. When provided,
            automatic splitting is skipped.
        groups : array-like of shape (n_samples,) or None
            Group labels (e.g. ``df['subject_id']``). When supplied and early
            stopping triggers an automatic split, ``StratifiedGroupKFold`` keeps
            groups intact and class proportions balanced across the split.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights, normalized to mean 1 internally. Applied
            throughout: the gradient/leaf fit, the categorical target encoder,
            the quantile bin borders, and the early-stopping metric on an
            automatically split (or bagged out-of-bag) validation set, so a
            zero-weight row never influences the model. An explicitly passed
            ``eval_set`` carries no weights and is scored unweighted.
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
            _check_eval_set(eval_set, self.n_features_in_,
                            classification=True)
            if not getattr(self, "_is_bag_member", False):
                _check_feature_names_match(self, eval_set[0])

            # Bag members are exempt: their OOB eval set may legitimately hold
            # a rare label their row sample missed, and the parent aligns
            # member probability columns to the global class set.
            if not getattr(self, "_is_bag_member", False):
                _check_eval_labels(eval_set, y)

        with _quality_applied(self):
            if self.n_ensembles and self.n_ensembles > 1:
                if callbacks is not None:
                    raise ValueError(
                        "callbacks are not supported with n_ensembles > 1.")

                # Fix the global class set up front: a member's bootstrap may
                # miss a rare class, and predict_proba aligns each member's
                # columns to this.
                yarr = np.asarray(y)
                self.classes_ = np.unique(yarr)
                self.n_classes_ = self.classes_.size
                if self.n_classes_ < 2:
                    raise ValueError(
                        f"Need at least 2 classes; got {self.n_classes_} "
                        "class(es).")

                self._multiclass = self.n_classes_ > 2
                self.estimators_ = _fit_bagged(self, X, yarr, cat_features,
                                               eval_set, groups, sample_weight)
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
        all_classes = self.classes_
        if self.n_classes_ < 2:
            raise ValueError(
                f"Need at least 2 classes; got {self.n_classes_} class(es).")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Kept for the optional full-data refit below: the auto split
        # reassigns X/y, but the refit retrains on every row.
        X_full, y_full, sw_full = X, y, sample_weight
        es_active, auto_split, X, y, sample_weight, eval_set = _auto_es_split(
            self, X, y, sample_weight, eval_set, groups,
            stratify=y)  # always stratify for classification

        if auto_split:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.size
            lost = np.setdiff1d(all_classes, self.classes_)
            if lost.size and not getattr(self, "_is_bag_member", False):
                # Without this the model silently trains without the lost class
                # and never predicts it -- its eval rows are even remapped onto
                # neighboring classes.
                raise ValueError(
                    f"The automatic early-stopping split left class(es) "
                    f"{lost.tolist()} entirely in the validation set, "
                    "usually because the class is rare or (with `groups`) "
                    "confined to a single group. Pass an explicit "
                    "eval_set, lower validation_fraction, or set "
                    "early_stopping=False.")

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50   # see GradientBoosting/Regressor note above

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds

        # Size-adaptive min_child_weight (see _auto_min_child_weight), resolved
        # on the FINAL training set, after the early-stopping split.
        if kw.get("min_child_weight") is None:
            kw["min_child_weight"] = _auto_min_child_weight(len(X))

        # An explicit depth=None resolves to the documented classifier default
        # (the regressor resolves it per loss); the booster requires an int.
        if kw.get("depth") is None:
            kw["depth"] = 6

        # leaf_estimation_iterations None = auto -> 3, the classifier's
        # long-standing effective value: it helps small and categorical binary
        # fits, and is inert where linear leaves take over or for multiclass.
        # None just keeps the API from advertising a concrete count that is dead
        # in those regimes. The booster requires an int.
        if kw.get("leaf_estimation_iterations") is None:
            kw["leaf_estimation_iterations"] = 3

        # colsample None = auto: full columns for a single model (the bagged
        # path resolves members to 0.85 before this runs; see _fit_bagged).
        if kw.get("colsample") is None:
            kw["colsample"] = 1.0

        # cat_combinations auto: on only for tractable all-categorical data,
        # which targets the all-categorical multiclass gap (car and friends).
        if kw.get("cat_combinations") is None:
            kw["cat_combinations"] = _auto_cat_combinations(
                cat_features, self.n_features_in_, len(X))

        self._multiclass = self.n_classes_ > 2

        # linear_leaves auto-default: ON for binary (a clean broad Brier win that
        # survives bagging), OFF for multiclass, where it is unsupported. An
        # explicit True on multiclass is a user error and raises; an explicit
        # False is honored everywhere.
        if self.linear_leaves is None:
            kw["linear_leaves"] = not self._multiclass
        elif self.linear_leaves and self._multiclass:
            raise NotImplementedError(
                "linear_leaves is not supported for multiclass classification "
                "yet; use it on regression or binary classification.")

        # Warn when an explicitly non-default setting will be silently inert on
        # the path about to run. The auto defaults stay quiet -- their precedence
        # is documented. The linear-leaf update owns the training step, so it
        # shadows both ordered boosting and leaf refinement, and multiclass never
        # refines leaves. leaf_estimation_iterations is checked against the raw
        # attribute (None = auto, no user intent) rather than the resolved kw,
        # and a refinement count of 1 asks for nothing, so neither warns.
        ll_shadows = (not self._multiclass and kw.get("linear_leaves")
                      and len(X) >= LINEAR_LEAVES_MIN_SAMPLES)
        lei_set = self.leaf_estimation_iterations
        lei_wanted = lei_set is not None and lei_set > 1
        if ll_shadows and self.ordered_boosting:
            warnings.warn(
                "ordered_boosting is ignored while linear leaves are active "
                "(the per-leaf linear model owns the training step); set "
                "linear_leaves=False to use it.", UserWarning, stacklevel=2)
        if ll_shadows and lei_wanted:
            warnings.warn(
                "leaf_estimation_iterations is ignored while linear leaves "
                "are active; set linear_leaves=False to use it.",
                UserWarning, stacklevel=2)
        if self._multiclass and lei_wanted:
            warnings.warn(
                "leaf_estimation_iterations is not implemented for multiclass "
                "and will be ignored.", UserWarning, stacklevel=2)

        # cross_features: None (the auto default) and True run the same
        # validation-selected race everywhere, multiclass included (M1); an
        # explicit False disables it.
        cal_Xv = cal_y = cal_w = None  # validation set used to calibrate temperature

        # Cheap selection (benchmarks/PARETO_PLAN.md step 2): when
        # selection_rounds is set and the cross refit will run (pair candidates
        # exist iff >= 2 numeric columns), the base fit is only an audition, so
        # cap it. It is refit in full below only if the augmented model loses.
        n_cats = len(cat_features) if cat_features else 0
        n_nums = X.shape[1] - n_cats
        fast = (self.selection_rounds is not None
                and self.cross_features is not False
                and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES
                and (n_nums >= 2 or (n_nums >= 1 and n_cats >= 1)))
        stop = _stop_after(self.selection_rounds) if fast else None

        # Shared across the base fit, the cross-augmented candidate and the
        # possible refit below: identical inputs, so preprocessing is computed
        # once (see _BaseBooster._prep_matrices).
        prep_cache = {}

        if self._multiclass:
            def _make(**extra):
                return MulticlassBoosting(**extra, **kw)
            y_fit = y
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_w = eval_set[2] if len(eval_set) > 2 else None
        else:
            def _make(**extra):
                return GradientBoosting(loss="Logloss", **extra, **kw)
            y_fit = (y == self.classes_[1]).astype(np.float64)
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_y = (np.asarray(eval_set[1]) == self.classes_[1]).astype(np.float64)
                # Preserve any val-row weights through the 0/1 relabeling.
                sw_v = eval_set[2] if len(eval_set) > 2 else None
                cal_w = sw_v
                eval_set = (cal_Xv, cal_y, sw_v)

        self.model_ = _make()
        self.model_.fit(X, y_fit, cat_features=cat_features, eval_set=eval_set,
                        sample_weight=sample_weight,
                        callbacks=_add_callback(callbacks, stop),
                        prep_cache=prep_cache)

        if self._multiclass:
            self.classes_ = self.model_.classes_
            if eval_set is not None:
                cal_y = np.searchsorted(self.classes_, np.asarray(eval_set[1]))

        # Numeric cross features (default-on auto): refit with difference and
        # product columns for the top numeric feature pairs of the base fit, and
        # keep the lower-validation-loss model -- the regressor's selection
        # pattern, see _cross_candidate_pairs. Binary judges on binary log loss,
        # multiclass on softmax log loss, at the same raced budget.
        self.cross_features_selected_ = None
        self.cross_pairs_ = None

        if (self.cross_features is not False
                and eval_set is not None and len(X) >= CROSS_MIN_SAMPLES):
            pairs = _cross_candidate_pairs(
                self.model_.feature_importances_, cat_features, X.shape[1])
            if pairs:
                aug = _make(cross_pairs=pairs)
                if fast:
                    # Symmetric race at the shared budget (see the regressor):
                    # judge both candidates on their first selection_rounds and
                    # kill a trailing augmented fit at the budget.
                    base_best = _best_val(self.model_)
                    aug.fit(X, y_fit, cat_features=cat_features,
                            eval_set=eval_set, sample_weight=sample_weight,
                            callbacks=_add_callback(
                                callbacks, _stop_if_behind(
                                    self.selection_rounds, base_best)),
                            prep_cache=prep_cache)
                    self.cross_features_selected_ = (
                        min(aug.valid_history_[:self.selection_rounds])
                        < base_best) if aug.valid_history_ else False
                else:
                    aug.fit(X, y_fit, cat_features=cat_features,
                            eval_set=eval_set, sample_weight=sample_weight,
                            callbacks=callbacks, prep_cache=prep_cache)
                    self.cross_features_selected_ = \
                        _best_val(aug) < _best_val(self.model_)

                if self.cross_features_selected_:
                    self.model_ = aug
                    self.cross_pairs_ = pairs
                elif fast and len(self.model_.valid_history_) >= self.selection_rounds:
                    # The incumbent audition really was truncated by the cap, so
                    # give the winning base variant its full fit. (An audition
                    # that early-stopped on its own already IS the full fit.)
                    self.model_ = _make()
                    self.model_.fit(X, y_fit, cat_features=cat_features,
                                    eval_set=eval_set,
                                    sample_weight=sample_weight,
                                    callbacks=callbacks,
                                    prep_cache=prep_cache)

        # The winner is chosen; free the cached binned matrices rather than
        # holding them through temperature scaling below.
        prep_cache.clear()

        # Temperature scaling on the validation set. Dividing raw scores by T > 0
        # is monotonic, so predict() is unchanged while predict_proba() becomes
        # better calibrated (lower log loss).
        self.temperature_ = 1.0
        if cal_Xv is not None:
            raw = self.model_.predict_raw(cal_Xv)
            self.temperature_ = _fit_temperature(raw, cal_y, self._multiclass,
                                                 sample_weight=cal_w)

        # Full-data refit (benchmarks/REFIT_PLAN.md): reclaim the auto-split data
        # tax once early stopping, selection and temperature scaling have
        # consumed the holdout. The temperature above was calibrated on the
        # early-stopping winner's validation scores and transfers to the refit
        # model.
        if self.refit_full and auto_split and self.model_.trees_:
            y_refit = (y_full if self._multiclass else
                       (y_full == self.classes_[1]).astype(np.float64))
            self.model_ = _refit_on_full(
                self, self.model_, X_full, y_refit, sw_full, cat_features, kw,
                replay=self.refit_full == "replay")
        elif (_bag_refit_rows(self) is not None and self.model_.trees_
                and not self._multiclass):
            # Binary only. Multiclass has no replay path -- `_refit_on_full`
            # rebuilds a vector-leaf model from scratch there -- so a member
            # refit would cost a whole extra fit per member instead of a cheap
            # structure replay. That is a different trade, and this evidence does
            # not cover it, so multiclass bags are unchanged.
            bx, byy, bsw, bfrac = _bag_refit_rows(self)
            y_refit = (byy == self.classes_[1]).astype(np.float64)
            self.model_ = _refit_on_full(
                self, self.model_, bx, y_refit, bsw, cat_features, kw,
                replay=True, train_frac=bfrac)

        return self

    def predict_proba(self, X):
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        if self.estimators_ is not None:
            # Soft-vote: average the members' calibrated probabilities, aligning
            # each member's class columns to the global class set. A member whose
            # sample missed a class simply contributes 0 to that column. One
            # shared conversion and factorization cache for the whole bag, and
            # one thread-limit switch (members' re-entries are no-ops).
            with _thread_limit(self.thread_count):
                Xc, ctx = _bag_predict_context(self, X)
                probas = [m._proba_impl(Xc, ctx) for m in self.estimators_]

            acc = np.zeros((probas[0].shape[0], self.n_classes_))
            for m, p in zip(self.estimators_, probas):
                cols = np.searchsorted(self.classes_, m.classes_)
                acc[:, cols] += p
            return acc / len(self.estimators_)

        return self._proba_impl(X)

    def _proba_impl(self, X, cat_ctx=None):
        """Calibrated probabilities of this single (non-bagged) model, without
        input validation -- the public entry points handle that."""
        raw = self.model_.predict_raw(X, cat_ctx) / self.temperature_
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
        """Per-round validation loss recorded during ``fit`` -- binary or softmax
        log loss -- as a list as long as the number of rounds run.

        Empty when no ``eval_set`` or early-stopping split was available; a list
        of the members' histories for a bagged model (``n_ensembles > 1``).
        """
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
        included exactly. Averaged across the bag when ``n_ensembles > 1``, an
        additive surrogate for the soft-voted probability. Multiclass is not
        supported yet. ``X_background`` overrides the reference distribution.
        """
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        if X_background is not None:
            bg = _check_predict_input(self, X_background)
            X_background = X_background if bg is None else bg

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