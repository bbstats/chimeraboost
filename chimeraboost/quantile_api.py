"""The multi-quantile estimator (benchmarks/QUANTILE_PLAN.md).

Kept out of ``sklearn_api`` because it shares that module's input-validation
helpers but almost none of its fit machinery: no loss family, no linear
leaves, no cross features, no bagging, no full-data refit. It reuses the
validation helpers by importing them, in the same flat, module-function style
`sklearn_api` already uses.
"""

import numpy as np
from sklearn.base import BaseEstimator

from . import quantile_metrics
from .booster import MultiQuantileBoosting
from .preprocessing import as_model_array
from .sklearn_api import (_auto_cat_combinations, _check_eval_set,
                          _check_predict_input, _make_eval_split,
                          _resolve_cat_features, _resolve_cat_feature_names,
                          _validate_fit_input, _validate_hyperparams)


# 0.05 ... 0.95 in steps of 0.05: nineteen levels, symmetric about the median,
# so the 90%, 80%, ... central intervals are adjacent column pairs.
DEFAULT_QUANTILES = np.round(np.arange(0.05, 0.9501, 0.05), 10)


def _resolve_quantiles(quantiles):
    """Validate and normalize the tau grid."""
    q = DEFAULT_QUANTILES if quantiles is None else np.asarray(
        quantiles, dtype=np.float64).ravel()
    if q.size == 0:
        raise ValueError("quantiles must contain at least one level.")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"quantiles must all be finite; got {q!r}.")
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError(
            f"quantiles must lie strictly inside (0, 1); got {q!r}.")
    if q.size > 1 and np.any(np.diff(q) <= 0.0):
        raise ValueError(
            "quantiles must be strictly ascending and unique; got "
            f"{q!r}. The column order of predict() is the order given here, "
            "so sort them first.")
    return np.ascontiguousarray(q)


def _auto_min_child_weight(taus):
    """Leaf-size floor implied by the most extreme level on the grid.

    A leaf estimating the 5% quantile from fewer than 20 rows is estimating it
    from under one expected observation in that tail, which is noise. Scaling
    with the grid keeps that honest for a user who asks for 0.01. It also
    lands on 20 for the default grid, which happens to match LightGBM's
    ``min_data_in_leaf`` -- convenient when comparing the two.
    """
    edge = min(float(taus[0]), 1.0 - float(taus[-1]))
    inv = 1.0 / max(edge, 1e-9)
    # `1.0 - taus[-1]` is not exact in binary: for a symmetric grid such as
    # (0.1, 0.5, 0.9) it lands a couple of ulps BELOW 0.1, so the reciprocal
    # creeps just past 10 and `ceil` rounds the floor up to 11. Snap to a whole
    # number when we are within rounding noise of one -- genuine fractions like
    # the 3.33 of a (0.3, 0.7) grid are far outside the tolerance and still
    # round up, which is the intended behaviour.
    nearest = round(inv)
    if abs(inv - nearest) <= 1e-9 * max(1.0, abs(inv)):
        inv = float(nearest)
    return float(np.ceil(inv))


def _median_index(taus):
    """``(i, w)`` such that ``Q[:, i] + w * (Q[:, i+1] - Q[:, i])`` is the
    predicted median, for a grid that may or may not contain 0.5 exactly."""
    K = taus.shape[0]
    if K == 1:
        return 0, 0.0
    j = int(np.searchsorted(taus, 0.5))
    if j < K and abs(taus[j] - 0.5) < 1e-12:
        return j, 0.0
    if j == 0:
        return 0, 0.0
    if j >= K:
        return K - 1, 0.0
    lo, hi = taus[j - 1], taus[j]
    return j - 1, float((0.5 - lo) / (hi - lo))


def _centre(Q, mi, mw):
    """Per-row predicted median from `_median_index`."""
    if mw == 0.0:
        return Q[:, mi]
    return Q[:, mi] + mw * (Q[:, mi + 1] - Q[:, mi])


def _cqr_scales(Q, y, taus, mi, mw):
    """Conformalized quantile regression as a per-level SCALE about the median.

    For each symmetric pair the conformity score is how far out the point sat
    in units of the predicted half-interval,

        E_i = max((c_i - y_i) / (c_i - q_lo,i), (y_i - c_i) / (q_hi,i - c_i))

    with c the predicted median, and the calibrated factor is that score's
    ``ceil((n+1)(1-alpha))``-th order statistic -- the standard conformal rank
    (Romano, Patterson & Candes 2019), which gives distribution-free marginal
    coverage on exchangeable data. Prediction then returns
    ``c + s * (q - c)``.

    Why scaling rather than the usual additive widening. An additive
    correction that SHRINKS an interval is a non-monotone offset vector, and
    the only way to apply one safely is out of the fit's narrowing budget --
    which the fit has already spent. Scaling about the median has no such
    problem: multiplying a monotone, sign-correct deviation by a non-negative
    factor cannot reorder anything, for any row, spent budget or not. Since a
    shrunk-fit quantile model is systematically OVER-dispersed -- every round's
    step is shrunk by the learning rate, so the grid never fully contracts --
    shrinking is the correction that is actually needed, and the additive form
    cannot deliver it.

    Two constraints are enforced:

    * Outer intervals get a factor at least as large as the ones nested inside
      them, which is what keeps the rescaled grid ordered.
    * The rank must exist: ``ceil((n+1)(1-alpha)) <= n`` needs
      ``n >= (1-alpha)/alpha``. Below that the interval is vacuous, and this
      raises rather than quietly returning an uncalibrated model.
    """
    K = taus.shape[0]
    s = np.ones(K)
    n = y.shape[0]
    c = _centre(Q, mi, mw)
    # Relative floor for the half-widths: a row whose predicted interval has
    # collapsed to a point would otherwise divide by zero.
    eps = 1e-9 * max(float(np.mean(Q[:, -1] - Q[:, 0])), 1e-12)
    pairs = []
    for k in range(K // 2):
        j = K - 1 - k
        if abs((taus[k] + taus[j]) - 1.0) > 1e-9:
            continue                       # not a symmetric pair
        alpha = 1.0 - (taus[j] - taus[k])
        need = int(np.ceil((1.0 - alpha) / max(alpha, 1e-12)))
        if n < need:
            raise ValueError(
                f"conformalize=True needs at least {need} calibration rows to "
                f"certify the {taus[k]:g}-{taus[j]:g} interval (nominal "
                f"coverage {1 - alpha:.0%}), but the calibration fold holds "
                f"only {n}. Raise calibration_fraction, supply more data, or "
                "drop the extreme levels from `quantiles`.")
        E = np.maximum((c - y) / np.maximum(c - Q[:, k], eps),
                       (y - c) / np.maximum(Q[:, j] - c, eps))
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        pairs.append([k, j, max(0.0, float(np.sort(E)[min(rank, n) - 1]))])
    # Outer factor >= inner factor. The deviations from the median already
    # grow outward, so a non-increasing factor toward the centre keeps their
    # product monotone -- and calibration naturally lands this way, since an
    # over-dispersed model needs the most shrinking near the median.
    for i in range(len(pairs) - 2, -1, -1):
        pairs[i][2] = max(pairs[i][2], pairs[i + 1][2])
    for k, j, sp in pairs:
        s[k] = s[j] = sp
    return s


class ChimeraBoostQuantileRegressor(BaseEstimator):
    """Gradient boosting for a whole predictive distribution at once.

    One booster, one tree structure per round, and a K-vector in every leaf,
    holding one entry per level in ``quantiles``. Against fitting one quantile
    regressor per level this is roughly K times less split-search work, and
    the predictions cannot cross: the 30% quantile is never returned above the
    70%. The guarantee is structural rather than repaired afterwards, so it
    holds at every intermediate stage of ``staged_predict`` too.

    Deliberately not a ``RegressorMixin``: ``predict`` returns a matrix, so the
    inherited ``score`` (which assumes one number per row) would be wrong.
    ``score`` here is negative CRPS, so higher is better, as sklearn requires.

    Read more in the [User Guide](https://bbstats.github.io/chimeraboost/quantiles/).

    Parameters
    ----------
    quantiles : array-like or None
        Ascending, unique levels strictly inside (0, 1). Default
        0.05, 0.10, ... 0.95. Column k of ``predict`` is level k.
    split_projection : {"rotate", "sum", "gram"}
        How the K gradient columns collapse into the single vector the tree
        grower accepts. ``"rotate"`` cycles a location and a spread contrast,
        two rounds of the former per round of the latter, and is the default
        because it measured best; ``"sum"`` is the literal channel sum, which
        is blind to spread entirely; ``"gram"`` picks the strongest contrast
        each round and measured no better than cycling. See
        `booster._fixed_contrasts`.
    exact_splits : bool
        Score splits on the exact summed-across-level gain instead of a
        projection. More faithful, and costs K histogram channels per feature
        instead of one -- a reference arm, not a working default.
    conformalize : bool
        Calibrate the intervals by conformalized quantile regression. Carves
        ``calibration_fraction`` of the rows off BEFORE the early-stopping
        split, so that fold influences neither the fit nor the stopping point.
        Raises if the fold is too small to certify the requested levels.
    calibration_fraction : float
        Share of training rows reserved for conformalization. Ignored unless
        ``conformalize=True``.

    Attributes
    ----------
    quantiles_ : ndarray of shape (n_quantiles,)
        The resolved grid.
    conformal_scale_ : ndarray of shape (n_quantiles,)
        Per-level conformal scale about the predicted median; all ones unless
        ``conformalize=True``.

    Notes
    -----
    Every other parameter carries its usual ChimeraBoost meaning. Two defaults
    are set for this head rather than inherited: ``depth`` is 4, because deep
    leaves overfit tail quantiles, and ``min_child_weight`` follows a floor
    implied by the most extreme level on the grid.
    """

    def __init__(self, quantiles=None, n_estimators=2000, learning_rate=None,
                 depth=None, l2_leaf_reg=1.0, max_bins=128, subsample=1.0,
                 colsample=None, cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None, min_child_weight=None,
                 thread_count=None, random_state=None, verbose=False,
                 cat_features=None, cat_combinations=None,
                 quantize_gradients=True, early_stopping=True,
                 validation_fraction=0.2, split_projection="rotate",
                 exact_splits=False, conformalize=False,
                 calibration_fraction=0.2):
        self.quantiles = quantiles
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
        self.min_child_weight = min_child_weight
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.cat_features = cat_features
        self.cat_combinations = cat_combinations
        self.quantize_gradients = quantize_gradients
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.split_projection = split_projection
        self.exact_splits = exact_splits
        self.conformalize = conformalize
        self.calibration_fraction = calibration_fraction

    def __sklearn_is_fitted__(self):
        return hasattr(self, "model_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN routed to a missing bin
        tags.input_tags.sparse = False
        return tags

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, callbacks=None):
        """Fit the model. Arguments carry the same meaning as
        `ChimeraBoostRegressor.fit`."""
        cat_features = _resolve_cat_features(self, cat_features)
        cat_features = _resolve_cat_feature_names(cat_features, X)
        _validate_hyperparams(self)
        if self.split_projection not in ("rotate", "sum", "gram"):
            raise ValueError(
                'split_projection must be "rotate", "sum" or "gram"; got '
                f"{self.split_projection!r}.")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1); got "
                             f"{self.calibration_fraction!r}.")
        y = _validate_fit_input(self, X, y, cat_features, sample_weight,
                                classification=False)
        if eval_set is not None:
            _check_eval_set(eval_set, self.n_features_in_)
        taus = _resolve_quantiles(self.quantiles)
        self.quantiles_ = taus
        self._median_idx_ = _median_index(taus)
        self.conformal_scale_ = np.ones(taus.shape[0])

        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # 1. Calibration fold FIRST, so it sees neither the fit nor the
        #    stopping decision. Carved before anything else touches y.
        cal = None
        if self.conformalize:
            split = _make_eval_split(X, y, self.calibration_fraction,
                                     self.random_state, groups=groups)
            if split is None:
                raise ValueError(
                    "conformalize=True could not carve a calibration fold from "
                    f"{len(y)} rows at calibration_fraction="
                    f"{self.calibration_fraction}. Supply more rows or lower "
                    "the fraction.")
            keep, cal_idx = split
            cal = (X[cal_idx], y[cal_idx])
            X, y = X[keep], y[keep]
            if sample_weight is not None:
                sample_weight = sample_weight[keep]
            if groups is not None:
                groups = np.asarray(groups)[keep]

        # 2. Early-stopping split out of whatever remains.
        es_active = bool(self.early_stopping)
        if es_active and eval_set is None:
            split = _make_eval_split(X, y, self.validation_fraction,
                                     self.random_state, groups=groups)
            if split is None:
                es_active = False           # too small to hold anything out
            else:
                tr, va = split
                sw_val = None if sample_weight is None else sample_weight[va]
                eval_set = (X[va], y[va], sw_val)
                X, y = X[tr], y[tr]
                if sample_weight is not None:
                    sample_weight = sample_weight[tr]
        es_rounds = self.early_stopping_rounds
        if not es_active:
            es_rounds = None
        elif eval_set is not None and es_rounds is None:
            es_rounds = 50

        depth = 4 if self.depth is None else self.depth
        mcw = (_auto_min_child_weight(taus) if self.min_child_weight is None
               else self.min_child_weight)
        cat_combos = self.cat_combinations
        if cat_combos is None:
            cat_combos = _auto_cat_combinations(
                cat_features, self.n_features_in_, len(y))

        self.model_ = MultiQuantileBoosting(
            quantiles=taus, split_projection=self.split_projection,
            exact_splits=self.exact_splits,
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            depth=depth, l2_leaf_reg=self.l2_leaf_reg, max_bins=self.max_bins,
            subsample=self.subsample,
            colsample=1.0 if self.colsample is None else self.colsample,
            cat_smoothing=self.cat_smoothing,
            cat_n_permutations=self.cat_n_permutations,
            early_stopping_rounds=es_rounds, min_child_weight=mcw,
            thread_count=self.thread_count, random_state=self.random_state,
            verbose=self.verbose, cat_combinations=cat_combos,
            quantize_gradients=self.quantize_gradients)
        self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                        sample_weight=sample_weight, callbacks=callbacks)

        # 3. Conformalize on the pristine fold.
        if cal is not None:
            mi, mw = self._median_idx_
            self.conformal_scale_ = _cqr_scales(
                self.model_.predict_raw(cal[0]), cal[1], taus, mi, mw)
        return self

    def _conformalize(self, Q):
        """Apply the calibrated scale about each row's predicted median. A
        no-op (all factors 1) unless ``conformalize=True``."""
        if np.all(self.conformal_scale_ == 1.0):
            return Q
        mi, mw = self._median_idx_
        c = _centre(Q, mi, mw)[:, None]
        return c + self.conformal_scale_[None, :] * (Q - c)

    def predict(self, X, kind="quantiles", alpha=None):
        """Predict the conditional distribution.

        ``kind="quantiles"`` (default) returns (n_samples, n_quantiles),
        column k being level ``quantiles_[k]``, non-decreasing along axis 1.

        ``kind="interval"`` returns (n_samples, 2): the central ``1 - alpha``
        interval, read off the ``alpha/2`` and ``1 - alpha/2`` levels, which
        must both be on the grid. No interpolation -- an interval the model
        was not fitted for is an error, not a guess.

        ``kind="mean"`` returns (n_samples,), the integral of the quantile
        function over tau: trapezoid across the grid plus flat extension of
        the edge levels out to 0 and 1. The flat extension is the honest
        reading of a finite grid, assuming nothing about tails the model never
        estimated.
        """
        Xv = _check_predict_input(self, X)
        Q = self._conformalize(
            self.model_.predict_raw(X if Xv is None else Xv))
        if kind == "quantiles":
            return Q
        if kind == "interval":
            if alpha is None:
                raise ValueError(
                    'predict(kind="interval") needs alpha, the miscoverage: '
                    "alpha=0.1 for a 90% interval.")
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
            lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
            taus = self.quantiles_
            i = int(np.argmin(np.abs(taus - lo)))
            j = int(np.argmin(np.abs(taus - hi)))
            if abs(taus[i] - lo) > 1e-9 or abs(taus[j] - hi) > 1e-9:
                raise ValueError(
                    f"alpha={alpha} needs levels {lo:g} and {hi:g}, which are "
                    "not on the fitted grid "
                    f"{np.array2string(taus, precision=4)}. Refit with those "
                    "levels in `quantiles`.")
            return np.column_stack([Q[:, i], Q[:, j]])
        if kind == "mean":
            return self._mean_from_quantiles(Q)
        raise ValueError(
            f'kind must be "quantiles", "interval" or "mean"; got {kind!r}.')

    def _mean_from_quantiles(self, Q):
        """Integral of the quantile function over [0, 1]: trapezoid across the
        grid, plus a rectangle for the flat extension below the first level and
        another above the last."""
        taus = self.quantiles_
        trapz = getattr(np, "trapezoid", None) or np.trapz
        inner = trapz(Q, x=taus, axis=1) if taus.shape[0] > 1 else 0.0
        return inner + taus[0] * Q[:, 0] + (1.0 - taus[-1]) * Q[:, -1]

    def staged_predict(self, X):
        """Yield the (n, K) quantile matrix after each successive tree. The
        conformal rescaling, being a post-fit transform, is applied at every
        stage so the last one equals ``predict``."""
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv
        for staged in self.model_.staged_predict_raw(X):
            yield self._conformalize(staged)

    def score(self, X, y, sample_weight=None):
        """Negative CRPS (mean pinball loss over the grid). Higher is better,
        per the sklearn convention."""
        return -quantile_metrics.crps(y, self.predict(X), self.quantiles_,
                                      sample_weight)

    def report(self, X, y, sample_weight=None):
        """`quantile_metrics.quantile_report` on this model's predictions:
        CRPS, per-level pinball, and coverage plus width for every symmetric
        interval."""
        return quantile_metrics.quantile_report(y, self.predict(X),
                                                self.quantiles_, sample_weight)

    @property
    def best_iteration_(self):
        return self.model_.best_iteration_

    @property
    def validation_history_(self):
        """Per-round validation CRPS from ``fit`` (empty without a validation
        set)."""
        return list(self.model_.valid_history_)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_
