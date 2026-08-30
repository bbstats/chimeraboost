"""The multi-quantile estimator (benchmarks/QUANTILE_PLAN.md).

Kept out of ``sklearn_api`` because it shares that module's input validation
but almost none of its fit machinery: no loss family, no linear leaves, no
cross features, no bagging, no full-data refit. It imports the validation
helpers and keeps the same flat, module-function style.
"""

import numpy as np
from sklearn.base import BaseEstimator

from . import quantile_metrics
from .booster import MultiQuantileBoosting
from .preprocessing import as_model_array
from .sklearn_api import (_auto_cat_combinations, _check_eval_set,
                          _check_feature_names_match, _check_predict_input,
                          _format_shap_importances, _make_eval_split,
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

    A leaf estimating the 5% quantile from fewer than 20 rows has under one
    expected observation in that tail, which is noise. Scaling with the grid
    keeps that honest for a user who asks for 0.01. On the default grid it
    lands on 20, which matches LightGBM's ``min_data_in_leaf`` -- convenient
    when comparing the two.

    Notes
    -----
    ``1.0 - taus[-1]`` is not exact in binary. For a symmetric grid such as
    (0.1, 0.5, 0.9) it lands a couple of ulps below 0.1, so the reciprocal
    creeps just past 10 and ``ceil`` rounds the floor up to 11. Hence the snap
    below. Genuine fractions -- the 3.33 of a (0.3, 0.7) grid, say -- are far
    outside the tolerance and still round up, as intended.
    """
    edge = min(float(taus[0]), 1.0 - float(taus[-1]))
    inv = 1.0 / max(edge, 1e-9)

    # Snap to a whole number when only float error separates us from one.
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

    For each symmetric pair the conformity score is how far out the point sat,
    in units of the predicted half-interval,

        E_i = max((c_i - y_i) / (c_i - q_lo,i), (y_i - c_i) / (q_hi,i - c_i))

    with c the predicted median. The calibrated factor is that score's
    ``ceil((n+1)(1-alpha))``-th order statistic -- the standard conformal rank
    (Romano, Patterson & Candes 2019), which gives distribution-free marginal
    coverage on exchangeable data. Prediction then returns ``c + s * (q - c)``.

    Why scale rather than widen additively. These scores are already
    rearranged, so every deviation from the median is sign-correct and ordered
    outward, and multiplying by a non-negative factor cannot reorder any row.
    An additive correction can: one that SHRINKS an interval is a non-monotone
    offset vector and can reorder the grid on its own. A factor also carries
    the right degree of freedom, since a shrunk fit can be over- or
    under-dispersed and a factor moves both ways.

    Two constraints are enforced:

    * An outer interval's factor is at least that of the ones nested inside
      it, which is what keeps the rescaled grid ordered.
    * The rank must exist: ``ceil((n+1)(1-alpha)) <= n`` needs
      ``n >= (1-alpha)/alpha``. Below that the interval is vacuous, and this
      raises rather than quietly returning an uncalibrated model.

    Only symmetric pairs (t and 1-t both on the grid) can be calibrated
    directly; a grid with none raises. Unpaired levels on asymmetric grids get
    a factor interpolated by distance from the median across the paired
    factors -- uncertified, but it keeps the factor profile monotone outward,
    which preserves the non-crossing guarantee.
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

    if not pairs:
        raise ValueError(
            "conformalize=True needs at least one symmetric pair of quantile "
            "levels (t and 1-t both on the grid) to calibrate against; got "
            f"{list(np.round(taus, 6))}. Add symmetric levels or set "
            "conformalize=False.")

    # Outer factor >= inner factor. Deviations from the median already grow
    # outward, so a factor profile that never rises toward the centre keeps
    # their product monotone. Calibration tends to land here anyway: an
    # over-dispersed model needs the most shrinking near the median.
    for i in range(len(pairs) - 2, -1, -1):
        pairs[i][2] = max(pairs[i][2], pairs[i + 1][2])
    for k, j, sp in pairs:
        s[k] = s[j] = sp

    # Unpaired levels (custom asymmetric grids) are interpolated as the
    # docstring describes. Leaving them at 1.0 would break non-crossing: a
    # shrunk outer pair jumps across an unshrunk inner level.
    paired = {k for k, _, _ in pairs} | {j for _, j, _ in pairs}
    if len(paired) < K:
        dp = np.array([0.5 - taus[k] for k, _, _ in pairs])[::-1]
        sp = np.array([p[2] for p in pairs])[::-1]
        for k in range(K):
            if k not in paired:
                s[k] = float(np.interp(abs(taus[k] - 0.5), dp, sp))

    return s


class ChimeraBoostQuantileRegressor(BaseEstimator):
    """Gradient boosting for a whole predictive distribution at once.

    One booster, one tree structure per round, and a K-vector in every leaf,
    one entry per level in ``quantiles``. Against one quantile regressor per
    level that is roughly K times less split-search work, and the predictions
    cannot cross: the 30% quantile is never returned above the 70%.

    Ordering is enforced per row by monotone rearrangement of the delivered
    scores. It is exact for every row and holds at every intermediate stage of
    ``staged_predict``. Rearrangement cannot cost accuracy -- sorting a
    crossing quantile curve never increases pinball loss at any level
    (Chernozhukov, Fernandez-Val & Galichon 2010).

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
        How split gain is scored across the quantile levels. Keep the default
        unless you are experimenting. ``"rotate"`` alternates a location and a
        spread contrast, two location rounds per spread round, and measured
        best. ``"sum"`` adds the levels up, which makes it blind to a change in
        spread. ``"gram"`` picks the strongest contrast each round and measured
        no better than rotating.
    exact_splits : bool
        Score splits exactly across every level instead of on a projection.
        More faithful, but the fit gets slower and more memory-hungry as the
        grid grows -- a reference setting, not one for routine use.
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

    def _carve_calibration_fold(self, X, y, sample_weight, groups):
        """Step 1: the conformal calibration fold FIRST, so it sees neither
        the fit nor the stopping decision. Carved before anything else
        touches y. Returns ``(cal, X, y, sample_weight, groups)``."""
        if not self.conformalize:
            return None, X, y, sample_weight, groups
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
        return cal, X, y, sample_weight, groups

    def _carve_es_split(self, X, y, sample_weight, eval_set, groups):
        """Step 2: the early-stopping split out of whatever remains.
        Returns ``(X, y, sample_weight, eval_set, es_rounds)``."""
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
        return X, y, sample_weight, eval_set, es_rounds

    def _make_mq_booster(self, taus, es_rounds, cat_features, n_rows):
        """Resolve the auto defaults and construct the booster."""
        depth = 4 if self.depth is None else self.depth
        mcw = (_auto_min_child_weight(taus) if self.min_child_weight is None
               else self.min_child_weight)

        cat_combos = self.cat_combinations
        if cat_combos is None:
            cat_combos = _auto_cat_combinations(
                cat_features, self.n_features_in_, n_rows)

        return MultiQuantileBoosting(
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
            quantize_gradients=self.quantize_gradients,
            # Pinned to the historical flat rate. The size fade became the
            # booster default in 0.30.0 on RMSE and Brier evidence
            # (benchmarks/SMALLDATA_PLAN.md), but was never measured against
            # pinball loss; inheriting it here would ship an unmeasured default
            # change to the quantile path. Measure before flipping.
            adaptive_learning_rate=False)

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
            _check_feature_names_match(self, eval_set[0])

        taus = _resolve_quantiles(self.quantiles)
        self.quantiles_ = taus
        self._median_idx_ = _median_index(taus)
        self.conformal_scale_ = np.ones(taus.shape[0])

        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        cal, X, y, sample_weight, groups = self._carve_calibration_fold(
            X, y, sample_weight, groups)
        X, y, sample_weight, eval_set, es_rounds = self._carve_es_split(
            X, y, sample_weight, eval_set, groups)

        self.model_ = self._make_mq_booster(taus, es_rounds, cat_features,
                                            len(y))
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

    def _interval_levels(self, alpha, caller="interval"):
        """Grid positions of the two levels bounding the central 1-alpha
        interval. Shared by `predict` and `shap_values` so an explanation can
        never target a different pair of levels than the prediction did.

        Raises rather than interpolating: an interval the model was not fitted
        for is an error, not a guess.
        """
        if alpha is None:
            raise ValueError(
                f'{caller} needs alpha, the miscoverage: alpha=0.1 for a 90% '
                "interval.")
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
        return i, j

    def predict(self, X, kind="quantiles", alpha=None, thresholds=None,
                n_samples=None, random_state=None):
        """Predict the conditional distribution.

        ``kind="quantiles"`` (default) returns (n_samples, n_quantiles),
        column k being level ``quantiles_[k]``, non-decreasing along axis 1.

        ``kind="interval"`` returns (n_samples, 2): the central ``1 - alpha``
        interval, read off the ``alpha/2`` and ``1 - alpha/2`` levels, which
        must both be on the grid. No interpolation -- an interval the model was
        not fitted for is an error, not a guess.

        ``kind="mean"`` returns (n_samples,), the integral of the quantile
        function over tau: trapezoid across the grid plus flat extension of the
        edge levels out to 0 and 1. The flat extension is the honest reading of
        a finite grid, assuming nothing about tails the model never estimated.

        ``kind="median"`` returns (n_samples,), the predicted median -- the
        0.5 level when the grid carries it, interpolated between its
        neighbours when it does not. This is the centre conformalization
        rescales about.

        ``kind="cdf"`` returns (n_samples, len(thresholds)): ``P(y <= t)`` for
        each ``t`` in ``thresholds``, by inverting the grid. Clamped to the
        outermost fitted levels rather than to 0 and 1, for the same reason
        ``"mean"`` extends flat.

        ``kind="sample"`` returns (n_samples_rows, n_samples): inverse-
        transform draws from the predicted distribution, for feeding a
        downstream simulation. ``random_state`` seeds them; draws stay inside
        the fitted level range.

        Unlike ``"interval"``, ``"cdf"`` and ``"sample"`` interpolate between
        levels. That is a different question -- reading a fitted curve at a
        point, rather than claiming a level was fitted when it was not.
        """
        Xv = _check_predict_input(self, X)
        Q = self._conformalize(
            self.model_.predict_raw(X if Xv is None else Xv))

        if kind == "quantiles":
            return Q

        if kind == "interval":
            i, j = self._interval_levels(alpha)
            return np.column_stack([Q[:, i], Q[:, j]])

        if kind == "mean":
            return self._mean_from_quantiles(Q)

        if kind == "median":
            mi, mw = self._median_idx_
            return _centre(Q, mi, mw)

        if kind == "cdf":
            return self._cdf_from_quantiles(Q, thresholds)

        if kind == "sample":
            return self._sample_from_quantiles(Q, n_samples, random_state)

        raise ValueError(
            'kind must be "quantiles", "interval", "mean", "median", "cdf" '
            f'or "sample"; got {kind!r}.')

    def _cdf_from_quantiles(self, Q, thresholds):
        """Invert the grid: P(y <= t) read off the predicted quantile
        function, one column per threshold.

        Linear between grid levels, and clamped outside it -- below the lowest
        fitted level the honest answer is "at most tau_0", not zero, but the
        grid cannot resolve further, so the edge levels are returned flat.
        Same convention as `_mean_from_quantiles`, which assumes nothing about
        tails the model never estimated.
        """
        if thresholds is None:
            raise ValueError(
                'predict(kind="cdf") needs `thresholds`: the values t at '
                "which to evaluate P(y <= t).")
        t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
        if t.ndim != 1:
            raise ValueError(
                f"thresholds must be a scalar or 1-D; got shape {t.shape}.")

        taus = self.quantiles_
        out = np.empty((Q.shape[0], t.shape[0]))
        for i in range(Q.shape[0]):
            out[i] = np.interp(t, Q[i], taus, left=taus[0], right=taus[-1])
        return out

    def _sample_from_quantiles(self, Q, n_samples, random_state):
        """Inverse-transform sampling from the predicted grid: draw u uniform
        and read the quantile function at u.

        Returns (n_rows, n_samples). Draws land strictly inside the fitted
        range -- the grid says nothing about the tails beyond it, so sampling
        them would be inventing data rather than reporting the model.
        """
        if n_samples is None:
            raise ValueError(
                'predict(kind="sample") needs `n_samples`, how many draws to '
                "take per row.")
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1; got {n_samples}.")

        taus = self.quantiles_
        rng = np.random.default_rng(random_state)
        u = rng.uniform(taus[0], taus[-1], size=(Q.shape[0], n_samples))
        out = np.empty_like(u)
        for i in range(Q.shape[0]):
            out[i] = np.interp(u[i], taus, Q[i])
        return out

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
        conformal rescaling is a post-fit transform, so it is applied at every
        stage and the last one equals ``predict``."""
        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv

        for staged in self.model_.staged_predict_raw(X):
            yield self._conformalize(staged)

    def score(self, X, y, sample_weight=None):
        """Negative CRPS (mean pinball loss over the grid). Higher is better,
        per the sklearn convention."""
        return -quantile_metrics.crps(y, self.predict(X), self.quantiles_,
                                      sample_weight)

    def report(self, X, y, sample_weight=None, baseline=None):
        """`quantile_metrics.quantile_report` on this model's predictions:
        CRPS and its skill score, per-level pinball, coverage plus width plus
        interval score for every symmetric interval, the PIT histogram, and
        the crossing rate.

        ``baseline`` sets what the skill score is measured against -- pass the
        training targets to score against the marginal distribution the model
        actually had, rather than the hindsight marginal of ``y``.
        """
        return quantile_metrics.quantile_report(y, self.predict(X),
                                                self.quantiles_, sample_weight,
                                                baseline)

    def _delivered_shap(self, phi_raw, base_raw, raw):
        """Move raw-level attributions onto the levels `predict` delivers.

        Two transforms, in the order `predict` applies them.

        **The sort** is a per-row permutation, so gathering `phi` and the
        baseline by it relabels levels without touching their values and
        efficiency is preserved exactly. The baseline becomes per-row because
        different rows reorder differently.

        These are therefore exact attributions of "whichever level landed
        here", not of the sorted map as a function. The latter is not
        available at any price: the sort acts on the SUMMED forest score, so it
        cannot be decomposed per tree, and a coalition game over every input
        feature instead of the <= D one tree touches is the exact blow-up
        obliviousness buys us out of. Recorded because it looks like an
        oversight otherwise.

        **The conformal rescale** is linear in the level vector with no
        constant term, so it pushes through onto `phi` and the baseline alike.
        Its matrix comes from pushing the identity through `_conformalize`
        itself rather than being rewritten here, so it cannot drift from what
        `predict` does.
        """
        # Stable, so tied channels break the same way on every numpy version.
        # An unstable sort would pick a different -- still self-consistent --
        # baseline from run to run.
        order = np.argsort(raw, axis=1, kind="stable")
        phi = np.take_along_axis(phi_raw, order[:, None, :], axis=2)
        base = base_raw[order]

        if not np.all(self.conformal_scale_ == 1.0):
            L = self._conformalize(np.eye(self.quantiles_.shape[0]))
            phi = phi @ L
            base = base @ L
        return phi, base

    def shap_values(self, X, X_background=None, kind="quantiles", alpha=None,
                    quantile=None, space="delivered"):
        """Exact interventional TreeSHAP for a predicted quantile grid.

        Explains what ``predict`` returned. Contributions plus
        ``expected_value_`` (set by this call) reconstruct it, level by level.

        ``kind`` selects the explained quantity:

        * ``"quantiles"`` -- ``(n_samples, n_features, n_quantiles)``, or
          ``(n_samples, n_features)`` when ``quantile`` names one fitted level.
        * ``"mean"`` -- the tau-integrated point prediction.
        * ``"width"`` -- the width of the central ``1 - alpha`` interval: which
          features make this row's prediction more uncertain, as opposed to
          higher or lower. Shapley values are linear in the value function, so
          the difference of two levels' attributions is exactly the attribution
          of their difference.

        ``space`` is for one specific job and most callers can ignore it.
        Predictions are rearranged on delivery, which relabels a row's levels,
        so the default ``"delivered"`` measures each row against its own
        reordering of the background and ``expected_value_`` is
        ``(n_samples, n_quantiles)``. Aggregating those across rows mixes rows
        that were reordered differently. ``space="raw"`` explains the
        pre-rearrangement levels instead, against one shared
        ``(n_quantiles,)`` baseline, which is what makes a cross-row average
        meaningful -- `shap_importances` uses it for exactly that reason. The
        two agree on any row whose levels were already in order.
        ``"mean"`` and ``"width"`` are order-dependent by construction and
        always read the delivered grid.
        """
        if space not in ("raw", "delivered"):
            raise ValueError(
                f'space must be "raw" or "delivered"; got {space!r}.')
        if kind not in ("quantiles", "mean", "width"):
            raise ValueError(
                f'kind must be "quantiles", "mean" or "width"; got {kind!r}.')

        Xv = _check_predict_input(self, X)
        X = X if Xv is None else Xv
        if X_background is not None:
            # Consumed positionally, so a reordered DataFrame would silently
            # skew every baseline.
            bg = _check_predict_input(self, X_background)
            X_background = X_background if bg is None else bg

        phi, base = self.model_.shap_values(X, background=X_background)

        if kind == "quantiles" and space == "raw":
            if quantile is not None:
                k = self._level_index(quantile)
                phi, base = phi[:, :, k], base[k]
            self.expected_value_ = base
            return phi

        phi, base = self._delivered_shap(phi, base,
                                         self.model_._raw_scores(X))

        if kind == "mean":
            w = self._mean_from_quantiles(np.eye(self.quantiles_.shape[0]))
            self.expected_value_ = base @ w
            return phi @ w

        if kind == "width":
            i, j = self._interval_levels(
                alpha, caller='shap_values(kind="width")')
            self.expected_value_ = base[:, j] - base[:, i]
            return phi[:, :, j] - phi[:, :, i]

        if quantile is not None:
            k = self._level_index(quantile)
            phi, base = phi[:, :, k], base[:, k]
        self.expected_value_ = base
        return phi

    def _level_index(self, quantile):
        """Grid position of one level, refusing anything not fitted -- the same
        rule `_interval_levels` applies to a pair."""
        taus = self.quantiles_
        k = int(np.argmin(np.abs(taus - quantile)))
        if abs(taus[k] - quantile) > 1e-9:
            raise ValueError(
                f"quantile={quantile} is not on the fitted grid "
                f"{np.array2string(taus, precision=4)}. Refit with that level "
                "in `quantiles`.")
        return k

    def shap_importances(self, X, feature_names=None, n_features=None,
                         prettified=False, quantile=None):
        """Global SHAP importance: ``mean(abs(shap_values(X)))`` per feature.

        Averaged over the whole grid by default, or over one level when
        ``quantile`` names it. Uses ``space="raw"`` so that every row is
        measured on the same footing -- see `shap_values` -- which is what a
        cross-row average needs.

        Returns a structured ``(feature, importance)`` array sorted descending,
        or a ``{feature: importance}`` dict when ``prettified=True``.
        """
        phi = self.shap_values(X, quantile=quantile, space="raw")
        imp = np.abs(phi).mean(axis=0)
        if imp.ndim == 2:                      # (n_features, n_quantiles)
            imp = imp.mean(axis=1)
        return _format_shap_importances(
            self, imp, feature_names=feature_names, n_features=n_features,
            prettified=prettified)

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
