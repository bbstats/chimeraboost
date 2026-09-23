"""Loss functions for ChimeraBoost.

Each loss provides:
  init(y)            -> scalar raw score to start every prediction from
  grad_hess(y, raw)  -> (gradient, hessian) of the loss wrt the raw score
  eval(y, raw)       -> scalar loss value (for early stopping / logging)

Raw scores are the additive model output *before* any link function.
For regression the raw score is the prediction itself; for binary
classification it is the log-odds, turned into a probability by a sigmoid.
"""

import numpy as np
from numba import njit, prange


def _weighted_quantile(values, weights, alpha):
    """Nearest-rank quantile at level *alpha*; unweighted when *weights* is None."""
    if weights is None:
        return float(np.quantile(values, alpha)) if values.size else 0.0
    if not values.size:
        return 0.0
    order = np.argsort(values)
    sv, sw = values[order], weights[order]
    cumw = np.cumsum(sw)
    idx = min(int(np.searchsorted(cumw, cumw[-1] * alpha)), len(sv) - 1)
    return float(sv[idx])


@njit(cache=True, parallel=True)
def _sigmoid(z):
    # Numerically stable logistic, parallel over rows. The sign branch keeps
    # exp() from overflowing: exp(-|z|) always lands in [0, 1].
    n = z.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        zi = z[i]
        if zi >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-zi))
        else:
            ez = np.exp(zi)
            out[i] = ez / (1.0 + ez)
    return out


@njit(cache=True, parallel=True)
def _logloss_grad_hess_kernel(raw, y):
    # Fused grad_hess for Logloss: the sigmoid row above plus `grad = p - y`
    # and `hess = max(p * (1 - p), 1e-6)` in one pass, so the output is
    # bit-identical to `_grad_hess_numpy` on the same machine. The sigmoid
    # loop is `_sigmoid`'s verbatim (same branch, same expressions); the grad
    # and hess lines are elementwise and in numpy's order, so they round
    # exactly as the numpy code does. The 1e-6 floor is a comparison because
    # that is `np.maximum(h, 1e-6)` on non-NaN input, and `h` lies in
    # [0, 0.25] for finite raw.
    n = raw.shape[0]
    grad = np.empty(n, dtype=np.float64)
    hess = np.empty(n, dtype=np.float64)
    for i in prange(n):
        zi = raw[i]
        if zi >= 0.0:
            p = 1.0 / (1.0 + np.exp(-zi))
        else:
            ez = np.exp(zi)
            p = ez / (1.0 + ez)
        grad[i] = p - y[i]
        h = p * (1.0 - p)
        hess[i] = h if h >= 1e-6 else 1e-6
    return grad, hess


@njit(cache=True, parallel=True)
def _logloss_ce_kernel(raw, y):
    # Per-row cross-entropy for Logloss.eval: sigmoid, clip and `-log` in one
    # pass, elementwise and in numpy's order, so the vector is bit-identical
    # to the `ce` line of `_eval_numpy` on the same machine. The mean stays in
    # numpy -- numpy's pairwise summation is not a sequential numba sum, and a
    # fused reduction would drift. The clip is two comparisons so a NaN passes
    # through exactly as `np.clip` passes it (a NaN fails both comparisons and
    # stays NaN); the upper bound is written as the literal expression
    # `1.0 - 1e-9`, the same expression numpy evaluates in the old body.
    # The 0/1 branches are exact: for y = 1 numpy computes
    # `-(1*log p + 0*log(1-p))`; `0*log(1-p)` is a signed zero, and adding a
    # signed zero to a finite float returns that float unchanged, so the
    # result is `-log p` to the bit; symmetrically for y = 0. The soft-label
    # branch is the general formula unchanged.
    n = raw.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        zi = raw[i]
        if zi >= 0.0:
            p = 1.0 / (1.0 + np.exp(-zi))
        else:
            ez = np.exp(zi)
            p = ez / (1.0 + ez)
        if p < 1e-9:
            p = 1e-9
        elif p > 1.0 - 1e-9:
            p = 1.0 - 1e-9
        yi = y[i]
        if yi == 1.0:
            out[i] = -np.log(p)
        elif yi == 0.0:
            out[i] = -np.log(1.0 - p)
        else:
            out[i] = -(yi * np.log(p) + (1.0 - yi) * np.log(1.0 - p))
    return out


def _scalar_pair_ok(raw, y):
    """True when both inputs are 1-D float64 C-contiguous arrays of one length."""
    return (isinstance(raw, np.ndarray) and isinstance(y, np.ndarray)
            and raw.dtype == np.float64 and y.dtype == np.float64
            and raw.ndim == 1 and y.ndim == 1
            and raw.flags.c_contiguous and y.flags.c_contiguous
            and raw.shape == y.shape)


class _UnitHessian:
    """Constant-hessian mixin: ``grad_hess`` returns a cached all-ones buffer
    instead of allocating ``np.ones_like`` every boosting round.

    The buffer is SHARED across rounds, so nothing on the fit path may write into
    a returned hessian. Today nothing does: the weighted paths build fresh arrays
    (``hess * w``), MVS returns fresh arrays, and the tree kernels only read it.
    Keep it that way -- in particular, never "optimize" ``hess = hess * w`` into
    ``np.multiply(hess, w, out=hess)``.

    The cache is dropped on pickle: ``loss_`` rides inside fitted boosters, and n
    floats of ones have no business in the payload.
    """

    def _unit_hess(self, like):
        h = getattr(self, "_hess_cache", None)
        if h is None or h.shape != like.shape:
            h = np.ones_like(like)
            self._hess_cache = h
        return h

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_hess_cache", None)
        return state


class RMSE(_UnitHessian):
    """Squared-error regression. grad = pred - y, hess = 1."""

    name = "RMSE"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        return float(np.average(y, weights=sample_weight))

    def grad_hess(self, y, raw):
        grad = raw - y
        return grad, self._unit_hess(raw)

    def eval(self, y, raw, sample_weight=None):
        return float(np.sqrt(np.average((raw - y) ** 2, weights=sample_weight)))

    def transform(self, raw):
        return raw


class Logloss:
    """Binary cross-entropy. raw = log-odds, p = sigmoid(raw).

    grad = p - y, hess = p * (1 - p), floored at 1e-6 so a saturated row cannot
    blow up a leaf denominator.
    """

    name = "Logloss"
    is_classification = True
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        p = np.clip(np.average(y, weights=sample_weight), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def _grad_hess_numpy(self, y, raw):
        p = _sigmoid(raw)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess

    def grad_hess(self, y, raw):
        if _scalar_pair_ok(raw, y):
            return _logloss_grad_hess_kernel(raw, y)
        return self._grad_hess_numpy(y, raw)

    def _eval_numpy(self, y, raw, sample_weight=None):
        p = np.clip(_sigmoid(raw), 1e-9, 1 - 1e-9)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.average(ce, weights=sample_weight))

    def eval(self, y, raw, sample_weight=None):
        if _scalar_pair_ok(raw, y):
            ce = _logloss_ce_kernel(raw, y)
            return float(np.average(ce, weights=sample_weight))
        return self._eval_numpy(y, raw, sample_weight)

    def transform(self, raw):
        return _sigmoid(raw)


class MAE(_UnitHessian):
    """Mean absolute error. grad = sign(pred - y), hess = 1.

    The sign gradient only picks the tree structure. Leaf values are the
    (weighted) median of the residuals, which is what minimizes absolute error.
    """

    name = "MAE"
    is_classification = False
    adjusts_leaves = True

    def leaf_value(self, residuals, weights=None):
        return _weighted_quantile(residuals, weights, 0.5)

    def init(self, y, sample_weight=None):
        return _weighted_quantile(y, sample_weight, 0.5)

    def grad_hess(self, y, raw):
        grad = np.sign(raw - y)
        return grad, self._unit_hess(raw)

    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.abs(raw - y), weights=sample_weight))

    def transform(self, raw):
        return raw


class Quantile(_UnitHessian):
    """Pinball loss for quantile regression at level `alpha` in (0, 1).

    grad = -alpha where y sits at or above the current estimate, 1 - alpha
    below; hess = 1. Leaf values are the weighted alpha-quantile of the
    residuals.
    """

    name = "Quantile"
    is_classification = False
    adjusts_leaves = True

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)

    def leaf_value(self, residuals, weights=None):
        return _weighted_quantile(residuals, weights, self.alpha)

    def init(self, y, sample_weight=None):
        return _weighted_quantile(y, sample_weight, self.alpha)

    def grad_hess(self, y, raw):
        a = self.alpha
        grad = np.where(y >= raw, -a, 1.0 - a)
        return grad, self._unit_hess(raw)

    def eval(self, y, raw, sample_weight=None):
        r = y - raw
        pinball = np.maximum(self.alpha * r, (self.alpha - 1.0) * r)
        return float(np.average(pinball, weights=sample_weight))

    def transform(self, raw):
        return raw


# Exponent cap for the log-link losses. exp(80) ~ 5.5e34 keeps every downstream
# product finite in float64, and no sane fit gets near it: raw = log(mu), so the
# cap sits far outside any real raw score.
_EXP_CLIP = 80.0


def _exp(z):
    return np.exp(np.clip(z, -_EXP_CLIP, _EXP_CLIP))


class Huber(_UnitHessian):
    """Huber regression: quadratic within `delta` of the target, linear beyond.

    grad = clip(pred - y, -delta, delta); hess = 1 in both regions, the standard
    GBDT treatment. `delta` is in y units and fixed, not quantile-adaptive, so
    scale it to the data.
    """

    name = "Huber"
    is_classification = False
    adjusts_leaves = False

    def __init__(self, delta=1.0):
        self.delta = float(delta)

    def init(self, y, sample_weight=None):
        return _weighted_quantile(y, sample_weight, 0.5)

    def grad_hess(self, y, raw):
        r = raw - y
        grad = np.clip(r, -self.delta, self.delta)
        return grad, self._unit_hess(raw)

    def eval(self, y, raw, sample_weight=None):
        r = np.abs(raw - y)
        d = self.delta
        loss = np.where(r <= d, 0.5 * r * r, d * (r - 0.5 * d))
        return float(np.average(loss, weights=sample_weight))

    def transform(self, raw):
        return raw


class Poisson:
    """Poisson regression for counts with a log link: raw = log(mu).

    grad = mu - y, hess = mu. Predictions (`transform`) are exp(raw) > 0.
    """

    name = "Poisson"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        if np.any(y < 0):
            raise ValueError("loss='Poisson' requires non-negative y.")

        mean = np.average(y, weights=sample_weight)
        if mean <= 0:
            raise ValueError("loss='Poisson' requires y with a positive mean.")

        return float(np.log(mean))

    def grad_hess(self, y, raw):
        mu = _exp(raw)
        return mu - y, np.maximum(mu, 1e-6)

    def eval(self, y, raw, sample_weight=None):
        """Mean Poisson deviance (2 * (y log(y/mu) - (y - mu)); y log y := 0 at 0)."""
        mu = _exp(raw)

        ylog = np.zeros_like(mu)
        nz = y > 0
        ylog[nz] = y[nz] * np.log(y[nz] / mu[nz])

        return float(np.average(2.0 * (ylog - (y - mu)),
                                weights=sample_weight))

    def transform(self, raw):
        return _exp(raw)


class Gamma:
    """Gamma regression for positive, right-skewed targets. Log link: raw = log(mu).

    grad = 1 - y/mu, hess = y/mu (the gamma NLL curvature).
    """

    name = "Gamma"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        if np.any(y <= 0):
            raise ValueError("loss='Gamma' requires strictly positive y.")
        return float(np.log(np.average(y, weights=sample_weight)))

    def grad_hess(self, y, raw):
        y_over_mu = y * _exp(-raw)
        return 1.0 - y_over_mu, np.maximum(y_over_mu, 1e-6)

    def eval(self, y, raw, sample_weight=None):
        """Mean gamma deviance: 2 * (log(mu/y) + y/mu - 1)."""
        y_over_mu = y * _exp(-raw)
        dev = 2.0 * (-np.log(y_over_mu) + y_over_mu - 1.0)
        return float(np.average(dev, weights=sample_weight))

    def transform(self, raw):
        return _exp(raw)


class Tweedie:
    """Tweedie regression (compound Poisson-gamma) with a log link: raw = log(mu).

    For non-negative targets with exact zeros plus a long right tail (insurance
    claims, rainfall). `power` p in (1, 2) interpolates Poisson -> Gamma.
    grad = mu^(2-p) - y mu^(1-p), hess = (2-p) mu^(2-p) - (1-p) y mu^(1-p).
    """

    name = "Tweedie"
    is_classification = False
    adjusts_leaves = False

    def __init__(self, power=1.5):
        power = float(power)
        if not 1.0 < power < 2.0:
            raise ValueError(
                f"Tweedie variance power must be in (1, 2); got {power!r}.")
        self.power = power

    def init(self, y, sample_weight=None):
        if np.any(y < 0):
            raise ValueError("loss='Tweedie' requires non-negative y.")

        mean = np.average(y, weights=sample_weight)
        if mean <= 0:
            raise ValueError("loss='Tweedie' requires y with a positive mean.")

        return float(np.log(mean))

    def grad_hess(self, y, raw):
        p = self.power
        e1 = _exp((1.0 - p) * raw)   # mu^(1-p)
        e2 = _exp((2.0 - p) * raw)   # mu^(2-p)
        grad = e2 - y * e1
        hess = (2.0 - p) * e2 - (1.0 - p) * y * e1
        return grad, np.maximum(hess, 1e-6)

    def eval(self, y, raw, sample_weight=None):
        """Mean Tweedie deviance at `power` (y = 0 contributes only the mu term)."""
        p = self.power
        mu1p = _exp((1.0 - p) * raw)
        mu2p = _exp((2.0 - p) * raw)
        dev = 2.0 * (np.power(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
                     - y * mu1p / (1.0 - p) + mu2p / (2.0 - p))
        return float(np.average(dev, weights=sample_weight))

    def transform(self, raw):
        return _exp(raw)


class CustomObjective:
    """Base class for user-defined regression objectives.

    Subclass and implement ``grad_hess(y, raw)`` -> (gradient, hessian) and
    ``eval(y, raw, sample_weight=None)`` -> scalar (lower is better; drives
    early stopping). Optionally override ``init(y, sample_weight=None)`` (the
    starting raw score, default 0.0) and ``transform(raw)`` (raw scores ->
    predictions, default identity).

    Pass an *instance* as the regressor's ``loss``. Instances must be stateless
    across fits and picklable, so define the subclass at module level: bagged
    members fit in worker processes.

    Read more in the [User Guide](https://bbstats.github.io/chimeraboost/recipes/).
    """

    name = "Custom"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        return 0.0

    def grad_hess(self, y, raw):
        raise NotImplementedError

    def eval(self, y, raw, sample_weight=None):
        raise NotImplementedError

    def transform(self, raw):
        return raw


# Largest class count for which the fused kernel below is bit-identical to
# `_softmax_numpy`. The limit is numpy's, not ours: a reduction over a short
# inner axis is summed left to right, but from length 8 numpy switches to
# pairwise blocking, and a row-at-a-time accumulator stops agreeing in the last
# bit. Measured over K = 2..50 -- the first disagreement is exactly at K = 8
# (`benchmarks/f4_c1_walltime.py`). Above the limit we keep numpy's own path
# rather than ship an answer that is merely close.
_SOFTMAX_MAX_K = 7


@njit(cache=True, parallel=True)
def _softmax_kernel(F):
    # Row-wise softmax in one pass per row, the multiclass twin of `_sigmoid`
    # above. Subtracting the row max before exp() is the same overflow guard
    # the numpy version uses.
    #
    # The final loop DIVIDES by the sum. Hoisting `inv = 1.0 / s` out and
    # multiplying is the obvious micro-optimization, costs nothing measurable,
    # and breaks bit-identity at every K -- a reciprocal followed by a multiply
    # does not round like a divide. Leave it as a divide.
    n, K = F.shape
    out = np.empty((n, K), dtype=np.float64)
    for i in prange(n):
        m = F[i, 0]
        for k in range(1, K):
            if F[i, k] > m:
                m = F[i, k]
        s = 0.0
        for k in range(K):
            e = np.exp(F[i, k] - m)
            out[i, k] = e
            s += e
        for k in range(K):
            out[i, k] = out[i, k] / s
    return out


@njit(cache=True, parallel=True)
def _softmax_grad_hess_kernel(F, Y):
    # Fused grad_hess for MultiSoftmax: the softmax row above plus
    # `grad = P - Y` and `hess = max(P * (1 - P), 1e-6)` in one pass, so the
    # output is bit-identical to `_grad_hess_numpy` on the same machine. The
    # exp loop is `_softmax_kernel`'s verbatim (same max, same accumulation
    # order into `s`; the exps rest in the grad buffer until the final loop
    # overwrites them). The divide stays a divide -- a hoisted reciprocal
    # does not round like a divide. The 1e-6 floor is a comparison because
    # that is `np.maximum(h, 1e-6)` on non-NaN input, and `h` cannot be NaN:
    # `p` is in [0, 1], so `h = p * (1 - p)` lands in [0, 0.25].
    n, K = F.shape
    grad = np.empty((n, K), dtype=np.float64)
    hess = np.empty((n, K), dtype=np.float64)
    for i in prange(n):
        m = F[i, 0]
        for k in range(1, K):
            if F[i, k] > m:
                m = F[i, k]
        s = 0.0
        for k in range(K):
            e = np.exp(F[i, k] - m)
            grad[i, k] = e
            s += e
        for k in range(K):
            p = grad[i, k] / s
            grad[i, k] = p - Y[i, k]
            h = p * (1.0 - p)
            hess[i, k] = h if h >= 1e-6 else 1e-6
    return grad, hess


@njit(cache=True, parallel=True)
def _softmax_ce_kernel(F, Y):
    # Per-row cross-entropy for MultiSoftmax.eval: the softmax row plus the
    # clip, the logs and the class sum in one pass, so the vector is
    # bit-identical to the `row_ce` line of `_eval_numpy` on the same machine.
    # Cross-libm caveat, as for the Logloss kernel: numba log() here, numpy log()
    # in the oracle -- bit-equal here, last-bit-different on some CPUs; tests use 4/8 ULP.
    # The max loop and the exp loop are `_softmax_kernel`'s verbatim (same row
    # max, same sequential accumulation into `s`); the final loop recomputes
    # `exp(F - m)` instead of reloading a stored buffer, which is exact
    # because exp is a pure function -- the same input reproduces the same
    # bits -- and the divide stays a divide, since a hoisted reciprocal does
    # not round like one. The clip is two comparisons so a NaN passes through
    # exactly as `np.clip` passes it (a NaN fails both comparisons and stays
    # NaN). The class sum runs k = 0..K-1 over EVERY term, including the
    # `0 * log(p)` signed zeros, because that is numpy's left-to-right order
    # for K <= 7, and the row is negated after the sum -- `-np.sum(...)`, not
    # a sum of negated terms. The mean stays in numpy: numpy's pairwise
    # summation is not a sequential numba sum, and a fused reduction would
    # drift.
    n, K = F.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        m = F[i, 0]
        for k in range(1, K):
            if F[i, k] > m:
                m = F[i, k]
        s = 0.0
        for k in range(K):
            s += np.exp(F[i, k] - m)
        total = 0.0
        for k in range(K):
            p = np.exp(F[i, k] - m) / s
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0:
                p = 1.0
            total += Y[i, k] * np.log(p)
        out[i] = -total
    return out


def _softmax_numpy(F):
    """The reference implementation, and the live path for K > 7.

    Kept as a named function because it is also the oracle the tests check the
    kernel against -- the bit-identity claim is only meaningful against the
    code that used to run."""
    z = F - F.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _softmax(F):
    if F.dtype == np.float64 and 0 < F.shape[1] <= _SOFTMAX_MAX_K:
        return _softmax_kernel(F)
    return _softmax_numpy(F)


class MultiSoftmax:
    """Multinomial logistic loss. Operates on raw scores F of shape (n, K).

    grad = softmax(F) - Y, hess = p * (1 - p) per class, floored at 1e-6.
    """

    name = "MultiClass"
    is_classification = True

    def __init__(self, n_classes):
        self.K = int(n_classes)

    def init(self, Y, sample_weight=None):  # Y one-hot (n, K)
        p = np.clip(np.average(Y, axis=0, weights=sample_weight), 1e-6, 1.0)
        return np.log(p)  # (K,)

    def _grad_hess_numpy(self, Y, F):  # F (n, K)
        P = _softmax(F)
        grad = P - Y
        hess = np.maximum(P * (1.0 - P), 1e-6)
        return grad, hess

    def grad_hess(self, Y, F):  # F (n, K)
        if (F.dtype == np.float64 and Y.dtype == np.float64
                and F.shape == Y.shape
                and 0 < F.shape[1] <= _SOFTMAX_MAX_K):
            if not F.flags.c_contiguous:
                F = np.ascontiguousarray(F)
            if not Y.flags.c_contiguous:
                Y = np.ascontiguousarray(Y)
            return _softmax_grad_hess_kernel(F, Y)
        return self._grad_hess_numpy(Y, F)

    def _eval_numpy(self, Y, F, sample_weight=None):
        P = np.clip(_softmax(F), 1e-12, 1.0)
        row_ce = -np.sum(Y * np.log(P), axis=1)
        return float(np.average(row_ce, weights=sample_weight))

    def eval(self, Y, F, sample_weight=None):
        if (F.dtype == np.float64 and Y.dtype == np.float64
                and F.shape == Y.shape
                and 0 < F.shape[1] <= _SOFTMAX_MAX_K):
            if not F.flags.c_contiguous:
                F = np.ascontiguousarray(F)
            if not Y.flags.c_contiguous:
                Y = np.ascontiguousarray(Y)
            row_ce = _softmax_ce_kernel(F, Y)
            return float(np.average(row_ce, weights=sample_weight))
        return self._eval_numpy(Y, F, sample_weight)

    def transform(self, F):
        return _softmax(F)


class MultiQuantile(_UnitHessian):
    """Pinball loss on a shared tau grid: K quantile channels at once.

    Raw scores are an (n, K) matrix; column k estimates the `taus[k]`
    conditional quantile. Nothing here couples the channels -- each column is
    exactly the scalar `Quantile` loss at its own alpha, and a one-element grid
    reproduces it term for term.

    Columns of ``F`` may therefore cross during a fit, and none of the maths
    below cares. The coupling lives in the booster, which shares one tree
    structure per round and rearranges each row before returning it.

    `taus` must be ascending, unique and strictly inside (0, 1); the estimator
    validates that before constructing this.
    """

    name = "MultiQuantile"
    is_classification = False
    adjusts_leaves = True

    def __init__(self, taus):
        self.taus = np.ascontiguousarray(taus, dtype=np.float64)
        self.K = self.taus.size

    def init(self, y, sample_weight=None):
        """Global (weighted) quantile at each level -- a (K,) starting vector."""
        q = np.array([_weighted_quantile(y, sample_weight, a)
                      for a in self.taus])

        # A no-op: quantiles of one sample are already monotone in alpha. Kept
        # because an ordered starting vector costs nothing and makes the
        # zero-tree prediction trivially well formed.
        q.sort()

        return q

    def grad_hess(self, y, F):
        """(n, K) pinball gradient; hessian is 1 everywhere.

        Sign convention matches the scalar `Quantile`: -alpha where the target
        sits at or above the current estimate, 1 - alpha below.
        """
        grad = np.where(y[:, None] >= F, -self.taus, 1.0 - self.taus)
        return grad, self._unit_hess(F)

    def eval(self, y, F, sample_weight=None):
        """Mean pinball loss across the grid.

        This is the CRPS convention used by `chimeraboost.quantile_metrics.crps`,
        and the early-stopping metric.
        """
        r = y[:, None] - F
        pinball = np.maximum(self.taus * r, (self.taus - 1.0) * r)
        return float(np.average(pinball.mean(axis=1), weights=sample_weight))

    def transform(self, F):
        return F


LOSSES = {"RMSE": RMSE, "Logloss": Logloss, "MAE": MAE, "Quantile": Quantile,
          "Huber": Huber, "Poisson": Poisson, "Gamma": Gamma,
          "Tweedie": Tweedie}

