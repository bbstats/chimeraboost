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


# --------------------------------------------------------------------------
# Weighted statistical helpers
# --------------------------------------------------------------------------

def _weighted_median(values, weights):
    """Weighted nearest-rank median.

    When *weights* is None this delegates to ``np.median`` so the
    ``sample_weight=None`` code path is bitwise identical to the original.
    """
    if weights is None:
        return float(np.median(values)) if values.size else 0.0
    if not values.size:
        return 0.0
    order = np.argsort(values)
    sv, sw = values[order], weights[order]
    cumw = np.cumsum(sw)
    idx = min(int(np.searchsorted(cumw, cumw[-1] * 0.5)), len(sv) - 1)
    return float(sv[idx])


def _weighted_quantile(values, weights, alpha):
    """Weighted nearest-rank quantile at level *alpha*.

    When *weights* is None this delegates to ``np.quantile`` so the
    ``sample_weight=None`` code path is bitwise identical to the original.
    """
    if weights is None:
        return float(np.quantile(values, alpha)) if values.size else 0.0
    if not values.size:
        return 0.0
    order = np.argsort(values)
    sv, sw = values[order], weights[order]
    cumw = np.cumsum(sw)
    idx = min(int(np.searchsorted(cumw, cumw[-1] * alpha)), len(sv) - 1)
    return float(sv[idx])


from numba import njit, prange


@njit(cache=True, parallel=True)
def _sigmoid(z):
    # Numerically stable logistic, parallelized over rows. Branching on sign
    # avoids overflow in exp(): exp(-|z|) is always in [0, 1].
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


class RMSE:
    """Squared-error regression. grad = pred - y, hess = 1."""

    name = "RMSE"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        return float(np.average(y, weights=sample_weight))

    def grad_hess(self, y, raw):
        grad = raw - y
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw, sample_weight=None):
        return float(np.sqrt(np.average((raw - y) ** 2, weights=sample_weight)))

    def transform(self, raw):
        return raw


class Logloss:
    """Binary cross-entropy. raw = log-odds, p = sigmoid(raw)."""

    name = "Logloss"
    is_classification = True
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        p = np.clip(np.average(y, weights=sample_weight), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def grad_hess(self, y, raw):
        p = _sigmoid(raw)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess

    def eval(self, y, raw, sample_weight=None):
        p = np.clip(_sigmoid(raw), 1e-9, 1 - 1e-9)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.average(ce, weights=sample_weight))

    def transform(self, raw):
        return _sigmoid(raw)


class MAE:
    """Mean absolute error. grad = sign(pred - y), hess = 1 (constant).

    Uses the gradient-step approximation for leaf values rather than exact
    medians, which is simple and converges fine with enough small steps.
    """

    name = "MAE"
    is_classification = False
    adjusts_leaves = True   # sign gradients only pick structure; set leaf = median

    def leaf_value(self, residuals, weights=None):
        return _weighted_median(residuals, weights)

    def init(self, y, sample_weight=None):
        return _weighted_median(y, sample_weight)

    def grad_hess(self, y, raw):
        grad = np.sign(raw - y)
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.abs(raw - y), weights=sample_weight))

    def transform(self, raw):
        return raw


class Quantile:
    """Pinball loss for quantile regression at level `alpha` in (0, 1)."""

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
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw, sample_weight=None):
        r = y - raw
        pinball = np.maximum(self.alpha * r, (self.alpha - 1.0) * r)
        return float(np.average(pinball, weights=sample_weight))

    def transform(self, raw):
        return raw


def _softmax(F):
    z = F - F.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


class MultiSoftmax:
    """Multinomial logistic loss. Operates on raw scores F of shape (n, K)."""

    name = "MultiClass"
    is_classification = True

    def __init__(self, n_classes):
        self.K = int(n_classes)

    def init(self, Y, sample_weight=None):  # Y one-hot (n, K)
        p = np.clip(np.average(Y, axis=0, weights=sample_weight), 1e-6, 1.0)
        return np.log(p)  # (K,)

    def grad_hess(self, Y, F):  # F (n, K)
        P = _softmax(F)
        grad = P - Y
        hess = np.maximum(P * (1.0 - P), 1e-6)
        return grad, hess

    def eval(self, Y, F, sample_weight=None):
        P = np.clip(_softmax(F), 1e-12, 1.0)
        row_ce = -np.sum(Y * np.log(P), axis=1)
        return float(np.average(row_ce, weights=sample_weight))

    def transform(self, F):
        return _softmax(F)


LOSSES = {"RMSE": RMSE, "Logloss": Logloss, "MAE": MAE, "Quantile": Quantile}

