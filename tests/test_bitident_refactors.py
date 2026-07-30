"""Guards for the output-identical performance refactors (2026-07-30 pass).

Each guard pins the invariant a refactor leans on, so a later change that
breaks the invariant fails here instead of silently changing models.
"""
import pickle

import numpy as np

from chimeraboost import ChimeraBoostRegressor
from chimeraboost.losses import MAE, RMSE, Huber, MultiQuantile, Quantile


def _reg_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X[:, 0] * 2 + np.sin(X[:, 1] * 3) + rng.normal(scale=0.3, size=n)
    w = rng.uniform(0.5, 2.0, n)
    return X, y, w


# ---------------------------------------------------------------------------
# _UnitHessian cache: shared across rounds, so it must stay all-ones through
# a full fit, and it must not ride along in pickles.
# ---------------------------------------------------------------------------

def test_unit_hessian_cache_is_never_mutated():
    # Weighted + subsampled hits every path that multiplies the hessian
    # (w-weighting, MVS): each must produce a fresh array, not write into
    # the shared cache.
    X, y, w = _reg_data()
    est = ChimeraBoostRegressor(n_estimators=60, subsample=0.7,
                                random_state=0, loss="MAE")
    est.fit(X, y, sample_weight=w)
    cache = est.model_.loss_._hess_cache
    assert cache is not None
    assert np.all(cache == 1.0)


def test_unit_hessian_cache_not_pickled():
    X, y, _ = _reg_data()
    est = ChimeraBoostRegressor(n_estimators=30, random_state=0)
    est.fit(X, y)
    assert est.model_.loss_._hess_cache is not None
    state = pickle.loads(pickle.dumps(est)).model_.loss_.__dict__
    assert "_hess_cache" not in state


def test_unit_hessian_values_and_reuse():
    for loss in (RMSE(), MAE(), Quantile(0.3), Huber(),
                 MultiQuantile(np.array([0.25, 0.75]))):
        raw = (np.zeros((50, 2)) if isinstance(loss, MultiQuantile)
               else np.zeros(50))
        y = np.ones(raw.shape[0])
        _, h1 = loss.grad_hess(y, raw)
        _, h2 = loss.grad_hess(y, raw)
        assert np.array_equal(h1, np.ones_like(raw))
        assert h1 is h2                      # cached, not reallocated
        _, h3 = loss.grad_hess(y[:20], raw[:20])   # shape change -> fresh
        assert h3.shape == raw[:20].shape
