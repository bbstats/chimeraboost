"""Guards for the output-identical performance refactors (2026-07-30 pass).

Each guard pins the invariant a refactor leans on, so a later change that
breaks the invariant fails here instead of silently changing models.
"""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from chimeraboost import ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.losses import MAE, RMSE, Huber, MultiQuantile, Quantile
from chimeraboost.tree import _SMALL_N


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


# ---------------------------------------------------------------------------
# Scalar _correct_leaves: the compiled K=1 kernel dispatch must reproduce the
# old argsort + per-leaf leaf_value() path exactly. The reference below IS
# that old path, verbatim.
# ---------------------------------------------------------------------------

def _correct_leaves_reference(loss, lr, n_leaves, leaf, y, F, sample_weight):
    residuals = y - F
    values = np.zeros(n_leaves)
    order = np.argsort(leaf, kind="stable")
    counts = np.bincount(leaf, minlength=n_leaves)
    stop = np.cumsum(counts)
    r_sorted = residuals[order]
    w_sorted = sample_weight[order] if sample_weight is not None else None
    for l in range(n_leaves):
        lo, hi = stop[l] - counts[l], stop[l]
        w = w_sorted[lo:hi] if w_sorted is not None else None
        values[l] = lr * loss.leaf_value(r_sorted[lo:hi], w)
    return values


@pytest.mark.parametrize("loss", [MAE(), Quantile(0.3), Quantile(0.9)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n", [200, _SMALL_N * 2])   # serial and parallel
def test_scalar_correct_leaves_matches_reference(loss, weighted, n):
    rng = np.random.default_rng(42)
    n_leaves = 16
    # Leave two leaves empty, and quantize y so tied residuals occur.
    leaf = rng.integers(0, n_leaves - 2, n).astype(np.int64)
    y = np.round(rng.normal(size=n) * 8) / 8
    F = np.round(rng.normal(size=n) * 4) / 4
    w = rng.uniform(0.0, 2.0, n) if weighted else None

    gb = GradientBoosting()
    gb.loss_, gb.lr_ = loss, 0.07
    tree = SimpleNamespace(values=np.zeros(n_leaves))
    gb._correct_leaves(tree, leaf, y, F, w)

    ref = _correct_leaves_reference(loss, 0.07, n_leaves, leaf, y, F, w)
    np.testing.assert_array_equal(tree.values, ref)


def test_apply_parallel_arm_matches_serial():
    # The parallel descend only fires above _ASSIGN_PAR_N rows -- too big for
    # the identity snapshot's eval sets -- so pin the equivalence directly.
    from chimeraboost.tree import ObliviousTree, _ASSIGN_PAR_N, _assign_leaves
    rng = np.random.default_rng(7)
    n = _ASSIGN_PAR_N + 5
    Xb = np.ascontiguousarray(rng.integers(0, 64, size=(5, n)),
                              dtype=np.uint16)
    sf = np.array([0, 3, 1, 4], dtype=np.int64)
    st = np.array([10, 30, 5, 50], dtype=np.int64)
    vals = rng.normal(size=16)
    tree = ObliviousTree(sf, st, vals)
    ref = _assign_leaves(Xb, sf, st)
    np.testing.assert_array_equal(tree.apply(Xb), ref)
    np.testing.assert_array_equal(tree.predict(Xb), vals[ref])
    # Below the gate both routes are the same serial kernel already.
    np.testing.assert_array_equal(tree.apply(Xb[:, :100]), ref[:100])


def test_custom_adjusts_leaves_loss_keeps_generic_path():
    # A user loss subclassing Quantile with its own leaf_value must NOT be
    # captured by the exact-type kernel dispatch.
    class Midhinge(Quantile):
        def leaf_value(self, residuals, weights=None):
            if not residuals.size:
                return 0.0
            return float(np.quantile(residuals, 0.25)
                         + np.quantile(residuals, 0.75)) / 2.0

    rng = np.random.default_rng(3)
    n, n_leaves = 500, 8
    leaf = rng.integers(0, n_leaves, n).astype(np.int64)
    y, F = rng.normal(size=n), rng.normal(size=n)
    loss = Midhinge(0.5)
    gb = GradientBoosting()
    gb.loss_, gb.lr_ = loss, 0.1
    tree = SimpleNamespace(values=np.zeros(n_leaves))
    gb._correct_leaves(tree, leaf, y, F, None)
    ref = _correct_leaves_reference(loss, 0.1, n_leaves, leaf, y, F, None)
    np.testing.assert_array_equal(tree.values, ref)
