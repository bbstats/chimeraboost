"""Bit-identity guards for the fast replay kernel (issue131, I066).

``_linear_leaf_fit_ref`` is a frozen copy of ``tree._linear_leaf_fit`` as of
2026-09-24, before the parallel rewrite. Every test asserts exact
``np.array_equal`` against it: any summation-order or expression change fails
in the last ulp.
"""
import numba
import numpy as np
import pytest
from numba import njit, prange

from chimeraboost.tree import (
    ObliviousTree,
    _assign_leaves,
    _leaf_values,
    _linear_leaf_fit,
    _solve_small,
    replay_oblivious_tree,
)

DEFAULT_THREADS = numba.get_num_threads()


@njit(cache=False, parallel=True)
def _linear_leaf_fit_ref(leaf, grad, hess, n_leaves, lin_feats, centers_std, Xb,  # noqa: C901 -- frozen reference kernel
                         l2_intercept, lin_lambda, lr):
    """Frozen copy of tree._linear_leaf_fit (2026-09-24)."""
    n = leaf.shape[0]
    k = lin_feats.shape[0]
    d = 1 + k
    coef = np.zeros((n_leaves, d))

    counts = np.zeros(n_leaves, dtype=np.int64)
    Gtot = np.zeros(n_leaves)
    Htot = np.zeros(n_leaves)
    for i in range(n):
        l = leaf[i]
        counts[l] += 1
        Gtot[l] += grad[i]
        Htot[l] += hess[i]

    start = np.zeros(n_leaves + 1, dtype=np.int64)
    for l in range(n_leaves):
        start[l + 1] = start[l] + counts[l]

    pos = start[:n_leaves].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        l = leaf[i]
        order[pos[l]] = i
        pos[l] += 1

    for l in prange(n_leaves):
        if counts[l] == 0:
            continue

        if counts[l] < 2 * d or k == 0:
            if Htot[l] > 0.0:
                coef[l, 0] = -lr * Gtot[l] / (Htot[l] + l2_intercept)
            continue

        Ml = np.zeros((d, d))
        rl = np.zeros(d)
        xrow = np.empty(k)

        for q in range(start[l], start[l + 1]):
            i = order[q]
            h = hess[i]
            g = grad[i]

            for j in range(k):
                f = lin_feats[j]
                v = centers_std[f, Xb[f, i]]
                xrow[j] = v if np.isfinite(v) else 0.0

            Ml[0, 0] += h
            rl[0] += -g
            for j in range(k):
                xj = xrow[j]
                Ml[0, 1 + j] += h * xj
                Ml[1 + j, 0] += h * xj
                rl[1 + j] += -g * xj
                for jj in range(k):
                    Ml[1 + j, 1 + jj] += h * xj * xrow[jj]

        Ml[0, 0] += l2_intercept
        for j in range(1, d):
            Ml[j, j] += lin_lambda
        for j in range(d):
            Ml[j, j] += 1e-9

        beta = _solve_small(Ml, rl)
        if np.isnan(beta[0]):
            if Htot[l] > 0.0:
                coef[l, 0] = -lr * Gtot[l] / (Htot[l] + l2_intercept)
            continue

        for j in range(d):
            coef[l, j] = lr * beta[j]

    return coef


def _make_inputs(n, n_leaves, k, seed):
    """Random kernel inputs covering empty, tiny (<2d) and 60%+ leaves."""
    rng = np.random.default_rng(seed)
    d = 1 + k
    n_features, max_bins = 8, 32

    leaf = np.empty(n, dtype=np.int64)
    n_big = max(int(0.6 * n), 2 * d)
    if n_big + 5 <= n:
        n_s1, n_s2 = 2, 3
    elif n_big + 2 <= n:
        n_s1, n_s2 = 1, 1
    else:
        n_big = n - 2
        n_s1, n_s2 = 1, 1
    leaf[:n_big] = 0
    leaf[n_big:n_big + n_s1] = 1
    leaf[n_big + n_s1:n_big + n_s1 + n_s2] = 2
    rest = n - (n_big + n_s1 + n_s2)
    if rest > 0:
        leaf[n_big + n_s1 + n_s2:] = rng.integers(3, n_leaves - 2, rest)

    grad = rng.standard_normal(n)
    hess = rng.random(n) + 0.1
    hess[rng.random(n) < 0.05] = 0.0
    lin_feats = np.arange(k, dtype=np.int64)
    centers_std = rng.standard_normal((n_features, max_bins))
    centers_std[rng.random((n_features, max_bins)) < 0.05] = np.nan
    Xb = rng.integers(0, max_bins, size=(n_features, n)).astype(np.uint16)
    return leaf, grad, hess, lin_feats, centers_std, Xb


@pytest.mark.parametrize("n,n_leaves", [(16, 8), (2000, 16), (200_000, 64)])
@pytest.mark.parametrize("k", [1, 6])
@pytest.mark.parametrize("n_threads", [1, DEFAULT_THREADS])
def test_linear_leaf_fit_matches_ref(n, n_leaves, k, n_threads):
    leaf, grad, hess, lin_feats, cs, Xb = _make_inputs(n, n_leaves, k, 1000 + n + k)
    prev = numba.get_num_threads()
    numba.set_num_threads(n_threads)
    try:
        ref = _linear_leaf_fit_ref(leaf, grad, hess, n_leaves, lin_feats, cs,
                                   Xb, 1.0, 1.0, 0.1)
        got = _linear_leaf_fit(leaf, grad, hess, n_leaves, lin_feats, cs,
                               Xb, 1.0, 1.0, 0.1)
    finally:
        numba.set_num_threads(prev)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def _make_donor(n_features, max_bins, depth, k, seed):
    rng = np.random.default_rng(seed)
    sf = rng.integers(0, n_features, depth).astype(np.int64)
    st = rng.integers(0, max_bins - 1, depth).astype(np.int64)
    n_leaves = 1 << depth
    vals = rng.standard_normal(n_leaves)
    lin_feats = np.arange(k, dtype=np.int64)
    lin_coef = np.zeros((n_leaves, 1 + k))
    return ObliviousTree(sf, st, vals, np.zeros(depth),
                         lin_feats=lin_feats, lin_coef=lin_coef,
                         centers_std=None), n_leaves


@pytest.mark.parametrize("n,depth", [(2000, 4), (200_000, 6)])
@pytest.mark.parametrize("k", [1, 6])
@pytest.mark.parametrize("n_threads", [1, DEFAULT_THREADS])
def test_replay_matches_ref(n, depth, k, n_threads):
    rng = np.random.default_rng(77 + n + k)
    n_features, max_bins = 8, 32
    donor, n_leaves = _make_donor(n_features, max_bins, depth, k, 5)
    Xb = rng.integers(0, max_bins, size=(n_features, n)).astype(np.uint16)
    grad = rng.standard_normal(n)
    hess = rng.random(n) + 0.1
    cs = rng.standard_normal((n_features, max_bins))
    cs[rng.random((n_features, max_bins)) < 0.05] = np.nan

    prev = numba.get_num_threads()
    numba.set_num_threads(n_threads)
    try:
        tree, leaf = replay_oblivious_tree(donor, Xb, grad, hess, 1.0, 0.1,
                                           linear_leaves=True, centers_std=cs,
                                           linear_lambda=1.0)
        leaf_ref = _assign_leaves(Xb, donor.splits_feat, donor.splits_thr)
        values_ref = _leaf_values(leaf_ref, grad, hess, n_leaves, 1.0, 0.1)
        coef_ref = _linear_leaf_fit_ref(leaf_ref, grad, hess, n_leaves,
                                        donor.lin_feats, cs, Xb, 1.0, 1.0, 0.1)
    finally:
        numba.set_num_threads(prev)
    assert np.array_equal(leaf, leaf_ref)
    assert np.array_equal(tree.values, values_ref)
    assert np.array_equal(tree.lin_coef, coef_ref)


def test_replay_constant_matches_ref():
    """Linear off: replay still matches (covers the assign swap alone)."""
    rng = np.random.default_rng(9)
    n, n_features, max_bins = 200_000, 8, 32
    donor, n_leaves = _make_donor(n_features, max_bins, 6, 2, 6)
    Xb = rng.integers(0, max_bins, size=(n_features, n)).astype(np.uint16)
    grad = rng.standard_normal(n)
    hess = rng.random(n) + 0.1
    tree, leaf = replay_oblivious_tree(donor, Xb, grad, hess, 1.0, 0.1)
    leaf_ref = _assign_leaves(Xb, donor.splits_feat, donor.splits_thr)
    values_ref = _leaf_values(leaf_ref, grad, hess, n_leaves, 1.0, 0.1)
    assert np.array_equal(leaf, leaf_ref)
    assert np.array_equal(tree.values, values_ref)
