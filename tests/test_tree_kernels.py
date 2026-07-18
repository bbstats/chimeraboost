"""Fused fit kernels must match the retained reference kernels bit for bit.

`_build_and_split` replaced the `_build_histograms_into` + `_best_split` pair
on the fit path (one launch, occupied-leaf skipping, per-feature bin counts,
transposed scan). The originals are kept purely as the equivalence oracle for
these tests: any diff here is a correctness bug, not a tolerance question.
"""

import numpy as np
import pytest

from chimeraboost.tree import (
    _best_split,
    _build_and_split,
    _build_histograms_into,
    _descend_leaves,
    _descend_leaves_serial,
    _linear_leaf_fit,
    _linear_leaf_fit_ref,
)


def _make_level(n, n_features, max_bins, depth_level, seed, skew):
    """Random binned matrix + a realistic leaf assignment for one level."""
    rng = np.random.RandomState(seed)
    nbins = np.full(n_features, max_bins, dtype=np.int64)
    nbins[: n_features // 4] = rng.randint(2, 12, size=n_features // 4)
    Xb = np.empty((n_features, n), dtype=np.uint16)
    for f in range(n_features):
        v = (rng.zipf(1.5, size=n) % nbins[f]) if skew \
            else rng.randint(0, nbins[f], size=n)
        Xb[f] = v.astype(np.uint16)
    grad = rng.randn(n)
    hess = np.ones(n)
    leaf = np.zeros(n, dtype=np.int64)
    for _ in range(depth_level):
        f = rng.randint(0, n_features)
        t = rng.randint(0, max(nbins[f] - 1, 1))
        leaf = (leaf << 1) + (Xb[f] > t).astype(np.int64)
    return Xb, grad, hess, leaf, 1 << depth_level, nbins


def test_fused_build_and_split_matches_reference_exactly():
    """Across levels, skew, column masks, non-unit hessians, and
    min_child_weight legality, the fused kernel must return the exact
    (feature, threshold, gain) of the reference pair — including when the
    fused kernel is handed a poisoned (NaN) buffer, proving it never reads a
    cell it did not zero or write."""
    n, n_features, max_bins, depth = 500, 13, 64, 6
    l2 = 3.0
    hist_ref = np.zeros((n_features, 1 << depth, max_bins, 2))
    hist_fused = np.empty_like(hist_ref)

    checked = 0
    for seed in range(12):
        for lev in range(depth):
            for skew in (False, True):
                Xb, g, h, leaf, nl, nbins = _make_level(
                    n, n_features, max_bins, lev, seed * 100 + lev, skew)
                mask = np.ones(n_features, dtype=np.int64)
                if seed % 3 == 1:
                    mask[::4] = 0                                # colsample
                if seed % 5 == 2:
                    h = np.random.RandomState(seed).rand(n)      # varied hess
                mcw = 5.0 if seed % 4 == 3 else 1.0              # legality
                active = np.flatnonzero(np.bincount(leaf, minlength=nl))

                _build_histograms_into(Xb, g, h, leaf, nl, hist_ref, mask)
                ref = _best_split(hist_ref, nbins, l2, mask, mcw, nl)

                hist_fused[:] = np.nan                            # poison
                got = _build_and_split(Xb, g, h, leaf, active, hist_fused,
                                       mask, nbins, l2, mcw)

                assert got[0] == ref[0] and got[1] == ref[1]
                assert got[2] == ref[2] or (
                    np.isneginf(got[2]) and np.isneginf(ref[2]))
                # The active hist slices themselves must be identical too.
                for f in np.flatnonzero(mask):
                    assert np.array_equal(
                        hist_ref[f][active][:, : nbins[f]],
                        hist_fused[f][active][:, : nbins[f]])
                checked += 1
    assert checked == 12 * depth * 2


def test_fused_kernel_accepts_superset_active_list():
    """Passing arange(n_leaves) (the large-n path) must give the same answer
    as the exact occupied list — empty rows are zeroed and contribute zeros."""
    Xb, g, h, leaf, nl, nbins = _make_level(300, 7, 32, 4, seed=7, skew=True)
    mask = np.ones(7, dtype=np.int64)
    hist = np.zeros((7, 16, 32, 2))
    exact = np.flatnonzero(np.bincount(leaf, minlength=nl))
    assert exact.size < nl, "test needs at least one empty leaf"
    r_exact = _build_and_split(Xb, g, h, leaf, exact, hist, mask, nbins, 3.0, 1.0)
    r_all = _build_and_split(Xb, g, h, leaf,
                             np.arange(nl, dtype=np.int64), hist, mask,
                             nbins, 3.0, 1.0)
    assert r_exact == r_all


def test_linear_leaf_fit_matches_reference_exactly():
    """The restructured ridge (row-major design table, mirrored intercept
    column, hoisted h*x) must reproduce the reference coefficients bit for
    bit -- across random leaf assignments (incl. empty and below-fallback
    leaves), NaN (missing) bin centers, varied k, and non-unit hessians."""
    checked = 0
    for seed in range(12):
        rng = np.random.RandomState(seed)
        n = (40, 400, 3000)[seed % 3]           # 40 forces the fallback path
        n_features, max_bins = 8, 32
        depth = (2, 4, 6)[seed % 3]
        n_leaves = 1 << depth
        Xb = rng.randint(0, max_bins, size=(n_features, n)).astype(np.uint16)
        grad = rng.randn(n)
        hess = rng.rand(n) + 0.1 if seed % 2 else np.ones(n)
        if seed % 3 == 2:                        # skew: empty + sparse leaves
            leaf = (rng.zipf(1.3, size=n) % n_leaves).astype(np.int64)
        else:
            leaf = rng.randint(0, n_leaves, size=n).astype(np.int64)
        k = 1 + seed % 5
        lin_feats = rng.choice(n_features, size=k, replace=False) \
            .astype(np.int64)
        centers_std = rng.randn(n_features, max_bins)
        centers_std[rng.rand(n_features, max_bins) < 0.1] = np.nan
        args = (leaf, grad, hess, n_leaves, lin_feats, centers_std, Xb,
                1.0, 0.7, 0.15)
        ref = _linear_leaf_fit_ref(*args)
        got = _linear_leaf_fit(*args)
        assert np.array_equal(got, ref)
        checked += 1
    assert checked == 12


def test_descend_serial_matches_parallel_exactly():
    rng = np.random.RandomState(3)
    Xf = rng.randint(0, 128, 5000).astype(np.uint16)
    for t in (0, 63, 126):
        base = rng.randint(0, 16, 5000).astype(np.int64)
        a, b = base.copy(), base.copy()
        _descend_leaves(a, Xf, t)
        _descend_leaves_serial(b, Xf, t)
        assert np.array_equal(a, b)
