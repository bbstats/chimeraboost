"""The thread-pool Binner.fit must be bit-identical to the serial loop.

Same convention as the _bin_matrix / _bin_matrix_serial twins: both dispatch
arms run the identical per-column arithmetic, so every fitted artifact must
match np.array_equal-exactly, across column pathologies, weights, and bin
budgets. The parallel/serial choice is forced through _PARALLEL_FIT_MIN_CELLS.
"""

import numpy as np
import pytest

import chimeraboost.binning as B


def _pathological_matrix(seed, n=800):
    """One column of every shape the binner special-cases."""
    rng = np.random.default_rng(seed)
    cols = [
        rng.standard_normal(n),                      # dense continuous
        np.where(rng.random(n) < 0.7, 0.0,
                 rng.standard_normal(n)),            # zero-inflated (greedy path)
        np.where(rng.random(n) < 0.1, np.nan,
                 rng.standard_normal(n)),            # NaN-laced
        rng.integers(0, 5, n).astype(np.float64),    # few uniques
        np.full(n, 3.25),                            # constant
        np.full(n, np.nan),                          # all missing
        np.round(rng.standard_normal(n), 1),         # heavy ties
    ]
    return np.column_stack(cols)


def _fit_forced(X, w, max_bins, force_parallel, monkeypatch):
    monkeypatch.setattr(
        B, "_PARALLEL_FIT_MIN_CELLS", 0 if force_parallel else 1 << 60)
    return B.Binner(max_bins).fit(X, w)


@pytest.mark.parametrize("max_bins", [2, 8, 16, 128])
@pytest.mark.parametrize("weighted", [False, True])
def test_parallel_fit_bit_identical(max_bins, weighted, monkeypatch):
    for seed in range(5):
        X = _pathological_matrix(seed)
        w = None
        if weighted:
            w = np.random.default_rng(seed + 100).random(X.shape[0]) + 0.25
            w[:: 7] = 0.0  # exercise the zero-weight drop
        ser = _fit_forced(X, w, max_bins, False, monkeypatch)
        par = _fit_forced(X, w, max_bins, True, monkeypatch)
        assert len(ser.borders_) == len(par.borders_)
        for a, b in zip(ser.borders_, par.borders_):
            assert np.array_equal(a, b)
        assert np.array_equal(ser.n_bins_, par.n_bins_)
        for a, b in zip(ser.bin_centers_, par.bin_centers_):
            assert np.array_equal(a, b, equal_nan=True)  # NaN bin center
        assert np.array_equal(ser._borders_flat, par._borders_flat)
        assert np.array_equal(ser._offsets, par._offsets)


def test_parallel_fit_per_feature_budgets(monkeypatch):
    # The cat-aware per-feature max_bins array must keep feature->budget
    # alignment under the pool (order comes from ex.map, not thread finish).
    X = _pathological_matrix(0)
    budgets = np.array([4, 8, 16, 32, 64, 128, 6], dtype=np.int64)
    ser = _fit_forced(X, None, budgets, False, monkeypatch)
    par = _fit_forced(X, None, budgets, True, monkeypatch)
    for a, b in zip(ser.borders_, par.borders_):
        assert np.array_equal(a, b)
    assert np.array_equal(ser.n_bins_, par.n_bins_)


def test_parallel_fit_propagates_errors(monkeypatch):
    # A failure inside a pooled column must surface, not vanish in a thread.
    monkeypatch.setattr(B, "_PARALLEL_FIT_MIN_CELLS", 0)

    def boom(col, max_bins, weights=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(B, "_feature_borders", boom)
    with pytest.raises(RuntimeError, match="boom"):
        B.Binner(16).fit(np.random.default_rng(0).standard_normal((64, 4)))
