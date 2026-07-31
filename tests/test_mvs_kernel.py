"""The numba MVS kernels must be bit-identical to the numpy code they replaced.

MVS picks a threshold lambda from the gradient magnitudes, turns it into a
per-row keep probability, and draws a mask against it. A lambda that moves by
one ulp flips rows near the knife edge and changes the model, so "close" is not
good enough here: the oracles below are the previous numpy implementations,
transcribed verbatim, and every comparison is exact.

Same convention as tests/test_greedy_kernel_and_group_bagging.py, which pins
the numba greedy-border sweep against a transcription of the old pure-Python
code, and tests/test_binner_parallel_fit.py, which forces both dispatch arms.
"""

import numpy as np
import pytest

import chimeraboost.booster as BO
from chimeraboost.tree import (_mvs_lambda_scan, _mvs_weights,
                               _mvs_weights_serial)


# --- oracles: the implementations in use before the numba port ------------

def _threshold_oracle(abs_g, target):
    n = len(abs_g)
    if target >= n:
        return 0.0
    sorted_g = np.sort(abs_g)[::-1]  # descending
    total = sorted_g.sum()
    if total < 1e-12:
        return 0.0
    prefix = np.empty(n)
    prefix[0] = 0.0
    prefix[1:] = np.cumsum(sorted_g[:-1])
    suffix = total - prefix
    remaining = target - np.arange(n, dtype=np.float64)
    cond = (remaining > 0) & (sorted_g * remaining <= suffix)
    if not cond.any():
        return 0.0
    k = int(np.argmax(cond))
    return suffix[k] / remaining[k]


def _weights_oracle(grad, u, lam, subsample):
    abs_g = np.abs(grad)
    prob = np.minimum(abs_g / lam, 1.0)
    mask = u < prob
    max_w = 1.0 / max(subsample, 1e-3)
    return np.where(mask, np.minimum(1.0 / np.maximum(prob, 1e-10), max_w), 0.0)


def _threshold_new(abs_g, target):
    """What booster._mvs_threshold now does: numpy sort + sum, fused scan."""
    n = len(abs_g)
    if target >= n:
        return 0.0
    sorted_g = np.sort(abs_g)[::-1]
    total = sorted_g.sum()
    if total < 1e-12:
        return 0.0
    return _mvs_lambda_scan(sorted_g, total, target)


# --- gradient shapes that stress the summation order ----------------------

def _gradients(name, n, rng):
    if name == "normal":
        return rng.standard_normal(n)
    if name == "heavy":            # magnitudes spanning 1e-8 .. 1e8
        return rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8, n)
    if name == "sparse":           # 95% exact zeros
        return rng.standard_normal(n) * (rng.random(n) < 0.05)
    if name == "ties":
        return np.round(rng.standard_normal(n), 2)
    if name == "logistic":         # the shape a binary-logloss residual takes
        return rng.random(n) - (rng.random(n) < 0.5)
    if name == "tiny":             # near the total < 1e-12 degenerate guard
        return rng.standard_normal(n) * 1e-13
    if name == "constant":
        return np.full(n, 0.3)
    if name == "nan":              # a broken fit, but must not change behaviour
        g = rng.standard_normal(n)
        g[::7] = np.nan
        return g
    raise AssertionError(name)


SHAPES = ["normal", "heavy", "sparse", "ties", "logistic", "tiny",
          "constant", "nan"]
SIZES = [1, 2, 3, 17, 999, 10_007, 200_003]
SUBSAMPLES = [0.05, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]


@pytest.mark.parametrize("shape", SHAPES)
def test_lambda_scan_matches_the_numpy_threshold_exactly(shape):
    rng = np.random.default_rng(hash(shape) % 2**32)
    for n in SIZES:
        abs_g = np.abs(_gradients(shape, n, rng))
        for sub in SUBSAMPLES:
            target = sub * n
            want = _threshold_oracle(abs_g, target)
            got = _threshold_new(abs_g, target)
            assert got == want or (np.isnan(want) and np.isnan(got)), (
                f"lambda drift: {shape} n={n} subsample={sub} "
                f"{got!r} != {want!r}")


@pytest.mark.parametrize("kernel", [_mvs_weights, _mvs_weights_serial])
@pytest.mark.parametrize("shape", SHAPES)
def test_weight_kernels_match_the_numpy_expression_exactly(kernel, shape):
    rng = np.random.default_rng(hash(shape) % 2**32 + 1)
    for n in [1, 17, 999, 10_007, 200_003]:
        for sub in [0.05, 0.3, 0.7, 0.95]:
            for scale in (1e-12, 1.0, 1e8):
                grad = _gradients(shape, n, rng) * scale
                u = rng.random(n)
                finite = np.abs(grad[np.isfinite(grad)])
                lam = float(np.median(finite)) if finite.size else 0.0
                if not lam:
                    lam = 1.0
                want = _weights_oracle(grad, u, lam, sub)
                got = kernel(grad, u, lam, 1.0 / max(sub, 1e-3))
                assert np.array_equal(want, got), (
                    f"weight drift: {shape} n={n} subsample={sub} scale={scale}")


def test_weight_kernel_twins_agree():
    """The parallel/serial dispatch may only change speed."""
    rng = np.random.default_rng(7)
    for n in (1, 31, 5_000, 100_000):
        grad = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 6, n)
        u = rng.random(n)
        assert np.array_equal(_mvs_weights(grad, u, 0.5, 2.0),
                              _mvs_weights_serial(grad, u, 0.5, 2.0))


def test_cap_and_drop_branches_are_both_reached():
    """Guards the grid above: a run where every row survives, or none does,
    would pass the equality checks while testing almost nothing."""
    rng = np.random.default_rng(11)
    grad = rng.standard_normal(20_000)
    u = rng.random(20_000)
    w = _mvs_weights(grad, u, float(np.median(np.abs(grad))), 1.0 / 0.7)
    assert (w == 0.0).any(), "no row was dropped"
    assert (w == 1.0 / 0.7).any(), "the 1/subsample cap was never hit"
    assert ((w > 1.0) & (w < 1.0 / 0.7)).any(), "no uncapped reweighted row"


@pytest.mark.parametrize("force_serial", [False, True])
def test_fitted_model_is_identical_across_both_dispatch_arms(force_serial,
                                                             monkeypatch):
    """End-to-end: the arm the booster picks must not change the model."""
    from chimeraboost import ChimeraBoostRegressor

    rng = np.random.default_rng(3)
    X = rng.standard_normal((4_000, 6))
    y = np.sin(X[:, 0] * 2) + X[:, 1] * X[:, 2] + rng.standard_normal(4_000) * 0.1
    kw = dict(n_estimators=40, subsample=0.7, random_state=0,
              early_stopping=False)

    monkeypatch.setattr(BO, "_SMALL_N", 1 << 60 if force_serial else 0)
    got = ChimeraBoostRegressor(**kw).fit(X, y).predict(X)
    monkeypatch.undo()
    want = ChimeraBoostRegressor(**kw).fit(X, y).predict(X)
    assert np.array_equal(want, got)
