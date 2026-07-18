"""Quantized-gradient histograms (QUANT_PLAN.md): the packed-int64 split
search must equal the float kernel EXACTLY on exactly-representable data,
and behave (deterministic, bounded, close) on real data.

The oracle trick: with power-of-two scales and integer-valued grad/hess
(g = m * 2**-10, |m| <= qmax, max present), quantization is lossless —
g * inv_dg is an exact integer, stochastic rounding floor(m + u) returns m
for every u in [0, 1), and both kernels compute bit-identical gains (all
partial sums are exact multiples of a common power of two, far below 2**53).
Any diff on such data is a correctness bug, not a tolerance question.
"""

import numpy as np
import pytest

from chimeraboost.booster import GradientBoosting, MulticlassBoosting
from chimeraboost.tree import _QMAX_CAP, _quantize_pack, build_oblivious_tree


def _exact_data(n, n_features, max_bins, seed):
    """Binned matrix + grad/hess that quantize losslessly (see module doc)."""
    rng = np.random.RandomState(seed)
    nbins = np.full(n_features, max_bins, dtype=np.int64)
    nbins[: n_features // 4] = rng.randint(2, 12, size=n_features // 4)
    Xb = np.empty((n_features, n), dtype=np.uint16)
    for f in range(n_features):
        Xb[f] = rng.randint(0, nbins[f], size=n).astype(np.uint16)
    mg = rng.randint(-_QMAX_CAP, _QMAX_CAP + 1, size=n)
    mg[0] = _QMAX_CAP                       # pin the abs-max so dg = 2**-10
    mh = rng.randint(1, _QMAX_CAP + 1, size=n)
    mh[1] = _QMAX_CAP                       # pin the max so dh = 2**-12
    grad = mg.astype(np.float64) * 2.0 ** -10
    hess = mh.astype(np.float64) * 2.0 ** -12
    return Xb, grad, hess, nbins


@pytest.mark.parametrize("n", [4096, 40000])  # small-n and large-n dispatch
def test_quantized_tree_matches_float_exactly_on_representable_data(n):
    Xb, grad, hess, nbins = _exact_data(n, 13, 64, seed=3)
    kw = dict(max_depth=6, l2=3.0, lr=0.5, min_child_weight=4.0)
    tref, leaf_ref = build_oblivious_tree(Xb, grad, hess, nbins, **kw)
    for qseed in (0, 997, (1 << 63) - 1):
        tq, leaf_q = build_oblivious_tree(Xb, grad, hess, nbins, **kw,
                                          quantize=True, qseed=qseed)
        assert np.array_equal(tref.splits_feat, tq.splits_feat)
        assert np.array_equal(tref.splits_thr, tq.splits_thr)
        assert np.array_equal(tref.gains, tq.gains)
        assert np.array_equal(tref.values, tq.values)
        assert np.array_equal(leaf_ref, leaf_q)
    assert tref.depth == 6      # the comparison exercised a full-depth tree


def test_quantize_pack_bounds_and_determinism():
    rng = np.random.RandomState(0)
    n = 1000
    grad = rng.randn(n) * 3.0
    hess = np.abs(rng.randn(n))
    qmax = 7                                # tiny range -> clamp edges hit
    gmax = np.abs(grad).max()
    hmax = hess.max()
    out1 = np.empty(n, dtype=np.int64)
    out2 = np.empty(n, dtype=np.int64)
    _quantize_pack(grad, hess, qmax / gmax, qmax / hmax, np.int64(qmax),
                   np.uint64(12345), out1)
    _quantize_pack(grad, hess, qmax / gmax, qmax / hmax, np.int64(qmax),
                   np.uint64(12345), out2)
    assert np.array_equal(out1, out2)       # counter-based rounding: pure
    qh = out1 & 0xFFFFFFFF
    qg = out1 >> 32
    assert qh.min() >= 0 and qh.max() <= qmax
    assert qg.min() >= -qmax and qg.max() <= qmax
    # A different seed must actually change some roundings.
    _quantize_pack(grad, hess, qmax / gmax, qmax / hmax, np.int64(qmax),
                   np.uint64(54321), out2)
    assert not np.array_equal(out1, out2)


def _toy_regression(n=2500, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    y = X[:, 0] * 2.0 + np.sin(X[:, 1] * 3.0) + 0.3 * rng.randn(n)
    return X, y


def test_quantized_booster_deterministic_and_close_to_float():
    X, y = _toy_regression()
    preds = []
    for _ in range(2):
        m = GradientBoosting(n_estimators=60, random_state=7,
                             quantize_gradients=True).fit(X, y)
        preds.append(m.predict_raw(X))
    assert np.array_equal(preds[0], preds[1])   # same seed -> same model

    ref = GradientBoosting(n_estimators=60, random_state=7).fit(X, y)
    rmse_q = np.sqrt(np.mean((preds[0] - y) ** 2))
    rmse_f = np.sqrt(np.mean((ref.predict_raw(X) - y) ** 2))
    # ~15-bit quantization only perturbs split choice; train RMSE must land
    # within a few percent of the float path (loose by design — drift class).
    assert rmse_q < rmse_f * 1.05


def test_quantized_binary_and_multiclass_smoke():
    rng = np.random.RandomState(1)
    X = rng.randn(1200, 6)
    yb = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    mb = GradientBoosting(loss="Logloss", n_estimators=40, random_state=3,
                          quantize_gradients=True).fit(X, yb)
    raw = mb.predict_raw(X)                     # log-odds
    assert np.isfinite(raw).all()
    assert ((yb == 1) == (raw > 0.0)).mean() > 0.9

    ym = (X[:, 0] > 0.5).astype(np.int64) + (X[:, 1] > 0).astype(np.int64)
    mm = MulticlassBoosting(n_estimators=30, random_state=3,
                            quantize_gradients=True).fit(X, ym)
    sm = mm.predict_raw(X)                      # (n, K) softmax scores
    assert sm.shape == (1200, 3)
    assert np.isfinite(sm).all()
    assert (mm.classes_[np.argmax(sm, axis=1)] == ym).mean() > 0.8


def test_sklearn_wrappers_pass_quantize_through():
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
    X, y = _toy_regression(n=600)
    r = ChimeraBoostRegressor(n_estimators=25, random_state=0,
                              early_stopping=False, cross_features=False,
                              linear_leaves=False, quantize_gradients=True)
    r.fit(X, y)
    assert r.model_.quantize_gradients is True
    c = ChimeraBoostClassifier(n_estimators=25, random_state=0,
                               early_stopping=False, cross_features=False,
                               linear_leaves=False, quantize_gradients=True)
    c.fit(X, (y > np.median(y)).astype(int))
    assert c.model_.quantize_gradients is True
