"""Shared-tree multi-quantile head (benchmarks/QUANTILE_PLAN.md): kernel
oracles, the non-crossing guarantee, conformal coverage, and path sanity."""

import pickle

import numpy as np
import pytest

from chimeraboost import ChimeraBoostQuantileRegressor, ChimeraBoostRegressor
from chimeraboost import quantile_metrics as qm
from chimeraboost.booster import MultiQuantileBoosting, _fixed_contrasts
from chimeraboost.losses import MultiQuantile, _weighted_quantile
from chimeraboost.quantile_api import (DEFAULT_QUANTILES,
                                       _auto_min_child_weight, _cqr_scales,
                                       _median_index)
from chimeraboost.tree import (_leaf_quantiles_vec, _leaf_quantiles_vec_serial,
                               _leaf_quantiles_vec_w,
                               _leaf_quantiles_vec_w_serial)

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 10)


def _heteroscedastic(n=4000, p=8, seed=0, narrow=True):
    """Centre driven by one column, spread by another. When `narrow`, half the
    rows have a conditional spread far TIGHTER than the pooled spread -- the
    case a monotone-increment-only construction cannot represent at all."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    mu = 2.0 * X[:, 0]
    sd = (0.1 if narrow else 1.0) + 2.0 * (X[:, 1] > 0)
    y = mu + sd * rng.standard_normal(n)
    return X, y


def _pinball(y, Q, taus=TAUS):
    r = np.asarray(y)[:, None] - Q
    return float(np.maximum(taus * r, (taus - 1.0) * r).mean())


# --------------------------------------------------------------------------
# Kernel oracles
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,n_leaves", [(1000, 16), (37, 8), (9, 4)])
def test_leaf_quantiles_match_numpy_exactly(n, n_leaves):
    """The registered bit-exact equivalence: column k of the leaf kernel IS
    `np.quantile` at taus[k] on that leaf's residuals. Not a tolerance
    question -- any diff is a correctness bug."""
    rng = np.random.default_rng(n)
    y = rng.standard_normal(n) * 3.0
    F = np.repeat(rng.standard_normal((n, 1)), TAUS.size, axis=1)
    leaf = rng.integers(0, n_leaves, size=n)
    lr = 0.1

    got = _leaf_quantiles_vec(leaf, y, F, TAUS, n_leaves, lr)
    ref = np.zeros((n_leaves, TAUS.size))
    for l in range(n_leaves):
        m = leaf == l
        if not m.any():
            continue
        for k in range(TAUS.size):
            ref[l, k] = lr * float(np.quantile(y[m] - F[m, k], TAUS[k]))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("n,n_leaves", [(1000, 16), (37, 8)])
def test_leaf_quantiles_weighted_match_reference(n, n_leaves):
    """Weighted twin reproduces `losses._weighted_quantile` (nearest rank on
    cumulative weight) exactly, which is the rule the scalar weighted quantile
    path already uses."""
    rng = np.random.default_rng(n + 1)
    y = rng.standard_normal(n) * 3.0
    F = np.repeat(rng.standard_normal((n, 1)), TAUS.size, axis=1)
    leaf = rng.integers(0, n_leaves, size=n)
    w = rng.uniform(0.2, 2.0, size=n)
    lr = 0.1

    got = _leaf_quantiles_vec_w(leaf, y, F, w, TAUS, n_leaves, lr)
    ref = np.zeros((n_leaves, TAUS.size))
    for l in range(n_leaves):
        m = leaf == l
        if not m.any():
            continue
        for k in range(TAUS.size):
            ref[l, k] = lr * _weighted_quantile(y[m] - F[m, k], w[m], TAUS[k])
    np.testing.assert_array_equal(got, ref)


def test_leaf_quantile_serial_twin_is_bit_identical():
    """Every parallel kernel in this repo has a serial twin for small n, and
    the two must be equal, not close."""
    rng = np.random.default_rng(7)
    n, n_leaves = 2000, 16
    y = rng.standard_normal(n)
    F = np.sort(rng.standard_normal((n, TAUS.size)), axis=1)
    leaf = rng.integers(0, n_leaves, size=n)
    w = rng.uniform(0.5, 1.5, size=n)

    np.testing.assert_array_equal(
        _leaf_quantiles_vec(leaf, y, F, TAUS, n_leaves, 0.1),
        _leaf_quantiles_vec_serial(leaf, y, F, TAUS, n_leaves, 0.1))
    np.testing.assert_array_equal(
        _leaf_quantiles_vec_w(leaf, y, F, w, TAUS, n_leaves, 0.1),
        _leaf_quantiles_vec_w_serial(leaf, y, F, w, TAUS, n_leaves, 0.1))


def test_projected_gradient_makes_no_ordering_assumption():
    """`_project_pinball` collapses the K gradient columns onto one direction
    without building them. It once did that by binary-searching the row's PIT
    rank, which silently returns the wrong channel set once a row of F crosses
    -- and rows of F are free to cross now. Score a deliberately crossing F
    against the dense definition."""
    from chimeraboost.tree import _project_pinball
    rng = np.random.default_rng(21)
    n, K = 500, TAUS.size
    y = rng.standard_normal(n)
    F = rng.standard_normal((n, K))          # unsorted on purpose
    assert (np.diff(F, axis=1) < 0).any(), "F must actually cross"
    c = rng.standard_normal(K)
    w = rng.uniform(0.5, 1.5, n)
    out = np.empty(n)
    _project_pinball(y, F, c, float(c @ TAUS), w, out)
    grad = np.where(y[:, None] >= F, -TAUS, 1.0 - TAUS)
    np.testing.assert_allclose(out, (grad @ c) * w, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# The non-crossing guarantee
# --------------------------------------------------------------------------

def test_predictions_never_cross():
    """Zero crossings, asserted exactly: every delivered row is rearranged."""
    X, y = _heteroscedastic(seed=1)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=200).fit(X, y)
    Q = m.predict(X)
    assert Q.shape == (len(y), TAUS.size)
    assert np.all(np.diff(Q, axis=1) >= 0.0)
    assert qm.crossing_rate(Q) == 0.0


def test_no_crossing_on_every_path():
    """Staged stages, the tiny-batch serial predict kernel, the weighted and
    subsampled paths, and the conformalized output all inherit the guarantee."""
    X, y = _heteroscedastic(n=2500, seed=2)
    rng = np.random.default_rng(0)
    m = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=120, subsample=0.7).fit(
            X, y, sample_weight=rng.uniform(0.5, 1.5, size=len(y)))
    for stage in m.staged_predict(X[:64]):
        assert np.all(np.diff(stage, axis=1) >= 0.0)
    for k in (1, 2, 4, 5, 64):          # straddles the serial/parallel switch
        assert np.all(np.diff(m.predict(X[:k]), axis=1) >= 0.0)

    mc = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=120, conformalize=True).fit(X, y)
    assert np.all(np.diff(mc.predict(X), axis=1) >= 0.0)


def test_narrow_regions_are_actually_narrower():
    """The band must track the LOCAL spread, not just beat the pooled one.

    The data has a 0.1 spread on one side and 2.1 on the other, so the honest
    width ratio is about 0.05. The predecessor to today's construction spent a
    global narrowing budget at its worst-case leaf, froze within tens of rounds
    and delivered bands 2x to 10x too wide on real data
    (benchmarks/LEAFTUNE_PLAN.md, P12) -- while still clearing a loose
    "narrower than pooled" bar. Hence the ratio, which is what actually
    failed."""
    X, y = _heteroscedastic(n=6000, seed=3, narrow=True)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=400).fit(X, y)
    Q = m.predict(X)
    width = Q[:, -1] - Q[:, 0]              # 5%-95% width
    tight, wide = X[:, 1] <= 0, X[:, 1] > 0
    pooled = float(np.quantile(y, 0.95) - np.quantile(y, 0.05))
    assert width[tight].mean() < 0.5 * pooled, (
        f"tight-region width {width[tight].mean():.3f} should be far below the "
        f"pooled width {pooled:.3f}")
    ratio = width[tight].mean() / width[wide].mean()
    assert ratio < 0.35, (
        f"tight/wide width ratio {ratio:.3f} should approach the true spread "
        f"ratio of ~0.05; a frozen global band sits near 1.0")


def test_coverage_does_not_ratchet_up_with_training():
    """The saturation signature, pinned as a regression test.

    When the width is frozen but the centre keeps sharpening, empirical
    coverage climbs AWAY from nominal as rounds accumulate -- P12 measured
    0.817 at round 1 and 0.995 from round 50. A healthy fit does the opposite:
    coverage settles toward nominal and the width keeps moving."""
    X, y = _heteroscedastic(n=6000, seed=5, narrow=True)
    Xt, yt, Xv, yv = X[:4000], y[:4000], X[4000:], y[4000:]
    m = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=400, early_stopping=False).fit(Xt, yt)
    lo, hi = 0, TAUS.size - 1               # the 0.05-0.95 band, nominal 0.90
    seen = {}
    for i, Q in enumerate(m.staged_predict(Xv), start=1):
        if i in (50, 400):
            seen[i] = (float(np.mean((yv >= Q[:, lo]) & (yv <= Q[:, hi]))),
                       float(np.mean(Q[:, hi] - Q[:, lo])))
    (cov50, wid50), (cov400, wid400) = seen[50], seen[400]

    assert abs(wid400 - wid50) / wid50 > 0.01, (
        f"width froze: {wid50:.5f} at round 50 vs {wid400:.5f} at 400")
    assert cov400 <= cov50 + 0.02, (
        f"coverage ratcheted up with training ({cov50:.3f} -> {cov400:.3f}), "
        f"which is the frozen-band signature")
    assert 0.84 <= cov400 <= 0.95, (
        f"final coverage {cov400:.3f} should sit near the nominal 0.90")


def test_rearrangement_never_increases_pinball():
    """Sorting a crossing quantile curve cannot raise the pinball loss at any
    level (Chernozhukov, Fernandez-Val & Galichon 2010). That is the whole
    licence for repairing order at delivery, so check it against the model's
    own raw accumulated scores."""
    X, y = _heteroscedastic(n=3000, seed=6)
    m = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=150, early_stopping=False).fit(X, y)

    Xb = np.ascontiguousarray(m.model_.prep_.transform(X).T)
    raw = np.tile(m.model_.init_, (X.shape[0], 1))
    for tree in m.model_.trees_:
        raw += tree.values[tree.apply(Xb)]
    sorted_out = m.predict(X)

    np.testing.assert_array_equal(sorted_out, np.sort(raw, axis=1))
    assert _pinball(y, sorted_out) <= _pinball(y, raw) + 1e-12


def test_staged_final_equals_predict():
    X, y = _heteroscedastic(n=1200, seed=4)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=60).fit(X, y)
    stages = list(m.staged_predict(X[:50]))
    assert len(stages) == m.best_iteration_
    np.testing.assert_array_equal(stages[-1], m.predict(X[:50]))


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------

def test_matches_scalar_quantile_regressor():
    """Per level, one shared tree structure must stay close to the library's
    own single-quantile regressors -- the in-repo stand-in for the LightGBM
    comparison, so the suite needs no optional dependency.

    Close, not better: K per-level models get K structures specialized to
    their own level, and the whole point of this head is to give that up for a
    K-fold cut in split-search work. The band below is what that trade costs
    on ordinary heteroscedastic data; benchmarks/quantile_head.py measures it
    against LightGBM properly."""
    X, y = _heteroscedastic(n=4000, seed=5, narrow=False)
    Xt, yt, Xv, yv = X[:3000], y[:3000], X[3000:], y[3000:]
    levels = [0.1, 0.5, 0.9]
    m = ChimeraBoostQuantileRegressor(
        quantiles=levels, random_state=0, n_estimators=300).fit(Xt, yt)
    Q = m.predict(Xv)
    for k, a in enumerate(levels):
        s = ChimeraBoostRegressor(loss="Quantile", alpha=a, random_state=0,
                                  n_estimators=300, depth=4).fit(Xt, yt)
        r_s = yv - s.predict(Xv)
        r_m = yv - Q[:, k]
        pb_s = float(np.maximum(a * r_s, (a - 1) * r_s).mean())
        pb_m = float(np.maximum(a * r_m, (a - 1) * r_m).mean())
        assert pb_m < 1.10 * pb_s, (
            f"tau={a}: shared-tree pinball {pb_m:.5f} vs per-level "
            f"{pb_s:.5f}")


def test_projection_arms_agree_with_exact_gain():
    """The default's whole justification: a rank-1 projection of the gradient
    picks structure about as good as scoring the exact summed-across-level
    gain, which costs K histogram channels."""
    X, y = _heteroscedastic(n=2500, seed=6)
    Xt, yt, Xv, yv = X[:1800], y[:1800], X[1800:], y[1800:]
    kw = dict(random_state=0, n_estimators=200, early_stopping=False)
    rot = ChimeraBoostQuantileRegressor(**kw).fit(Xt, yt)
    exa = ChimeraBoostQuantileRegressor(exact_splits=True, **kw).fit(Xt, yt)
    p_rot, p_exa = _pinball(yv, rot.predict(Xv)), _pinball(yv, exa.predict(Xv))
    assert p_rot < 1.10 * p_exa, f"rotate {p_rot:.5f} vs exact {p_exa:.5f}"


def test_sum_projection_is_blind_to_pure_spread():
    """The literal channel sum cancels on a symmetric grid when only the
    spread varies, which is why it is not the default. Recorded as a test so
    the reasoning cannot quietly stop being true."""
    rng = np.random.default_rng(9)
    n = 6000
    X = rng.standard_normal((n, 6))
    y = (0.3 + 2.7 * (X[:, 1] > 0)) * rng.standard_normal(n)   # centre is 0
    Xt, yt, Xv, yv = X[:4000], y[:4000], X[4000:], y[4000:]
    kw = dict(random_state=0, n_estimators=300, early_stopping=False)
    p_sum = _pinball(yv, ChimeraBoostQuantileRegressor(
        split_projection="sum", **kw).fit(Xt, yt).predict(Xv))
    p_rot = _pinball(yv, ChimeraBoostQuantileRegressor(**kw).fit(Xt, yt).predict(Xv))
    assert p_rot < p_sum


def test_learns_the_conditional_median():
    X, y = _heteroscedastic(n=3000, seed=8)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=300).fit(X, y)
    med = m.predict(X)[:, TAUS.size // 2]
    assert float(np.corrcoef(med, 2.0 * X[:, 0])[0, 1]) > 0.95


# --------------------------------------------------------------------------
# Conformalization
# --------------------------------------------------------------------------

def test_conformal_coverage_is_near_nominal():
    """CQR on a fold that saw no training, no early stopping and no model
    selection should land every interval within ~2 points of nominal at
    n=10k.

    Scored on 20k fresh rows so that test-set sampling noise (about 0.9 points
    on a 2k set) is not what the tolerance is measuring."""
    X, y = _heteroscedastic(n=30000, p=10, seed=10, narrow=False)
    Xt, yt, Xv, yv = X[:10000], y[:10000], X[10000:], y[10000:]
    m = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=400, conformalize=True).fit(Xt, yt)
    for iv in qm.interval_coverage(yv, m.predict(Xv), m.quantiles_):
        assert abs(iv["coverage"] - iv["nominal"]) < 0.02, iv


def test_conformal_hits_nominal_exactly_on_its_own_fold():
    """The finite-sample guarantee itself, isolated from generalization: the
    calibrated interval covers its own calibration fold at the nominal rate,
    by construction of the conformal rank. Any test-set shortfall is variance
    in how well that fold represents the next one, not a broken calibration --
    worth separating, because the two get confused."""
    rng = np.random.default_rng(30)
    n = 8000
    Q = np.sort(rng.standard_normal((n, TAUS.size)) * 0.1
                + 3.0 * np.tile(TAUS - 0.5, (n, 1)), axis=1)
    y = rng.standard_normal(n)
    mi, mw = _median_index(TAUS)
    s = _cqr_scales(Q, y, TAUS, mi, mw)
    c = Q[:, mi][:, None]
    for iv in qm.interval_coverage(y, c + s[None, :] * (Q - c), TAUS):
        assert iv["coverage"] >= iv["nominal"] - 1e-9, iv


def test_conformal_scale_is_ordered_and_nonnegative():
    X, y = _heteroscedastic(n=4000, seed=11)
    m = ChimeraBoostQuantileRegressor(
        random_state=0, n_estimators=150, conformalize=True).fit(X, y)
    s = m.conformal_scale_
    assert np.all(s >= 0.0)
    half = TAUS.size // 2
    # Outer intervals must carry a factor at least as large as inner ones.
    assert np.all(np.diff(s[:half]) <= 1e-12)


def test_conformalize_fails_loudly_when_the_fold_is_too_small():
    X, y = _heteroscedastic(n=200, seed=12)
    with pytest.raises(ValueError, match="calibration"):
        ChimeraBoostQuantileRegressor(
            quantiles=[0.001, 0.5, 0.999], conformalize=True,
            calibration_fraction=0.02, n_estimators=10,
            random_state=0).fit(X, y)


def test_cqr_scales_shrink_an_overdispersed_model():
    """A model whose intervals are twice too wide should calibrate to a factor
    near one half."""
    rng = np.random.default_rng(13)
    n = 20000
    y = rng.standard_normal(n)
    from scipy.stats import norm
    Q = 2.0 * np.tile(norm.ppf(TAUS), (n, 1))     # exactly 2x too wide
    mi, mw = _median_index(TAUS)
    s = _cqr_scales(Q, y, TAUS, mi, mw)
    assert 0.45 < s[0] < 0.55, s


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def test_predict_kinds():
    X, y = _heteroscedastic(n=1500, seed=14)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=80).fit(X, y)
    assert m.predict(X, kind="quantiles").shape == (len(y), TAUS.size)
    iv = m.predict(X, kind="interval", alpha=0.1)
    assert iv.shape == (len(y), 2)
    np.testing.assert_array_equal(iv[:, 0], m.predict(X)[:, 0])
    np.testing.assert_array_equal(iv[:, 1], m.predict(X)[:, -1])
    mean = m.predict(X, kind="mean")
    assert mean.shape == (len(y),)
    Q = m.predict(X)
    assert np.all(mean >= Q[:, 0]) and np.all(mean <= Q[:, -1])


def test_mean_of_a_symmetric_grid_matches_the_trapezoid_by_hand():
    """The flat-tail extension is a documented choice, so pin the arithmetic."""
    X, y = _heteroscedastic(n=800, seed=15)
    m = ChimeraBoostQuantileRegressor(quantiles=[0.25, 0.5, 0.75],
                                      random_state=0, n_estimators=40).fit(X, y)
    Q = m.predict(X)
    taus = m.quantiles_
    hand = (np.trapezoid(Q, x=taus, axis=1) if hasattr(np, "trapezoid")
            else np.trapz(Q, x=taus, axis=1)) + 0.25 * Q[:, 0] + 0.25 * Q[:, -1]
    np.testing.assert_allclose(m.predict(X, kind="mean"), hand, rtol=0,
                               atol=1e-12)


def test_pickle_round_trip():
    X, y = _heteroscedastic(n=1200, seed=16)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=60,
                                      conformalize=True).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.predict(X), m.predict(X))


def test_categorical_and_nan_are_handled():
    rng = np.random.default_rng(17)
    n = 1500
    num = rng.standard_normal((n, 3))
    cat = rng.integers(0, 6, size=n).astype(np.float64)
    X = np.column_stack([num, cat])
    y = num[:, 0] + cat * 0.4 + rng.standard_normal(n) * 0.5
    X[::50, 0] = np.nan
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=80).fit(
        X, y, cat_features=[3])
    Q = m.predict(X)
    assert np.isfinite(Q).all()
    assert np.all(np.diff(Q, axis=1) >= 0.0)


def test_reuses_the_shared_validation_helpers():
    X, y = _heteroscedastic(n=600, seed=18)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=30)
    with pytest.raises(Exception):
        m.predict(X)                       # not fitted
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :3])                # wrong feature count
    with pytest.raises(ValueError):
        ChimeraBoostQuantileRegressor(depth=99).fit(X, y)
    with pytest.raises(ValueError):
        ChimeraBoostQuantileRegressor(n_estimators=30).fit(
            X, y, sample_weight=np.full(len(y), -1.0))


@pytest.mark.parametrize("bad", [[0.5, 0.1], [0.0, 0.5], [0.5, 1.0],
                                 [0.3, 0.3], []])
def test_quantile_grid_validation(bad):
    X, y = _heteroscedastic(n=300, seed=19)
    with pytest.raises(ValueError):
        ChimeraBoostQuantileRegressor(quantiles=bad, n_estimators=10).fit(X, y)


def test_single_quantile_grid_reduces_cleanly():
    """K=1 has no spread or skew contrast and no interval; it must still fit."""
    X, y = _heteroscedastic(n=1000, seed=20)
    m = ChimeraBoostQuantileRegressor(quantiles=[0.5], random_state=0,
                                      n_estimators=60).fit(X, y)
    Q = m.predict(X)
    assert Q.shape == (len(y), 1)
    assert np.isfinite(m.predict(X, kind="mean")).all()


def test_fixed_contrasts_are_orthonormal():
    B = np.stack(_fixed_contrasts(TAUS), axis=1)
    np.testing.assert_allclose(B.T @ B, np.eye(B.shape[1]), rtol=0, atol=1e-12)
    # The first contrast is the plain channel sum, up to normalization.
    np.testing.assert_allclose(B[:, 0], np.ones(TAUS.size) / np.sqrt(TAUS.size),
                               rtol=0, atol=1e-12)


def test_loss_reduces_to_the_scalar_quantile_gradient():
    """A one-level grid must reproduce `losses.Quantile` term for term."""
    from chimeraboost.losses import Quantile
    rng = np.random.default_rng(21)
    y = rng.standard_normal(500)
    raw = rng.standard_normal(500)
    mq = MultiQuantile(np.array([0.3]))
    sq = Quantile(0.3)
    g_m, h_m = mq.grad_hess(y, raw[:, None])
    g_s, h_s = sq.grad_hess(y, raw)
    np.testing.assert_array_equal(g_m[:, 0], g_s)
    np.testing.assert_array_equal(h_m[:, 0], h_s)
    np.testing.assert_allclose(mq.eval(y, raw[:, None]), sq.eval(y, raw),
                               rtol=0, atol=1e-12)


def test_booster_keeps_no_narrowing_budget():
    """The budget is gone and must stay gone: `gap_` was its public trace, and
    a half-reverted state that recreates the attribute without the machinery
    (or vice versa) would be worse than either. The raw accumulated scores are
    free to cross -- that is the point -- while every delivered row is
    ordered."""
    X, y = _heteroscedastic(n=3000, seed=22)
    b = MultiQuantileBoosting(quantiles=TAUS, n_estimators=200, depth=4,
                              random_state=0, min_child_weight=20.0)
    b.fit(X, y)
    assert not hasattr(b, "gap_")
    assert np.all(np.diff(b.predict_raw(X), axis=1) >= 0.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_pinball_and_crps():
    y = np.array([0.0, 1.0])
    Q = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    taus = np.array([0.25, 0.5, 0.75])
    per = qm.pinball_loss(y, Q, taus)
    # row 0: r = 0-(-1)=1 -> .25*1 ; 0-0=0 -> 0 ; 0-1=-1 -> .25*1
    # row 1: r = 2 -> .5 ; 1 -> .5 ; 0 -> 0
    np.testing.assert_allclose(per, [(0.25 + 0.5) / 2, (0.0 + 0.5) / 2,
                                     (0.25 + 0.0) / 2])
    np.testing.assert_allclose(qm.crps(y, Q, taus), per.mean())


def test_interval_coverage_and_width():
    y = np.array([0.0, 5.0])
    Q = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    taus = np.array([0.1, 0.5, 0.9])
    iv = qm.interval_coverage(y, Q, taus)
    assert len(iv) == 1
    assert iv[0]["nominal"] == pytest.approx(0.8)
    assert iv[0]["coverage"] == pytest.approx(0.5)
    assert iv[0]["width"] == pytest.approx(2.0)


def test_crossing_rate_detects_crossings():
    assert qm.crossing_rate(np.array([[0.0, 1.0, 2.0]])) == 0.0
    assert qm.crossing_rate(np.array([[0.0, 2.0, 1.0]])) == pytest.approx(0.5)


def test_report_and_score():
    X, y = _heteroscedastic(n=1000, seed=23)
    m = ChimeraBoostQuantileRegressor(random_state=0, n_estimators=60).fit(X, y)
    rep = m.report(X, y)
    assert set(rep) == {"crps", "pinball", "taus", "intervals", "crossing_rate"}
    assert rep["crossing_rate"] == 0.0
    assert m.score(X, y) == pytest.approx(-rep["crps"])
    assert isinstance(qm.format_report(rep, "t"), str)


@pytest.mark.parametrize("grid, want", [
    ([0.1, 0.5, 0.9], 10),          # 1.0 - 0.9 is a few ulps under 0.1
    ([0.2, 0.8], 5),                # so is 1.0 - 0.8
    ([0.05, 0.5, 0.95], 20),
    (list(DEFAULT_QUANTILES), 20),  # matches LightGBM's min_data_in_leaf
    ([0.25, 0.75], 4),
    ([0.01, 0.99], 100),
    ([0.001, 0.999], 1000),
    ([0.3, 0.7], 4),                # 1/0.3 is GENUINELY 3.33 -- must round up
    ([0.15, 0.85], 7),              # likewise 6.67
])
def test_auto_min_child_weight_is_exact_on_symmetric_grids(grid, want):
    """The floor is 1/edge rounded up. Computing `edge` as `1.0 - taus[-1]`
    loses a couple of ulps, which used to push the reciprocal just past a whole
    number and cost an extra row of leaf floor -- 11 instead of 10 for the
    (0.1, 0.5, 0.9) grid. Fractional edges must still round up."""
    assert _auto_min_child_weight(np.asarray(grid)) == want


def test_auto_min_child_weight_unrounded_grid_matches_rounded():
    """A user building the default grid by arithmetic rather than by
    `np.round` lands on 0.9500000000000001, which must not change the floor."""
    unrounded = np.array([0.05 + 0.05 * i for i in range(19)])
    assert (_auto_min_child_weight(unrounded)
            == _auto_min_child_weight(DEFAULT_QUANTILES) == 20)


def test_metric_shape_errors():
    y = np.zeros(3)
    with pytest.raises(ValueError):
        qm.pinball_loss(y, np.zeros((3, 2)), np.array([0.5]))
    with pytest.raises(ValueError):
        qm.pinball_loss(y, np.zeros((2, 2)), np.array([0.25, 0.75]))
    with pytest.raises(ValueError):
        qm.pinball_loss(y, np.zeros(3), np.array([0.5]))
