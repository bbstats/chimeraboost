"""The synthetic quantile screen (Q-B2): oracle correctness, no models.

Every number the screen judges on is excess CRPS over an oracle, so the
oracles have to be right: monotone in tau, calibrated on a large draw, and
-- for the bimodal mixture, whose quantiles come from a root-find -- exact
inversions of the mixture CDF.

Deterministic fixtures only -- no network, no fitting.
"""
import os
import sys

import numpy as np
import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import quantile_synth as qsyn  # noqa: E402

BAND = np.array([0.05, 0.95])


@pytest.mark.parametrize("regime", sorted(qsyn.REGIMES))
def test_oracle_monotone_in_tau(regime):
    X, y, Q = qsyn.REGIMES[regime](5000, 0)
    assert y.shape == (5000,)
    assert Q.shape == (5000, len(qsyn.TAUS))
    assert np.all(np.diff(Q, axis=1) >= 0)


@pytest.mark.parametrize("regime", sorted(qsyn.REGIMES))
def test_oracle_band_covers_90_percent(regime):
    # No model: the oracle's own [0.05, 0.95] band on 200k fresh rows must
    # cover 0.90 within half a point (se ~ 0.0007, so this is ~7 sigma).
    _, y, Q = qsyn.REGIMES[regime](200_000, 1, BAND)
    assert Q.shape == (200_000, 2)
    cov = float(np.mean((y >= Q[:, 0]) & (y <= Q[:, 1])))
    assert abs(cov - 0.90) < 0.005


def test_catscale_has_string_levels_and_varying_spread():
    X, _, Q = qsyn.regime_catscale(2000, 0)
    assert X.shape == (2000, 9)
    levels = X[:, 8]
    assert all(isinstance(v, str) for v in levels)
    # Twenty uniform levels at n=2000: every one is drawn, for sure
    # (missing even one has probability ~1e-43).
    assert len(set(levels)) == 20
    width = Q[:, -1] - Q[:, 0]
    per_level = [width[levels == lv].mean() for lv in set(levels)]
    assert max(per_level) - min(per_level) > 1.0


def test_bimodal_oracle_inverts_its_cdf():
    from scipy.special import expit
    X, _, Q = qsyn.regime_bimodal(2000, 0)
    p = expit(2.0 * X[:, 1])
    mu = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 2]) + 0.5 * X[:, 3] * X[:, 4]
    F = qsyn.bimodal_cdf(Q - mu[:, None], p[:, None])
    want = np.tile(np.asarray(qsyn.TAUS), (F.shape[0], 1))
    np.testing.assert_allclose(F, want, atol=1e-8, rtol=0)


def test_bimodal_oracle_matches_brentq_to_1e_10():
    from scipy.optimize import brentq
    from scipy.special import expit
    X, _, Q = qsyn.regime_bimodal(300, 2)
    p = expit(2.0 * X[:, 1])
    mu = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 2]) + 0.5 * X[:, 3] * X[:, 4]
    taus = np.asarray(qsyn.TAUS)
    for i in range(0, 300, 15):
        for j, tau in enumerate(taus):
            ref = brentq(lambda q: float(qsyn.bimodal_cdf(q, p[i])) - tau,
                          -7.0, 7.0, xtol=1e-13, rtol=1e-13)
            assert abs(Q[i, j] - (mu[i] + ref)) < 1e-10


def test_stream_is_stable_shared_and_nested():
    # Same (regime, n, seed) twice: the identical draw.
    Xa, ya, Qa = qsyn.regime_location(500, 0)
    Xb, yb, Qb = qsyn.regime_location(500, 0)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)
    assert np.array_equal(Qa, Qb)
    # Same (n, seed) across regimes: identical X (numeric columns for
    # catscale, whose 9th column carries the levels).
    Xh, _, _ = qsyn.regime_hetero(500, 0)
    assert np.array_equal(Xa, Xh)
    Xc, _, _ = qsyn.regime_catscale(500, 0)
    assert np.array_equal(np.asarray(Xc[:, :8], dtype=float), Xa)
    # Nested across sizes: the n=500 rows are the n=1000 prefix.
    Xbig, _, _ = qsyn.regime_location(1000, 0)
    assert np.array_equal(Xbig[:500], Xa)


def test_screen_definition_and_location_median():
    assert sorted(qsyn.REGIMES) == ["bimodal", "catscale", "heavy", "hetero",
                                   "location", "skewed"]
    assert tuple(qsyn.SIZES) == (1000, 10000)
    assert qsyn.dataset_key("hetero", 1000) == "qsyn:hetero/n1000"
    # The location median is mu(x) exactly (Phi^-1(0.5) = 0), pinning mu.
    X, _, Q = qsyn.regime_location(500, 0)
    j = int(np.argmin(np.abs(np.asarray(qsyn.TAUS) - 0.5)))
    mu = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 2]) + 0.5 * X[:, 3] * X[:, 4]
    np.testing.assert_array_equal(Q[:, j], mu)
