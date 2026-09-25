"""Random slopes, slice 2 (#113): solver unit tests plus estimator integration.

The model is ``y = F(X) + b_g + a_g * z + eps`` with independent
``b ~ N(0, s2b)`` and ``a ~ N(0, s2a)``, ``z = x - centre``. Solver oracles
are the textbook matrix forms on tiny data: posterior mean via an explicit
inverse, REML via explicit log-determinants. The library path must match
both, since the O(G) sufficient-statistic forms are only algebra away.
"""

import pickle

import numpy as np
import pytest

from chimeraboost import ChimeraBoostRegressor
from chimeraboost.random_effects import (
    _slope_reml_from_stats,
    _slope_suff_stats,
    codes_for_labels,
    estimate_ratio_reml,
    estimate_slope_ratios_reml,
    solve_intercepts,
    solve_slopes,
)


def _brute_posterior_mean(resid, codes, n_groups, z, s2b, s2a, s2e=1.0):
    n = resid.shape[0]
    zb = np.zeros((n, n_groups))
    zb[np.arange(n), codes] = 1.0
    za = zb * z[:, None]
    big_z = np.concatenate([zb, za], axis=1)
    prior = np.diag([s2b] * n_groups + [s2a] * n_groups)
    v = big_z @ prior @ big_z.T + s2e * np.eye(n)
    u = prior @ big_z.T @ np.linalg.solve(v, resid)
    return u[:n_groups], u[n_groups:]


def _brute_reml_profile(resid, codes, n_groups, z, ratio_b, ratio_a):
    """-2 REML at one ratio pair, explicit matrices, s2e profiled out."""
    n = resid.shape[0]
    zb = np.zeros((n, n_groups))
    zb[np.arange(n), codes] = 1.0
    za = zb * z[:, None]
    sigma = zb @ zb.T / ratio_b + za @ za.T / ratio_a + np.eye(n)
    sinv = np.linalg.inv(sigma)
    x = np.column_stack([np.ones(n), z])
    xtx = x.T @ sinv @ x
    xtr = x.T @ sinv @ resid
    qc = resid @ sinv @ resid - xtr @ np.linalg.solve(xtx, xtr)
    _, logdet = np.linalg.slogdet(sigma)
    _, logdetx = np.linalg.slogdet(xtx)
    return (n - 2) * np.log(qc) + logdet + logdetx


def _grouped_resid(seed=0, n_groups=6, lo=2, hi=12):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(lo, hi + 1, size=n_groups)
    codes = np.repeat(np.arange(n_groups), sizes)
    resid = rng.normal(size=codes.shape[0])
    z = rng.normal(size=codes.shape[0])
    z[3] = np.nan  # the missing-x rule (z = 0) rides along in every oracle
    return resid, codes, n_groups, z


def test_solve_matches_brute_force_posterior_mean():
    resid, codes, n_groups, z = _grouped_resid()
    zc = np.where(np.isnan(z), 0.0, z)
    for s2b, s2a, s2e in [(2.0, 1.0, 1.0), (0.25, 3.0, 1.0),
                          (1.0, 0.5, 2.0)]:
        b, a = solve_slopes(resid, codes, n_groups, s2e / s2b, s2e / s2a, z)
        want_b, want_a = _brute_posterior_mean(resid, codes, n_groups, zc,
                                               s2b, s2a, s2e)
        np.testing.assert_allclose(b, want_b, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(a, want_a, rtol=1e-10, atol=1e-10)


def test_solve_weighted_gradient_is_zero():
    resid, codes, n_groups, z = _grouped_resid(seed=1)
    zc = np.where(np.isnan(z), 0.0, z)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, size=resid.shape[0])
    lb, la = 1.5, 0.7
    b, a = solve_slopes(resid, codes, n_groups, lb, la, z, weights=w)
    # Defining property: gradient of the penalized weighted objective
    # (factor 2 dropped) is zero at the solution.
    s_w = np.bincount(codes, weights=w, minlength=n_groups)
    s_z = np.bincount(codes, weights=w * zc, minlength=n_groups)
    s_zz = np.bincount(codes, weights=w * zc * zc, minlength=n_groups)
    s_r = np.bincount(codes, weights=w * resid, minlength=n_groups)
    s_zr = np.bincount(codes, weights=w * zc * resid, minlength=n_groups)
    grad_b = (s_r - b * s_w - a * s_z) - lb * b
    grad_a = (s_zr - b * s_z - a * s_zz) - la * a
    np.testing.assert_allclose(grad_b, np.zeros(n_groups), atol=1e-10)
    np.testing.assert_allclose(grad_a, np.zeros(n_groups), atol=1e-10)


def test_inf_slope_ratio_reduces_to_intercepts():
    resid, codes, n_groups, z = _grouped_resid(seed=2)
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 2.0, size=resid.shape[0])
    for weights in (None, w):
        b, a = solve_slopes(resid, codes, n_groups, 1.5, np.inf, z,
                            weights=weights)
        want = solve_intercepts(resid, codes, n_groups, 1.5,
                                weights=weights)
        np.testing.assert_allclose(b, want, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(a, np.zeros(n_groups))


def test_reml_matches_brute_force_up_to_constant():
    rng = np.random.default_rng(3)
    sizes = rng.integers(2, 13, size=6)
    codes = np.repeat(np.arange(6), sizes)
    z = rng.normal(size=codes.shape[0])
    z = z - z.mean()
    # Plant group signal for an interior optimum at both ratios.
    resid = (rng.normal(size=codes.shape[0])
             + rng.normal(0.0, 2.0, size=6)[codes]
             + rng.normal(0.0, 1.0, size=6)[codes] * z)
    stats = _slope_suff_stats(resid, codes, 6, z, None)
    pairs = [(0.5, 2.0), (3.0, 0.7), (10.0, 10.0), (0.1, 0.1)]
    got = np.array([_slope_reml_from_stats(*stats, np.log10(pb),
                                           np.log10(pa))
                    for pb, pa in pairs])
    want = np.array([_brute_reml_profile(resid, codes, 6, z, pb, pa)
                     for pb, pa in pairs])
    # Up to the additive constant: differences from the first pair agree.
    np.testing.assert_allclose(got - got[0], want - want[0], rtol=1e-8,
                               atol=1e-8)


def test_reml_recovers_known_ratios():
    rng = np.random.default_rng(4)
    n_groups, per = 60, 25
    s2b, s2a, s2e = 4.0, 1.0, 1.0
    b = rng.normal(0.0, np.sqrt(s2b), size=n_groups)
    a = rng.normal(0.0, np.sqrt(s2a), size=n_groups)
    codes = np.repeat(np.arange(n_groups), per)
    z = rng.normal(size=codes.shape[0])
    resid = b[codes] + a[codes] * z + rng.normal(0.0, 1.0,
                                                size=codes.shape[0])
    got_b, got_a = estimate_slope_ratios_reml(resid, codes, n_groups, z)
    assert abs(np.log10(got_b) - np.log10(s2e / s2b)) < 0.3
    assert abs(np.log10(got_a) - np.log10(s2e / s2a)) < 0.3


def test_reml_degenerate_inputs_return_inf():
    r = np.array([1.0, 2.0, 3.0])
    z = np.array([0.5, -0.5, 0.0])
    # One group: the fixed effects absorb everything; nothing to separate.
    assert estimate_slope_ratios_reml(r, np.array([0, 0, 0]), 1, z) == (
        np.inf, np.inf)
    # All singletons: no within-group information at all.
    assert estimate_slope_ratios_reml(r, np.array([0, 1, 2]), 3, z) == (
        np.inf, np.inf)
    # x constant within every group: slopes have no support (inf), while
    # the intercept ratio is slice 1's answer on the same data.
    rc = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cc = np.array([0, 0, 1, 1, 2, 2])
    xc = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    got_b, got_a = estimate_slope_ratios_reml(rc, cc, 3, xc - xc.mean())
    assert got_a == np.inf
    assert got_b == estimate_ratio_reml(rc, cc, 3)


def test_out_of_range_codes_raise_cleanly():
    r = np.array([1.0, 2.0, 3.0])
    z = np.array([0.5, -0.5, 0.0])
    with pytest.raises(ValueError, match="out of range"):
        estimate_slope_ratios_reml(r, np.array([0, 1, 5]), 3, z)
    with pytest.raises(ValueError, match="out of range"):
        solve_slopes(r, np.array([0, -1, 1]), 3, 1.0, 1.0, z)


# --- estimator integration -----------------------------------------------


def _sloped_data(seed=0, n_groups=25, per=40, group_sd=3.0, slope_sd=2.0,
                 noise=1.0):
    """y = X @ beta + b_g + a_g * (x0 - mean) + eps, truth known."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    codes = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(n, 5))
    beta = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    b_true = rng.normal(0.0, group_sd, size=n_groups)
    a_true = rng.normal(0.0, slope_sd, size=n_groups)
    xc = X[:, 0]
    y = (X @ beta + b_true[codes] + a_true[codes] * (xc - xc.mean())
         + rng.normal(0.0, noise, size=n))
    return X, y, codes, b_true, a_true


def _fit_rs(X, y, groups, fit_kw=None, **kw):
    params = dict(n_estimators=300, random_state=0, random_effects=True,
                  random_slopes=[0])
    params.update(kw)
    return ChimeraBoostRegressor(**params).fit(X, y, groups=groups,
                                              **(fit_kw or {}))


def _hand_predict(m, X, groups):
    """Trees-only prediction plus hand-built random effects.

    Uses only the fitted attributes (``group_intercepts_``,
    ``group_slopes_``, ``group_slope_center_``, ``group_slope_range_``),
    never the ``groups=...`` predict path -- so it disagrees with
    ``predict`` exactly when ``predict`` drops the slope term. The
    operation order mirrors ``_group_offsets`` + ``predict``, so equality
    with ``predict`` is bit-exact, not approximate.
    """
    base = m.predict(X)
    xc = np.asarray(X)[:, 0]
    codes = codes_for_labels(groups, m.group_labels_)
    n = base.shape[0]
    off = np.zeros(n, dtype=np.float64)
    seen = codes >= 0
    off[seen] = m.group_intercepts_[codes[seen]]
    lo, hi = m.group_slope_range_
    xclip = np.clip(xc, lo, hi)
    have_x = seen & ~np.isnan(xclip)
    off[have_x] += (m.group_slopes_[codes[have_x]]
                    * (xclip[have_x] - m.group_slope_center_))
    return base + off


def test_slopes_recover_truth_and_win_on_seen_groups():
    X, y, codes, b_true, a_true = _sloped_data()
    m = _fit_rs(X, y, codes)
    assert np.corrcoef(m.group_slopes_, a_true)[0, 1] > 0.95
    assert np.corrcoef(m.group_intercepts_, b_true)[0, 1] > 0.95

    rng = np.random.default_rng(1)
    n_te = 600
    te_codes = rng.integers(0, 25, size=n_te)
    Xte = rng.normal(size=(n_te, 5))
    beta = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    yte = (Xte @ beta + b_true[te_codes]
           + a_true[te_codes] * (Xte[:, 0] - X[:, 0].mean())
           + rng.normal(size=n_te))
    rmse_rs = np.sqrt(np.mean((m.predict(Xte, groups=te_codes) - yte) ** 2))
    m_re = ChimeraBoostRegressor(n_estimators=300, random_state=0,
                                 random_effects=True).fit(X, y, groups=codes)
    rmse_re = np.sqrt(
        np.mean((m_re.predict(Xte, groups=te_codes) - yte) ** 2))
    assert rmse_rs < rmse_re


def test_predict_matches_hand_built_prediction_exactly():
    # The #113 gate finding: predict skipped the slope term entirely, so
    # this hand-built comparison (which never touches the groups=...
    # predict path) is what would have caught it. Seen rows only, with
    # some x beyond the fit range and some NaN x.
    X, y, codes, _, _ = _sloped_data()
    m = _fit_rs(X, y, codes)
    rng = np.random.default_rng(11)
    n_te = 300
    te_codes = rng.integers(0, 25, size=n_te)
    Xte = rng.normal(size=(n_te, 5))
    lo, hi = m.group_slope_range_
    Xte[:40, 0] = hi + 50.0
    Xte[40:80, 0] = lo - 50.0
    Xte[80:120, 0] = np.nan
    np.testing.assert_array_equal(
        m.predict(Xte, groups=te_codes), _hand_predict(m, Xte, te_codes))


def test_planted_slopes_win_by_twenty_percent_on_seen_rows():
    # Per-group slope sd 2.0 (>= 1.5) on the N(0, 1) column X0: fitting
    # the slopes must cut the seen-row RMSE by at least 20% against
    # intercepts alone on the same data.
    X, y, codes, _, _ = _sloped_data(slope_sd=2.0)
    m = _fit_rs(X, y, codes)
    m_re = ChimeraBoostRegressor(n_estimators=300, random_state=0,
                                 random_effects=True).fit(X, y, groups=codes)
    rmse_rs = np.sqrt(np.mean((m.predict(X, groups=codes) - y) ** 2))
    rmse_re = np.sqrt(
        np.mean((m_re.predict(X, groups=codes) - y) ** 2))
    assert rmse_rs < 0.8 * rmse_re


def test_predict_differs_from_intercepts_where_slope_nonzero():
    X, y, codes, _, _ = _sloped_data()
    m = _fit_rs(X, y, codes)
    got = m.predict(X, groups=codes)
    base = m.predict(X)
    cc = codes_for_labels(codes, m.group_labels_)
    inter_only = base + m.group_intercepts_[cc]
    lo, hi = m.group_slope_range_
    slope_term = (m.group_slopes_[cc]
                  * (np.clip(X[:, 0], lo, hi) - m.group_slope_center_))
    nz = slope_term != 0.0
    assert nz.any()
    assert np.all(got[nz] != inter_only[nz])
    np.testing.assert_array_equal(got[~nz], inter_only[~nz])


def test_null_slopes_shrink_toward_zero():
    X, y, codes, _, _ = _sloped_data(seed=7, slope_sd=0.0)
    m = _fit_rs(X, y, codes)
    assert np.abs(m.group_slopes_).max() < 1e-3
    assert m.group_slope_ratio_ > 1e3


def test_unseen_nan_and_clip():
    X, y, codes, _, _ = _sloped_data()
    m = _fit_rs(X, y, codes)
    rng = np.random.default_rng(2)
    n_te = 200
    te_codes = rng.integers(0, 25, size=n_te)
    Xte = rng.normal(size=(n_te, 5))

    # Unseen groups: exactly the trees-only prediction.
    f_only = m.predict(Xte)
    np.testing.assert_array_equal(
        m.predict(Xte, groups=np.full(n_te, "never-seen")), f_only)
    # predict_raw stays trees-only (RMSE: identity link, no offset).
    np.testing.assert_array_equal(f_only, m.predict_raw(Xte))

    # A NaN x carries no slope term: exactly the hand-built prediction
    # (trees + intercepts, no slope on NaN rows) -- not another path that
    # also skips the slope.
    Xnan = Xte.copy()
    Xnan[:, 0] = np.nan
    np.testing.assert_array_equal(m.predict(Xnan, groups=te_codes),
                                  _hand_predict(m, Xnan, te_codes))

    # x beyond the fit range is clipped to it: predict at far-out x equals
    # the hand-built prediction at the range edge (trees bin both the
    # same, so only the clipped slope term can differ).
    lo, hi = m.group_slope_range_
    for wide, edge in ((hi + 50.0, hi), (lo - 50.0, lo)):
        Xwide = Xte.copy()
        Xwide[:, 0] = wide
        Xedge = Xte.copy()
        Xedge[:, 0] = edge
        np.testing.assert_array_equal(
            m.predict(Xwide, groups=te_codes),
            _hand_predict(m, Xedge, te_codes))


def test_none_is_bit_identical_to_no_slopes():
    X, y, codes, _, _ = _sloped_data()
    plain_re = ChimeraBoostRegressor(n_estimators=300, random_state=0,
                                     random_effects=True)
    plain_re.fit(X, y, groups=codes)
    m = ChimeraBoostRegressor(n_estimators=300, random_state=0,
                              random_effects=True,
                              random_slopes=None)
    m.fit(X, y, groups=codes)
    np.testing.assert_array_equal(m.predict(X, groups=codes),
                                  plain_re.predict(X, groups=codes))
    np.testing.assert_array_equal(m.group_intercepts_,
                                  plain_re.group_intercepts_)
    assert m.group_ratio_ == plain_re.group_ratio_
    assert m.group_slopes_ is None
    assert m.group_slope_ratio_ is None
    assert m.group_slope_center_ is None
    assert m.group_slope_range_ is None


def test_error_paths():
    X, y, codes, _, _ = _sloped_data()
    with pytest.raises(ValueError, match="random_effects=True"):
        ChimeraBoostRegressor(random_slopes=[0]).fit(X, y)
    with pytest.raises(ValueError, match="exactly one"):
        ChimeraBoostRegressor(random_effects=True,
                              random_slopes=[0, 1]).fit(X, y, groups=codes)
    with pytest.raises(ValueError, match="categorical"):
        ChimeraBoostRegressor(random_effects=True,
                              random_slopes=[1]).fit(
            X, y, groups=codes, cat_features=[1])
    with pytest.raises(ValueError, match="out of range"):
        ChimeraBoostRegressor(random_effects=True,
                              random_slopes=[99]).fit(X, y, groups=codes)
    with pytest.raises(ValueError, match="no column names"):
        ChimeraBoostRegressor(random_effects=True,
                              random_slopes=["x0"]).fit(X, y, groups=codes)


def test_feature_names_and_object_path():
    pd = pytest.importorskip("pandas")
    X, y, codes, _, _ = _sloped_data(n_groups=10, per=30)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["city"] = np.where(np.arange(len(df)) % 2, "a", "b")
    labels = np.array([f"g{c}" for c in codes], dtype=object)
    m = ChimeraBoostRegressor(
        n_estimators=300, random_state=0, random_effects=True,
        random_slopes=["x0"]).fit(df, y, groups=labels, cat_features=["city"])
    assert np.isfinite(m.group_slopes_).all()
    assert np.isfinite(m.predict(df, groups=labels)).all()
    with pytest.raises(ValueError, match="not a column"):
        ChimeraBoostRegressor(
            random_effects=True, random_slopes=["nope"]).fit(
            df, y, groups=labels, cat_features=["city"])


def test_pickle_round_trip_preserves_slopes():
    X, y, codes, _, _ = _sloped_data(n_groups=10, per=30)
    m = _fit_rs(X, y, codes)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.group_slopes_, m.group_slopes_)
    np.testing.assert_array_equal(m2.group_intercepts_,
                                  m.group_intercepts_)
    assert m2.group_slope_ratio_ == m.group_slope_ratio_
    assert m2.group_slope_center_ == m.group_slope_center_
    assert m2.group_slope_range_ == m.group_slope_range_
    np.testing.assert_array_equal(m2.predict(X, groups=codes),
                                  m.predict(X, groups=codes))


def test_deterministic_under_seed():
    X, y, codes, _, _ = _sloped_data(n_groups=10, per=30)
    m1 = _fit_rs(X, y, codes)
    m2 = _fit_rs(X, y, codes)
    np.testing.assert_array_equal(m1.group_slopes_, m2.group_slopes_)
    np.testing.assert_array_equal(m1.group_intercepts_,
                                  m2.group_intercepts_)
    np.testing.assert_array_equal(m1.predict(X, groups=codes),
                                  m2.predict(X, groups=codes))


def test_sample_weights_flow_through():
    X, y, codes, _, _ = _sloped_data(n_groups=10, per=30)
    w = np.ones(len(y))
    w[codes == 0] = 0.0  # ghost group: contributes nothing anywhere
    m = _fit_rs(X, y, codes, fit_kw={"sample_weight": w})
    zero_code = codes_for_labels([0], m.group_labels_)[0]
    assert m.group_intercepts_[zero_code] == 0.0
    assert m.group_slopes_[zero_code] == 0.0


def test_staged_predict_final_matches_predict():
    X, y, codes, _, _ = _sloped_data(n_groups=10, per=30)
    m = _fit_rs(X, y, codes)
    stages = list(m.staged_predict(X, groups=codes))
    # The last stage carries the slopes too, not just the intercepts:
    # it equals the hand-built prediction, which the slope-skipping
    # pass-1 code fails.
    np.testing.assert_allclose(stages[-1], _hand_predict(m, X, codes),
                               rtol=1e-12)
    np.testing.assert_allclose(stages[-1], m.predict(X, groups=codes),
                               rtol=1e-12)
    stages_f = list(m.staged_predict(X))
    np.testing.assert_allclose(stages_f[-1], m.predict(X), rtol=1e-12)
