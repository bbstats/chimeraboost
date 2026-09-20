"""Random intercepts, slice 1 (#109): solver unit tests with brute-force oracles.

The oracle is the textbook matrix form on tiny data: posterior mean via an
explicit inverse, REML via explicit log-determinants over a dense grid. The
library path must match both, since the O(G) sufficient-statistic forms are
only algebra away from them.
"""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostRegressor
from chimeraboost.random_effects import (
    codes_for_labels,
    estimate_ratio_reml,
    solve_intercepts,
)
from chimeraboost.target_encoding import factorize


def _brute_posterior_mean(resid, codes, n_groups, s2b, s2e):
    n = resid.shape[0]
    z = np.zeros((n, n_groups))
    z[np.arange(n), codes] = 1.0
    v = s2b * z @ z.T + s2e * np.eye(n)
    return s2b * z.T @ np.linalg.solve(v, resid)


def _brute_reml_profile(resid, codes, n_groups, ratio):
    """-2 REML at one ratio, explicit matrices, s2e profiled out."""
    n = resid.shape[0]
    z = np.zeros((n, n_groups))
    z[np.arange(n), codes] = 1.0
    s2b, s2e = 1.0, ratio  # the profiled criterion depends only on the ratio
    v = s2b * z @ z.T + s2e * np.eye(n)
    vinv = np.linalg.inv(v)
    one = np.ones(n)
    xtvx = float(one @ vinv @ one)
    rpr = float(resid @ vinv @ resid
                - (one @ vinv @ resid) ** 2 / xtvx)
    sign, logdet = np.linalg.slogdet(v / ratio)
    # rpr and xtvx both carry 1/ratio from V^-1 = Sigma^-1 / ratio.
    return ((n - 1) * np.log(rpr * ratio) + logdet
            + np.log(xtvx * ratio))


def _grouped_resid(seed=0, n_groups=6, lo=2, hi=12):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(lo, hi + 1, size=n_groups)
    codes = np.repeat(np.arange(n_groups), sizes)
    resid = rng.normal(size=codes.shape[0])
    return resid, codes, n_groups


def test_solve_matches_brute_force_posterior_mean():
    resid, codes, n_groups = _grouped_resid()
    for s2b, s2e in [(2.0, 1.0), (0.25, 1.0), (1.0, 3.0)]:
        got = solve_intercepts(resid, codes, n_groups, s2e / s2b)
        want = _brute_posterior_mean(resid, codes, n_groups, s2b, s2e)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_solve_weighted_matches_brute_force():
    resid, codes, n_groups = _grouped_resid(seed=1)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, size=resid.shape[0])
    got = solve_intercepts(resid, codes, n_groups, 1.5, weights=w)
    # Defining property: gradient of the penalized weighted objective is
    # zero at the solution.
    grad = np.bincount(codes, weights=w * (resid - got[codes]),
                       minlength=n_groups) - 1.5 * got
    np.testing.assert_allclose(grad, np.zeros(n_groups), atol=1e-10)


def test_solve_limits():
    rng = np.random.default_rng(2)
    # Huge groups: no shrinkage, the raw means.
    codes = np.repeat([0, 1], 5000)
    resid = np.concatenate([rng.normal(2.0, size=5000),
                            rng.normal(-3.0, size=5000)])
    got = solve_intercepts(resid, codes, 2, 1.0)
    np.testing.assert_allclose(got, [2.0, -3.0], atol=0.05)
    # Infinite ratio: exact zeros, whatever the residuals.
    np.testing.assert_array_equal(
        solve_intercepts(resid, codes, 2, np.inf), [0.0, 0.0])
    # Empty groups are exactly 0, even at ratio 0.
    got = solve_intercepts(np.array([1.0, 2.0]), np.array([0, 0]), 3, 0.0)
    np.testing.assert_array_equal(got[1:], [0.0, 0.0])


def test_reml_matches_brute_force_grid():
    resid, codes, n_groups = _grouped_resid(seed=3)
    # Pure noise has its optimum at ratio = inf (off any grid), so plant
    # group signal for an interior optimum both methods must find.
    rng = np.random.default_rng(3)
    resid = resid + rng.normal(0.0, 2.0, size=n_groups)[codes]
    grid = np.linspace(-4.0, 4.0, 161)
    vals = np.array([_brute_reml_profile(resid, codes, n_groups, 10.0 ** t)
                     for t in grid])
    want = grid[int(np.argmin(vals))]
    got = np.log10(estimate_ratio_reml(resid, codes, n_groups))
    assert abs(got - want) < 0.05 + (grid[1] - grid[0])


def test_reml_recovers_known_ratio():
    rng = np.random.default_rng(4)
    n_groups, per = 60, 25
    s2b, s2e = 4.0, 1.0
    b = rng.normal(0.0, np.sqrt(s2b), size=n_groups)
    codes = np.repeat(np.arange(n_groups), per)
    resid = b[codes] + rng.normal(0.0, np.sqrt(s2e), size=codes.shape[0])
    got = np.log10(estimate_ratio_reml(resid, codes, n_groups))
    assert abs(got - np.log10(s2e / s2b)) < 0.3


def test_reml_degenerate_inputs_return_inf():
    # One group: the intercept absorbs everything; nothing to separate.
    r = np.array([1.0, 2.0, 3.0])
    assert estimate_ratio_reml(r, np.array([0, 0, 0]), 1) == np.inf
    # All singletons: no within-group information at all.
    assert estimate_ratio_reml(r, np.array([0, 1, 2]), 3) == np.inf
    # Zero within-group variation: the group means are exact, so the answer
    # is no-pooling (the search floor), not "no group signal".
    assert estimate_ratio_reml(np.full(6, 2.0),
                               np.array([0, 0, 1, 1, 2, 2]), 3) == 1e-8


def test_out_of_range_codes_raise_cleanly():
    r = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="out of range"):
        estimate_ratio_reml(r, np.array([0, 1, 5]), 3)
    with pytest.raises(ValueError, match="out of range"):
        solve_intercepts(r, np.array([0, -1, 1]), 3, 1.0)


def test_codes_round_trip_factorize():
    labels = (["b", "a", "b", None, "a", np.nan, 1, 1.0, True, "1",
               "nan", "b", None])
    codes, cats = factorize(labels)
    back = codes_for_labels(labels, cats)
    np.testing.assert_array_equal(back, codes)


def test_codes_unseen_is_minus_one():
    _, cats = factorize(["x", "y", "x"])
    got = codes_for_labels(["x", "zzz", "y", None, 42], cats)
    np.testing.assert_array_equal(got, [0, -1, 1, -1, -1])


def test_codes_rejects_nothing_unhashable():
    _, cats = factorize(["x", "y"])
    got = codes_for_labels([["nested"], ("t",), "x"], cats)
    assert got[2] == 0 and got[0] == -1


# --- estimator integration -----------------------------------------------


def _grouped_data(seed=0, n_groups=30, per=40, group_sd=5.0, noise=1.0):
    """y = X @ beta + b_g + eps with known groups and intercepts."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    codes = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(n, 5))
    beta = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    b_true = rng.normal(0.0, group_sd, size=n_groups)
    y = X @ beta + b_true[codes] + rng.normal(0.0, noise, size=n)
    return X, y, codes, b_true


def _fit_re(X, y, groups, fit_kw=None, **kw):
    params = dict(n_estimators=300, random_state=0, random_effects=True)
    params.update(kw)
    return ChimeraBoostRegressor(**params).fit(X, y, groups=groups,
                                              **(fit_kw or {}))


def test_intercepts_recover_truth_and_win_on_seen_groups():
    X, y, codes, b_true = _grouped_data()
    m = _fit_re(X, y, codes)
    assert np.corrcoef(m.group_intercepts_, b_true)[0, 1] > 0.99

    rng = np.random.default_rng(1)
    n_te = 600
    te_codes = rng.integers(0, 30, size=n_te)
    Xte = rng.normal(size=(n_te, 5))
    beta = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    yte = Xte @ beta + b_true[te_codes] + rng.normal(size=n_te)

    rmse_re = np.sqrt(np.mean((m.predict(Xte, groups=te_codes) - yte) ** 2))
    plain = ChimeraBoostRegressor(n_estimators=300, random_state=0)
    plain.fit(X, y)
    rmse_plain = np.sqrt(np.mean((plain.predict(Xte) - yte) ** 2))
    # Group sd 5 dwarfs noise 1: the intercepts must win by a mile.
    assert rmse_re < 0.5 * rmse_plain


def test_unseen_groups_fall_back_to_f_only_exactly():
    X, y, codes, _ = _grouped_data()
    m = _fit_re(X, y, codes)
    rng = np.random.default_rng(2)
    Xte = rng.normal(size=(50, 5))
    f_only = m.predict(Xte)
    unseen = m.predict(Xte, groups=np.full(50, "never-seen"))
    np.testing.assert_array_equal(unseen, f_only)
    # And F-only equals the trees-only raw scores (RMSE: identity link).
    np.testing.assert_array_equal(f_only, m.predict_raw(Xte))


def test_string_labels_round_trip():
    X, y, codes, _ = _grouped_data(n_groups=10, per=30)
    labels = np.array([f"g{c}" for c in codes], dtype=object)
    m = _fit_re(X, y, labels)
    assert m.group_labels_.shape == m.group_intercepts_.shape
    pred = m.predict(X, groups=labels)
    assert np.isfinite(pred).all()
    back = codes_for_labels(labels, m.group_labels_)
    np.testing.assert_array_equal(back, codes)


def test_missing_labels_without_split():
    # None/NaN labels break sklearn's group splitter (pre-existing, flag-off
    # fits fail identically), so exercise them with no automatic split.
    X, y, codes, _ = _grouped_data(n_groups=10, per=30)
    labels = np.array([f"g{c}" for c in codes], dtype=object)
    labels[::37] = None
    m = _fit_re(X, y, labels, early_stopping=False)
    # The None rows matched the fit's "__nan__" group, not the fallback:
    # their offsets are that group's intercept.
    nan_code = codes_for_labels([None], m.group_labels_)[0]
    assert nan_code >= 0
    off = m.predict(X[::37], groups=labels[::37]) - m.predict(X[::37])
    np.testing.assert_allclose(off, m.group_intercepts_[nan_code],
                               rtol=1e-12)


def test_deterministic_under_seed():
    X, y, codes, _ = _grouped_data()
    m1 = _fit_re(X, y, codes)
    m2 = _fit_re(X, y, codes)
    np.testing.assert_array_equal(m1.group_intercepts_,
                                  m2.group_intercepts_)
    np.testing.assert_array_equal(m1.predict(X, groups=codes),
                                  m2.predict(X, groups=codes))


def test_noise_groups_shrink_to_zero_and_hold_accuracy():
    X, y, _, _ = _grouped_data()
    rng = np.random.default_rng(3)
    noise_groups = rng.integers(0, 400, size=len(y))  # ~3 rows each
    m = _fit_re(X, y, noise_groups)
    assert np.abs(m.group_intercepts_).max() < 0.5
    # Within-model: the shrunk offsets barely move predictions off F-only.
    np.testing.assert_allclose(m.predict(X, groups=noise_groups),
                               m.predict(X), atol=0.6)
    # Against the resampling-only fit on the same split: no real damage.
    plain = ChimeraBoostRegressor(n_estimators=300, random_state=0)
    plain.fit(X, y, groups=noise_groups)
    Xte = rng.normal(size=(400, 5))
    rmse_re = np.sqrt(np.mean((m.predict(Xte) - Xte @ [3, -2, 0, 1, 0.5]) ** 2))
    rmse_plain = np.sqrt(np.mean(
        (plain.predict(Xte) - Xte @ [3, -2, 0, 1, 0.5]) ** 2))
    assert rmse_re < 1.25 * rmse_plain


def test_flag_off_ignores_nothing_and_changes_nothing():
    # groups=... without the flag is resampling-only, exactly as before:
    # no intercepts are stored and predictions take no groups.
    X, y, codes, _ = _grouped_data()
    m = ChimeraBoostRegressor(n_estimators=300, random_state=0)
    m.fit(X, y, groups=codes)
    assert m.group_intercepts_ is None
    with pytest.raises(ValueError, match="without random_effects"):
        m.predict(X, groups=codes)


def test_error_paths():
    X, y, codes, _ = _grouped_data()
    with pytest.raises(ValueError, match="groups="):
        ChimeraBoostRegressor(random_effects=True).fit(X, y)
    with pytest.raises(ValueError, match="RMSE"):
        ChimeraBoostRegressor(random_effects=True, loss="MAE").fit(
            X, y, groups=codes)
    with pytest.raises(ValueError, match="n_ensembles"):
        ChimeraBoostRegressor(random_effects=True, n_ensembles=2).fit(
            X, y, groups=codes)
    with pytest.raises(ValueError, match="one label per training row"):
        ChimeraBoostRegressor(random_effects=True).fit(
            X, y, groups=codes[:100])
    m = _fit_re(X, y, codes)
    with pytest.raises(ValueError, match="one label per prediction row"):
        m.predict(X, groups=codes[:100])


def test_explicit_eval_set_and_no_es_and_no_refit_paths():
    X, y, codes, _ = _grouped_data()
    # Explicit eval rows carry no groups: scored F-only, fit still works.
    m = _fit_re(X[:1000], y[:1000], codes[:1000],
                fit_kw={"eval_set": (X[1000:], y[1000:])})
    assert np.isfinite(m.group_intercepts_).all()
    # No early stopping at all.
    m = _fit_re(X, y, codes, early_stopping=False)
    assert np.isfinite(m.group_intercepts_).all()
    # No full-data refit: val-split groups still get intercepts.
    m = _fit_re(X, y, codes, refit_full=False)
    assert np.isfinite(m.group_intercepts_).all()
    assert len(m.group_intercepts_) == 30
    # From-scratch refit (not replay) on the adjusted target.
    m = _fit_re(X, y, codes, refit_full=True)
    assert np.isfinite(m.group_intercepts_).all()


def test_sample_weights_flow_through():
    X, y, codes, _ = _grouped_data()
    w = np.ones(len(y))
    w[codes == 0] = 0.0  # ghost group: contributes nothing anywhere
    m = _fit_re(X, y, codes, fit_kw={"sample_weight": w})
    zero_code = codes_for_labels([0], m.group_labels_)[0]
    assert m.group_intercepts_[zero_code] == 0.0


def test_pickle_round_trip_preserves_intercepts():
    import pickle

    X, y, codes, _ = _grouped_data()
    m = _fit_re(X, y, codes)
    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)
    np.testing.assert_array_equal(m2.group_intercepts_, m.group_intercepts_)
    np.testing.assert_array_equal(m2.predict(X, groups=codes),
                                  m.predict(X, groups=codes))


def test_winner_is_plain_prefit_bit_identical():
    # Before the refit, the RE winner is a plain fit on raw y (groups enter
    # only at the solve, and the val split is random like a plain fit's):
    # same seed => same trees, bit for bit.
    X, y, codes, _ = _grouped_data()
    m = _fit_re(X, y, codes, refit_full=False)
    plain = ChimeraBoostRegressor(n_estimators=300, random_state=0,
                                 refit_full=False)
    plain.fit(X, y)
    assert len(m.model_.trees_) == len(plain.model_.trees_)
    np.testing.assert_array_equal(m.predict_raw(X), plain.predict_raw(X))


def test_staged_predict_final_matches_predict():
    X, y, codes, _ = _grouped_data()
    m = _fit_re(X, y, codes)
    stages = list(m.staged_predict(X, groups=codes))
    np.testing.assert_allclose(stages[-1], m.predict(X, groups=codes),
                               rtol=1e-12)
    stages_f = list(m.staged_predict(X))
    np.testing.assert_allclose(stages_f[-1], m.predict(X), rtol=1e-12)
