"""SHAP for the multi-quantile head.

The efficiency identity is the whole contract: contributions plus the baseline
must reconstruct the explained quantity. What varies between these tests is
*which* quantity, because the head applies two transforms on the way out --
a per-row monotone rearrangement, then an optional conformal rescale -- and
each has to be carried through exactly.

The kernel itself is pinned separately, bit-for-bit against the scalar SHAP
kernel at K=1, in tests/test_tree_kernels.py.
"""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostQuantileRegressor

TAUS19 = list(np.round(np.arange(0.05, 0.9501, 0.05), 2))


def _data(n=1200, seed=0):
    """Heteroscedastic on purpose: feature 0 moves the location, feature 1
    moves the spread. Interval-width attribution has to find feature 1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = X[:, 0] * 2 + (0.2 + 1.8 * (X[:, 1] > 0)) * rng.standard_normal(n)
    return X, y


def _fit(quantiles=TAUS19, conformalize=False, n=1200):
    X, y = _data(n)
    m = ChimeraBoostQuantileRegressor(quantiles=quantiles, n_estimators=120,
                                      random_state=0,
                                      conformalize=conformalize).fit(X, y)
    return m, X[:60]


@pytest.mark.parametrize("quantiles,conformalize", [
    (TAUS19, False),
    (TAUS19, True),
    ([0.1, 0.5, 0.9], False),
    ([0.1, 0.5, 0.9], True),
    ([0.1, 0.4, 0.6, 0.9], False),     # median interpolated: mw != 0
    ([0.1, 0.4, 0.6, 0.9], True),      # ... and rescaled through it
    ([0.5], False),                    # K=1 degenerate
])
def test_raw_and_delivered_attributions_are_both_efficient(quantiles,
                                                           conformalize):
    m, Xt = _fit(quantiles, conformalize)

    # The default explains what predict() returns.
    phid = m.shap_values(Xt)
    assert phid.shape == (60, 6, len(quantiles))
    assert m.expected_value_.shape == (60, len(quantiles))
    err = np.abs(phid.sum(axis=1) + m.expected_value_ - m.predict(Xt)).max()
    assert err < 1e-8, err

    # Raw space reconstructs the pre-rearrangement scores instead.
    phi = m.shap_values(Xt, space="raw")
    assert phi.shape == (60, 6, len(quantiles))
    assert np.shape(m.expected_value_) == (len(quantiles),)
    raw = m.model_._raw_scores(Xt)
    assert np.abs(phi.sum(axis=1) + m.expected_value_ - raw).max() < 1e-8


def test_conformal_rescale_is_actually_exercised():
    """Guards the short-circuit: `_delivered_shap` skips the rescale when
    every factor is 1, so a conformalized test that silently took that branch
    would prove nothing."""
    m, _ = _fit(conformalize=True)
    assert not np.all(m.conformal_scale_ == 1.0)


def test_the_permutation_really_is_the_delivery_sort():
    """`_delivered_shap` gathers by argsort and trusts that this reproduces
    the sort `_predict_raw_impl` applies. Check it rather than assume it."""
    m, Xt = _fit()
    raw = m.model_._raw_scores(Xt)
    order = np.argsort(raw, axis=1, kind="stable")
    assert np.array_equal(np.take_along_axis(raw, order, axis=1),
                          np.sort(raw, axis=1))
    assert np.array_equal(np.sort(raw, axis=1), m.predict(Xt)) or \
        not np.all(m.conformal_scale_ == 1.0)


def test_the_two_spaces_agree_exactly_when_no_row_crosses():
    """Raw and delivered differ only through the permutation, so on a grid
    where nothing crosses they must be identical. This is what makes the
    default safe on small grids."""
    m, Xt = _fit(quantiles=[0.1, 0.5, 0.9])
    raw = m.model_._raw_scores(Xt)
    assert not (np.diff(raw, axis=1) < 0).any(), "grid crossed; test is moot"
    assert np.array_equal(m.shap_values(Xt, space="raw"),
                          m.shap_values(Xt, space="delivered"))


def test_mean_attribution_reconstructs_the_point_prediction():
    m, Xt = _fit()
    phi = m.shap_values(Xt, kind="mean")
    assert phi.shape == (60, 6)
    err = np.abs(phi.sum(axis=1) + m.expected_value_
                 - m.predict(Xt, kind="mean")).max()
    assert err < 1e-8, err


def test_width_attribution_reconstructs_the_interval_width():
    m, Xt = _fit()
    iv = m.predict(Xt, kind="interval", alpha=0.1)
    phi = m.shap_values(Xt, kind="width", alpha=0.1)
    assert phi.shape == (60, 6)
    err = np.abs(phi.sum(axis=1) + m.expected_value_
                 - (iv[:, 1] - iv[:, 0])).max()
    assert err < 1e-8, err


def test_width_attribution_finds_the_feature_that_drives_the_spread():
    """The claim the feature is sold on. Feature 1 sets the conditional
    spread and barely moves the median; feature 0 does the opposite. The
    width ranking must put feature 1 on top even though the median ranking
    does not."""
    m, Xt = _fit(n=3000)
    width = np.abs(m.shap_values(Xt, kind="width", alpha=0.1)).mean(axis=0)
    median = np.abs(m.shap_values(Xt, quantile=0.5)).mean(axis=0)

    assert width.argmax() == 1, width
    assert median.argmax() == 0, median
    # ... and it is not a photo finish in either direction.
    assert width[1] > 2 * width[0]
    assert median[0] > 2 * median[1]


def test_single_level_selection_matches_the_full_cube():
    m, Xt = _fit()
    full = m.shap_values(Xt)
    one = m.shap_values(Xt, quantile=0.9)
    k = int(np.argmin(np.abs(m.quantiles_ - 0.9)))
    assert np.array_equal(one, full[:, :, k])


def test_levels_and_kinds_off_the_grid_raise_rather_than_guess():
    m, Xt = _fit(quantiles=[0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="not on the fitted grid"):
        m.shap_values(Xt, quantile=0.42)
    with pytest.raises(ValueError, match="not on the fitted grid"):
        m.shap_values(Xt, kind="width", alpha=0.1)   # needs 0.05 / 0.95
    with pytest.raises(ValueError, match="needs alpha"):
        m.shap_values(Xt, kind="width")
    with pytest.raises(ValueError, match="space must be"):
        m.shap_values(Xt, space="sorted")
    with pytest.raises(ValueError, match="kind must be"):
        m.shap_values(Xt, kind="interval")


def test_importances_rank_features_and_respect_a_level():
    m, Xt = _fit(n=3000)
    imp = m.shap_importances(Xt)
    assert imp.shape == (6,)
    assert imp["feature"][0] == 0            # location driver dominates overall

    # At an extreme level the spread driver must gain ground on it.
    def gap(**kw):
        d = m.shap_importances(Xt, prettified=True, **kw)
        return d[1] / d[0]

    assert gap(quantile=0.95) > gap(quantile=0.5)


def test_unused_feature_gets_near_zero_attribution_at_every_level():
    m, Xt = _fit(n=3000)
    imp = np.abs(m.shap_values(Xt)).mean(axis=0)     # (n_features, K)
    # Features 2..5 enter y not at all.
    assert imp[2:].max() < 0.1 * imp[:2].max()


def test_categoricals_land_in_original_feature_space():
    rng = np.random.default_rng(3)
    n = 900
    cat = rng.integers(0, 5, n)
    X = np.column_stack([rng.standard_normal(n), cat.astype(float),
                         rng.standard_normal(n)])
    y = X[:, 0] + cat * 0.5 + 0.3 * rng.standard_normal(n)
    m = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.5, 0.9],
                                      n_estimators=60, random_state=0,
                                      cat_features=[1], audition=False).fit(X, y)
    # Explicitly raw, so this pins the original-space mapping rather than
    # relying on a 3-level grid never crossing (which would make the default
    # coincide with raw and the assertion pass for the wrong reason).
    phi = m.shap_values(X[:40], space="raw")
    assert phi.shape == (40, 3, 3)
    err = np.abs(phi.sum(axis=1) + m.expected_value_
                 - m.model_._raw_scores(X[:40])).max()
    assert err < 1e-8

    # The default is still well-formed on categorical data.
    assert m.shap_values(X[:40]).shape == (40, 3, 3)


def test_explicit_background_moves_the_baseline():
    m, Xt = _fit()
    X, _ = _data()
    m.shap_values(Xt)
    default_base = m.expected_value_.copy()
    m.shap_values(Xt, X_background=X[:100] + 5.0)
    assert not np.allclose(default_base, m.expected_value_)


def test_an_unfitted_forest_explains_to_the_init_value():
    """n_estimators is honoured, so a model that kept no trees still has to
    return a well-shaped zero attribution rather than fall over."""
    X, y = _data(400)
    m = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.5, 0.9],
                                      n_estimators=1, random_state=0,
                                      early_stopping=False).fit(X, y)
    m.model_.trees_ = []
    phi = m.shap_values(X[:5])
    assert phi.shape == (5, 6, 3)
    assert not phi.any()
    assert np.allclose(m.expected_value_, m.model_.init_)


def test_importances_stay_on_the_raw_grid_whatever_the_default_is():
    """`shap_importances` averages across rows, and the delivered view
    reorders levels per row, so averaging it would mix rows that were
    reordered differently. It must pin space="raw" rather than inherit
    whatever `shap_values` defaults to."""
    m, Xt = _fit(n=3000)
    got = m.shap_importances(Xt, prettified=True)
    want = m.shap_importances(Xt, prettified=True, quantile=None)
    assert got == want

    phi_raw = m.shap_values(Xt, space="raw")
    expected = np.abs(phi_raw).mean(axis=0).mean(axis=1)
    assert np.allclose([got[j] for j in sorted(got)],
                       [expected[j] for j in sorted(got)])


def _aud_split_shap(X, y, split_seed):
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=split_seed)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=0)
    return (Xf, Xv, yf, yv), (Xte, yte)


def _aud_data_shap(kind):
    if kind == "head":
        rng = np.random.default_rng(1)
        n = 1000
        X = rng.standard_normal((n, 5))
        y = 2.0 * X[:, 0] + np.exp(0.6 * X[:, 1]) * rng.standard_normal(n)
        return X, y, 2
    if kind == "bins":
        rng = np.random.default_rng(7)
        n = 1200
        X = rng.uniform(-1.0, 1.0, size=(n, 3))
        y = ((X[:, 0] > 0.712345).astype(float) * 5.0
             + rng.standard_normal(n) * 0.2)
        return X, y, 0
    if kind == "recentred":
        rng = np.random.default_rng(2)
        n = 1000
        X = rng.standard_normal((n, 4))
        y = 2.0 * X[:, 0] + 0.2 * rng.standard_normal(n)
        return X, y, 0
    if kind == "fixed":
        rng = np.random.default_rng(3)
        n = 600
        X = rng.standard_normal((n, 6))
        y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(n)
        return X, y, 1
    assert kind == "scaled"
    rng = np.random.default_rng(41)
    n = 2500
    X = rng.standard_normal((n, 5))
    y = 2.0 * X[:, 0] + np.exp(0.5 * X[:, 1]) * rng.standard_normal(n)
    return X, y, 4


def _aud_fit_shap(kind):
    from chimeraboost import ChimeraBoostQuantileRegressor
    taus = [0.05, 0.25, 0.5, 0.75, 0.95]
    X, y, sseed = _aud_data_shap(kind)
    (Xf, Xv, yf, yv), (Xte, yte) = _aud_split_shap(X, y, sseed)
    m = ChimeraBoostQuantileRegressor(
        quantiles=taus, n_estimators=300, early_stopping_rounds=50,
        thread_count=1, random_state=0).fit(Xf, yf, eval_set=(Xv, yv))
    return m, Xte


def _unfloored_rows(m, Xt):
    """Rows whose spread sits above its floor -- where N's attribution is
    exact."""
    spread = m._spread_model_
    s_raw = np.asarray(
        spread.model_.predict_raw(Xt) + spread.quantile_offset_,
        dtype=np.float64).ravel()
    return s_raw > float(m._spread_floor_)


def test_audition_winner_shap_is_locally_accurate():
    """SHAP reconstructs the winner for H/B/R/S/N, quantiles and width.

    N is checked on unfloored rows only: the floor breaks the linearity
    the attribution assumes."""
    for kind in ("head", "bins", "recentred", "fixed", "scaled"):
        m, Xte = _aud_fit_shap(kind)
        assert m.audition_["selected"] == kind
        Xt = Xte[:20]
        rows = slice(None)
        if kind == "scaled":
            rows = _unfloored_rows(m, Xt)
            assert rows.any(), "no unfloored rows; test is moot"
        phi = m.shap_values(Xt)
        assert phi.shape == (20, Xt.shape[1], 5)
        assert np.allclose(phi[rows].sum(axis=1) + m.expected_value_,
                           m.predict(Xt)[rows])
        iv = m.predict(Xt, kind="interval", alpha=0.1)
        phiw = m.shap_values(Xt, kind="width", alpha=0.1)
        assert phiw.shape == (20, Xt.shape[1])
        if kind == "fixed":
            assert not np.any(phiw), "fixed width attribution is zero"
        assert np.allclose(phiw[rows].sum(axis=1) + m.expected_value_,
                           iv[rows, 1] - iv[rows, 0])
