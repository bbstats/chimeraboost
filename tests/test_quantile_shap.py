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

    # Raw space reconstructs the pre-rearrangement scores.
    phi = m.shap_values(Xt)
    assert phi.shape == (60, 6, len(quantiles))
    assert np.shape(m.expected_value_) == (len(quantiles),)
    raw = m.model_._raw_scores(Xt)
    assert np.abs(phi.sum(axis=1) + m.expected_value_ - raw).max() < 1e-8

    # Delivered space reconstructs exactly what predict() returns.
    phid = m.shap_values(Xt, space="delivered")
    assert m.expected_value_.shape == (60, len(quantiles))
    err = np.abs(phid.sum(axis=1) + m.expected_value_ - m.predict(Xt)).max()
    assert err < 1e-8, err


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
    assert np.array_equal(m.shap_values(Xt),
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
                                      cat_features=[1]).fit(X, y)
    phi = m.shap_values(X[:40])
    assert phi.shape == (40, 3, 3)
    err = np.abs(phi.sum(axis=1) + m.expected_value_
                 - m.model_._raw_scores(X[:40])).max()
    assert err < 1e-8


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
