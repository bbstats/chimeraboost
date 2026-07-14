"""Numeric cross features (``cross_features=True``): validation-selected
difference/product columns for top numeric feature pairs."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting


def _interaction_reg(n=4000, seed=0):
    """Regression whose signal is a comparison + a product -- exactly what an
    oblivious tree staircases and a cross column captures in one split."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (3.0 * (X[:, 0] > X[:, 1]) + X[:, 2] * X[:, 3]
         + 0.1 * rng.standard_normal(n))
    return X, y


def _interaction_clf(n=6000, seed=0):
    """XOR of a comparison and a product sign: linear leaves can't express it,
    cross columns crack both parts with one split each."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    z = (X[:, 0] > X[:, 1]) != (X[:, 2] * X[:, 3] > 0)
    p = np.where(z, 0.9, 0.1)
    y = (rng.random(n) < p).astype(int)
    return X, y


# ---- booster-level cross_pairs -------------------------------------------

def test_booster_cross_pairs_transform_roundtrip():
    X, y = _interaction_reg()
    pairs = [(0, 1, "diff"), (2, 3, "prod")]
    b = GradientBoosting(n_estimators=50, random_state=0, cross_pairs=pairs)
    b.fit(X, y)
    # transform reproduces the cross columns on new data of ORIGINAL width;
    # binned width = 5 numerics + 2 crosses.
    Xb = b.prep_.transform(X[:10])
    assert Xb.shape == (10, 7)
    assert b.prep_.is_numeric_binned_.shape == (7,)
    assert b.prep_.is_numeric_binned_.all()
    # feature map folds crosses into the lower-indexed parent.
    assert list(b.prep_.feature_map_) == [0, 1, 2, 3, 4, 0, 2]
    # importances stay in the ORIGINAL feature space.
    assert b.feature_importances_.shape == (5,)


def test_booster_cross_pairs_help_interaction_data():
    X, y = _interaction_reg()
    Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]
    plain = GradientBoosting(n_estimators=150, random_state=0)
    plain.fit(Xtr, ytr)
    crossed = GradientBoosting(n_estimators=150, random_state=0,
                               cross_pairs=[(0, 1, "diff"), (2, 3, "prod")])
    crossed.fit(Xtr, ytr)
    rmse = lambda m: np.sqrt(np.mean((yte - m.predict_raw(Xte)) ** 2))
    assert rmse(crossed) < rmse(plain) * 0.9


def test_cross_block_nan_propagates():
    X, y = _interaction_reg(n=2500)
    X[5, 0] = np.nan
    b = GradientBoosting(n_estimators=20, random_state=0,
                         cross_pairs=[(0, 1, "diff")])
    b.fit(X, y)
    assert np.isfinite(b.predict_raw(X[:10])).all()


# ---- wrapper-level selection ---------------------------------------------

def test_regressor_selects_crosses_on_interaction_data():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is True
    assert m.cross_pairs_
    # predicts on ORIGINAL-width input, no user-side augmentation.
    assert m.predict(X[:20]).shape == (20,)
    base = ChimeraBoostRegressor(n_estimators=200, random_state=0).fit(X, y)
    Xte, yte = _interaction_reg(seed=1)
    rmse = lambda mm: np.sqrt(np.mean((yte - mm.predict(Xte)) ** 2))
    assert rmse(m) < rmse(base)


def test_classifier_selects_crosses_on_interaction_data():
    X, y = _interaction_clf()
    m = ChimeraBoostClassifier(n_estimators=200, random_state=0,
                               cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is True
    proba = m.predict_proba(X[:20])
    assert proba.shape == (20, 2)


def test_default_off_is_identical():
    X, y = _interaction_reg()
    a = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              cross_features=False).fit(X, y)
    np.testing.assert_array_equal(a.predict(X[:50]), b.predict(X[:50]))
    assert b.cross_features_selected_ is None


def test_selection_can_reject_crosses():
    # Pure-noise target: crosses cannot beat the base on validation reliably;
    # whatever the verdict, the final model must be usable and recorded.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 4))
    y = rng.standard_normal(3000)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ in (True, False)
    if not m.cross_features_selected_:
        assert m.cross_pairs_ is None
    assert m.predict(X[:5]).shape == (5,)


def test_skipped_below_min_samples_and_without_validation():
    X, y = _interaction_reg(n=900)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is None

    X, y = _interaction_reg(n=3000)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True,
                              early_stopping=False).fit(X, y)
    assert m.cross_features_selected_ is None


def test_skipped_for_mae_loss():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0, loss="MAE",
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is None


def test_multiclass_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 4))
    y = rng.integers(0, 3, 600)
    with pytest.raises(NotImplementedError):
        ChimeraBoostClassifier(cross_features=True).fit(X, y)


def test_crosses_skip_categorical_columns():
    rng = np.random.default_rng(0)
    n = 4000
    Xnum = rng.standard_normal((n, 3))
    cat = rng.integers(0, 4, n).astype(str)
    X = np.column_stack([Xnum.astype(object), cat.astype(object)])
    y = (2.0 * (Xnum[:, 0] > Xnum[:, 1]) + (cat == "2") * 1.5
         + 0.1 * rng.standard_normal(n))
    m = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              cross_features=True)
    m.fit(X, y, cat_features=[3])
    if m.cross_pairs_:
        flat = {i for i, j, _ in m.cross_pairs_} | {j for i, j, _ in m.cross_pairs_}
        assert 3 not in flat
    assert m.predict(X[:10]).shape == (10,)


def test_shap_stays_in_original_feature_space():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_features=True).fit(X, y)
    if m.cross_features_selected_:
        contrib = m.shap_values(X[:16])
        assert contrib.shape == (16, 5)
        recon = contrib.sum(axis=1) + m.expected_value_
        np.testing.assert_allclose(recon, m.predict(X[:16]), rtol=1e-6,
                                   atol=1e-8)
