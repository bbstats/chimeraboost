"""Group-centered categorical crosses (op="gdiff"): validation-selected
``x_num - mean(x_num | cat)`` columns. Target-free, so the same fitted map
serves fit and predict; unseen categories fall back to the global mean."""

import pickle

import numpy as np

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.preprocessing import FeaturePreprocessor
from chimeraboost.sklearn_api import _cross_candidate_pairs


def _group_reg(n=5000, k=40, seed=0):
    """Regression whose signal is the deviation of x0 from its category's
    baseline -- a per-category threshold an oblivious tree must staircase but
    a gdiff column captures in one split."""
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, k, n)
    baseline = rng.normal(scale=3.0, size=k)
    x0 = baseline[cats] + rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 4.0 * (x0 - baseline[cats] > 0) + 0.1 * rng.standard_normal(n)
    X = np.column_stack([x0, x1, cats.astype(object)]).astype(object)
    return X, y


# ---- preprocessor mechanism ----------------------------------------------

def test_gdiff_column_values_and_fallback():
    X = np.array([[1.0, "a"], [3.0, "a"], [10.0, "b"], [14.0, "b"]],
                 dtype=object)
    prep = FeaturePreprocessor(cross_pairs=[(0, 1, "gdiff")])
    prep.fit_transform(X, encode_targets=[np.array([0., 1., 0., 1.])],
                       cat_features=[1])
    block = prep._cross_block(X)
    # a-mean 2.0, b-mean 12.0 -> centered deviations.
    np.testing.assert_allclose(block[:, 0], [-1.0, 1.0, -2.0, 2.0])
    # Unseen category -> global mean (7.0); NaN numeric propagates.
    Xn = np.array([[9.0, "zzz"], [np.nan, "a"]], dtype=object)
    block = prep._cross_block(Xn)
    np.testing.assert_allclose(block[0, 0], 2.0)
    assert np.isnan(block[1, 0])


def test_gdiff_nan_category_is_its_own_group():
    X = np.array([[1.0, "a"], [3.0, "a"], [100.0, None], [104.0, None]],
                 dtype=object)
    prep = FeaturePreprocessor(cross_pairs=[(0, 1, "gdiff")])
    prep.fit_transform(X, encode_targets=[np.zeros(4)], cat_features=[1])
    block = prep._cross_block(X)
    np.testing.assert_allclose(block[:, 0], [-1.0, 1.0, -2.0, 2.0])


def test_gdiff_zero_weight_rows_do_not_shape_means():
    X = np.array([[1.0, "a"], [3.0, "a"], [999.0, "a"], [10.0, "b"],
                  [14.0, "b"]], dtype=object)
    y = np.zeros(5)
    w = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    full = FeaturePreprocessor(cross_pairs=[(0, 1, "gdiff")])
    full.fit_transform(X, [y], [1], sample_weight=w)
    dropped = FeaturePreprocessor(cross_pairs=[(0, 1, "gdiff")])
    dropped.fit_transform(X[w > 0], [y[w > 0]], [1])
    assert full.gdiff_maps_[0][0] == dropped.gdiff_maps_[0][0]


def test_feature_map_folds_gdiff_into_numeric_parent():
    X, y = _group_reg()
    b = GradientBoosting(n_estimators=30, random_state=0,
                         cross_pairs=[(0, 2, "gdiff"), (0, 1, "diff")])
    b.fit(X, y, cat_features=[2])
    # Columns: numerics [0, 1] | crosses [gdiff->0, diff->0] | TS cat block.
    assert list(b.prep_.feature_map_[:4]) == [0, 1, 0, 0]
    assert b.feature_importances_.shape == (3,)


# ---- candidate generation -------------------------------------------------

def test_candidates_without_cats_are_unchanged():
    imp = np.array([5.0, 4.0, 3.0, 2.0])
    pairs = _cross_candidate_pairs(imp, None, 4)
    assert all(op in ("diff", "prod") for _, _, op in pairs)
    assert len(pairs) == 12   # C(4,2) * 2


def test_candidates_with_cats_add_gdiff():
    imp = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pairs = _cross_candidate_pairs(imp, [3, 4], 5)
    gdiff = [(i, j) for i, j, op in pairs if op == "gdiff"]
    # top numerics {0,1,2} x cats {3,4}, cat rank by importance.
    assert set(gdiff) == {(i, j) for i in (0, 1, 2) for j in (3, 4)}
    numnum = [p for p in pairs if p[2] != "gdiff"]
    assert len(numnum) == 6   # C(3,2) * 2, cats excluded


def test_candidates_single_numeric_plus_cat_engage():
    pairs = _cross_candidate_pairs(np.array([1.0, 1.0]), [1], 2)
    assert pairs == [(0, 1, "gdiff")]


# ---- end-to-end selection -------------------------------------------------

def test_regressor_selects_gdiff_on_group_data_and_wins():
    X, y = _group_reg()
    Xtr, Xte, ytr, yte = X[:4000], X[4000:], y[:4000], y[4000:]
    on = ChimeraBoostRegressor(random_state=0).fit(
        Xtr, ytr, cat_features=[2])
    off = ChimeraBoostRegressor(random_state=0, cross_features=False).fit(
        Xtr, ytr, cat_features=[2])
    assert on.cross_features_selected_
    assert any(op == "gdiff" for _, _, op in on.cross_pairs_)
    rmse_on = np.sqrt(np.mean((on.predict(Xte) - yte) ** 2))
    rmse_off = np.sqrt(np.mean((off.predict(Xte) - yte) ** 2))
    assert rmse_on < rmse_off


def test_classifier_gdiff_pickle_roundtrip():
    X, y = _group_reg()
    yb = (y > np.median(y)).astype(int)
    m = ChimeraBoostClassifier(random_state=0).fit(X[:4000], yb[:4000],
                                                   cat_features=[2])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict_proba(X[4000:]),
                                  m2.predict_proba(X[4000:]))
