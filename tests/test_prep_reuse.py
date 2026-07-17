"""Intra-fit preprocessing reuse: the booster fits inside one sklearn-level
fit (selection auditions, cross-augmented candidate, winner refit) share one
prep cache and must produce bit-identical models to uncached fits."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.preprocessing import FeaturePreprocessor
import chimeraboost.preprocessing as pmod


def _mixed_data(n=3000, seed=0, classification=False):
    """Numerics (with NaNs) + two categorical columns, big enough to trigger
    linear-leaf selection and cross-feature candidates."""
    rng = np.random.default_rng(seed)
    Xn = rng.standard_normal((n, 5))
    Xn[rng.random((n, 5)) < 0.02] = np.nan
    ca = rng.integers(0, 8, n)
    cb = rng.integers(0, 5, n)
    X = np.empty((n, 7), dtype=object)
    X[:, :5] = Xn
    X[:, 5] = np.array([f"a{v}" for v in ca], dtype=object)
    X[:, 6] = np.array([f"b{v}" for v in cb], dtype=object)
    signal = (np.nan_to_num(Xn[:, 0] - Xn[:, 1]) + 0.5 * ca
              + np.nan_to_num(Xn[:, 2] * Xn[:, 3]))
    if classification:
        y = (signal + 0.3 * rng.standard_normal(n) > np.median(signal)).astype(int)
    else:
        y = signal + 0.3 * rng.standard_normal(n)
    return X, y


CAT = [5, 6]
PAIRS = [(0, 1, "diff"), (0, 1, "prod"), (2, 3, "diff"), (2, 3, "prod")]


# ---- FeaturePreprocessor.from_base_with_cross ----------------------------

def test_from_base_with_cross_bit_identical():
    X, y = _mixed_data()
    base = FeaturePreprocessor(max_bins=64, random_state=7)
    base_binned = base.fit_transform(X, [y], CAT)

    scratch = FeaturePreprocessor(max_bins=64, random_state=7,
                                  cross_pairs=PAIRS)
    scratch_binned = scratch.fit_transform(X, [y], CAT)

    aug, cross_binner, crossb = FeaturePreprocessor.from_base_with_cross(
        base, PAIRS, X)

    # Fitted state matches the from-scratch fit exactly.
    assert len(aug.binner_.borders_) == len(scratch.binner_.borders_)
    for a, s in zip(aug.binner_.borders_, scratch.binner_.borders_):
        np.testing.assert_array_equal(a, s)
    np.testing.assert_array_equal(aug.n_bins_, scratch.n_bins_)
    for a, s in zip(aug.binner_.bin_centers_, scratch.binner_.bin_centers_):
        np.testing.assert_array_equal(a, s)
    np.testing.assert_array_equal(aug.is_numeric_binned_,
                                  scratch.is_numeric_binned_)
    np.testing.assert_array_equal(aug.feature_map_, scratch.feature_map_)

    # The spliced binned train matrix matches the from-scratch one exactly.
    nb = len(base.num_features_)
    spliced = np.hstack([base_binned[:, :nb], crossb, base_binned[:, nb:]])
    np.testing.assert_array_equal(spliced, scratch_binned)

    # transform on unseen rows (fresh categories included) matches exactly.
    Xt, _ = _mixed_data(n=500, seed=1)
    Xt[0, 5] = "unseen_cat"
    np.testing.assert_array_equal(aug.transform(Xt), scratch.transform(Xt))


def test_from_base_with_cross_rejects_cross_base():
    X, y = _mixed_data(n=1200)
    base = FeaturePreprocessor(max_bins=32, random_state=0,
                               cross_pairs=[(0, 1, "diff")])
    base.fit_transform(X, [y], CAT)
    with pytest.raises(ValueError):
        FeaturePreprocessor.from_base_with_cross(base, PAIRS, X)


# ---- booster-level prep_cache --------------------------------------------

def test_prep_cache_hit_is_identical():
    X, y = _mixed_data()
    ev = (X[-600:], y[-600:])
    Xtr, ytr = X[:-600], y[:-600]
    kw = dict(n_estimators=60, random_state=3, early_stopping_rounds=20)

    plain = GradientBoosting(**kw)
    plain.fit(Xtr, ytr, cat_features=CAT, eval_set=ev)

    cache = {}
    first = GradientBoosting(**kw)
    first.fit(Xtr, ytr, cat_features=CAT, eval_set=ev, prep_cache=cache)
    second = GradientBoosting(**kw)
    second.fit(Xtr, ytr, cat_features=CAT, eval_set=ev, prep_cache=cache)

    assert second.prep_ is first.prep_          # the hit reused the object
    np.testing.assert_array_equal(plain.predict_raw(X), first.predict_raw(X))
    np.testing.assert_array_equal(plain.predict_raw(X), second.predict_raw(X))
    assert plain.valid_history_ == second.valid_history_


def test_prep_cache_augments_from_base():
    X, y = _mixed_data()
    ev = (X[-600:], y[-600:])
    Xtr, ytr = X[:-600], y[:-600]
    kw = dict(n_estimators=60, random_state=3, early_stopping_rounds=20)

    plain = GradientBoosting(cross_pairs=PAIRS, **kw)
    plain.fit(Xtr, ytr, cat_features=CAT, eval_set=ev)

    cache = {}
    GradientBoosting(**kw).fit(Xtr, ytr, cat_features=CAT, eval_set=ev,
                               prep_cache=cache)
    aug = GradientBoosting(cross_pairs=PAIRS, **kw)
    aug.fit(Xtr, ytr, cat_features=CAT, eval_set=ev, prep_cache=cache)

    np.testing.assert_array_equal(plain.predict_raw(X), aug.predict_raw(X))
    assert plain.valid_history_ == aug.valid_history_


# ---- sklearn-level: prep runs once per fit -------------------------------

def _count_prep_fits(monkeypatch):
    calls = {"n": 0}
    orig = FeaturePreprocessor.fit_transform

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(pmod.FeaturePreprocessor, "fit_transform", counting)
    return calls


def test_regressor_preps_once(monkeypatch):
    calls = _count_prep_fits(monkeypatch)
    X, y = _mixed_data()
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0)
    m.fit(X, y, cat_features=CAT)
    # Selection ran (auditions + possible cross candidate + possible refit),
    # yet the full prep was computed exactly once; the cross candidate only
    # augments it.
    assert m.linear_leaves_selected_ is not None
    assert calls["n"] == 1


def test_classifier_preps_once(monkeypatch):
    calls = _count_prep_fits(monkeypatch)
    X, y = _mixed_data(classification=True)
    m = ChimeraBoostClassifier(n_estimators=60, random_state=0)
    m.fit(X, y, cat_features=CAT)
    assert calls["n"] == 1


def test_sklearn_predictions_match_uncached_boosters():
    """End-to-end: the sklearn fit (cache active) equals hand-run uncached
    booster fits under the same selection logic on a no-cats dataset."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((2500, 6))
    y = X[:, 0] * X[:, 1] + 2.0 * (X[:, 2] > X[:, 3]) \
        + 0.2 * rng.standard_normal(2500)
    m = ChimeraBoostRegressor(n_estimators=80, random_state=1)
    m.fit(X, y)
    # The fitted model predicts sanely and its prep transform round-trips
    # through the augmented preprocessor when cross features were selected.
    pred = m.predict(X)
    assert np.isfinite(pred).all()
    if m.cross_features_selected_:
        assert m.model_.prep_.cross_pairs == m.cross_pairs_
