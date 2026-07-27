"""Named operating points (``quality=1..5``; benchmarks/SELECT_PLAN.md).

``quality`` only pins parameters that already exist, so the tests that matter
are: the default path is untouched, each rung is reachable on both estimators
(multiclass included), the recipe wins a clash loudly, and the bagged rungs do
not re-apply themselves inside their own members.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.sklearn_api import QUALITY_NAMES


def _reg_data(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = 2.0 * X[:, 0] + X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(n)
    return X, y


def _clf_data(n=2500, k=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    score = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.5 * rng.standard_normal(n)
    if k == 2:
        y = (score > 0).astype(int)
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    return X, y


def test_default_none_is_identical():
    """quality=None must reproduce the pre-parameter model bit for bit."""
    X, y = _reg_data()
    a = ChimeraBoostRegressor(n_estimators=120, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=120, random_state=0,
                              quality=None).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_accurate_rung_is_the_default():
    """Since 0.25.0 the default is rung 3 -- the strongest non-bagged rung."""
    X, y = _reg_data()
    a = ChimeraBoostRegressor(n_estimators=120, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=120, random_state=0,
                              quality=3).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_balanced_rung_is_the_pre_025_default():
    """Rung 2 is the old default: the search, without the full-data refit."""
    X, y = _reg_data()
    a = ChimeraBoostRegressor(n_estimators=120, random_state=0,
                              refit_full=False).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=120, random_state=0,
                              quality=2).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    # ...and it is genuinely cheaper than the default now is.
    c = ChimeraBoostRegressor(n_estimators=120, random_state=0).fit(X, y)
    assert not np.array_equal(b.predict(X), c.predict(X))


@pytest.mark.parametrize("q", sorted(QUALITY_NAMES))
def test_regressor_rungs_fit_and_predict(q):
    X, y = _reg_data()
    m = ChimeraBoostRegressor(n_estimators=80, quality=q,
                              ensemble_n_jobs=1, random_state=0).fit(X, y)
    assert np.isfinite(m.predict(X[:20])).all()


@pytest.mark.parametrize("q", sorted(QUALITY_NAMES))
@pytest.mark.parametrize("k", [2, 3])
def test_classifier_rungs_fit_and_predict(q, k):
    """Rung 1 must not trip the multiclass linear-leaves guard."""
    X, y = _clf_data(k=k)
    m = ChimeraBoostClassifier(n_estimators=80, quality=q,
                               ensemble_n_jobs=1, random_state=0).fit(X, y)
    p = m.predict_proba(X[:20])
    assert p.shape == (20, k)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_fast_rung_pins_the_search_off():
    X, y = _reg_data()
    m = ChimeraBoostRegressor(n_estimators=80, quality=1,
                              random_state=0).fit(X, y)
    # One booster fit: nothing was auditioned, so no selection was recorded.
    assert m.cross_features_selected_ is None
    assert m.linear_leaves_selected_ is None


def test_fit_does_not_rewrite_constructor_params():
    """sklearn requires get_params to keep returning what the user passed."""
    X, y = _reg_data()
    m = ChimeraBoostRegressor(n_estimators=80, quality=1, random_state=0)
    m.fit(X, y)
    p = m.get_params()
    assert p["quality"] == 1
    assert p["cross_features"] is None      # pinned only for the fit's duration
    assert p["linear_leaves"] is None
    assert p["refit_full"] == "replay"      # restored to the class default


def test_params_restored_even_when_fit_raises():
    X, y = _reg_data(n=300)
    m = ChimeraBoostRegressor(quality=1, n_estimators=80, random_state=0)
    with pytest.raises(ValueError):
        m.fit(X, y[:-1])                    # length mismatch
    assert m.get_params()["cross_features"] is None


def test_clash_warns_and_the_recipe_wins():
    X, y = _reg_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = ChimeraBoostRegressor(n_estimators=80, quality=1,
                                  cross_features=True, random_state=0).fit(X, y)
    msgs = [str(w.message) for w in caught if "quality=1" in str(w.message)]
    assert msgs, "a clash must warn"
    assert "cross_features=False" in msgs[0]
    # The recipe won: the search was skipped despite cross_features=True.
    assert m.cross_features_selected_ is None


def test_no_warning_when_the_user_agrees_with_the_recipe():
    X, y = _reg_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ChimeraBoostRegressor(n_estimators=80, quality=1, cross_features=False,
                              random_state=0).fit(X, y)
    assert not [w for w in caught if "quality=1" in str(w.message)]


def test_bagged_rungs_do_not_recurse():
    """Members must be built with quality=None, or rungs 4/5 nest forever."""
    X, y = _reg_data()
    m = ChimeraBoostRegressor(n_estimators=60, quality=4, ensemble_n_jobs=1,
                              random_state=0).fit(X, y)
    assert len(m.estimators_) == 5
    for member in m.estimators_:
        assert member.get_params()["quality"] is None
        assert member.get_params()["n_ensembles"] is None


@pytest.mark.parametrize("bad", [0, 6, 2.5, True, "fast", -1])
def test_invalid_quality_rejected(bad):
    X, y = _reg_data(n=300)
    with pytest.raises(ValueError, match="quality"):
        ChimeraBoostRegressor(quality=bad).fit(X, y)


def test_clone_round_trip():
    e = ChimeraBoostRegressor(quality=4)
    assert clone(e).get_params()["quality"] == 4
