"""Full-data refit (``refit_full=True``; benchmarks/REFIT_PLAN.md)."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


def _reg_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = 2.0 * X[:, 0] + X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(n)
    return X, y


def _clf_data(n=3000, k=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    score = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.5 * rng.standard_normal(n)
    if k == 2:
        y = (score > 0).astype(int)
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    return X, y


def test_default_off_is_identical():
    X, y = _reg_data()
    a = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              refit_full=False).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_refit_changes_predictions_and_extends_rounds():
    X, y = _reg_data()
    base = ChimeraBoostRegressor(n_estimators=200, random_state=0).fit(X, y)
    re = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                               refit_full=True).fit(X, y)
    t_star = len(base.model_.trees_)
    expect = min(int(np.ceil(t_star / (1 - re.validation_fraction))), 200)
    assert len(re.model_.trees_) == expect
    assert not np.array_equal(base.predict(X), re.predict(X))
    # The ES fit's curve is preserved for introspection.
    assert re.validation_history_ == base.validation_history_


def test_refit_trains_on_the_holdout_rows():
    # The RMSE booster's init is the mean of ITS training targets: the base
    # model's is the 80% split's mean, the refit's must be the full-data mean.
    X, y = _reg_data(n=2500)
    base = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
    re = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                               refit_full=True).fit(X, y)
    assert re.model_.init_ == pytest.approx(float(np.mean(y)), abs=1e-12)
    assert base.model_.init_ != pytest.approx(float(np.mean(y)), abs=1e-9)


def test_noop_with_explicit_eval_set():
    X, y = _reg_data()
    ev = (X[:400], y[:400])
    a = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(
        X[400:], y[400:], eval_set=ev)
    b = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              refit_full=True).fit(X[400:], y[400:],
                                                   eval_set=ev)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_noop_without_early_stopping():
    X, y = _reg_data()
    a = ChimeraBoostRegressor(n_estimators=60, early_stopping=False,
                              random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=60, early_stopping=False,
                              random_state=0, refit_full=True).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_quantile_keeps_conformal_holdout():
    X, y = _reg_data()
    a = ChimeraBoostRegressor(loss="Quantile", alpha=0.9, n_estimators=80,
                              random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(loss="Quantile", alpha=0.9, n_estimators=80,
                              random_state=0, refit_full=True).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert b.quantile_offset_ == a.quantile_offset_


def test_binary_classifier_refit_and_temperature_transfer():
    X, y = _clf_data()
    base = ChimeraBoostClassifier(n_estimators=150, random_state=0).fit(X, y)
    re = ChimeraBoostClassifier(n_estimators=150, random_state=0,
                                refit_full=True).fit(X, y)
    assert re.temperature_ == base.temperature_   # calibrated pre-refit
    p = re.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert not np.array_equal(base.predict_proba(X), p)
    assert len(re.model_.trees_) >= len(base.model_.trees_)


def test_multiclass_refit():
    X, y = _clf_data(k=3)
    re = ChimeraBoostClassifier(n_estimators=120, random_state=0,
                                refit_full=True).fit(X, y)
    base = ChimeraBoostClassifier(n_estimators=120, random_state=0).fit(X, y)
    assert re.predict_proba(X).shape == (len(X), 3)
    assert set(np.unique(re.predict(X))) <= set(np.unique(y))
    assert len(re.model_.trees_) >= len(base.model_.trees_)


def test_bagged_members_unaffected():
    X, y = _reg_data(n=2500)
    a = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              n_ensembles=3, ensemble_n_jobs=1).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=80, random_state=0, n_ensembles=3,
                              ensemble_n_jobs=1, refit_full=True).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_refit_full_validation():
    X, y = _reg_data(n=500)
    with pytest.raises(ValueError, match="refit_full"):
        ChimeraBoostRegressor(refit_full="yes").fit(X, y)
