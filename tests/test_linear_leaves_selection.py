"""Validation-selected linear leaves (regressor ``linear_leaves=None``)."""

import numpy as np

from chimeraboost import ChimeraBoostRegressor


def _data(n=3000, smooth=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    if smooth:
        y = 2.0 * X[:, 0] + X[:, 1] * X[:, 2] + 0.1 * rng.standard_normal(n)
    else:
        y = (X[:, 0] > 0).astype(float) + 0.5 * rng.standard_normal(n)
    return X, y


def _fit(ll, X, y, **kw):
    m = ChimeraBoostRegressor(n_estimators=150, linear_leaves=ll,
                              random_state=0, **kw)
    return m.fit(X, y)


def test_selection_winner_matches_the_explicit_variant():
    X, y = _data()
    m = _fit(None, X, y)
    assert m.linear_leaves_selected_ in (True, False)
    explicit = _fit(m.linear_leaves_selected_, X, y)
    np.testing.assert_array_equal(m.predict(X), explicit.predict(X))


def test_selection_prefers_linear_on_smooth_data():
    X, y = _data(smooth=True)
    m = _fit(None, X, y)
    assert m.linear_leaves_selected_ is True


def test_no_selection_below_min_samples_or_without_validation():
    X, y = _data(n=600)
    m = _fit(None, X, y)
    assert m.linear_leaves_selected_ is None
    np.testing.assert_array_equal(m.predict(X), _fit(False, X, y).predict(X))

    X, y = _data(n=3000)
    m = _fit(None, X, y, early_stopping=False)
    assert m.linear_leaves_selected_ is None
    np.testing.assert_array_equal(
        m.predict(X), _fit(False, X, y, early_stopping=False).predict(X))


def test_no_selection_for_mae_loss():
    X, y = _data()
    m = _fit(None, X, y, loss="MAE")
    assert m.linear_leaves_selected_ is None


def test_default_false_behavior_unchanged():
    X, y = _data()
    a = _fit(False, X, y)
    b = ChimeraBoostRegressor(n_estimators=150, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert b.linear_leaves_selected_ is None
