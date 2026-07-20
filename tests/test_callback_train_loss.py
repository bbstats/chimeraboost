"""The fit loop only evaluates per-round train loss for callbacks that read it.

Internal selection-audition callbacks are tagged ``_cb_needs_train_loss=False``
and receive ``None``; untagged (user) callbacks keep receiving the float. The
model itself is unaffected either way -- train loss feeds no fit decision.
"""

import numpy as np

from chimeraboost import ChimeraBoostClassifier
from chimeraboost.booster import _callbacks_need_train_loss
from chimeraboost.sklearn_api import _stop_after, _stop_if_behind


def _data(n=3000, f=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=n) > 0)
    return X, y.astype(int)


def test_internal_callbacks_are_tagged():
    assert _stop_after(5)._cb_needs_train_loss is False
    assert _stop_if_behind(5, 1.0)._cb_needs_train_loss is False
    assert _callbacks_need_train_loss(None) is False
    assert _callbacks_need_train_loss([_stop_after(5)]) is False
    assert _callbacks_need_train_loss([_stop_after(5), lambda *a: None]) is True


def test_user_callback_still_receives_train_loss():
    X, y = _data()
    seen = []

    def cb(iteration, train_loss, val_loss, model):
        seen.append(train_loss)

    ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(
        X, y, callbacks=cb)
    assert seen and all(isinstance(t, float) for t in seen)


def test_tagged_callback_receives_none_and_fit_is_identical():
    X, y = _data()
    seen = []

    def spy(iteration, train_loss, val_loss, model):
        seen.append(train_loss)
    spy._cb_needs_train_loss = False

    a = ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(
        X, y, callbacks=spy)
    b = ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert seen and all(t is None for t in seen)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
