"""Size-adaptive auto learning rate (benchmarks/SMALLDATA_PLAN.md, C4).

CatBoost's only size-dependent default is its learning rate, and denying it
that schedule costs it 57% of its edge over us at quarter size. Sweeping our
own rate found a knee at 0.07: +0.275% of the primary metric at quarter size
for 1.21x fit, while at full size the same arm is a wash that would cost
1.28x. ``adaptive_learning_rate=True`` therefore fades the auto rate from 0.07
on small data back to the historical 0.1 on large data.

These tests lock the CONTRACT, not the accuracy: the default is off and off is
byte-identical, an explicit rate still wins, the fade hits its documented
endpoints, an unknown row count degrades to the historical default, and the
rate that chose the early-stopped budget is the rate the refit replays at.
"""

import numpy as np
import pytest
from sklearn.base import clone

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import (_AUTO_LR_HI, _AUTO_LR_LARGE, _AUTO_LR_LO,
                                  _AUTO_LR_SMALL, _auto_learning_rate)

FIT = dict(n_estimators=150, early_stopping_rounds=20, random_state=0)


def _data(n=1200, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.normal(scale=0.4, size=n)
    return X, y


# --------------------------------------------------------------------------
# The fade function itself
# --------------------------------------------------------------------------

def test_fade_hits_its_documented_endpoints():
    small = _auto_learning_rate(2000, True, n_train=_AUTO_LR_LO, adaptive=True)
    large = _auto_learning_rate(2000, True, n_train=_AUTO_LR_HI, adaptive=True)
    assert small == pytest.approx(_AUTO_LR_SMALL)
    assert large == pytest.approx(_AUTO_LR_LARGE)


def test_fade_is_monotone_and_bounded():
    ns = [10, 500, 2_000, _AUTO_LR_LO, 7_500, 10_000, _AUTO_LR_HI, 50_000]
    rates = [_auto_learning_rate(2000, True, n_train=n, adaptive=True)
             for n in ns]
    assert rates == sorted(rates)
    assert min(rates) == pytest.approx(_AUTO_LR_SMALL)
    assert max(rates) == pytest.approx(_AUTO_LR_LARGE)


def test_unknown_row_count_falls_back_to_the_historical_default():
    """Never silently pick the SMALL rate when the size is unknown."""
    assert _auto_learning_rate(2000, True, n_train=None,
                               adaptive=True) == _AUTO_LR_LARGE


def test_flag_does_not_touch_the_no_early_stopping_branch():
    """Without early stopping the rate already scales with the round budget."""
    for n_est in (50, 500, 2000):
        assert (_auto_learning_rate(n_est, False, n_train=10, adaptive=True)
                == _auto_learning_rate(n_est, False))


def test_off_is_exactly_the_historical_function():
    for n in (10, 5_000, 50_000, None):
        assert _auto_learning_rate(2000, True, n_train=n,
                                   adaptive=False) == _AUTO_LR_LARGE


# --------------------------------------------------------------------------
# Estimator contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_default_is_off(Est):
    assert Est().adaptive_learning_rate is False


def test_off_is_byte_identical_to_an_explicit_flat_rate():
    """The shipped default must not move a single prediction."""
    X, y = _data()
    a = ChimeraBoostRegressor(**FIT).fit(X, y)
    b = ChimeraBoostRegressor(learning_rate=_AUTO_LR_LARGE, **FIT).fit(X, y)
    assert np.array_equal(a.predict(X[:200]), b.predict(X[:200]))


def test_on_engages_and_matches_the_explicit_small_rate():
    """Below the low threshold the fade must be exactly the knee rate."""
    X, y = _data()                      # 1200 rows, well under _AUTO_LR_LO
    off = ChimeraBoostRegressor(**FIT).fit(X, y)
    on = ChimeraBoostRegressor(adaptive_learning_rate=True, **FIT).fit(X, y)
    explicit = ChimeraBoostRegressor(learning_rate=_AUTO_LR_SMALL,
                                     **FIT).fit(X, y)
    assert not np.array_equal(off.predict(X[:200]), on.predict(X[:200]))
    assert np.array_equal(on.predict(X[:200]), explicit.predict(X[:200]))


def test_explicit_learning_rate_still_wins():
    X, y = _data()
    a = ChimeraBoostRegressor(learning_rate=0.2, adaptive_learning_rate=True,
                              **FIT).fit(X, y)
    b = ChimeraBoostRegressor(learning_rate=0.2, **FIT).fit(X, y)
    assert np.array_equal(a.predict(X[:200]), b.predict(X[:200]))


def test_classifier_path_engages():
    X, y = _data()
    yc = (y > np.median(y)).astype(int)
    off = ChimeraBoostClassifier(**FIT).fit(X, yc)
    on = ChimeraBoostClassifier(adaptive_learning_rate=True, **FIT).fit(X, yc)
    assert not np.array_equal(off.predict_proba(X[:200]),
                              on.predict_proba(X[:200]))


def test_refit_replays_at_the_rate_that_chose_the_budget():
    """The full-data refit sees MORE rows than the early-stopping fit, so a
    naive re-resolve would fade to a different rate than the one the round
    budget was chosen at. _refit_on_full pins winner.lr_; lock that."""
    X, y = _data(n=1200)
    m = ChimeraBoostRegressor(adaptive_learning_rate=True, **FIT).fit(X, y)
    assert m.model_.lr_ == pytest.approx(_AUTO_LR_SMALL)


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_get_params_and_clone_roundtrip(Est):
    m = Est(adaptive_learning_rate=True)
    assert m.get_params()["adaptive_learning_rate"] is True
    assert clone(m).get_params()["adaptive_learning_rate"] is True
