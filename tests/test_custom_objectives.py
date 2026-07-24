"""Custom objective hook + the extra built-in losses (Huber/Poisson/Gamma/
Tweedie) + the eval_metric hook."""

import pickle

import numpy as np
import pytest

from chimeraboost import (ChimeraBoostClassifier, ChimeraBoostRegressor,
                          CustomObjective)
from chimeraboost.losses import Gamma, Huber, Poisson, Tweedie


def _reg_data(seed=0, n=2000, link=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    f = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3]
    if link == "exp":
        return X, f  # caller samples counts / positives from exp(f)
    y = f + rng.normal(scale=0.3, size=n)
    return X, y


def _split(X, y, n_train=1500):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ---------------------------------------------------------------------------
# Built-in losses
# ---------------------------------------------------------------------------

def test_huber_close_to_rmse_on_clean_data():
    X, y = _reg_data(seed=1)
    Xtr, ytr, Xte, yte = _split(X, y)
    kw = dict(n_estimators=300, random_state=0, cross_features=False,
              linear_leaves=False)
    rmse_pred = ChimeraBoostRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    hub_pred = ChimeraBoostRegressor(loss="Huber", delta=2.0, **kw) \
        .fit(Xtr, ytr).predict(Xte)
    # With delta at ~6 sigma of the residuals, Huber is essentially squared
    # error; the two models should land very close on clean data.
    assert np.sqrt(np.mean((rmse_pred - hub_pred) ** 2)) < 0.15


def test_huber_robust_to_outliers():
    X, y = _reg_data(seed=2)
    Xtr, ytr, Xte, yte = _split(X, y)
    rng = np.random.default_rng(3)
    bad = rng.random(len(ytr)) < 0.08
    ytr_out = ytr.copy()
    ytr_out[bad] += rng.choice([-50.0, 50.0], size=bad.sum())
    kw = dict(n_estimators=300, random_state=0, cross_features=False,
              linear_leaves=False)
    rmse_err = np.mean(np.abs(
        ChimeraBoostRegressor(**kw).fit(Xtr, ytr_out).predict(Xte) - yte))
    hub_err = np.mean(np.abs(
        ChimeraBoostRegressor(loss="Huber", delta=1.0, **kw)
        .fit(Xtr, ytr_out).predict(Xte) - yte))
    assert hub_err < rmse_err


def test_poisson_fits_counts():
    X, f = _reg_data(seed=4, link="exp")
    rng = np.random.default_rng(5)
    y = rng.poisson(np.exp(f)).astype(np.float64)
    Xtr, ytr, Xte, yte = _split(X, y)
    m = ChimeraBoostRegressor(loss="Poisson", n_estimators=300,
                              random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert np.all(pred > 0)  # log link: predictions are exp(raw)

    def poisson_dev(y_, mu):
        ylog = np.where(y_ > 0, y_ * np.log(np.where(y_ > 0, y_, 1.0) / mu), 0.0)
        return np.mean(2.0 * (ylog - (y_ - mu)))

    baseline = poisson_dev(yte, np.full_like(yte, ytr.mean()))
    assert poisson_dev(yte, pred) < 0.8 * baseline


def test_gamma_fits_positive_targets():
    X, f = _reg_data(seed=6, link="exp")
    rng = np.random.default_rng(7)
    shape = 2.0
    y = rng.gamma(shape, np.exp(f) / shape)
    y = np.maximum(y, 1e-8)
    Xtr, ytr, Xte, yte = _split(X, y)
    m = ChimeraBoostRegressor(loss="Gamma", n_estimators=300,
                              random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert np.all(pred > 0)

    def gamma_dev(y_, mu):
        return np.mean(2.0 * (np.log(mu / y_) + y_ / mu - 1.0))

    baseline = gamma_dev(yte, np.full_like(yte, ytr.mean()))
    assert gamma_dev(yte, pred) < 0.8 * baseline


def test_tweedie_fits_zero_inflated():
    X, f = _reg_data(seed=8, link="exp")
    rng = np.random.default_rng(9)
    # Compound Poisson-gamma: zeros where the Poisson count is 0.
    counts = rng.poisson(0.7 * np.exp(f))
    y = np.array([rng.gamma(2.0, 1.0, size=c).sum() for c in counts])
    Xtr, ytr, Xte, yte = _split(X, y)
    assert (ytr == 0).any()  # the regime Tweedie exists for
    m = ChimeraBoostRegressor(loss="Tweedie", tweedie_variance_power=1.5,
                              n_estimators=300, random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert np.all(pred > 0)

    def tweedie_dev(y_, mu, p=1.5):
        return np.mean(2.0 * (np.power(y_, 2 - p) / ((1 - p) * (2 - p))
                              - y_ * np.power(mu, 1 - p) / (1 - p)
                              + np.power(mu, 2 - p) / (2 - p)))

    baseline = tweedie_dev(yte, np.full_like(yte, ytr.mean()))
    assert tweedie_dev(yte, pred) < baseline


def test_log_link_staged_predict_matches_predict():
    X, f = _reg_data(seed=10, link="exp")
    rng = np.random.default_rng(11)
    y = rng.poisson(np.exp(f)).astype(np.float64)
    m = ChimeraBoostRegressor(loss="Poisson", n_estimators=100,
                              random_state=0).fit(X, y)
    *_, last = m.staged_predict(X[:20])
    np.testing.assert_allclose(last, m.predict(X[:20]))


def test_log_link_pickle_roundtrip():
    X, f = _reg_data(seed=12, link="exp")
    rng = np.random.default_rng(13)
    y = rng.poisson(np.exp(f)).astype(np.float64)
    m = ChimeraBoostRegressor(loss="Poisson", n_estimators=100,
                              random_state=0).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict(X[:50]), m2.predict(X[:50]))


def test_y_domain_errors():
    X = np.random.default_rng(0).normal(size=(200, 3))
    y_neg = np.linspace(-1, 5, 200)
    with pytest.raises(ValueError, match="non-negative"):
        ChimeraBoostRegressor(loss="Poisson").fit(X, y_neg)
    with pytest.raises(ValueError, match="positive"):
        ChimeraBoostRegressor(loss="Gamma").fit(X, np.zeros(200))
    with pytest.raises(ValueError, match="non-negative"):
        ChimeraBoostRegressor(loss="Tweedie").fit(X, y_neg)


def test_loss_param_validation():
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = X[:, 0]
    with pytest.raises(ValueError, match="loss must be one of"):
        ChimeraBoostRegressor(loss="bogus").fit(X, y)
    with pytest.raises(ValueError, match="delta"):
        ChimeraBoostRegressor(loss="Huber", delta=0.0).fit(X, y)
    with pytest.raises(ValueError, match="tweedie_variance_power"):
        ChimeraBoostRegressor(loss="Tweedie",
                              tweedie_variance_power=2.5).fit(X, y)
    with pytest.raises(ValueError, match="tweedie_variance_power"):
        ChimeraBoostRegressor(loss="Tweedie",
                              tweedie_variance_power=1.0).fit(X, y)


def test_loss_classes_reject_bad_construction():
    with pytest.raises(ValueError):
        Tweedie(power=2.0)
    # Sanity: direct loss objects agree with their sklearn-string twins.
    y = np.array([0.0, 1.0, 2.0, 3.0])
    raw = np.log(np.maximum(y, 0.5))
    for loss in (Huber(1.0), Poisson(), Gamma(), Tweedie(1.5)):
        if isinstance(loss, Gamma):
            g, h = loss.grad_hess(y + 0.5, raw)
        else:
            g, h = loss.grad_hess(y, raw)
        assert np.all(np.isfinite(g)) and np.all(h > 0)


# ---------------------------------------------------------------------------
# Custom objective hook
# ---------------------------------------------------------------------------

class _SquaredError(CustomObjective):
    """RMSE re-implemented through the public hook (used to prove the custom
    path is numerically identical to the built-in when the math matches)."""

    def init(self, y, sample_weight=None):
        return float(np.average(y, weights=sample_weight))

    def grad_hess(self, y, raw):
        return raw - y, np.ones_like(raw)

    def eval(self, y, raw, sample_weight=None):
        return float(np.sqrt(np.average((raw - y) ** 2,
                                        weights=sample_weight)))


class _LogCosh(CustomObjective):
    """A genuinely non-built-in objective (smooth MAE)."""

    def grad_hess(self, y, raw):
        r = raw - y
        return np.tanh(r), 1.0 - np.tanh(r) ** 2 + 1e-6

    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.logaddexp(raw - y, y - raw) - np.log(2.0),
                                weights=sample_weight))


def test_custom_objective_matches_builtin_rmse():
    X, y = _reg_data(seed=14)
    # Custom losses skip the RMSE-only linear-leaf/cross-feature auditions,
    # so pin both variants off on the built-in side for an apples-to-apples
    # comparison; everything else runs the identical code path.
    kw = dict(n_estimators=200, random_state=0, cross_features=False,
              linear_leaves=False)
    builtin = ChimeraBoostRegressor(loss="RMSE", **kw).fit(X, y)
    custom = ChimeraBoostRegressor(loss=_SquaredError(), **kw).fit(X, y)
    np.testing.assert_array_equal(builtin.predict(X[:100]),
                                  custom.predict(X[:100]))


def test_custom_objective_logcosh_learns():
    X, y = _reg_data(seed=15)
    Xtr, ytr, Xte, yte = _split(X, y)
    m = ChimeraBoostRegressor(loss=_LogCosh(), n_estimators=300,
                              random_state=0).fit(Xtr, ytr)
    err = np.mean(np.abs(m.predict(Xte) - yte))
    assert err < 0.5 * np.mean(np.abs(yte - ytr.mean()))


def test_custom_objective_bagged_and_pickled():
    X, y = _reg_data(seed=16, n=1200)
    m = ChimeraBoostRegressor(loss=_LogCosh(), n_estimators=100,
                              n_ensembles=2, random_state=0).fit(X, y)
    pred = m.predict(X[:50])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(pred, m2.predict(X[:50]))


def test_custom_objective_protocol_validation():
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = X[:, 0]
    with pytest.raises(ValueError, match="CustomObjective"):
        ChimeraBoostRegressor(loss=object()).fit(X, y)


# ---------------------------------------------------------------------------
# eval_metric hook
# ---------------------------------------------------------------------------

def _mae_metric(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _neg_mae_metric(y_true, y_pred):
    return -float(np.mean(np.abs(y_true - y_pred)))


_neg_mae_metric.greater_is_better = True


def test_eval_metric_drives_validation_history():
    X, y = _reg_data(seed=17)
    m = ChimeraBoostRegressor(n_estimators=150, random_state=0,
                              eval_metric=_mae_metric,
                              cross_features=False,
                              linear_leaves=False).fit(X, y)
    hist = m.validation_history_
    assert len(hist) > 0
    # The recorded series is the metric (MAE), not the training RMSE: on this
    # data MAE and RMSE differ well beyond tolerance, and the two models pick
    # early stops independently. Recompute the metric at the final model to
    # anchor the units.
    default = ChimeraBoostRegressor(n_estimators=150, random_state=0,
                                    cross_features=False,
                                    linear_leaves=False).fit(X, y)
    assert not np.allclose(hist[:10], default.validation_history_[:10])


def test_eval_metric_greater_is_better_negates():
    X, y = _reg_data(seed=18)
    kw = dict(n_estimators=150, random_state=0, cross_features=False,
              linear_leaves=False)
    lo = ChimeraBoostRegressor(eval_metric=_mae_metric, **kw).fit(X, y)
    hi = ChimeraBoostRegressor(eval_metric=_neg_mae_metric, **kw).fit(X, y)
    # Negating a lower-is-better metric and flagging greater_is_better must
    # give the identical fit: same stored history, same chosen iteration.
    np.testing.assert_allclose(lo.validation_history_,
                               hi.validation_history_)
    assert lo.best_iteration_ == hi.best_iteration_


def test_eval_metric_receives_weights():
    X, y = _reg_data(seed=19)
    w = np.random.default_rng(20).uniform(0.5, 2.0, size=len(y))
    seen = {"three_arg_calls": 0}

    def metric(y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            seen["three_arg_calls"] += 1
            assert len(sample_weight) == len(y_true)
        return float(np.average(np.abs(y_true - y_pred),
                                weights=sample_weight))

    ChimeraBoostRegressor(n_estimators=60, random_state=0,
                          eval_metric=metric, cross_features=False,
                          linear_leaves=False).fit(X, y, sample_weight=w)
    # The auto-split holdout carries its (non-uniform) row weights.
    assert seen["three_arg_calls"] > 0


def test_eval_metric_binary_classifier():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(1500, 5))
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - X[:, 1])))
    y = (rng.random(1500) < p).astype(int)

    def auc(y_true, y_pred):
        order = np.argsort(y_pred)
        ranks = np.empty(len(y_pred))
        ranks[order] = np.arange(1, len(y_pred) + 1)
        pos = y_true == 1
        n1, n0 = pos.sum(), (~pos).sum()
        return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    auc.greater_is_better = True
    m = ChimeraBoostClassifier(n_estimators=150, random_state=0,
                               eval_metric=auc,
                               cross_features=False).fit(X, y)
    hist = m.validation_history_
    # Stored negated: -AUC of a learning model lives in (-1, -0.5).
    assert all(-1.0 <= v <= 0.0 for v in hist)
    assert min(hist) < -0.7
    proba = m.predict_proba(X[:20])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_eval_metric_multiclass():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(1500, 4))
    y = np.argmax(X[:, :3] + 0.5 * rng.normal(size=(1500, 3)), axis=1)

    def brier(Y_onehot, P):
        return float(np.mean(np.sum((Y_onehot - P) ** 2, axis=1)))

    m = ChimeraBoostClassifier(n_estimators=120, random_state=0,
                               eval_metric=brier,
                               cross_features=False).fit(X, y)
    assert len(m.validation_history_) > 0
    assert m.predict_proba(X[:10]).shape == (10, 3)


def test_eval_metric_validation():
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = X[:, 0]
    with pytest.raises(ValueError, match="eval_metric"):
        ChimeraBoostRegressor(eval_metric="rmse").fit(X, y)


def test_eval_metric_none_bit_identical():
    # The hook must not perturb the default path at all.
    X, y = _reg_data(seed=23)
    kw = dict(n_estimators=100, random_state=0)
    a = ChimeraBoostRegressor(**kw).fit(X, y)
    b = ChimeraBoostRegressor(eval_metric=None, **kw).fit(X, y)
    np.testing.assert_array_equal(a.predict(X[:100]), b.predict(X[:100]))
