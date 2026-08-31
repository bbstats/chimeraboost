"""The shipped self-scoring layer, and the predict kinds that read the grid.

`chimeraboost.metrics` has to agree with sklearn and with
`benchmarks/run_benchmarks.py` exactly -- a number a user reads from
`model.report()` should be comparable with the project's own tables -- so most
of these tests are equivalence checks against those definitions rather than
self-consistency checks.
"""

import numpy as np
import pytest

from chimeraboost import (ChimeraBoostClassifier, ChimeraBoostQuantileRegressor,
                          ChimeraBoostRegressor, metrics)


def _reg_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = X[:, 0] * 2 + X[:, 1] + 0.5 * rng.standard_normal(n)
    return X, y


def _clf_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = (X[:, 0] + X[:, 2] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    return X, y


# --- regression ---------------------------------------------------------------

def test_regression_report_matches_sklearn():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rng = np.random.default_rng(1)
    y = rng.standard_normal(500)
    pred = y + 0.3 * rng.standard_normal(500)

    rep = metrics.regression_report(y, pred)
    assert rep["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, pred)))
    assert rep["mae"] == pytest.approx(mean_absolute_error(y, pred))
    # With no explicit baseline the reference mean is y's own, which is
    # exactly sklearn's R2.
    assert rep["r2"] == pytest.approx(r2_score(y, pred))
    assert rep["n"] == 500


def test_regression_r2_is_zero_for_the_mean_and_one_for_the_truth():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.regression_report(y, np.full(4, y.mean()))["r2"] == \
        pytest.approx(0.0)
    assert metrics.regression_report(y, y)["r2"] == pytest.approx(1.0)
    # Worse than the mean goes negative, which is the point of a skill score.
    assert metrics.regression_report(y, np.full(4, 100.0))["r2"] < 0.0


def test_regression_r2_against_a_training_baseline_differs_from_hindsight():
    """Scoring against the mean the model actually had is not the same as the
    hindsight mean of the scored fold, and the option exists to say which."""
    y = np.array([10.0, 11.0, 12.0])
    pred = np.array([10.5, 11.0, 11.5])
    hindsight = metrics.regression_report(y, pred)["r2"]
    shifted = metrics.regression_report(y, pred, baseline=np.zeros(50))["r2"]
    assert shifted != pytest.approx(hindsight)


def test_regression_report_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="pred has shape"):
        metrics.regression_report(np.zeros(4), np.zeros(3))


def test_regression_r2_is_nan_when_the_target_is_constant():
    rep = metrics.regression_report(np.ones(5), np.ones(5))
    assert np.isnan(rep["r2"]) and rep["rmse"] == 0.0


# --- classification -----------------------------------------------------------

def test_classification_report_matches_sklearn_and_the_harness():
    from sklearn.metrics import f1_score, log_loss
    X, y = _clf_data()
    m = ChimeraBoostClassifier(random_state=0, n_estimators=60).fit(X, y)
    proba = m.predict_proba(X)
    rep = metrics.classification_report(y, proba, m.classes_)

    assert rep["log_loss"] == pytest.approx(
        log_loss(y, proba, labels=m.classes_))
    # The harness's multiclass Brier, verbatim.
    onehot = (y[:, None] == m.classes_[None, :]).astype(float)
    assert rep["brier"] == pytest.approx(
        float(np.mean(np.sum((proba - onehot) ** 2, axis=1))))
    pred = m.classes_[proba.argmax(axis=1)]
    assert rep["f1_macro"] == pytest.approx(
        f1_score(y, pred, average="macro"))


def test_brier_skill_is_one_for_a_perfect_forecast_and_zero_for_the_prior():
    y = np.array([0, 1, 0, 1, 1])
    classes = np.array([0, 1])
    perfect = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [0., 1.]])
    assert metrics.classification_report(y, perfect, classes)["brier_skill"] \
        == pytest.approx(1.0)

    prior = np.tile([0.4, 0.6], (5, 1))
    assert metrics.classification_report(y, prior, classes)["brier_skill"] \
        == pytest.approx(0.0)


def test_miscalibration_is_zero_for_a_calibrated_forecast_and_positive_when_skewed():
    """A monotone recalibration cannot improve an already-calibrated forecast,
    so MCB is 0; squashing the probabilities toward the middle leaves room."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 8000)
    y = (rng.uniform(size=8000) < p).astype(int)
    classes = np.array([0, 1])

    honest = np.column_stack([1 - p, p])
    assert metrics.classification_report(y, honest, classes)[
        "calibration_mcb"] < 0.002

    # Same ranking, wrong scale: isotonic can fix it, so MCB reports the gap.
    bad = 0.5 + (p - 0.5) * 0.3
    skewed = np.column_stack([1 - bad, bad])
    assert metrics.classification_report(y, skewed, classes)[
        "calibration_mcb"] > 0.01


def test_classification_report_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="proba must be"):
        metrics.classification_report(np.zeros(3), np.zeros((3, 5)),
                                      np.array([0, 1]))
    with pytest.raises(ValueError, match="rows but y has"):
        metrics.classification_report(np.zeros(3), np.zeros((4, 2)),
                                      np.array([0, 1]))


# --- the estimators' own report() ---------------------------------------------

def test_estimators_report_and_format():
    X, y = _reg_data()
    r = ChimeraBoostRegressor(random_state=0, n_estimators=60).fit(X, y)
    rep = r.report(X, y)
    assert set(rep) == {"n", "rmse", "mae", "r2"}
    assert rep["r2"] > 0.9
    assert "R2 skill" in metrics.format_report(rep, "t")

    Xc, yc = _clf_data()
    c = ChimeraBoostClassifier(random_state=0, n_estimators=60).fit(Xc, yc)
    crep = c.report(Xc, yc)
    assert set(crep) == {"n", "log_loss", "brier", "brier_skill", "accuracy",
                         "f1_macro", "calibration_mcb"}
    assert "miscalibration" in metrics.format_report(crep)


def test_report_honours_sample_weight():
    X, y = _reg_data(n=600)
    r = ChimeraBoostRegressor(random_state=0, n_estimators=40).fit(X, y)
    w = np.ones(len(y))
    w[:300] = 0.0
    assert r.report(X, y, sample_weight=w)["rmse"] == pytest.approx(
        r.report(X[300:], y[300:])["rmse"])


# --- predict kinds ------------------------------------------------------------

def _q(quantiles=None, n=1500):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 2 + 0.5 * rng.standard_normal(n)
    m = ChimeraBoostQuantileRegressor(quantiles=quantiles, random_state=0,
                                      n_estimators=80).fit(X, y)
    return m, X[:200]


def test_median_kind_is_the_conformal_centre():
    m, Xt = _q()
    Q = m.predict(Xt)
    k = int(np.argmin(np.abs(m.quantiles_ - 0.5)))
    assert np.array_equal(m.predict(Xt, kind="median"), Q[:, k])


def test_median_kind_interpolates_when_the_grid_omits_one_half():
    m, Xt = _q(quantiles=[0.1, 0.4, 0.6, 0.9])
    Q = m.predict(Xt)
    got = m.predict(Xt, kind="median")
    assert np.allclose(got, Q[:, 1] + 0.5 * (Q[:, 2] - Q[:, 1]))
    assert np.all((got > Q[:, 1]) & (got < Q[:, 2]))


def test_cdf_inverts_the_grid():
    m, Xt = _q()
    Q = m.predict(Xt)
    taus = m.quantiles_
    # Evaluated at a fitted level's own value, the CDF must return that level.
    for k in (0, 5, 9, 18):
        got = np.array([m.predict(Xt[i:i + 1], kind="cdf",
                                  thresholds=[Q[i, k]])[0, 0]
                        for i in range(10)])
        assert np.allclose(got, taus[k]), k


def test_cdf_is_monotone_and_clamped_to_the_fitted_range():
    m, Xt = _q()
    cdf = m.predict(Xt, kind="cdf", thresholds=[-100.0, -1.0, 0.0, 1.0, 100.0])
    assert cdf.shape == (200, 5)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)
    # Clamped to the outermost fitted levels, not to 0 and 1: the grid says
    # nothing about the tails beyond it.
    assert np.allclose(cdf[:, 0], m.quantiles_[0])
    assert np.allclose(cdf[:, -1], m.quantiles_[-1])


def test_cdf_accepts_a_scalar_threshold():
    m, Xt = _q()
    assert m.predict(Xt, kind="cdf", thresholds=0.0).shape == (200, 1)


def test_cdf_reads_2d_thresholds_row_against_row():
    m, Xt = _q()
    t = np.column_stack([np.linspace(-2.0, 2.0, 200), np.full(200, 0.5)])
    got = m.predict(Xt, kind="cdf", thresholds=t)
    assert got.shape == (200, 2)
    # Row i of a 2-D thresholds array must answer exactly as row i asked with
    # its own thresholds shared -- per-row is the same inversion, not new math.
    for i in (0, 7, 199):
        assert np.array_equal(
            got[i], m.predict(Xt[i:i + 1], kind="cdf", thresholds=t[i])[0])


def test_cdf_2d_thresholds_must_carry_one_row_per_prediction_row():
    m, Xt = _q()
    with pytest.raises(ValueError, match="one row per prediction row"):
        m.predict(Xt, kind="cdf", thresholds=np.zeros((2, 2)))


def test_predict_thresh_is_the_complement_of_the_cdf():
    m, Xt = _q()
    t = [-1.0, 0.0, 1.0]
    cdf = m.predict(Xt, kind="cdf", thresholds=t)
    assert np.array_equal(m.predict_thresh(Xt, t, direction="less"), cdf)
    assert np.array_equal(m.predict_thresh(Xt, t), 1.0 - cdf)


def test_predict_thresh_shapes_follow_the_thresholds():
    m, Xt = _q()
    assert m.predict_thresh(Xt, 0.0).shape == (200,)
    assert m.predict_thresh(Xt, [0.0, 1.0]).shape == (200, 2)
    assert m.predict_thresh(Xt, np.zeros((200, 3))).shape == (200, 3)


def test_predict_thresh_reads_a_level_back_at_its_own_value():
    m, Xt = _q()
    Q = m.predict(Xt)
    taus = m.quantiles_
    # A per-row threshold sitting exactly on a fitted level must read that
    # level back: P(y <= q_k(x)) = tau_k.
    for k in (0, 9, 18):
        got = m.predict_thresh(Xt, Q[:, [k]], direction="less")
        assert np.allclose(got[:, 0], taus[k]), k


def test_predict_thresh_clamps_to_the_fitted_levels():
    m, Xt = _q()
    lo, hi = m.quantiles_[0], m.quantiles_[-1]
    # Beyond the grid the honest answer is the edge level, never 0 or 1.
    assert np.allclose(m.predict_thresh(Xt, -100.0), 1.0 - lo)
    assert np.allclose(m.predict_thresh(Xt, 100.0), 1.0 - hi)
    assert np.allclose(m.predict_thresh(Xt, -100.0, direction="less"), lo)


def test_predict_thresh_refuses_bad_arguments():
    m, Xt = _q()
    with pytest.raises(ValueError, match="direction must be"):
        m.predict_thresh(Xt, 0.0, direction="above")
    with pytest.raises(ValueError, match="needs `thresholds`"):
        m.predict_thresh(Xt, None)


def test_samples_follow_the_predicted_distribution_and_are_seeded():
    m, Xt = _q()
    s = m.predict(Xt, kind="sample", n_samples=4000, random_state=0)
    assert s.shape == (200, 4000)
    assert np.array_equal(
        s, m.predict(Xt, kind="sample", n_samples=4000, random_state=0))
    assert not np.array_equal(
        s, m.predict(Xt, kind="sample", n_samples=4000, random_state=1))

    # The empirical median of the draws must track the predicted median.
    med = m.predict(Xt, kind="median")
    assert np.abs(np.median(s, axis=1) - med).max() < 0.25
    # Draws stay inside the fitted range; the grid cannot speak for the tails.
    Q = m.predict(Xt)
    assert s.min() >= Q[:, 0].min() - 1e-9
    assert s.max() <= Q[:, -1].max() + 1e-9


def test_new_kinds_refuse_missing_arguments():
    m, Xt = _q()
    with pytest.raises(ValueError, match="needs `thresholds`"):
        m.predict(Xt, kind="cdf")
    with pytest.raises(ValueError, match="needs `n_samples`"):
        m.predict(Xt, kind="sample")
    with pytest.raises(ValueError, match="n_samples must be"):
        m.predict(Xt, kind="sample", n_samples=0)
    with pytest.raises(ValueError, match="thresholds must be"):
        m.predict(Xt, kind="cdf", thresholds=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="kind must be"):
        m.predict(Xt, kind="quantile")
