"""Regression tests for the 2026-07-29 bug-hunt fixes: eval_set validation,
calibration weight handling, conformal scaling on asymmetric grids, bagged
rare-class handling, and assorted guards.

Each test names the bug it pins down; all of them failed (silently wrong
results or crashes) before the fixes."""

import warnings

import numpy as np
import pandas as pd
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


def _toy_regression(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _toy_binary(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


# ------------------------------------------------- eval_set target validation


def test_nan_in_eval_y_raises():
    """A NaN in eval y used to zero out early stopping and silently ship a
    one-tree model; now it is rejected up front like training y."""
    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    yv = yv.copy()
    yv[7] = np.nan
    with pytest.raises(ValueError, match="eval_set y"):
        ChimeraBoostRegressor(n_estimators=30).fit(X, y, eval_set=(Xv, yv))


def test_inf_in_eval_y_raises():
    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    yv = yv.copy()
    yv[0] = np.inf
    with pytest.raises(ValueError, match="eval_set y"):
        ChimeraBoostRegressor(n_estimators=30).fit(X, y, eval_set=(Xv, yv))


def test_gamma_eval_y_zero_raises_not_truncates():
    """loss='Gamma' with a zero in eval y makes the eval deviance infinite;
    the early stopper now raises instead of silently keeping round 0."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 5))
    y = np.exp(X[:, 0]) + 0.1
    Xv = rng.normal(size=(200, 5))
    yv = np.exp(Xv[:, 0]) + 0.1
    yv[3] = 0.0                      # finite, so it passes the NaN check...
    with pytest.raises(ValueError, match="[Vv]alidation score"):
        ChimeraBoostRegressor(loss="Gamma", n_estimators=30).fit(
            X, y, eval_set=(Xv, yv))


def test_custom_eval_metric_nan_raises():
    def bad_metric(y_true, y_pred):
        return float("nan")

    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    with pytest.raises(ValueError, match="[Vv]alidation score"):
        ChimeraBoostRegressor(n_estimators=30, eval_metric=bad_metric).fit(
            X, y, eval_set=(Xv, yv))


def test_reordered_eval_set_columns_raise():
    """A reordered eval-set DataFrame is consumed positionally and used to
    silently wreck early stopping; it must raise like predict does."""
    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    dfv = pd.DataFrame(Xv, columns=cols)[list(reversed(cols))]
    with pytest.raises(ValueError, match="feature names"):
        ChimeraBoostRegressor(n_estimators=30).fit(df, y, eval_set=(dfv, yv))

    Xc, yc = _toy_binary(800)
    Xcv, ycv = _toy_binary(200, seed=1)
    dfc = pd.DataFrame(Xc, columns=cols)
    dfcv = pd.DataFrame(Xcv, columns=cols)[list(reversed(cols))]
    with pytest.raises(ValueError, match="feature names"):
        ChimeraBoostClassifier(n_estimators=30).fit(dfc, yc,
                                                    eval_set=(dfcv, ycv))


def test_matching_eval_set_columns_fit_clean():
    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    dfv = pd.DataFrame(Xv, columns=cols)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = ChimeraBoostRegressor(n_estimators=30).fit(df, y,
                                                       eval_set=(dfv, yv))
    assert np.isfinite(m.predict(dfv)).all()


def test_bagged_fit_with_dataframe_and_oob_eval_no_name_warning():
    """Members build ndarray OOB eval sets internally; a DataFrame bag fit
    must not spray name-mismatch warnings from inside the members."""
    X, y = _toy_regression(400)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ChimeraBoostRegressor(n_estimators=20, n_ensembles=3).fit(df, y)
    assert not [w for w in caught if "feature names" in str(w.message)]


def test_shap_background_reordered_columns_raise():
    X, y = _toy_regression(400)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    m = ChimeraBoostRegressor(n_estimators=20).fit(df, y)
    with pytest.raises(ValueError, match="feature names"):
        m.shap_values(df.head(10), X_background=df[list(reversed(cols))])


# -------------------------------------- zero-weight rows vs post-fit calibration


def test_zero_weight_rows_cannot_move_conformal_offset():
    """Zero-weight rows with corrupted targets used to set the conformal
    quantile offset when they landed in the auto-split holdout, shifting
    every prediction; the docstring promises they never influence the model."""
    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    w = np.ones(n)
    bad = rng.choice(n, size=60, replace=False)
    y = y.copy()
    y[bad] += 1000.0                 # corrupted rows...
    w[bad] = 0.0                     # ...that carry zero weight
    m = ChimeraBoostRegressor(loss="Quantile", alpha=0.9,
                              n_estimators=40, random_state=0)
    m.fit(X, y, sample_weight=w)
    assert abs(m.quantile_offset_) < 10.0


def test_uniform_weights_match_unweighted_conformal_offset():
    """The weighted conformal rank must reduce exactly to the unweighted
    k-th order statistic when every weight is 1."""
    rng = np.random.default_rng(4)
    n = 600
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    kw = dict(loss="Quantile", alpha=0.8, n_estimators=40, random_state=0)
    plain = ChimeraBoostRegressor(**kw).fit(X, y)
    ones = ChimeraBoostRegressor(**kw).fit(X, y, sample_weight=np.ones(n))
    assert plain.quantile_offset_ == pytest.approx(ones.quantile_offset_)


def test_zero_weight_rows_cannot_move_temperature():
    from chimeraboost.sklearn_api import _fit_temperature

    rng = np.random.default_rng(5)
    raw = rng.normal(size=200)
    y = (raw + 0.5 * rng.normal(size=200) > 0).astype(np.float64)
    # Garbage rows: confidently wrong labels, weight zero.
    raw_aug = np.concatenate([raw, 8.0 * np.ones(50)])
    y_aug = np.concatenate([y, np.zeros(50)])
    w_aug = np.concatenate([np.ones(200), np.zeros(50)])
    t_clean = _fit_temperature(raw, y, False)
    t_aug = _fit_temperature(raw_aug, y_aug, False, sample_weight=w_aug)
    assert t_aug == pytest.approx(t_clean, rel=1e-3)
    # And uniform weights match no weights.
    t_ones = _fit_temperature(raw, y, False, sample_weight=np.ones(200))
    assert t_ones == pytest.approx(t_clean, rel=1e-6)


def test_classifier_fit_with_zero_weights_smoke():
    Xc, yc = _toy_binary(600)
    w = np.ones(600)
    w[:50] = 0.0
    c = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(
        Xc, yc, sample_weight=w)
    assert np.isfinite(c.temperature_)
    assert np.isfinite(c.predict_proba(Xc)).all()


# ------------------------------------- conformal scaling on asymmetric grids


def test_conformalize_asymmetric_grid_keeps_quantiles_ordered():
    """Unpaired levels used to keep scale 1.0 while their neighbors shrank,
    reordering ~95% of rows on this exact configuration."""
    from chimeraboost import ChimeraBoostQuantileRegressor

    rng = np.random.default_rng(6)
    X = rng.normal(size=(3000, 4))
    y = 5.0 * X[:, 0] + 0.1 * rng.normal(size=3000)
    m = ChimeraBoostQuantileRegressor(
        quantiles=[0.1, 0.3, 0.9], conformalize=True,
        n_estimators=30, early_stopping=False, random_state=0)
    m.fit(X[:2000], y[:2000])
    P = m.predict(X[2000:])
    assert np.all(np.diff(P, axis=1) >= -1e-9)
    # The unpaired 0.3 level must actually receive a factor from its
    # neighbors, not silently keep 1.0 while they shrink.
    scales = np.asarray(m.conformal_scale_)
    if scales[0] != pytest.approx(1.0):
        assert scales[1] != pytest.approx(1.0)


def test_conformalize_pair_free_grid_raises():
    """A grid with no symmetric pairs cannot be conformalized; it used to be
    a silent no-op that still paid the calibration data tax."""
    from chimeraboost import ChimeraBoostQuantileRegressor

    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=500)
    m = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.25, 0.8],
                                      conformalize=True, n_estimators=10)
    with pytest.raises(ValueError, match="symmetric pair"):
        m.fit(X, y)


def test_conformalize_symmetric_grid_unchanged():
    """The default symmetric grid keeps working (and stays ordered)."""
    from chimeraboost import ChimeraBoostQuantileRegressor

    rng = np.random.default_rng(8)
    X = rng.normal(size=(1500, 4))
    y = 2.0 * X[:, 0] + 0.2 * rng.normal(size=1500)
    m = ChimeraBoostQuantileRegressor(conformalize=True, n_estimators=20,
                                      random_state=0)
    m.fit(X[:1000], y[:1000])
    P = m.predict(X[1000:])
    assert np.all(np.diff(P, axis=1) >= -1e-9)


# --------------------------------------------------- bagged rare-class fits


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bagged_binary_rare_class_does_not_crash(seed):
    """A bootstrap member that drew zero positives used to crash the whole
    bag with 'Need at least 2 classes' on plainly two-class data."""
    rng = np.random.default_rng(9)
    n = 300
    X = rng.normal(size=(n, 5))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=2, replace=False)] = 1     # 2 positives in 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = ChimeraBoostClassifier(n_estimators=15, n_ensembles=4,
                                   max_samples=0.5, random_state=seed)
        c.fit(X, y)
    proba = c.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()
    assert list(c.classes_) == [0, 1]


def test_bagged_multiclass_rare_class_still_aligns():
    rng = np.random.default_rng(10)
    n = 300
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(int)
    y[rng.choice(n, size=3, replace=False)] = 2     # rare third class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = ChimeraBoostClassifier(n_estimators=15, n_ensembles=4,
                                   max_samples=0.5, random_state=0)
        c.fit(X, y)
    proba = c.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


# --------------------------------------------------------------- misc guards


def test_ordered_boosting_l2_zero_does_not_crash():
    """Singleton leaves made the leave-one-out step divide by exactly zero
    when l2_leaf_reg=0; both values pass parameter validation."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(500, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=500)
    m = ChimeraBoostRegressor(ordered_boosting=True, l2_leaf_reg=0.0,
                              n_estimators=10, random_state=0).fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_refit_on_ndarray_clears_stale_feature_names():
    """After df fit -> ndarray refit, the old names used to stick around and
    misfire the column-order guard both ways."""
    X, y = _toy_regression(300)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0)
    m.fit(df, y)
    assert list(m.feature_names_in_) == ["a", "b", "c", "d", "e"]
    X2, y2 = _toy_regression(300, seed=1)
    m.fit(X2, y2)                       # refit on a plain ndarray
    assert not hasattr(m, "feature_names_in_")
    # ndarray predict is silent; a df with new names warns (names on one
    # side only) but must NOT raise against the stale first-fit names.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict(X2)
    df2 = pd.DataFrame(X2, columns=["x1", "x2", "x3", "x4", "x5"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isfinite(m.predict(df2)).all()


def test_weighted_k1_multiquantile_matches_scalar_quantile_trees():
    """K=1 multi-quantile is the same algorithm as the scalar Quantile
    booster; before the fix the multi-quantile split search dropped sample
    weights from the hessian and the two diverged tree-for-tree under a
    weight skew (0 of 40 trees matched)."""
    from chimeraboost.booster import GradientBoosting, MultiQuantileBoosting

    rng = np.random.default_rng(12)
    n = 800
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.3 * rng.normal(size=n)
    w = np.ones(n)
    w[rng.choice(n, size=200, replace=False)] = 100.0

    kw = dict(n_estimators=40, random_state=0, depth=4, learning_rate=0.1)
    scalar = GradientBoosting(loss="Quantile", loss_kwargs={"alpha": 0.5},
                              **kw)
    scalar.fit(X, y, sample_weight=w)
    multi = MultiQuantileBoosting(quantiles=[0.5], **kw)
    multi.fit(X, y, sample_weight=w)

    assert len(scalar.trees_) == len(multi.trees_)
    for ts, tm in zip(scalar.trees_, multi.trees_):
        np.testing.assert_array_equal(ts.splits_feat, tm.splits_feat)
        np.testing.assert_array_equal(ts.splits_thr, tm.splits_thr)


def test_member_oob_eval_respects_groups():
    """A bag member's OOB eval rows must not share a group with its training
    rows; before the fix the group boundary was silently ignored under
    bagging."""
    from chimeraboost.sklearn_api import _member_oob_eval_indices

    n = 100
    groups = np.repeat(np.arange(10), 10)      # 10 groups of 10 rows
    idx = np.arange(0, 35)                     # covers groups 0-3 (partially)
    oob = _member_oob_eval_indices(idx, n, groups)
    assert len(oob) > 0
    assert not np.intersect1d(np.unique(groups[oob]),
                              np.unique(groups[idx])).size
    # Without groups: the plain complement, unchanged behavior.
    plain = _member_oob_eval_indices(idx, n, None)
    np.testing.assert_array_equal(plain, np.arange(35, 100))
    # Every group straddling the sample -> empty (caller falls back to the
    # member's group-aware auto-split).
    idx_all = np.arange(0, 100, 2)             # one row from every group
    assert len(_member_oob_eval_indices(idx_all, n, groups)) == 0


def test_bagged_fit_with_groups_smoke():
    rng = np.random.default_rng(13)
    n = 400
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=n)
    groups = rng.integers(0, 20, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ChimeraBoostRegressor(n_estimators=15, n_ensembles=3,
                                  random_state=0)
        m.fit(X, y, groups=groups)
    assert np.isfinite(m.predict(X)).all()


# ------------------------------------------------- binner heavy-hitter fix


def test_binner_dominant_minimum_does_not_collapse():
    """99%+ mass on the minimum used to collapse the borders to [min], which
    (border <= v goes right) binned every row identically -- a perfectly
    predictive sparse feature silently died."""
    from chimeraboost.binning import _feature_borders

    rng = np.random.default_rng(14)
    col = np.zeros(20_000)
    nz = rng.choice(20_000, size=400, replace=False)
    col[nz] = rng.uniform(1, 100, size=400)
    borders = _feature_borders(col, 128)
    # The floor guarantees real resolution among the nonzeros (mass-
    # proportional alone would round to ~1 border).
    assert borders.size >= 5
    assert borders[0] > 0.0                 # zeros isolated in their own bin
    assert borders[0] < 1.0                 # below the smallest nonzero


def test_binner_dominant_minimum_feature_stays_predictive():
    rng = np.random.default_rng(15)
    n = 20_000
    col = np.zeros(n)
    nz = rng.choice(n, size=400, replace=False)
    col[nz] = rng.uniform(1, 100, size=400)
    y = np.where(col > 0, 100.0, 0.0) + 0.1 * rng.normal(size=n)
    m = ChimeraBoostRegressor(n_estimators=30, random_state=0).fit(
        col.reshape(-1, 1), y)
    lo = m.predict(np.array([[0.0]]))[0]
    hi = m.predict(np.array([[50.0]]))[0]
    assert hi - lo > 90.0


def test_binner_non_colliding_borders_bit_identical():
    """Columns whose quantile levels land on distinct borders must keep the
    plain quantile borders exactly -- the fallback only fires on collision."""
    from chimeraboost.binning import _feature_borders

    rng = np.random.default_rng(16)
    col = rng.normal(size=5000)             # continuous: no collisions
    got = _feature_borders(col, 128)
    qs = np.linspace(0.0, 1.0, 129)[1:-1]
    np.testing.assert_array_equal(got, np.unique(np.quantile(col, qs)))


def test_binner_weighted_dominant_minimum():
    from chimeraboost.binning import _feature_borders

    rng = np.random.default_rng(17)
    n = 20_000
    col = np.zeros(n)
    nz = rng.choice(n, size=400, replace=False)
    col[nz] = rng.uniform(1, 100, size=400)
    w = np.ones(n)
    borders = _feature_borders(col, 128, weights=w)
    assert borders.size >= 5
    assert 0.0 < borders[0] < 1.0


def test_binner_dominant_maximum_still_fine():
    from chimeraboost.binning import _feature_borders

    rng = np.random.default_rng(18)
    n = 20_000
    col = np.full(n, 100.0)
    nz = rng.choice(n, size=400, replace=False)
    col[nz] = rng.uniform(0, 99, size=400)
    borders = _feature_borders(col, 128)
    assert borders.size >= 5
    assert borders[-1] > 99.0               # the tied max isolated above


def test_clean_eval_set_still_fits():
    X, y = _toy_regression(800)
    Xv, yv = _toy_regression(200, seed=1)
    m = ChimeraBoostRegressor(n_estimators=30).fit(X, y, eval_set=(Xv, yv))
    assert np.isfinite(m.predict(Xv)).all()

    Xc, yc = _toy_binary(800)
    Xcv, ycv = _toy_binary(200, seed=1)
    c = ChimeraBoostClassifier(n_estimators=30).fit(Xc, yc,
                                                    eval_set=(Xcv, ycv))
    assert np.isfinite(c.predict_proba(Xcv)).all()
