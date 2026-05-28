"""Test suite for ChimeraBoost. Run with: pytest -q"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


def test_regressor_beats_mean_baseline():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = ChimeraBoostRegressor(iterations=300, random_state=0).fit(Xtr, ytr)
    rmse = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
    baseline = np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean())))
    # diabetes is tiny and noisy; this is a single split, so the bound is loose
    # on purpose -- it checks the model meaningfully beats the mean, not a precise
    # ratio. (With early stopping or min_child_weight tuning it does better, but
    # this test exercises the bare default path.)
    assert rmse < 0.93 * baseline


def test_classifier_high_auc():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=300, random_state=0).fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    assert auc > 0.97
    proba = m.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_ordered_ts_resists_leakage():
    """Ordered target stats should generalize on a high-cardinality categorical
    far better than the train/test gap a leaky encoder would show."""
    rng = np.random.default_rng(0)
    n, n_levels = 5000, 2500
    cat = rng.integers(0, n_levels, n)
    num = rng.normal(size=(n, 3))
    logit = 1.2 * num[:, 0] - num[:, 1] + rng.normal(0, 1, n)
    y = (logit > np.median(logit)).astype(int)
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = np.array([f"id_{c}" for c in cat], dtype=object)
    X[:, 1:] = num
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=200, random_state=1)
    m.fit(Xtr, ytr, cat_features=[0])
    tr = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
    te = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    assert te > 0.85          # generalizes
    assert tr - te < 0.10     # small gap, i.e. not memorizing the noise column


def test_early_stopping_trims_trees():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(
        iterations=1000, early_stopping_rounds=20, random_state=0
    )
    m.fit(Xtr, ytr, eval_set=(Xte, yte))
    assert m.best_iteration_ < 1000


def test_handles_nan_and_unseen_categories():
    rng = np.random.default_rng(0)
    n = 1500
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = np.array([f"c{c}" for c in rng.integers(0, 8, n)], dtype=object)
    num = rng.normal(size=(n, 2))
    num[rng.random(n) < 0.1, 0] = np.nan
    X[:, 1:] = num
    y = ((num[:, 1] > 0) | (rng.random(n) < 0.3)).astype(int)
    m = ChimeraBoostClassifier(iterations=80, random_state=0)
    m.fit(X, y, cat_features=[0])
    Xnew = np.array([["c_UNSEEN", np.nan, 0.5], ["c3", 1.0, -0.5]], dtype=object)
    p = m.predict_proba(Xnew)
    assert p.shape == (2, 2)
    assert np.all((p >= 0) & (p <= 1))


def test_explicit_lr_overrides_auto():
    X, y = load_diabetes(return_X_y=True)
    m = ChimeraBoostRegressor(iterations=50, learning_rate=0.123).fit(X, y)
    assert m.model_.lr_ == 0.123


def test_multiclass_accuracy():
    from sklearn.datasets import load_wine, load_iris
    for load in (load_wine, load_iris):
        X, y = load(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        m = ChimeraBoostClassifier(iterations=200, random_state=0).fit(Xtr, ytr)
        assert m.n_classes_ == 3
        proba = m.predict_proba(Xte)
        assert proba.shape == (len(yte), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (m.predict(Xte) == yte).mean() > 0.9


def test_multiclass_preserves_string_labels_and_categoricals():
    rng = np.random.default_rng(0)
    n = 2000
    region = rng.choice(["N", "S", "E"], n)
    x = rng.normal(size=(n, 2))
    score = np.select([region == "N", region == "S"], [1.5, -1.0], 0.0) + 0.4 * x[:, 0]
    y = np.array(["low", "mid", "high"])[np.digitize(score, [-0.3, 1.0])]
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = region
    X[:, 1:] = x
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    m = ChimeraBoostClassifier(iterations=150, random_state=1)
    m.fit(Xtr, ytr, cat_features=[0])
    assert set(m.classes_) == {"low", "mid", "high"}
    assert set(np.unique(m.predict(Xte))).issubset({"low", "mid", "high"})


def test_feature_importances():
    rng = np.random.default_rng(0)
    n = 3000
    strong = rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    y = (strong + 0.1 * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([strong, noise])
    m = ChimeraBoostClassifier(iterations=100, random_state=0).fit(X, y)
    imp = m.feature_importances_
    assert imp.shape == (5,)
    assert abs(imp.sum() - 1.0) < 1e-6
    assert imp.argmax() == 0          # the informative feature dominates


def test_mae_loss_beats_rmse_on_mae_metric():
    from sklearn.metrics import mean_absolute_error
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    mae = ChimeraBoostRegressor(iterations=300, loss="MAE", random_state=0).fit(Xtr, ytr)
    rmse = ChimeraBoostRegressor(iterations=300, loss="RMSE", random_state=0).fit(Xtr, ytr)
    assert (mean_absolute_error(yte, mae.predict(Xte))
            <= mean_absolute_error(yte, rmse.predict(Xte)) + 1.0)


def test_quantile_calibration_on_large_data():
    rng = np.random.default_rng(0)
    n = 10000
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] + rng.normal(0, 1, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    # Early stopping prevents overfitting the training quantiles, improving test calibration.
    qlo = ChimeraBoostRegressor(iterations=2000, depth=4, loss="Quantile",
                                alpha=0.1, early_stopping=True,
                                early_stopping_rounds=50, random_state=0).fit(Xtr, ytr)
    qhi = ChimeraBoostRegressor(iterations=2000, depth=4, loss="Quantile",
                                alpha=0.9, early_stopping=True,
                                early_stopping_rounds=50, random_state=0).fit(Xtr, ytr)
    cov = np.mean((yte >= qlo.predict(Xte)) & (yte <= qhi.predict(Xte)))
    assert cov > 0.77                 # ~0.80 target; tight only with early stopping



def test_staged_predict_matches_final():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    r = ChimeraBoostRegressor(iterations=50, random_state=0).fit(Xtr, ytr)
    stages = list(r.staged_predict(Xte))
    assert len(stages) == r.best_iteration_
    assert np.allclose(stages[-1], r.predict(Xte))


def test_colsample_runs_and_keeps_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=150, colsample=0.5,
                               random_state=0).fit(Xtr, ytr)
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.97


def test_thread_count_records_effective_threads():
    import numba
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(iterations=30, thread_count=1, random_state=0).fit(X, y)
    assert m.model_.n_threads_ == 1
    # None -> all detected cores
    m2 = ChimeraBoostClassifier(iterations=30, thread_count=None, random_state=0).fit(X, y)
    assert m2.model_.n_threads_ == numba.config.NUMBA_NUM_THREADS
    # over-request is clamped, never exceeds detected cores
    m3 = ChimeraBoostClassifier(iterations=30, thread_count=9999, random_state=0).fit(X, y)
    assert m3.model_.n_threads_ <= numba.config.NUMBA_NUM_THREADS


def test_thread_count_does_not_change_predictions():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    a = ChimeraBoostRegressor(iterations=80, thread_count=1, random_state=0).fit(Xtr, ytr)
    b = ChimeraBoostRegressor(iterations=80, thread_count=None, random_state=0).fit(Xtr, ytr)
    # histogram sums are deterministic regardless of thread count
    assert np.allclose(a.predict(Xte), b.predict(Xte))


def test_min_child_weight_controls_depth_overfitting():
    """With min_child_weight active, increasing depth should NOT degrade test
    accuracy (the constraint stops growth before sparse leaves overfit). This is
    the property that fixes the oblivious-tree depth anomaly."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=4000, n_features=30, n_informative=20,
                           noise=20, random_state=1000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)

    def rmse_at(depth, mcw):
        m = ChimeraBoostRegressor(iterations=1500, depth=depth,
                                  min_child_weight=mcw, early_stopping_rounds=50,
                                  random_state=0).fit(Xf, yf, eval_set=(Xv, yv))
        return np.sqrt(np.mean((yte - m.predict(Xte)) ** 2))

    # Unconstrained (mcw=1): deeper overfits -> depth 8 clearly worse than depth 4.
    assert rmse_at(8, 1) > rmse_at(4, 1)
    # Constrained (mcw=20): depth 8 should be no worse than a small tolerance
    # above depth 6 -- growth is capped, so extra depth is harmless.
    assert rmse_at(8, 20) <= rmse_at(6, 20) + 0.5


def test_min_child_weight_param_plumbing():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(iterations=50, min_child_weight=30,
                               random_state=0).fit(X, y)
    assert m.model_.min_child_weight == 30.0


def test_shared_histogram_buffers_match_standalone():
    """A tree built with pre-allocated shared buffers must be identical to one
    built with its own freshly-allocated buffers (same math, no realloc)."""
    import numpy as np
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 12))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 800)).astype(float)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    nb = prep.n_bins_
    grad = (y - y.mean()); hess = np.ones(len(y))

    depth = 6
    standalone = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1)
    nfeat = Xb.shape[1]; maxbins = int(nb.max()); maxleaves = 1 << depth
    bufs = (np.zeros((nfeat, maxleaves, maxbins)),
            np.zeros((nfeat, maxleaves, maxbins)))
    shared = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1,
                                  hist_buffers=bufs)
    assert np.array_equal(standalone.splits_feat, shared.splits_feat)
    assert np.array_equal(standalone.splits_thr, shared.splits_thr)
    assert np.allclose(standalone.values, shared.values)

    # Reusing the SAME buffers for a second, different tree must not leak state.
    y2 = (X[:, 3] - X[:, 4] + rng.normal(0, 0.5, 800)).astype(float)
    g2 = (y2 - y2.mean())
    again = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1,
                                 hist_buffers=bufs)
    fresh = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1)
    assert np.array_equal(again.splits_feat, fresh.splits_feat)
    assert np.allclose(again.values, fresh.values)


# ---------------------------------------------------------------------------
# sample_weight tests
# ---------------------------------------------------------------------------

def test_sample_weight_uniform_equals_no_weight_rmse():
    """sample_weight=ones must give bitwise-identical predictions to no weight
    for RMSE: normalized ones leave grad/hess unchanged, np.average(y,w=None)==mean."""
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    w = np.ones(len(ytr))
    m_none = ChimeraBoostRegressor(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostRegressor(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict(Xte), m_ones.predict(Xte))


def test_sample_weight_uniform_equals_no_weight_logloss():
    """Same exact-equality check for binary classification (Logloss)."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    w = np.ones(len(ytr))
    m_none = ChimeraBoostClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict_proba(Xte), m_ones.predict_proba(Xte))


def test_sample_weight_uniform_equals_no_weight_multiclass():
    """Same exact-equality check for multiclass (softmax)."""
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    w = np.ones(len(ytr))
    m_none = ChimeraBoostClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict_proba(Xte), m_ones.predict_proba(Xte))


def test_sample_weight_shifts_predictions():
    """Up-weighting the high-y half of the training set should push the mean
    prediction higher on held-out data relative to the unweighted model."""
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(size=(n, 5))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.5, n)   # strong signal in col 0
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    # Build weights: samples with above-median y get weight 5, others get 1.
    w_high = np.where(ytr >= np.median(ytr), 5.0, 1.0)
    w_low  = np.where(ytr <  np.median(ytr), 5.0, 1.0)

    m_base = ChimeraBoostRegressor(iterations=150, random_state=0).fit(Xtr, ytr)
    m_high = ChimeraBoostRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_high
    )
    m_low  = ChimeraBoostRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_low
    )
    mean_base = m_base.predict(Xte).mean()
    mean_high = m_high.predict(Xte).mean()
    mean_low  = m_low.predict(Xte).mean()

    # Up-weighting high-y samples → higher mean predictions, and vice-versa.
    assert mean_high > mean_base > mean_low


def test_sample_weight_early_stopping_slices_correctly():
    """When early_stopping=True, the weight array must be sliced to match the
    training split; the fit should complete without error and stop early."""
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 2.0, len(y))
    m = ChimeraBoostClassifier(
        iterations=500, early_stopping=True, validation_fraction=0.15,
        early_stopping_rounds=20, random_state=0
    ).fit(X, y, sample_weight=w)
    assert m.best_iteration_ < 500


def test_empty_tree_stops_boosting_early():
    """When splits are exhausted, the booster should stop rather than bank
    useless depth-0 trees until the iteration ceiling."""
    import numpy as np
    # One informative feature, aggressive min_child_weight -> splits run out fast.
    X = np.array([[0.0]] * 60 + [[1.0]] * 60)
    y = np.array([0.0] * 60 + [1.0] * 60)
    m = ChimeraBoostRegressor(iterations=1000, min_child_weight=30,
                              random_state=0).fit(X, y)
    assert len(m.model_.trees_) < 1000
