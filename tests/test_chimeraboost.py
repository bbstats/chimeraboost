"""Test suite for ChimeraBoost. Run with: pytest -q"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


def test_descend_leaves_matches_numpy_reference():
    """The in-place njit per-level leaf descent must equal the old numpy
    expression `(leaf<<1) + (Xf > t).astype(int64)` exactly, across leaf depths,
    bin values and thresholds (incl. t=-1 and t at the max bin). Guards the
    fit-time speedup (the descent was ~1/3 of fit)."""
    from chimeraboost.tree import _descend_leaves

    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 6000))
        d = int(rng.integers(0, 6))
        leaf = rng.integers(0, 1 << d, size=n).astype(np.int64)
        Xf = rng.integers(0, 260, size=n).astype(np.uint16)
        t = int(rng.integers(-1, 260))
        ref = (leaf << 1) + (Xf > t).astype(np.int64)
        got = leaf.copy()
        _descend_leaves(got, Xf, t)
        assert np.array_equal(ref, got)


def test_binning_transform_matches_searchsorted_reference():
    """The njit row-parallel binning kernel must be bit-identical to the old
    per-column np.searchsorted(side='right') logic, including NaN/+-inf routing to
    the missing bin and few-distinct/constant columns. Guards the predict-time
    binning speedup (the kernel is ~half of inference cost)."""
    from chimeraboost.binning import Binner, BIN_DTYPE

    def reference(binner, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=BIN_DTYPE)
        for f in range(X.shape[1]):
            col = X[:, f]
            borders = binner.borders_[f]
            binned = np.searchsorted(borders, col, side="right").astype(BIN_DTYPE)
            binned[~np.isfinite(col)] = len(borders) + 1
            out[:, f] = binned
        return out

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 10))
    X[:, 2] = rng.integers(0, 4, size=4000)          # few distinct values
    X[:, 5] = 1.0                                      # constant column
    X[rng.integers(0, 4000, 300), rng.integers(0, 10, 300)] = np.nan
    X[rng.integers(0, 4000, 100), rng.integers(0, 10, 100)] = np.inf
    X[rng.integers(0, 4000, 100), rng.integers(0, 10, 100)] = -np.inf

    bn = Binner(max_bins=64).fit(X)
    assert np.array_equal(bn.transform(X), reference(bn, X))
    assert np.array_equal(bn.transform(X[:1]), reference(bn, X[:1]))  # single row
    assert bn.transform(X[:0]).shape == (0, 10)                       # empty


def test_regressor_beats_mean_baseline():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = ChimeraBoostRegressor(n_estimators=300, random_state=0).fit(Xtr, ytr)
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
    m = ChimeraBoostClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
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
    m = ChimeraBoostClassifier(n_estimators=200, random_state=1)
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
        n_estimators=1000, early_stopping_rounds=20, random_state=0
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
    m = ChimeraBoostClassifier(n_estimators=80, random_state=0)
    m.fit(X, y, cat_features=[0])
    Xnew = np.array([["c_UNSEEN", np.nan, 0.5], ["c3", 1.0, -0.5]], dtype=object)
    p = m.predict_proba(Xnew)
    assert p.shape == (2, 2)
    assert np.all((p >= 0) & (p <= 1))


def test_explicit_lr_overrides_auto():
    X, y = load_diabetes(return_X_y=True)
    m = ChimeraBoostRegressor(n_estimators=50, learning_rate=0.123).fit(X, y)
    assert m.model_.lr_ == 0.123


def test_multiclass_accuracy():
    from sklearn.datasets import load_wine, load_iris
    for load in (load_wine, load_iris):
        X, y = load(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        m = ChimeraBoostClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
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
    m = ChimeraBoostClassifier(n_estimators=150, random_state=1)
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
    m = ChimeraBoostClassifier(n_estimators=100, random_state=0).fit(X, y)
    imp = m.feature_importances_
    assert imp.shape == (5,)
    assert abs(imp.sum() - 1.0) < 1e-6
    assert imp.argmax() == 0          # the informative feature dominates


def test_mae_loss_beats_rmse_on_mae_metric():
    from sklearn.metrics import mean_absolute_error
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    mae = ChimeraBoostRegressor(n_estimators=300, loss="MAE", random_state=0).fit(Xtr, ytr)
    rmse = ChimeraBoostRegressor(n_estimators=300, loss="RMSE", random_state=0).fit(Xtr, ytr)
    assert (mean_absolute_error(yte, mae.predict(Xte))
            <= mean_absolute_error(yte, rmse.predict(Xte)) + 1.0)


def test_quantile_calibration_on_large_data():
    rng = np.random.default_rng(0)
    n = 10000
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] + rng.normal(0, 1, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    # Early stopping prevents overfitting the training quantiles, improving test calibration.
    qlo = ChimeraBoostRegressor(n_estimators=2000, depth=4, loss="Quantile",
                                alpha=0.1, early_stopping=True,
                                early_stopping_rounds=50, random_state=0).fit(Xtr, ytr)
    qhi = ChimeraBoostRegressor(n_estimators=2000, depth=4, loss="Quantile",
                                alpha=0.9, early_stopping=True,
                                early_stopping_rounds=50, random_state=0).fit(Xtr, ytr)
    cov = np.mean((yte >= qlo.predict(Xte)) & (yte <= qhi.predict(Xte)))
    assert cov > 0.77                 # ~0.80 target; tight only with early stopping



def test_staged_predict_matches_final():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    r = ChimeraBoostRegressor(n_estimators=50, random_state=0).fit(Xtr, ytr)
    stages = list(r.staged_predict(Xte))
    assert len(stages) == r.best_iteration_
    assert np.allclose(stages[-1], r.predict(Xte))


def test_colsample_runs_and_keeps_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(n_estimators=150, colsample=0.5,
                               random_state=0).fit(Xtr, ytr)
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.97


def test_thread_count_records_effective_threads():
    import numba
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(n_estimators=30, thread_count=1, random_state=0).fit(X, y)
    assert m.model_.n_threads_ == 1
    # None -> all detected cores
    m2 = ChimeraBoostClassifier(n_estimators=30, thread_count=None, random_state=0).fit(X, y)
    assert m2.model_.n_threads_ == numba.config.NUMBA_NUM_THREADS
    # over-request is clamped, never exceeds detected cores
    m3 = ChimeraBoostClassifier(n_estimators=30, thread_count=9999, random_state=0).fit(X, y)
    assert m3.model_.n_threads_ <= numba.config.NUMBA_NUM_THREADS


def test_thread_count_does_not_change_predictions():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    a = ChimeraBoostRegressor(n_estimators=80, thread_count=1, random_state=0).fit(Xtr, ytr)
    b = ChimeraBoostRegressor(n_estimators=80, thread_count=None, random_state=0).fit(Xtr, ytr)
    # histogram sums are deterministic regardless of thread count
    assert np.allclose(a.predict(Xte), b.predict(Xte))


def test_min_child_weight_regularizes_sparse_leaves():
    """min_child_weight regularizes by forbidding SPARSE non-empty leaves, so at a
    fixed (deep) depth, raising it reduces overfitting.

    History (read before changing): an earlier version asserted that mcw *caps
    depth* -- that depth 8 ~= depth 6 because growth stops. That encoded a BUG.
    The oblivious veto rejected a shared split whenever any leaf gained an EMPTY
    child (a pure leaf, all samples one way), which is normal in symmetric trees
    (cf. CatBoost). One pure leaf vetoed the whole level, so effective depth
    self-capped ~4-6 regardless of the `depth` arg, and large interaction-heavy
    datasets (e.g. pol) were stuck ~79% of sklearn with no way to improve. The
    fix exempts empty children (only 0 < mass < mcw is illegal). depth is a real
    lever again; mcw still guards sparse leaves. Do NOT reassert the depth cap."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=4000, n_features=30, n_informative=20,
                           noise=20, random_state=1000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)

    def rmse_at(depth, mcw):
        m = ChimeraBoostRegressor(n_estimators=1500, depth=depth,
                                  min_child_weight=mcw, early_stopping_rounds=50,
                                  random_state=0).fit(Xf, yf, eval_set=(Xv, yv))
        return np.sqrt(np.mean((yte - m.predict(Xte)) ** 2))

    # depth is a real lever: unconstrained (mcw=1), deeper overfits this noisy
    # target -> depth 8 clearly worse than depth 4.
    assert rmse_at(8, 1) > rmse_at(4, 1)
    # mcw regularizes sparse leaves: at the same deep depth, a strong mcw sharply
    # reduces the overfit a weak one allows.
    assert rmse_at(8, 80) < rmse_at(8, 1)
    # ...and it is monotone in the right direction (more mass -> less overfit).
    assert rmse_at(8, 80) <= rmse_at(8, 20)


def test_min_child_weight_param_plumbing():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(n_estimators=50, min_child_weight=30,
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
    # Tree builder consumes a feature-major (n_features, n_samples) matrix.
    Xb = np.ascontiguousarray(prep.fit_transform(X, [y], None).T)
    nb = prep.n_bins_
    grad = (y - y.mean()); hess = np.ones(len(y))

    depth = 6
    standalone, _ = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1)
    nfeat = Xb.shape[0]; maxbins = int(nb.max()); maxleaves = 1 << depth
    bufs = np.zeros((nfeat, maxleaves, maxbins, 2))   # interleaved grad/hess
    shared, _ = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1,
                                     hist_buffers=bufs)
    assert np.array_equal(standalone.splits_feat, shared.splits_feat)
    assert np.array_equal(standalone.splits_thr, shared.splits_thr)
    assert np.allclose(standalone.values, shared.values)

    # Reusing the SAME buffers for a second, different tree must not leak state.
    y2 = (X[:, 3] - X[:, 4] + rng.normal(0, 0.5, 800)).astype(float)
    g2 = (y2 - y2.mean())
    again, _ = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1,
                                    hist_buffers=bufs)
    fresh, _ = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1)
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
    m_none = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(
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
    m_none = ChimeraBoostClassifier(n_estimators=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(n_estimators=80, random_state=0).fit(
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
    m_none = ChimeraBoostClassifier(n_estimators=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(n_estimators=80, random_state=0).fit(
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

    m_base = ChimeraBoostRegressor(n_estimators=150, random_state=0).fit(Xtr, ytr)
    m_high = ChimeraBoostRegressor(n_estimators=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_high
    )
    m_low  = ChimeraBoostRegressor(n_estimators=150, random_state=0).fit(
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
        n_estimators=500, early_stopping=True, validation_fraction=0.15,
        early_stopping_rounds=20, random_state=0
    ).fit(X, y, sample_weight=w)
    assert m.best_iteration_ < 500


def test_groups_kept_intact_in_early_stopping_split():
    """The grouped early-stopping split must keep every group entirely on one
    side of the train/validation boundary, on both the regression
    (GroupShuffleSplit) and classification (StratifiedGroupKFold) paths. The
    end-to-end classifier fit with groups should also run and predict."""
    from chimeraboost.sklearn_api import _make_eval_split
    rng = np.random.default_rng(0)
    n = 400
    groups = rng.integers(0, 40, size=n)        # 40 groups, repeated across rows
    X = rng.normal(size=(n, 5))
    y_cls = rng.integers(0, 2, size=n)
    y_reg = rng.normal(size=n)

    # Regression path: GroupShuffleSplit, no stratification.
    tr, va = _make_eval_split(X, y_reg, 0.2, 0, groups=groups, stratify=None)
    assert set(groups[tr]).isdisjoint(set(groups[va]))

    # Classification path: StratifiedGroupKFold.
    tr, va = _make_eval_split(X, y_cls, 0.2, 0, groups=groups, stratify=y_cls)
    assert set(groups[tr]).isdisjoint(set(groups[va]))

    # End-to-end: early stopping + groups fits and predicts the right shape.
    m = ChimeraBoostClassifier(n_estimators=200, early_stopping=True,
                               validation_fraction=0.2, early_stopping_rounds=15,
                               random_state=0).fit(X, y_cls, groups=groups)
    assert m.predict(X).shape == (n,)


def test_bagging_none_matches_single_model():
    """n_ensembles=None and =1 must be the plain single model, bit-identical."""
    X, y = load_diabetes(return_X_y=True)
    base = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(X, y)
    none_ = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                                  n_ensembles=None).fit(X, y)
    one = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                                n_ensembles=1).fit(X, y)
    assert base.estimators_ is None and one.estimators_ is None
    assert np.array_equal(base.predict(X), none_.predict(X))
    assert np.array_equal(base.predict(X), one.predict(X))


def test_bagging_regressor_runs_and_averages_members():
    """A bagged regressor trains the requested members and its prediction is
    exactly the mean of the members' predictions, and it beats the naive
    mean-baseline on held-out data."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1500, n_features=15, noise=25.0,
                           random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    bag = ChimeraBoostRegressor(n_estimators=150, random_state=0,
                                n_ensembles=8).fit(Xtr, ytr)
    assert len(bag.estimators_) == 8
    # The ensemble prediction is the average of its members.
    members = np.mean([m.predict(Xte) for m in bag.estimators_], axis=0)
    assert np.allclose(bag.predict(Xte), members)
    # Sanity: clearly better than predicting the training mean.
    base_rmse = np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean())))
    bag_rmse = np.sqrt(mean_squared_error(yte, bag.predict(Xte)))
    assert bag_rmse < 0.5 * base_rmse


def test_bagging_classifier_multiclass_proba():
    """Bagged multiclass classifier: proper proba shape, normalized rows, and
    preserved class labels."""
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    clf = ChimeraBoostClassifier(n_estimators=120, random_state=0,
                                 n_ensembles=8).fit(X, y)
    assert len(clf.estimators_) == 8
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(clf.classes_, np.unique(y))
    assert clf.predict(X).shape == (len(y),)


def test_bagging_parallel_matches_sequential():
    """ensemble_n_jobs only changes scheduling: members are independently
    seeded, so predictions must be identical to the sequential fit."""
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    seq = ChimeraBoostClassifier(n_estimators=80, random_state=3,
                                 n_ensembles=4, ensemble_n_jobs=1).fit(X, y)
    par = ChimeraBoostClassifier(n_estimators=80, random_state=3,
                                 n_ensembles=4, ensemble_n_jobs=2).fit(X, y)
    assert np.allclose(seq.predict_proba(X), par.predict_proba(X))


def test_bagging_with_categoricals():
    """Bagging forwards cat_features to every member (the advantage over a
    sklearn.ensemble.Bagging wrapper, which would drop it)."""
    rng = np.random.default_rng(0)
    n = 800
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = rng.choice(["a", "b", "c"], n)
    X[:, 1] = rng.normal(size=n)
    X[:, 2] = rng.choice(["x", "y"], n)
    y = ((X[:, 0] == "a").astype(int) ^ (X[:, 2] == "x").astype(int))
    clf = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                                 n_ensembles=5).fit(X, y, cat_features=[0, 2])
    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_empty_tree_stops_boosting_early():
    """When splits are exhausted, the booster should stop rather than bank
    useless depth-0 trees until the iteration ceiling."""
    import numpy as np
    # One informative feature, aggressive min_child_weight -> splits run out fast.
    X = np.array([[0.0]] * 60 + [[1.0]] * 60)
    y = np.array([0.0] * 60 + [1.0] * 60)
    m = ChimeraBoostRegressor(n_estimators=1000, min_child_weight=30,
                              random_state=0).fit(X, y)
    assert len(m.model_.trees_) < 1000


# ---------------------------------------------------------------------------
# Input validation & scikit-learn compatibility (robustness pass)
# ---------------------------------------------------------------------------
def _Xy(n=40, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    return X, X[:, 0], (X[:, 0] > 0).astype(int)


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_predict_before_fit_raises_not_fitted(Est):
    from sklearn.exceptions import NotFittedError
    X, _, _ = _Xy()
    with pytest.raises(NotFittedError):
        Est().predict(X)


def test_feature_count_mismatch_raises():
    X, yr, _ = _Xy()
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0).fit(X, yr)
    with pytest.raises(ValueError, match="features"):
        m.predict(np.random.default_rng(1).normal(size=(5, 7)))


def test_fit_input_validation_messages():
    X, yr, _ = _Xy()
    R = ChimeraBoostRegressor(n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="2D"):
        R.fit(X[:, 0], yr)                      # 1-D X
    with pytest.raises(ValueError, match="inconsistent lengths"):
        R.fit(X, yr[:10])                       # X/y mismatch
    with pytest.raises(ValueError, match="NaN or infinity"):
        R.fit(X, np.r_[np.inf, yr[1:]])         # inf in y
    with pytest.raises(ValueError, match="infinity"):
        R.fit(np.r_[[[np.inf, 0, 0, 0]], X[1:]], yr)   # inf in X
    with pytest.raises(ValueError, match="y is None"):
        R.fit(X, None)                          # missing y
    with pytest.raises(ValueError, match="sample_weight"):
        R.fit(X, yr, sample_weight=np.ones(10))


def test_nan_in_X_is_accepted_as_missing():
    X, yr, _ = _Xy()
    Xn = X.copy(); Xn[::5, 0] = np.nan
    m = ChimeraBoostRegressor(n_estimators=20, random_state=0).fit(Xn, yr)
    assert np.isfinite(m.predict(Xn)).all()     # NaN handled, not rejected


def test_n_features_in_and_feature_names_in():
    pd = pytest.importorskip("pandas")
    X, yr, _ = _Xy()
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0).fit(X, yr)
    assert m.n_features_in_ == 4
    df = pd.DataFrame(X, columns=list("abcd"))
    m2 = ChimeraBoostRegressor(n_estimators=10, random_state=0).fit(df, yr)
    assert list(m2.feature_names_in_) == list("abcd")


def test_column_vector_y_is_raveled_with_warning():
    from sklearn.exceptions import DataConversionWarning
    X, yr, _ = _Xy()
    with pytest.warns(DataConversionWarning):
        m = ChimeraBoostRegressor(n_estimators=10, random_state=0).fit(X, yr.reshape(-1, 1))
    assert m.predict(X).shape == (40,)


def test_continuous_target_to_classifier_raises():
    X, yr, _ = _Xy()
    with pytest.raises(ValueError, match="[Uu]nknown label|continuous"):
        ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(X, yr)


@pytest.mark.parametrize("params, match", [
    (dict(n_estimators=0), "n_estimators"),
    (dict(depth=0), "depth"),
    (dict(depth=30), "depth"),
    (dict(learning_rate=-0.1), "learning_rate"),
    (dict(learning_rate=0.0), "learning_rate"),
    (dict(l2_leaf_reg=-1.0), "l2_leaf_reg"),
    (dict(subsample=0.0), "subsample"),
    (dict(subsample=1.5), "subsample"),
    (dict(colsample=2.0), "colsample"),
    (dict(cat_smoothing=0.0), "cat_smoothing"),  # 0 pseudocount -> 0/0 in ordered TS
    (dict(cat_smoothing=-1.0), "cat_smoothing"),
    (dict(min_child_weight=-3.0), "min_child_weight"),
    (dict(validation_fraction=1.0), "validation_fraction"),
    (dict(cat_n_permutations=0), "cat_n_permutations"),
    (dict(leaf_estimation_iterations=0), "leaf_estimation_iterations"),
    (dict(loss="bogus"), "loss"),
    (dict(loss="Quantile", alpha=0.0), "alpha"),
    (dict(loss="Quantile", alpha=1.0), "alpha"),
])
def test_invalid_hyperparams_raise(params, match):
    X, yr, _ = _Xy()
    with pytest.raises(ValueError, match=match):
        ChimeraBoostRegressor(**params).fit(X, yr)


def test_sample_weight_value_validation():
    X, yr, _ = _Xy()
    n = X.shape[0]
    with pytest.raises(ValueError, match="NaN or infinity"):
        ChimeraBoostRegressor(n_estimators=10).fit(X, yr, sample_weight=np.full(n, np.nan))
    with pytest.raises(ValueError, match="non-negative"):
        ChimeraBoostRegressor(n_estimators=10).fit(X, yr, sample_weight=-np.ones(n))
    with pytest.raises(ValueError, match="sums to zero"):
        ChimeraBoostRegressor(n_estimators=10).fit(X, yr, sample_weight=np.zeros(n))


def test_cat_features_index_validation():
    X, _, yc = _Xy()                       # 4 numeric columns
    with pytest.raises(ValueError, match="out of range"):
        ChimeraBoostClassifier(n_estimators=10).fit(X, yc, cat_features=[9])
    with pytest.raises(ValueError, match="out of range"):
        ChimeraBoostClassifier(n_estimators=10).fit(X, yc, cat_features=[-1])
    with pytest.raises(ValueError, match="duplicate"):
        ChimeraBoostClassifier(n_estimators=10).fit(X, yc, cat_features=[1, 1])


def test_eval_set_shape_validation():
    X, yr, _ = _Xy()
    Xt, yt, Xv, yv = X[:30], yr[:30], X[30:], yr[30:]
    with pytest.raises(ValueError, match="features"):
        ChimeraBoostRegressor(n_estimators=10).fit(Xt, yt, eval_set=(Xv[:, :2], yv))
    with pytest.raises(ValueError, match="inconsistent lengths"):
        ChimeraBoostRegressor(n_estimators=10).fit(Xt, yt, eval_set=(Xv, yv[:3]))


def test_nonnumeric_column_error_names_the_column():
    pd = pytest.importorskip("pandas")
    X, _, yc = _Xy()
    df = pd.DataFrame(X, columns=list("abcd"))
    df["g"] = np.random.default_rng(0).choice(list("XY"), len(df))
    with pytest.raises(ValueError, match="cat_features"):
        ChimeraBoostClassifier(n_estimators=10).fit(df, yc)
    # The friendly message names the offending column.
    try:
        ChimeraBoostClassifier(n_estimators=10).fit(df, yc)
    except ValueError as e:
        assert "'g'" in str(e)


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_predict_enforces_feature_names(Est):
    pd = pytest.importorskip("pandas")
    X, yr, yc = _Xy()
    y = yr if Est is ChimeraBoostRegressor else yc
    df = pd.DataFrame(X, columns=list("abcd"))
    m = Est(n_estimators=20, random_state=0).fit(df, y)
    # Same order -> fine.
    m.predict(df.iloc[:3])
    # Reordered columns -> raise (would otherwise be silently wrong).
    with pytest.raises(ValueError, match="feature names"):
        m.predict(df[list("dcba")])
    # Renamed columns -> raise.
    with pytest.raises(ValueError, match="feature names"):
        m.predict(df.rename(columns={"a": "Z"}))
    # Fitted with names, predicted without -> warn (sklearn-consistent).
    with pytest.warns(UserWarning, match="without feature names"):
        m.predict(X[:3])


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_inf_rejected_at_predict(Est):
    X, yr, yc = _Xy()
    y = yr if Est is ChimeraBoostRegressor else yc
    m = Est(n_estimators=20, random_state=0).fit(X, y)
    Xinf = X[:1].copy(); Xinf[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        m.predict(Xinf)
    # sklearn's assume_finite config skips the predict-time finiteness scan.
    from sklearn import config_context
    with config_context(assume_finite=True):
        assert np.isfinite(m.predict(Xinf)).shape == (1,)  # no raise


def test_linear_leaves_warns_when_dropped_for_mae_quantile():
    X, yr, _ = _Xy()
    with pytest.warns(UserWarning, match="linear_leaves"):
        ChimeraBoostRegressor(n_estimators=20, loss="MAE", linear_leaves=True).fit(X, yr)
    with pytest.warns(UserWarning, match="linear_leaves"):
        ChimeraBoostRegressor(n_estimators=20, loss="Quantile", alpha=0.5,
                              linear_leaves=True).fit(X, yr)


def test_cat_features_constructor_param():
    """cat_features can be set on the constructor (so GridSearchCV/Pipeline can
    carry it); the fit argument overrides and never mutates the stored param."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV
    rng = np.random.default_rng(0)
    n = 600
    city = rng.choice(["NYC", "SF", "LA"], n)
    age = rng.normal(40, 10, n)
    y = ((city == "SF") | (age > 45)).astype(int)
    X = np.empty((n, 2), dtype=object); X[:, 0] = city; X[:, 1] = age

    # Constructor cat_features is used when fit gets none.
    m = ChimeraBoostClassifier(n_estimators=40, random_state=0,
                               cat_features=[0]).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.9
    # It survives clone and a meta-estimator (the whole point).
    assert clone(m).get_params()["cat_features"] == [0]
    gs = GridSearchCV(ChimeraBoostClassifier(n_estimators=30, random_state=0,
                                             cat_features=[0]), {"depth": [3, 6]}, cv=3)
    gs.fit(X, y)                                   # would crash if cat col hit float cast
    # The fit argument overrides, without mutating the stored constructor value.
    m2 = ChimeraBoostClassifier(n_estimators=20, random_state=0, cat_features=[1])
    m2.fit(X, y, cat_features=[0])
    assert m2.cat_features == [1]


def test_cat_features_by_column_name():
    """Categoricals can be marked by DataFrame column name (resolved to the same
    positions as integer indices), as a fit arg, a constructor arg, or a mix."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "city": rng.choice(["NYC", "SF", "LA"], n),
        "age": rng.normal(40, 10, n),
        "plan": rng.choice(["free", "pro"], n),
    })
    y = ((df["city"] == "SF") | (df["age"] > 45)).astype(int).to_numpy()

    by_name = ChimeraBoostClassifier(n_estimators=40, random_state=0).fit(
        df, y, cat_features=["city", "plan"])
    by_index = ChimeraBoostClassifier(n_estimators=40, random_state=0).fit(
        df, y, cat_features=[0, 2])
    # Names resolve to the same columns -> identical predictions.
    assert np.array_equal(by_name.predict(df), by_index.predict(df))
    # A mix of names and positions works too.
    mixed = ChimeraBoostClassifier(n_estimators=40, random_state=0).fit(
        df, y, cat_features=["city", 2])
    assert np.array_equal(mixed.predict(df), by_index.predict(df))
    # Names also work via the constructor (for GridSearchCV/Pipeline).
    by_ctor = ChimeraBoostClassifier(n_estimators=40, random_state=0,
                                     cat_features=["city", "plan"]).fit(df, y)
    assert np.array_equal(by_ctor.predict(df), by_index.predict(df))

    # An unknown name, or names without column metadata, raise clearly.
    with pytest.raises(ValueError, match="not a column"):
        ChimeraBoostClassifier(n_estimators=10).fit(df, y, cat_features=["nope"])
    with pytest.raises(ValueError, match="no column names"):
        ChimeraBoostClassifier(n_estimators=10).fit(
            df.to_numpy(dtype=object), y, cat_features=["city"])


def test_pyarrow_feature_names_not_polluted_by_data():
    pa = pytest.importorskip("pyarrow")
    X, _, yc = _Xy()
    tbl = pa.table({c: X[:, i] for i, c in enumerate("abcd")})
    m = ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(tbl, yc)
    # .columns is column DATA in pyarrow; names must come from .column_names.
    assert list(m.feature_names_in_) == list("abcd")
    assert m.n_features_in_ == 4


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_masked_array_rejected(Est):
    X, yr, yc = _Xy()
    y = yr if Est is ChimeraBoostRegressor else yc
    Xm = np.ma.array(X, mask=np.zeros_like(X, dtype=bool))
    Xm[0, 0] = np.ma.masked
    with pytest.raises(TypeError, match="[Mm]asked"):
        Est(n_estimators=10).fit(Xm, y)
    m = Est(n_estimators=10, random_state=0).fit(X, y)
    with pytest.raises(TypeError, match="[Mm]asked"):
        m.predict(Xm)


def test_quantile_depth_default_is_loss_adaptive():
    """depth=None resolves to 6 for RMSE/MAE (unchanged) but 4 for Quantile,
    because deep oblivious leaves overfit the tail quantile -- predicted
    quantiles otherwise collapse toward the median on held-out data."""
    X, yr, _ = _Xy()
    assert ChimeraBoostRegressor(n_estimators=10, random_state=0).fit(X, yr).model_.depth == 6
    assert ChimeraBoostRegressor(n_estimators=10, loss="MAE", random_state=0).fit(X, yr).model_.depth == 6
    mq = ChimeraBoostRegressor(n_estimators=10, loss="Quantile", alpha=0.9,
                               random_state=0).fit(X, yr)
    assert mq.model_.depth == 4
    # Explicit depth still wins.
    assert ChimeraBoostRegressor(n_estimators=10, loss="Quantile", alpha=0.9, depth=8,
                                 random_state=0).fit(X, yr).model_.depth == 8


def test_quantile_calibration_beats_deep_trees():
    """The shallower default quantile depth is better calibrated on held-out data
    than a deep model: the tails are less collapsed toward the median."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 8))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 1.5, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    def coverage(depth, a):
        m = ChimeraBoostRegressor(n_estimators=400, loss="Quantile", alpha=a,
                                  depth=depth, random_state=0).fit(Xtr, ytr)
        return float(np.mean(yte <= m.predict(Xte)))

    # Default (None -> 4) vs an explicit deep (8) model.
    lo_def = coverage(None, 0.1); hi_def = coverage(None, 0.9)
    lo_deep = coverage(8, 0.1);   hi_deep = coverage(8, 0.9)
    # 80% prediction interval: the default covers closer to the nominal 0.80.
    assert (hi_def - lo_def) > (hi_deep - lo_deep)
    # And lands in a sensible band (not collapsed to the median).
    assert 0.78 <= hi_def <= 0.95
    assert 0.05 <= lo_def <= 0.22


def test_auto_min_child_weight_is_size_adaptive():
    """Classifier default min_child_weight=None resolves to a size-adaptive veto:
    full (~1) on small data, off (~0) on large -- monotone in training size."""
    from chimeraboost.sklearn_api import _auto_min_child_weight as f
    assert f(300) == 1.0 and f(5000) == 0.0          # endpoints clamp
    assert f(400) > f(1250) > f(2500)                 # monotone decreasing
    # The resolved value lands on the fitted booster.
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(400, 6)); Xl = rng.normal(size=(4000, 6))
    cs = ChimeraBoostClassifier(n_estimators=20, random_state=0).fit(Xs, (Xs[:, 0] > 0).astype(int))
    cl = ChimeraBoostClassifier(n_estimators=20, random_state=0).fit(Xl, (Xl[:, 0] > 0).astype(int))
    assert cs.model_.min_child_weight == 1.0          # small -> full veto
    assert cl.model_.min_child_weight == 0.0          # large -> no veto
    # An explicit value is still honored (overrides auto).
    ce = ChimeraBoostClassifier(n_estimators=20, min_child_weight=0.5,
                                random_state=0).fit(Xl, (Xl[:, 0] > 0).astype(int))
    assert ce.model_.min_child_weight == 0.5


def test_auto_cat_combinations_helper():
    """cat_combinations=None enables combos only for tractable all-categorical
    data; the resource caps and the mixed/no-cat cases stay off."""
    from chimeraboost.sklearn_api import (
        _auto_cat_combinations as f,
        _AUTO_CAT_COMBO_MAX_PAIRS, _AUTO_CAT_COMBO_MAX_CELLS)
    assert f([0, 1, 2, 3], 4, 1000) is True            # all-categorical -> on
    assert f([0, 1], 4, 1000) is False                 # mixed (2 of 4) -> off
    assert f(None, 4, 1000) is False                   # no cats -> off
    assert f([0], 1, 1000) is False                    # need >=2 to combine
    # Resource guards: too many pairs, or too many pairs*rows, -> off.
    big = list(range(60)); assert (60 * 59 // 2) > _AUTO_CAT_COMBO_MAX_PAIRS
    assert f(big, 60, 1000) is False
    n_feat = 40; small_pairs = n_feat * (n_feat - 1) // 2
    assert small_pairs <= _AUTO_CAT_COMBO_MAX_PAIRS
    big_rows = int(_AUTO_CAT_COMBO_MAX_CELLS // small_pairs) + 10
    assert f(list(range(n_feat)), n_feat, big_rows) is False
    # numpy-array cat_features must not raise (ambiguous truth value).
    assert f(np.array([0, 1, 2]), 3, 1000) is True


def test_auto_cat_combinations_on_estimators():
    """The resolved cat_combinations lands on the fitted preprocessor; explicit
    True/False override the auto rule."""
    rng = np.random.default_rng(0)
    n = 1500
    fcat = rng.integers(0, 4, (n, 4))
    y = ((fcat[:, 0] == fcat[:, 1]) ^ (fcat[:, 2] > 1)).astype(int)
    Xcat = fcat.astype(object)
    allcat = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(
        Xcat, y, cat_features=[0, 1, 2, 3])
    assert allcat.model_.prep_.combo_pairs_                 # auto-on
    # Mixed data leaves combos off.
    Xmix = np.column_stack([fcat[:, :2].astype(object),
                            rng.normal(size=(n, 2)).astype(object)])
    mixed = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(
        Xmix, y, cat_features=[0, 1])
    assert not mixed.model_.prep_.combo_pairs_             # auto-off
    # Explicit False on all-categorical data overrides the auto rule.
    off = ChimeraBoostClassifier(cat_combinations=False, n_estimators=30,
                                 random_state=0).fit(Xcat, y, cat_features=[0, 1, 2, 3])
    assert not off.model_.prep_.combo_pairs_
    # Regressor honors the same auto rule.
    yr = fcat[:, 0] + 2.0 * (fcat[:, 1] == fcat[:, 2])
    rgr = ChimeraBoostRegressor(n_estimators=30, random_state=0).fit(
        Xcat, yr, cat_features=[0, 1, 2, 3])
    assert rgr.model_.prep_.combo_pairs_


# ---------------------------------------------------------------------------
# hierarchical shrinkage (hs_lambda)
# ---------------------------------------------------------------------------

def _hs_leaf_case():
    """A depth-2 (4-leaf) case with one empty leaf, for kernel-level tests.
    leaf0: g=-2,h=1 | leaf1: g=3,h=2 | leaf2: EMPTY | leaf3: g=1,h=1."""
    leaf = np.array([0, 1, 1, 3], dtype=np.int64)
    grad = np.array([-2.0, 1.5, 1.5, 1.0])
    hess = np.array([1.0, 1.0, 1.0, 1.0])
    return leaf, grad, hess


def test_hs_kernel_matches_plain_newton_at_tiny_lambda():
    """As hs_lambda -> 0+, non-empty leaves match the plain per-leaf Newton
    value (the shrinkage weight a = H/(H+hs) -> 1)."""
    from chimeraboost.tree import _leaf_values, _leaf_values_hs
    leaf, grad, hess = _hs_leaf_case()
    l2, lr = 1.0, 0.3
    plain = _leaf_values(leaf, grad, hess, 4, l2, lr)
    hs = _leaf_values_hs(leaf, grad, hess, 4, l2, lr, 1e-9)
    mass = np.bincount(leaf, weights=hess, minlength=4)
    nonempty = mass > 0
    assert np.allclose(plain[nonempty], hs[nonempty], atol=1e-6)


def test_hs_kernel_collapses_to_root_at_large_lambda():
    """As hs_lambda -> infinity every leaf is pulled to the root's global Newton
    value (one shared step over all samples)."""
    from chimeraboost.tree import _leaf_values_hs
    leaf, grad, hess = _hs_leaf_case()
    l2, lr = 1.0, 1.0
    root = lr * (-grad.sum() / (hess.sum() + l2))
    hs = _leaf_values_hs(leaf, grad, hess, 4, l2, lr, 1e12)
    assert np.allclose(hs, root, atol=1e-3)


def test_hs_kernel_shrinks_monotonically():
    """Increasing hs_lambda pulls leaf values toward their ancestors, so the
    spread (variance) of leaf values is non-increasing."""
    from chimeraboost.tree import _leaf_values_hs
    leaf, grad, hess = _hs_leaf_case()
    spreads = [float(np.var(_leaf_values_hs(leaf, grad, hess, 4, 1.0, 1.0, hl)))
               for hl in (0.0, 0.5, 2.0, 10.0, 100.0)]
    assert all(a >= b - 1e-12 for a, b in zip(spreads, spreads[1:]))


def test_hs_kernel_empty_leaf_inherits_ancestor():
    """An empty leaf (no samples) takes its parent's shrunk value -- unlike the
    plain Newton estimate, which would leave it at 0."""
    from chimeraboost.tree import _leaf_values, _leaf_values_hs
    leaf, grad, hess = _hs_leaf_case()
    plain = _leaf_values(leaf, grad, hess, 4, 1.0, 1.0)
    hs = _leaf_values_hs(leaf, grad, hess, 4, 1.0, 1.0, 1.0)
    assert plain[2] == 0.0          # leaf2 is empty -> plain leaves it at 0
    assert hs[2] != 0.0             # HS fills it from the ancestor instead


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_hs_lambda_plumbs_and_changes_predictions(Est):
    """hs_lambda reaches the fitted booster, hs=0 is the default path, and a
    positive value perturbs predictions while keeping the model usable."""
    if Est is ChimeraBoostClassifier:
        X, y = load_breast_cancer(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=0)
        base = Est(n_estimators=120, random_state=0).fit(Xtr, ytr)
        hs = Est(n_estimators=120, hs_lambda=2.0, random_state=0).fit(Xtr, ytr)
        assert base.model_.hs_lambda == 0.0 and hs.model_.hs_lambda == 2.0
        pb = base.predict_proba(Xte)[:, 1]
        ph = hs.predict_proba(Xte)[:, 1]
        assert not np.allclose(pb, ph)
        assert roc_auc_score(yte, ph) > 0.95           # still strong
    else:
        X, y = load_diabetes(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=0)
        base = Est(n_estimators=120, random_state=0).fit(Xtr, ytr)
        hs = Est(n_estimators=120, hs_lambda=2.0, random_state=0).fit(Xtr, ytr)
        assert base.model_.hs_lambda == 0.0 and hs.model_.hs_lambda == 2.0
        assert not np.allclose(base.predict(Xte), hs.predict(Xte))


# ---------------------------------------------------------------------------
# linear leaf models (linear_leaves)
# ---------------------------------------------------------------------------

def test_linear_leaves_beat_constant_on_smooth_target():
    """On a smooth ~linear target with a limited tree budget, per-leaf linear
    models fit the slope within each leaf far better than constant leaves."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(3000, 3))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + 0.05 * rng.normal(size=3000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    common = dict(n_estimators=60, depth=3, random_state=0, thread_count=4)
    const = ChimeraBoostRegressor(**common).fit(Xtr, ytr)
    lin = ChimeraBoostRegressor(linear_leaves=True, **common).fit(Xtr, ytr)
    r_const = mean_squared_error(yte, const.predict(Xte)) ** 0.5
    r_lin = mean_squared_error(yte, lin.predict(Xte)) ** 0.5
    assert r_lin < 0.5 * r_const          # a large, unambiguous improvement


def _big_reg(n=1500, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.4 * X[:, 2] + 0.1 * rng.normal(size=n)
    return X, y


def test_linear_leaves_default_off_uses_fused_path():
    """linear_leaves defaults to off: the fast fused-forest path is used
    (no centers table built), so default behavior is unchanged."""
    X, y = _big_reg()
    off = ChimeraBoostRegressor(n_estimators=30, random_state=0).fit(X, y)
    on = ChimeraBoostRegressor(n_estimators=30, linear_leaves=True,
                               random_state=0).fit(X, y)
    assert off.model_._centers_std_ is None        # fused (constant) path
    assert on.model_._centers_std_ is not None      # linear path active (n>=1000)
    assert on.model_.linear_leaves is True


def test_linear_leaves_small_data_guard_falls_back_to_constant():
    """Below LINEAR_LEAVES_MIN_SAMPLES rows, linear leaves silently fall back to
    constant leaves (noisy small data overfits per-leaf slopes) -- so the result
    is bitwise identical to a plain constant-leaf model."""
    from chimeraboost.booster import LINEAR_LEAVES_MIN_SAMPLES
    X, y = load_diabetes(return_X_y=True)          # 442 rows < the guard
    assert len(X) < LINEAR_LEAVES_MIN_SAMPLES
    const = ChimeraBoostRegressor(n_estimators=40, random_state=0).fit(X, y)
    lin = ChimeraBoostRegressor(n_estimators=40, linear_leaves=True,
                                random_state=0).fit(X, y)
    assert lin.model_._centers_std_ is None        # guard tripped -> constant
    assert np.array_equal(const.predict(X), lin.predict(X))   # bit-identical


def test_linear_leaves_predict_matches_staged_and_is_finite():
    """The fused-bypass predict path and the staged per-tree path agree, and
    linear-leaf predictions are finite (no solve blow-ups)."""
    X, y = _big_reg(seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    m = ChimeraBoostRegressor(n_estimators=40, linear_leaves=True,
                              random_state=1, early_stopping=False).fit(Xtr, ytr)
    assert m.model_._centers_std_ is not None       # linear actually engaged
    pred = m.predict(Xte)
    staged_last = list(m.staged_predict(Xte))[-1]
    assert np.all(np.isfinite(pred))
    assert np.allclose(pred, staged_last)


def test_linear_leaves_classifier_auto_default():
    """The classifier default (linear_leaves=None) auto-enables linear leaves for
    BINARY (above the size guard) and disables them for multiclass WITHOUT
    raising -- only an explicit True on multiclass raises."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 5))
    yb = (rng.random(n) < 1 / (1 + np.exp(-(1.4 * X[:, 0] - X[:, 1])))).astype(int)
    mb = ChimeraBoostClassifier(random_state=0, thread_count=4).fit(X, yb)
    assert mb.model_.linear_leaves is True                # auto-on for binary
    assert mb.model_._centers_std_ is not None
    # multiclass default must NOT raise and must train fine (linear auto-off).
    ym = rng.integers(0, 3, size=n)
    mm = ChimeraBoostClassifier(random_state=0, thread_count=4).fit(X, ym)
    assert mm.predict(X).shape == (n,)


def test_linear_leaves_multiclass_explicit_true_raises():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    with pytest.raises(NotImplementedError, match="multiclass"):
        ChimeraBoostClassifier(linear_leaves=True, random_state=0).fit(X, y)


def test_linear_leaves_binary_runs_and_keeps_auc():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))                  # >= guard so linear engages
    logit = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.8 * X[:, 2]
    y = (rng.random(2000) < 1 / (1 + np.exp(-logit))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    m = ChimeraBoostClassifier(linear_leaves=True, random_state=0,
                               thread_count=4).fit(Xtr, ytr)
    assert m.model_._centers_std_ is not None        # linear engaged
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.85


@pytest.mark.parametrize("Est", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_sklearn_check_estimator_compliance(Est):
    """Full sklearn check_estimator must pass, except the one documented
    deviation (sample_weight is not bit-exactly equivalent to row repetition)."""
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(Est(), expected_failed_checks={
        "check_sample_weight_equivalence_on_dense_data":
            "weights reweight the loss but are not bit-exactly equivalent to "
            "integer sample repetition (documented deviation)",
    })


# ---- SHAP (exact interventional TreeSHAP) -----------------------------------

def _shap_efficiency_err(model_pred, phi, expected_value):
    """Max |sum_features(phi) + expected_value - prediction| over rows."""
    return np.abs(phi.sum(axis=1) + expected_value - model_pred).max()


def test_shap_efficiency_regression_linear_leaves():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 6))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3] + 0.3 * rng.normal(size=n)
    m = ChimeraBoostRegressor(n_estimators=80, depth=5, linear_leaves=True,
                              random_state=0).fit(X, y)
    phi = m.shap_values(X[:50])
    # Shapley efficiency must hold exactly, with the linear-leaf slopes included.
    assert _shap_efficiency_err(m.predict(X[:50]), phi, m.expected_value_) < 1e-6
    assert phi.shape == (50, 6)


def test_shap_efficiency_regression_constant_leaves():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 5))
    y = X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=800)
    m = ChimeraBoostRegressor(n_estimators=60, depth=4, linear_leaves=False,
                              random_state=0).fit(X, y)
    phi = m.shap_values(X[:40])
    assert _shap_efficiency_err(m.predict(X[:40]), phi, m.expected_value_) < 1e-6


def test_shap_efficiency_binary_logodds():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1500, 6))
    score = 2 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3]
    y = (score + 0.3 * rng.normal(size=1500) > 0).astype(int)
    m = ChimeraBoostClassifier(n_estimators=80, depth=5, random_state=0).fit(X, y)
    phi = m.shap_values(X[:50])
    # Classifier SHAP is in pre-temperature log-odds (margin) space.
    raw = m.model_.predict_raw(X[:50])
    assert _shap_efficiency_err(raw, phi, m.expected_value_) < 1e-6


def test_shap_efficiency_bagged():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 6))
    y = 2 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=1500)
    m = ChimeraBoostRegressor(n_estimators=60, depth=4, n_ensembles=3,
                              linear_leaves=True, random_state=0).fit(X, y)
    phi = m.shap_values(X[:40])
    # The bag prediction is the members' mean, so averaged SHAP stays exact.
    assert _shap_efficiency_err(m.predict(X[:40]), phi, m.expected_value_) < 1e-6


def test_shap_null_feature_is_negligible():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1500, 6))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.3 * rng.normal(size=1500)  # 5 unused
    m = ChimeraBoostRegressor(n_estimators=80, depth=5, random_state=0).fit(X, y)
    imp = np.abs(m.shap_values(X[:100])).mean(axis=0)
    # A feature absent from the target should carry near-zero attribution.
    assert imp[5] < 0.1 * imp[0]


def test_shap_maps_to_original_feature_space_with_categoricals():
    rng = np.random.default_rng(5)
    num = rng.normal(size=600)
    cat = rng.integers(0, 4, size=600).astype(object)
    X = np.column_stack([num, cat])
    y = (num + (cat.astype(int) == 2)).astype(float)
    m = ChimeraBoostRegressor(n_estimators=40, depth=4, random_state=0)
    m.fit(X, y, cat_features=[1])
    phi = m.shap_values(X[:30])
    # Attribution is reported in the user's 2-column input space, not the wider
    # internal (target-encoded / combo) matrix, and still satisfies efficiency.
    assert phi.shape == (30, 2)
    assert _shap_efficiency_err(m.predict(X[:30]), phi, m.expected_value_) < 1e-6


def test_shap_custom_background():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(1200, 5))
    y = X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=1200)
    m = ChimeraBoostRegressor(n_estimators=60, depth=4, random_state=0).fit(X, y)
    bg = X[:100]
    phi = m.shap_values(X[:30], X_background=bg)
    # expected_value_ must equal the mean prediction over the supplied background.
    assert abs(m.expected_value_ - m.predict(bg).mean()) < 1e-6
    assert _shap_efficiency_err(m.predict(X[:30]), phi, m.expected_value_) < 1e-6


def test_shap_multiclass_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 4))
    y = rng.integers(0, 3, size=300)
    m = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        m.shap_values(X[:10])


# --- validation_history_ property and callbacks= fit hook ---------------------

def test_validation_history_best_iteration_is_curve_argmin():
    """Under early stopping the recorded curve covers every round actually run
    (best round + patience), and best_iteration_ is its argmin + 1 -- so the
    kept-tree count is recoverable from the curve alone."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    PATIENCE = 20
    m = ChimeraBoostClassifier(
        n_estimators=1000, early_stopping_rounds=PATIENCE, random_state=0)
    m.fit(Xtr, ytr, eval_set=(Xte, yte))
    hist = m.validation_history_
    assert m.best_iteration_ < 1000
    assert int(np.argmin(hist)) + 1 == m.best_iteration_
    # Stopped because patience ran out after the best round (not at the horizon).
    assert len(hist) == m.best_iteration_ + PATIENCE


def test_validation_history_full_curve_without_early_stopping():
    """early_stopping=False + explicit eval_set => the complete curve to the
    horizon (n_estimators), never truncated by the stopper."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    HORIZON = 60
    m = ChimeraBoostClassifier(
        n_estimators=HORIZON, early_stopping=False, random_state=0)
    m.fit(Xtr, ytr, eval_set=(Xte, yte))
    assert len(m.validation_history_) == HORIZON


def test_validation_history_matches_manual_staged_logloss():
    """The cheap per-round curve equals a manual staged log-loss evaluation on
    the same eval_set -- so the harness can trust valid_history_ as the metric
    at every iteration count from a single fit."""
    from chimeraboost.losses import LOSSES
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    m = ChimeraBoostClassifier(
        n_estimators=40, early_stopping=False, random_state=0)
    m.fit(Xtr, ytr, eval_set=(Xte, yte))
    # Manual: staged raw scores -> Logloss against the 0/1 eval labels.
    y01 = (yte == m.classes_[1]).astype(np.float64)
    loss = LOSSES["Logloss"]()
    manual = [loss.eval(y01, F)
              for F in m.model_.staged_predict_raw(Xte)]
    assert np.allclose(m.validation_history_, manual, atol=1e-9)


def test_callbacks_fire_once_per_round_and_can_stop():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    seen = []

    def record(it, train_loss, val_loss, model):
        seen.append((it, train_loss, val_loss))

    m = ChimeraBoostClassifier(
        n_estimators=15, early_stopping=False, random_state=0)
    m.fit(Xtr, ytr, eval_set=(Xte, yte), callbacks=record)
    # One call per round, monotonically increasing iteration index.
    assert [s[0] for s in seen] == list(range(15))
    # val_loss passed to the callback tracks the recorded curve.
    assert np.allclose([s[2] for s in seen], m.validation_history_, atol=1e-9)

    stopped = []

    def stop_at_5(it, train_loss, val_loss, model):
        stopped.append(it)
        return it >= 5

    m2 = ChimeraBoostClassifier(
        n_estimators=100, early_stopping=False, random_state=0)
    m2.fit(Xtr, ytr, eval_set=(Xte, yte), callbacks=stop_at_5)
    assert max(stopped) == 5
    assert m2.best_iteration_ == 6   # rounds 0..5 kept


def test_callbacks_rejected_for_bagging():
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(n_estimators=20, n_ensembles=3, random_state=0)
    with pytest.raises(ValueError, match="callbacks"):
        m.fit(X, y, callbacks=lambda *a: None)


def test_validation_history_regressor_and_multiclass():
    # Regressor curve present with explicit eval set.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 5))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=800)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    r = ChimeraBoostRegressor(
        n_estimators=30, early_stopping=False, random_state=0)
    r.fit(Xtr, ytr, eval_set=(Xte, yte))
    assert len(r.validation_history_) == 30
    # Multiclass keeps a single combined softmax-logloss curve.
    yc = rng.integers(0, 3, size=800)
    Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.3, random_state=0)
    c = ChimeraBoostClassifier(
        n_estimators=25, early_stopping=False, random_state=0)
    c.fit(Xtr, ytr, eval_set=(Xte, yte))
    assert len(c.validation_history_) == 25


# --- C1: one-hot low-cardinality categoricals (default-off flag) --------------

def test_onehot_low_card_is_noop_on_numeric_data():
    """The flag must be a true no-op when there are no categoricals (or none low-
    cardinality): predictions byte-identical to the default."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    a = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False).fit(X, y)
    b = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False,
                               onehot_low_card=True).fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_onehot_low_card_encodes_only_low_cardinality():
    """Only columns with <= onehot_max_card levels get a one-hot block; high-
    cardinality columns keep TS only. The flag changes predictions."""
    rng = np.random.default_rng(0)
    n = 2000
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = np.array([f"c{c}" for c in rng.integers(0, 5, n)], dtype=object)
    X[:, 1] = np.array([f"h{c}" for c in rng.integers(0, 50, n)], dtype=object)
    X[:, 2] = rng.normal(size=n)
    y = ((X[:, 0] == "c1") | (rng.random(n) < 0.3)).astype(int)
    m = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                               early_stopping=False, onehot_low_card=True)
    m.fit(X, y, cat_features=[0, 1])
    # The 5-level column (index 0 among cats) is one-hot; the 50-level one isn't.
    assert m.model_.prep_.onehot_specs_ == [(0, 5)]
    base = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                  early_stopping=False).fit(X, y, cat_features=[0, 1])
    assert not np.allclose(m.predict_proba(X), base.predict_proba(X))
    # Unseen categories at predict route to an all-zero one-hot row (no crash).
    Xnew = np.array([["c_UNSEEN", "h_UNSEEN", 0.1], ["c1", "h3", -0.2]],
                    dtype=object)
    assert m.predict_proba(Xnew).shape == (2, 2)


def test_onehot_max_card_validation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="onehot_max_card"):
        ChimeraBoostClassifier(onehot_max_card=1).fit(X, y)


# --- C3: selective cat_combinations on mixed data (default-off flag) -----------

def test_cat_combinations_selective_is_noop_on_numeric_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    a = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False).fit(X, y)
    b = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False,
                               cat_combinations_selective=True).fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_cat_combinations_selective_fires_on_mixed_data():
    """On MIXED data the plain auto-rule keeps no combos, but the selective flag
    picks high-MI interaction pairs (here A x B drives an XOR target) and changes
    predictions. Unseen categories at predict still work."""
    rng = np.random.default_rng(0)
    n = 3000
    A, B, C = (rng.integers(0, 4, n), rng.integers(0, 4, n),
               rng.integers(0, 6, n))
    num = rng.normal(size=(n, 2))
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = [f"a{v}" for v in A]
    X[:, 1] = [f"b{v}" for v in B]
    X[:, 2] = [f"c{v}" for v in C]
    X[:, 3], X[:, 4] = num[:, 0], num[:, 1]
    y = (((A % 2) ^ (B % 2)).astype(bool) | (num[:, 0] > 1.2)).astype(int)
    base = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                  early_stopping=False).fit(X, y,
                                                            cat_features=[0, 1, 2])
    sel = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                 early_stopping=False,
                                 cat_combinations_selective=True
                                 ).fit(X, y, cat_features=[0, 1, 2])
    assert base.model_.prep_.combo_pairs_ == []          # auto-off on mixed
    assert (0, 1) in sel.model_.prep_.combo_pairs_       # the XOR-driving pair
    assert not np.allclose(base.predict_proba(X), sel.predict_proba(X))
    Xnew = X[:3].copy()
    Xnew[0, 0] = "a_UNSEEN"
    assert sel.predict_proba(Xnew).shape == (3, 2)


def test_cat_combinations_max_pairs_caps_selection():
    rng = np.random.default_rng(1)
    n = 2000
    X = np.empty((n, 5), dtype=object)
    cols = [rng.integers(0, 4, n) for _ in range(5)]
    for j, c in enumerate(cols):
        X[:, j] = [f"v{v}" for v in c]
    y = ((cols[0] % 2) ^ (cols[1] % 2) ^ (cols[2] % 2)).astype(int)
    m = ChimeraBoostClassifier(n_estimators=40, random_state=0,
                               early_stopping=False,
                               cat_combinations_selective=True,
                               cat_combinations_max_pairs=2
                               ).fit(X, y, cat_features=[0, 1, 2, 3, 4])
    assert len(m.model_.prep_.combo_pairs_) <= 2


# --- G1: forest-level joint leaf refit (default-off flag) ---------------------

def test_forest_leaf_refit_is_noop_when_off():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=1500)
    a = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                              early_stopping=False).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                              early_stopping=False).fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X))    # determinism baseline
    c = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                              early_stopping=False,
                              forest_leaf_refit=True).fit(X, y)
    assert not np.allclose(a.predict(X), c.predict(X))


def test_forest_leaf_refit_solves_ridge_optimum():
    """The joint refit must reach the ridge stationarity point: with prediction
    init + Z @ w (Z = leaf-membership over all trees), the RMSE-ridge gradient
    Z.T(pred - y) + lambda * w is ~0. This pins the matrix-free CG result to the
    explicit closed-form solve."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3))
    y = 0.7 * X[:, 0] + 0.3 * rng.normal(size=400)
    lam = 2.0
    m = ChimeraBoostRegressor(n_estimators=8, depth=3, random_state=0,
                              early_stopping=False, l2_leaf_reg=lam,
                              forest_leaf_refit=True,
                              forest_refit_iterations=2).fit(X, y)
    gb = m.model_
    Xb = np.ascontiguousarray(gb.prep_.transform(X).T)
    trees = [t for t in gb.trees_ if t.depth > 0 and t.lin_coef is None]
    sizes = [t.values.shape[0] for t in trees]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    n, L = Xb.shape[1], int(offs[-1])
    Z = np.zeros((n, L))
    for k, t in enumerate(trees):
        Z[np.arange(n), offs[k] + t.apply(Xb)] = 1.0
    w = np.concatenate([t.values for t in trees])
    pred = gb.init_ + Z @ w
    assert np.allclose(pred, m.predict(X), atol=1e-6)          # consistency
    grad = Z.T @ (pred - y) + lam * w
    assert np.linalg.norm(grad) < 1e-2                         # ridge optimum
    # And it lowers training RMSE vs no refit.
    base = ChimeraBoostRegressor(n_estimators=8, depth=3, random_state=0,
                                 early_stopping=False, l2_leaf_reg=lam).fit(X, y)
    assert np.sqrt(np.mean((m.predict(X) - y) ** 2)) < \
        np.sqrt(np.mean((base.predict(X) - y) ** 2))


def test_forest_leaf_refit_binary_lowers_logloss():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    base = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                  early_stopping=False).fit(Xtr, ytr)
    ref = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                 early_stopping=False,
                                 forest_leaf_refit=True).fit(Xtr, ytr)
    from sklearn.metrics import log_loss
    assert log_loss(ytr, ref.predict_proba(Xtr)) < log_loss(ytr, base.predict_proba(Xtr))


def test_forest_leaf_refit_multiclass_warns_and_ignored():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = rng.integers(0, 3, size=600)
    with pytest.warns(UserWarning, match="forest_leaf_refit"):
        ChimeraBoostClassifier(n_estimators=30, random_state=0,
                               early_stopping=False,
                               forest_leaf_refit=True).fit(X, y)


# --- G4: ordered_boosting + leaf_estimation reconciliation (default-off flag) --

def test_ordered_leaf_estimation_fires_only_under_ordered_boosting():
    """ordered_leaf_estimation refines the leaf values that predict uses while
    F keeps the ordered LOO trajectory. It must change predictions when
    ordered_boosting is on, and be a byte-identical no-op when it is off (the
    flag only acts in the ordered-boosting branch)."""
    X, y = load_breast_cancer(return_X_y=True)
    ob = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                early_stopping=False,
                                ordered_boosting=True).fit(X, y)
    g4 = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                early_stopping=False, ordered_boosting=True,
                                ordered_leaf_estimation=True).fit(X, y)
    assert not np.allclose(ob.predict_proba(X), g4.predict_proba(X))
    # off -> no-op
    base = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                  early_stopping=False).fit(X, y)
    flag = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                                  early_stopping=False,
                                  ordered_leaf_estimation=True).fit(X, y)
    assert np.allclose(base.predict_proba(X), flag.predict_proba(X))


# --- G3: size-adaptive leaf_estimation_iterations (default-off flag) -----------

def test_adaptive_leaf_estimation_resolves_by_size_and_is_off_by_default():
    from chimeraboost.sklearn_api import _adaptive_leaf_estimation_iterations as f
    assert f(300) == 1 and f(2000) == 3 and f(40000) == 6   # monotone, capped
    X, y = load_breast_cancer(return_X_y=True)               # ~400 train rows
    base = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                  early_stopping=False).fit(X, y)
    adapt = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                   early_stopping=False,
                                   adaptive_leaf_estimation=True).fit(X, y)
    assert base.model_.leaf_estimation_iterations == 3       # class default
    assert adapt.model_.leaf_estimation_iterations == 1      # size-resolved
    assert not np.allclose(base.predict_proba(X), adapt.predict_proba(X))


# --- G2: mass-adaptive per-leaf shrinkage (default-off, alpha=0) ---------------

def test_adaptive_leaf_shrinkage_noop_at_zero_and_shrinks_when_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=1500)
    base = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                                 early_stopping=False).fit(X, y)
    zero = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                                 early_stopping=False,
                                 adaptive_leaf_shrinkage=0.0).fit(X, y)
    assert np.allclose(base.predict(X), zero.predict(X))      # alpha=0 -> no-op
    on = ChimeraBoostRegressor(n_estimators=50, random_state=1,
                               early_stopping=False,
                               adaptive_leaf_shrinkage=5.0).fit(X, y)
    assert not np.allclose(base.predict(X), on.predict(X))
    with pytest.raises(ValueError, match="adaptive_leaf_shrinkage"):
        ChimeraBoostRegressor(adaptive_leaf_shrinkage=-1.0).fit(X, y)


def test_adaptive_leaf_shrinkage_kernel_matches_formula():
    """The mass-shrinkage kernel must equal -lr*G/(H+l2) * H/(H+alpha)."""
    from chimeraboost.tree import _leaf_values, _leaf_values_adaptive
    rng = np.random.default_rng(2)
    leaf = rng.integers(0, 4, 200).astype(np.int64)
    g = rng.normal(size=200)
    h = rng.random(200) + 0.1
    plain = _leaf_values(leaf, g, h, 4, 1.0, 0.1)
    alpha = 3.0
    adapt = _leaf_values_adaptive(leaf, g, h, 4, 1.0, 0.1, alpha)
    # reconstruct H per leaf to form the expected factor
    H = np.zeros(4)
    np.add.at(H, leaf, h)
    factor = H / (H + alpha)
    assert np.allclose(adapt, plain * factor)


# --- C4: cat-aware binning (default-off flag) ---------------------------------

def test_cat_aware_binning_gives_cat_columns_more_bins_and_is_noop_on_numeric():
    rng = np.random.default_rng(0)
    n = 3000
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = [f"c{c}" for c in rng.integers(0, 30, n)]   # high-card categorical
    X[:, 1] = rng.normal(size=n)
    X[:, 2] = rng.normal(size=n)
    y = ((rng.integers(0, 30, n) % 2) | (X[:, 1].astype(float) > 0.5)).astype(int)
    base = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                  early_stopping=False).fit(X, y, cat_features=[0])
    caw = ChimeraBoostClassifier(n_estimators=60, random_state=0,
                                 early_stopping=False,
                                 cat_aware_binning=True).fit(X, y, cat_features=[0])
    # The target-encoded categorical column gets the larger (cat_max_bins) budget;
    # numeric columns keep max_bins.
    assert caw.model_.prep_.binner_.n_bins_.max() > \
        base.model_.prep_.binner_.n_bins_.max()
    assert not np.allclose(base.predict_proba(X), caw.predict_proba(X))
    # No categoricals -> no effect.
    Xn = rng.normal(size=(1500, 4))
    yn = (Xn[:, 0] > 0).astype(int)
    a = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False).fit(Xn, yn)
    b = ChimeraBoostClassifier(n_estimators=40, random_state=1,
                               early_stopping=False,
                               cat_aware_binning=True).fit(Xn, yn)
    assert np.allclose(a.predict_proba(Xn), b.predict_proba(Xn))
