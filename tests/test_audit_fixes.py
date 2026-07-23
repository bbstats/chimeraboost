"""Regression tests for the 2026-07-19 audit fixes: silent-failure guards,
thread hygiene, pickle slimming, and the previously-untested invariants
(pickle round-trip, same-seed determinism)."""

import pickle
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


# ---------------------------------------------------------------- invariants


def test_pickle_round_trip_identical_predictions():
    X, y = _toy_regression()
    m = ChimeraBoostRegressor(n_estimators=30, random_state=0).fit(X, y)
    clone = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict(X), clone.predict(X))

    Xc, yc = _toy_binary()
    c = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(Xc, yc)
    cclone = pickle.loads(pickle.dumps(c))
    np.testing.assert_array_equal(c.predict_proba(Xc),
                                  cclone.predict_proba(Xc))


def test_pickle_not_bloated_by_predict_cache():
    """The lazily-built packed-forest cache is dropped at pickle time, so a
    model pickled after predicting is no bigger than one pickled fresh."""
    X, y = _toy_regression(n=800)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
    before = len(pickle.dumps(m))
    m.predict(X)                       # builds the packed-forest cache
    after = len(pickle.dumps(m))
    assert after <= before * 1.05
    # And the reloaded model still predicts (cache rebuilds lazily).
    clone = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict(X), clone.predict(X))


def test_same_seed_bit_identical():
    X, y = _toy_regression()
    p1 = ChimeraBoostRegressor(n_estimators=30, random_state=7).fit(X, y).predict(X)
    p2 = ChimeraBoostRegressor(n_estimators=30, random_state=7).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)

    Xc, yc = _toy_binary()
    q1 = ChimeraBoostClassifier(n_estimators=30, random_state=7).fit(Xc, yc)
    q2 = ChimeraBoostClassifier(n_estimators=30, random_state=7).fit(Xc, yc)
    np.testing.assert_array_equal(q1.predict_proba(Xc), q2.predict_proba(Xc))


# ------------------------------------------------------------ thread hygiene


def test_thread_count_restored_after_fit_and_predict():
    """fit/predict with an explicit thread_count must not leak the setting
    into the process (numba.set_num_threads is global)."""
    import numba
    ambient = numba.get_num_threads()
    X, y = _toy_regression(n=200)
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0,
                              thread_count=1).fit(X, y)
    assert numba.get_num_threads() == ambient
    m.predict(X)
    assert numba.get_num_threads() == ambient


# ------------------------------------------------------- loud-failure guards


def test_eval_set_label_unseen_in_training_raises():
    Xc, yc = _toy_binary()
    Xv, yv = Xc[:50].copy(), yc[:50].copy()
    yv[0] = 2                                 # label the training y never has
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="not present in y"):
        clf.fit(Xc, yc, eval_set=(Xv, yv))


def test_eval_set_label_unseen_multiclass_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.choice([0, 1, 3], size=300)       # note: no class 2
    yv = y[:60].copy()
    yv[0] = 2
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="not present in y"):
        clf.fit(X, y, eval_set=(X[:60], yv))


def test_grouped_split_never_silently_drops_a_class():
    """A class confined to one group must either survive the auto-split or
    raise -- never silently train as a (K-1)-class model."""
    rng = np.random.default_rng(0)
    n0 = 135
    X = rng.normal(size=(2 * n0 + 30, 4))
    y = np.array([0] * n0 + [1] * n0 + [2] * 30)
    groups = np.concatenate([rng.integers(0, 10, size=2 * n0),
                             np.full(30, 10)])   # class 2 lives in group 10 only
    saw_raise = False
    for seed in range(12):
        clf = ChimeraBoostClassifier(n_estimators=10, random_state=seed,
                                     validation_fraction=0.5)
        try:
            clf.fit(X, y, groups=groups)
        except ValueError as e:
            assert "entirely in the validation set" in str(e)
            saw_raise = True
        else:
            assert clf.n_classes_ == 3
    assert saw_raise, "expected at least one seed to put group 10 in validation"


def test_classifier_depth_none_resolves_to_default():
    Xc, yc = _toy_binary(n=200)
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0,
                                 depth=None).fit(Xc, yc)
    assert clf.predict(Xc).shape == (200,)


# ------------------------------------------------------ cat_features guards


def test_cat_features_accepts_numpy_int_array():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=300),
                         rng.integers(0, 4, size=300).astype(float)])
    y = X[:, 0] + X[:, 1]
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0)
    m.fit(X, y, cat_features=np.array([1]))    # ndarray, not list
    assert np.isfinite(m.predict(X)).all()


def test_cat_features_float_array_names_sample_weight():
    """fit(X, y, w) binds w to cat_features; the error must name the real
    mistake instead of a generic type complaint."""
    X, y = _toy_regression(n=100)
    w = np.abs(np.random.default_rng(0).normal(size=100))
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="sample_weight=w"):
        m.fit(X, y, w)


# ------------------------------------------------------- inert-knob warnings


def test_ordered_boosting_warns_when_linear_leaves_shadow_it():
    Xc, yc = _toy_binary(n=1600)               # >= 1000 post-split train rows
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0,
                                 ordered_boosting=True)
    with pytest.warns(UserWarning, match="ordered_boosting is ignored"):
        clf.fit(Xc, yc)


def test_leaf_estimation_iterations_default_is_none_auto():
    # H1/H2: the classifier's default is None (auto), not a concrete 3, so the
    # API stops advertising a refinement count that is inert for multiclass and
    # shadowed by linear leaves. None resolves to 3 for the constant-leaf path.
    assert ChimeraBoostClassifier().leaf_estimation_iterations is None


def test_leaf_estimation_iterations_auto_resolves_to_three():
    # The auto default must reproduce the historical effective value (3) exactly
    # -- i.e. be bit-identical to explicitly passing 3 -- on a constant-leaf
    # binary fit where refinement is live (linear leaves off).
    Xc, yc = _toy_binary(n=1500, seed=2)
    common = dict(n_estimators=40, random_state=0, linear_leaves=False)
    p_auto = ChimeraBoostClassifier(**common).fit(Xc, yc).predict_proba(Xc)
    p_three = ChimeraBoostClassifier(leaf_estimation_iterations=3,
                                     **common).fit(Xc, yc).predict_proba(Xc)
    np.testing.assert_array_equal(p_auto, p_three)


def test_leaf_estimation_iterations_auto_is_quiet_on_multiclass():
    # The auto default (None) must NOT warn on multiclass -- only an explicitly
    # set, ignored value should.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.choice([0, 1, 2], size=300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        ChimeraBoostClassifier(n_estimators=10, random_state=0).fit(X, y)


def test_regressor_accepts_none_leaf_estimation_iterations():
    # The shared validator now accepts None (the classifier's auto default);
    # the regressor resolves an explicit None to its concrete default 1.
    X, y = _toy_regression(n=300)
    common = dict(n_estimators=20, random_state=0)
    p_none = ChimeraBoostRegressor(leaf_estimation_iterations=None,
                                   **common).fit(X, y).predict(X)
    p_one = ChimeraBoostRegressor(leaf_estimation_iterations=1,
                                  **common).fit(X, y).predict(X)
    np.testing.assert_array_equal(p_none, p_one)


@pytest.mark.parametrize("lei", [2, 3, 8])
def test_leaf_estimation_iterations_warns_on_multiclass(lei):
    # Any explicitly-set lei>1 that is ignored on multiclass warns honestly
    # (the former default 3 no longer gets a silent exemption).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.choice([0, 1, 2], size=300)
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0,
                                 leaf_estimation_iterations=lei)
    with pytest.warns(UserWarning, match="not implemented for multiclass"):
        clf.fit(X, y)


@pytest.mark.parametrize("lei", [2, 3])
def test_leaf_estimation_iterations_warns_when_linear_leaves_shadow_it(lei):
    Xc, yc = _toy_binary(n=1600)               # >= 1000 rows -> linear leaves on
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0,
                                 leaf_estimation_iterations=lei)
    with pytest.warns(UserWarning,
                      match="leaf_estimation_iterations is ignored"):
        clf.fit(Xc, yc)


def test_ordered_boosting_warns_on_mae():
    X, y = _toy_regression()
    m = ChimeraBoostRegressor(n_estimators=10, random_state=0, loss="MAE",
                              ordered_boosting=True)
    with pytest.warns(UserWarning, match="ordered_boosting is ignored"):
        m.fit(X, y)


# ------------------------------------------------- bagged feature names (H6)


def test_bagged_members_carry_feature_names():
    X, y = _toy_regression(n=600)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    bag = ChimeraBoostRegressor(n_estimators=20, random_state=0, n_ensembles=2,
                                learning_rate=0.1, colsample=1.0,
                                ensemble_n_jobs=1).fit(df, y)
    for member in bag.estimators_:
        assert list(member.feature_names_in_) == list(df.columns)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        bag.predict(df)
    assert not [w for w in rec
                if "fitted without feature names" in str(w.message)]
    # The member-level column-order guard now applies too.
    with pytest.raises(ValueError, match="feature names"):
        bag.predict(df[list(df.columns[::-1])])


# ---------------------------------------------------------------- H3: weights
# sample_weight=0 rows must not shape the target encoder, the bin borders, or
# the early-stopping metric. See the audit note H3 / project-audit-2026-07-19.


def _garbage_weight_data(seed=0, n=4000, n_garbage=800):
    """Clean signal plus zero-weight 'garbage' rows with wild targets and
    extreme feature values. A correct fit treats the garbage as ghosts."""
    rng = np.random.default_rng(seed)
    Xc = rng.normal(size=(n, 6))
    yc = Xc[:, 0] * 2.0 + Xc[:, 1] - 0.5 * Xc[:, 2] + 0.1 * rng.normal(size=n)
    Xg = rng.normal(size=(n_garbage, 6)) * 50.0
    yg = rng.normal(size=n_garbage) * 1000.0 + 5000.0
    X = np.vstack([Xc, Xg])
    y = np.concatenate([yc, yg])
    w = np.concatenate([np.ones(n), np.zeros(n_garbage)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm], w[perm]


def test_h3_encoder_totals_are_weighted():
    """The ordered-target encoder's prior and per-category totals weight each
    row by sample_weight, so a zero-weight row contributes nothing."""
    from chimeraboost.target_encoding import OrderedTargetEncoder
    rng = np.random.default_rng(0)
    n, g = 2000, 400
    codes = np.concatenate([rng.integers(0, 5, n), rng.integers(0, 5, g)])
    y = np.concatenate([rng.normal(size=n), np.full(g, 9999.0)])
    w = np.concatenate([np.ones(n), np.zeros(g)])
    enc = OrderedTargetEncoder(smoothing=1.0, random_state=0, n_permutations=4)
    enc.fit_transform(codes.reshape(-1, 1), y, w)
    # Prior is the weighted mean -> the 9999 garbage rows drop out entirely.
    assert np.isclose(enc.prior_, np.average(y, weights=w))
    assert enc.prior_ < 10.0            # not dragged toward 9999
    # Per-category totals equal the weighted sums/counts over positive rows.
    for c in range(5):
        m = codes == c
        assert np.isclose(enc.sums_[0][c], np.sum(w[m] * y[m]))
        assert np.isclose(enc.counts_[0][c], np.sum(w[m]))


def test_h3_bin_borders_drop_zero_weight_rows():
    """Zero-weight rows must not place a bin edge: extreme values carried only
    by weight-0 rows never appear in the learned borders."""
    from chimeraboost.binning import _feature_borders
    col = np.array([0.0, 1.0, 2.0, 100.0, 100.0])
    w = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    borders = _feature_borders(col, max_bins=128, weights=w)
    # Only {0,1,2} survive -> midpoints, and nothing near the weight-0 value 100.
    np.testing.assert_allclose(borders, [0.5, 1.5])
    assert borders.max() < 3.0
    # The unweighted path still sees the 100s (regression guard on the default).
    plain = _feature_borders(col, max_bins=128, weights=None)
    assert plain.max() > 3.0


def test_h3_early_stopping_metric_ignores_zero_weight_rows():
    """The headline leak: zero-weight rows landing in the auto-split validation
    fold must not corrupt the early-stopping metric or wreck the fit."""
    X, y, w = _garbage_weight_data(seed=2)
    keep = w > 0
    m_zero = ChimeraBoostRegressor(
        n_estimators=500, early_stopping=True, validation_fraction=0.2,
        early_stopping_rounds=20, random_state=0).fit(X, y, sample_weight=w)
    m_drop = ChimeraBoostRegressor(
        n_estimators=500, early_stopping=True, validation_fraction=0.2,
        early_stopping_rounds=20, random_state=0).fit(X[keep], y[keep])
    # Validation loss is on the signal scale (~0.1), not the garbage scale (~2e3).
    assert min(m_zero.model_.valid_history_) < 1.0
    # And the model is not truncated to near-nothing by a corrupted metric.
    assert m_zero.best_iteration_ > 20
    rng = np.random.default_rng(99)
    Xte = rng.normal(size=(2000, 6))
    yte = Xte[:, 0] * 2.0 + Xte[:, 1] - 0.5 * Xte[:, 2]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    # Predictions track the rows-removed model closely (not off by an order
    # of magnitude, as the corrupted-metric fit was).
    assert rmse(m_zero.predict(Xte), m_drop.predict(Xte)) < 0.1


def test_h3_classifier_early_stopping_ignores_zero_weight_rows():
    """Same guard for the binary classifier (eval_set is relabeled 0/1 and must
    carry the val weights through)."""
    rng = np.random.default_rng(0)
    n, g = 3000, 600
    Xc = rng.normal(size=(n, 6))
    yc = (Xc[:, 0] + Xc[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    Xg = rng.normal(size=(g, 6)) * 50.0
    yg = (rng.normal(size=g) > 0).astype(int)   # noise labels on extreme X
    X = np.vstack([Xc, Xg]); y = np.concatenate([yc, yg])
    w = np.concatenate([np.ones(n), np.zeros(g)])
    perm = rng.permutation(len(y))
    X, y, w = X[perm], y[perm], w[perm]
    m = ChimeraBoostClassifier(
        n_estimators=300, early_stopping=True, validation_fraction=0.2,
        early_stopping_rounds=20, random_state=0).fit(X, y, sample_weight=w)
    # Logloss on the clean signal is well under a random-guess ~0.69, which a
    # garbage-corrupted validation metric would never allow.
    assert min(m.model_.valid_history_) < 0.5


def test_h3_uniform_weight_bit_identical_with_cats_and_es():
    """Uniform weights collapse to the unweighted path everywhere, including the
    new weighted encoder/binner/val-metric code, so predictions stay bitwise
    identical to sample_weight=None even with categoricals + early stopping."""
    rng = np.random.default_rng(0)
    n = 2500
    cats = rng.integers(0, 6, size=n).astype(float)
    Xnum = rng.normal(size=(n, 4))
    X = np.column_stack([cats, Xnum])
    y = cats * 0.5 + Xnum[:, 0] - Xnum[:, 1] + 0.1 * rng.normal(size=n)
    Xte = np.column_stack([rng.integers(0, 6, 500).astype(float),
                           rng.normal(size=(500, 4))])
    kw = dict(n_estimators=120, early_stopping=True, early_stopping_rounds=20,
              random_state=0)
    m_none = ChimeraBoostRegressor(**kw).fit(X, y, cat_features=[0])
    m_ones = ChimeraBoostRegressor(**kw).fit(
        X, y, cat_features=[0], sample_weight=np.ones(n))
    np.testing.assert_array_equal(m_none.predict(Xte), m_ones.predict(Xte))


def _small_reg_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_a1_auto_split_notice_verbose_regressor(capsys):
    """verbose=True announces the silent 20% early-stopping holdout (A1)."""
    X, y = _small_reg_data()
    ChimeraBoostRegressor(n_estimators=5, verbose=True,
                          random_state=0).fit(X, y)
    out = capsys.readouterr().out
    assert "holding out 60 of 300 rows" in out
    assert "early_stopping=False" in out


def test_a1_auto_split_notice_verbose_classifier(capsys):
    X, y = _small_reg_data()
    y = (y > 0).astype(int)
    ChimeraBoostClassifier(n_estimators=5, verbose=True,
                           random_state=0).fit(X, y)
    out = capsys.readouterr().out
    assert "holding out 60 of 300 rows" in out


def test_a1_auto_split_notice_silent_by_default(capsys):
    """Default verbose=False stays quiet; so does an explicit eval_set."""
    X, y = _small_reg_data()
    ChimeraBoostRegressor(n_estimators=5, random_state=0).fit(X, y)
    assert "holding out" not in capsys.readouterr().out
    ChimeraBoostRegressor(n_estimators=5, verbose=True, random_state=0).fit(
        X[:240], y[:240], eval_set=(X[240:], y[240:]))
    assert "holding out" not in capsys.readouterr().out


# ------------------------------------------- 2026-07-23 semi-bug hunt fixes


def test_mae_quantile_leaf_estimation_iterations_is_truly_inert():
    """For MAE/Quantile, _correct_leaves sets the exact minimizer (median /
    alpha-quantile); leaf_estimation_iterations > 1 must not run sign-gradient
    Newton steps on top of it. The sklearn layer has always *warned* the
    parameter has no effect for these losses -- this pins that the warning is
    honest: predictions are identical whatever the value."""
    X, y = _toy_regression(n=600, seed=3)
    for loss in ("MAE", "Quantile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = ChimeraBoostRegressor(
                loss=loss, n_estimators=40, leaf_estimation_iterations=1,
                random_state=0).fit(X, y)
            lei3 = ChimeraBoostRegressor(
                loss=loss, n_estimators=40, leaf_estimation_iterations=3,
                random_state=0).fit(X, y)
        np.testing.assert_array_equal(base.predict(X), lei3.predict(X))


def test_depth0_stop_keeps_best_validation_prefix(monkeypatch):
    """A depth-0 tree ends boosting early; that exit must keep the best
    validation prefix exactly like patience and budget exhaustion do. Scripted
    trees: round 0 improves validation, rounds 1-2 worsen it, round 3 returns
    depth-0 -- the fitted model must keep only the one good tree."""
    import chimeraboost.booster as bst

    class _FakeTree:
        def __init__(self, depth, step):
            self.depth = depth
            self.values = np.full(2, step)
            self.gains = np.zeros(max(depth, 1))
            self.splits_feat = np.zeros(depth, dtype=np.int64)
            self.splits_thr = np.zeros(depth, dtype=np.int64)
            self.lin_feats = None
            self.lin_coef = None
            self.centers_std = None

        def predict(self, Xb):
            return np.full(Xb.shape[1], self.values[0])

    steps = iter([0.5, -0.5, -0.5])

    def fake_build(Xb, g, h, *args, **kw):
        step = next(steps, None)
        if step is None:
            return _FakeTree(0, 0.0), np.zeros(Xb.shape[1], dtype=np.int64)
        return _FakeTree(1, step), np.zeros(Xb.shape[1], dtype=np.int64)

    monkeypatch.setattr(bst, "build_oblivious_tree", fake_build)
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(64, 2))
    ytr = np.zeros(64)                    # init_ = 0
    Xv = rng.normal(size=(16, 2))
    yv = np.ones(16)                      # val RMSE: 1.0 -> 0.5 -> 1.0 -> 1.5
    b = bst.GradientBoosting(loss="RMSE", n_estimators=10,
                             early_stopping_rounds=50, linear_leaves=False,
                             learning_rate=0.1, random_state=0)
    b.fit(Xtr, ytr, eval_set=(Xv, yv))
    assert len(b.trees_) == 1, (
        f"depth-0 exit kept {len(b.trees_)} trees; the best validation "
        "prefix is 1 tree")


def test_grouped_classification_split_honors_random_state():
    """The grouped stratified ES split used an unshuffled StratifiedGroupKFold:
    random_state was inert and the holdout was always the same first fold.
    With a seed it now shuffles fold selection; None keeps the historical
    deterministic split."""
    from chimeraboost.sklearn_api import _make_eval_split

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) > 0.5).astype(int)
    groups = np.repeat(np.arange(40), 10)

    def val_groups(rs):
        tr, va = _make_eval_split(X, y, 0.2, rs, groups=groups, stratify=y)
        return frozenset(np.unique(groups[va]).tolist())

    # Seeded splits are reproducible...
    assert val_groups(0) == val_groups(0)
    assert val_groups(None) == val_groups(None)
    # ...and the seed actually selects different holdout groups.
    assert len({val_groups(rs) for rs in range(6)}) > 1


def test_tiny_batch_serial_predict_bit_identical():
    """Predict batches at/below the serial-dispatch threshold take serial
    kernel twins (the OpenMP fork/join dwarfs a 1-row walk); each row's result
    must equal the parallel path's bit-for-bit, on the constant, linear-leaf,
    and multiclass forests."""
    rng = np.random.default_rng(7)
    n = 1300
    X = rng.normal(size=(n, 4))
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    reg = ChimeraBoostRegressor(n_estimators=25, random_state=0).fit(X, y)
    yb = (y > 0).astype(int)
    clf = ChimeraBoostClassifier(n_estimators=25, random_state=0).fit(X, yb)
    ym = np.digitize(X[:, 0], [-0.5, 0.5])
    mc = ChimeraBoostClassifier(n_estimators=25, random_state=0).fit(X, ym)
    for m, pred in ((reg, reg.predict), (clf, clf.predict_proba),
                    (mc, mc.predict_proba)):
        big = pred(X[:16])
        for i in range(3):
            np.testing.assert_array_equal(pred(X[i:i + 1])[0], big[i])
