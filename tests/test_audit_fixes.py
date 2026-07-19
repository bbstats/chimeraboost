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


def test_leaf_estimation_iterations_warns_on_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.choice([0, 1, 2], size=300)
    clf = ChimeraBoostClassifier(n_estimators=10, random_state=0,
                                 leaf_estimation_iterations=8)
    with pytest.warns(UserWarning, match="not implemented for multiclass"):
        clf.fit(X, y)


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
