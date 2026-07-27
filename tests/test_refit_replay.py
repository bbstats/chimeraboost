"""Structure-transfer full-data refit (``refit_full="replay"``;
benchmarks/REPLAY_PLAN.md).

The full refit re-grows the early-stopping winner on all rows; replay reuses
the winner's split structures and refits only the leaf values, which skips the
split search (83-85% of a fit) while still letting the held-out rows reach the
leaf estimates.
"""

import pickle

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.tree import ObliviousTree, replay_oblivious_tree

from test_refit_full import _clf_data, _reg_data


def test_replay_matches_full_refit_round_count():
    """Same rounds scaling as the full refit -- only the cost differs."""
    X, y = _reg_data()
    full = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                 refit_full=True).fit(X, y)
    rep = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full="replay").fit(X, y)
    assert len(rep.model_.trees_) == len(full.model_.trees_)


def test_replay_is_its_own_model():
    """Distinct from both the no-refit and the from-scratch-refit models."""
    X, y = _reg_data()
    off = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full=False).fit(X, y)
    full = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                 refit_full=True).fit(X, y)
    rep = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full="replay").fit(X, y)
    assert not np.array_equal(rep.predict(X), off.predict(X))
    assert not np.array_equal(rep.predict(X), full.predict(X))


def test_replay_reuses_the_donor_structures():
    """The replayed trees carry the ES winner's splits verbatim; only the
    trailing rounds with no donor are grown fresh."""
    X, y = _reg_data()
    off = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full=False).fit(X, y)
    rep = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full="replay").fit(X, y)
    donor, replayed = off.model_.trees_, rep.model_.trees_
    assert len(replayed) >= len(donor)
    for d, r in zip(donor, replayed):
        np.testing.assert_array_equal(d.splits_feat, r.splits_feat)
        np.testing.assert_array_equal(d.splits_thr, r.splits_thr)
    # Leaf values were refit on more rows, so they must have moved.
    assert any(not np.array_equal(d.values, r.values)
               for d, r in zip(donor, replayed))


def test_replay_trains_on_the_holdout_rows():
    """The RMSE booster's init is the mean of ITS training targets: a replay
    refit must see the full-data mean, not the 80% split's."""
    X, y = _reg_data()
    off = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                                refit_full=False).fit(X, y)
    rep = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                                refit_full="replay").fit(X, y)
    assert rep.model_.init_ == pytest.approx(float(np.mean(y)))
    assert off.model_.init_ != pytest.approx(float(np.mean(y)))


def test_replay_preserves_the_validation_curve():
    X, y = _reg_data()
    off = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full=False).fit(X, y)
    rep = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                refit_full="replay").fit(X, y)
    assert rep.validation_history_ == off.validation_history_


@pytest.mark.parametrize("k", [2, 3])
def test_replay_fits_and_predicts_classification(k):
    """Binary replays; multiclass deliberately falls back to the from-scratch
    refit (the vector-leaf round has its own loop), so it must still fit."""
    X, y = _clf_data(k=k)
    rep = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                                 refit_full="replay").fit(X, y)
    p = rep.predict_proba(X)
    assert p.shape == (len(X), k)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_multiclass_replay_equals_full_refit():
    """The documented fallback: replay is a no-op for multiclass."""
    X, y = _clf_data(k=3)
    full = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                                  refit_full=True).fit(X, y)
    rep = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                                 refit_full="replay").fit(X, y)
    np.testing.assert_array_equal(full.predict_proba(X), rep.predict_proba(X))


def test_replay_does_not_leak_the_target_through_cat_encoding():
    """Regression test for a real bug: the first replay implementation built
    its training matrix with the donor's ``transform``, i.e. the INFERENCE-time
    target statistics, in which a category's mean includes the label of the row
    being encoded. On a pure-noise 2500-level column that handed the noise 54%
    of the model's importance and cost 6 points of test AUC.

    The fix refits the ordered (non-leaky) statistics on the refit's own rows
    and keeps only the donor's bin borders.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n, n_levels = 5000, 2500
    cat = rng.integers(0, n_levels, n)          # pure noise, unrelated to y
    num = rng.normal(size=(n, 3))
    logit = 1.2 * num[:, 0] - num[:, 1] + rng.normal(0, 1, n)
    y = (logit > np.median(logit)).astype(int)
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = np.array([f"id_{c}" for c in cat], dtype=object)
    X[:, 1:] = num
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1,
                                          stratify=y)
    m = ChimeraBoostClassifier(n_estimators=200, random_state=1,
                               refit_full="replay").fit(Xtr, ytr,
                                                        cat_features=[0])
    tr = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
    te = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    imp = m.feature_importances_
    assert te > 0.85                       # generalizes
    assert tr - te < 0.10                  # not memorizing the noise column
    # The leak showed up here first and most loudly: 54% vs a few percent.
    assert imp[0] / imp.sum() < 0.15


def test_replay_with_categoricals():
    """The donor's preprocessor (encoder + binner) is reused verbatim, because
    splits_thr are bin indices -- a re-fitted binner would move every
    threshold underneath the structures."""
    rng = np.random.default_rng(0)
    n = 3000
    num = rng.standard_normal((n, 3))
    cat = rng.integers(0, 12, size=(n, 1)).astype(float)
    X = np.hstack([num, cat])
    y = num[:, 0] + 0.4 * cat[:, 0] + 0.3 * rng.standard_normal(n)
    rep = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                                refit_full="replay").fit(X, y, cat_features=[3])
    assert np.isfinite(rep.predict(X)).all()


def test_replay_model_does_not_pickle_the_donor():
    """The donor holds a whole forest; the shipped model must not carry it."""
    X, y = _reg_data()
    rep = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                                refit_full="replay").fit(X, y)
    assert rep.model_.replay_donor is None
    back = pickle.loads(pickle.dumps(rep))
    np.testing.assert_array_equal(back.predict(X), rep.predict(X))


def test_replay_survives_get_params_round_trip():
    from sklearn.base import clone
    est = ChimeraBoostRegressor(refit_full="replay")
    assert clone(est).get_params()["refit_full"] == "replay"


def test_bad_refit_full_value_rejected():
    X, y = _reg_data(n=300)
    with pytest.raises(ValueError, match="refit_full"):
        ChimeraBoostRegressor(refit_full="nope").fit(X, y)


def test_replay_kernel_reuses_splits_and_refits_values():
    """Unit-level: same splits out, values driven by the supplied gradients."""
    rng = np.random.default_rng(0)
    Xb = rng.integers(0, 8, size=(3, 500)).astype(np.uint16)
    sf = np.array([0, 2], dtype=np.int64)
    st = np.array([3, 4], dtype=np.int64)
    donor = ObliviousTree(sf, st, np.zeros(4))
    g = rng.standard_normal(500)
    h = np.ones(500)
    tree, leaf = replay_oblivious_tree(donor, Xb, g, h, 1.0, 0.1)
    np.testing.assert_array_equal(tree.splits_feat, sf)
    np.testing.assert_array_equal(tree.splits_thr, st)
    assert leaf.min() >= 0 and leaf.max() < 4
    # Newton value of the leaf the gradients actually landed in.
    for l in range(4):
        m = leaf == l
        if m.sum():
            expect = -0.1 * g[m].sum() / (h[m].sum() + 1.0)
            assert tree.values[l] == pytest.approx(expect)


def test_replay_kernel_handles_a_depth_zero_donor():
    Xb = np.zeros((2, 10), dtype=np.uint16)
    donor = ObliviousTree(np.zeros(0, dtype=np.int64),
                          np.zeros(0, dtype=np.int64), np.zeros(1))
    tree, leaf = replay_oblivious_tree(donor, Xb, np.ones(10), np.ones(10),
                                       1.0, 0.1)
    assert tree.depth == 0
    assert leaf.shape == (10,)
