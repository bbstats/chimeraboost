"""Per-member full-data refit in the bagged path (benchmarks/BREAKTHROUGH_PLAN.md C1).

A bag member trains on ``max_samples`` of the rows and early-stops on its
out-of-bag complement, so ``auto_split`` is False and the ordinary full-data
refit never fires -- every member's leaf values come from 0.8n rows.
``refit_members=True`` replays each member's OWN structure against all-row
gradients, which is safe for the ensemble because bag lift is structural
diversity (the LRE post-mortem), not leaf-value diversity.

These tests lock the contract, not the accuracy: off is byte-identical, on
engages, structures survive, and the paths that must not change do not.
"""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


def _data(n=1500, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.normal(scale=0.4, size=n)
    return X, y


BAG = dict(n_estimators=120, early_stopping_rounds=20, n_ensembles=3,
           ensemble_n_jobs=1, random_state=0)


def test_default_is_off_and_off_is_byte_identical():
    """The default must not change any shipped prediction."""
    X, y = _data()
    Xte = X[:200]
    assert ChimeraBoostRegressor().refit_members is False
    a = ChimeraBoostRegressor(**BAG).fit(X, y)
    b = ChimeraBoostRegressor(refit_members=False, **BAG).fit(X, y)
    assert np.array_equal(a.predict(Xte), b.predict(Xte))


def test_on_changes_predictions_regressor():
    X, y = _data()
    Xte = X[:200]
    a = ChimeraBoostRegressor(**BAG).fit(X, y)
    c = ChimeraBoostRegressor(refit_members=True, **BAG).fit(X, y)
    assert not np.allclose(a.predict(Xte), c.predict(Xte))


def test_on_changes_probabilities_classifier():
    X, y = _data()
    yc = (y > np.median(y)).astype(int)
    Xte = X[:200]
    a = ChimeraBoostClassifier(**BAG).fit(X, yc)
    c = ChimeraBoostClassifier(refit_members=True, **BAG).fit(X, yc)
    assert not np.allclose(a.predict_proba(Xte), c.predict_proba(Xte))


def test_member_structures_are_preserved():
    """Only leaf values may move -- the splits are the diversity the bag
    trades on, so each member must keep the structure its own bag grew."""
    X, y = _data()
    plain = ChimeraBoostRegressor(**BAG).fit(X, y)
    refit = ChimeraBoostRegressor(refit_members=True, **BAG).fit(X, y)
    for mp, mr in zip(plain.estimators_, refit.estimators_):
        tp, tr = mp.model_.trees_, mr.model_.trees_
        # the refit replays every donor tree, then may grow a scaled-up tail
        assert len(tr) >= len(tp)
        for a, b in zip(tp, tr[:len(tp)]):
            assert np.array_equal(a.splits_feat, b.splits_feat)
            assert np.array_equal(a.splits_thr, b.splits_thr)

    sigs = {tuple(tuple(t.splits_feat.tolist()) for t in m.model_.trees_[:3])
            for m in refit.estimators_}
    assert len(sigs) == len(refit.estimators_), "members must stay distinct"


def test_single_model_path_untouched():
    """refit_members is a bagged-path parameter; a single model must ignore it."""
    X, y = _data()
    Xte = X[:200]
    kw = dict(n_estimators=120, early_stopping_rounds=20, random_state=0)
    a = ChimeraBoostRegressor(**kw).fit(X, y)
    b = ChimeraBoostRegressor(refit_members=True, **kw).fit(X, y)
    assert np.array_equal(a.predict(Xte), b.predict(Xte))


def test_explicit_eval_set_disables_member_refit():
    """With a user eval_set the members never carve OOB rows, so there is no
    member-level data tax to reclaim and the arm must stay inert."""
    X, y = _data()
    Xte = X[:200]
    ev = (X[:300], y[:300])
    a = ChimeraBoostRegressor(**BAG).fit(X, y, eval_set=ev)
    b = ChimeraBoostRegressor(refit_members=True, **BAG).fit(X, y, eval_set=ev)
    assert np.array_equal(a.predict(Xte), b.predict(Xte))


def test_multiclass_bag_is_unchanged():
    """Multiclass has no replay path, so a member refit there would cost a
    whole extra fit per member rather than a cheap structure replay. Gated off."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1200, 5))
    y = rng.integers(0, 3, size=1200)
    Xte = X[:200]
    a = ChimeraBoostClassifier(**BAG).fit(X, y)
    b = ChimeraBoostClassifier(refit_members=True, **BAG).fit(X, y)
    assert np.array_equal(a.predict_proba(Xte), b.predict_proba(Xte))


def test_members_fit_on_more_rows_than_their_bag():
    """The point of the arm: leaf values stop being estimated from 0.8n rows.
    Train-set error should improve, since every row now informs some leaf."""
    X, y = _data(n=2000)
    plain = ChimeraBoostRegressor(**BAG).fit(X, y)
    refit = ChimeraBoostRegressor(refit_members=True, **BAG).fit(X, y)
    rp = float(np.sqrt(np.mean((y - plain.predict(X)) ** 2)))
    rr = float(np.sqrt(np.mean((y - refit.predict(X)) ** 2)))
    assert rr < rp


@pytest.mark.parametrize("cls", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_get_params_roundtrip(cls):
    """sklearn clone contract: the parameter must survive get/set_params."""
    est = cls(refit_members=True)
    assert est.get_params()["refit_members"] is True
    from sklearn.base import clone
    assert clone(est).refit_members is True
