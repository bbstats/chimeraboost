"""Vector-leaf multiclass (benchmarks/A1_PLAN.md): kernel oracles, packed
predict bit-identity, legacy (pre-0.25.0) forest fallback, and path sanity."""

import copy

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier
from chimeraboost.booster import MulticlassBoosting
from chimeraboost.tree import (ObliviousTree, _leaf_values, _leaf_values_vec,
                               pack_forest_vec, _predict_forest_vec_rm,
                               _predict_forest_vec_rm_serial)


def _toy_multiclass(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, k - 1))
         .astype(np.int64))
    return X, y


def test_leaf_values_vec_matches_per_class_oracle():
    """Column k of the vector kernel == the scalar `_leaf_values` on
    (grad[:, k], hess[:, k] * coupling) with the shared partition — the
    registered bit-exact equivalence."""
    rng = np.random.default_rng(3)
    n, K, n_leaves = 500, 5, 16
    leaf = rng.integers(0, n_leaves, size=n)
    grad = rng.standard_normal((n, K))
    hess = rng.uniform(1e-6, 0.25, size=(n, K))
    coupling = (K - 1) / K
    l2, lr = 1.0, 0.1
    got = _leaf_values_vec(leaf, grad, hess, coupling, n_leaves, l2, lr)
    for k in range(K):
        want = _leaf_values(leaf, np.ascontiguousarray(grad[:, k]),
                            np.ascontiguousarray(hess[:, k]) * coupling,
                            n_leaves, l2, lr)
        np.testing.assert_array_equal(got[:, k], want)


def test_packed_vector_forest_matches_per_tree_loop():
    """The fused vector predict kernels == init + per-tree values[apply],
    bit for bit, and the serial twin == the parallel kernel."""
    X, y = _toy_multiclass()
    b = MulticlassBoosting(n_estimators=20, random_state=0, depth=4)
    b.fit(X, y)
    Xb_fm = np.ascontiguousarray(b.prep_.transform(X).T)   # feature-major
    F = np.tile(b.init_, (X.shape[0], 1))
    for tree in b.trees_:
        F += tree.values[tree.apply(Xb_fm)]
    Xb_rm = b.prep_.transform(X)                           # row-major
    feats, thrs, depths, vals, voff, K = pack_forest_vec(b.trees_, b.depth)
    par = _predict_forest_vec_rm(Xb_rm, feats, thrs, depths, vals, voff,
                                 K, b.init_)
    ser = _predict_forest_vec_rm_serial(Xb_rm, feats, thrs, depths, vals,
                                        voff, K, b.init_)
    np.testing.assert_array_equal(par, ser)
    np.testing.assert_allclose(par, F, rtol=0, atol=1e-12)
    # And the public path agrees with the packed kernel exactly.
    np.testing.assert_array_equal(b.predict_raw(X), par)


def test_legacy_ktrees_forest_still_predicts():
    """A model whose `trees_` is rounds-of-K per-class trees (any pickle
    from <= 0.24.0) must keep predicting: rebuild that structure from a
    vector model (class k's forest = column k of every vector tree) and
    check the legacy path reproduces the vector path's scores."""
    X, y = _toy_multiclass()
    b = MulticlassBoosting(n_estimators=15, random_state=1, depth=4)
    b.fit(X, y)
    want = b.predict_raw(X)
    legacy = copy.copy(b)
    legacy.trees_ = [
        [ObliviousTree(t.splits_feat, t.splits_thr,
                       np.ascontiguousarray(t.values[:, k]), t.gains)
         for k in range(b.n_classes_)]
        for t in b.trees_]
    legacy._forest_ = None
    legacy._forests_ = None
    got = legacy.predict_raw(X)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_serial_and_parallel_predict_agree_via_public_api():
    """1-row predictions (serial twin) must equal the same rows of a batch
    prediction (parallel kernel)."""
    X, y = _toy_multiclass()
    clf = ChimeraBoostClassifier(n_estimators=25, random_state=0)
    clf.fit(X, y)
    batch = clf.predict_proba(X[:10])
    rows = np.vstack([clf.predict_proba(X[i:i + 1]) for i in range(10)])
    np.testing.assert_array_equal(batch, rows)


def test_vector_leaf_quality_sanity():
    """Separable 4-class problem: the vector-leaf path must actually learn
    (guards against a silently broken sketch or leaf table)."""
    X, y = _toy_multiclass(n=2000, seed=7)
    clf = ChimeraBoostClassifier(n_estimators=150, random_state=0)
    clf.fit(X[:1500], y[:1500])
    acc = float((clf.predict(X[1500:]) == y[1500:]).mean())
    assert acc > 0.8, f"vector-leaf multiclass underfits: acc={acc:.3f}"


def test_ordered_boosting_and_subsample_paths_run():
    """The per-class LOO update and the sketch-derived MVS row selection
    both produce finite, learnable models."""
    X, y = _toy_multiclass(n=800, seed=11)
    for kw in ({"ordered_boosting": True}, {"subsample": 0.7}):
        b = MulticlassBoosting(n_estimators=30, random_state=0, **kw)
        b.fit(X, y)
        raw = b.predict_raw(X)
        assert np.all(np.isfinite(raw))
        acc = float((b.classes_[raw.argmax(axis=1)] == y).mean())
        assert acc > 0.6, f"{kw}: train acc {acc:.3f}"


def test_round_count_and_importances_contract():
    """`trees_` is flat, one tree per round; `best_iteration_` counts rounds;
    `feature_importances_` normalizes over the vector trees."""
    X, y = _toy_multiclass()
    b = MulticlassBoosting(n_estimators=12, random_state=0)
    b.fit(X, y)
    assert not isinstance(b.trees_[0], list)
    assert b.best_iteration_ == len(b.trees_) <= 12
    imp = b.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert imp.sum() == pytest.approx(1.0)
