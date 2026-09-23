"""Exactness guards for the fused multiclass eval kernel (S5, I026).

`_softmax_ce_kernel` folds the per-round validation cross-entropy -- softmax,
clip, logs, class sum -- into one numba pass returning the per-row vector,
with the weighted mean left in numpy. The oracle is the OLD body, kept byte
for byte as `MultiSoftmax._eval_numpy`. Both sides run the same numba exp()
(the oracle's softmax dispatches to `_softmax_kernel` for these inputs) on
the same machine, and a 2M-value probe found numba's log() bit-agreeing with
numpy's over the whole clipped [1e-12, 1.0] range on this machine -- but
numba's log() is LLVM libm where numpy's may take a SIMD path that rounds
the last bits differently on other hosts, so kernel-vs-oracle comparisons
hold to 4 ULP per row and 8 ULP on the averaged scalar (the F4 C3
convention), not exact equality. Same-machine bit-identity stays guarded by
benchmarks/identity_snapshot.py.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification

import chimeraboost.losses as losses
from chimeraboost import ChimeraBoostClassifier
from chimeraboost.losses import MultiSoftmax, _SOFTMAX_MAX_K


# Copied from the F4 C3 Logloss guards in tests/test_bitident_refactors.py.
def _assert_ce_equal(a, b):
    # The per-row cross-entropy: the kernel's log() is LLVM libm, the oracle's
    # is numpy's SIMD log(), and those may round the last bits differently on
    # the CI runner pool -- the same 4-ULP caveat as the softmax pins there.
    # Same-machine bit-identity stays guarded by identity_snapshot.py. NaN
    # positions must agree exactly; the finite rest is held to 4 ULP.
    assert a.shape == b.shape
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    assert np.array_equal(nan_a, nan_b)
    if np.any(~nan_a):
        np.testing.assert_array_max_ulp(a[~nan_a], b[~nan_a], maxulp=4)


def _assert_eval_equal(a, b):
    # The averaged scalar: a mean of rows each within 4 ULP is itself within a
    # few ULP; both-NaN counts as equal -- empty and NaN-poisoned inputs
    # evaluate to NaN on both arms.
    if np.isnan(a) or np.isnan(b):
        assert np.isnan(a) and np.isnan(b)
        return
    np.testing.assert_array_max_ulp(np.array([a]), np.array([b]), maxulp=8)


def _one_hot(rng, n, K):
    return np.eye(K)[rng.integers(0, K, n)]


def _extreme_F(rng, n, K):
    # Rows where the clip and the max-shift both bind: whole rows at +/-50,
    # and rows split between +700 and -700 against a normal background.
    F = rng.normal(size=(n, K))
    F[::3] = 50.0
    F[1::3] = -50.0
    F[2::3, 0] = 700.0
    if K > 1:
        F[2::3, 1] = -700.0
    return F


def _row_ce_ref(F, Y):
    # The oracle's per-row vector, transcribed from `_eval_numpy`.
    P = np.clip(losses._softmax(F), 1e-12, 1.0)
    return -np.sum(Y * np.log(P), axis=1), P


def test_eval_matches_oracle_exactly(monkeypatch):
    # K = 2..7, normal and extreme rows, with and without weights. The spy
    # proves the fused kernel -- not the numpy fallback -- produced every
    # dispatch below.
    calls = []
    orig = losses._softmax_ce_kernel

    def spy(F, Y):
        calls.append(F.shape)
        return orig(F, Y)

    monkeypatch.setattr(losses, "_softmax_ce_kernel", spy)
    rng = np.random.default_rng(31)
    n_calls = 0
    for K in range(2, _SOFTMAX_MAX_K + 1):
        loss = MultiSoftmax(K)
        for tag, F in (("gauss", rng.normal(size=(2000, K))),
                       ("extreme", _extreme_F(rng, 2000, K))):
            F = np.ascontiguousarray(F, dtype=np.float64)
            Y = _one_hot(rng, 2000, K)
            ref, P = _row_ce_ref(F, Y)
            _assert_ce_equal(orig(F, Y), ref)
            if tag == "extreme":
                # The clip transcription is only proven where it engages.
                assert np.any(P == 1e-12)
            w = rng.uniform(0.5, 2.0, 2000)
            for weights in (None, w):
                _assert_eval_equal(loss.eval(Y, F, sample_weight=weights),
                    loss._eval_numpy(Y, F, sample_weight=weights))
                n_calls += 1
    assert len(calls) == n_calls


def test_eval_handles_non_contiguous_inputs(monkeypatch):
    # Fortran-order and strided views still take the fused path (copied to
    # C-contiguous first, which changes layout, not values).
    calls = []
    orig = losses._softmax_ce_kernel

    def spy(F, Y):
        calls.append(F.shape)
        return orig(F, Y)

    monkeypatch.setattr(losses, "_softmax_ce_kernel", spy)
    rng = np.random.default_rng(32)
    n, K = 2000, 5
    loss = MultiSoftmax(K)
    F = rng.normal(size=(n, K))
    Y = _one_hot(rng, n, K)
    views = [(np.asfortranarray(F), Y),
             (F, np.asfortranarray(Y)),
             (np.asfortranarray(F), np.asfortranarray(Y)),
             (F[::2], Y[::2])]
    for Fv, Yv in views:
        assert not Fv.flags.c_contiguous or not Yv.flags.c_contiguous
        _assert_eval_equal(loss.eval(Yv, Fv), loss._eval_numpy(Yv, Fv))
    assert len(calls) == len(views)


def test_above_the_guard_uses_numpy_untouched(monkeypatch):
    # K = 8 must never reach the fused kernel. If it does, the test explodes
    # instead of silently comparing two numpy paths.
    def boom(F, Y):
        raise AssertionError("fused kernel must not run for K > 7")

    monkeypatch.setattr(losses, "_softmax_ce_kernel", boom)
    rng = np.random.default_rng(33)
    loss = MultiSoftmax(8)
    F = rng.normal(size=(1500, 8))
    Y = _one_hot(rng, 1500, 8)
    w = rng.uniform(0.5, 2.0, 1500)
    assert loss.eval(Y, F) == loss._eval_numpy(Y, F)
    assert loss.eval(Y, F, sample_weight=w) == \
        loss._eval_numpy(Y, F, sample_weight=w)


def test_float32_falls_back_to_numpy(monkeypatch):
    # Either input in float32 takes the numpy path (same tripwire as above).
    def boom(F, Y):
        raise AssertionError("fused kernel must not run on float32")

    monkeypatch.setattr(losses, "_softmax_ce_kernel", boom)
    rng = np.random.default_rng(34)
    F64 = rng.normal(size=(800, 4))
    Y64 = _one_hot(rng, 800, 4)
    loss = MultiSoftmax(4)
    for F, Y in ((F64.astype(np.float32), Y64),
                 (F64, Y64.astype(np.float32))):
        assert loss.eval(Y, F) == loss._eval_numpy(Y, F)


@pytest.mark.parametrize("n_classes", [3, 6])
def test_eval_refactor_is_end_to_end_identical(monkeypatch, n_classes):
    X, y = make_classification(n_samples=2500, n_features=12, n_informative=8,
                               n_redundant=2, n_classes=n_classes,
                               n_clusters_per_class=1, random_state=0)
    Xtr, ytr = X[:2000], y[:2000]
    Xho, yho = X[2000:], y[2000:]

    def fit():
        m = ChimeraBoostClassifier(random_state=0)
        m.fit(Xtr, ytr, eval_set=(Xho[:250], yho[:250]))
        return m

    kern = fit()
    monkeypatch.setattr(losses.MultiSoftmax, "eval",
                        losses.MultiSoftmax._eval_numpy)
    npy = fit()
    np.testing.assert_array_equal(kern.predict_proba(Xho),
                                  npy.predict_proba(Xho))
    assert kern.best_iteration_ == npy.best_iteration_
    kh, nh = kern.model_.valid_history_, npy.model_.valid_history_
    assert len(kh) > 0 and len(kh) == len(nh)
    for a, b in zip(kh, nh):
        _assert_eval_equal(a, b)
