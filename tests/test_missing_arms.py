"""Contracts for the MISSING_PLAN bake-off arms (benchmarks/missing_arms.py).

What makes the comparison fair: no-NaN data passes through untouched (so the
plain strata tie the baseline exactly), nothing is learned from test rows,
column bookkeeping is right, imputers leave no NaN behind, and the novel
masked imputer actually learns cross-column structure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

import missing_arms as ma  # noqa: E402

FAST = ["ind", "mia", "mean", "missforest", "cbmice", "masked"]
try:
    import miceforest  # noqa: F401
    ALL = FAST + ["miceforest"]
except ImportError:
    ALL = FAST


def _planted(n=1500, rate=0.3, seed=0):
    """Column 2 is a near-exact linear function of columns 0 and 1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    X[:, 2] = 2 * X[:, 0] - X[:, 1] + 0.05 * rng.standard_normal(n)
    full = X.copy()
    M = rng.random(n) < rate
    X[M, 2] = np.nan
    X[rng.random(n) < rate, 3] = np.nan
    return X, full, M


@pytest.mark.parametrize("arm", ALL)
def test_no_nan_is_exact_passthrough(arm):
    X = np.random.default_rng(0).standard_normal((200, 3))
    tf = ma.ARMS[arm]()
    assert tf.fit_transform(X, None) is X
    assert tf.transform(X[:50]) is X[:50] or np.array_equal(tf.transform(X[:50]), X[:50])


@pytest.mark.parametrize("arm", ALL)
def test_shapes_and_no_nan_left(arm):
    X, _, _ = _planted()
    tf = ma.ARMS[arm]()
    out = tf.fit_transform(X[:1000], None)
    te = tf.transform(X[1000:])
    n_extra = {"ind": 2, "mia": 4}.get(arm, 2)
    assert out.shape == (1000, 4 + n_extra)
    assert te.shape == (500, 4 + n_extra)
    # indicators match the original mask
    ind = out[:, 4:6]
    assert np.array_equal(ind.astype(bool), np.isnan(X[:1000, 2:4]))
    if arm not in ("ind", "mia"):
        assert not np.isnan(out[:, :4]).any() and not np.isnan(te[:, :4]).any()
    else:
        # original columns keep their NaN (the booster's top bin still sees it)
        assert np.array_equal(np.isnan(out[:, :4]), np.isnan(X[:1000]))
    if arm == "mia":
        mirror = out[:, 6:8]
        assert not np.isnan(mirror).any()
        assert (mirror[np.isnan(X[:1000, 2]), 0]
                < np.nanmin(X[:1000, 2])).all()


@pytest.mark.parametrize("arm", [a for a in ALL if a != "miceforest"])
def test_test_rows_are_independent(arm):
    """Transforming a test row gives the same answer whatever rows accompany
    it -- nothing is learned from the test set."""
    X, _, _ = _planted()
    tf = ma.ARMS[arm]()
    tf.fit_transform(X[:1000], None)
    a = tf.transform(X[1000:])
    b = tf.transform(X[1000:1100])
    assert np.allclose(a[:100], b, equal_nan=True)


def test_categoricals_pass_through():
    X, _, _ = _planted(n=400)
    Xo = X.astype(object)
    Xo[:, 1] = "k"
    tf = ma.ARMS["mean"]()
    out = tf.fit_transform(Xo, [1])
    assert (out[:, 1] == "k").all()
    assert out.shape[1] == 4 + 2


@pytest.mark.parametrize("arm", [a for a in ALL if a not in ("ind", "mia", "mean")])
def test_model_imputers_beat_mean_on_planted_relation(arm):
    X, full, M = _planted()
    tr = slice(0, 1000)
    mean = ma.ARMS["mean"]().fit_transform(X[tr], None)[:, 2]
    tf = ma.ARMS[arm]()
    got = tf.fit_transform(X[tr], None)[:, 2]
    m = M[tr]
    rmse = lambda v: np.sqrt(np.mean((v[m] - full[tr][m, 2]) ** 2))  # noqa: E731
    assert rmse(got) < 0.5 * rmse(mean), (rmse(got), rmse(mean))
    # Test rows too -- the miceforest new-data path once returned its random
    # initial fill here while the training fill looked fine.
    te = slice(1000, None)
    got_te = tf.transform(X[te])[:, 2]
    mean_te = np.where(np.isnan(X[te, 2]), np.nanmean(X[tr, 2]), X[te, 2])
    mt = M[te]
    err = np.sqrt(np.mean((got_te[mt] - full[te][mt, 2]) ** 2))
    err_mean = np.sqrt(np.mean((mean_te[mt] - full[te][mt, 2]) ** 2))
    assert err < 0.5 * err_mean, (err, err_mean)
