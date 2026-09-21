"""Prepare-once categorical factorization: one full-matrix pass per fit.

Each leg of a fit (train split, validation split, full-data refit) derives its
categorical factorization from a single parent pass over the full matrix by
gathering and re-numbering in first-appearance order. The oracle throughout is
``factorize`` on the leg's own rows, compared exactly: codes array-equal,
same category count, and pairwise ``==`` with equal hashes (representatives
may differ in type where cross-type-equal values like ``1`` and ``1.0``
coexist -- no dict lookup can see that).
"""

import contextlib
from unittest import mock

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.preprocessing import (
    CatTransformCache,
    _rerank_first_appearance,
)
import chimeraboost.preprocessing as pmod
import chimeraboost.sklearn_api as skmod
from chimeraboost.sklearn_api import _shared_cat_ctxs
from chimeraboost.target_encoding import _factorize_numeric, factorize


N = 400
SEED = 7


def _columns_400():
    """The five adversarial object columns, each (400,) with a fixed seed."""
    rng = np.random.default_rng(SEED)
    cols = {}

    # 1. plain strings with repeats.
    cols["strings"] = rng.choice(
        [f"level-{i}" for i in range(9)], size=N).astype(object)

    # 2. strings with None and NaN mixed in.
    s = rng.choice(["red", "green", "blue", "yellow"], size=N).astype(object)
    miss = rng.choice(N, size=80, replace=False)
    s[miss[:40]] = None
    s[miss[40:]] = float("nan")
    cols["strings_missing"] = s

    # 3. integer-coded categories stored as Python ints in an object array.
    cols["ints"] = rng.integers(0, 12, size=N).astype(object)

    # 4. mixed ints, floats, bools and strings; 1, 1.0 and True all occur
    # (one ==/hash category under factorize's dict).
    pool = [1, 1.0, True, 0, 0.0, False, 2, 2.5, "x", "yy", "1", 3.5, 7]
    idx = rng.integers(0, len(pool), size=N)
    m = np.array([pool[i] for i in idx], dtype=object)
    m[5], m[6], m[7] = 1, 1.0, True
    cols["mixed"] = m

    # 5. all numbers except ONE string at row 0, which lands outside every leg
    # below -- so the parent and each leg take different factorize paths
    # (hashed vs numeric) with different representative types.
    nums = rng.integers(0, 20, size=N).astype(object)
    nums[0] = "the-string"
    assert _factorize_numeric(nums) is None  # premise: parent is not numeric
    cols["one_string_outside_leg"] = nums

    return cols


def _subsets(col):
    """Three row subsets over rows 1..N-1 (row 0 stays outside every leg)."""
    rng = np.random.default_rng(1234)
    rest = np.arange(1, N)
    out = [
        rng.permutation(rest)[: N // 2],  # a random half in random order
        np.sort(rng.choice(rest, size=int(N * 0.8), replace=False)),
    ]
    # A subset missing one category entirely: drop every row holding the first
    # non-missing value (plain != is category-aware here -- NaN/None rows are
    # kept since they compare unequal, cross-type-equals are dropped).
    v = next(x for x in col[1:] if x is not None and x == x)
    out.append(np.array([i for i in rest if col[i] != v], dtype=np.int64))
    return out


def _assert_factorization_equal(got, exp):
    got_codes, got_cats = got
    exp_codes, exp_cats = exp
    assert np.array_equal(got_codes, exp_codes)
    assert len(got_cats) == len(exp_cats)
    for a, b in zip(got_cats, exp_cats):
        assert a == b
        assert hash(a) == hash(b)


@pytest.mark.parametrize("name", list(_columns_400()))
def test_child_column_matches_leg_factorize(name):
    col = _columns_400()[name]
    X = col.reshape(-1, 1)
    for rows in _subsets(col):
        if name == "one_string_outside_leg":
            # Premise: the leg really is all-numeric while the parent is not.
            assert _factorize_numeric(col[rows]) is not None
        parent = CatTransformCache()
        child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
        _assert_factorization_equal(child.column(X[rows], 0),
                                    factorize(X[rows][:, 0]))


def test_child_combo_matches_fresh_cache():
    rng = np.random.default_rng(11)
    a = rng.choice([f"a{i}" for i in range(12)], size=N).astype(object)
    b = rng.choice([f"b{i}" for i in range(7)], size=N).astype(object)
    X = np.column_stack([a, b])
    rows = rng.permutation(N)[:173]
    parent = CatTransformCache()
    child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
    got_codes, got_cats = child.combo(X[rows], 0, 1)
    exp_codes, exp_cats = CatTransformCache().combo(X[rows], 0, 1)
    assert np.array_equal(got_codes, exp_codes)
    assert got_cats == exp_cats


def test_child_guard_falls_back_on_row_mismatch():
    rng = np.random.default_rng(3)
    col = rng.choice(["u", "v", "w"], size=N).astype(object)
    X = col.reshape(-1, 1)
    rows = rng.permutation(N)[:200]
    parent = CatTransformCache()
    child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
    other = rng.choice(["p", "q", "r", "s"], size=150).astype(object)
    other = other.reshape(-1, 1)
    _assert_factorization_equal(child.column(other, 0), factorize(other[:, 0]))


def test_rerank_edge_cases():
    out, keys = _rerank_first_appearance(
        np.array([2, 0, 2, 1], dtype=np.int64), 3)
    assert out.tolist() == [0, 1, 0, 2]
    assert keys.tolist() == [2, 0, 1]
    assert out.dtype == np.int64 and keys.dtype == np.int64

    out, keys = _rerank_first_appearance(np.zeros(0, dtype=np.int64), 4)
    assert out.size == 0 and keys.size == 0
    out, keys = _rerank_first_appearance(np.zeros(0, dtype=np.int64), 0)
    assert out.size == 0 and keys.size == 0
    out, keys = _rerank_first_appearance(np.zeros(4, dtype=np.int64), 0)
    assert out.size == 0 and keys.size == 0


# ---- end to end: sharing is bit-identical and really engages --------------


E2E_CAT = [4, 5, 6]


def _e2e_data(seed=0, n=3000, n_test=1000):
    """Mixed dataset: 4 numerics + 3 string categoricals (~5/40/600 levels)."""
    rng = np.random.default_rng(seed)

    def _make(m):
        Xn = rng.standard_normal((m, 4))
        c0 = np.array([f"g{i}" for i in rng.integers(0, 5, m)], dtype=object)
        c1 = np.array([f"h{i}" for i in rng.integers(0, 40, m)], dtype=object)
        c2 = np.array([f"k{i}" for i in rng.integers(0, 600, m)], dtype=object)
        X = np.empty((m, 7), dtype=object)
        X[:, :4] = Xn
        X[:, 4], X[:, 5], X[:, 6] = c0, c1, c2
        return X, Xn

    Xtr, Xn_tr = _make(n)
    Xte, _ = _make(n_test)
    lvl = np.array([int(v[1:]) for v in Xtr[:, 6]])
    sig = (Xn_tr[:, 0] - 0.7 * Xn_tr[:, 1] + 0.3 * Xn_tr[:, 2]
           + (lvl % 7) * 0.15)
    y_reg = sig + 0.5 * rng.standard_normal(n)
    y_bin = (sig + 0.5 * rng.standard_normal(n) > np.median(sig)).astype(int)
    noisy = sig + 0.5 * rng.standard_normal(n)
    y_mc = np.digitize(noisy, np.quantile(noisy, [1 / 3, 2 / 3]))
    return Xtr, Xte, y_reg, y_bin, y_mc


def _fit_counted(est, X, y, disable_sharing):
    """Fit, counting real ``factorize`` calls; optionally disable sharing."""
    calls = []
    real = pmod.factorize

    def counted(col):
        calls.append(1)
        return real(col)

    patches = [mock.patch.object(pmod, "factorize", counted)]
    if disable_sharing:
        patches.append(mock.patch.object(
            skmod, "_shared_cat_ctxs", lambda *a, **k: (None, None, None)))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        est.fit(X, y, cat_features=E2E_CAT)
    return len(calls)


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
def test_end_to_end_bit_identical(kind):
    Xtr, Xte, y_reg, y_bin, y_mc = _e2e_data()
    if kind == "regression":
        make, y = ChimeraBoostRegressor, y_reg
    else:
        make = ChimeraBoostClassifier
        y = y_bin if kind == "binary" else y_mc

    est_on = make(random_state=0)
    n_on = _fit_counted(est_on, Xtr, y, disable_sharing=False)
    est_off = make(random_state=0)
    n_off = _fit_counted(est_off, Xtr, y, disable_sharing=True)

    assert np.array_equal(est_on.predict(Xte), est_off.predict(Xte))
    if kind != "regression":
        assert np.array_equal(est_on.predict_proba(Xte),
                              est_off.predict_proba(Xte))
    assert est_on.best_iteration_ == est_off.best_iteration_
    # The sharing arm really engaged: strictly fewer factorize passes.
    assert n_on < n_off


# ---- inert without categoricals or without an auto split --------------------


def test_shared_ctxs_inert_directly():
    X = np.zeros((10, 2), dtype=object)
    assert _shared_cat_ctxs(X, None, [0]) == (None, None, None)
    assert _shared_cat_ctxs(X, (np.arange(8), np.arange(8, 10)), []) == (
        None, None, None)
    assert _shared_cat_ctxs(X, None, None) == (None, None, None)


def test_eval_set_and_nocat_fits_build_no_context():
    Xtr, _, y_reg, _, _ = _e2e_data(n=500, n_test=100)
    Xn = np.asarray(Xtr[:, :4], dtype=np.float64)
    seen = []
    real = skmod._shared_cat_ctxs

    def record(*a, **k):
        out = real(*a, **k)
        seen.append(out)
        return out

    with mock.patch.object(skmod, "_shared_cat_ctxs", record):
        # User-supplied eval_set (categoricals present): no auto split.
        ChimeraBoostRegressor(
            random_state=0, n_estimators=10).fit(
            Xtr[:400], y_reg[:400], cat_features=E2E_CAT,
            eval_set=(Xtr[400:], y_reg[400:]))
        # No categoricals at all: the auto split engages, nothing to share.
        ChimeraBoostRegressor(
            random_state=0, n_estimators=10).fit(Xn, y_reg)

    assert seen
    assert all(r == (None, None, None) for r in seen)
