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


# ---- numeric block: cast once per fit, gather per leg -----------------------


def _numeric_400():
    """Object matrix (400, 5): 3 mixed-type numeric cols + 2 string cat cols.

    Each numeric column mixes Python floats, Python ints, numpy floats, bools
    and NaN, so the element-wise object->float64 cast sees every spelling.
    """
    rng = np.random.default_rng(20260921)
    X = np.empty((N, 5), dtype=object)
    for j in range(3):
        col = np.empty(N, dtype=object)
        for i in range(N):
            r = rng.random()
            if r < 0.35:
                col[i] = float(rng.standard_normal())  # Python float
            elif r < 0.55:
                col[i] = int(rng.integers(-1000, 1000))  # Python int
            elif r < 0.70:
                col[i] = np.float64(rng.standard_normal())  # numpy float
            elif r < 0.85:
                col[i] = bool(rng.integers(0, 2))  # bool
            else:
                col[i] = float("nan")
        X[:, j] = col
    X[:, 3] = rng.choice([f"c{i}" for i in range(11)], size=N)
    X[:, 4] = rng.choice([f"d{i}" for i in range(23)], size=N)
    return X


_NUM_400 = [0, 1, 2]


def _assert_block_equal(got, exp):
    assert got.shape == exp.shape
    assert got.dtype == np.float64 and exp.dtype == np.float64
    assert np.array_equal(got, exp, equal_nan=True)
    assert got.tobytes() == exp.tobytes()


def test_child_numeric_matches_leg_cast():
    X = _numeric_400()
    for rows in _subsets(X[:, 3]):
        parent = CatTransformCache()
        child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
        _assert_block_equal(child.numeric(X[rows], _NUM_400),
                            pmod._cast_numeric_block(X[rows], _NUM_400))


def test_child_numeric_all_positions_and_empty():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((N, 4))
    X[::17, 0] = np.nan
    rows = np.sort(rng.choice(N, size=251, replace=False))
    parent = CatTransformCache()
    child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
    full = list(range(X.shape[1]))
    _assert_block_equal(child.numeric(X[rows], full),
                        pmod._cast_numeric_block(X[rows], full))

    Xo = _numeric_400()
    rows_o = np.sort(rng.choice(N, size=251, replace=False))
    child_o = CatTransformCache(parent=CatTransformCache(), parent_X=Xo,
                                rows=rows_o)
    got = child_o.numeric(Xo[rows_o], [])
    assert got.shape == (len(rows_o), 0)
    assert got.dtype == np.float64
    _assert_block_equal(got, pmod._cast_numeric_block(Xo[rows_o], []))


def test_child_numeric_cached_parent_casts_once():
    X = _numeric_400()
    rows = _subsets(X[:, 3])[0]
    real = pmod._cast_numeric_block
    calls = []

    def counted(X_, num_features):
        calls.append(1)
        return real(X_, num_features)

    with mock.patch.object(pmod, "_cast_numeric_block", counted):
        parent = CatTransformCache()
        child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
        first = child.numeric(X[rows], _NUM_400)
        second = child.numeric(X[rows], _NUM_400)
    assert first is second
    assert len(calls) == 1


def test_child_numeric_guard_falls_back_on_row_mismatch():
    X = _numeric_400()
    rng = np.random.default_rng(5)
    rows = rng.permutation(N)[:200]
    parent = CatTransformCache()
    child = CatTransformCache(parent=parent, parent_X=X, rows=rows)
    other = np.empty((150, 5), dtype=object)
    other[:, :3] = rng.standard_normal((150, 3))
    other[:, 3] = "z"
    other[:, 4] = "w"
    _assert_block_equal(child.numeric(other, _NUM_400),
                        pmod._cast_numeric_block(other, _NUM_400))


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_full_ctx_numeric_block_unmodified_by_fit(kind):
    Xtr, _, y_reg, y_bin, _ = _e2e_data()
    if kind == "regression":
        make, y = ChimeraBoostRegressor, y_reg
    else:
        make, y = ChimeraBoostClassifier, y_bin
    seen = []
    real = skmod._shared_cat_ctxs

    def record(*a, **k):
        out = real(*a, **k)
        seen.append((a, out))
        return out

    with mock.patch.object(skmod, "_shared_cat_ctxs", record):
        make(random_state=0).fit(Xtr, y, cat_features=E2E_CAT)

    assert seen
    num = [f for f in range(Xtr.shape[1]) if f not in E2E_CAT]
    checked = 0
    for a, out in seen:
        full_ctx = out[0]
        if full_ctx is None:
            continue
        key = tuple(num)
        assert key in full_ctx._numeric
        cached = full_ctx._numeric[key]
        fresh = pmod._cast_numeric_block(a[0], num)
        assert cached.shape == fresh.shape
        assert cached.tobytes() == fresh.tobytes()
        checked += 1
    assert checked


def _fit_counted_numeric(est, X, y, disable_sharing):
    """Fit, counting real ``_cast_numeric_block`` calls; optionally no sharing."""
    calls = []
    real = pmod._cast_numeric_block

    def counted(X_, num_features):
        calls.append(1)
        return real(X_, num_features)

    patches = [mock.patch.object(pmod, "_cast_numeric_block", counted)]
    if disable_sharing:
        patches.append(mock.patch.object(
            skmod, "_shared_cat_ctxs", lambda *a, **k: (None, None, None)))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        est.fit(X, y, cat_features=E2E_CAT)
    return len(calls)


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
def test_end_to_end_numeric_sharing_fewer_casts_bit_identical(kind):
    Xtr, Xte, y_reg, y_bin, y_mc = _e2e_data()
    if kind == "regression":
        make, y = ChimeraBoostRegressor, y_reg
    else:
        make = ChimeraBoostClassifier
        y = y_bin if kind == "binary" else y_mc

    est_on = make(random_state=0)
    n_on = _fit_counted_numeric(est_on, Xtr, y, disable_sharing=False)
    est_off = make(random_state=0)
    n_off = _fit_counted_numeric(est_off, Xtr, y, disable_sharing=True)

    assert np.array_equal(est_on.predict(Xte), est_off.predict(Xte))
    if kind != "regression":
        assert np.array_equal(est_on.predict_proba(Xte),
                              est_off.predict_proba(Xte))
    # The sharing arm really engaged: strictly fewer numeric casts.
    assert n_on < n_off
