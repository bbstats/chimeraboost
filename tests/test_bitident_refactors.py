"""Guards for the output-identical performance refactors (2026-07-30 pass).

Each guard pins the invariant a refactor leans on, so a later change that
breaks the invariant fails here instead of silently changing models.
"""
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.losses import (MAE, RMSE, Huber, MultiQuantile, MultiSoftmax,
                                 Quantile, _SOFTMAX_MAX_K, _softmax,
                                 _softmax_numpy)
from chimeraboost.preprocessing import _factorize_int
from chimeraboost.target_encoding import (_factorize_hashed,
                                          _factorize_numeric, factorize)
from chimeraboost.tree import _SMALL_N


def _reg_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X[:, 0] * 2 + np.sin(X[:, 1] * 3) + rng.normal(scale=0.3, size=n)
    w = rng.uniform(0.5, 2.0, n)
    return X, y, w


# ---------------------------------------------------------------------------
# _UnitHessian cache: shared across rounds, so it must stay all-ones through
# a full fit, and it must not ride along in pickles.
# ---------------------------------------------------------------------------

def test_unit_hessian_cache_is_never_mutated():
    # Weighted + subsampled hits every path that multiplies the hessian
    # (w-weighting, MVS): each must produce a fresh array, not write into
    # the shared cache.
    X, y, w = _reg_data()
    est = ChimeraBoostRegressor(n_estimators=60, subsample=0.7,
                                random_state=0, loss="MAE")
    est.fit(X, y, sample_weight=w)
    cache = est.model_.loss_._hess_cache
    assert cache is not None
    assert np.all(cache == 1.0)


def test_unit_hessian_cache_not_pickled():
    X, y, _ = _reg_data()
    est = ChimeraBoostRegressor(n_estimators=30, random_state=0)
    est.fit(X, y)
    assert est.model_.loss_._hess_cache is not None
    state = pickle.loads(pickle.dumps(est)).model_.loss_.__dict__
    assert "_hess_cache" not in state


def test_unit_hessian_values_and_reuse():
    for loss in (RMSE(), MAE(), Quantile(0.3), Huber(),
                 MultiQuantile(np.array([0.25, 0.75]))):
        raw = (np.zeros((50, 2)) if isinstance(loss, MultiQuantile)
               else np.zeros(50))
        y = np.ones(raw.shape[0])
        _, h1 = loss.grad_hess(y, raw)
        _, h2 = loss.grad_hess(y, raw)
        assert np.array_equal(h1, np.ones_like(raw))
        assert h1 is h2                      # cached, not reallocated
        _, h3 = loss.grad_hess(y[:20], raw[:20])   # shape change -> fresh
        assert h3.shape == raw[:20].shape


# ---------------------------------------------------------------------------
# Scalar _correct_leaves: the compiled K=1 kernel dispatch must reproduce the
# old argsort + per-leaf leaf_value() path exactly. The reference below IS
# that old path, verbatim.
# ---------------------------------------------------------------------------

def _correct_leaves_reference(loss, lr, n_leaves, leaf, y, F, sample_weight):
    residuals = y - F
    values = np.zeros(n_leaves)
    order = np.argsort(leaf, kind="stable")
    counts = np.bincount(leaf, minlength=n_leaves)
    stop = np.cumsum(counts)
    r_sorted = residuals[order]
    w_sorted = sample_weight[order] if sample_weight is not None else None
    for l in range(n_leaves):
        lo, hi = stop[l] - counts[l], stop[l]
        w = w_sorted[lo:hi] if w_sorted is not None else None
        values[l] = lr * loss.leaf_value(r_sorted[lo:hi], w)
    return values


@pytest.mark.parametrize("loss", [MAE(), Quantile(0.3), Quantile(0.9)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n", [200, _SMALL_N * 2])   # serial and parallel
def test_scalar_correct_leaves_matches_reference(loss, weighted, n):
    rng = np.random.default_rng(42)
    n_leaves = 16
    # Leave two leaves empty, and quantize y so tied residuals occur.
    leaf = rng.integers(0, n_leaves - 2, n).astype(np.int64)
    y = np.round(rng.normal(size=n) * 8) / 8
    F = np.round(rng.normal(size=n) * 4) / 4
    w = rng.uniform(0.0, 2.0, n) if weighted else None

    gb = GradientBoosting()
    gb.loss_, gb.lr_ = loss, 0.07
    tree = SimpleNamespace(values=np.zeros(n_leaves))
    gb._correct_leaves(tree, leaf, y, F, w)

    ref = _correct_leaves_reference(loss, 0.07, n_leaves, leaf, y, F, w)
    np.testing.assert_array_equal(tree.values, ref)


# ---------------------------------------------------------------------------
# factorize: both fast paths must group and order exactly like the dict loop.
# The numeric one must REFUSE anything the audit can't prove numeric (numeric-
# looking strings are the killer: astype parses "1.5" happily); the hashed one
# keeps the dict itself, so it need only refuse what its missing-value mask
# cannot see. The reference below is the old loop, verbatim.
# ---------------------------------------------------------------------------

def _factorize_reference(column):
    col = np.asarray(column, dtype=object)
    codes = np.empty(col.shape[0], dtype=np.int64)
    mapping = {}
    cats = []
    for i, v in enumerate(col.tolist()):
        if v is None:
            v = "__nan__"
        else:
            try:
                if v != v:
                    v = "__nan__"
            except (TypeError, ValueError):
                v = "__nan__"
        code = mapping.get(v)
        if code is None:
            code = mapping[v] = len(cats)
            cats.append(v)
        codes[i] = code
    return codes, np.asarray(cats, dtype=object)


class _RaisesOnCompare:
    """The loop's third missing-value class: `!=` raises, so the loop calls it
    missing. No vectorized mask can see that, so the fast path must refuse."""

    def __eq__(self, other):
        raise ValueError("no comparison here")

    def __ne__(self, other):
        raise ValueError("no comparison here")

    def __hash__(self):
        return 0


FACTORIZE_CASES = [
    [3.5, 1.0, 3.5, np.nan, 2.0, None, 1.0],
    [1, 2, 3, 1, 2],
    [True, False, 1, 0, 1.0],           # bool/int/float share ==/hash classes
    [np.nan, np.nan],                   # all missing
    [],                                 # empty
    [7.25],                             # single value
    [0.0, -0.0, 0.0],                   # signed zeros are one category
    [np.inf, -np.inf, np.inf, None],
    [np.float32(0.5), 0.5, np.int64(7), 7],
    ["a", "b", "a", None],              # strings -> general loop
    ["1.5", "2", "1.5"],                # numeric strings must NOT parse
    [1.5, "1.5"],                       # ...and stay distinct from the float
    ["nan", "nan"],                     # the string "nan" is a real category
    ["__nan__", None],                  # sentinel-collision quirk preserved
    [2 ** 53 + 1, 2 ** 53 + 2],         # past float53 -> must not merge
    ["x", np.nan, "y", "x", None],      # strings + both flavours of missing
    ["b", "a", "b", "c", "a"],          # first-appearance order != sorted order
    ["", "", "a"],                      # the empty string is a category
    ["a\x00", "a"],                     # a NUL is part of the string, not padding
    ["e", "é", "e"],               # non-ASCII stays distinct
    [np.str_("a"), "a"],                # str subclass: one ==/hash class
    ["a", 1.5, "a"],                    # mixed types keep the dict's classes
    ["a", b"a"],                        # bytes never equal str
    ["a", True, 1, "a"],                # bool/int merge, string does not
    [_RaisesOnCompare(), "a"],          # raises on != -> counts as missing
]


@pytest.mark.parametrize("case", FACTORIZE_CASES,
                         ids=[str(i) for i in range(len(FACTORIZE_CASES))])
def test_factorize_matches_reference(case):
    codes, cats = factorize(case)
    ref_codes, ref_cats = _factorize_reference(case)
    np.testing.assert_array_equal(codes, ref_codes)
    assert codes.dtype == np.int64
    assert cats.dtype == object
    assert len(cats) == len(ref_cats)
    # Representatives may be the float image of the original (7 -> 7.0);
    # they must stay ==/hash-interchangeable, which is what downstream
    # category maps key on.
    for a, b in zip(cats.tolist(), ref_cats.tolist()):
        assert a == b and hash(a) == hash(b)


def test_factorize_fast_path_refuses_non_numeric():
    for case in (["a", "b"], ["1.5", "2"], ["nan"], [1.5, "1.5"],
                 ["__nan__", None], [2 ** 53 + 1, 2 ** 53 + 2], [b"x"]):
        col = np.asarray(case, dtype=object)
        assert _factorize_numeric(col) is None, case


def test_factorize_hashed_refuses_what_it_cannot_vectorize():
    # The mask is the only non-structural step: an element that raises on `!=`
    # is missing to the loop and invisible to any mask, and an unhashable one
    # must reach the loop so it raises there, as it always did.
    for case in ([_RaisesOnCompare()], ["a", _RaisesOnCompare()],
                 [["unhashable"], "a"]):
        col = np.asarray(case, dtype=object)
        assert _factorize_hashed(col) is None, case


def test_factorize_hashed_engages_on_the_case_it_exists_for():
    # String columns are why this path exists (18% of the hc:okcupid-stem fit);
    # they must not reach the loop, with or without missing values.
    for case in (["a", "b", "a"], ["x", None, "y"], ["x", np.nan],
                 ["", "a"], ["__nan__", None], ["1.5", "2"], ["a", 1.5]):
        col = np.asarray(case, dtype=object)
        assert _factorize_hashed(col) is not None, case


def test_factorize_int_matches_reference():
    rng = np.random.default_rng(0)
    for vals in (rng.integers(-5, 20, 300).astype(np.int64),
                 np.array([7], dtype=np.int64),
                 np.empty(0, dtype=np.int64)):
        codes, keys = _factorize_int(vals)
        mapping, ref_keys = {}, []
        ref_codes = np.empty(vals.size, dtype=np.int64)
        for i, v in enumerate(vals.tolist()):
            c = mapping.get(v)
            if c is None:
                c = mapping[v] = len(ref_keys)
                ref_keys.append(v)
            ref_codes[i] = c
        np.testing.assert_array_equal(codes, ref_codes)
        assert codes.dtype == np.int64
        assert keys == ref_keys
        assert all(type(k) is int for k in keys)


def test_apply_parallel_arm_matches_serial():
    # The parallel descend only fires above _ASSIGN_PAR_N rows -- too big for
    # the identity snapshot's eval sets -- so pin the equivalence directly.
    from chimeraboost.tree import ObliviousTree, _ASSIGN_PAR_N, _assign_leaves
    rng = np.random.default_rng(7)
    n = _ASSIGN_PAR_N + 5
    Xb = np.ascontiguousarray(rng.integers(0, 64, size=(5, n)),
                              dtype=np.uint16)
    sf = np.array([0, 3, 1, 4], dtype=np.int64)
    st = np.array([10, 30, 5, 50], dtype=np.int64)
    vals = rng.normal(size=16)
    tree = ObliviousTree(sf, st, vals)
    ref = _assign_leaves(Xb, sf, st)
    np.testing.assert_array_equal(tree.apply(Xb), ref)
    np.testing.assert_array_equal(tree.predict(Xb), vals[ref])
    # Below the gate both routes are the same serial kernel already.
    np.testing.assert_array_equal(tree.apply(Xb[:, :100]), ref[:100])


def test_custom_adjusts_leaves_loss_keeps_generic_path():
    # A user loss subclassing Quantile with its own leaf_value must NOT be
    # captured by the exact-type kernel dispatch.
    class Midhinge(Quantile):
        def leaf_value(self, residuals, weights=None):
            if not residuals.size:
                return 0.0
            return float(np.quantile(residuals, 0.25)
                         + np.quantile(residuals, 0.75)) / 2.0

    rng = np.random.default_rng(3)
    n, n_leaves = 500, 8
    leaf = rng.integers(0, n_leaves, n).astype(np.int64)
    y, F = rng.normal(size=n), rng.normal(size=n)
    loss = Midhinge(0.5)
    gb = GradientBoosting()
    gb.loss_, gb.lr_ = loss, 0.1
    tree = SimpleNamespace(values=np.zeros(n_leaves))
    gb._correct_leaves(tree, leaf, y, F, None)
    ref = _correct_leaves_reference(loss, 0.1, n_leaves, leaf, y, F, None)
    np.testing.assert_array_equal(tree.values, ref)


# ---------------------------------------------------------------------------
# _softmax fused kernel (F4 C1): matches the numpy path it replaced, for every
# class count it is allowed to run on. The oracle is the OLD code
# (`_softmax_numpy`), not a re-derivation of what softmax ought to be.
#
# The bound is 4 ULP, not exact equality: numba's exp() (LLVM libm) and
# numpy's exp() may round the last bits differently depending on the host
# CPU's SIMD path. Exact cross-implementation equality held for a month of
# CI only because the runner pool was homogeneous; on 2026-08-30 ubuntu
# runners started serving hardware where ~2% of elements differ (up to
# 3 ULP observed at the near-overflow scale). A genuine algorithmic
# regression diverges by orders of magnitude more, so the guard keeps its
# power. Same-machine bit-identity -- the property the F4 refactor actually
# leaned on -- is still guarded exactly, by benchmarks/identity_snapshot.py
# and the multiclass goldens.
# ---------------------------------------------------------------------------

def test_softmax_kernel_matches_numpy_to_a_few_ulp():
    rng = np.random.default_rng(0)
    for K in range(2, _SOFTMAX_MAX_K + 1):
        for scale in (1e-3, 1.0, 30.0):    # tiny, ordinary and near-overflow
            F = rng.normal(scale=scale, size=(4000, K))
            np.testing.assert_array_max_ulp(_softmax(F), _softmax_numpy(F),
                                            maxulp=4)


def test_softmax_above_the_guard_uses_numpy_untouched():
    # K >= 8 is where a row-at-a-time sum stops matching numpy's pairwise
    # blocking, so the kernel must not run there at all -- exercised through
    # the public entry point, since that dispatch IS the guarantee.
    rng = np.random.default_rng(1)
    for K in (_SOFTMAX_MAX_K + 1, 12, 26):
        F = rng.normal(size=(2000, K))
        np.testing.assert_array_equal(_softmax(F), _softmax_numpy(F))


def test_softmax_kernel_matches_numpy_on_degenerate_rows():
    # Constant rows, huge negatives (every exp underflows but the max),
    # duplicate maxima, and a single-column matrix.
    cases = [np.zeros((5, 3)),
             np.full((4, 3), -1e5),
             np.array([[800.0, 800.0, -800.0], [-1e300, 1e-300, 0.0]]),
             np.array([[1.0], [2.0]]),
             np.repeat(np.array([[3.0, 3.0, 3.0]]), 7, axis=0)]
    for F in cases:
        np.testing.assert_array_equal(_softmax(F), _softmax_numpy(F))


def test_softmax_non_float64_falls_back_to_numpy():
    # The kernel writes a float64 out-array; anything else must take the
    # numpy path rather than be silently upcast.
    F = np.random.default_rng(2).normal(size=(64, 3)).astype(np.float32)
    out = _softmax(F)
    np.testing.assert_array_equal(out, _softmax_numpy(F))
    assert out.dtype == np.float32


def test_softmax_rows_sum_to_one_and_probabilities_are_valid():
    F = np.random.default_rng(3).normal(scale=5.0, size=(1000, 5))
    P = _softmax(F)
    assert np.all((P >= 0.0) & (P <= 1.0))
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_multiclass_grad_hess_and_eval_go_through_the_kernel():
    # The callers, not just the helper: grad_hess is 40% of a multiclass fit
    # and eval another 5%, so both are pinned against the numpy oracle --
    # within the same cross-libm tolerance as the kernel test above (grad and
    # hess inherit the kernel's last-bit exp() variation, and the derived
    # arithmetic can compound it, so the bounds are a few ULP of the
    # probabilities involved, not exact equality).
    rng = np.random.default_rng(4)
    n, K = 800, 4
    F = rng.normal(size=(n, K))
    Y = np.eye(K)[rng.integers(0, K, n)]
    loss = MultiSoftmax(K)
    P = _softmax_numpy(F)
    grad, hess = loss.grad_hess(Y, F)
    np.testing.assert_allclose(grad, P - Y, rtol=0, atol=2e-15)
    np.testing.assert_allclose(hess, np.maximum(P * (1.0 - P), 1e-6),
                               rtol=2e-15, atol=0)
    ref_eval = float(np.average(-np.sum(
        Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))
    assert abs(loss.eval(Y, F) - ref_eval) <= 1e-14
    np.testing.assert_array_max_ulp(loss.transform(F), P, maxulp=4)


# ---------------------------------------------------------------------------
# _softmax_grad_hess fused kernel (F4 C1b): the softmax row plus the grad and
# hess elementwise passes in one numba kernel. The oracle is the OLD code
# (`MultiSoftmax._grad_hess_numpy`), kept byte for byte -- and since both
# sides run the same numba exp() on the same machine, the comparison is EXACT
# equality, unlike the cross-libm softmax tests above.
# ---------------------------------------------------------------------------

def _one_hot(rng, n, K):
    return np.eye(K)[rng.integers(0, K, n)]


def test_fused_grad_hess_matches_oracle_exactly(monkeypatch):
    # K = 2..7 at tiny, ordinary and near-overflow scales. The spy proves the
    # fused kernel -- not the numpy fallback -- produced every one of these.
    import chimeraboost.losses as losses

    calls = []
    orig = losses._softmax_grad_hess_kernel

    def spy(F, Y):
        calls.append(F.shape)
        return orig(F, Y)

    monkeypatch.setattr(losses, "_softmax_grad_hess_kernel", spy)
    rng = np.random.default_rng(11)
    for K in range(2, _SOFTMAX_MAX_K + 1):
        loss = MultiSoftmax(K)
        for scale in (1e-3, 1.0, 30.0):
            F = rng.normal(scale=scale, size=(4000, K))
            Y = _one_hot(rng, 4000, K)
            grad, hess = loss.grad_hess(Y, F)
            egrad, ehess = loss._grad_hess_numpy(Y, F)
            np.testing.assert_array_equal(grad, egrad)
            np.testing.assert_array_equal(hess, ehess)
    assert len(calls) == len(range(2, _SOFTMAX_MAX_K + 1)) * 3


def test_fused_grad_hess_matches_oracle_on_degenerate_rows():
    # Constant rows, huge negatives, duplicate maxima, a single column --
    # each with an all-zero Y and a one-hot Y.
    cases = [np.zeros((5, 3)),
             np.full((4, 3), -1e5),
             np.array([[800.0, 800.0, -800.0], [-1e300, 1e-300, 0.0]]),
             np.array([[1.0], [2.0]]),
             np.repeat(np.array([[3.0, 3.0, 3.0]]), 7, axis=0)]
    rng = np.random.default_rng(12)
    for F in cases:
        n, K = F.shape
        loss = MultiSoftmax(K)
        for Y in (np.zeros((n, K)), _one_hot(rng, n, K)):
            grad, hess = loss.grad_hess(Y, F)
            egrad, ehess = loss._grad_hess_numpy(Y, F)
            np.testing.assert_array_equal(grad, egrad)
            np.testing.assert_array_equal(hess, ehess)


def test_fused_grad_hess_handles_non_contiguous_inputs():
    # Fortran-order and strided views still take the fused path (copied to
    # C-contiguous first, which changes layout, not values).
    rng = np.random.default_rng(13)
    n, K = 3000, 5
    loss = MultiSoftmax(K)
    F = rng.normal(size=(n, K))
    Y = _one_hot(rng, n, K)
    views = [(np.asfortranarray(F), Y),
             (F, np.asfortranarray(Y)),
             (np.asfortranarray(F), np.asfortranarray(Y)),
             (F[::2], Y[::2])]
    for Fv, Yv in views:
        assert not Fv.flags.c_contiguous or not Yv.flags.c_contiguous
        grad, hess = loss.grad_hess(Yv, Fv)
        egrad, ehess = loss._grad_hess_numpy(Yv, Fv)
        np.testing.assert_array_equal(grad, egrad)
        np.testing.assert_array_equal(hess, ehess)


def test_fused_grad_hess_above_the_guard_uses_numpy_untouched(monkeypatch):
    # K >= 8 must never reach the fused kernel. If it does, the test explodes
    # instead of silently comparing two numpy paths.
    import chimeraboost.losses as losses

    def boom(F, Y):
        raise AssertionError("fused kernel must not run for K > 7")

    monkeypatch.setattr(losses, "_softmax_grad_hess_kernel", boom)
    rng = np.random.default_rng(14)
    for K in (_SOFTMAX_MAX_K + 1, 12):
        loss = MultiSoftmax(K)
        F = rng.normal(size=(2000, K))
        Y = _one_hot(rng, 2000, K)
        grad, hess = loss.grad_hess(Y, F)
        egrad, ehess = loss._grad_hess_numpy(Y, F)
        np.testing.assert_array_equal(grad, egrad)
        np.testing.assert_array_equal(hess, ehess)


def test_fused_grad_hess_float32_falls_back_to_numpy(monkeypatch):
    # Either input in float32 takes the numpy path (same tripwire as above).
    import chimeraboost.losses as losses

    def boom(F, Y):
        raise AssertionError("fused kernel must not run on float32")

    monkeypatch.setattr(losses, "_softmax_grad_hess_kernel", boom)
    rng = np.random.default_rng(15)
    F64 = rng.normal(size=(500, 4))
    Y64 = _one_hot(rng, 500, 4)
    loss = MultiSoftmax(4)
    for F, Y in ((F64.astype(np.float32), Y64),
                 (F64, Y64.astype(np.float32))):
        grad, hess = loss.grad_hess(Y, F)
        egrad, ehess = loss._grad_hess_numpy(Y, F)
        np.testing.assert_array_equal(grad, egrad)
        np.testing.assert_array_equal(hess, ehess)


def test_fused_grad_hess_invariants():
    rng = np.random.default_rng(16)
    loss = MultiSoftmax(5)
    F = rng.normal(scale=5.0, size=(1000, 5))
    Y = _one_hot(rng, 1000, 5)
    grad, hess = loss.grad_hess(Y, F)
    assert np.all(hess >= 1e-6)
    np.testing.assert_allclose(grad.sum(axis=1), 0.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Complexity-refactor guards (stage 0 of the C901 program). The identity
# snapshot pins model outputs exactly, but two behavior surfaces are invisible
# to it: which file/line a warning is attributed to (an extracted helper adds
# a stack frame; without a stacklevel bump the warning moves), and the exact
# text plus first-fire order of validation errors. Pin both here.
# ---------------------------------------------------------------------------

def _recorded(warns, match):
    return [w for w in warns if match in str(w.message)]


def test_inert_knob_warnings_keep_their_attribution_and_order():
    # linear_leaves=True with >= LINEAR_LEAVES_MIN_SAMPLES post-split rows
    # shadows ordered boosting and leaf refinement; both warnings must keep
    # firing, in this order, attributed to sklearn_api.py (stacklevel must be
    # bumped +1 for every frame an extraction adds).
    X, y, _ = _reg_data(n=2000, seed=5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ChimeraBoostRegressor(
            n_estimators=15, random_state=0, linear_leaves=True,
            ordered_boosting=True, leaf_estimation_iterations=3).fit(X, y)
    ob = _recorded(rec, "ordered_boosting is ignored")
    lei = _recorded(rec, "leaf_estimation_iterations is ignored")
    assert len(ob) == 1 and len(lei) == 1
    assert ob[0].category is UserWarning
    assert ob[0].filename.endswith("sklearn_api.py")
    assert lei[0].filename.endswith("sklearn_api.py")
    assert rec.index(ob[0]) < rec.index(lei[0])


def test_multiclass_lei_warning_keeps_its_attribution():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 4))
    y = rng.integers(0, 3, 300)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ChimeraBoostClassifier(n_estimators=10, random_state=0,
                               leaf_estimation_iterations=3).fit(X, y)
    w = _recorded(rec, "not implemented for multiclass")
    assert len(w) == 1
    assert w[0].category is UserWarning
    assert w[0].filename.endswith("sklearn_api.py")


def test_column_vector_y_warning_keeps_its_attribution():
    X, y, _ = _reg_data(n=200, seed=7)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ChimeraBoostRegressor(n_estimators=10, random_state=0,
                              early_stopping=False).fit(X, y.reshape(-1, 1))
    w = _recorded(rec, "A column-vector y was passed")
    assert len(w) == 1
    assert w[0].filename.endswith("sklearn_api.py")


# The refactor contract for the validators: byte-identical messages and
# unchanged first-fire order. Regex matching would let a reworded message
# slide through, so compare the full string.
@pytest.mark.parametrize("params, expected", [
    ({"n_estimators": 0},
     "n_estimators must be an integer >= 1; got 0."),
    ({"depth": 30},
     "depth must be an integer in [1, 16] or None; got 30."),
    ({"learning_rate": -0.1},
     "learning_rate must be in (0.0, inf]; got -0.1."),
    ({"subsample": 0.0},
     "subsample must be in (0.0, 1.0]; got 0.0."),
    ({"loss": "LogCosh"},
     "loss must be one of ('RMSE', 'MAE', 'Quantile', 'Huber', 'Poisson', "
     "'Gamma', 'Tweedie') or a custom objective instance; got 'LogCosh'."),
    ({"refit_full": "yes"},
     'refit_full must be True, False, "replay" or None; got \'yes\'.'),
])
def test_hyperparam_error_messages_are_pinned(params, expected):
    X, y, _ = _reg_data(n=50)
    with pytest.raises(ValueError) as e:
        ChimeraBoostRegressor(**params).fit(X, y)
    assert str(e.value) == expected


def test_classifier_accepts_always_and_stays_inert_below_the_gate():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, 50)
    m = ChimeraBoostClassifier(cross_features="always").fit(X, y)
    assert m.cross_features_selected_ is None


def test_fit_input_error_messages_are_pinned():
    X, y, w = _reg_data(n=50)
    est = ChimeraBoostRegressor(n_estimators=10)

    with pytest.raises(ValueError) as e:
        est.fit(X, None)
    assert str(e.value) == ("This estimator requires y to be passed, but the "
                            "target y is None.")

    with pytest.raises(ValueError) as e:
        est.fit(X, y[:-1])
    assert str(e.value) == ("X and y have inconsistent lengths: X has 50 "
                            "samples, y has 49.")

    # The classic slip: fit(X, y, w) binds weights to cat_features.
    with pytest.raises(ValueError) as e:
        est.fit(X, y, w)
    assert str(e.value) == (
        "cat_features must be integer column indices or column names. Got an "
        "array of floats -- if these are per-sample weights, pass them by "
        "keyword: fit(X, y, sample_weight=w).")

    with pytest.raises(ValueError) as e:
        est.fit(X, y, sample_weight=-w)
    assert str(e.value) == "sample_weight must be non-negative."

    Xinf = X.copy()
    Xinf[0, 0] = np.inf
    with pytest.raises(ValueError) as e:
        est.fit(Xinf, y)
    assert str(e.value) == ("X contains infinity. NaN is accepted (treated as "
                            "missing), but inf is not -- clip or clean it "
                            "first.")


# ---------------------------------------------------------------------------
# Logloss fused kernels (F4 C3): the scalar twin of the multiclass fusion.
# `_logloss_grad_hess_kernel` and `_logloss_ce_kernel` transcribe the numpy
# bodies element by element, so both sides run the same numba exp() on the
# same machine and the comparison is EXACT equality -- same-machine
# bit-identity. (The ce kernel's log() is numba's where the oracle's is
# numpy's; the two agree to the bit on the clipped probability range on this
# machine -- a 2.5M-value probe found zero bit differences before shipping --
# and the tests below pin exact agreement on their data.)
# ---------------------------------------------------------------------------

def _assert_bit_equal(a, b):
    # grad and hess: both arms run numba's exp() on the same machine, so the
    # comparison is exact everywhere the tests run.
    assert np.array_equal(a, b, equal_nan=True)
    assert a.tobytes() == b.tobytes()


def _assert_ce_equal(a, b):
    # The per-row cross-entropy: the kernel's log() is LLVM libm, the oracle's
    # is numpy's SIMD log(), and those may round the last bits differently on
    # the CI runner pool -- the same 4-ULP caveat the softmax pins above carry.
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


def _logloss_raw_families(rng):
    """(tag, raw) pairs covering the sigmoid's branches, saturation and edges."""
    return [("gauss", rng.normal(0.0, 3.0, 10_007)),
            ("extreme", np.array([40.0, -40.0, 700.0, -700.0, 1e6, -1e6]
                                 * 500, dtype=np.float64)),
            ("inf", np.array([np.inf, -np.inf, 0.0, 5.0, -5.0] * 200,
                             dtype=np.float64)),
            ("empty", np.empty(0, dtype=np.float64)),
            ("len1", np.array([0.7]))]


def test_logloss_grad_hess_matches_oracle_exactly(monkeypatch):
    # The spy proves the fused kernel -- not the numpy fallback -- produced
    # every one of these.
    import chimeraboost.losses as losses

    calls = []
    orig = losses._logloss_grad_hess_kernel

    def spy(raw, y):
        calls.append(raw.shape)
        return orig(raw, y)

    monkeypatch.setattr(losses, "_logloss_grad_hess_kernel", spy)
    rng = np.random.default_rng(21)
    loss = losses.Logloss()
    for tag, raw in _logloss_raw_families(rng):
        raw = np.ascontiguousarray(raw, dtype=np.float64)
        y = rng.integers(0, 2, raw.shape[0]).astype(np.float64)
        grad, hess = loss.grad_hess(y, raw)
        egrad, ehess = loss._grad_hess_numpy(y, raw)
        _assert_bit_equal(grad, egrad)
        _assert_bit_equal(hess, ehess)
        if tag == "extreme":
            # The floor must actually engage on both sides, or this family
            # proves nothing about the comparison transcription.
            assert np.any(hess[raw > 0] == 1e-6)
            assert np.any(hess[raw < 0] == 1e-6)
    assert len(calls) == 5


def test_logloss_eval_matches_oracle_exactly(monkeypatch):
    import chimeraboost.losses as losses

    calls = []
    orig = losses._logloss_ce_kernel

    def spy(raw, y):
        calls.append(raw.shape)
        return orig(raw, y)

    monkeypatch.setattr(losses, "_logloss_ce_kernel", spy)
    rng = np.random.default_rng(22)
    loss = losses.Logloss()
    for tag, raw in _logloss_raw_families(rng):
        raw = np.ascontiguousarray(raw, dtype=np.float64)
        y = rng.integers(0, 2, raw.shape[0]).astype(np.float64)
        # Per-row vector vs the old per-row ce expression in numpy.
        p = np.clip(losses._sigmoid(raw), 1e-9, 1 - 1e-9)
        ce_ref = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        _assert_ce_equal(orig(raw, y), ce_ref)
        # Empty weights sum to zero, which np.average rejects -- pinned as an
        # identical raise on both arms below, not as a value here.
        ws = [None] if tag == "empty" else [None, rng.uniform(0.5, 2.0, raw.shape[0])]
        for w in ws:
            _assert_eval_equal(loss.eval(y, raw, sample_weight=w),
                               loss._eval_numpy(y, raw, sample_weight=w))
    with pytest.raises(ZeroDivisionError):
        loss.eval(np.empty(0), np.empty(0), sample_weight=np.empty(0))
    with pytest.raises(ZeroDivisionError):
        loss._eval_numpy(np.empty(0), np.empty(0), sample_weight=np.empty(0))
    assert len(calls) == 10


def test_logloss_eval_matches_oracle_on_soft_labels():
    import chimeraboost.losses as losses

    rng = np.random.default_rng(23)
    loss = losses.Logloss()
    raw = rng.normal(0.0, 3.0, 5000)
    soft = rng.uniform(0.0, 1.0, 5000)
    assert np.all((soft != 0.0) & (soft != 1.0))  # neither branch fires
    mixed = soft.copy()
    mixed[::3] = 0.0
    mixed[1::3] = 1.0
    for y in (soft, mixed):
        p = np.clip(losses._sigmoid(raw), 1e-9, 1 - 1e-9)
        ce_ref = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        _assert_ce_equal(losses._logloss_ce_kernel(raw, y), ce_ref)
        for w in (None, rng.uniform(0.5, 2.0, raw.shape[0])):
            _assert_eval_equal(loss.eval(y, raw, sample_weight=w),
                               loss._eval_numpy(y, raw, sample_weight=w))


def test_logloss_kernels_pass_nan_through():
    import chimeraboost.losses as losses

    rng = np.random.default_rng(24)
    loss = losses.Logloss()
    raw = rng.normal(0.0, 3.0, 2000)
    raw[::500] = np.nan
    y = rng.integers(0, 2, 2000).astype(np.float64)
    p = np.clip(losses._sigmoid(raw), 1e-9, 1 - 1e-9)
    ce_ref = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    ce = losses._logloss_ce_kernel(raw, y)
    _assert_ce_equal(ce, ce_ref)
    grad, hess = loss.grad_hess(y, raw)
    egrad, ehess = loss._grad_hess_numpy(y, raw)
    assert np.array_equal(np.isnan(grad), np.isnan(egrad))
    _assert_bit_equal(grad, egrad)
    # The hess floor is the specified comparison (`h if h >= 1e-6 else 1e-6`),
    # which yields 1e-6 on NaN where np.maximum yields NaN -- the documented
    # non-NaN scope of the transcription, same as the shipped multiclass twin.
    # NaN raw is unreachable on the fit path (raw scores stay finite).
    nan_pos = np.isnan(ehess)
    assert np.any(nan_pos)
    assert np.all(hess[nan_pos] == 1e-6)
    _assert_bit_equal(hess[~nan_pos], ehess[~nan_pos])
    _assert_eval_equal(loss.eval(y, raw), loss._eval_numpy(y, raw))


def test_logloss_fallbacks_tripwire(monkeypatch):
    # Every input the guard refuses must take the numpy path: if a kernel runs
    # where it should not, the spy below records it and the test fails.
    import chimeraboost.losses as losses

    gh_calls, ce_calls = [], []
    orig_gh = losses._logloss_grad_hess_kernel
    orig_ce = losses._logloss_ce_kernel

    def gh_spy(raw, y):
        gh_calls.append(raw.shape)
        return orig_gh(raw, y)

    def ce_spy(raw, y):
        ce_calls.append(raw.shape)
        return orig_ce(raw, y)

    monkeypatch.setattr(losses, "_logloss_grad_hess_kernel", gh_spy)
    monkeypatch.setattr(losses, "_logloss_ce_kernel", ce_spy)
    rng = np.random.default_rng(25)
    loss = losses.Logloss()
    base = rng.normal(size=200)
    yb = rng.integers(0, 2, 200).astype(np.float64)
    # float32 raw and a non-contiguous slice run the numpy path fine.
    for raw, y in ((base.astype(np.float32), yb), (base[::2], yb[:100])):
        assert not losses._scalar_pair_ok(raw, y)
        grad, hess = loss.grad_hess(y, raw)
        egrad, ehess = loss._grad_hess_numpy(y, raw)
        _assert_bit_equal(grad, egrad)
        _assert_bit_equal(hess, ehess)
        _assert_eval_equal(loss.eval(y, raw), loss._eval_numpy(y, raw))
    # 2-D raw and mismatched lengths also take the numpy path -- which raises,
    # exactly as the old body did (a numba TypingError from _sigmoid for 2-D,
    # a broadcast ValueError for the mismatch). The kernels stay silent.
    with pytest.raises(Exception):
        loss.grad_hess(yb, base.reshape(-1, 1))
    with pytest.raises(Exception):
        loss.eval(yb, base.reshape(-1, 1))
    with pytest.raises(ValueError):
        loss.grad_hess(yb[:-1], base)
    with pytest.raises(ValueError):
        loss.eval(yb[:-1], base)
    assert gh_calls == [] and ce_calls == []
    # A plain contiguous float64 pair takes the kernels -- one call each.
    loss.grad_hess(yb, base)
    loss.eval(yb, base)
    assert len(gh_calls) == 1 and len(ce_calls) == 1


def test_logloss_refactor_is_end_to_end_identical(monkeypatch):
    import chimeraboost.losses as losses

    rng = np.random.default_rng(26)
    X = rng.normal(size=(4000, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] * X[:, 3] > 0).astype(np.int64)
    Xtr, ytr = X[:3000], y[:3000]
    Xho, yho = X[3000:], y[3000:]

    def fit():
        m = ChimeraBoostClassifier(n_estimators=100, random_state=0)
        m.fit(Xtr, ytr, eval_set=(Xho[:500], yho[:500]))
        return m

    kern = fit()
    monkeypatch.setattr(losses.Logloss, "grad_hess",
                        losses.Logloss._grad_hess_numpy)
    monkeypatch.setattr(losses.Logloss, "eval", losses.Logloss._eval_numpy)
    npy = fit()
    np.testing.assert_array_equal(kern.predict_proba(Xho),
                                  npy.predict_proba(Xho))
    assert kern.best_iteration_ == npy.best_iteration_
    kh, nh = kern.model_.valid_history_, npy.model_.valid_history_
    assert len(kh) > 0 and len(kh) == len(nh)
    for a, b in zip(kh, nh):
        _assert_eval_equal(a, b)
