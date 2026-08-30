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


def test_forced_cross_on_classifier_message_is_pinned():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, 50)
    with pytest.raises(ValueError) as e:
        ChimeraBoostClassifier(cross_features="always").fit(X, y)
    assert str(e.value) == (
        'cross_features="always" is only supported on the regressor; '
        "the classifier's cross features are validation-raced "
        "(cross_features=None or True).")


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
