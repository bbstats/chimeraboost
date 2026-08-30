"""Oblivious (symmetric) decision trees, numba-accelerated.

An oblivious tree of depth D uses the SAME (feature, bin-threshold) split at
every node of a given level. A row's leaf is therefore just a D-bit number, one
bit per level: bit_d = 1 if X[feature_d] > threshold_d else 0. This makes:

  * prediction a handful of comparisons + an array lookup (very fast), and
  * the model strongly regularized (only D splits per tree, shared across the
    whole level), which is a big part of why the defaults don't overfit.

We grow level by level. At each level we build per-(feature, current-leaf, bin)
gradient/hessian histograms and pick the single split that maximizes the summed
XGBoost-style gain over all current leaves.
"""

import numpy as np
from numba import njit, prange


# Below this many samples, per-level cost is fixed overhead (kernel launches,
# empty-leaf zero/scan, strided split scans) rather than sample count, so
# build_oblivious_tree switches to the small-n variants: occupied-leaf lists
# and a serial descend.
#
# Both sides of every dispatch are bit-identical, so this only moves speed.
# 32768 sits between the measured regimes: 1.7x fused win at 2k, parity at 200k.
_SMALL_N = 32768

# Above this many rows, ObliviousTree.apply descends level by level with the
# parallel `_descend_leaves` instead of the serial fused `_assign_leaves`. The
# per-level fork/join only pays for itself once the serial O(n x depth) walk
# clearly dominates it.
#
# Both sides are bit-identical, so this only moves speed. It matters once per
# boosting round: eval-set scoring and replay refits assign leaves once per
# retained tree.
_ASSIGN_PAR_N = 32768

# Passed as the fused level kernel's occupancy buffer on the large-n path,
# where it is never touched. Shared and read-only, so safe across concurrent
# fits.
_EMPTY_I64 = np.empty(0, dtype=np.int64)


# --- MVS gradient row sampling -------------------------------------------
# These run once per boosting round, before any tree work. They live here
# rather than in booster.py because booster.py holds no kernels: it imports
# every one of them from this module.
#
# The tree build is numba-parallel across every core, while this step used to
# be single-threaded numpy allocating ~10 full-length temporaries per round.
# At 200k rows subsampling therefore cost more than the trees it fed: 61% of
# a subsample=0.7 fit. Whole-fit effect of the port, 12 threads, best of 2
# (2026-07-30):
#
#                             case   subsample     base      new
#     regression 200k x 20, 200 trees      0.3    2.94 s   2.13 s   1.38x
#                                          0.7    3.09 s   2.16 s   1.43x
#     regression   2M x 20,  50 trees      0.7    7.58 s   5.67 s   1.34x
#     binary     200k x 20, 200 trees      0.7    3.96 s   3.00 s   1.32x
#     multiclass 200k x 20, 100 trees      0.7    5.02 s   4.63 s   1.08x
#
# Multiclass gains least: its vector-leaf round does K channels of work around
# one shared row draw, so sampling is a smaller slice to begin with. What is
# left is dominated by the np.sort, which stays in numpy on purpose (see
# _mvs_lambda_scan) and which numpy 2.x vectorizes better than we could.


@njit(cache=True)
def _mvs_lambda_scan(sorted_g, total, target):
    """MVS threshold: the first k where sorted_g[k] <= suffix[k]/remaining[k].

    `sorted_g` is |grad| sorted DESCENDING and `total` is its sum. The caller
    computes both in numpy and they MUST stay there: they are the only two
    operations in the MVS path that rely on numpy's pairwise summation. Redoing
    them here would move lambda in the last ulp, and a moved lambda changes which
    rows the mask keeps -- so it changes the model.

    Everything else is exact, so this loop is bit-identical to the vectorized
    form it replaces:

      * `prefix` reproduces `np.cumsum(sorted_g[:-1])` exactly -- cumsum is
        `add.accumulate`, sequential by definition, so there is no summation
        order to disagree about.
      * `remaining` equals `target - np.arange(n)[k]` exactly for n < 2**53.
      * the comparison and the final divide are single IEEE-754 double
        operations on the same operands as the vectorized form. No fastmath
        anywhere in this module, and no `a*b + c` shape here, so nothing can be
        contracted into an FMA.

    Fusing also buys an early exit and drops six full-length temporaries (prefix,
    suffix, remaining, the product, the boolean cond, its argmax). Once
    `remaining` hits zero every later k fails too, so bailing there matches
    `cond.any() == False` -> lambda 0, the "all rows forced" signal the caller
    reads as "use the uniform fallback".
    """
    n = sorted_g.shape[0]
    prefix = 0.0

    for k in range(n):
        remaining = target - np.float64(k)
        if remaining <= 0.0:
            return 0.0

        suffix = total - prefix
        if sorted_g[k] * remaining <= suffix:
            return suffix / remaining

        prefix += sorted_g[k]

    return 0.0


@njit(cache=True, parallel=True)
def _mvs_weights(grad, u, lam, max_w):
    """Per-row MVS importance weights: 1/p where the row survives, else 0.

    One fused pass replaces seven numpy passes and their temporaries (`np.abs`,
    the divide, `np.minimum`, the mask compare, `np.maximum`, the cap
    `np.minimum`, `np.where`). All of those are elementwise, so evaluating them
    per row reproduces the vectorized result exactly -- including the NaN
    corners, where `p > 1.0` is False so `p` stays NaN, `u < NaN` is False, and
    the row lands on the zero branch just as `np.where(mask, ..., 0.0)` puts it.

    `u` is the caller's `rng.random(n)` draw rather than one generated here:
    that draw is a single call on the booster's numpy Generator and the stream
    position is golden-frozen.

    The cap `max_w = 1/subsample` bounds the reweighting of near-zero-gradient
    rows, whose effective contribution g_i/p_i is lambda regardless.
    """
    n = grad.shape[0]
    w = np.empty(n, dtype=np.float64)
    for i in prange(n):
        p = abs(grad[i]) / lam
        if p > 1.0:
            p = 1.0
        if u[i] < p:
            q = p if p > 1e-10 else 1e-10
            v = 1.0 / q
            w[i] = v if v <= max_w else max_w
        else:
            w[i] = 0.0
    return w


@njit(cache=True)
def _mvs_weights_serial(grad, u, lam, max_w):
    """Serial twin of `_mvs_weights` for small n, where the parallel fork/join
    costs more than the pass itself.

    Every write is independent, so the two are bit-identical; the booster
    dispatches on `_SMALL_N`.
    """
    n = grad.shape[0]
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = abs(grad[i]) / lam
        if p > 1.0:
            p = 1.0
        if u[i] < p:
            q = p if p > 1e-10 else 1e-10
            v = 1.0 / q
            w[i] = v if v <= max_w else max_w
        else:
            w[i] = 0.0
    return w


@njit(cache=True, parallel=True)
def _build_histograms_into(Xb, grad, hess, leaf, n_leaves, hist, feat_mask):
    """Fill per-feature gradient/hessian histograms into a pre-allocated buffer.

    REFERENCE KERNEL: the fit path now uses the fused `_build_and_split`. This
    is kept (with `_best_split`) as the plainly-readable equivalence oracle for
    tests/test_tree_kernels.py.

    `Xb` is feature-major (n_features, n_samples), so `Xb[f]` is a contiguous
    row and the inner sample loop reads bins, grads and hessians sequentially.

    `hist` has shape (n_features, max_leaves, max_bins, 2). Grad and hess for a
    bin are interleaved on the last axis so each scatter write touches one cache
    line instead of two arrays. The buffer is reused across every tree and level,
    and we zero only the (n_leaves) slice we are about to write. Parallel over
    features, so each thread owns a disjoint slice -- no races.

    Features with feat_mask[f] == 0 (column subsampling) are skipped entirely.
    `_best_split` honors the same mask and never reads their slice, so the stale
    data left there is harmless and the whole scan is saved. At colsample=c that
    removes a (1-c) fraction of histogram work.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        for l in range(n_leaves):
            for b in range(max_bins):
                hist[f, l, b, 0] = 0.0
                hist[f, l, b, 1] = 0.0

        Xf = Xb[f]
        for i in range(n_samples):
            l = leaf[i]
            b = Xf[i]
            hist[f, l, b, 0] += grad[i]
            hist[f, l, b, 1] += hess[i]


@njit(cache=True, parallel=True)
def _descend_leaves(leaf, Xf, t):
    """Push every sample one level deeper, in place: leaf = (leaf<<1) + (Xf > t).

    The fit path descends inside `_build_split_descend`. This kernel is the
    large-n arm of `ObliviousTree.apply` (per-round eval scoring, replay refits)
    and the descend oracle for tests/test_tree_kernels.py.

    It replaces the numpy expression
    ``leaf = (leaf << 1) + (Xb[f] > t).astype(np.int64)``, which allocated
    several n-sample temporaries (the bool mask, its int64 cast, the shifted
    array, the sum) on every one of the (max_depth x n_trees) level steps —
    measured at ~⅓ of total fit time. One parallel pass over the contiguous
    feature row, no temporaries, bit-identical bucketing.
    """
    for i in prange(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > t else 0)


@njit(cache=True)
def _descend_leaves_serial(leaf, Xf, t):
    """Serial twin of `_descend_leaves` for small n, where the parallel
    fork/join costs more than the pass itself (~4.7us vs 0.9us at n=2k).

    Every write is independent, so serial and parallel are bit-identical;
    `build_oblivious_tree` dispatches on `_SMALL_N`.
    """
    for i in range(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > t else 0)


@njit(cache=True, parallel=True)
def _build_and_split(Xb, grad, hess, leaf, active, hist, feat_mask,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                     n_bins_per_feature, l2, min_child_weight):
    """Fused histogram build + best-split search: one parallel launch per level
    instead of two, with the split scan reading the hist slice the same thread
    just wrote (cache-hot).

    REFERENCE KERNEL: the fit path runs `_build_split_descend`, which adds the
    level's descend and occupancy to this search. Kept as that kernel's
    split-search oracle for tests/test_tree_kernels.py.

    Output is EXACTLY `_build_histograms_into` followed by `_best_split` --
    checked by an exact-equality test -- while cutting the small-n fixed cost
    three ways:

      * `active` lists the leaf rows that hold samples (any superset works, e.g.
        arange(n_leaves) when counting isn't worth it). Empty leaves are all-zero
        rows, so skipping them skips zeroing and scanning cells that contribute
        nothing. Occupancy is feature-independent, so one list serves every
        feature.
      * Only bins [0, n_bins_[f]) are zeroed and scanned — the scatter never
        writes past a feature's actual bin count.
      * The scan is transposed (leaf-outer, bin-inner), so the prefix and gain
        passes stream each hist row instead of striding across leaf rows per
        threshold. gain[t] still accumulates leaves in ascending order, so every
        float sum matches `_best_split` bit for bit. The parent term
        gt*gt/(ht+l2) is computed once per leaf: same value, one divide instead
        of nb-1.

    Legality (`min_child_weight`) matches the reference: a threshold dies if ANY
    contributing leaf would gain a sparse non-empty child. Leaves with no hessian
    mass cast neither gain nor veto.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        # Zero this feature's leaf rows, then scatter the samples in.
        nb = n_bins_per_feature[f]
        for k in range(n_active):
            l = active[k]
            for b in range(nb):
                hist[f, l, b, 0] = 0.0
                hist[f, l, b, 1] = 0.0

        Xf = Xb[f]
        for i in range(n_samples):
            l = leaf[i]
            b = Xf[i]
            hist[f, l, b, 0] += grad[i]
            hist[f, l, b, 1] += hess[i]

        # Score every threshold, summing gain across the leaves.
        gain = np.zeros(max_bins)
        legal = np.ones(max_bins, dtype=np.uint8)

        for k in range(n_active):
            l = active[k]
            gt = 0.0
            ht = 0.0
            for b in range(nb):
                gt += hist[f, l, b, 0]
                ht += hist[f, l, b, 1]

            if ht <= 0.0:
                continue

            par = gt * gt / (ht + l2)
            gl = 0.0
            hl = 0.0

            for t in range(nb - 1):
                gl += hist[f, l, t, 0]
                hl += hist[f, l, t, 1]
                hr = ht - hl
                if (hl > 0.0 and hl < min_child_weight) or \
                   (hr > 0.0 and hr < min_child_weight):
                    legal[t] = 0
                else:
                    gr = gt - gl
                    gain[t] += (gl * gl / (hl + l2)
                                + gr * gr / (hr + l2)
                                - par)

        best_g = -np.inf
        best_t = -1
        for t in range(nb - 1):
            if legal[t] and gain[t] > best_g:
                best_g = gain[t]
                best_t = t

        feat_gain[f] = best_g
        feat_thr[f] = best_t

    # Best feature overall.
    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f

    return best_f, feat_thr[best_f], best_gain


@njit(cache=True, parallel=True)
def _build_split_descend(Xb, grad, hess, leaf, active, hist, feat_mask,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                         n_bins_per_feature, l2, min_child_weight, min_gain,
                         small, n_leaves_next, next_active):
    """`_build_and_split` plus the level's follow-up work in the same launch.

    When the split it finds is usable (legal threshold, gain > min_gain) this
    kernel also pushes every sample one level deeper, and on the small-n path
    emits the next level's occupied-leaf list.

    Per level it replaces one split launch, one descend launch and a
    bincount/flatnonzero numpy pair. At small n the per-level cost is fixed
    overhead, not arithmetic (GROW_PLAN.md Phase 0: per-tree Python residue is
    8-15% of fit on Grinsztajn-sized sets). That overhead is what this removes.

    Bit-identity, piece by piece:

      * the split search is `_build_and_split`'s code verbatim (that kernel is
        this one's oracle, and is itself oracle-tested against
        `_build_histograms_into` + `_best_split`);
      * the descend is `_descend_leaves(_serial)`'s integer update, fused with
        an integer occupancy count, exact in any order;
      * the occupancy list is ascending nonzero-count indices — exactly
        flatnonzero(bincount(leaf, n_leaves_next)).

    The descend fires iff the caller's continue-predicate holds (NOT (gain <=
    min_gain or t < 0)), so a rejected level leaves `leaf` untouched, the way
    the old Python-side break did.

    Returns (best_f, best_t, best_gain, n_next). n_next is the occupancy list
    length, or -1 when no list was built (large n, or no descend).
    `next_active` needs room for n_leaves_next entries on the small-n path; it
    is never touched otherwise.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        # Zero this feature's leaf rows, then scatter the samples in.
        nb = n_bins_per_feature[f]
        for k in range(n_active):
            l = active[k]
            for b in range(nb):
                hist[f, l, b, 0] = 0.0
                hist[f, l, b, 1] = 0.0

        Xf = Xb[f]
        for i in range(n_samples):
            l = leaf[i]
            b = Xf[i]
            hist[f, l, b, 0] += grad[i]
            hist[f, l, b, 1] += hess[i]

        # Score every threshold, summing gain across the leaves.
        gain = np.zeros(max_bins)
        legal = np.ones(max_bins, dtype=np.uint8)

        for k in range(n_active):
            l = active[k]
            gt = 0.0
            ht = 0.0
            for b in range(nb):
                gt += hist[f, l, b, 0]
                ht += hist[f, l, b, 1]

            if ht <= 0.0:
                continue

            par = gt * gt / (ht + l2)
            gl = 0.0
            hl = 0.0

            for t in range(nb - 1):
                gl += hist[f, l, t, 0]
                hl += hist[f, l, t, 1]
                hr = ht - hl
                if (hl > 0.0 and hl < min_child_weight) or \
                   (hr > 0.0 and hr < min_child_weight):
                    legal[t] = 0
                else:
                    gr = gt - gl
                    gain[t] += (gl * gl / (hl + l2)
                                + gr * gr / (hr + l2)
                                - par)

        best_g = -np.inf
        best_t = -1
        for t in range(nb - 1):
            if legal[t] and gain[t] > best_g:
                best_g = gain[t]
                best_t = t

        feat_gain[f] = best_g
        feat_thr[f] = best_t

    # Best feature overall.
    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    best_t = feat_thr[best_f]

    # Descend one level. At small n, list the leaves that are now occupied so
    # the next level can skip the empty rows.
    n_next = -1
    if best_t >= 0 and best_gain > min_gain:
        Xf = Xb[best_f]
        if small:
            counts = np.zeros(n_leaves_next, dtype=np.int64)
            for i in range(n_samples):
                nl = (leaf[i] << 1) + (1 if Xf[i] > best_t else 0)
                leaf[i] = nl
                counts[nl] += 1

            n_next = 0
            for l in range(n_leaves_next):
                if counts[l] > 0:
                    next_active[n_next] = l
                    n_next += 1
        else:
            for i in prange(n_samples):
                leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > best_t else 0)

    return best_f, best_t, best_gain, n_next


# Quantized-gradient histograms (QUANT_PLAN.md, a LightGBM-4-style adaptation).
# grad/hess are quantized per tree to integers and packed into ONE int64 per
# sample, so the histogram scatter does a single integer read-modify-write per
# (sample, feature) instead of two float64 ones, and the buffer footprint halves.
#
# _QMAX_CAP bounds the quantized range at 15 bits. build_oblivious_tree shrinks
# it further for huge n so that any cell or prefix sum keeps
# |sum qg| <= n*qmax < 2**31 and 0 <= sum qh < 2**32. The packed halves can then
# never bleed into each other and shift/mask unpacking is exact.
_QMAX_CAP = 32767


@njit(cache=True, parallel=True)
def _gh_absmax(grad, hess):
    """Fused (max |grad|, max hess) reduction — the quantization scales.

    Avoids numpy temporaries: np.abs(grad).max() would allocate n floats.
    """
    gmax = 0.0
    hmax = 0.0
    for i in prange(grad.shape[0]):
        ag = abs(grad[i])
        gmax = max(gmax, ag)
        hmax = max(hmax, hess[i])
    return gmax, hmax


@njit(cache=True, parallel=True)
def _quantize_pack(grad, hess, inv_dg, inv_dh, qmax, qseed, out):
    """out[i] = (qg << 32) + qh with stochastic rounding qX = floor(x*inv + u).

    The uniform pair u comes from a counter-based splitmix64(qseed + i), which
    buys two things. It is deterministic given the seed, so models stay
    reproducible without threading RNG state through numba. And the rounding is
    unbiased: round-to-nearest would bias every histogram cell the same way,
    whereas stochastic errors cancel by sqrt(n) — the LightGBM
    quantized-training result.

    The scales put qg in [-qmax, qmax] and qh in [0, qmax] by construction. The
    clamps only guard the edge where gmax * (qmax/gmax) rounds a hair above qmax,
    which keeps the caller's overflow bound exact. Hessians are non-negative for
    every library loss, so qh's lower clamp is defensive only.
    """
    n = grad.shape[0]
    for i in prange(n):
        # splitmix64 of the row index, split into two uniforms.
        z = (qseed + np.uint64(i)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        u1 = (z & np.uint64(0xFFFFFFFF)) * (1.0 / 4294967296.0)
        u2 = (z >> np.uint64(32)) * (1.0 / 4294967296.0)

        qg = np.int64(np.floor(grad[i] * inv_dg + u1))
        qh = np.int64(np.floor(hess[i] * inv_dh + u2))
        qg = min(max(qg, -qmax), qmax)
        qh = min(max(qh, np.int64(0)), qmax)
        out[i] = (qg << 32) + qh


@njit(cache=True, parallel=True)
def _build_split_descend_q(Xb, q, leaf, active, histq, feat_mask,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                           n_bins_per_feature, dg, dh, l2, min_child_weight,
                           min_gain, small, n_leaves_next, next_active):
    """Packed-int64 twin of `_build_split_descend` for quantized training.

    The structure is that kernel's verbatim except for the histogram. `histq` is
    int64 (n_features, max_leaves, max_bins), the scatter adds the packed sample
    value once, and the scan runs packed-integer prefix sums (exact — see
    _QMAX_CAP) that are unpacked by arithmetic shift and mask, then dequantized
    with dg/dh only where the float gain formula needs them.

    On exactly-representable grad/hess (integer multiples of power-of-two
    scales) this reproduces the float kernel bit for bit — the oracle test in
    tests/test_tree_kernels.py. On real data it differs only by the quantization
    noise. hr = ht - hl is computed in float like the reference; multiplying by
    a positive scale is monotone, so hr never goes negative.
    """
    n_features, n_samples = Xb.shape
    max_bins = histq.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        # Zero this feature's leaf rows, then scatter the samples in.
        nb = n_bins_per_feature[f]
        for k in range(n_active):
            l = active[k]
            for b in range(nb):
                histq[f, l, b] = 0

        Xf = Xb[f]
        for i in range(n_samples):
            histq[f, leaf[i], Xf[i]] += q[i]

        # Score every threshold, unpacking each integer prefix sum as it goes.
        gain = np.zeros(max_bins)
        legal = np.ones(max_bins, dtype=np.uint8)

        for k in range(n_active):
            l = active[k]
            tot = np.int64(0)
            for b in range(nb):
                tot += histq[f, l, b]

            ht = (tot & 0xFFFFFFFF) * dh
            gt = (tot >> 32) * dg
            if ht <= 0.0:
                continue

            par = gt * gt / (ht + l2)
            acc = np.int64(0)

            for t in range(nb - 1):
                acc += histq[f, l, t]
                hl = (acc & 0xFFFFFFFF) * dh
                gl = (acc >> 32) * dg
                hr = ht - hl
                if (hl > 0.0 and hl < min_child_weight) or \
                   (hr > 0.0 and hr < min_child_weight):
                    legal[t] = 0
                else:
                    gr = gt - gl
                    gain[t] += (gl * gl / (hl + l2)
                                + gr * gr / (hr + l2)
                                - par)

        best_g = -np.inf
        best_t = -1
        for t in range(nb - 1):
            if legal[t] and gain[t] > best_g:
                best_g = gain[t]
                best_t = t

        feat_gain[f] = best_g
        feat_thr[f] = best_t

    # Best feature overall.
    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    best_t = feat_thr[best_f]

    # Descend one level. At small n, list the leaves that are now occupied so
    # the next level can skip the empty rows.
    n_next = -1
    if best_t >= 0 and best_gain > min_gain:
        Xf = Xb[best_f]
        if small:
            counts = np.zeros(n_leaves_next, dtype=np.int64)
            for i in range(n_samples):
                nl = (leaf[i] << 1) + (1 if Xf[i] > best_t else 0)
                leaf[i] = nl
                counts[nl] += 1

            n_next = 0
            for l in range(n_leaves_next):
                if counts[l] > 0:
                    next_active[n_next] = l
                    n_next += 1
        else:
            for i in prange(n_samples):
                leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > best_t else 0)

    return best_f, best_t, best_gain, n_next


@njit(cache=True, parallel=True)
def _best_split(hist, n_bins_per_feature, l2, feat_mask, min_child_weight,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                n_leaves):
    """Find the (feature, threshold) with the highest total gain.

    REFERENCE KERNEL: the fit path now uses the fused `_build_and_split`. This
    is kept as the equivalence oracle for tests/test_tree_kernels.py.

    `hist` is the interleaved (n_features, max_leaves, max_bins, 2) buffer:
    [..., 0] is grad, [..., 1] is hess. `n_leaves` is how many leaf rows are
    active at this level, so we read only those.

    For a candidate threshold t, bins <= t go left and bins > t go right, the
    same way in every current leaf. Gain is summed across leaves. Features with
    feat_mask[f] == 0 are skipped (column subsampling).

    A threshold is legal unless some leaf would gain a *sparse non-empty* child
    (0 < hessian mass < min_child_weight). That is the sparse-leaf overfit risk,
    and since the split is shared it is rejected for the whole level. Children
    that come out EMPTY (a leaf whose samples all go one way) are exempt: pure
    leaves are normal in an oblivious tree, and blocking the shared split on them
    caps effective depth far below what the data supports.
    """
    n_features = hist.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        # Totals per leaf for this feature (same regardless of threshold).
        nb = n_bins_per_feature[f]
        Gt = np.zeros(n_leaves)
        Ht = np.zeros(n_leaves)
        for l in range(n_leaves):
            for b in range(nb):
                Gt[l] += hist[f, l, b, 0]
                Ht[l] += hist[f, l, b, 1]

        GL = np.zeros(n_leaves)
        HL = np.zeros(n_leaves)
        best_g = -np.inf
        best_t = -1

        # Threshold t means "left = bins [0..t]". Last bin can't be a threshold.
        for t in range(nb - 1):
            # Pass 1: advance the running prefix sums for all leaves
            # unconditionally, so GL/HL carry into the next threshold.
            for l in range(n_leaves):
                GL[l] += hist[f, l, t, 0]
                HL[l] += hist[f, l, t, 1]

            # Pass 2: gain of this threshold, and its legality. Only a sparse
            # non-empty child vetoes the shared split (see the docstring).
            gain = 0.0
            legal = True
            for l in range(n_leaves):
                if Ht[l] > 0.0:
                    hl = HL[l]
                    hr = Ht[l] - hl

                    # An empty child (hl == 0 or hr == 0) is exempt.
                    if (hl > 0.0 and hl < min_child_weight) or \
                       (hr > 0.0 and hr < min_child_weight):
                        legal = False
                        break

                    gl = GL[l]
                    gr = Gt[l] - gl
                    gain += (
                        gl * gl / (hl + l2)
                        + gr * gr / (hr + l2)
                        - Gt[l] * Gt[l] / (Ht[l] + l2)
                    )

            if legal and gain > best_g:
                best_g = gain
                best_t = t

        feat_gain[f] = best_g
        feat_thr[f] = best_t

    # Best feature overall.
    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f

    return best_f, feat_thr[best_f], best_gain


@njit(cache=True)
def _assign_leaves(Xb, splits_feat, splits_thr):
    """Leaf index of every sample given the splits. `Xb` is feature-major, so
    each level reads one contiguous feature row."""
    depth = splits_feat.shape[0]
    n = Xb.shape[1]
    leaf = np.zeros(n, dtype=np.int64)
    for d in range(depth):
        Xf = Xb[splits_feat[d]]
        t = splits_thr[d]
        for i in range(n):
            leaf[i] = leaf[i] * 2 + (1 if Xf[i] > t else 0)
    return leaf


@njit(cache=True)
def _leaf_values(leaf, grad, hess, n_leaves, l2, lr):
    """Newton leaf values: value = -G / (H + l2), scaled by learning rate."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for i in range(leaf.shape[0]):
        G[leaf[i]] += grad[i]
        H[leaf[i]] += hess[i]

    values = np.zeros(n_leaves)
    for l in range(n_leaves):
        if H[l] > 0.0:
            values[l] = -lr * G[l] / (H[l] + l2)
    return values


@njit(cache=True)
def _solve_small(A, b):  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
    """Solve ``A x = b`` for a small dense system via LU with partial pivoting.

    Drop-in replacement for ``np.linalg.solve`` on the tiny (d x d, d <= depth+1)
    per-leaf normal equations. Same algorithm family as LAPACK's gesv, but
    hand-rolled to avoid instantiating numba's LAPACK bindings: those alone cost
    several seconds of JIT compile time on the first fit in a fresh environment.

    ``A`` and ``b`` are modified in place. Returns the solution, or a vector of
    NaN if a pivot underflows -- the caller then falls back to the constant
    Newton leaf value. With the ridge and jitter on the diagonal, that cannot
    trigger in practice.
    """
    d = A.shape[0]
    x = np.empty(d)

    for c in range(d):
        # Pick the pivot row.
        p = c
        amax = abs(A[c, c])
        for r in range(c + 1, d):
            ar = abs(A[r, c])
            if ar > amax:
                amax = ar
                p = r

        if amax < 1e-300:
            for j in range(d):
                x[j] = np.nan
            return x

        if p != c:
            for j in range(d):
                tmp = A[c, j]
                A[c, j] = A[p, j]
                A[p, j] = tmp
            tmp = b[c]
            b[c] = b[p]
            b[p] = tmp

        # Eliminate below the pivot.
        inv = 1.0 / A[c, c]
        for r in range(c + 1, d):
            f = A[r, c] * inv
            if f != 0.0:
                A[r, c] = 0.0
                for j in range(c + 1, d):
                    A[r, j] -= f * A[c, j]
                b[r] -= f * b[c]

    # Back substitution.
    for r in range(d - 1, -1, -1):
        s = b[r]
        for j in range(r + 1, d):
            s -= A[r, j] * x[j]
        x[r] = s / A[r, r]

    return x


@njit(cache=True, parallel=True)
def _linear_leaf_fit(leaf, grad, hess, n_leaves, lin_feats, centers_std, Xb,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                     l2_intercept, lin_lambda, lr):
    """Fit a small hessian-weighted ridge per leaf (local linear-leaf models).

    For the samples in a leaf we solve the second-order objective

        min_beta  sum_i [ g_i f_i + 1/2 h_i f_i^2 ] + 1/2 ( l2*b^2 + lin*||w||^2 )

    with f_i = b + w . x_std_i over the leaf's numeric split features -- i.e. the
    normal equations  (A^T diag(h) A + Lambda) beta = -A^T g,  A = [1, x_std],
    accumulated directly, with no per-leaf design matrix. The output is
    `lr * beta`.

    Leaves too small to support a slope, and empty leaves, fall back to the plain
    constant Newton value, so the linear model only ever ADDS slope where the
    data supports it. Returns `lin_coef` of shape (n_leaves, 1 + len(lin_feats));
    column 0 is the intercept. `centers_std` is the per-feature table of
    standardized bin-center values, with NaN (missing) bins treated as 0, the
    feature mean.

    Notes
    -----
    Parallel over leaves and bit-identical to the old serial global scan. A
    stable counting sort groups sample indices by leaf in original order, so each
    leaf accumulates in exactly the float-add sequence the serial version used --
    a leaf only ever saw its own samples, in increasing i. Thread-count invariant
    for the same reason.

    Design values are gathered per sample inside the leaf loop: no (k, n) scratch
    matrix, and one parallel region holds the JIT compile cost down.
    """
    n = leaf.shape[0]
    k = lin_feats.shape[0]
    d = 1 + k
    coef = np.zeros((n_leaves, d))

    # Per-leaf grad/hess totals (for the constant fallback) and counts.
    counts = np.zeros(n_leaves, dtype=np.int64)
    Gtot = np.zeros(n_leaves)
    Htot = np.zeros(n_leaves)
    for i in range(n):
        l = leaf[i]
        counts[l] += 1
        Gtot[l] += grad[i]
        Htot[l] += hess[i]

    # Stable counting sort: order[start[l]:start[l+1]] = leaf-l samples in
    # increasing original index.
    start = np.zeros(n_leaves + 1, dtype=np.int64)
    for l in range(n_leaves):
        start[l + 1] = start[l] + counts[l]

    pos = start[:n_leaves].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        l = leaf[i]
        order[pos[l]] = i
        pos[l] += 1

    # Per-leaf normal equations + solve; leaves are independent.
    for l in prange(n_leaves):
        if counts[l] == 0:
            continue

        if counts[l] < 2 * d or k == 0:
            if Htot[l] > 0.0:
                coef[l, 0] = -lr * Gtot[l] / (Htot[l] + l2_intercept)
            continue

        Ml = np.zeros((d, d))
        rl = np.zeros(d)
        xrow = np.empty(k)

        for q in range(start[l], start[l + 1]):
            i = order[q]
            h = hess[i]
            g = grad[i]

            # Standardized design values for this sample; missing bins -> 0.
            for j in range(k):
                f = lin_feats[j]
                v = centers_std[f, Xb[f, i]]
                xrow[j] = v if np.isfinite(v) else 0.0

            Ml[0, 0] += h
            rl[0] += -g
            for j in range(k):
                xj = xrow[j]
                Ml[0, 1 + j] += h * xj
                Ml[1 + j, 0] += h * xj
                rl[1 + j] += -g * xj
                for jj in range(k):
                    Ml[1 + j, 1 + jj] += h * xj * xrow[jj]

        Ml[0, 0] += l2_intercept
        for j in range(1, d):
            Ml[j, j] += lin_lambda
        for j in range(d):
            Ml[j, j] += 1e-9              # jitter: keep the solve well-posed

        beta = _solve_small(Ml, rl)
        if np.isnan(beta[0]):
            # Singular pivot (unreachable given the diagonal ridge + jitter):
            # keep the plain constant Newton value rather than a broken slope.
            if Htot[l] > 0.0:
                coef[l, 0] = -lr * Gtot[l] / (Htot[l] + l2_intercept)
            continue

        for j in range(d):
            coef[l, j] = lr * beta[j]

    return coef


@njit(cache=True, parallel=True)
def _linear_predict(leaf, lin_feats, lin_coef, centers_std, Xb):
    """Per-sample output of a linear-leaf tree: intercept + slope . x_std.

    Parallel over samples. Each out[i] is independent, so this is bit-identical
    to the serial loop.
    """
    n = leaf.shape[0]
    k = lin_feats.shape[0]
    out = np.empty(n)
    for i in prange(n):
        l = leaf[i]
        s = lin_coef[l, 0]
        for j in range(k):
            f = lin_feats[j]
            v = centers_std[f, Xb[f, i]]
            if np.isfinite(v):
                s += lin_coef[l, 1 + j] * v
        out[i] = s
    return out


@njit(cache=True)
def _leaf_values_vec(leaf, grad, hess, coupling, n_leaves, l2, lr):
    """Vector (K-output) Newton leaf values on ONE shared leaf partition.

    ``grad``/``hess`` are the (n, K) per-class softmax gradient/hessian matrices.
    ``coupling`` is the softmax rank correction ((K-1)/K), applied to the hessian
    exactly as the per-class-tree path applies it before its scalar
    `_leaf_values` call.

    Column k of the result is what
    `_leaf_values(leaf, grad[:, k], hess[:, k] * coupling, ...)` returns — the
    same Newton formula, K columns per leaf instead of K separate trees. That
    equivalence is the oracle test in tests/test_vector_leaf.py.
    """
    n, K = grad.shape
    G = np.zeros((n_leaves, K))
    H = np.zeros((n_leaves, K))
    for i in range(n):
        l = leaf[i]
        for k in range(K):
            G[l, k] += grad[i, k]
            # Couple per element, BEFORE the sum, exactly like the scalar path's
            # `hess[:, k] * coupling` argument. Keeps the oracle equivalence
            # bit-exact: c*Σh ≠ Σ(c*h) in floats.
            H[l, k] += hess[i, k] * coupling

    values = np.zeros((n_leaves, K))
    for l in range(n_leaves):
        for k in range(K):
            if H[l, k] > 0.0:
                values[l, k] = -lr * G[l, k] / (H[l, k] + l2)
    return values


# ---------------------------------------------------------------------------
# Multi-quantile vector leaves (benchmarks/QUANTILE_PLAN.md).
#
# The quantile head's leaf value is not a Newton step but the exact empirical
# quantile of the leaf's residuals, one per tau. It is the same override the
# scalar MAE/Quantile path applies in `booster._correct_leaves`, widened to K
# channels.
#
# The channels are not constrained against one another here; ordering is imposed
# per row on delivered predictions (`booster.MultiQuantileBoosting`).
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _project_pinball(y, F, c, cdot, w, out):
    """Projected pinball gradient ``(grad @ c)`` without ever forming grad.

    Row i's channel-k gradient is ``1{y_i < F_ik} - tau_k``, so the projection
    onto the direction ``c`` is

        g_i = sum over {k : F_ik > y_i} of c_k, minus dot(c, taus)

    which is one accumulating pass over the row instead of a K-element dot
    product against a materialized (n, K) gradient. Skipping that matrix is why
    a K-level head costs about what a 1-level one costs per round, outside the
    leaf refit.

    ``w`` is a per-row weight array, ones when unweighted -- multiplying by
    exactly 1.0 is bit-exact, so the unweighted path is unchanged.

    Notes
    -----
    TRAP, and the reason this is a plain scan: an earlier version read the sum
    out of a precomputed suffix table by binary-searching the row's PIT rank,
    which is only valid while ``F_i`` is non-decreasing. Leaf vectors are no
    longer constrained against one another (ordering is imposed per row on
    delivered predictions instead), so a row's channels may cross during training
    and ``{k : F_ik > y_i}`` need not be a suffix. The scan below assumes no
    ordering at all, costing K comparisons rather than log K over the same cache
    lines.
    """
    n, K = F.shape
    for i in prange(n):
        yi = y[i]
        s = 0.0
        for k in range(K):
            if F[i, k] > yi:
                s += c[k]
        out[i] = (s - cdot) * w[i]


@njit(cache=True, parallel=True)
def _add_leaf_values(F, values, leaf):
    """``F += values[leaf]`` in place.

    Avoids the (n, K) gather that fancy indexing would allocate every round.
    """
    n, K = F.shape
    for i in prange(n):
        l = leaf[i]
        for k in range(K):
            F[i, k] += values[l, k]


@njit(cache=True)
def _lerp_np(a, b, t):
    """NumPy's `_lerp` (numpy/lib/function_base.py), branch included.

    The obvious ``a + (b - a) * t`` disagrees with `np.quantile` on roughly 7% of
    random inputs. Reproducing the ``t >= 0.5`` branch makes the leaf quantile
    bit-identical to the scalar path's `np.quantile` call, which is what lets
    tests/test_quantile_head.py assert exact equality.
    """
    diff = b - a
    out = a + diff * t
    if t >= 0.5:
        out = b - diff * (1.0 - t)
    return out


@njit(cache=True)
def _select_kth(buf, lo, hi, k):  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
    """Quickselect: reorder ``buf[lo:hi]`` so ``buf[k]`` is its (k-lo)-th
    smallest element, everything left of k no greater and everything right no
    smaller. Median-of-three pivot.

    O(m) expected versus O(m log m) for a full sort -- and the leaf refit runs
    this K times per leaf per round, so that difference is the head's dominant
    per-round cost.
    """
    left = lo
    right = hi - 1

    while left < right:
        # Median-of-three pivot.
        mid = (left + right) // 2
        x = buf[left]
        y = buf[mid]
        z = buf[right]
        if x > y:
            x, y = y, x
        if y > z:
            y = z
        if x > y:
            y = x
        pivot = y

        i = left
        j = right
        while i <= j:
            while buf[i] < pivot:
                i += 1
            while buf[j] > pivot:
                j -= 1
            if i <= j:
                buf[i], buf[j] = buf[j], buf[i]
                i += 1
                j -= 1

        # Keep narrowing onto the side that holds k.
        if k <= j:
            right = j
        elif k >= i:
            left = i
        else:
            return


@njit(cache=True)
def _quantile_slice(buf, lo, hi, alpha):
    """The ``alpha`` quantile of ``buf[lo:hi]``, matching `np.quantile`'s default
    ('linear') method bit for bit. Reorders the slice in place.
    """
    m = hi - lo
    if m <= 0:
        return 0.0
    if m == 1:
        return buf[lo]

    pos = alpha * (m - 1)
    j = int(np.floor(pos))
    if j >= m - 1:          # alpha == 1.0: the maximum, nothing to interpolate
        _select_kth(buf, lo, hi, hi - 1)
        return buf[hi - 1]

    frac = pos - j
    _select_kth(buf, lo, hi, lo + j)
    a = buf[lo + j]
    if frac == 0.0:
        return a
    # The slice is partitioned around lo+j, so the next order statistic is the
    # smallest element of the tail -- no second quickselect needed.
    b = buf[lo + j + 1]
    for t in range(lo + j + 2, hi):
        if buf[t] < b:
            b = buf[t]
    return _lerp_np(a, b, frac)


@njit(cache=True)
def _sort_pairs(buf, wbuf, lo, hi, wlo):
    """Sort ``buf[lo:hi]`` ascending, carrying ``wbuf`` along.

    Insertion sort: only the weighted path uses it, and leaves are small.
    """
    for i in range(1, hi - lo):
        v = buf[lo + i]
        wv = wbuf[wlo + i]
        j = i - 1
        while j >= 0 and buf[lo + j] > v:
            buf[lo + j + 1] = buf[lo + j]
            wbuf[wlo + j + 1] = wbuf[wlo + j]
            j -= 1
        buf[lo + j + 1] = v
        wbuf[wlo + j + 1] = wv


@njit(cache=True)
def _weighted_quantile_slice(buf, wbuf, lo, hi, alpha, wlo):
    """Nearest-rank weighted quantile of ``buf[lo:hi]``, reproducing
    `losses._weighted_quantile`.

    Sort by value, walk the cumulative weight, take the first element whose
    running total reaches ``alpha`` of the whole. Reorders both slices in place.
    """
    m = hi - lo
    if m <= 0:
        return 0.0

    _sort_pairs(buf, wbuf, lo, hi, wlo)

    total = 0.0
    for i in range(m):
        total += wbuf[wlo + i]

    target = total * alpha
    run = 0.0
    for i in range(m):
        run += wbuf[wlo + i]
        if run >= target:       # np.searchsorted(..., side='left')
            return buf[lo + i]
    return buf[hi - 1]


@njit(cache=True)
def _leaf_row_index(leaf, n_leaves):
    """Group rows by leaf into contiguous, disjoint slices.

    Returns ``(off, rows)`` where leaf l owns ``rows[off[l]:off[l+1]]``, rows
    listed in ascending order within each leaf -- the same grouping
    `booster._correct_leaves` gets from a stable argsort.

    Disjointness is what makes the `prange` over leaves race-free: each leaf
    writes only its own slice of the shared scratch buffer.
    """
    n = leaf.shape[0]
    off = np.zeros(n_leaves + 1, dtype=np.int64)
    for i in range(n):
        off[leaf[i] + 1] += 1
    for l in range(n_leaves):
        off[l + 1] += off[l]

    fill = off.copy()
    rows = np.empty(n, dtype=np.int64)
    for i in range(n):
        l = leaf[i]
        rows[fill[l]] = i
        fill[l] += 1
    return off, rows


@njit(cache=True, parallel=True)
def _leaf_quantiles_vec(leaf, y, F, taus, n_leaves, lr):
    """Per-leaf, per-tau empirical quantile of the residuals ``y - F[:, k]``,
    scaled by the learning rate.

    The (n_leaves, K) analogue of `booster._correct_leaves`: the tree structure
    was chosen by the projected gradient, this sets the step. Column k is
    bit-identical to what the scalar quantile path computes for the same
    partition at ``alpha = taus[k]`` -- the oracle test in
    tests/test_quantile_head.py.

    Each channel is an independent step and nothing constrains them against one
    another here; ordering is imposed per row on delivered predictions instead
    (see `booster.MultiQuantileBoosting`). `lr` multiplies from the outside,
    matching `_correct_leaves`.
    """
    n = leaf.shape[0]
    K = taus.shape[0]
    off, rows = _leaf_row_index(leaf, n_leaves)
    values = np.zeros((n_leaves, K))

    # One scratch run per (leaf, channel). Costs an (n, K) buffer -- the same
    # footprint as the score matrix itself -- and buys load balance: a real
    # oblivious tree's leaves are badly uneven, so a prange over leaves alone
    # leaves most threads idle waiting on the biggest one. Leaf l owns
    # buf[off[l]*K : off[l+1]*K], so the runs stay disjoint.
    buf = np.empty(n * K, dtype=np.float64)

    for t in prange(n_leaves * K):
        l = t // K
        k = t - l * K
        lo = off[l]
        m = off[l + 1] - lo
        if m > 0:
            # Forward strided scan of one F column, contiguous write. The leaf's
            # rows are ascending so the prefetcher tracks it. Gathering
            # channel-major in one pass was measured SLOWER -- it trades these
            # reads for a transposing write that costs more than it saves.
            st = lo * K + k * m
            for j in range(m):
                i = rows[lo + j]
                buf[st + j] = y[i] - F[i, k]
            values[l, k] = lr * _quantile_slice(buf, st, st + m, taus[k])
    return values


@njit(cache=True)
def _leaf_quantiles_vec_serial(leaf, y, F, taus, n_leaves, lr):
    """Serial twin of `_leaf_quantiles_vec` for small n, where the parallel
    fork/join costs more than the pass.

    Every write is to a disjoint slice, so the two are bit-identical; the booster
    dispatches on `_SMALL_N`.
    """
    n = leaf.shape[0]
    K = taus.shape[0]
    off, rows = _leaf_row_index(leaf, n_leaves)
    values = np.zeros((n_leaves, K))
    buf = np.empty(n, dtype=np.float64)

    for l in range(n_leaves):
        lo = off[l]
        hi = off[l + 1]
        if hi > lo:
            # One forward strided scan of F per channel. The leaf's rows are
            # ascending, so the prefetcher tracks it, and the leaf's slice of F
            # stays resident across the K passes. Gathering channel-major
            # instead was measured SLOWER: it trades these reads for a
            # transposing write, which costs more than it saves.
            for k in range(K):
                for j in range(lo, hi):
                    i = rows[j]
                    buf[j] = y[i] - F[i, k]
                values[l, k] = lr * _quantile_slice(buf, lo, hi, taus[k])
    return values


@njit(cache=True, parallel=True)
def _leaf_quantiles_vec_w(leaf, y, F, w, taus, n_leaves, lr):
    """Weighted `_leaf_quantiles_vec`.

    Reproduces `losses._weighted_quantile` (nearest rank on cumulative weight),
    the rule the scalar weighted quantile path already uses. Rows carrying zero
    weight -- MVS drops, or a user's zero sample weights -- contribute nothing,
    so the leaf value sees exactly the rows the split search saw.
    """
    n = leaf.shape[0]
    K = taus.shape[0]
    off, rows = _leaf_row_index(leaf, n_leaves)
    values = np.zeros((n_leaves, K))

    # Per-(leaf, channel) runs, as in the unweighted twin. The weights need their
    # own copy per run because the paired sort reorders them.
    buf = np.empty(n * K, dtype=np.float64)
    wbuf = np.empty(n * K, dtype=np.float64)

    for t in prange(n_leaves * K):
        l = t // K
        k = t - l * K
        lo = off[l]
        m = off[l + 1] - lo
        if m > 0:
            st = lo * K + k * m
            for j in range(m):
                i = rows[lo + j]
                buf[st + j] = y[i] - F[i, k]
                wbuf[st + j] = w[i]
            values[l, k] = lr * _weighted_quantile_slice(
                buf, wbuf, st, st + m, taus[k], st)
    return values


@njit(cache=True)
def _leaf_quantiles_vec_w_serial(leaf, y, F, w, taus, n_leaves, lr):
    """Serial twin of `_leaf_quantiles_vec_w` (see `_leaf_quantiles_vec_serial`)."""
    n = leaf.shape[0]
    K = taus.shape[0]
    off, rows = _leaf_row_index(leaf, n_leaves)
    values = np.zeros((n_leaves, K))
    buf = np.empty(n, dtype=np.float64)
    wbuf = np.empty(n, dtype=np.float64)

    for l in range(n_leaves):
        lo = off[l]
        hi = off[l + 1]
        if hi > lo:
            for k in range(K):
                for j in range(lo, hi):
                    i = rows[j]
                    buf[j] = y[i] - F[i, k]
                    wbuf[j] = w[i]
                values[l, k] = lr * _weighted_quantile_slice(
                    buf, wbuf, lo, hi, taus[k], lo)
    return values


@njit(cache=True)
def _loo_leaf_step(leaf, grad, hess, n_leaves, l2, lr):
    """Leave-one-out training step for every row, fused into two passes.

    The first pass scatters per-leaf grad/hess totals. The second gathers each
    row's totals, removes the row's own contribution, and forms the shrunk
    Newton step. Replaces two np.bincount calls plus several NumPy temporaries
    with one scatter and one compute loop over `leaf`.
    """
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    n = leaf.shape[0]
    for i in range(n):
        l = leaf[i]
        G[l] += grad[i]
        H[l] += hess[i]

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        l = leaf[i]
        denom = H[l] - hess[i]
        if denom < 0.0:
            denom = 0.0
        if denom + l2 <= 0.0:
            # Singleton leaf (or all co-leaf rows subsampled away) with
            # l2_leaf_reg=0: no curvature left once the row's own contribution
            # is removed, so there is no leave-one-out step to take. Mirrors the
            # H[l] > 0 guard in _leaf_values.
            out[i] = 0.0
        else:
            out[i] = -lr * (G[l] - grad[i]) / (denom + l2)
    return out


@njit(cache=True)
def _predict_tree(Xb, splits_feat, splits_thr, values):
    """Route each sample to its leaf and return that leaf's value."""
    leaf = _assign_leaves(Xb, splits_feat, splits_thr)
    out = np.empty(Xb.shape[1], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


@njit(cache=True, parallel=True)
def _predict_forest(Xb, feats, thrs, depths, vals, voff, init):
    """Sum a whole ensemble of oblivious trees in one parallel pass over samples.

    The trees arrive packed into flat arrays (see `pack_forest`): `feats`/`thrs`
    are (n_trees, max_depth) split tables, `depths[t]` is the real depth of tree
    t, and `vals`/`voff` form a ragged leaf-value table where tree t's values
    live at vals[voff[t] : voff[t+1]].

    Parallelizing over samples rather than trees lets each sample load its
    handful of feature bins once and keep them hot while walking every tree. The
    per-sample accumulation runs init + tree0 + tree1 + ... in tree order,
    matching the serial `F += tree.predict(Xb)` loop bit for bit.
    """
    n = Xb.shape[1]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        acc = init
        for t in range(n_trees):
            # A depth-0 tree found no legal split; like ObliviousTree.predict it
            # contributes nothing (its lone leaf value is never applied).
            if depths[t] == 0:
                continue

            leaf = 0
            for d in range(depths[t]):
                if Xb[feats[t, d], i] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            acc += vals[voff[t] + leaf]
        out[i] = acc
    return out


@njit(cache=True, parallel=True)
def _predict_forest_rm(Xb, feats, thrs, depths, vals, voff, init):
    """`_predict_forest` for a row-major (n_samples, n_features) binned matrix.

    Predict-time binning produces row-major output. Consuming it directly keeps
    each sample's feature bins in one or two cache lines for the whole forest
    walk, and skips the feature-major transpose copy entirely. Same arithmetic
    and per-sample accumulation order as `_predict_forest`, so the two are
    bit-identical.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        acc = init
        for t in range(n_trees):
            # A depth-0 tree found no legal split; like ObliviousTree.predict it
            # contributes nothing (its lone leaf value is never applied).
            if depths[t] == 0:
                continue

            leaf = 0
            for d in range(depths[t]):
                if Xb[i, feats[t, d]] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            acc += vals[voff[t] + leaf]
        out[i] = acc
    return out


@njit(cache=True)
def _predict_forest_rm_serial(Xb, feats, thrs, depths, vals, voff, init):
    """Serial twin of `_predict_forest_rm` for tiny batches.

    The OpenMP fork/join (~20us on 12 threads) exceeds the whole 1-row walk, and
    the parallel kernel only overtakes serial around n~5. Bit-identical
    (independent per-row writes); the booster dispatches on
    `binning._SERIAL_PREDICT_N`.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc = init
        for t in range(n_trees):
            if depths[t] == 0:
                continue
            leaf = 0
            for d in range(depths[t]):
                if Xb[i, feats[t, d]] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            acc += vals[voff[t] + leaf]
        out[i] = acc
    return out


def pack_forest(trees, max_depth):
    """Flatten a list of ObliviousTrees into the arrays `_predict_forest` wants.

    Returns (feats, thrs, depths, vals, voff). The booster caches this after fit
    so repeated predict calls skip the rebuild.
    """
    n_trees = len(trees)
    feats = np.zeros((n_trees, max_depth), dtype=np.int64)
    thrs = np.zeros((n_trees, max_depth), dtype=np.int64)
    depths = np.empty(n_trees, dtype=np.int64)
    voff = np.empty(n_trees + 1, dtype=np.int64)
    voff[0] = 0

    for t, tree in enumerate(trees):
        d = tree.depth
        depths[t] = d
        feats[t, :d] = tree.splits_feat
        thrs[t, :d] = tree.splits_thr
        voff[t + 1] = voff[t] + tree.values.shape[0]

    vals = np.empty(voff[-1], dtype=np.float64)
    for t, tree in enumerate(trees):
        vals[voff[t]:voff[t + 1]] = tree.values
    return feats, thrs, depths, vals, voff


def pack_forest_vec(trees, max_depth):
    """Flatten a forest of vector-leaf (K-output) oblivious trees for
    `_predict_forest_vec_rm`.

    Same layout as `pack_forest` except that tree t's leaf-value block is
    leaf-major (n_leaves, K) flattened: class k of leaf l lives at
    vals[voff[t] + l*K + k]. Returns (feats, thrs, depths, vals, voff, K).
    """
    n_trees = len(trees)
    K = trees[0].values.shape[1]
    feats = np.zeros((n_trees, max_depth), dtype=np.int64)
    thrs = np.zeros((n_trees, max_depth), dtype=np.int64)
    depths = np.empty(n_trees, dtype=np.int64)
    voff = np.empty(n_trees + 1, dtype=np.int64)
    voff[0] = 0

    for t, tree in enumerate(trees):
        d = tree.depth
        depths[t] = d
        feats[t, :d] = tree.splits_feat
        thrs[t, :d] = tree.splits_thr
        voff[t + 1] = voff[t] + tree.values.shape[0] * K

    vals = np.empty(voff[-1], dtype=np.float64)
    for t, tree in enumerate(trees):
        vals[voff[t]:voff[t + 1]] = tree.values.reshape(-1)
    return feats, thrs, depths, vals, voff, K


@njit(cache=True, parallel=True)
def _predict_forest_vec_rm(Xb, feats, thrs, depths, vals, voff, K, init):
    """Sum a forest of vector-leaf oblivious trees in one parallel pass over a
    row-major (n_samples, n_features) binned matrix — the K-output analogue of
    `_predict_forest_rm`.

    One tree walk serves all K classes; the per-class-forest path walked the same
    rows K times. Per-sample accumulation runs init + tree0 + tree1 + ... in tree
    order per class, matching the fit loop's `F += tree.values[leaf]` bit for bit.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty((n, K), dtype=np.float64)

    for i in prange(n):
        for k in range(K):
            out[i, k] = init[k]

        for t in range(n_trees):
            # A depth-0 tree found no legal split; it contributes nothing.
            if depths[t] == 0:
                continue

            leaf = 0
            for d in range(depths[t]):
                if Xb[i, feats[t, d]] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            base = voff[t] + leaf * K
            for k in range(K):
                out[i, k] += vals[base + k]
    return out


@njit(cache=True)
def _predict_forest_vec_rm_serial(Xb, feats, thrs, depths, vals, voff, K,
                                  init):
    """Serial twin of `_predict_forest_vec_rm` for tiny batches — see
    `_predict_forest_rm_serial`.

    Bit-identical (independent per-row writes); the booster dispatches on
    `binning._SERIAL_PREDICT_N`.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty((n, K), dtype=np.float64)
    for i in range(n):
        for k in range(K):
            out[i, k] = init[k]
        for t in range(n_trees):
            if depths[t] == 0:
                continue
            leaf = 0
            for d in range(depths[t]):
                if Xb[i, feats[t, d]] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            base = voff[t] + leaf * K
            for k in range(K):
                out[i, k] += vals[base + k]
    return out


def pack_forest_linear(trees, max_depth):
    """Flatten a forest of (possibly) linear-leaf trees for `_predict_forest_linear`.

    A constant-leaf tree is just a linear tree with k=0 features -- its coef
    block is the leaf intercepts -- so one packed layout and one kernel serve
    both. Per tree: `lin_k[t]` linear features at
    `lin_feat_idx[featoff[t]:featoff[t+1]]`, and a leaf-major coef block at
    `coef[coefoff[t]:coefoff[t+1]]` of shape (n_leaves, 1 + lin_k[t]) flattened,
    column 0 being the intercept.
    """
    n_trees = len(trees)
    feats = np.zeros((n_trees, max_depth), dtype=np.int64)
    thrs = np.zeros((n_trees, max_depth), dtype=np.int64)
    depths = np.empty(n_trees, dtype=np.int64)
    lin_k = np.empty(n_trees, dtype=np.int64)
    featoff = np.empty(n_trees + 1, dtype=np.int64)
    coefoff = np.empty(n_trees + 1, dtype=np.int64)
    featoff[0] = 0
    coefoff[0] = 0

    for t, tree in enumerate(trees):
        d = tree.depth
        depths[t] = d
        feats[t, :d] = tree.splits_feat
        thrs[t, :d] = tree.splits_thr
        n_leaves = (1 << d) if d > 0 else 1
        k = tree.lin_feats.shape[0] if tree.lin_coef is not None else 0
        lin_k[t] = k
        featoff[t + 1] = featoff[t] + k
        coefoff[t + 1] = coefoff[t] + n_leaves * (1 + k)

    lin_feat_idx = np.empty(featoff[-1], dtype=np.int64)
    coef = np.empty(coefoff[-1], dtype=np.float64)

    for t, tree in enumerate(trees):
        if lin_k[t] > 0:
            lin_feat_idx[featoff[t]:featoff[t + 1]] = tree.lin_feats
            coef[coefoff[t]:coefoff[t + 1]] = tree.lin_coef.reshape(-1)
        else:
            coef[coefoff[t]:coefoff[t + 1]] = tree.values
    return feats, thrs, depths, lin_k, featoff, lin_feat_idx, coefoff, coef


@njit(cache=True, parallel=True)
def _predict_forest_linear(Xb, feats, thrs, depths, lin_k, featoff,
                           lin_feat_idx, coefoff, coef, centers_std, init):
    """Sum a forest of linear-leaf (or constant, k=0) oblivious trees in one
    parallel pass over samples -- the linear-leaf analogue of `_predict_forest`.

    Each leaf contributes intercept + sum_j slope_j * centers_std[feat_j, bin],
    matching `_linear_predict`/`ObliviousTree.predict`, so the fused path agrees
    with the per-tree path bit for bit (same accumulation order).
    """
    n = Xb.shape[1]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        acc = init
        for t in range(n_trees):
            d = depths[t]
            if d == 0:
                continue

            leaf = 0
            for dd in range(d):
                if Xb[feats[t, dd], i] > thrs[t, dd]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2

            k = lin_k[t]
            row = coefoff[t] + leaf * (1 + k)
            val = coef[row]                      # intercept
            fb = featoff[t]
            for j in range(k):
                f = lin_feat_idx[fb + j]
                v = centers_std[f, Xb[f, i]]
                if np.isfinite(v):
                    val += coef[row + 1 + j] * v
            acc += val
        out[i] = acc
    return out


@njit(cache=True, parallel=True)
def _predict_forest_linear_rm(Xb, feats, thrs, depths, lin_k, featoff,
                              lin_feat_idx, coefoff, coef, centers_std, init):
    """`_predict_forest_linear` for a row-major (n_samples, n_features) binned
    matrix — see `_predict_forest_rm` for why.

    Bit-identical to the feature-major kernel: same arithmetic, same
    accumulation order.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        acc = init
        for t in range(n_trees):
            d = depths[t]
            if d == 0:
                continue

            leaf = 0
            for dd in range(d):
                if Xb[i, feats[t, dd]] > thrs[t, dd]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2

            k = lin_k[t]
            row = coefoff[t] + leaf * (1 + k)
            val = coef[row]                      # intercept
            fb = featoff[t]
            for j in range(k):
                f = lin_feat_idx[fb + j]
                v = centers_std[f, Xb[i, f]]
                if np.isfinite(v):
                    val += coef[row + 1 + j] * v
            acc += val
        out[i] = acc
    return out


@njit(cache=True)
def _predict_forest_linear_rm_serial(Xb, feats, thrs, depths, lin_k, featoff,
                                     lin_feat_idx, coefoff, coef, centers_std,
                                     init):
    """Serial twin of `_predict_forest_linear_rm` for tiny batches — see
    `_predict_forest_rm_serial`.
    """
    n = Xb.shape[0]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        acc = init
        for t in range(n_trees):
            d = depths[t]
            if d == 0:
                continue

            leaf = 0
            for dd in range(d):
                if Xb[i, feats[t, dd]] > thrs[t, dd]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2

            k = lin_k[t]
            row = coefoff[t] + leaf * (1 + k)
            val = coef[row]                      # intercept
            fb = featoff[t]
            for j in range(k):
                f = lin_feat_idx[fb + j]
                v = centers_std[f, Xb[i, f]]
                if np.isfinite(v):
                    val += coef[row + 1 + j] * v
            acc += val
        out[i] = acc
    return out


@njit(cache=True, parallel=True)
def _shap_forest_linear(Xb, Rb, feats, thrs, depths, lin_k, featoff,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                        lin_feat_idx, coefoff, coef, centers_std,
                        feat_orig, n_orig, fact):
    """Exact interventional TreeSHAP for a forest of oblivious (linear-leaf or
    constant, k=0) trees, returned in the user's ORIGINAL feature space.

    For each instance x (a column of Xb) and background reference r (a column of
    Rb), the per-tree Shapley values come from exact enumeration over subsets of
    the distinct ORIGINAL features the tree uses. That is tractable precisely
    because the trees are oblivious: a depth-D tree touches at most D distinct
    features, so the coalition game has at most D players (<=2**D subsets), not
    one per input column. A feature in coalition S takes its value from x and the
    rest from r; the leaf -- and any linear-leaf slope -- is evaluated under that
    mix, so linear leaves are explained faithfully rather than ignored.

    Contributions are averaged over the background and summed over trees, giving
    every instance the Shapley-efficiency identity (to float tolerance)

        sum_orig phi[i, orig] == predict_trees(x_i) - mean_r predict_trees(r).

    Two internal columns mapping to the same original feature (categorical
    combos, multi-target encodings) count as ONE player, so the attribution lands
    directly in input-feature space. `fact[s]` is s!, precomputed up to depth.
    Parallel over instances; each thread owns a disjoint row of `phi`.
    """
    n = Xb.shape[1]
    nbg = Rb.shape[1]
    n_trees = feats.shape[0]
    phi = np.zeros((n, n_orig))
    inv_nbg = 1.0 / nbg

    for i in prange(n):
        for t in range(n_trees):
            d = depths[t]
            if d == 0:
                continue

            k = lin_k[t]
            fb = featoff[t]
            cb = coefoff[t]

            # The distinct original features this tree uses are the coalition
            # players U. level_u[dd] is the U-slot of level dd's feature, so a
            # feature reused across levels moves as one player.
            U = np.empty(d, dtype=np.int64)
            level_u = np.empty(d, dtype=np.int64)
            u = 0
            for dd in range(d):
                o = feat_orig[feats[t, dd]]
                idx = -1
                for q in range(u):
                    if U[q] == o:
                        idx = q
                        break
                if idx < 0:
                    U[u] = o
                    idx = u
                    u += 1
                level_u[dd] = idx

            lin_u = np.empty(k, dtype=np.int64)
            for j in range(k):
                o = feat_orig[lin_feat_idx[fb + j]]
                for q in range(u):
                    if U[q] == o:
                        lin_u[j] = q
                        break

            nsub = 1 << u

            # x-side level bits and standardized linear values; both are
            # independent of the reference, so they are computed once.
            xbit = np.empty(d, dtype=np.int64)
            for dd in range(d):
                xbit[dd] = 1 if Xb[feats[t, dd], i] > thrs[t, dd] else 0

            xval = np.empty(k)
            for j in range(k):
                f = lin_feat_idx[fb + j]
                v = centers_std[f, Xb[f, i]]
                xval[j] = v if np.isfinite(v) else 0.0

            fval = np.empty(nsub)
            rbit = np.empty(d, dtype=np.int64)
            rval = np.empty(k)

            for b in range(nbg):
                for dd in range(d):
                    rbit[dd] = 1 if Rb[feats[t, dd], b] > thrs[t, dd] else 0
                for j in range(k):
                    f = lin_feat_idx[fb + j]
                    vv = centers_std[f, Rb[f, b]]
                    rval[j] = vv if np.isfinite(vv) else 0.0

                # Output of every coalition: bits and linear values follow x
                # inside S, r outside it.
                for mask in range(nsub):
                    leaf = 0
                    for dd in range(d):
                        if (mask >> level_u[dd]) & 1:
                            bit = xbit[dd]
                        else:
                            bit = rbit[dd]
                        leaf = leaf * 2 + bit
                    row = cb + leaf * (1 + k)
                    val = coef[row]
                    for j in range(k):
                        vv = xval[j] if (mask >> lin_u[j]) & 1 else rval[j]
                        val += coef[row + 1 + j] * vv
                    fval[mask] = val

                # Shapley value of each player: the weighted marginal over
                # every coalition that excludes it.
                for ui in range(u):
                    bit_ui = 1 << ui
                    contrib = 0.0
                    for mask in range(nsub):
                        if (mask >> ui) & 1:
                            continue
                        s = 0
                        mm = mask
                        while mm:
                            s += mm & 1
                            mm >>= 1
                        w = fact[s] * fact[u - s - 1] / fact[u]
                        contrib += w * (fval[mask | bit_ui] - fval[mask])

                    phi[i, U[ui]] += contrib * inv_nbg

    return phi


class ObliviousTree:
    """A single symmetric tree. Stores its splits and leaf values.

    `apply` and `predict` take a feature-major binned matrix
    (n_features, n_samples) -- the same layout the builder consumes.
    """

    __slots__ = ("splits_feat", "splits_thr", "values", "gains", "depth",
                 "lin_feats", "lin_coef", "centers_std")

    def __init__(self, splits_feat, splits_thr, values, gains=None,
                 lin_feats=None, lin_coef=None, centers_std=None):
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.values = values
        self.gains = gains if gains is not None else np.zeros(len(splits_feat))
        self.depth = len(splits_feat)

        # Optional linear-leaf models (None => plain constant leaves).
        self.lin_feats = lin_feats
        self.lin_coef = lin_coef
        self.centers_std = centers_std

    def apply(self, Xb):
        """Return the leaf index of each sample."""
        if self.depth == 0:
            return np.zeros(Xb.shape[1], dtype=np.int64)

        n = Xb.shape[1]
        if n > _ASSIGN_PAR_N:
            # Level-by-level parallel descend; each Xb[f] is a contiguous
            # feature row. Bit-identical to the serial fused walk.
            leaf = np.zeros(n, dtype=np.int64)
            for d in range(self.depth):
                _descend_leaves(leaf, Xb[self.splits_feat[d]],
                                self.splits_thr[d])
            return leaf

        return _assign_leaves(Xb, self.splits_feat, self.splits_thr)

    def predict(self, Xb):
        if self.depth == 0:
            return np.zeros(Xb.shape[1], dtype=np.float64)

        if self.lin_coef is not None:
            leaf = self.apply(Xb)
            return _linear_predict(leaf, self.lin_feats, self.lin_coef,
                                   self.centers_std, Xb)

        if Xb.shape[1] > _ASSIGN_PAR_N:
            return self.values[self.apply(Xb)]

        return _predict_tree(Xb, self.splits_feat, self.splits_thr, self.values)


def replay_oblivious_tree(donor, Xb, grad, hess, l2, lr, linear_leaves=False,
                          centers_std=None, linear_lambda=1.0):
    """Refit one tree's LEAF VALUES on new data, reusing ``donor``'s splits.

    The split search is what makes growing expensive -- a histogram pass over
    every feature at every level. Given a structure already known to be good, the
    leaf values follow from one leaf assignment plus a scatter-add of the
    gradients: O(n * depth) instead of O(n * features * depth).

    Used by the full-data refit (``refit_full="replay"``). The early-stopping
    winner's structures are replayed round by round against gradients computed on
    ALL rows, so the held-out validation rows reach the leaf estimates without
    re-paying for the split search. Returns ``(tree, leaf)``, matching
    ``build_oblivious_tree``.

    ``Xb`` must be binned by the DONOR's binner: ``splits_thr`` are bin indices,
    so a re-fitted binner would silently move every threshold.
    """
    sf, st = donor.splits_feat, donor.splits_thr
    if len(sf) == 0:                       # degenerate donor; caller stops
        return (ObliviousTree(sf, st, np.zeros(1), np.zeros(0)),
                np.zeros(Xb.shape[1], dtype=np.int64))

    leaf = _assign_leaves(Xb, sf, st)
    n_leaves = 1 << len(sf)
    values = _leaf_values(leaf, grad, hess, n_leaves, l2, lr)

    # Linear leaves are refit too: same features the donor split on, new
    # coefficients from the full-data gradients.
    lin_feats = lin_coef = None
    if (linear_leaves and centers_std is not None
            and donor.lin_feats is not None):
        lin_feats = donor.lin_feats
        lin_coef = _linear_leaf_fit(leaf, grad, hess, n_leaves, lin_feats,
                                    centers_std, Xb, l2, linear_lambda, lr)

    # Carry the donor's split gains: the structure IS the donor's, so its gains
    # are the honest attribution for it. Zeroing them would leave
    # ``feature_importances_`` summing only the trailing grown trees.
    tree = ObliviousTree(sf, st, values, donor.gains,
                         lin_feats=lin_feats, lin_coef=lin_coef,
                         centers_std=centers_std if lin_coef is not None
                         else None)
    return tree, leaf


@njit(cache=True, parallel=True)
def _build_split_descend_vec(Xb, grad, rw, leaf, n_active, hist, histw,  # noqa: C901 -- frozen numba kernel, see benchmarks/REFACTOR_AUDIT.md
                             feat_mask, n_bins_per_feature, l2,
                             min_child_weight, min_gain):
    """EXACT multi-channel split search: the K-deep twin of
    `_build_split_descend`, used only by ``exact_splits=True``.

    Scores the true summed-across-channel gain. Because the multi-quantile
    hessian is 1 in every channel, every channel shares one weight total per
    node, and the summed gain collapses to a norm:

        sum_k [Gl_k²/(Hl+l2) + Gr_k²/(Hr+l2) - Gt_k²/(Ht+l2)]
          = ‖Gl‖²/(Hl+l2) + ‖Gr‖²/(Hr+l2) - ‖Gt‖²/(Ht+l2)

    which is basis-free. That identity is what makes the default path's single
    projection an honest rank-1 truncation of this quantity rather than a
    different algorithm. It costs K histogram channels per feature instead of
    one, which is exactly why it is the reference arm and not the default.

    Simplified relative to the scalar kernel: no occupancy list and no small-n
    path (level d always scans leaves 0..n_active-1). Both are speed
    optimizations, and this kernel is not on the fast path.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    K = grad.shape[1]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue

        # Zero this feature's leaf rows, then scatter the samples in.
        nb = n_bins_per_feature[f]
        for l in range(n_active):
            for b in range(nb):
                histw[f, l, b] = 0.0
                for k in range(K):
                    hist[f, l, b, k] = 0.0

        Xf = Xb[f]
        for i in range(n_samples):
            l = leaf[i]
            b = Xf[i]
            histw[f, l, b] += rw[i]
            for k in range(K):
                hist[f, l, b, k] += grad[i, k]

        # Score every threshold on the summed-across-channel gain.
        gain = np.zeros(max_bins)
        legal = np.ones(max_bins, dtype=np.uint8)
        gt = np.empty(K)
        gl = np.empty(K)

        for l in range(n_active):
            ht = 0.0
            for k in range(K):
                gt[k] = 0.0
            for b in range(nb):
                ht += histw[f, l, b]
                for k in range(K):
                    gt[k] += hist[f, l, b, k]

            if ht <= 0.0:
                continue

            sq = 0.0
            for k in range(K):
                sq += gt[k] * gt[k]
            par = sq / (ht + l2)

            hl = 0.0
            for k in range(K):
                gl[k] = 0.0

            for t in range(nb - 1):
                hl += histw[f, l, t]
                for k in range(K):
                    gl[k] += hist[f, l, t, k]
                hr = ht - hl
                if (hl > 0.0 and hl < min_child_weight) or \
                   (hr > 0.0 and hr < min_child_weight):
                    legal[t] = 0
                else:
                    sl = 0.0
                    sr = 0.0
                    for k in range(K):
                        gr = gt[k] - gl[k]
                        sl += gl[k] * gl[k]
                        sr += gr * gr
                    gain[t] += sl / (hl + l2) + sr / (hr + l2) - par

        best_g = -np.inf
        best_t = -1
        for t in range(nb - 1):
            if legal[t] and gain[t] > best_g:
                best_g = gain[t]
                best_t = t

        feat_gain[f] = best_g
        feat_thr[f] = best_t

    # Best feature overall, then descend one level if the split is usable.
    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    best_t = feat_thr[best_f]

    if best_t >= 0 and best_gain > min_gain:
        Xf = Xb[best_f]
        for i in prange(n_samples):
            leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > best_t else 0)

    return best_f, best_t, best_gain


def alloc_exact_hist(n_features, max_depth, n_bins_per_feature, K):
    """Buffers for `build_oblivious_tree_exact`, allocated once per fit.

    Sized (n_features, 2**max_depth, max_bins, K) plus a shared weight histogram.
    That is K times the default path's footprint -- at depth 6, 50 features, 128
    bins and K=19 it is roughly 62 MB. The honest price of the exact gain, and
    another reason it is not the default.
    """
    max_bins = n_features and int(n_bins_per_feature.max())
    max_leaves = 1 << max_depth
    return (np.zeros((n_features, max_leaves, max_bins, K)),
            np.zeros((n_features, max_leaves, max_bins)))


def build_oblivious_tree_exact(Xb, grad, n_bins_per_feature, max_depth, l2,
                               min_gain=1e-8, feature_mask=None,
                               min_child_weight=1.0, row_weight=None,
                               hist_buffers=None):
    """Grow one oblivious tree on the EXACT summed-across-channel gain.

    The reference arm for the multi-quantile head's split search (see
    `_build_split_descend_vec`). Returns ``(tree, train_leaf)`` like
    `build_oblivious_tree`, but leaf values are left as zeros: the quantile
    booster overwrites them with the exact per-tau residual quantiles either way,
    so computing a Newton step here would be wasted work.

    ``row_weight`` is the per-row weight (sample weights times any MVS importance
    weight); None means uniform. It plays the hessian's role, one total shared by
    every channel.
    """
    n_features, n_samples = Xb.shape
    if feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)

    K = grad.shape[1]
    if hist_buffers is None:
        hist, histw = alloc_exact_hist(n_features, max_depth,
                                       n_bins_per_feature, K)
    else:
        hist, histw = hist_buffers

    rw = np.ones(n_samples) if row_weight is None else row_weight
    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(n_samples, dtype=np.int64)
    n_active = 1

    for d in range(max_depth):
        f, t, gain = _build_split_descend_vec(
            Xb, grad, rw, leaf, n_active, hist, histw, feature_mask,
            n_bins_per_feature, l2, min_child_weight, min_gain)
        if gain <= min_gain or t < 0:
            break

        splits_feat.append(f)
        splits_thr.append(t)
        splits_gain.append(gain)
        n_active <<= 1

    sf = np.array(splits_feat, dtype=np.int64)
    st = np.array(splits_thr, dtype=np.int64)
    values = np.zeros(1 << len(splits_feat))
    tree = ObliviousTree(sf, st, values,
                         np.array(splits_gain, dtype=np.float64))
    return tree, leaf


def build_oblivious_tree(Xb, grad, hess, n_bins_per_feature,  # noqa: C901 -- complexity baseline, removed in stage 7
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         linear_leaves=False, centers_std=None, is_numeric=None,
                         linear_lambda=1.0, quantize=False, qbuf=None,
                         qseed=0):
    """Grow one oblivious tree level by level.

    Returns (tree, train_leaf), where train_leaf is the tree's leaf index for
    every training sample.

    Parameters
    ----------
    Xb : feature-major binned matrix (n_features, n_samples).

    feature_mask : 0/1 array over features; 0 disables a feature for this tree
        (column subsampling). None means every feature is eligible.

    min_child_weight : minimum hessian mass each side of a split must retain in
        every non-empty leaf. Growth stops once no legal split remains, which
        prevents sparse-leaf overfitting at higher depth.

    hist_buffers : buffer reused across trees to avoid per-level allocation.
        Interleaved (n_features, 2**max_depth, max_bins, 2) float64, or with
        quantize=True int64 (n_features, 2**max_depth, max_bins). If None it is
        allocated here, for one-off calls and tests.

    linear_leaves : when True, attach a per-leaf ridge linear model over the
        tree's numeric split features. Requires `centers_std` and `is_numeric`;
        `linear_lambda` is the slope penalty. Low-count leaves fall back to the
        constant Newton value. The split search is unaffected.

    quantize : run the SPLIT SEARCH on packed-int64 quantized grad/hess
        (QUANT_PLAN.md) — one integer read-modify-write per scatter, half the
        histogram footprint. Leaf values and the linear-leaf ridge still use the
        original float64 grad/hess, so quantization noise touches only the
        structure choice. `qbuf` is an optional reusable int64 (n_samples)
        scratch for the packed values. `qseed` seeds the stochastic rounding;
        pass a fresh draw per tree for decorrelated rounding noise.
    """
    n_features, n_samples = Xb.shape
    max_bins = n_features and int(n_bins_per_feature.max())
    if feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)

    if hist_buffers is None:
        hist = (np.zeros((n_features, 1 << max_depth, max_bins),
                         dtype=np.int64) if quantize
                else np.zeros((n_features, 1 << max_depth, max_bins, 2)))
    else:
        hist = hist_buffers

    if quantize:
        # qmax keeps every packed cell and prefix sum overflow-safe (see the
        # _QMAX_CAP comment); the scales map the observed grad/hess range onto
        # [-qmax, qmax] and [0, qmax]. All-zero grad or hess degenerates to
        # qg/qh = 0, which yields zero gains — the same no-split outcome as the
        # float kernel.
        qmax = min(_QMAX_CAP, (2 ** 31 - 1) // max(n_samples, 1))
        gmax, hmax = _gh_absmax(grad, hess)
        inv_dg = qmax / gmax if gmax > 0.0 else 0.0
        inv_dh = qmax / hmax if hmax > 0.0 else 0.0
        dg = gmax / qmax if gmax > 0.0 else 0.0
        dh = hmax / qmax if hmax > 0.0 else 0.0
        if qbuf is None:
            qbuf = np.empty(n_samples, dtype=np.int64)

        _quantize_pack(grad, hess, inv_dg, inv_dh, np.int64(qmax),
                       np.uint64(qseed), qbuf)

    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(n_samples, dtype=np.int64)

    # One fused launch per level: split search, descend, and at small n the next
    # level's occupied-leaf list, which lets the kernel skip zeroing and scanning
    # empty leaf rows. Any superset of the occupied rows is exact (empty rows are
    # all-zero once zeroed), so at large n we pass all rows — there the scatter
    # dominates and the trim is noise. The two occupancy buffers ping-pong so the
    # kernel never writes the buffer `active` currently views.
    small = n_samples < _SMALL_N
    act_w = np.empty(1 << max_depth, dtype=np.int64) if small else _EMPTY_I64
    act_r = np.empty(1 << max_depth, dtype=np.int64) if small else _EMPTY_I64
    active = np.arange(1, dtype=np.int64)            # level 0: the root
    n_leaves_next = 2

    for d in range(max_depth):
        if quantize:
            f, t, gain, n_next = _build_split_descend_q(
                Xb, qbuf, leaf, active, hist, feature_mask,
                n_bins_per_feature, dg, dh, l2, min_child_weight, min_gain,
                small, n_leaves_next, act_w)
        else:
            f, t, gain, n_next = _build_split_descend(
                Xb, grad, hess, leaf, active, hist, feature_mask,
                n_bins_per_feature, l2, min_child_weight, min_gain, small,
                n_leaves_next, act_w)
        if gain <= min_gain or t < 0:
            break

        splits_feat.append(f)
        splits_thr.append(t)
        splits_gain.append(gain)

        if small:
            active = act_w[:n_next]
            act_w, act_r = act_r, act_w
        else:
            active = np.arange(n_leaves_next, dtype=np.int64)
        n_leaves_next <<= 1

    sf = np.array(splits_feat, dtype=np.int64)
    st = np.array(splits_thr, dtype=np.int64)
    n_leaves = 1 << len(splits_feat)
    values = _leaf_values(leaf, grad, hess, n_leaves, l2, lr)

    lin_feats = lin_coef = None
    if linear_leaves and len(splits_feat) > 0 and centers_std is not None:
        # Linear term uses the NUMERIC features the tree actually split on.
        seen = []
        for f in splits_feat:
            if is_numeric[f] and f not in seen:
                seen.append(f)

        if seen:
            lin_feats = np.array(seen, dtype=np.int64)
            lin_coef = _linear_leaf_fit(leaf, grad, hess, n_leaves, lin_feats,
                                        centers_std, Xb, l2, linear_lambda, lr)

    tree = ObliviousTree(sf, st, values, np.array(splits_gain, dtype=np.float64),
                         lin_feats=lin_feats, lin_coef=lin_coef,
                         centers_std=centers_std if lin_coef is not None else None)

    # `leaf` is the training-set assignment, returned so callers (LOO update,
    # leaf correction) reuse it instead of recomputing tree.apply(Xb).
    return tree, leaf