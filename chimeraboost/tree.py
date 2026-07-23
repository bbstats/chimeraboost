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


# Below this many samples, per-level work is fixed-cost bound (kernel launches,
# empty-leaf zero/scan, strided split scans) rather than sample bound;
# build_oblivious_tree switches to the small-n variants (occupied-leaf lists,
# serial descend). Both sides of every dispatch are bit-identical, so the
# threshold only affects speed. 32768 sits comfortably between the measured
# regimes (1.7x fused win at 2k, parity at 200k).
_SMALL_N = 32768

# Placeholder passed as the fused level kernel's occupancy buffer on the
# large-n path, where it is never touched (shared and read-only, so safe
# across concurrent fits).
_EMPTY_I64 = np.empty(0, dtype=np.int64)


@njit(cache=True, parallel=True)
def _build_histograms_into(Xb, grad, hess, leaf, n_leaves, hist, feat_mask):
    """Fill per-feature gradient/hessian histograms into a pre-allocated buffer.

    REFERENCE KERNEL: the fit path now uses the fused `_build_and_split`; this
    is kept (with `_best_split`) as the plainly-readable equivalence oracle for
    tests/test_tree_kernels.py.

    `Xb` is feature-major (n_features, n_samples), so `Xb[f]` is a contiguous
    row and the inner sample loop reads bins, grads, and hessians sequentially.

    `hist` has shape (n_features, max_leaves, max_bins, 2): grad and hess for a
    bin are interleaved on the last axis so each scatter write touches a single
    cache line instead of two separate arrays. Reused across every tree and
    level; we zero only the (n_leaves) slice we are about to write. Parallelized
    over features so each thread owns a disjoint slice -- no write races.

    Features with feat_mask[f] == 0 (column subsampling) are skipped entirely:
    `_best_split` never reads their slice (it honors the same mask), so the
    stale data left there is harmless and the whole scan is saved. At
    colsample=c that removes a (1-c) fraction of histogram work.
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

    REFERENCE KERNEL: the fit path now descends inside `_build_split_descend`;
    kept (with the serial twin) as the descend oracle for
    tests/test_tree_kernels.py.

    Replaces the per-level numpy expression
    ``leaf = (leaf << 1) + (Xb[f] > t).astype(np.int64)`` which allocated several
    n-sample temporaries (the bool mask, its int64 cast, the shifted array, the
    sum) on every one of the (max_depth x n_trees) level steps — measured at ~⅓
    of total fit time. One parallel pass over the contiguous feature row, no
    temporaries; bit-identical bucketing.
    """
    for i in prange(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > t else 0)


@njit(cache=True)
def _descend_leaves_serial(leaf, Xf, t):
    """Serial twin of `_descend_leaves` for small n, where the parallel
    fork/join costs more than the pass itself (~4.7us vs 0.9us at n=2k).
    Every write is independent, so serial and parallel are bit-identical;
    `build_oblivious_tree` dispatches on `_SMALL_N`."""
    for i in range(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > t else 0)


@njit(cache=True, parallel=True)
def _build_and_split(Xb, grad, hess, leaf, active, hist, feat_mask,
                     n_bins_per_feature, l2, min_child_weight):
    """Fused histogram build + best-split search: one parallel launch per
    level instead of two, and the split scan runs on the hist slice the same
    thread just wrote (cache-hot).

    REFERENCE KERNEL: the fit path now runs `_build_split_descend` (this
    kernel's search plus the level's descend/occupancy in the same launch);
    this is retained as its split-search oracle for
    tests/test_tree_kernels.py.

    Produces EXACTLY the outputs of `_build_histograms_into` followed by
    `_best_split` (the retained reference kernels in this module) — verified
    by an exact-equality test — while cutting the small-n fixed cost three
    ways:

      * `active` lists the leaf rows that actually contain samples (callers
        may pass any superset, e.g. arange(n_leaves) when counting isn't
        worth it). Empty leaves are all-zero histogram rows: skipping them
        skips zeroing and scanning cells that contribute nothing. Occupancy
        is feature-independent, so one list serves every feature.
      * Only bins [0, n_bins_[f]) are zeroed and scanned per feature — the
        scatter never writes past a feature's actual bin count.
      * The split scan is transposed (leaf-outer, bin-inner): the prefix and
        gain passes stream each hist row sequentially instead of striding
        across leaf rows per threshold. gain[t] still accumulates leaves in
        ascending order, so every floating-point sum matches the reference
        `_best_split` bit for bit; the parent term gl*gl/(ht+l2) is computed
        once per leaf (identical value, one divide instead of nb-1).

    Legality (`min_child_weight`) matches the reference: a threshold dies if
    ANY contributing leaf would gain a sparse non-empty child; leaves with no
    hessian mass contribute neither gain nor legality vetoes.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
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

    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    return best_f, feat_thr[best_f], best_gain


@njit(cache=True, parallel=True)
def _build_split_descend(Xb, grad, hess, leaf, active, hist, feat_mask,
                         n_bins_per_feature, l2, min_child_weight, min_gain,
                         small, n_leaves_next, next_active):
    """`_build_and_split` plus the level's follow-up work in the same launch:
    when the found split is usable (legal threshold, gain > min_gain) the
    kernel also pushes every sample one level deeper, and on the small-n path
    emits the next level's occupied-leaf list.

    Replaces, per level: one split launch + one descend launch + a
    bincount/flatnonzero numpy pair. At small n the per-level cost is
    launch/fixed-cost bound (GROW_PLAN.md Phase 0: per-tree Python residue
    8-15% of fit on Grinsztajn-sized sets) — that fixed cost is what this
    removes; the arithmetic is unchanged.

    Bit-identity: the split search is `_build_and_split`'s code verbatim
    (that kernel is retained as this one's oracle, itself oracle-tested
    against `_build_histograms_into` + `_best_split`); the descend is
    `_descend_leaves(_serial)`'s integer update, fused with an integer
    occupancy count (exact in any order); the occupancy list is ascending
    nonzero-count indices — exactly flatnonzero(bincount(leaf,
    n_leaves_next)). The descend fires iff the caller's continue-predicate
    holds (NOT (gain <= min_gain or t < 0)), so a rejected level leaves
    `leaf` untouched, like the old Python-side break did.

    Returns (best_f, best_t, best_gain, n_next); n_next is the occupancy
    list length, or -1 when no list was built (large n, or no descend).
    `next_active` needs room for n_leaves_next entries on the small-n path;
    it is never touched otherwise.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
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

    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    best_t = feat_thr[best_f]

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


# Quantized-gradient histograms (QUANT_PLAN.md, LightGBM-4-style adaptation):
# grad/hess are quantized per tree to integers and packed into ONE int64 per
# sample, so the histogram scatter does a single integer RMW per (sample,
# feature) instead of two float64 RMWs, and the buffer footprint halves.
# _QMAX_CAP bounds the quantized range at 15 bits; build_oblivious_tree
# shrinks it further for huge n so that any cell/prefix sum keeps
# |sum qg| <= n*qmax < 2**31 and 0 <= sum qh < 2**32 — the packed halves can
# then never bleed into each other and shift/mask unpacking is exact.
_QMAX_CAP = 32767


@njit(cache=True, parallel=True)
def _gh_absmax(grad, hess):
    """Fused (max |grad|, max hess) reduction — the quantization scales —
    without numpy temporaries (np.abs(grad).max() would allocate n floats)."""
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

    The uniform pair u comes from counter-based splitmix64(qseed + i):
    deterministic given the seed (reproducible models, no RNG state threaded
    through numba), unbiased rounding (round-to-nearest would bias every
    histogram cell the same way; stochastic errors cancel by sqrt(n) — the
    LightGBM quantized-training result). qg lands in [-qmax, qmax] and qh in
    [0, qmax] by construction of the scales; the clamps only guard the edge
    where gmax * (qmax/gmax) rounds a hair above qmax, keeping the caller's
    overflow bound exact. Hessians are non-negative for every library loss,
    so qh's lower clamp is defensive only."""
    n = grad.shape[0]
    for i in prange(n):
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
def _build_split_descend_q(Xb, q, leaf, active, histq, feat_mask,
                           n_bins_per_feature, dg, dh, l2, min_child_weight,
                           min_gain, small, n_leaves_next, next_active):
    """Packed-int64 twin of `_build_split_descend` for quantized training.

    Structure is that kernel's verbatim except the histogram: `histq` is
    int64 (n_features, max_leaves, max_bins), the scatter adds the packed
    sample value once, and the scan runs packed-integer prefix sums (exact —
    see _QMAX_CAP) that are unpacked (arithmetic shift / mask) and
    dequantized with dg/dh only where the float gain formula needs them.
    On exactly-representable grad/hess (integer multiples of power-of-two
    scales) this reproduces the float kernel bit for bit — that is the
    oracle test in tests/test_tree_kernels.py; on real data it differs from
    the float kernel only by the quantization noise. hr = ht - hl is
    computed in float like the reference; multiplication by a positive
    scale is monotone, so hr never goes negative."""
    n_features, n_samples = Xb.shape
    max_bins = histq.shape[2]
    n_active = active.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        for k in range(n_active):
            l = active[k]
            for b in range(nb):
                histq[f, l, b] = 0
        Xf = Xb[f]
        for i in range(n_samples):
            histq[f, leaf[i], Xf[i]] += q[i]

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

    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    best_t = feat_thr[best_f]

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
def _best_split(hist, n_bins_per_feature, l2, feat_mask, min_child_weight,
                n_leaves):
    """Find the (feature, threshold) with the highest total gain.

    REFERENCE KERNEL: the fit path now uses the fused `_build_and_split`; kept
    as the equivalence oracle for tests/test_tree_kernels.py.

    `hist` is the interleaved (n_features, max_leaves, max_bins, 2) buffer:
    [..., 0] is grad, [..., 1] is hess. `n_leaves` says how many leaf rows are
    actually active at this level, so we only read those.

    For a candidate threshold t, bins <= t go left and bins > t go right, the
    same way in every current leaf. Gain is summed across leaves. Features with
    feat_mask[f] == 0 are skipped (column subsampling).

    A threshold is legal unless some leaf would gain a *sparse non-empty* child
    (0 < hessian mass < min_child_weight) -- that is the sparse-leaf overfit risk,
    and since the split is shared it is rejected for the whole level. Children
    that come out EMPTY (a leaf whose samples all go one way) are exempt: pure
    leaves are normal in an oblivious tree and must not block the shared split,
    or effective depth caps far below what the data supports.
    """
    n_features = hist.shape[0]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        # Totals per leaf for this feature (same regardless of threshold).
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
            # Pass 1: Advance running prefix sums for all leaves unconditionally
            # so GL/HL carry correctly into the next threshold.
            for l in range(n_leaves):
                GL[l] += hist[f, l, t, 0]
                HL[l] += hist[f, l, t, 1]

            # Pass 2: gain of this threshold, and its legality (see docstring:
            # only a sparse non-empty child vetoes the shared split).
            gain = 0.0
            legal = True
            for l in range(n_leaves):
                if Ht[l] > 0.0:
                    hl = HL[l]
                    hr = Ht[l] - hl
                    # Empty child (hl==0 or hr==0) is exempt; only 0 < mass <
                    # min_child_weight is illegal.
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
def _solve_small(A, b):
    """Solve ``A x = b`` for a small dense system via LU with partial pivoting.

    Drop-in replacement for ``np.linalg.solve`` on the tiny (d x d, d <= depth+1)
    per-leaf normal equations. Same algorithm family as LAPACK's gesv (LU with
    partial pivoting) but hand-rolled, which avoids instantiating numba's LAPACK
    bindings: those alone account for several seconds of JIT compile time on the
    first fit in a fresh environment. ``A`` and ``b`` are modified in place.
    Returns the solution, or a vector of NaN if a pivot underflows (the caller
    then falls back to the constant Newton leaf value; with the ridge + jitter
    on the diagonal this cannot trigger in practice).
    """
    d = A.shape[0]
    x = np.empty(d)
    for c in range(d):
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
        inv = 1.0 / A[c, c]
        for r in range(c + 1, d):
            f = A[r, c] * inv
            if f != 0.0:
                A[r, c] = 0.0
                for j in range(c + 1, d):
                    A[r, j] -= f * A[c, j]
                b[r] -= f * b[c]
    for r in range(d - 1, -1, -1):
        s = b[r]
        for j in range(r + 1, d):
            s -= A[r, j] * x[j]
        x[r] = s / A[r, r]
    return x


@njit(cache=True, parallel=True)
def _linear_leaf_fit(leaf, grad, hess, n_leaves, lin_feats, centers_std, Xb,
                     l2_intercept, lin_lambda, lr):
    """Fit a small hessian-weighted ridge per leaf (local linear-leaf models).

    For samples in a leaf we solve the second-order objective
        min_beta  sum_i [ g_i f_i + 1/2 h_i f_i^2 ] + 1/2 ( l2*b^2 + lin*||w||^2 )
    with f_i = b + w . x_std_i over the leaf's numeric split features -- i.e. the
    normal equations  (A^T diag(h) A + Lambda) beta = -A^T g,  A = [1, x_std],
    accumulated directly (no per-leaf design matrix). The fitted output is
    `lr * beta`. Leaves with too few samples to support the slope (or empty
    leaves) fall back to the plain constant Newton value, so the linear model
    only ever ADDS local slope where the data supports it. Returns `lin_coef` of
    shape (n_leaves, 1 + len(lin_feats)) (column 0 = intercept).

    `centers_std` is the per-feature table of standardized bin-center values;
    NaN (missing) bins are treated as 0 (= the feature mean).

    Parallel over leaves, bit-identically to the old serial global scan: a
    stable counting sort groups sample indices by leaf in original order, so
    each leaf's normal equations accumulate in exactly the same float-add
    sequence the serial version used (a leaf only ever saw its own samples,
    in increasing i). Thread-count invariant for the same reason. The
    standardized design values are gathered per sample inside the leaf loop
    (no (k, n) scratch matrix, and a single parallel region keeps the JIT
    compile cost down)."""
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

    Parallel over samples; each out[i] is independent, so bit-identical to
    the serial loop."""
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
def _loo_leaf_step(leaf, grad, hess, n_leaves, l2, lr):
    """Leave-one-out training step for every row, fused into two passes.

    First pass scatters per-leaf grad/hess totals; second pass gathers each
    row's totals, removes the row's own contribution, and forms the shrunk
    Newton step. Replaces two np.bincount calls plus several NumPy temporaries
    with one scatter and one compute loop over `leaf`."""
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

    Parameters are the trees packed into flat arrays (see `pack_forest`):
    `feats`/`thrs` are (n_trees, max_depth) split tables, `depths[t]` the real
    depth of tree t, and `vals`/`voff` a ragged leaf-value table (tree t's leaf
    values live at vals[voff[t] : voff[t+1]]).

    Parallelizing over samples (not trees) means each sample loads its handful
    of feature bins once and keeps them hot in cache while walking every tree.
    The per-sample accumulation runs init + tree0 + tree1 + ... in tree order,
    matching the serial `F += tree.predict(Xb)` loop bit-for-bit."""
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

    Predict-time binning produces row-major output; consuming it directly
    keeps each sample's feature bins in one or two cache lines for the whole
    forest walk and skips the feature-major transpose copy entirely. Same
    arithmetic and per-sample accumulation order as `_predict_forest`, so the
    two are bit-identical."""
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
    """Serial twin of `_predict_forest_rm` for tiny batches: the OpenMP
    fork/join (~20us on 12 threads) exceeds the whole 1-row walk, and the
    parallel kernel only overtakes serial around n~5. Bit-identical
    (independent per-row writes); the booster dispatches on
    `binning._SERIAL_PREDICT_N`."""
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

    Returns (feats, thrs, depths, vals, voff). Cached by the booster after fit
    so repeated predict calls skip the rebuild."""
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


def pack_forest_linear(trees, max_depth):
    """Flatten a forest of (possibly) linear-leaf trees for `_predict_forest_linear`.

    A constant-leaf tree is just a linear tree with k=0 features (its coef block
    is the leaf intercepts), so one packed layout + kernel serves both. Per tree:
    `lin_k[t]` linear features at `lin_feat_idx[featoff[t]:featoff[t+1]]`, and a
    leaf-major coef block at `coef[coefoff[t]:coefoff[t+1]]` of shape
    (n_leaves, 1 + lin_k[t]) flattened (column 0 = intercept)."""
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
    matching `_linear_predict`/`ObliviousTree.predict` so the fused path agrees
    with the per-tree path bit-for-bit (same accumulation order)."""
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
    matrix — see `_predict_forest_rm` for why. Bit-identical to the
    feature-major kernel (same arithmetic, same accumulation order)."""
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
    `_predict_forest_rm_serial`."""
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
def _shap_forest_linear(Xb, Rb, feats, thrs, depths, lin_k, featoff,
                        lin_feat_idx, coefoff, coef, centers_std,
                        feat_orig, n_orig, fact):
    """Exact interventional TreeSHAP for a forest of oblivious (linear-leaf or
    constant, k=0) trees, returned in the user's ORIGINAL feature space.

    For each instance x (column of Xb) and background reference r (column of Rb)
    the per-tree Shapley values are computed by exact enumeration over subsets of
    the distinct ORIGINAL features the tree uses. This is tractable precisely
    because the trees are oblivious: a depth-D tree touches at most D distinct
    features, so the coalition game has at most D players (<=2**D subsets), not
    one per input column. A feature in coalition S takes its value from x, the
    rest from r; the leaf -- and any linear-leaf slope term -- is evaluated under
    that mix, so the linear leaves are explained faithfully rather than ignored.

    Contributions are averaged over the background and summed over trees, giving
    for every instance the Shapley-efficiency identity (to float tolerance)
        sum_orig phi[i, orig] == predict_trees(x_i) - mean_r predict_trees(r).
    Two internal columns mapping to the same original feature (categorical combos
    / multi-target encodings) are treated as ONE player, so the attribution lands
    directly in input-feature space. `fact[s]` is s! (precomputed up to depth).
    Parallelized over instances; each thread owns a disjoint row of `phi`."""
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
            # Distinct original features used by this tree = coalition players U;
            # level_u[dd] is the U-slot of level dd's feature (features reused
            # across levels share a slot, so they move together in a coalition).
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
            # x-side: level bits and standardized linear values (ref-independent).
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
                # Output of every coalition: bits/linear-values follow x inside S,
                # r outside it.
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
                # Shapley value of each player: weighted marginal over every
                # coalition that excludes it.
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

    Its `apply`/`predict` take a feature-major binned matrix (n_features,
    n_samples) -- the same layout the builder consumes."""

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
        return _assign_leaves(Xb, self.splits_feat, self.splits_thr)

    def predict(self, Xb):
        if self.depth == 0:
            return np.zeros(Xb.shape[1], dtype=np.float64)
        if self.lin_coef is not None:
            leaf = _assign_leaves(Xb, self.splits_feat, self.splits_thr)
            return _linear_predict(leaf, self.lin_feats, self.lin_coef,
                                   self.centers_std, Xb)
        return _predict_tree(Xb, self.splits_feat, self.splits_thr, self.values)


def build_oblivious_tree(Xb, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         linear_leaves=False, centers_std=None, is_numeric=None,
                         linear_lambda=1.0, quantize=False, qbuf=None,
                         qseed=0):
    """Grow one oblivious tree level by level. Returns (tree, train_leaf), where
    train_leaf is the tree's leaf index for every training sample.

    Xb: feature-major binned matrix (n_features, n_samples).
    feature_mask: optional 0/1 array over features; 0 disables a feature for
    this tree (column subsampling). None means all features are eligible.
    min_child_weight: minimum hessian mass each side of a split must retain in
    every non-empty leaf. Stops the tree growing once no legal split remains,
    which prevents sparse-leaf overfitting at higher depth.
    hist_buffers: optional buffer reused across trees to avoid per-level
    allocation: interleaved (n_features, 2**max_depth, max_bins, 2) float64,
    or with quantize=True int64 (n_features, 2**max_depth, max_bins). If
    None, it is allocated here (for one-off calls and tests).
    linear_leaves: when True, attach a per-leaf ridge linear model over the
    tree's numeric split features (`centers_std`/`is_numeric` required;
    `linear_lambda` is the slope penalty). Low-count leaves fall back to the
    constant Newton value. The split search is unaffected.
    quantize: run the SPLIT SEARCH on packed-int64 quantized grad/hess
    (QUANT_PLAN.md) — one integer RMW per scatter write, half the histogram
    footprint. Leaf values (and the linear-leaf ridge) still use the
    original float64 grad/hess, so quantization noise touches only the
    structure choice. `qbuf` is an optional reusable int64 (n_samples)
    scratch for the packed values; `qseed` seeds the stochastic rounding
    (pass a fresh draw per tree for decorrelated rounding noise).
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
        # qmax keeps every packed cell/prefix sum overflow-safe (see _QMAX_CAP
        # comment); scales map the observed grad/hess range onto [-qmax, qmax]
        # and [0, qmax]. All-zero grad or hess degenerates to qg/qh = 0, which
        # yields zero gains — the same no-split outcome as the float kernel.
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

    small = n_samples < _SMALL_N
    # One fused launch per level: split search + descend + (small n) the next
    # level's occupied-leaf list, which lets the kernel skip zeroing/scanning
    # empty leaf rows. Any superset is exact (empty rows are all-zero once
    # zeroed), so at large n we pass all rows — there the scatter dominates
    # and the trim is noise. Occupancy buffers ping-pong so the kernel never
    # writes the buffer `active` currently views.
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