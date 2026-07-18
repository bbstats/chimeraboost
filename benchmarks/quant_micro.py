"""Phase-0 microbench for QUANT_PLAN.md — packed-int histogram kernel vs float.

Races the library's fused level kernel (`_build_split_descend`, float64
interleaved grad/hess histograms) against script-local prototypes:

  B: packed int64 histograms — q[i] = (qg << 32) + qh, ONE integer RMW per
     (sample, feature) in the scatter, packed-integer prefix sums in the scan
     (exact by QMAX construction), dequantized floats only in the gain formula.
     Includes the per-tree quantize+pack cost (scales + one pass over n).
  C: packed int32 (16+16, LightGBM-low-bit-style, 4-bit hess / 5-bit grad)
     with int64 unpack-accumulate in the scan — TIMING BOUND ONLY for the
     two-tier scheme; not a ship candidate as-is.

Method per GROW_PLAN lessons: script file, dispatchers called directly (never
`__wrapped__`), one JIT-warm untimed run per shape, median of warm reps of the
full 6-level per-tree loop, planted signal so every timed run reaches depth 6.

Usage:  python benchmarks/quant_micro.py
Writes: benchmarks/results/quant-phase0-micro.md (+ prints the table).
"""

import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chimeraboost  # noqa: E402
from chimeraboost.tree import _build_split_descend  # noqa: E402

DEPTH = 6
BINS = 128
L2 = 1.0
MCW = 1.0
MIN_GAIN = 1e-8
SMALL_N = 32768          # mirror tree._SMALL_N dispatch
REPS = 25


# ---------------------------------------------------------------- B: int64 ---

@njit(cache=True, parallel=True)
def _gh_absmax(grad, hess):
    """Fused (max|g|, max h) reduction — no numpy temporaries. In the real
    implementation logloss can skip this entirely (|g| < 1, h <= 0.25 are loss
    constants); this micro times the general (regression/weighted) case."""
    gmax = 0.0
    hmax = 0.0
    for i in prange(grad.shape[0]):
        ag = abs(grad[i])
        gmax = max(gmax, ag)
        hmax = max(hmax, hess[i])
    return gmax, hmax


@njit(cache=True, parallel=True)
def _quantize_pack(grad, hess, inv_dg, inv_dh, qseed, out):
    """q[i] = (qg << 32) + qh with stochastic rounding qX = floor(x*inv + u).

    u pair comes from counter-based splitmix64(qseed + i): deterministic given
    the seed, no RNG state threaded through. qh >= 0 always; the packed sum of
    any subset keeps 0 <= sum(qh) < 2^32 by the caller's QMAX bound, so the
    halves never bleed and arithmetic-shift/mask unpack is exact."""
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
        out[i] = (qg << 32) + qh


@njit(cache=True, parallel=True)
def _build_split_descend_q(Xb, q, leaf, active, histq, feat_mask,
                           n_bins_per_feature, dg, dh, l2, min_child_weight,
                           min_gain, small, n_leaves_next, next_active):
    """Packed-int64 twin of tree._build_split_descend.

    Scatter: histq[f, l, b] += q[i]  (one RMW, half the buffer footprint).
    Scan: packed prefix sums (exact ints), unpack via >>32 / &0xFFFFFFFF,
    dequantize with dg/dh, then the float gain formula verbatim."""
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


# ------------------------------------------- C: int32 (timing bound only) ---

@njit(cache=True, parallel=True)
def _quantize_pack32(grad, hess, inv_dg, inv_dh, qseed, out):
    """16+16 packed int32, 5-bit grad / 4-bit hess (LightGBM-low-bit-style)."""
    n = grad.shape[0]
    for i in prange(n):
        z = (qseed + np.uint64(i)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        u1 = (z & np.uint64(0xFFFFFFFF)) * (1.0 / 4294967296.0)
        u2 = (z >> np.uint64(32)) * (1.0 / 4294967296.0)
        qg = np.int32(np.floor(grad[i] * inv_dg + u1))
        qh = np.int32(np.floor(hess[i] * inv_dh + u2))
        out[i] = (qg << 16) + qh


@njit(cache=True, parallel=True)
def _build_split_descend_q32(Xb, q, leaf, active, histq, feat_mask,
                             n_bins_per_feature, dg, dh, l2, min_child_weight,
                             min_gain, small, n_leaves_next, next_active):
    """int32-cell twin: cells hold (qg<<16)+qh; the scan unpacks each cell into
    int64 accumulators (cell halves stay in range on uniform synthetic data;
    whole-leaf totals would overflow 16 bits, hence per-bin unpack)."""
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
            gtot = np.int64(0)
            htot = np.int64(0)
            for b in range(nb):
                v = histq[f, l, b]
                gtot += v >> 16
                htot += v & 0xFFFF
            ht = htot * dh
            gt = gtot * dg
            if ht <= 0.0:
                continue
            par = gt * gt / (ht + l2)
            ga = np.int64(0)
            ha = np.int64(0)
            for t in range(nb - 1):
                v = histq[f, l, t]
                ga += v >> 16
                ha += v & 0xFFFF
                hl = ha * dh
                gl = ga * dg
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


# ------------------------------------------------------------------ harness ---

def make_data(n, nf, seed):
    """Uniform bins + planted hierarchical signal so trees reach depth 6."""
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, BINS, size=(nf, n)).astype(np.uint16)
    base = (0.6 * (Xb[0] > 64) + 0.3 * (Xb[1 % nf] > 32)
            + 0.4 * (Xb[2 % nf].astype(np.float64) / BINS))
    p = 1.0 / (1.0 + np.exp(-(base - base.mean())))
    grad = (base - base.mean()) + 0.5 * rng.standard_normal(n)
    hess = p * (1.0 - p)                       # logloss-shaped, in (0, .25]
    return np.ascontiguousarray(Xb), grad, hess


def run_tree_float(Xb, grad, hess, leaf, hist, fmask, nbins, act_w, small):
    leaf[:] = 0
    active = np.arange(1, dtype=np.int64)
    n_leaves_next = 2
    d = 0
    while d < DEPTH:
        f, t, gain, n_next = _build_split_descend(
            Xb, grad, hess, leaf, active, hist, fmask, nbins, L2, MCW,
            MIN_GAIN, small, n_leaves_next, act_w)
        if gain <= MIN_GAIN or t < 0:
            break
        d += 1
        active = (act_w[:n_next].copy() if small
                  else np.arange(n_leaves_next, dtype=np.int64))
        n_leaves_next <<= 1
    return d


def run_tree_q(Xb, grad, hess, leaf, histq, qbuf, fmask, nbins, act_w, small,
               qseed, pack_ms):
    n = grad.shape[0]
    qmax = min(32767, (2 ** 31 - 1) // n)
    t0 = time.perf_counter()
    gmax, hmax = _gh_absmax(grad, hess)
    inv_dg = qmax / gmax if gmax > 0 else 0.0
    inv_dh = qmax / hmax if hmax > 0 else 0.0
    dg = gmax / qmax if gmax > 0 else 0.0
    dh = hmax / qmax if hmax > 0 else 0.0
    _quantize_pack(grad, hess, inv_dg, inv_dh, np.uint64(qseed), qbuf)
    pack_ms.append((time.perf_counter() - t0) * 1e3)

    leaf[:] = 0
    active = np.arange(1, dtype=np.int64)
    n_leaves_next = 2
    d = 0
    while d < DEPTH:
        f, t, gain, n_next = _build_split_descend_q(
            Xb, qbuf, leaf, active, histq, fmask, nbins, dg, dh, L2, MCW,
            MIN_GAIN, small, n_leaves_next, act_w)
        if gain <= MIN_GAIN or t < 0:
            break
        d += 1
        active = (act_w[:n_next].copy() if small
                  else np.arange(n_leaves_next, dtype=np.int64))
        n_leaves_next <<= 1
    return d


def run_tree_q32(Xb, grad, hess, leaf, histq, qbuf, fmask, nbins, act_w, small,
                 qseed):
    qmax_g, qmax_h = 15, 15                    # 5-bit signed grad, 4-bit hess
    gmax = float(np.abs(grad).max())
    hmax = float(hess.max())
    inv_dg = qmax_g / gmax if gmax > 0 else 0.0
    inv_dh = qmax_h / hmax if hmax > 0 else 0.0
    dg = gmax / qmax_g if gmax > 0 else 0.0
    dh = hmax / qmax_h if hmax > 0 else 0.0
    _quantize_pack32(grad, hess, inv_dg, inv_dh, np.uint64(qseed), qbuf)

    leaf[:] = 0
    active = np.arange(1, dtype=np.int64)
    n_leaves_next = 2
    d = 0
    while d < DEPTH:
        f, t, gain, n_next = _build_split_descend_q32(
            Xb, qbuf, leaf, active, histq, fmask, nbins, dg, dh, L2, MCW,
            MIN_GAIN, small, n_leaves_next, act_w)
        if gain <= MIN_GAIN or t < 0:
            break
        d += 1
        active = (act_w[:n_next].copy() if small
                  else np.arange(n_leaves_next, dtype=np.int64))
        n_leaves_next <<= 1
    return d


def main():
    print(f"chimeraboost: {chimeraboost.__file__}")
    shapes = [(8_000, 10), (8_000, 32), (37_500, 10), (37_500, 32),
              (75_000, 10), (75_000, 32), (200_000, 10), (200_000, 32)]
    rows = []
    for n, nf in shapes:
        Xb, grad, hess = make_data(n, nf, seed=42)
        small = n < SMALL_N
        fmask = np.ones(nf, dtype=np.int64)
        nbins = np.full(nf, BINS, dtype=np.int64)
        leaf = np.zeros(n, dtype=np.int64)
        act_w = np.empty(1 << DEPTH, dtype=np.int64)
        hist = np.zeros((nf, 1 << DEPTH, BINS, 2))
        histq = np.zeros((nf, 1 << DEPTH, BINS), dtype=np.int64)
        hist32 = np.zeros((nf, 1 << DEPTH, BINS), dtype=np.int32)
        qbuf = np.empty(n, dtype=np.int64)
        qbuf32 = np.empty(n, dtype=np.int32)

        # JIT warm + depth sanity (one untimed run per variant).
        dA = run_tree_float(Xb, grad, hess, leaf, hist, fmask, nbins, act_w,
                            small)
        dB = run_tree_q(Xb, grad, hess, leaf, histq, qbuf, fmask, nbins,
                        act_w, small, 1234, [])
        dC = run_tree_q32(Xb, grad, hess, leaf, hist32, qbuf32, fmask, nbins,
                          act_w, small, 1234)

        tA, tB, tC, pack_ms = [], [], [], []
        for r in range(REPS):
            t0 = time.perf_counter()
            run_tree_float(Xb, grad, hess, leaf, hist, fmask, nbins, act_w,
                           small)
            tA.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            run_tree_q(Xb, grad, hess, leaf, histq, qbuf, fmask, nbins,
                       act_w, small, 1234 + r, pack_ms)
            tB.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            run_tree_q32(Xb, grad, hess, leaf, hist32, qbuf32, fmask, nbins,
                         act_w, small, 1234 + r)
            tC.append((time.perf_counter() - t0) * 1e3)

        mA = float(np.median(tA))
        mB = float(np.median(tB))
        mC = float(np.median(tC))
        mP = float(np.median(pack_ms))
        rows.append((n, nf, dA, dB, dC, mA, mB, mP, mC,
                     mA / mB if mB else 0.0, mA / mC if mC else 0.0))
        print(f"n={n:>7} nf={nf:>2} depth A/B/C={dA}/{dB}/{dC}  "
              f"A={mA:8.2f}ms  B={mB:8.2f}ms (pack {mP:5.2f})  "
              f"C={mC:8.2f}ms  A/B={mA / mB:5.2f}x  A/C={mA / mC:5.2f}x")

    lines = [
        "# Phase-0 micro: packed-int histogram kernel vs float "
        "(quant_micro.py)",
        "",
        "Median of %d warm reps, full %d-level per-tree loop, uniform bins + "
        "planted signal," % (REPS, DEPTH),
        "logloss-shaped hessians. B includes the per-tree quantize+pack cost "
        "(also shown alone).",
        "C is the int32 (16+16) TIMING BOUND, not a ship candidate.",
        "",
        "| n | nf | depth A/B/C | A float ms | B int64 ms | pack ms | "
        "C int32 ms | A/B | A/C |",
        "|--:|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for (n, nf, dA, dB, dC, mA, mB, mP, mC, rB, rC) in rows:
        lines.append(f"| {n} | {nf} | {dA}/{dB}/{dC} | {mA:.2f} | {mB:.2f} | "
                     f"{mP:.2f} | {mC:.2f} | {rB:.2f}x | {rC:.2f}x |")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "quant-phase0-micro.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
