"""F4 candidate C1 ceiling read: how much of a multiclass fit is really softmax?

C2 taught this family a lesson that this script exists to apply: a cProfile
percentage is an UPPER BOUND, not an estimate. `factorize` profiled at 18% of
the okcupid-stem fit and was 8% by wall clock, because cProfile charges its
per-call overhead on every one of 1.6M `dict.get` calls. `_softmax` profiled at
39%, and its calls are few and fat (767 numpy calls, not a million tiny ones),
so it should NOT suffer the same inflation -- but "should not" is a forecast,
and the plan file says C1 may not be forecast off the profile alone.

So: measure, then forecast. Nothing is optimized here.

Two reads, both wall clock:

  1. IN-FIT -- `losses._softmax` is wrapped with a `perf_counter` pair and a
     real fit is run. At ~800 calls the wrapper costs microseconds in total, so
     unlike the C2 case the instrument does not move what it measures. Calls
     are attributed to their caller (`grad_hess` / `eval` / `transform`) and
     the (n, K) shapes seen are recorded, because the ceiling is only readable
     against the shape it happens at.

  2. OP BREAKDOWN + CEILING -- on the dominant shape, each numpy pass (max,
     subtract, exp, sum, divide) is timed on its own, and two candidate
     rewrites are timed against the current one:
       B  in-place: the same ufuncs in the same order with `out=`, which
          removes two temporaries and must be bit-identical;
       C  a numba fused kernel: one pass over each row doing max/exp/sum/div
          together, which is NOT bit-identical by assumption (numba's libm
          `exp` need not agree with numpy's SIMD `exp` to the last bit).
     Both are checked against the current output for exact equality and the
     result of that check is printed, not assumed.

The ceiling for the whole candidate is (in-fit softmax share) x (best speedup
the microbench actually shows). If that product is under a couple of points of
fit time, C1 dies here at S0 and no library code is written.

Run: python benchmarks/f4_c1_walltime.py [--datasets hc:okcupid-stem]
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402  (the repo's one dataset loader)

import chimeraboost.losses as losses  # noqa: E402
from chimeraboost import ChimeraBoostClassifier  # noqa: E402

PANEL = ["hc:okcupid-stem"]

_ORIG_SOFTMAX = losses._softmax
_CENSUS = {"calls": 0, "seconds": 0.0, "by_caller": {}, "shapes": {}}


def _timed_softmax(F):
    """Time every `_softmax` call and tag it with the method that made it.

    `MultiSoftmax` calls the module global, so patching it here is what the
    library actually runs. `sys._getframe(1)` is read OUTSIDE the timed region
    -- the instrument must never do work inside the interval it reports, which
    is the mistake that made C2's first A/B unreadable."""
    caller = sys._getframe(1).f_code.co_name
    t0 = time.perf_counter()
    out = _ORIG_SOFTMAX(F)
    dt = time.perf_counter() - t0
    _CENSUS["seconds"] += dt
    _CENSUS["calls"] += 1
    _CENSUS["by_caller"][caller] = _CENSUS["by_caller"].get(caller, 0.0) + dt
    key = tuple(F.shape)
    _CENSUS["shapes"][key] = _CENSUS["shapes"].get(key, 0) + 1
    return out


def softmax_inplace(F):
    """Candidate B: same ufuncs, same order, two fewer temporaries."""
    z = F - F.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def softmax_colfold(F):
    """Candidate D: replace both axis=1 reduces with explicit column folds.

    The op breakdown says the `max` reduce alone is ~43% of the call, which is
    absurd for K=3 -- a reduction over a length-3 inner axis pays numpy's
    per-row reduce machinery on every row. Folding over COLUMNS instead does
    the same arithmetic as full-length vector ops.

    Bit-identity is a claim about op order, and it is checked, not assumed:
      - `max` is exact (it selects, it never rounds), so any fold order agrees;
      - `sum` is NOT exact, so the left fold here matches numpy's reduce only
        while numpy sums in order -- true for a small inner axis, false once
        pairwise blocking kicks in. That is why `exact_upto_k` is measured
        across K below rather than argued from the numpy source."""
    m = F[:, 0].copy()
    for k in range(1, F.shape[1]):
        np.maximum(m, F[:, k], out=m)
    z = F - m[:, None]
    np.exp(z, out=z)
    s = z[:, 0].copy()
    for k in range(1, z.shape[1]):
        s += z[:, k]
    z /= s[:, None]
    return z


def _build_numba_softmax():
    """Candidate C: one fused pass per row. Returns None if numba is absent."""
    try:
        import numba
    except ImportError:  # pragma: no cover - numba is a hard dep here
        return None

    @numba.njit(cache=True, parallel=True, fastmath=False)
    def _kernel(F, out, reciprocal):
        n, K = F.shape
        for i in numba.prange(n):
            m = F[i, 0]
            for k in range(1, K):
                if F[i, k] > m:
                    m = F[i, k]
            s = 0.0
            for k in range(K):
                e = np.exp(F[i, k] - m)
                out[i, k] = e
                s += e
            if reciprocal:
                inv = 1.0 / s          # one divide, K multiplies -- NOT the
                for k in range(K):     # same rounding as K divides
                    out[i, k] *= inv
            else:
                for k in range(K):
                    out[i, k] = out[i, k] / s

    def softmax_numba(F):
        out = np.empty_like(F)
        _kernel(F, out, True)
        return out

    def softmax_numba_div(F):
        """Candidate E: as C but dividing rather than multiplying by 1/s, which
        removes one of the two possible sources of last-bit disagreement (the
        other being numba's libm `exp` vs numpy's vectorized one)."""
        out = np.empty_like(F)
        _kernel(F, out, False)
        return out

    return softmax_numba, softmax_numba_div


def exactness_sweep(cands, repeats):
    """Does each candidate stay bit-identical as the class count grows?

    The fold's `sum` is the one at risk: numpy sums a short inner axis in
    order but switches to pairwise blocking as it lengthens, and the moment it
    does, a left fold stops agreeing to the last bit. Every multiclass set the
    suites contain has a K, so the answer decides whether a candidate ships
    unconditionally or behind a `K <=` guard."""
    print("\n--- exactness vs class count (n=20000, random F) ---")
    rng = np.random.default_rng(0)
    header = "  K    " + "".join(f"{n:>22s}" for n, _ in cands)
    print(header)
    limits = {n: None for n, _ in cands}
    for K in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 26, 50):
        F = rng.normal(size=(20000, K))
        ref = _ORIG_SOFTMAX(F)
        cells = []
        for name, fn in cands:
            ok = bool(np.array_equal(fn(F.copy()), ref))
            cells.append(f"{'exact' if ok else 'DRIFTS':>22s}")
            if not ok and limits[name] is None:
                limits[name] = K
        print(f"  {K:<5d}" + "".join(cells))
    for name, first_drift in limits.items():
        if first_drift is None:
            print(f"  {name}: bit-identical at every K tested")
        else:
            print(f"  {name}: bit-identical for K < {first_drift}, "
                  f"drifts from K = {first_drift}")
    return limits


def _median_time(fn, arg, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def op_breakdown(shape, repeats):
    """Time each pass of the current implementation separately."""
    rng = np.random.default_rng(0)
    F = rng.normal(size=shape)
    print(f"\n--- op breakdown at shape {shape} "
          f"(median of {repeats}, synthetic F) ---")
    mx = _median_time(lambda a: a.max(axis=1, keepdims=True), F, repeats)
    m = F.max(axis=1, keepdims=True)
    sub = _median_time(lambda a: a - m, F, repeats)
    z = F - m
    ex = _median_time(np.exp, z, repeats)
    ez = np.exp(z)
    sm = _median_time(lambda a: a.sum(axis=1, keepdims=True), ez, repeats)
    s = ez.sum(axis=1, keepdims=True)
    dv = _median_time(lambda a: a / s, ez, repeats)
    whole = _median_time(_ORIG_SOFTMAX, F, repeats)
    parts = {"max": mx, "subtract": sub, "exp": ex, "sum": sm, "divide": dv}
    for name, t in parts.items():
        print(f"  {name:9s} {t * 1e3:7.2f} ms  ({t / whole * 100:4.1f}% of the call)")
    print(f"  {'TOTAL':9s} {whole * 1e3:7.2f} ms  (measured whole-call)")
    return F, whole


def candidate_bench(F, whole, repeats):
    """Time the rewrites against the current call and check exact equality."""
    print(f"\n--- candidates at shape {F.shape} "
          f"(median of {repeats}) ---")
    ref = _ORIG_SOFTMAX(F)
    rows = []
    built = _build_numba_softmax()
    cands = [("B in-place ufuncs", softmax_inplace),
             ("D column fold", softmax_colfold)]
    if built is not None:
        numba_softmax, numba_div = built
        numba_softmax(F[:64])                      # warm the JIT before timing
        numba_div(F[:64])
        cands.append(("C numba fused", numba_softmax))
        cands.append(("E numba fused (div)", numba_div))
    for name, fn in cands:
        got = fn(F.copy())
        exact = bool(np.array_equal(got, ref))
        maxulp = float(np.max(np.abs(got - ref)))
        t = _median_time(fn, F.copy(), repeats)
        rows.append((name, t, t / whole, exact, maxulp))
        print(f"  {name:20s} {t * 1e3:7.2f} ms   {t / whole:5.2f}x current   "
              f"bit-identical={exact}   max|diff|={maxulp:.2e}")
    print(f"  {'A current':20s} {whole * 1e3:7.2f} ms   1.00x")
    return rows, cands


def run_dataset(key, n_estimators):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    n_classes = len(np.unique(ytr))
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, "
          f"n_features={Xtr.shape[1]}, n_classes={n_classes}) ===", flush=True)
    if task != "multiclass" and n_classes <= 2:
        print("  SKIP: binary/regression -- MultiSoftmax never runs here.")
        return None

    est_kw = dict(n_estimators=n_estimators, random_state=0,
                  early_stopping=True, early_stopping_rounds=50,
                  validation_fraction=0.15)
    # Warm the JIT and the bin/download caches before anything is timed.
    ChimeraBoostClassifier(n_estimators=5, random_state=0).fit(
        Xtr[:min(500, len(Xtr))], ytr[:min(500, len(Xtr))],
        cat_features=cat_idx)

    for k in _CENSUS:
        _CENSUS[k] = 0 if k == "calls" else (0.0 if k == "seconds" else {})
    losses._softmax = _timed_softmax
    t0 = time.perf_counter()
    est = ChimeraBoostClassifier(**est_kw).fit(Xtr, ytr, cat_features=cat_idx)
    fit_s = time.perf_counter() - t0
    losses._softmax = _ORIG_SOFTMAX

    share = _CENSUS["seconds"] / fit_s * 100
    print(f"  fit {fit_s:.2f}s, best_iteration_={est.best_iteration_}")
    print(f"  _softmax {_CENSUS['seconds']:.3f}s in {_CENSUS['calls']} calls "
          f"= {share:.1f}% of fit  (cProfile said 39%)")
    for caller, secs in sorted(_CENSUS["by_caller"].items(),
                               key=lambda kv: -kv[1]):
        print(f"    from {caller:12s} {secs:.3f}s ({secs / fit_s * 100:.1f}% of fit)")
    for shape, n in sorted(_CENSUS["shapes"].items(), key=lambda kv: -kv[1]):
        print(f"    shape {shape}: {n} calls")
    dominant = max(_CENSUS["shapes"].items(), key=lambda kv: kv[1])[0]
    return {"dataset": key, "fit_s": fit_s, "softmax_s": _CENSUS["seconds"],
            "share": share, "calls": _CENSUS["calls"], "shape": dominant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()

    rows = [r for r in (run_dataset(k, args.n_estimators)
                        for k in args.datasets) if r]
    if not rows:
        print("\nNo multiclass dataset in the panel -- nothing to measure.")
        return

    shape = rows[0]["shape"]
    F, whole = op_breakdown(shape, args.repeats)
    cands, fns = candidate_bench(F, whole, args.repeats)
    exactness_sweep(fns, args.repeats)

    print("\n| dataset | fit s | softmax s | share of fit | calls |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['fit_s']:.2f} | {r['softmax_s']:.2f} | "
              f"{r['share']:.1f}% | {r['calls']} |")

    best = min(cands, key=lambda c: c[1]) if cands else None
    if best:
        name, t, ratio, exact, _ = best
        ceiling = rows[0]["share"] * (1 - ratio)
        print(f"\nCEILING: softmax is {rows[0]['share']:.1f}% of fit; the best "
              f"candidate ({name}) runs at {ratio:.2f}x, so the most it can "
              f"remove is {ceiling:.1f}% of fit on {rows[0]['dataset']} "
              f"(bit-identical={exact}).")
        print("Read that against the ~1% same-process noise floor before "
              "forecasting anything.")


if __name__ == "__main__":
    main()
