"""F4 candidate C1 speed read: the fused `_softmax` kernel, in real fits.

The ceiling script (`f4_c1_walltime.py`) established that `_softmax` is 45.4%
of an hc:okcupid-stem fit by wall clock and that a fused numba kernel runs it
32x faster while staying bit-identical at K <= 7. This script asks the only
question left: how much of that ceiling turns into fit time.

Method -- a SAME-PROCESS A/B, as for C2, so the cross-run ~2% fit-time noise
floor does not apply and the tighter ~1% same-process floor does. One process
loads each dataset once, then alternates:

    OFF: `losses._softmax` monkeypatched to `_softmax_numpy`, which IS the
         pre-change function, so the OFF arm is the old library byte for byte.
    ON:  the guarded dispatcher live.

The panel is every multiclass shape the decision tier actually contains --
K = 3 at 38K rows, K = 3 at 53K rows, K = 6 at 2K rows -- plus a binary set as
the zero-change control. `MultiSoftmax` never runs on binary data, so that arm
must read flat, and if it does not, the reading is drift and the multiclass
numbers must be discounted by the same amount.

It also times the kernel serial vs parallel at the real shapes. A `prange` over
tens of thousands of rows of 3-wide work may be paying thread setup for nothing,
and the fit is already parallel elsewhere, so the serial kernel is the safer
neighbour if it is not measurably slower.

Run:
    python benchmarks/f4_c1_speed.py
    python benchmarks/f4_c1_speed.py --datasets hc:cjs --repeats 5
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
from chimeraboost import (ChimeraBoostClassifier,  # noqa: E402
                          ChimeraBoostRegressor)

# Three of the eight multiclass rows in `--decide` (all high-card; Grinsztajn
# has no multiclass task at all), chosen to span the shape space: big/K=3,
# bigger/K=3, small/K=6. Last entry is the binary control.
PANEL = ["hc:okcupid-stem", "hc:Traffic_violations", "hc:cjs", "hc:kick"]

_ON = losses._softmax
_OFF = losses._softmax_numpy
_CENSUS = {"calls": 0, "seconds": 0.0}


def _counted(fn):
    """Wrap an arm so the run can report how much time the loss layer took.

    Call counts here are in the hundreds, so the wrapper is microseconds in
    total -- the C2 rule (never let the instrument do work proportional to the
    arm) is satisfied by size, and the wrapper is applied to BOTH arms anyway,
    so whatever it costs it costs symmetrically."""
    def wrapped(F):
        t0 = time.perf_counter()
        out = fn(F)
        _CENSUS["seconds"] += time.perf_counter() - t0
        _CENSUS["calls"] += 1
        return out
    return wrapped


def _fit_once(Est, Xtr, ytr, cat_idx, n_estimators):
    est = Est(n_estimators=n_estimators, random_state=0, early_stopping=True,
              early_stopping_rounds=50, validation_fraction=0.15)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr, cat_features=cat_idx)
    return time.perf_counter() - t0


def run_dataset(key, repeats, n_estimators):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    K = len(np.unique(ytr))
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    print(f"\n=== {key}  ({task}, K={K}, n_train={len(Xtr)}, "
          f"n_features={Xtr.shape[1]}) ===", flush=True)
    if task == "multiclass" and K > losses._SOFTMAX_MAX_K:
        print(f"  NOTE: K={K} is above the guard -- this set takes the numpy "
              f"path and is expected to read flat.")

    # Warm the JIT (including the new kernel) and the caches before timing.
    Est(n_estimators=5, random_state=0).fit(
        Xtr[:min(500, len(Xtr))], ytr[:min(500, len(Xtr))],
        cat_features=cat_idx)

    off, on, cens = [], [], {}
    for i in range(repeats):
        for arm, fn, times in (("OFF", _OFF, off), ("ON", _ON, on)):
            losses._softmax = _counted(fn)
            _CENSUS["calls"], _CENSUS["seconds"] = 0, 0.0
            times.append(_fit_once(Est, Xtr, ytr, cat_idx, n_estimators))
            cens[arm] = dict(_CENSUS)
        print(f"  repeat {i + 1}: OFF {off[-1]:6.2f}s   ON {on[-1]:6.2f}s   "
              f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    losses._softmax = _ON

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    print(f"  OFF median {statistics.median(off):6.2f}s   "
          f"ON median {statistics.median(on):6.2f}s")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: {c['calls']} softmax calls, {c['seconds']:.3f}s in "
              f"softmax")
    return {"dataset": key, "task": task, "K": K, "median_ratio": med,
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "softmax_off_s": cens["OFF"]["seconds"],
            "softmax_on_s": cens["ON"]["seconds"],
            "calls": cens["ON"]["calls"]}


def serial_vs_parallel(shapes, repeats):
    """Is `prange` earning its thread setup on 3-wide rows?"""
    try:
        import numba
    except ImportError:  # pragma: no cover
        return

    @numba.njit(cache=True, parallel=False)
    def _serial(F):
        n, K = F.shape
        out = np.empty((n, K), dtype=np.float64)
        for i in range(n):
            m = F[i, 0]
            for k in range(1, K):
                if F[i, k] > m:
                    m = F[i, k]
            s = 0.0
            for k in range(K):
                e = np.exp(F[i, k] - m)
                out[i, k] = e
                s += e
            for k in range(K):
                out[i, k] = out[i, k] / s
        return out

    print("\n--- fused kernel: serial vs parallel (median of "
          f"{repeats}) ---")
    rng = np.random.default_rng(0)
    _serial(rng.normal(size=(64, 3)))              # warm both JITs
    losses._softmax_kernel(rng.normal(size=(64, 3)))
    for shape in shapes:
        F = rng.normal(size=shape)
        out = {}
        for name, fn in (("serial", _serial),
                         ("parallel", losses._softmax_kernel)):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn(F)
                times.append(time.perf_counter() - t0)
            out[name] = statistics.median(times)
        same = bool(np.array_equal(_serial(F), losses._softmax_kernel(F)))
        print(f"  {str(shape):>14s}  serial {out['serial'] * 1e3:6.3f} ms   "
              f"parallel {out['parallel'] * 1e3:6.3f} ms   "
              f"serial/parallel {out['serial'] / out['parallel']:.2f}x   "
              f"same result={same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=2000)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()

    rows = [run_dataset(k, args.repeats, args.n_estimators)
            for k in args.datasets]
    serial_vs_parallel([(32377, 3), (5714, 3), (52755, 3), (2097, 6)],
                       args.repeats * 3)

    print("\n| dataset | task | K | OFF s | ON s | fit-time change "
          "| softmax s OFF -> ON | calls |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['task']} | {r['K']} | "
              f"{r['off_median']:.2f} | {r['on_median']:.2f} | "
              f"{(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{r['softmax_off_s']:.2f} -> {r['softmax_on_s']:.2f} | "
              f"{r['calls']} |")


if __name__ == "__main__":
    main()
