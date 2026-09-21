"""F4 candidate C1b speed read: fused multiclass grad/hess kernel, in real fits.

C1b folds `grad = P - Y` and `hess = max(P * (1 - P), 1e-6)` into the numba
softmax kernel so `MultiSoftmax.grad_hess` makes one pass instead of three.
The ceiling script (`f4_c1b_walltime.py`) put that arithmetic at 4.4% of an
hc:okcupid-stem fit and 7.5% of hc:cjs. This script asks how much of that
converts into fit time.

Method -- the same-process A/B used for C1 and C2, so the ~1% same-process
floor applies rather than the ~2% cross-run one. One process loads each
dataset once, then alternates:

    OFF: `MultiSoftmax.grad_hess` monkeypatched to `_grad_hess_numpy`, which
         IS the pre-change method body, so the OFF arm is the old library.
    ON:  the guarded dispatcher live.

The panel is the three multiclass shapes the decision tier contains plus a
binary set as the zero-change control: `MultiSoftmax` never runs on binary
data, so that arm must read flat, and if it does not the multiclass numbers
are discounted by the same drift.

Run:
    python benchmarks/f4_c1b_speed.py
    python benchmarks/f4_c1b_speed.py --datasets hc:cjs --repeats 5
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

import chimeraboost.losses as losses  # noqa: E402
from chimeraboost import (ChimeraBoostClassifier,  # noqa: E402
                          ChimeraBoostRegressor)

PANEL = ["hc:okcupid-stem", "hc:Traffic_violations", "hc:cjs", "hc:kick"]

_ON = losses.MultiSoftmax.grad_hess
_OFF = losses.MultiSoftmax._grad_hess_numpy
_CENSUS = {"calls": 0, "seconds": 0.0}


def _counted(fn):
    """Wrap an arm so the run can report the loss layer's own time.

    A few hundred calls per fit, so the wrapper costs microseconds in total,
    and it is applied to both arms so whatever it costs it costs
    symmetrically."""
    def wrapped(self, Y, F):
        t0 = time.perf_counter()
        out = fn(self, Y, F)
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
            losses.MultiSoftmax.grad_hess = _counted(fn)
            _CENSUS["calls"], _CENSUS["seconds"] = 0, 0.0
            times.append(_fit_once(Est, Xtr, ytr, cat_idx, n_estimators))
            cens[arm] = dict(_CENSUS)
        print(f"  repeat {i + 1}: OFF {off[-1]:6.2f}s   ON {on[-1]:6.2f}s   "
              f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    losses.MultiSoftmax.grad_hess = _ON

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    print(f"  OFF median {statistics.median(off):6.2f}s   "
          f"ON median {statistics.median(on):6.2f}s")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: {c['calls']} grad_hess calls, {c['seconds']:.3f}s in "
              f"grad_hess")
    return {"dataset": key, "task": task, "K": K, "median_ratio": med,
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "gh_off_s": cens["OFF"]["seconds"],
            "gh_on_s": cens["ON"]["seconds"],
            "calls": cens["ON"]["calls"]}


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

    print("\n| dataset | task | K | OFF s | ON s | fit-time change "
          "| grad_hess s OFF -> ON | calls |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['task']} | {r['K']} | "
              f"{r['off_median']:.2f} | {r['on_median']:.2f} | "
              f"{(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{r['gh_off_s']:.2f} -> {r['gh_on_s']:.2f} | "
              f"{r['calls']} |")


if __name__ == "__main__":
    main()
