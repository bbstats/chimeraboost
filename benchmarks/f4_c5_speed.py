"""F4 candidate C5 speed read: the fused multiclass eval, in real fits.

C5 folds the per-round validation cross-entropy -- softmax, clip, logs,
class sum -- into one numba pass that returns the per-row vector
(`_softmax_ce_kernel`), with the weighted mean left in numpy. The profile
(I026) put `MultiSoftmax.eval` at 4.9-6.5% of a multiclass fit on the hc
sets. This script asks how much of that converts into fit time.

Method -- the same-process A/B used for C1 through C4a-2, so the ~1%
same-process floor applies rather than the ~2% cross-run one. One process
loads each dataset once, then alternates:

    OFF: `MultiSoftmax.eval` monkeypatched to its `_eval_numpy` body,
         which IS the pre-change method.
    ON:  the guarded dispatcher live.

With the lesson from the S1 rung: the arm order alternates each repeat
(odd repeats ON then OFF, even repeats OFF then ON) so a warm-up drift
cannot systematically favor one arm. Both arms fit the DEFAULT estimator
on the harness's 75% split, seed 0. The census wraps `eval` in both arms
and reports its own seconds, so the read shows the mechanism and not only
the total. Every repeat asserts the two arms' fitted outputs are exactly
identical (exact rewrite: any difference is a bug, not noise).

The panel is the four multiclass hc sets plus gr:clf_num/Higgs as the
binary control: it never calls MultiSoftmax, so it must read flat (and
the script asserts its eval census is zero calls).

Run:
    python benchmarks/f4_c5_speed.py
    python benchmarks/f4_c5_speed.py --datasets hc:cjs --repeats 1
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb

import chimeraboost.losses as losses
from chimeraboost import ChimeraBoostClassifier

PANEL = ["hc:okcupid-stem", "hc:Traffic_violations", "hc:cjs",
         "hc:eucalyptus", "gr:clf_num/Higgs"]
BINARY_CONTROL = "gr:clf_num/Higgs"
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "campaign-f4c5-speed-20260923.txt")

_EV_ON = losses.MultiSoftmax.eval
_EV_OFF = losses.MultiSoftmax._eval_numpy
_CENSUS = {"ev_calls": 0, "ev_seconds": 0.0}


def _counted(fn):
    """Wrap an arm so the run can report the eval layer's own time.

    A few hundred calls per fit, so the wrapper costs microseconds in total,
    and it is applied to both arms so whatever it costs it costs
    symmetrically."""
    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(self, *args, **kwargs)
        _CENSUS["ev_seconds"] += time.perf_counter() - t0
        _CENSUS["ev_calls"] += 1
        return out
    return wrapped


def _set_arm(ev):
    losses.MultiSoftmax.eval = _counted(ev)


def _reset_census():
    _CENSUS["ev_calls"] = 0
    _CENSUS["ev_seconds"] = 0.0


def _fit_once(Xtr, ytr, Xho, cat_idx):
    m = ChimeraBoostClassifier(random_state=0)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr, cat_features=cat_idx)
    dt = time.perf_counter() - t0
    return dt, m.predict_proba(Xho), m.best_iteration_


def run_dataset(key, repeats):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xho, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, "
          f"n_features={Xtr.shape[1]}, cats={len(cat_idx) if cat_idx else 0})"
          f" ===", flush=True)

    _fit_once(Xtr, ytr, Xho, cat_idx)  # untimed: JIT + caches, ON arm live

    off, on, cens = [], [], {}
    try:
        for rep in range(1, repeats + 1):
            # S1 lesson: alternate which arm runs first each repeat.
            order = (("OFF", _EV_OFF, off), ("ON", _EV_ON, on))
            if rep % 2 == 1:
                order = (order[1], order[0])
            probas, iters = {}, {}
            for arm, ev, times in order:
                _set_arm(ev)
                _reset_census()
                dt, proba, it = _fit_once(Xtr, ytr, Xho, cat_idx)
                times.append(dt)
                cens[arm] = dict(_CENSUS)
                probas[arm], iters[arm] = proba, it
            assert np.array_equal(probas["OFF"], probas["ON"]), \
                f"{key} repeat {rep}: fitted probas differ between arms"
            assert iters["OFF"] == iters["ON"], \
                f"{key} repeat {rep}: best_iteration differs between arms"
            print(f"  repeat {rep}: OFF {off[-1]:6.3f}s   ON {on[-1]:6.3f}s   "
                  f"ratio {on[-1] / off[-1]:.3f}   "
                  f"({'OFF,ON' if order[0][0] == 'OFF' else 'ON,OFF'}) "
                  f"identical", flush=True)
    finally:
        losses.MultiSoftmax.eval = _EV_ON

    if key == BINARY_CONTROL:
        # The control never calls MultiSoftmax: both censuses must be empty.
        for arm in ("OFF", "ON"):
            assert cens[arm]["ev_calls"] == 0, \
                f"{key}: {arm} called MultiSoftmax.eval {cens[arm]['ev_calls']}x"

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: eval {c['ev_calls']} calls {c['ev_seconds']:.3f}s")
    return {"dataset": key, "task": task, "median_ratio": med,
            "lo": ratios[0], "hi": ratios[-1],
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "off": cens["OFF"], "on": cens["ON"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(losses.__file__)}", flush=True)

    rows = [run_dataset(k, args.repeats) for k in args.datasets]

    table = ["| dataset | task | OFF s | ON s | fit-time change | range "
             "| eval s OFF -> ON | calls ev |",
             "|---|---|--:|--:|--:|---|---|---|"]
    for r in rows:
        o, n = r["off"], r["on"]
        table.append(
            f"| {r['dataset']} | {r['task']} | {r['off_median']:.3f} | "
            f"{r['on_median']:.3f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
            f"{(r['lo'] - 1) * 100:+.1f}..{(r['hi'] - 1) * 100:+.1f}% | "
            f"{o['ev_seconds']:.3f} -> {n['ev_seconds']:.3f} | "
            f"{n['ev_calls']} |")
    print("\n" + "\n".join(table))
    header = ["# F4 C5 speed read: fused MultiSoftmax.eval (one numba pass, "
              "mean in numpy)",
              f"# {time.strftime('%Y-%m-%d %H:%M')}  repeats={args.repeats}  "
              f"chimeraboost from {os.path.dirname(losses.__file__)}",
              "# OFF = eval patched to _eval_numpy; ON = live dispatcher.",
              "# Order alternates per repeat (odd ON,OFF / even OFF,ON); "
              "fitted probas asserted identical every repeat.",
              ""]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(header + table) + "\n")
    print(f"\nwrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
