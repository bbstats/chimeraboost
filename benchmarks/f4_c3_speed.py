"""F4 candidate C3 speed read: the fused binary Logloss layer, in real fits.

C3 folds `grad = p - y` and `hess = max(p * (1 - p), 1e-6)` into the numba
sigmoid pass (`_logloss_grad_hess_kernel`) and the per-round validation
cross-entropy -- sigmoid, clip, logs, products -- into one pass that returns
the per-row vector (`_logloss_ce_kernel`), with the mean left in numpy. The
profile (I023) put `grad_hess` at 4.2-4.9% and `eval` at 4.0-5.1% of a
Grinsztajn binary fit. This script asks how much of that converts into fit
time.

Method -- the same-process A/B used for C1, C1b, C2, C4a and C4a-2, so the
~1% same-process floor applies rather than the ~2% cross-run one. One process
loads each dataset once, then alternates:

    OFF: `Logloss.grad_hess` and `Logloss.eval` monkeypatched to their
         `_numpy` bodies, which ARE the pre-change methods.
    ON:  the guarded dispatchers live.

Both arms fit the DEFAULT estimator. The census wraps both methods in both
arms and reports their own seconds, so the read shows the mechanism and not
only the total. The panel is the three Grinsztajn binary sets the profile
named plus hc:kick (binary, high-card), with a regression set and a
multiclass set as zero-change controls: `Logloss` is not their loss, so those
arms must read flat.

Run:
    python benchmarks/f4_c3_speed.py
    python benchmarks/f4_c3_speed.py --datasets gr:clf_num/Higgs --repeats 7
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
from chimeraboost import (ChimeraBoostClassifier,
                          ChimeraBoostRegressor)

PANEL = ["gr:clf_num/MagicTelescope", "gr:clf_num/Higgs",
         "gr:clf_cat/road-safety", "hc:kick",
         "gr:reg_num/cpu_act", "hc:okcupid-stem"]

_GH_ON = losses.Logloss.grad_hess
_GH_OFF = losses.Logloss._grad_hess_numpy
_EV_ON = losses.Logloss.eval
_EV_OFF = losses.Logloss._eval_numpy
_CENSUS = {"gh_calls": 0, "gh_seconds": 0.0, "ev_calls": 0, "ev_seconds": 0.0}


def _counted(fn, tag):
    """Wrap an arm so the run can report the loss layer's own time.

    A few hundred calls per fit, so the wrapper costs microseconds in total,
    and it is applied to both arms so whatever it costs it costs
    symmetrically."""
    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(self, *args, **kwargs)
        _CENSUS[tag + "_seconds"] += time.perf_counter() - t0
        _CENSUS[tag + "_calls"] += 1
        return out
    return wrapped


def _set_arm(gh, ev):
    losses.Logloss.grad_hess = _counted(gh, "gh")
    losses.Logloss.eval = _counted(ev, "ev")


def _reset_census():
    for k in _CENSUS:
        _CENSUS[k] = 0 if k.endswith("calls") else 0.0


def _fit_once(Est, Xtr, ytr, cat_idx):
    t0 = time.perf_counter()
    Est(random_state=0).fit(Xtr, ytr, cat_features=cat_idx)
    return time.perf_counter() - t0


def run_dataset(key, repeats):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    Est = ChimeraBoostRegressor if task == "regression" \
        else ChimeraBoostClassifier
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, "
          f"n_features={Xtr.shape[1]}, cats={len(cat_idx) if cat_idx else 0})"
          f" ===", flush=True)

    _fit_once(Est, Xtr, ytr, cat_idx)      # untimed: JIT + caches, ON arm live

    off, on, cens = [], [], {}
    try:
        for i in range(repeats):
            for arm, gh, ev, times in (("OFF", _GH_OFF, _EV_OFF, off),
                                       ("ON", _GH_ON, _EV_ON, on)):
                _set_arm(gh, ev)
                _reset_census()
                times.append(_fit_once(Est, Xtr, ytr, cat_idx))
                cens[arm] = dict(_CENSUS)
            print(f"  repeat {i + 1}: OFF {off[-1]:6.3f}s   ON {on[-1]:6.3f}s   "
                  f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    finally:
        losses.Logloss.grad_hess = _GH_ON
        losses.Logloss.eval = _EV_ON

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: grad_hess {c['gh_calls']} calls {c['gh_seconds']:.3f}s"
              f"   eval {c['ev_calls']} calls {c['ev_seconds']:.3f}s")
    return {"dataset": key, "task": task, "median_ratio": med,
            "lo": ratios[0], "hi": ratios[-1],
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "off": cens["OFF"], "on": cens["ON"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(losses.__file__)}", flush=True)

    rows = [run_dataset(k, args.repeats) for k in args.datasets]

    print("\n| dataset | task | OFF s | ON s | fit-time change | range "
          "| grad_hess s OFF -> ON | eval s OFF -> ON | calls gh/ev |")
    print("|---|---|--:|--:|--:|---|---|---|---|")
    for r in rows:
        o, n = r["off"], r["on"]
        print(f"| {r['dataset']} | {r['task']} | {r['off_median']:.3f} | "
              f"{r['on_median']:.3f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{(r['lo'] - 1) * 100:+.1f}..{(r['hi'] - 1) * 100:+.1f}% | "
              f"{o['gh_seconds']:.3f} -> {n['gh_seconds']:.3f} | "
              f"{o['ev_seconds']:.3f} -> {n['ev_seconds']:.3f} | "
              f"{n['gh_calls']}/{n['ev_calls']} |")


if __name__ == "__main__":
    main()
