"""F4 candidate C4a speed read: factorize once per fit, in real default fits.

C4's split (`f4_other_walltime.py --prep-detail`) found that one default fit
factorizes overlapping rows about 2.2 times -- the selection fits' training
rows, the validation rows twice, then every row again in the refit -- and that
factorize alone is 11-30% of fit on the high-cardinality sets. C4a factorizes
the full matrix once and derives each leg's codes by an integer re-rank. This
script asks how much of the ceiling converts into fit time.

Method -- the same-process A/B used for C1, C1b and C2, so the ~1% same-process
floor applies rather than the ~2% cross-run one. One process loads each
dataset once, then alternates:

    OFF: `sklearn_api._shared_cat_ctxs` monkeypatched to return no contexts,
         which sends every leg down the pre-change path (its own `factorize`).
    ON:  the helper live.

Both arms fit the DEFAULT estimator, since the default's selection fits,
calibration and refit are exactly where the repeated passes come from. The
census counts real `factorize` calls and their seconds per arm, so the read
shows the mechanism and not only the total. The panel is four high-card sets
(string-coded and integer-coded categoricals, binary and multiclass) plus a
numeric Grinsztajn set as the zero-change control: without a categorical
column no context is ever built, so that arm must read flat.

Run:
    python benchmarks/f4_c4a_speed.py
    python benchmarks/f4_c4a_speed.py --datasets hc:kick --repeats 7
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

import chimeraboost.preprocessing as pmod
import chimeraboost.sklearn_api as skmod
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

PANEL = ["hc:kick", "hc:sf-police-incidents", "hc:porto-seguro",
         "hc:okcupid-stem", "gr:clf_num/Higgs"]

_ON = skmod._shared_cat_ctxs
_FACTORIZE = pmod.factorize
_CENSUS = {"calls": 0, "seconds": 0.0}


def _off(X_full, split_idx, cat_features):
    return None, None, None


def _counted_factorize(col):
    t0 = time.perf_counter()
    out = _FACTORIZE(col)
    _CENSUS["seconds"] += time.perf_counter() - t0
    _CENSUS["calls"] += 1
    return out


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
    pmod.factorize = _counted_factorize
    try:
        for i in range(repeats):
            for arm, helper, times in (("OFF", _off, off), ("ON", _ON, on)):
                skmod._shared_cat_ctxs = helper
                _CENSUS["calls"], _CENSUS["seconds"] = 0, 0.0
                times.append(_fit_once(Est, Xtr, ytr, cat_idx))
                cens[arm] = dict(_CENSUS)
            print(f"  repeat {i + 1}: OFF {off[-1]:6.3f}s   ON {on[-1]:6.3f}s   "
                  f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    finally:
        skmod._shared_cat_ctxs = _ON
        pmod.factorize = _FACTORIZE

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: {c['calls']} factorize calls, {c['seconds']:.3f}s")
    return {"dataset": key, "task": task, "median_ratio": med,
            "lo": ratios[0], "hi": ratios[-1],
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "fz_off": cens["OFF"], "fz_on": cens["ON"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(skmod.__file__)}", flush=True)

    rows = [run_dataset(k, args.repeats) for k in args.datasets]

    print("\n| dataset | task | OFF s | ON s | fit-time change | range "
          "| factorize calls OFF -> ON | factorize s OFF -> ON |")
    print("|---|---|--:|--:|--:|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['task']} | {r['off_median']:.3f} | "
              f"{r['on_median']:.3f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{(r['lo'] - 1) * 100:+.1f}..{(r['hi'] - 1) * 100:+.1f}% | "
              f"{r['fz_off']['calls']} -> {r['fz_on']['calls']} | "
              f"{r['fz_off']['seconds']:.3f} -> {r['fz_on']['seconds']:.3f} |")


if __name__ == "__main__":
    main()
