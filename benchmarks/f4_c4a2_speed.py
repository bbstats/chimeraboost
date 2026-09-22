"""F4 candidate C4a-2 speed read: the numeric block cast once per fit.

C4a (PR #125) factorizes the categorical columns once per fit and derives each
leg's codes from that pass. The float64 numeric block rode on the same
contexts but was still cast on every leg -- the selection fits' training
rows, the validation rows, the cross candidate twice, calibration, then every
row again in the refit -- for 4.7% of fit on hc:kick and 10.1% on
hc:porto-seguro (I026). C4a-2 casts the full matrix once and gathers rows per
leg. This script asks how much of that converts into fit time.

Method -- the same-process A/B used for C1, C1b, C2 and C4a, so the ~1%
same-process floor applies rather than the ~2% cross-run one. One process
loads each dataset once, then alternates:

    OFF: `FeaturePreprocessor._numeric_block` monkeypatched to ignore the
         context and cast the leg it is handed, the pre-change path. The
         categorical sharing from C4a stays ON in both arms, so the read
         isolates the numeric block.
    ON:  the method live.

Both arms fit the DEFAULT estimator. The census counts real
`_cast_numeric_block` calls and their seconds per arm, so the read shows the
mechanism and not only the total. The panel is the four high-card sets C4a
used (kick and porto-seguro carry the numeric cost; sf-police and
okcupid-stem are under 1% and read as controls) plus a numeric Grinsztajn set
as the zero-change control: without a categorical column no context is ever
built, so that arm must read flat.

Run:
    python benchmarks/f4_c4a2_speed.py
    python benchmarks/f4_c4a2_speed.py --datasets hc:porto-seguro --repeats 7
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

_ON = pmod.FeaturePreprocessor._numeric_block
_CAST = pmod._cast_numeric_block
_CENSUS = {"calls": 0, "seconds": 0.0}


def _off(self, X, cat_ctx=None):
    """The pre-change path: cast the leg at hand, whatever context is given."""
    return pmod._cast_numeric_block(X, self.num_features_)


def _counted_cast(X, num_features):
    t0 = time.perf_counter()
    out = _CAST(X, num_features)
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
    pmod._cast_numeric_block = _counted_cast
    try:
        for i in range(repeats):
            for arm, method, times in (("OFF", _off, off), ("ON", _ON, on)):
                pmod.FeaturePreprocessor._numeric_block = method
                _CENSUS["calls"], _CENSUS["seconds"] = 0, 0.0
                times.append(_fit_once(Est, Xtr, ytr, cat_idx))
                cens[arm] = dict(_CENSUS)
            print(f"  repeat {i + 1}: OFF {off[-1]:6.3f}s   ON {on[-1]:6.3f}s   "
                  f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    finally:
        pmod.FeaturePreprocessor._numeric_block = _ON
        pmod._cast_numeric_block = _CAST

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    for arm in ("OFF", "ON"):
        c = cens[arm]
        print(f"  {arm}: {c['calls']} numeric casts, {c['seconds']:.3f}s")
    return {"dataset": key, "task": task, "median_ratio": med,
            "lo": ratios[0], "hi": ratios[-1],
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "nb_off": cens["OFF"], "nb_on": cens["ON"]}


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
          "| numeric casts OFF -> ON | cast s OFF -> ON |")
    print("|---|---|--:|--:|--:|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['task']} | {r['off_median']:.3f} | "
              f"{r['on_median']:.3f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{(r['lo'] - 1) * 100:+.1f}..{(r['hi'] - 1) * 100:+.1f}% | "
              f"{r['nb_off']['calls']} -> {r['nb_on']['calls']} | "
              f"{r['nb_off']['seconds']:.3f} -> {r['nb_on']['seconds']:.3f} |")


if __name__ == "__main__":
    main()
