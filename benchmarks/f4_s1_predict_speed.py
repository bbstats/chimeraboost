"""F4 S1 speed read: direct per-row categorical codes at predict time.

`FeaturePreprocessor._codes_for_transform` factorizes each categorical column
of the predict batch (hashing every row) only to map the uniques through the
fit-time dict. For a single model that factorization is pure overhead: one
`dict.get` per row (`_direct_codes`) yields the same codes. This script asks
how much predict time that saves.

Method -- the same-process A/B used for C4a, so the ~1% same-process floor
applies. One process loads each dataset once, fits the DEFAULT estimator once
on the harness's 75% split (seed 0), then times predict on the 25% test rows:

    OFF: `_direct_codes` monkeypatched to return None (pre-change path).
    ON:  the helper live.

15 repeats per arm with alternating order (even repeats OFF then ON, odd
repeats ON then OFF) so neither arm always runs on warm caches. Each timing
averages enough consecutive predicts to cover ~100ms, so a fixed scheduling
overhead cannot dominate a 13ms control. The report shows median predict
seconds per arm, the ratio ON/OFF, and asserts the two arms' outputs are
identical. Panel: four string-coded high-card sets (the win),
hc:porto-seguro (its categoricals arrive from the harness as numeric STRINGS,
so the direct path engages there too -- not a flat control), kick-intcoded
(hc:kick re-coded to Python ints: the numeric-coded case), allcat-combos
(hc:kick's categorical columns only, so cat_combinations turns on by default:
the combo case), and gr:clf_num/Higgs (numeric control: no categorical
column, must read flat).

Run:
    python benchmarks/f4_s1_predict_speed.py
    python benchmarks/f4_s1_predict_speed.py --datasets hc:kick --repeats 15
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
from chimeraboost.target_encoding import factorize

PANEL = ["hc:kick", "hc:okcupid-stem", "hc:sf-police-incidents",
         "hc:Traffic_violations", "hc:porto-seguro", "gr:clf_num/Higgs",
         "kick-intcoded", "allcat-combos"]

_ON = pmod._direct_codes


def _off(col, mapping):
    return None


def _intcode_kick(X0, y, cat0, task):
    """hc:kick with every categorical column re-coded to Python ints."""
    X = np.array(X0, dtype=object, copy=True)
    for f in cat0:
        c, _ = factorize(X[:, f])
        X[:, f] = [int(v) for v in c.tolist()]
    return X, y, list(cat0), task


def _allcat_kick(X0, y, cat0, task):
    """hc:kick's categorical columns only (all-categorical, combos on)."""
    X = np.array(X0[:, list(cat0)], dtype=object, copy=True)
    return X, y, list(range(X.shape[1])), task


def _load_panel_data(key):
    if key in ("kick-intcoded", "allcat-combos"):
        X0, y, cat0, task = rb.DATASETS["hc:kick"](
            1, np.random.default_rng(0))
        if key == "kick-intcoded":
            return _intcode_kick(X0, y, cat0, task)
        return _allcat_kick(X0, y, cat0, task)
    return rb.DATASETS[key](1, np.random.default_rng(0))


def _predict_once(est, Xte, is_clf):
    t0 = time.perf_counter()
    pred = est.predict(Xte)
    dt = time.perf_counter() - t0
    proba = est.predict_proba(Xte) if is_clf else None
    return dt, pred, proba


def run_dataset(key, repeats):
    X, y, cat_idx, task = _load_panel_data(key)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    Est = ChimeraBoostRegressor if task == "regression" \
        else ChimeraBoostClassifier
    is_clf = task != "regression"
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, n_test={len(Xte)}, "
          f"n_features={Xtr.shape[1]}, cats={len(cat_idx) if cat_idx else 0})"
          f" ===", flush=True)

    est = Est(random_state=0)
    est.fit(Xtr, ytr, cat_features=cat_idx)
    if key == "allcat-combos":
        n_combo = len(est.model_.prep_.combo_pairs_)
        print(f"  combos engaged: {n_combo} pairs", flush=True)
    dt_on, _, _ = _predict_once(est, Xte, is_clf)
    pmod._direct_codes = _off
    try:
        dt_off, _, _ = _predict_once(est, Xte, is_clf)
    finally:
        pmod._direct_codes = _ON
    single = max(dt_on, dt_off, 1e-6)
    n_inner = max(1, min(10, int(0.1 / single)))

    off, on = [], []
    ref_pred, ref_proba = None, None
    try:
        for i in range(repeats):
            if i % 2 == 0:
                order = (("OFF", _off, off), ("ON", _ON, on))
            else:
                order = (("ON", _ON, on), ("OFF", _off, off))
            for arm, helper, times in order:
                pmod._direct_codes = helper
                dt_sum = 0.0
                for _ in range(n_inner):
                    dt, pred, proba = _predict_once(est, Xte, is_clf)
                    dt_sum += dt
                    if ref_pred is None:
                        ref_pred, ref_proba = pred, proba
                    else:
                        assert np.array_equal(pred, ref_pred), \
                            f"{key} {arm} predict differs from reference"
                        if is_clf:
                            assert np.array_equal(proba, ref_proba), \
                                f"{key} {arm} predict_proba differs"
                times.append(dt_sum / n_inner)
            print(f"  repeat {i + 1}: OFF {off[-1]:6.3f}s   ON {on[-1]:6.3f}s   "
                  f"ratio {on[-1] / off[-1]:.3f}  (x{n_inner})", flush=True)
    finally:
        pmod._direct_codes = _ON

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% predict time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    print("  outputs identical between arms: yes")
    return {"dataset": key, "task": task, "median_ratio": med,
            "lo": ratios[0], "hi": ratios[-1],
            "off_median": statistics.median(off),
            "on_median": statistics.median(on)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(skmod.__file__)}", flush=True)

    rows = [run_dataset(k, args.repeats) for k in args.datasets]

    lines = []
    lines.append("| dataset | task | OFF s | ON s | predict-time change | range "
                 "| identical |")
    lines.append("|---|---|--:|--:|--:|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['task']} | {r['off_median']:.3f} | "
            f"{r['on_median']:.3f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
            f"{(r['lo'] - 1) * 100:+.1f}..{(r['hi'] - 1) * 100:+.1f}% | yes |")
    table = "\n".join(lines)
    print("\n" + table)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results",
                            "campaign-f4s1-predict-speed-20260922.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# F4 S1 predict-cat-lookup speed read ({args.repeats} repeats/arm)\n"
                f"# chimeraboost from {os.path.dirname(skmod.__file__)}\n\n"
                f"{table}\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
