"""M1 tier-1 screen read: slice the synth A/B by the registered partition.

Slices (benchmarks/M1_PLAN.md):
  eligible     multiclass sets clearing the cross gates -> the treatment read
               (primary F1 + multiclass Brier, sign test + mean)
  mc-inelig    multiclass sets under the gates           -> must be EXACT TIES
  reg/binary   untouched code paths                      -> must be EXACT TIES
  canaries     suites.CANARIES                           -> must be EXACT TIES

Usage: python benchmarks/m1_screen_read.py BASE.json NEW.json [--model M]
"""
import argparse
import sys

sys.path.insert(0, "benchmarks")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from compare_runs import per_dataset_metric
from synthgen import api, suites
from chimeraboost.sklearn_api import CROSS_MIN_SAMPLES


def eligible_partition():
    elig, mc_inelig, other = set(), set(), set()
    for key in suites.frozen_keys("screen"):
        if api.task_of(key) != "multiclass":
            other.add(key)
            continue
        X, y, cat, task, meta = api.build_dataset(key)
        n_num = X.shape[1] - len(cat or [])
        n_fit = int(X.shape[0] * 0.75 * 0.8)
        if n_fit >= CROSS_MIN_SAMPLES and n_num >= 2:
            elig.add(key)
        else:
            mc_inelig.add(key)
    return elig, mc_inelig, other


def slice_report(name, keys, base, new, must_tie):
    shared = sorted(set(base) & set(new) & keys)
    if not shared:
        print(f"{name}: no shared sets")
        return
    wins = losses = ties = 0
    rels, broken = [], []
    for ds in shared:
        d = new[ds] - base[ds]
        rels.append(d / abs(base[ds]) if abs(base[ds]) > 1e-12 else 0.0)
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
        if abs(d) > 1e-9 and must_tie:
            broken.append((ds, d))
    line = (f"{name}: {wins}W-{losses}L-{ties}T of {len(shared)}, "
            f"mean {np.mean(rels):+.3%}")
    if must_tie:
        line += "  [MUST TIE: " + ("OK" if not broken else
                                   f"BROKEN {broken[:5]}") + "]"
    print(line)
    if not must_tie:
        deltas = sorted(((new[ds] - base[ds]) /
                         (abs(base[ds]) if abs(base[ds]) > 1e-12 else 1.0), ds)
                        for ds in shared)
        for r, ds in deltas[:3]:
            print(f"    worst {ds}: {r:+.2%}")
        for r, ds in deltas[-3:]:
            print(f"    best  {ds}: {r:+.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_path")
    ap.add_argument("new_path")
    ap.add_argument("--model", default="ChimeraBoost")
    args = ap.parse_args()

    elig, mc_inelig, other = eligible_partition()
    canaries = {api.key_for(i) for i in suites.CANARIES}

    for metric in ("primary", "brier"):
        print(f"== metric: {metric} ==")
        base = per_dataset_metric(args.base_path, args.model, metric)
        new = per_dataset_metric(args.new_path, args.model, metric)
        slice_report("eligible-mc ", elig, base, new, must_tie=False)
        slice_report("mc-ineligible", mc_inelig, base, new, must_tie=True)
        slice_report("reg/binary  ", other, base, new, must_tie=True)
        slice_report("canaries    ", canaries, base, new, must_tie=True)
        print()


if __name__ == "__main__":
    main()
