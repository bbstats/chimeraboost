"""M1 tier-1 extension read: 6-seed eligible-slice verdict + validity canary.

1. Validity: per-(set, seed) primary for seeds 0-2 must EXACTLY match the
   original 3-seed runs (BASE vs 20260717-103015, NEW vs 20260717-192856) --
   deterministic per (dataset, seed) on this box, so any drift = bad arm.
2. Verdict (pre-stated in M1_PLAN.md): on the 15-set slice at 6 seeds,
   wins > losses AND mean > 0 on primary; Brier not negative beyond noise.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE6, NEW6 = sys.argv[1], sys.argv[2]
BASE3 = "benchmarks/results/20260717-103015.json"
NEW3 = "benchmarks/results/20260717-192856.json"


def per_set_seed(path, metric="primary"):
    out = {}
    for r in json.load(open(path))["records"]:
        if r["model"] != "ChimeraBoost":
            continue
        if r["metrics"].get(metric) is None:
            continue
        out[(r["dataset"], r["seed"])] = r["metrics"][metric]
    return out


def canary(six, three, label):
    six_d, three_d = per_set_seed(six), per_set_seed(three)
    shared = [k for k in six_d if k in three_d]
    bad = [(k, six_d[k], three_d[k]) for k in shared
           if six_d[k] != three_d[k]]
    print(f"canary {label}: {len(shared)} (set,seed) pairs vs original run; "
          f"{'EXACT MATCH' if not bad else f'MISMATCH {bad[:4]}'}")


canary(BASE6, BASE3, "BASE seeds0-2")
canary(NEW6, NEW3, "NEW  seeds0-2")

for metric in ("primary", "brier"):
    b, n = per_set_seed(BASE6, metric), per_set_seed(NEW6, metric)
    sign = -1.0 if metric == "brier" else 1.0
    sets = defaultdict(list)
    for k in b:
        if k in n:
            sets[k[0]].append((sign * b[k], sign * n[k]))
    wins = losses = ties = 0
    rels = []
    rows = []
    for ds, pairs in sorted(sets.items()):
        bm = float(np.mean([p[0] for p in pairs]))
        nm = float(np.mean([p[1] for p in pairs]))
        d = nm - bm
        rel = d / abs(bm) if abs(bm) > 1e-12 else 0.0
        rels.append(rel)
        if d > 1e-9:
            wins += 1
            tag = "WIN"
        elif d < -1e-9:
            losses += 1
            tag = "loss"
        else:
            ties += 1
            tag = "tie"
        sw = sum(1 for p in pairs if p[1] > p[0] + 1e-12)
        sl = sum(1 for p in pairs if p[1] < p[0] - 1e-12)
        rows.append(f"  {ds:14s} {bm:+.4f} -> {nm:+.4f}  {rel:+.2%}  {tag}"
                    f"  (seed pairs {sw}W-{sl}L)")
    print(f"\n== {metric}, 6 seeds, {len(sets)} sets ==")
    for r in rows:
        print(r)
    print(f"SLICE: {wins}W-{losses}L-{ties}T, mean {np.mean(rels):+.3%}")
    sp_w = sum(1 for ds, pairs in sets.items() for p in pairs
               if p[1] > p[0] + 1e-12)
    sp_l = sum(1 for ds, pairs in sets.items() for p in pairs
               if p[1] < p[0] - 1e-12)
    print(f"pooled (set,seed) pairs: {sp_w}W-{sp_l}L")
