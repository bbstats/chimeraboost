"""M1 gate read: eligible-set detail (F1 + Brier per seed) + cat_multiclass
eligibility verification."""
import json
import sys

sys.path.insert(0, "benchmarks")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from run_benchmarks import DATASETS
from chimeraboost.sklearn_api import CROSS_MIN_SAMPLES

BASE = "benchmarks/results/20260717-195727.json"
NEW = "benchmarks/results/20260717-200023.json"

X, y, cat, task = DATASETS["cat_multiclass"](1.0, np.random.default_rng(0))
n_num = X.shape[1] - len(cat or [])
print(f"cat_multiclass: n={X.shape[0]} d={X.shape[1]} num={n_num} "
      f"K={len(np.unique(y))} n_fit~{int(X.shape[0] * 0.75 * 0.8)} "
      f"(gate {CROSS_MIN_SAMPLES}) -> eligible="
      f"{int(X.shape[0] * 0.75 * 0.8) >= CROSS_MIN_SAMPLES and n_num >= 2}")

ELIG = ["cat_multiclass", "oml:optdigits", "oml:satimage", "oml:pendigits",
        "oml:letter"]


def recs(path):
    out = {}
    for r in json.load(open(path))["records"]:
        if r["model"] == "ChimeraBoost" and r["dataset"] in ELIG:
            out[(r["dataset"], r["seed"])] = r["metrics"]
    return out


b, n = recs(BASE), recs(NEW)
print(f"\n{'set':>16} seed   F1 delta     Brier delta (+ = NEW better)")
f1_rels, br_rels = {}, {}
for ds in ELIG:
    for s in (0, 1, 2):
        bm, nm = b[(ds, s)], n[(ds, s)]
        df = nm["primary"] - bm["primary"]
        db = bm["brier"] - nm["brier"]
        f1_rels.setdefault(ds, []).append(df / abs(bm["primary"]))
        br_rels.setdefault(ds, []).append(db / max(abs(bm["brier"]), 1e-12))
        print(f"{ds:>16}   {s}   {df:+.4f}      {db:+.5f}")
print("\nper-set means (rel):")
for ds in ELIG:
    print(f"{ds:>16}  F1 {np.mean(f1_rels[ds]):+.3%}   "
          f"Brier {np.mean(br_rels[ds]):+.3%}")
print(f"\npooled eligible: F1 {np.mean([np.mean(f1_rels[d]) for d in ELIG]):+.3%}"
      f"   Brier {np.mean([np.mean(br_rels[d]) for d in ELIG]):+.3%}")
