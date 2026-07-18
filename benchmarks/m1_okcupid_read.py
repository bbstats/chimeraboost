"""M1 tier-2 hc: paired per-seed okcupid-stem read (F1 + Brier), both arms."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = "benchmarks/results/20260717-155202.json"
NEW = "benchmarks/results/20260717-193744.json"


def recs(path, model, ds="hc:okcupid-stem"):
    out = {}
    for r in json.load(open(path))["records"]:
        if r["model"] == model and r["dataset"] == ds:
            out[r["seed"]] = r
    return out


for model in ("ChimeraBoost", "ChimeraBoostEns8"):
    b, n = recs(BASE, model), recs(NEW, model)
    print(f"== {model} on hc:okcupid-stem (paired seeds) ==")
    for s in sorted(b):
        bm, nm = b[s]["metrics"], n[s]["metrics"]
        print(f"  seed {s}: F1 {bm['primary']:.4f} -> {nm['primary']:.4f} "
              f"({nm['primary'] - bm['primary']:+.4f})   "
              f"Brier {bm['brier']:.4f} -> {nm['brier']:.4f} "
              f"({bm['brier'] - nm['brier']:+.4f} = + is NEW better)   "
              f"fit {b[s]['fit_time']:.1f}s -> {n[s]['fit_time']:.1f}s")
