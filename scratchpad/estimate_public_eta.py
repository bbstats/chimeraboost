"""Rough ETA for the public-suite run, calibrated on the Grinsztajn run.

The harness buffers its output, so a running benchmark gives no progress signal.
This builds a crude cost model from the run that just finished on the same box
with almost the same field, then applies it to the public suite's shapes.

Deliberately crude, and the caveats matter more than the number:
  * Grinsztajn is 36 regression + 23 binary and has NO multiclass, so the 9
    multiclass public sets are extrapolated off the end of the calibration.
  * High-cardinality categoricals (18,210 levels on one) are far outside
    Grinsztajn's range and cost target-encoding work this model cannot see.
  * Early stopping means tree counts vary hugely per dataset; this uses mean
    observed cost per row-feature, which averages that away.
"""
import json
import re
import sys

import numpy as np

RUN = "benchmarks/results/20260801-171224.json"
PLAN = "benchmarks/PUBLIC_PLAN.md"
# The public run's field (no sklearn_HGB).
FIELD = ["ChimeraBoost", "ChimeraBoostEns5", "ChimeraBoostEns8",
         "ChimeraBoostNoRefit", "ChimeraBoostOneLin", "CatBoost", "LightGBM"]

blob = json.load(open(RUN, "r", encoding="utf-8"))
meta = blob["datasets"]

# --- calibrate: mean fit seconds per (row x feature) unit, per model ---------
per_model = {m: [] for m in FIELD}
for r in blob["records"]:
    m = r["model"]
    if m not in per_model:
        continue
    d = meta.get(r["dataset"])
    ft = r.get("fit_time")
    if not d or ft is None:
        continue
    units = d["n_train"] * d["n_features"]
    if units > 0:
        per_model[m].append(ft / units)

rate = {m: float(np.median(v)) for m, v in per_model.items() if v}
print("calibration — median fit seconds per million row-feature units:")
for m in FIELD:
    if m in rate:
        print(f"  {m:22s} {rate[m] * 1e6:8.3f}")

# --- public suite shapes, parsed from the plan table ------------------------
rows = []
for line in open(PLAN, "r", encoding="utf-8"):
    if not line.startswith("| pub:"):
        continue
    cells = [c.strip() for c in line.split("|")]
    name, task, shape = cells[1], cells[3], cells[5]
    mm = re.match(r"([\d,]+)\s*[x×]\s*([\d,]+)", shape)
    if not mm:
        continue
    n = int(mm.group(1).replace(",", ""))
    p = int(mm.group(2).replace(",", ""))
    rows.append((name, task, n, p))

print(f"\npublic suite: {len(rows)} datasets parsed")

SEEDS = 3
TRAIN_FRAC = 0.8          # the harness holds out a test split
total = 0.0
for name, task, n, p in rows:
    n_tr = int(n * TRAIN_FRAC)
    for m in FIELD:
        if m not in rate:
            continue
        total += rate[m] * n_tr * p * SEEDS

JOBS = 5
wall_h = total / JOBS / 3600.0
print(f"\nestimated total fit-seconds : {total:,.0f}")
print(f"estimated wall hours at jobs={JOBS} : {wall_h:.1f} h")
print("\n(fit time only — excludes data load, preprocessing, prediction and "
      "scoring, and assumes perfect packing across the 5 workers)")

# --- self-check: does the model reproduce the run it was calibrated on? -----
# The Grinsztajn run took ~28 min wall (17:12 -> ~17:40) with 8 models. If the
# model badly under-predicts a run it has seen, its public number is a floor,
# not an estimate -- which is the only claim worth making from it.
gr_fit = sum(r["fit_time"] for r in blob["records"]
             if r.get("fit_time") is not None)
print(f"\nself-check on the Grinsztajn run it was calibrated on:")
print(f"  summed actual fit-seconds across all models/seeds : {gr_fit:,.0f}")
print(f"  implied wall hours at jobs=5                      : "
      f"{gr_fit / JOBS / 3600.0:.2f} h")
print(f"  ACTUAL observed wall time                         : ~0.47 h (28 min)")
print(f"  => non-fit overhead + packing loss inflate wall time by "
      f"{0.47 / max(gr_fit / JOBS / 3600.0, 1e-9):.1f}x")
