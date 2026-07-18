"""M1: measured fit-time tax on the gate's eligible multiclass sets."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE = "benchmarks/results/20260717-195727.json"
NEW = "benchmarks/results/20260717-200023.json"
ELIG = ["cat_multiclass", "oml:optdigits", "oml:satimage", "oml:pendigits",
        "oml:letter"]


def fit_means(path):
    acc = defaultdict(list)
    for r in json.load(open(path))["records"]:
        if r["model"] == "ChimeraBoost" and r["dataset"] in ELIG:
            acc[r["dataset"]].append(r["fit_time"])
    return {d: float(np.mean(v)) for d, v in acc.items()}


b, n = fit_means(BASE), fit_means(NEW)
for d in ELIG:
    print(f"{d:>16}: {b[d]:6.2f}s -> {n[d]:6.2f}s  ({n[d] / b[d]:.2f}x)")
print(f"{'sum':>16}: {sum(b.values()):6.2f}s -> {sum(n.values()):6.2f}s  "
      f"({sum(n.values()) / sum(b.values()):.2f}x)")
