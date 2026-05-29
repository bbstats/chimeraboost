"""Sign-test two run_benchmarks JSONs against each other (same model).

Usage:
    python benchmarks/compare_runs.py BASE.json NEW.json [base_label new_label]

Compares the per-dataset mean of the 'primary' metric (always higher-is-better:
negative RMSE for regression, F1/accuracy for classification). Reports per-dataset
deltas and a sign test (how many datasets NEW beats BASE).
"""
import json
import sys
from collections import defaultdict

import numpy as np


def per_dataset_primary(path):
    recs = json.load(open(path))["records"]
    bucket = defaultdict(list)
    for r in recs:
        bucket[r["dataset"]].append(r["metrics"]["primary"])
    return {ds: float(np.mean(v)) for ds, v in bucket.items()}


def main():
    base_path, new_path = sys.argv[1], sys.argv[2]
    base_label = sys.argv[3] if len(sys.argv) > 3 else "BASE"
    new_label = sys.argv[4] if len(sys.argv) > 4 else "NEW"

    base = per_dataset_primary(base_path)
    new = per_dataset_primary(new_path)
    shared = sorted(set(base) & set(new))

    wins = losses = ties = 0
    print(f"{'dataset':22s} {base_label:>12s} {new_label:>12s} {'delta':>12s}  result")
    rel_deltas = []
    for ds in shared:
        b, n = base[ds], new[ds]
        d = n - b                       # primary is higher-better
        # relative improvement (guard tiny/zero base)
        rel = d / abs(b) if abs(b) > 1e-12 else 0.0
        rel_deltas.append(rel)
        if d > 1e-9:
            wins += 1; tag = f"{new_label} wins"
        elif d < -1e-9:
            losses += 1; tag = f"{base_label} wins"
        else:
            ties += 1; tag = "tie"
        print(f"{ds:22s} {b:12.4f} {n:12.4f} {d:+12.4f}  {tag}  ({rel:+.2%})")

    n = len(shared)
    print(f"\n{new_label} vs {base_label}: {wins} wins / {losses} losses / {ties} ties  "
          f"(of {n} datasets)")
    print(f"mean relative change in primary: {np.mean(rel_deltas):+.3%}")
    need = n // 2 + 1
    verdict = "PASS" if wins >= need else "FAIL"
    print(f"sign-test bar (> half = {need}+ wins): {verdict}")


if __name__ == "__main__":
    main()
