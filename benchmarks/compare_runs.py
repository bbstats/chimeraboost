"""Sign-test two run_benchmarks JSONs against each other (same model).

Usage:
    python benchmarks/compare_runs.py BASE.json NEW.json [base_label new_label]
                                      [--model ChimeraBoost]

Compares the per-dataset mean of the 'primary' metric (always higher-is-better:
negative RMSE for regression, F1/accuracy for classification). Reports per-dataset
deltas and a sign test (how many datasets NEW beats BASE).

--model filters records to one model first. Without it, a multi-model JSON
blends every model's records into the per-dataset mean (fine when both runs
hold the other models fixed, but the deltas are diluted).
--model-new names the NEW run's records when they differ (e.g. baseline
ChimeraBoost vs an arm's ChimeraBoostEns2).
--metric brier judges on Brier instead: classification sets only (regression
records carry no Brier), oriented so NEW wins = lower Brier.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def per_dataset_metric(path, model=None, metric="primary"):
    recs = json.load(open(path))["records"]
    sign = -1.0 if metric == "brier" else 1.0   # orient higher = better
    bucket = defaultdict(list)
    for r in recs:
        if model is not None and r["model"] != model:
            continue
        if r["metrics"].get(metric) is None:
            continue
        bucket[r["dataset"]].append(sign * r["metrics"][metric])
    return {ds: float(np.mean(v)) for ds, v in bucket.items()}


def per_dataset_primary(path, model=None):
    return per_dataset_metric(path, model, "primary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_path")
    ap.add_argument("new_path")
    ap.add_argument("base_label", nargs="?", default="BASE")
    ap.add_argument("new_label", nargs="?", default="NEW")
    ap.add_argument("--model", default=None,
                    help="restrict to one model's records (e.g. ChimeraBoost).")
    ap.add_argument("--model-new", default=None,
                    help="model name for the NEW run's records (default: --model).")
    ap.add_argument("--metric", choices=["primary", "brier"], default="primary",
                    help="judge metric; brier = classification only, "
                         "oriented so NEW wins = lower Brier.")
    args = ap.parse_args()
    base_label, new_label = args.base_label, args.new_label

    base = per_dataset_metric(args.base_path, args.model, args.metric)
    new = per_dataset_metric(args.new_path, args.model_new or args.model,
                             args.metric)
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
    mtag = args.model or ""
    if args.model_new and args.model_new != args.model:
        mtag = f"{mtag}->{args.model_new}"
    print(f"\n{new_label} vs {base_label}: {wins} wins / {losses} losses / {ties} ties  "
          f"(of {n} datasets)"
          + (f"  [model={mtag}]" if mtag else ""))
    print(f"mean relative change in {args.metric} (+ = better): "
          f"{np.mean(rel_deltas):+.3%}")
    need = n // 2 + 1
    verdict = "PASS" if wins >= need else "FAIL"
    print(f"sign-test bar (> half = {need}+ wins): {verdict}")


if __name__ == "__main__":
    main()
