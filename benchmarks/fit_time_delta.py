"""Sum a model's fit seconds in two benchmark runs and print the delta.

The pareto slowdown column is LightGBM-normalized, so run-to-run LightGBM
drift pollutes cross-run speed reads; this compares raw summed fit_time on
the intersection of (dataset, seed) pairs instead.

Run: python benchmarks/fit_time_delta.py BASE.json NEW.json --model ChimeraBoostEns5
"""
import argparse
import json


def load(path, model):
    with open(path) as f:
        data = json.load(f)
    return {(r["dataset"], r["seed"]): r["fit_time"]
            for r in data["records"] if r["model"] == model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--model", default="ChimeraBoostEns5")
    args = ap.parse_args()
    b, n = load(args.base, args.model), load(args.new, args.model)
    keys = sorted(set(b) & set(n))
    tb, tn = sum(b[k] for k in keys), sum(n[k] for k in keys)
    print(f"{args.model} on {len(keys)} (dataset,seed) pairs:")
    print(f"  BASE {tb:8.1f}s   NEW {tn:8.1f}s   NEW/BASE {tn / tb:.3f}")
    by_task = {}
    for k in keys:
        pre = k[0].split(":")[0] + ":" + (k[0].split("/")[0].split(":")[1]
                                          if "/" in k[0] else "")
        by_task.setdefault(pre, [0.0, 0.0])
        by_task[pre][0] += b[k]
        by_task[pre][1] += n[k]
    for pre, (pb, pn) in sorted(by_task.items()):
        print(f"    {pre:12s} BASE {pb:7.1f}s  NEW {pn:7.1f}s  ratio {pn / pb:.3f}")


if __name__ == "__main__":
    main()
