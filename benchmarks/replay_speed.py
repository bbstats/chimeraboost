"""Summed ChimeraBoost fit time across two runs, for the REPLAY A/B.

The sign test reads accuracy only; this reads the other axis. Sums per-dataset
mean fit time over the datasets both runs share, so a dataset that failed to
load in one arm cannot skew the ratio.

    python benchmarks/replay_speed.py BASE.json NEW.json [--model ChimeraBoost]
"""
import argparse
import collections
import json

import numpy as np


def load(path, model):
    recs = json.load(open(path, encoding="utf-8"))
    rows = recs["records"] if isinstance(recs, dict) else recs
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("model") != model:
            continue
        t = r.get("fit_time", r.get("fit_secs"))
        if t is not None:
            by[r["dataset"]].append(float(t))
    meta = recs.get("config", {}) if isinstance(recs, dict) else {}
    return {k: float(np.mean(v)) for k, v in by.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--model", default="ChimeraBoost")
    a = ap.parse_args()

    b, mb = load(a.base, a.model)
    n, mn = load(a.new, a.model)
    for m, name in ((mb, "base"), (mn, "new")):
        if m.get("timing") and m["timing"] != "fit_only":
            print(f"WARNING: {name} run stamped timing={m['timing']!r} "
                  f"(not fit_only) -- ratios mix fit and scoring.")
    shared = sorted(set(b) & set(n))
    if not shared:
        raise SystemExit("no shared datasets")
    tb = sum(b[k] for k in shared)
    tn = sum(n[k] for k in shared)
    ratios = np.array([n[k] / b[k] for k in shared if b[k] > 0])

    print(f"{a.model} fit time over {len(shared)} shared datasets")
    print(f"  base summed  {tb:9.1f}s")
    print(f"  new  summed  {tn:9.1f}s   ({tn / tb:.3f}x, "
          f"{(1 - tn / tb) * 100:+.1f}%)")
    print(f"  per-dataset ratio: median {np.median(ratios):.3f}x  "
          f"mean {ratios.mean():.3f}x  "
          f"[p10 {np.percentile(ratios, 10):.3f} "
          f"p90 {np.percentile(ratios, 90):.3f}]")
    faster = int((ratios < 1.0).sum())
    print(f"  faster on {faster}/{len(ratios)} datasets")


if __name__ == "__main__":
    main()
