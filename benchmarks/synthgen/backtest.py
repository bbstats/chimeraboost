"""Backtest the frozen synthgen suite against ledger verdicts (adoption gate).

Runs the screen suite baseline + one arm per known-outcome lever, sequentially
(one benchmark at a time), then scores sign agreement. The suite earns its
place in the /experiment protocol only if >= 7/9 arms agree AND the
cat_combinations canary slice (saturated & cats) is not positive.

Usage:
  python benchmarks/synthgen/backtest.py                 # run everything
  python benchmarks/synthgen/backtest.py --arms 1 4 9    # subset
  python benchmarks/synthgen/backtest.py --score-only    # re-score existing JSONs

Each run saves to benchmarks/results/synv1-<arm>.txt/.json (kept out of git).
All models run in every arm (user decision 2026-07-14); verdicts read
ChimeraBoost records only via the --model filter.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH)
RESULTS = os.path.join(_BENCH, "results")
EPS = 1e-9

# (arm#, name, extra run_benchmarks args, expectation, expectation kind)
# kinds: "win" (sign test should lean +), "loss" (lean -), "flat" (|mean|~0),
#        "not_win" (must NOT decisively win)
ARMS = [
    (1, "crossfeat", ["--chimera-cross-features"], "win",
     "cross_features shipped: Grinsztajn 51W/8L +1.5% (signal: crossfeat-scope slice)"),
    (2, "linleaves", ["--chimera-linear-leaves"], "win",
     "linear_leaves shipped for smooth regression (signal: reg, func linear/neural)"),
    (3, "catcombo", ["--chimera-cat-combinations"], "not_win",
     "cat_combinations auto-on ONLY for all-cat data; forced-on is mixed on real "
     "data; canary slice (saturated&cats) must NOT be positive"),
    (4, "patience300", ["--patience", "300"], "flat",
     "patience 300 = flat (cascade selftest anchor)"),
    (5, "orderedboost", ["--chimera-ordered-boosting"], "not_win",
     "ordered boosting tested, never shipped as forced default"),
    (6, "mcw1", ["--chimera-mcw", "1"], "loss",
     "disabling size-adaptive min_child_weight hurts small-n classification"),
    (7, "depth8", ["--chimera-depth", "8"], "not_win",
     "depth U-shape, default 6 at the bottom (PMLB/knob characterization)"),
    (8, "depth4", ["--chimera-depth", "4"], "loss",
     "depth 4 under-capacity on real suites"),
    (9, "lr03", ["--lr", "0.3"], "loss",
     "lr 0.3 = right arm of the U (knob characterization)"),
]


def _run(name, extra, seeds, suite, jobs):
    out = os.path.join(RESULTS, f"synv1-{name}.txt")
    if os.path.exists(out.replace(".txt", ".json")):
        print(f"[{name}] exists, skipping run", flush=True)
        return
    cmd = [sys.executable, os.path.join(_BENCH, "run_benchmarks.py"),
           "--synth", "--synth-suite", suite, "--seeds", str(seeds),
           "--jobs", str(jobs), "--save", out] + extra
    print(f"[{name}] {' '.join(cmd[1:])}", flush=True)
    res = subprocess.run(cmd, cwd=os.path.dirname(_BENCH))
    if res.returncode != 0:
        raise SystemExit(f"arm {name} failed (exit {res.returncode})")


def _per_dataset(path, model="ChimeraBoost"):
    data = json.load(open(path, encoding="utf-8"))
    bucket = defaultdict(list)
    for r in data["records"]:
        if r["model"] == model:
            bucket[r["dataset"]].append(r["metrics"]["primary"])
    return ({ds: float(np.mean(v)) for ds, v in bucket.items()},
            data["datasets"])


def _slice_mean(deltas, metas, pred):
    d = [v for ds, v in deltas.items() if pred(metas[ds]["synth"])]
    return (float(np.mean(d)), len(d)) if d else (0.0, 0)


def score(arm_names):
    base_path = os.path.join(RESULTS, "synv1-baseline.json")
    base, metas = _per_dataset(base_path)
    agree, results = 0, []
    for num, name, extra, kind, why in ARMS:
        if name not in arm_names:
            continue
        path = os.path.join(RESULTS, f"synv1-{name}.json")
        if not os.path.exists(path):
            results.append((num, name, kind, "MISSING", False, ""))
            continue
        new, metas_new = _per_dataset(path)
        for k, v in metas_new.items():
            metas.setdefault(k, v)
        deltas = {}
        for ds in set(base) & set(new):
            b = base[ds]
            deltas[ds] = (new[ds] - b) / max(abs(b), 1e-12)
        d = np.array(list(deltas.values()))
        w, l = int((d > EPS).sum()), int((d < -EPS).sum())
        mean = float(d.mean())
        decisive_win = w > l and mean > 0.001
        decisive_loss = l > w and mean < -0.001
        if kind == "win":
            ok = decisive_win
        elif kind == "loss":
            ok = decisive_loss
        elif kind == "flat":
            ok = abs(mean) < 0.001 and abs(w - l) <= max(3, 0.15 * len(d))
        else:  # not_win
            ok = not decisive_win
        note = f"W{w}-L{l} mean {mean:+.3%}"
        if name == "catcombo":
            cm, cn = _slice_mean(deltas, metas,
                                 lambda s: s["saturated"] and s["n_cat"] > 0)
            canary_ok = cn == 0 or cm <= EPS
            note += f" | canary {cm:+.3%}@{cn} {'OK' if canary_ok else 'FAIL'}"
            ok = ok and canary_ok
        if name == "crossfeat":
            sm, sn = _slice_mean(
                deltas, metas,
                lambda s: (s["task"] in ("regression", "binary")
                           and s["n"] >= 2000 and s["cat_fraction"] < 0.5
                           and s["interaction_depth"] >= 2))
            note += f" | scope-slice {sm:+.3%}@{sn}"
        agree += ok
        results.append((num, name, kind, note, ok, why))

    print("\n== backtest scorecard ==", flush=True)
    for num, name, kind, note, ok, why in results:
        print(f"  {num}. {name:12s} expect={kind:8s} {note:44s} "
              f"[{'AGREE' if ok else 'DISAGREE'}]", flush=True)
    n_run = sum(1 for r in results if r[3] != "MISSING")
    print(f"\nagreement: {agree}/{n_run} arms "
          f"(gate: >=7/9 and canary not positive)", flush=True)
    return agree, n_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", type=int, default=None,
                    help="arm numbers to run (default: all)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--suite", default="screen")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    wanted = {n for n, *_ in ARMS} if args.arms is None else set(args.arms)
    names = [name for num, name, *_ in ARMS if num in wanted]
    if not args.score_only:
        _run("baseline", [], args.seeds, args.suite, args.jobs)
        for num, name, extra, _, _ in ARMS:
            if num in wanted:
                _run(name, extra, args.seeds, args.suite, args.jobs)
    score(names)


if __name__ == "__main__":
    main()
