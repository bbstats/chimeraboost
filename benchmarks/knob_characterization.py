"""
benchmarks/knob_characterization.py — does each hyperparameter knob actually
WORK, is it HELPFUL, and how much?

Motivation: `depth` was once a SILENT NO-OP. The oblivious min_child_weight veto
capped realized tree depth at ~4 regardless of the requested `depth`, so turning
the knob changed nothing (depth 6/8/10 gave byte-identical predictions). We only
caught it by directly checking that predictions actually moved. This harness does
that check for EVERY tunable knob, sweeping each against the OUT-OF-BOX model and
reporting, per swept value:

  dMETRIC% relative change vs the out-of-box default (held-out RMSE for
           regression / Brier for classification; negative = IMPROVED)
  dPRED    mean |prediction change| vs out-of-box, normalized by its spread
           (~0 across the whole range  ==  the knob is a NO-OP on this data)
  TREES    mean fitted tree count (the early-stopping response to the knob)

Two questions answered per knob:
  • WORKS?   max dPRED over the swept values clears a small floor -> it does something.
  • HELPS?   best swept value's dMETRIC% is meaningfully < 0 -> tuning it pays off.

The baseline is ALWAYS the out-of-box model (no kwargs) — exactly what a user
gets from `ChimeraBoostX().fit(X, y)`. It's fit once per (dataset, seed) and
every knob is diffed against it, so e.g. `depth=6` is compared to the regressor's
loss-adaptive default rather than assumed equal to it.

Suites:
  --suite offline    7 built-in datasets (reg/binary/multiclass/cat), NO network.
  --suite grinsztajn 59 real Grinsztajn datasets (HuggingFace). Dataset-outer so
                     each CSV downloads once and is reused across all knobs.

Report-only: this CHARACTERIZES the knobs. It does NOT tune defaults — nothing
ships off these numbers without the full synthetic->Grinsztajn->OpenML pipeline.

Usage:
  python benchmarks/knob_characterization.py                       # offline, all knobs
  python benchmarks/knob_characterization.py --knob colsample subsample
  python benchmarks/knob_characterization.py --suite grinsztajn --runs 1
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402  (exact Grinsztajn model config)
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Offline panel: the built-in datasets, tagged by group.
#   group "all"     — every dataset
#   group "cat"     — has categoricals (cat_smoothing / perms / combos)
#   group "reg_bin" — regression or binary (linear_leaves has no multiclass path)
# ---------------------------------------------------------------------------
OFFLINE_PANEL = ["diabetes", "friedman1", "synthetic_reg",   # regression
                 "breast_cancer", "cat_binary",              # binary
                 "wine", "cat_multiclass"]                   # multiclass


# Knob -> explicit probe values + the group of datasets it applies to. The
# baseline is the out-of-box model (kwarg omitted), so each list is just the set
# of explicit values to try; a value equal to the class default should show
# dPRED~0 (a built-in sanity check that "explicit default == out-of-box").
KNOBS = {
    "depth":                      dict(values=[4, 6, 8, 10],        group="all"),
    "learning_rate":              dict(values=[0.03, 0.1, 0.3],     group="all"),
    "l2_leaf_reg":                dict(values=[0.0, 1.0, 3.0, 10.0], group="all"),
    "min_child_weight":           dict(values=[0.0, 1.0, 5.0],      group="all"),
    "subsample":                  dict(values=[0.8, 0.5],           group="all"),
    "colsample":                  dict(values=[0.8, 0.5],           group="all"),
    "leaf_estimation_iterations": dict(values=[1, 3, 5, 10],        group="all"),
    "hs_lambda":                  dict(values=[1.0, 5.0, 20.0],     group="all"),
    "max_bins":                   dict(values=[64, 128, 254],       group="all"),
    "ordered_boosting":           dict(values=[True],               group="all"),
    # cat_smoothing=0.0 is invalid (0/0 pseudocount -> rejected); 0.3 is the
    # smallest sensible probe below the 1.0 default.
    "cat_smoothing":              dict(values=[0.3, 1.0, 10.0],     group="cat"),
    "cat_n_permutations":         dict(values=[1, 4, 8],            group="cat"),
    "cat_combinations":           dict(values=[True],               group="cat"),
    "linear_leaves":              dict(values=[True],               group="reg_bin"),
}


def _group_ok(group, task, has_cats):
    if group == "cat":
        return has_cats
    if group == "reg_bin":
        return task != "multiclass"
    return True


def _split(X, y, task, seed):
    strat = y if task != "regression" else None
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)


def _fit_eval(task, kw, Xtr, ytr, Xte, yte, cat, seed, threads):
    """Fit out-of-box (internal early-stop split) with kwargs `kw` (empty ==
    out-of-box baseline). Returns (metric, prediction_vector, n_trees)."""
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    m = Est(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
            random_state=seed, thread_count=threads, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    if task == "regression":
        pred = np.asarray(m.predict(Xte), dtype=float)
        metric = float(np.sqrt(np.mean((pred - yte) ** 2)))            # RMSE
    else:
        proba = np.asarray(m.predict_proba(Xte), dtype=float)
        classes = np.asarray(m.classes_)
        onehot = (np.asarray(yte)[:, None] == classes[None, :]).astype(float)
        metric = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))  # Brier (sum form)
        pred = proba.ravel()
    return metric, pred, int(m.best_iteration_)


def run_suite(knobs, dataset_keys, runs, threads, verbose=False):
    """Dataset-outer characterization with a shared out-of-box baseline.

    For each (dataset, seed): build once, fit the out-of-box baseline once, then
    fit every applicable (knob, value) and diff against that baseline. Returns
    accumulators keyed by (knob, value)."""
    # acc[knob][value] = dict(rel=[], pred=[], trees=[], per_ds={ds:[metrics]})
    acc = {k: {v: dict(rel=[], pred=[], trees=[], per_ds={})
               for v in KNOBS[k]["values"]} for k in knobs}

    for ds in dataset_keys:
        task = rb._task_of(ds)
        for run_idx in range(runs):
            seed = 3000 + run_idx
            rng = np.random.default_rng(seed)
            try:
                X, y, cat, _task = rb.DATASETS[ds](1.0, rng)
            except Exception as e:
                print(f"  ! build failed {ds} seed={seed}: {e}", flush=True)
                continue
            has_cats = cat is not None
            Xtr, Xte, ytr, yte = _split(X, y, task, seed)
            t0 = time.time()
            try:
                base_m, base_p, base_t = _fit_eval(
                    task, {}, Xtr, ytr, Xte, yte, cat, seed, threads)
            except Exception as e:
                print(f"  ! baseline failed {ds} seed={seed}: {e}", flush=True)
                continue
            spread = float(np.std(base_p)) + 1e-12

            for k in knobs:
                if not _group_ok(KNOBS[k]["group"], task, has_cats):
                    continue
                for v in KNOBS[k]["values"]:
                    try:
                        m, p, nt = _fit_eval(task, {k: v}, Xtr, ytr, Xte, yte,
                                             cat, seed, threads)
                    except Exception as e:
                        print(f"  ! {k}={v} failed {ds}: {e}", flush=True)
                        continue
                    a = acc[k][v]
                    a["rel"].append((m - base_m) / (abs(base_m) + 1e-12))
                    a["pred"].append(float(np.mean(np.abs(p - base_p))) / spread)
                    a["trees"].append(nt)
                    a["per_ds"].setdefault(ds, []).append(m)
            print(f"  [{ds}] seed={seed} task={task} baseline_trees={base_t} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return acc


def report(knobs, acc, verbose=False):
    summary = []
    for k in knobs:
        spec = KNOBS[k]
        print(f"\n=== {k}  (group={spec['group']}) ===")
        print(f"{'value':>10} | {'dMETRIC%':>9} | {'dPRED':>8} | {'trees':>6} | {'n':>4}")
        print("-" * 52)
        best_v, best_d, max_pc = None, 0.0, 0.0
        for v in spec["values"]:
            a = acc[k][v]
            if not a["rel"]:
                print(f"{str(v):>10} |    (no data)")
                continue
            d = 100.0 * float(np.mean(a["rel"]))
            pc = float(np.mean(a["pred"]))
            nt = float(np.mean(a["trees"]))
            n = len(a["rel"])
            max_pc = max(max_pc, pc)
            if d < best_d - 1e-9:
                best_d, best_v = d, v
            print(f"{str(v):>10} | {d:>+8.2f}% | {pc:>8.4f} | {nt:>6.0f} | {n:>4}")
        works = max_pc > 1e-3
        helps = best_v is not None and best_d < -0.5
        parts = ["WORKS" if works else "NO-OP (predictions ~unchanged)"]
        if helps:
            parts.append(f"HELPS: best={best_v!r} ({best_d:+.2f}%)")
        elif works:
            parts.append("works; out-of-box ~best on this suite")
        print(f"VERDICT: {' | '.join(parts)}")
        if verbose:
            # which datasets move most for this knob (best swept value vs nothing)
            moved = []
            for v in spec["values"]:
                for ds, ms in acc[k][v]["per_ds"].items():
                    moved.append((ds, v, float(np.mean(ms))))
            # show per-dataset spread across values for the knob
            by_ds = {}
            for ds, v, m in moved:
                by_ds.setdefault(ds, {})[v] = m
            for ds in sorted(by_ds):
                cells = "  ".join(f"{v}={by_ds[ds][v]:.4f}" for v in spec["values"]
                                  if v in by_ds[ds])
                print(f"    {ds:34s} {cells}")
        summary.append(dict(knob=k, works=works, helps=helps,
                            best_value=best_v, best_delta_pct=best_d, max_pred=max_pc))

    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"{'knob':>28} | {'works?':>6} | {'helps?':>6} | best (dMETRIC%)")
    print("-" * 72)
    for s in summary:
        w = "yes" if s["works"] else "NO-OP"
        h = "yes" if s["helps"] else "-"
        b = f"{s['best_value']!r} ({s['best_delta_pct']:+.2f}%)" if s["helps"] else ""
        print(f"{s['knob']:>28} | {w:>6} | {h:>6} | {b}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knob", nargs="+", default=["all"],
                    help=f"knobs to characterize, or 'all'. Available: {list(KNOBS)}")
    ap.add_argument("--suite", choices=["offline", "grinsztajn"], default="offline")
    ap.add_argument("--runs", type=int, default=None,
                    help="seeds per (dataset, value); default 3 offline / 1 grinsztajn")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="per-dataset breakdown")
    args = ap.parse_args()

    knobs = list(KNOBS) if args.knob == ["all"] else [k for k in args.knob if k in KNOBS]
    bad = [k for k in args.knob if k not in KNOBS and k != "all"]
    for k in bad:
        print(f"  ! unknown knob {k!r}, skipping")

    if args.suite == "grinsztajn":
        rb._add_grinsztajn_datasets()
        dataset_keys = [k for k in rb.DATASETS if k.startswith("gr:")]
        runs = args.runs if args.runs is not None else 1
    else:
        dataset_keys = OFFLINE_PANEL
        runs = args.runs if args.runs is not None else 3

    print(f"Knob characterization | suite={args.suite} | knobs={knobs} | "
          f"runs={runs} | {len(dataset_keys)} datasets", flush=True)
    acc = run_suite(knobs, dataset_keys, runs, args.threads, args.verbose)
    report(knobs, acc, args.verbose)


if __name__ == "__main__":
    main()
