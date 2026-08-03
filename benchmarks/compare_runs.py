"""Sign-test two run_benchmarks JSONs against each other (same model).

Usage:
    python benchmarks/compare_runs.py BASE.json NEW.json [base_label new_label]
                                      [--model ChimeraBoost] [--by-suite]
                                      [--metric brier]

Compares the per-dataset mean of the 'primary' metric (always higher-is-better:
negative RMSE for regression, F1/accuracy for classification). Reports per-dataset
deltas and a sign test (how many datasets NEW beats BASE).

READ THE SIGN TEST AND THE MEDIAN, NOT THE MEAN
-----------------------------------------------
The win/loss/tie count and the MEDIAN relative gap are what this project
decides on. The mean of relative gaps is printed for continuity with numbers
quoted in older plan files and is never a verdict: a single dataset whose base
loss is near zero can move it by thousands of percent. It has read -144% and
-8e21% here on comparisons the sign test read as fine (the -144% one was 54
wins to 31 losses). See NEAR-SOLVED DATASETS below.

--by-suite reports one INDEPENDENT sign test per stratum (suite x variant)
instead of one over the union. Mandatory for reading a --decide run: the
decision suites answer different questions, and a variant (@sus25/@sus50/@time)
is a derived view of its parent dataset, so a pooled test counts the same rows
twice. The tool warns loudly if you pool strata without it.
--metric brier judges on Brier instead: classification sets only (regression
records carry no Brier), oriented so NEW wins = lower Brier.
--model filters records to one model first. Without it, a multi-model JSON
blends every model's records into the per-dataset mean (fine when both runs
hold the other models fixed, but the deltas are diluted).
--model-new names the NEW run's records when they differ (e.g. baseline
ChimeraBoost vs an arm's ChimeraBoostEns2).

NEAR-SOLVED DATASETS
--------------------
A dataset every model solves to a practically-zero loss carries no information
about a change, but it wrecks a RELATIVE mean: the ratio of two tiny numbers is
numerical noise. Measured on a real historical screen, one such dataset
(syn:v2/117, base Brier ~0) contributed -12555% by itself and dragged the
reported mean of an 88-dataset comparison to -144.7% while the sign test read
54 wins / 31 losses. The old floor here (`abs(base) > 1e-12`) only caught
values that were literally zero to floating point, and silently scored them as
0.0 -- everything in the wide band between 1e-12 and "actually solved" sailed
through and distorted the mean.

The exclusion now uses the same thresholds as the rest of the analysis stack
(summarize.py, make_tables.py): regression drops when best NRMSE (RMSE / target
std) is below NEAR_SOLVED_NRMSE, classification when best Brier is below
NEAR_SOLVED_BRIER. Excluded datasets are always named in the output -- never
dropped silently. `--keep-near-solved` restores the old unguarded arithmetic
for auditing numbers quoted in older plan files.

The SIGN TEST and its PASS/FAIL bar deliberately still run over every shared
dataset, so this fix cannot silently flip a historical verdict. The sign test
is sign-based and was never distorted by this. The retained-only sign test is
printed alongside as a diagnostic, with a warning if the two disagree.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402  (sibling module)
from summarize import NEAR_SOLVED_NRMSE  # noqa: E402  (shared with make_tables)
from summarize import load as _load_json, timing_warning  # noqa: E402

# Classification analog of NEAR_SOLVED_NRMSE, matching summarize.primary_scores.
NEAR_SOLVED_BRIER = 1e-3


def load_run(path, model=None, metric="primary"):
    """(metric values, rmse, brier, dataset metadata), each keyed by dataset
    and averaged over seeds. rmse/brier come along regardless of the compared
    metric because the near-solved test needs them."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    sign = -1.0 if metric == "brier" else 1.0   # orient higher = better
    bucket, rmse, brier = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in data["records"]:
        if model is not None and r["model"] != model:
            continue
        m = r["metrics"]
        if m.get("rmse") is not None:
            rmse[r["dataset"]].append(m["rmse"])
        if m.get("brier") is not None:
            brier[r["dataset"]].append(m["brier"])
        if m.get(metric) is None:
            continue
        bucket[r["dataset"]].append(sign * m[metric])

    def _mean(d):
        return {k: float(np.mean(v)) for k, v in d.items()}

    return _mean(bucket), _mean(rmse), _mean(brier), data.get("datasets", {})


def is_near_solved(ds, ds_meta, rmse_b, rmse_n, brier_b, brier_n):
    """Is `ds` solved well enough that a relative delta on it is meaningless?

    Uses the best (lowest) loss either run achieved, mirroring summarize's
    "best across models" rule.

    The two rules degrade differently on an older JSON with no dataset
    metadata, and deliberately so. The regression rule needs y_std to form an
    NRMSE, so without it a dataset is never excluded. The Brier rule needs no
    metadata at all -- a Brier below 1e-3 is solved whatever the record says --
    so it still applies to anything that recorded one. Regression records carry
    no Brier, so they cannot be caught by it accidentally.
    """
    meta = ds_meta.get(ds) or {}
    if meta.get("task") == "regression":
        y_std = meta.get("y_std")
        vals = [v for v in (rmse_b.get(ds), rmse_n.get(ds)) if v is not None]
        return bool(y_std) and bool(vals) and min(vals) / y_std < NEAR_SOLVED_NRMSE
    vals = [v for v in (brier_b.get(ds), brier_n.get(ds)) if v is not None]
    return bool(vals) and min(vals) < NEAR_SOLVED_BRIER


def _warn_pooled_strata(ds_names):
    """Shout when one pooled sign test is about to span several strata.

    A variant (@sus25/@sus50/@time) is a derived VIEW of its parent dataset, so
    pooling the two counts the same rows twice and the test claims a larger
    sample than it has. Grinsztajn and the high-card suite are separate
    questions besides. CLAUDE.md therefore requires --by-suite for a --decide
    run. Warning only: nothing below it changes and the exit code stays 0.
    """
    strata = summarize.split_strata(ds_names)
    if len(strata) < 2:
        return
    found = ", ".join(f"{summarize.stratum_label(s)} ({len(d)})"
                      for s, d in strata.items())
    bar = "!" * 72
    print(f"{bar}\n"
          f"WARNING: pooling {len(strata)} strata into one sign test: {found}.\n"
          f"A variant is a derived view of its parent dataset, so a pooled test\n"
          f"scores the same rows twice and overstates its own sample size.\n"
          f"Re-run with --by-suite for the per-stratum tests the protocol wants.\n"
          f"{bar}\n")


def _sign_counts(pairs):
    """(wins, losses, ties) over (base, new) pairs on a higher-is-better metric."""
    wins = sum(1 for b, n in pairs if n - b > 1e-9)
    losses = sum(1 for b, n in pairs if n - b < -1e-9)
    return wins, losses, len(pairs) - wins - losses


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
    ap.add_argument("--keep-near-solved", action="store_true",
                    help="do NOT exclude near-solved datasets from the mean "
                         "(reproduces pre-fix numbers quoted in older plans).")
    ap.add_argument("--by-suite", action="store_true",
                    help="report an independent sign test per stratum (suite x "
                         "variant) instead of one over the union -- mandatory "
                         "for reading a --decide run.")
    args = ap.parse_args()
    base_label, new_label = args.base_label, args.new_label

    base, rmse_b, brier_b, meta_b = load_run(
        args.base_path, args.model, args.metric)
    new, rmse_n, brier_n, meta_n = load_run(
        args.new_path, args.model_new or args.model, args.metric)
    ds_meta = {**meta_b, **meta_n}
    shared = sorted(set(base) & set(new))

    # Strength comparisons are unaffected by the timing convention, but say so
    # loudly if the two runs straddle the _finish fix — anyone reading a speed
    # number off these files needs to know.
    warn = timing_warning(_load_json(args.base_path), _load_json(args.new_path))
    if warn:
        print(warn + "\n")

    if args.by_suite:
        # One INDEPENDENT sign test per stratum. The decision suites answer
        # different questions and a variant reuses its parent's rows, so a
        # pooled bar over the union would be a different (weaker) test than the
        # protocol asks for. No combined verdict is printed on purpose.
        strata = summarize.split_strata(shared)
        for i, (stratum, ds_names) in enumerate(strata.items()):
            if i:
                print()
            print(f"########## {summarize.stratum_label(stratum)} "
                  f"({len(ds_names)} datasets) ##########")
            _report(ds_names, base, new, ds_meta, rmse_b, rmse_n,
                    brier_b, brier_n, args, base_label, new_label)
        return

    _warn_pooled_strata(shared)
    _report(shared, base, new, ds_meta, rmse_b, rmse_n, brier_b, brier_n,
            args, base_label, new_label)


def _report(shared, base, new, ds_meta, rmse_b, rmse_n, brier_b, brier_n,
            args, base_label, new_label):
    """Per-dataset rows, the mean/median, and the sign-test bar for one set of
    datasets. Behaviour on the full set is unchanged from before --by-suite."""
    near = set() if args.keep_near_solved else {
        ds for ds in shared
        if is_near_solved(ds, ds_meta, rmse_b, rmse_n, brier_b, brier_n)}

    wins = losses = ties = 0
    print(f"{'dataset':22s} {base_label:>12s} {new_label:>12s} {'delta':>12s}  result")
    rel_deltas, all_pairs, rel_named = [], [], []
    for ds in shared:
        b, n = base[ds], new[ds]
        d = n - b                       # primary is higher-better
        all_pairs.append((b, n))
        if d > 1e-9:
            wins += 1; tag = f"{new_label} wins"
        elif d < -1e-9:
            losses += 1; tag = f"{base_label} wins"
        else:
            ties += 1; tag = "tie"
        if ds in near:
            print(f"{ds:22s} {b:12.4f} {n:12.4f} {d:+12.4f}  {tag}  "
                  f"(near-solved: excluded from mean)")
            continue
        # relative improvement (guard tiny/zero base)
        rel = d / abs(b) if abs(b) > 1e-12 else 0.0
        rel_deltas.append(rel)
        rel_named.append((rel, ds))
        print(f"{ds:22s} {b:12.4f} {n:12.4f} {d:+12.4f}  {tag}  ({rel:+.2%})")

    n_ds = len(shared)
    mtag = args.model or ""
    if args.model_new and args.model_new != args.model:
        mtag = f"{mtag}->{args.model_new}"
    print(f"\n{new_label} vs {base_label}: {wins} wins / {losses} losses / {ties} ties  "
          f"(of {n_ds} datasets)"
          + (f"  [model={mtag}]" if mtag else ""))

    if near:
        print(f"excluded {len(near)} near-solved dataset(s) from the mean "
              f"(regression NRMSE < {NEAR_SOLVED_NRMSE}, "
              f"classification Brier < {NEAR_SOLVED_BRIER}): "
              + ", ".join(sorted(near)))
    if rel_deltas:
        mean_v, med_v = float(np.mean(rel_deltas)), float(np.median(rel_deltas))
        print(f"mean relative change in {args.metric} (+ = better): "
              f"{mean_v:+.3%}   "
              f"[median {med_v:+.3%}, n={len(rel_deltas)}]")
        # A mean far from its median is being carried by one or two datasets --
        # usually a near-zero denominator that sits just ABOVE the near-solved
        # cutoff, so the guard above never fired. This project has been misread
        # that way three times (-144%, -8e21%, -1.171%), so name the culprit
        # rather than leaving the mean to be quoted on its own. Print-only: no
        # verdict depends on it (benchmarks/GATE_ROBUSTNESS.md #3).
        if rel_named and abs(mean_v) > 3 * abs(med_v) + 1e-12:
            worst = max(rel_named, key=lambda t: abs(t[0] - med_v))
            without = [r for r, d in rel_named if d != worst[1]]
            if without:
                print(f"  !! mean is {abs(mean_v) / (abs(med_v) + 1e-12):.0f}x "
                      f"its median -- largest single contributor is "
                      f"{worst[1]} at {worst[0]:+.2%}; "
                      f"mean without it: {float(np.mean(without)):+.3%}. "
                      f"Read the sign test and the median.")
    else:
        print(f"mean relative change in {args.metric}: n/a (no scored datasets)")

    need = n_ds // 2 + 1
    verdict = "PASS" if wins >= need else "FAIL"
    print(f"sign-test bar (> half = {need}+ wins): {verdict}")

    # Diagnostic only: the bar above intentionally stays over ALL datasets so
    # this guard cannot silently flip a verdict recorded in an older plan file.
    if near:
        kept = [p for ds, p in zip(shared, all_pairs) if ds not in near]
        kw, kl, kt = _sign_counts(kept)
        k_need = len(kept) // 2 + 1
        k_verdict = "PASS" if kw >= k_need else "FAIL"
        note = "" if k_verdict == verdict else "   <-- DISAGREES with the bar above"
        print(f"  (excluding near-solved: {kw} wins / {kl} losses / {kt} ties, "
              f"bar {k_need}+ = {k_verdict}){note}")


if __name__ == "__main__":
    main()
