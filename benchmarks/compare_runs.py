"""Sign-test two run_benchmarks JSONs against each other (same model).

Usage:
    python benchmarks/compare_runs.py BASE.json NEW.json [base_label new_label]
                                      [--model ChimeraBoost] [--by-suite]
                                      [--metric decision|primary|brier|crps]

Judges each dataset on the DECISION metric by default -- RMSE for regression,
Brier for classification, the pair every ship gate, the harness SUMMARY block
and the Pareto chart score on -- oriented so NEW wins = lower loss. Reports
per-dataset deltas and a sign test (how many datasets NEW beats BASE).

Before 2026-09-22 the default was ``primary`` (negative RMSE for regression
but F1 for classification), so classification bars quoted in log entries
before CAMPAIGN_PLAN I044 were read on F1 while the gate names Brier; pass
``--metric primary`` to reproduce those numbers.

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
--metric primary judges on the harness's 'primary' field (negative RMSE for
regression, F1 for classification) -- the pre-2026-09-22 default, kept for
reproducing older entries.
--metric brier judges on Brier alone: classification sets only (regression
records carry no Brier), oriented so NEW wins = lower Brier.
--metric crps judges on CRPS, for quantile_suite.py runs, oriented so NEW wins
= lower CRPS. Those runs already store primary = -CRPS, so the default reads
the same ordering; --metric crps exists to print the CRPS itself.
--model filters records to one model first. Without it, a multi-model JSON
blends every model's records into the per-dataset mean (fine when both runs
hold the other models fixed, but the deltas are diluted).
--model-new names the NEW run's records when they differ (e.g. baseline
ChimeraBoost vs an arm's ChimeraBoostEns2).
--expect-inert declares that the change is conditionally gated and must be inert
somewhere; see THE INERT SLICE IS A CONTROL below.

THE INERT SLICE IS A CONTROL
----------------------------
The sign-test bar counts ties against the change, so a conditionally-gated
change reads worse the better its gating -- that is how a 6W-0L result once read
FAIL (GATE_ROBUSTNESS.md #4). Every comparison therefore also prints the tie
count as a CONTROL (an exact tie where the change cannot engage is positive
evidence that it did what it claims and nothing else) and an engaged-only sign
test, which answers "when this engaged, did it help?". Neither changes the bar.

THE ENGAGED SLICE IS MEASURED, NOT JUST COUNTED
-----------------------------------------------
Two plan-file bars were computed by hand because no tool printed them
(CAMPAIGN_PLAN I020/I021): the engaged-slice MEDIAN relative change and how
sure that read is. Every comparison now prints, for the engaged datasets
(near-solved ones excluded as in the mean above): the median relative change
with a 95% percentile bootstrap CI over datasets (10 000 resamples, seed 0,
the summarize.bootstrap_winrate_ci convention), and PER-SEED AGREEMENT -- how
many engaged datasets have every seed's NEW-BASE carry the sign of the mean
delta, and which ones split. A 2-1 seed split is not the win a 3-0 one is
(GATE_ROBUSTNESS.md #1: "run more seeds" reproduces the same table; this is
the number that would have said so). Print-only, no verdict moves.

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


def _judged_value(record, metric, tasks):
    """The higher-is-better value a record contributes under ``metric``, or
    None when the record carries none.

    ``decision`` is the gate's own pair: negative RMSE on regression datasets,
    negative Brier on classification ones (the task comes from the run's
    dataset metadata; a record with an ``rmse`` and no task is regression).
    The losses ``brier`` and ``crps`` are negated too; ``primary`` is used as
    stored (the harness already orients it)."""
    m = record["metrics"]
    if metric == "decision":
        task = tasks.get(record["dataset"], {}).get("task")
        if task is None:
            task = "regression" if m.get("rmse") is not None else "binary"
        key = "rmse" if task == "regression" else "brier"
        v = m.get(key)
        return None if v is None else -float(v)
    v = m.get(metric)
    if v is None:
        return None
    return -float(v) if metric in ("brier", "crps") else float(v)


def load_run(path, model=None, metric="decision"):
    """(metric values, rmse, brier, dataset metadata), each keyed by dataset
    and averaged over seeds. rmse/brier come along regardless of the compared
    metric because the near-solved test needs them."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    tasks = data.get("datasets", {})
    bucket, rmse, brier = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in data["records"]:
        if model is not None and r["model"] != model:
            continue
        m = r["metrics"]
        if m.get("rmse") is not None:
            rmse[r["dataset"]].append(m["rmse"])
        if m.get("brier") is not None:
            brier[r["dataset"]].append(m["brier"])
        v = _judged_value(r, metric, tasks)
        if v is None:
            continue
        bucket[r["dataset"]].append(v)

    def _mean(d):
        return {k: float(np.mean(v)) for k, v in d.items()}

    return _mean(bucket), _mean(rmse), _mean(brier), data.get("datasets", {})


def load_run_seeds(path, model=None, metric="decision"):
    """{dataset: {seed: metric value}}, oriented higher-is-better like
    ``load_run`` -- the per-seed view the agreement read needs. ``load_run``'s
    seed-averaged path is untouched."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    tasks = data.get("datasets", {})
    out = defaultdict(dict)
    for r in data["records"]:
        if model is not None and r["model"] != model:
            continue
        v = _judged_value(r, metric, tasks)
        if v is None:
            continue
        out[r["dataset"]][r.get("seed", 0)] = v
    return dict(out)


def engaged_slice_stats(rels, n_boot=10000, seed=0):
    """(median, ci_lo, ci_hi) of relative changes: the median over datasets and
    its 95% percentile bootstrap CI, resampling datasets with replacement
    (deterministic for a given seed). None for an empty slice."""
    if not rels:
        return None
    arr = np.asarray(rels, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    meds = np.median(arr[idx], axis=1)
    return (float(np.median(arr)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def seed_agreement(ds_names, seeds_b, seeds_n):
    """(unanimous, split, n_multi): over the datasets given, how many have
    every shared seed's NEW-BASE carry the sign of the seed-mean delta
    (``unanimous``), which do not (``split``, names), and how many had more
    than one shared seed at all (``n_multi`` -- with a single seed there is
    nothing to agree on). Datasets with no shared seed are skipped."""
    unanimous, split, n_multi = 0, [], 0
    for ds in ds_names:
        sb, sn = seeds_b.get(ds, {}), seeds_n.get(ds, {})
        common = sorted(set(sb) & set(sn))
        if len(common) < 2:
            continue
        n_multi += 1
        deltas = np.array([sn[s] - sb[s] for s in common])
        ref = np.sign(deltas.mean())
        if ref != 0 and np.all(np.sign(deltas) == ref):
            unanimous += 1
        else:
            split.append(ds)
    return unanimous, split, n_multi


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

    Quantile runs (`quantile_suite.py`, task "quantile") are never excluded:
    they carry neither RMSE nor Brier, so both rules fall through to False.
    That is the safe direction -- nothing is dropped that should have been
    counted -- and it only affects the MEAN, which is never the verdict here.
    Add a CRPS/y_std rule if a near-solved quantile dataset ever distorts one.
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


# A stratum with fewer decided datasets than this cannot separate a real
# regression from noise (benchmarks/GATE_ROBUSTNESS.md #2: SMALLDATA's hc:time
# FAIL on 7 sets later measured 21W-21L). Decided = wins + losses; exact ties
# are the inert slice, not evidence either way. The label is print-only: the
# PASS/FAIL word stays so verdicts recorded in older plan files still match.
POINTER_MIN_DECIDED = 8


def pointer_label(decided):
    """The suffix a sign-test line carries when it is a pointer, not a gate."""
    if decided >= POINTER_MIN_DECIDED:
        return ""
    return (f"   [POINTER, not a gate: {decided} decided "
            f"< {POINTER_MIN_DECIDED}]")


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
    ap.add_argument("--metric", choices=["decision", "primary", "brier", "crps"],
                    default="decision",
                    help="judge metric. decision (default) = RMSE on "
                         "regression, Brier on classification -- what every "
                         "gate scores; primary = the harness field (F1 for "
                         "classification), the pre-2026-09-22 default; brier "
                         "= classification only; crps = quantile_suite.py "
                         "runs. Losses are oriented so NEW wins = lower.")
    ap.add_argument("--keep-near-solved", action="store_true",
                    help="do NOT exclude near-solved datasets from the mean "
                         "(reproduces pre-fix numbers quoted in older plans).")
    ap.add_argument("--expect-inert", action="store_true",
                    help="declare that this change is conditionally gated and "
                         "MUST be inert on part of the suite; the control line "
                         "then fails loudly if it engaged everywhere.")
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
    seeds = (load_run_seeds(args.base_path, args.model, args.metric),
             load_run_seeds(args.new_path, args.model_new or args.model,
                            args.metric))

    # Strength comparisons are unaffected by the timing convention, but say so
    # loudly if the two runs straddle the _finish fix — anyone reading a speed
    # number off these files needs to know.
    warn = timing_warning(_load_json(args.base_path), _load_json(args.new_path))
    if warn:
        print(warn + "\n")
    if args.metric == "decision":
        print("metric: decision = RMSE (regression) / Brier (classification), "
              "the gate's own pair; entries before 2026-09-22 were read on "
              "--metric primary (F1 for classification).\n")

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
                    brier_b, brier_n, args, base_label, new_label, seeds)
        return

    _warn_pooled_strata(shared)
    _report(shared, base, new, ds_meta, rmse_b, rmse_n, brier_b, brier_n,
            args, base_label, new_label, seeds)


def _report(shared, base, new, ds_meta, rmse_b, rmse_n, brier_b, brier_n,
            args, base_label, new_label, seeds=None):
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
    print(f"sign-test bar (> half = {need}+ wins): {verdict}"
          + pointer_label(wins + losses))

    # Diagnostic only: the bar above intentionally stays over ALL datasets so
    # this guard cannot silently flip a verdict recorded in an older plan file.
    if near:
        kept = [p for ds, p in zip(shared, all_pairs) if ds not in near]
        kw, kl, kt = _sign_counts(kept)
        k_need = len(kept) // 2 + 1
        k_verdict = "PASS" if kw >= k_need else "FAIL"
        note = "" if k_verdict == verdict else "   <-- DISAGREES with the bar above"
        print(f"  (excluding near-solved: {kw} wins / {kl} losses / {kt} ties, "
              f"bar {k_need}+ = {k_verdict}){note}" + pointer_label(kw + kl))

    _control_line(shared, all_pairs, ties, args, near, seeds)


def _control_line(shared, all_pairs, ties, args, near=(), seeds=None):
    """Read the exact ties as a CONTROL rather than as a penalty.

    A conditionally-gated change (size-gated, feature-gated, dtype-gated) is
    deliberately inert on part of the suite. The bar above counts those ties
    against it, so the better a change's gating the worse it reads -- that is
    how a 6W-0L result read FAIL (GATE_ROBUSTNESS.md #4).

    Read the other way round, an exact tie where the change cannot engage is
    POSITIVE evidence: it did what it claims and nothing else. Two readings
    follow from that, and both are printed here:

      - the inert slice, which is the control; and
      - the engaged-only sign test, which answers "when this engaged, did it
        help?" -- the question the all-dataset bar cannot answer.

    Below those, the engaged slice is MEASURED: its median relative change
    with a bootstrap CI over datasets, and per-seed agreement (see the module
    docstring). ``near`` names the near-solved datasets to keep out of the
    median; ``seeds`` is the (base, new) pair of per-seed views.

    Print-only. No verdict above changes, and the exit code stays 0.
    """
    engaged = [(ds, p) for ds, p in zip(shared, all_pairs)
               if abs(p[1] - p[0]) > 1e-9]

    if ties:
        print(f"control (inert slice): {ties} of {len(shared)} datasets are exact "
              f"ties -- the change did not engage there.")
    else:
        print(f"control (inert slice): none -- the change engaged on all "
              f"{len(shared)} datasets.")
        if args.expect_inert:
            print("  !! CONTROL FAIL: --expect-inert was declared and nothing "
                  "was inert. Either the gate is not doing what it claims, or "
                  "the arm is misconfigured. Find out before reading the bar.")

    if engaged and ties:
        ew, el, et = _sign_counts([p for _, p in engaged])
        e_need = len(engaged) // 2 + 1
        e_verdict = "PASS" if ew >= e_need else "FAIL"
        plural = "dataset" if len(engaged) == 1 else "datasets"
        print(f"  engaged only ({len(engaged)} {plural}): {ew} wins / {el} losses, "
              f"bar {e_need}+ = {e_verdict}   <-- 'when it engaged, did it help?'"
              + pointer_label(ew + el))

    _engaged_slice_lines(engaged, near, seeds, args.metric)


def _engaged_slice_lines(engaged, near, seeds, metric):
    """The engaged slice measured: median relative change with its bootstrap
    CI, and per-seed agreement. Prints nothing when nothing engaged."""
    if not engaged:
        return
    rels = [(n - b) / abs(b) for ds, (b, n) in engaged
            if ds not in near and abs(b) > 1e-12]
    stats = engaged_slice_stats(rels)
    if stats is None:
        print("  engaged slice: every engaged dataset is near-solved -- "
              "no median to read")
    else:
        med, lo, hi = stats
        print(f"  engaged slice ({len(rels)} scored): median relative change "
              f"in {metric} {med:+.3%}  [95% bootstrap CI {lo:+.3%}..{hi:+.3%}]"
              + pointer_label(len(rels)))
    if seeds is None:
        return
    unanimous, split, n_multi = seed_agreement(
        [ds for ds, _ in engaged], seeds[0], seeds[1])
    if n_multi == 0:
        print("  per-seed agreement: single seed -- nothing to agree on")
        return
    line = (f"  per-seed agreement: {unanimous} of {n_multi} engaged datasets "
            f"unanimous across seeds, {len(split)} split")
    if split:
        line += " (" + ", ".join(split) + ")"
    print(line)


if __name__ == "__main__":
    main()
