# Gate robustness — how our benchmark process fools us

Companion to the `/experiment` skill, which says *what* to run. This says where
the process is brittle, with the incident that proved each one.

The theme, and it recurs: **a number that looks like evidence and isn't.** In
one day (2026-08-01, SMALLDATA) three separate readings nearly decided a ship
wrongly — duplicated seeds, a four-dataset stratum, and a mean driven by one
near-zero denominator. Each was individually reasonable and each was empty. The
common defence is the same in all three cases: **print the effective sample size
and the dominant contributor next to every aggregate**, so an empty number
cannot look full.

---

## 1. Sample size that does not exist

`_temporal_split` takes `TEMPORAL_CUTS[seed % 3]` and has no other
seed-dependent randomness. **Seed 3 reproduces seed 0 exactly.** A `--seeds 6`
run on `@time` is still three windows per dataset; the extra seeds return
identical numbers and the run looks twice as strong as it is.

The `@time` universe is 7 datasets × 3 cuts. That is all of it.

- **Do**: vary `TEMPORAL_CUTS` in a probe when you need more temporal windows.
- **Don't**: add seeds and believe the denominator.
- **Guard added**: `run_benchmarks.py` now warns when `--seeds` exceeds the
  number of distinct cuts and `@time` datasets are in the run.

*Incident*: SMALLDATA C5. The instinct on seeing a 4-dataset regression was
"run more seeds", which would have produced the identical table and a false
sense of confirmation.

## 2. Verdicts rendered on strata too small to carry one

Tier-2 prints PASS/FAIL per stratum including `hc:sus50` (n=2, both ties) and
`hc:time` (7 datasets, 4 of which engaged the change). One of those FAILs was
carried into a shipping recommendation and later measured as **21W-21L,
p=1.000** — a coin flip.

- A stratum below roughly 8 decided datasets cannot distinguish a real
  regression from noise. Read it as a pointer, never as a gate.
- Before letting a single stratum block a ship, **probe that stratum directly**
  with more windows or more sources. It is one run and it is decisive.

*Incident*: SMALLDATA tier-2 → C5.

## 3. Hard thresholds always have a just-above case

The near-solved guard excludes classification sets below Brier 0.001. On the
tier-1 screen `syn:v2/117` sat at **0.0018** — just above — and moved
0.0018 → 0.0036, an absolute delta of 0.0018 that reads as **−99.85%**. That
single dataset dragged an 86-set mean to −1.171%; without it the mean is
−0.010%.

This is the **second** time this class of bug has bitten: the original
near-solved fix (PR #31) was itself a response to an 88-set mean reading −144%.
Adding a threshold did not remove the failure mode, it moved it one step out.

- **Never gate on a mean.** The house rule is the sign test and the median.
- **Guard added**: `compare_runs.py` names the single largest contributor to
  the mean whenever `|mean| > 3×|median|`, so a distorted mean announces itself
  instead of being quietly read.

## 4. The sign bar counts ties against the change

`wins >= n//2 + 1` over **all** datasets. A change that is byte-identical on
part of the suite therefore fails by construction: tier-2's `hc:base` was
**6W-0L — zero losses** — and read FAIL because 8 ties outvoted it.

This will get worse, not better. Every conditionally-gated change we ship
(size-gated, feature-gated, dtype-gated) is deliberately inert on part of the
suite, so its tie count grows with the quality of its gating.

- Read the **decided-only** count beside the all-dataset bar. `compare_runs`
  already prints an "excluding near-solved" line; the decided-only reading is
  the one that answers "when this engaged, did it help?".
- Record both in plan files. "6W-0L-8T, bar FAIL" is honest; "FAIL" alone is
  not.

**Turn it around: the inert slice is a control, not a penalty.** An exact tie
where a gated change *cannot* engage is positive evidence — the change did what
it claims and nothing else. That is the one control this project's decision tier
otherwise lacks; the synthetic suite has had one for a year
(`synthgen/backtest.py`, the saturated-and-cat-bearing canary slice that must not
go positive) and `--decide` had none.

- **Guard added**: `compare_runs.py` prints a `control (inert slice)` line on
  every comparison — the exact-tie count, plus an **engaged-only** sign test.
  Nothing above it changes; the bar is still over all datasets.
- Pass `--expect-inert` when the change is conditionally gated. The control then
  fails loudly if it engaged everywhere, which means either the gate is not doing
  what it claims or the arm is misconfigured. A change that should be inert
  somewhere and isn't is a bug report, not a benchmark result.

## 5. Cross-program comparisons that mix statistics

Plan files record means and medians inconsistently. A candidate was **killed**
by comparing its median against `refit_members`' mean — "a quarter of the
strength for three to five times the cost". On like-for-like medians it was
+0.275% vs +0.304%, i.e. the same trade, and the kill was withdrawn.

- Plan files record **both**, always labelled.
- Any comparison across programs states which statistic it is using.

## 6. Decision-grade conclusions from non-decision-grade instruments

The first learning-rate probe printed fit-time ratios with `thread_count`
unpinned, carrying an explicit "never compare to a harness slowdown" caveat —
and the cost axis then became the axis the decision turned on. The caveat was
right there and still nearly lost.

- If a number will gate a decision, produce it under pinned conditions. If it
  cannot be, the probe should refuse to print the column rather than print it
  with a warning nobody weighs.

## 7. Sweep design: saturation hides the knee

A rate sweep of 0.1 / 0.05 / 0.03 showed strength saturating by 0.05 while cost
kept climbing. The best point — 0.07, which dominates 0.05 on both axes — was
never sampled.

- **If the metric saturates at your first alternative while cost keeps rising,
  the optimum is between it and the baseline.** Sample there before concluding
  anything about the family.

## 8. Pre-registration has to survive contact with the data

The 0.07 arm was chosen *after* seeing the pilot, which makes that run
exploratory, not confirmatory — it cannot be cited as a passed test.

- Label post-hoc arm selection as such, in the plan file, at the time.
- Confirm on data that did not select it. Here tier-1 (synth) and tier-2
  (decide) did exactly that, which is why the result stands.

---

## The short version

Before a stratum, a mean, or a seed count is allowed to decide anything, ask:

1. **How many independent things is this actually?** (not how many rows the
   table has)
2. **What happens to it if I drop the single biggest contributor?**
3. **Am I comparing the same statistic on both sides?**
4. **Was this instrument built to be decided on?**

Three of this project's worst readings — −144%, −8e21%, and −1.171% — all
failed question 2.
