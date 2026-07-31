---
name: pareto
description: Regenerate or read the strength vs slowdown Pareto (the north-star chart; head-to-head win-rate axis)
---

Run `python benchmarks/make_pareto.py` (newest `benchmarks/results/*.json`; pass a path for a
specific run; `--no-image` for the text tables only). It emits `images/pareto.png` (win-rate
axis) + `images/winrate_matrix.png` (who-beats-whom companion) + phone-readable tables.
Show the tables to the user.

Headline axis (STRENGTH_VIZ_PLAN.md, Nathan 2026-07-18): **head-to-head win rate** — % of
(dataset × opponent) matchups won on the per-dataset primary metric (RMSE reg / Brier clf;
exact ties ½ each; 95% bootstrap CI over datasets). 50% = mid-pack. Equals mean rank rescaled:
(k − mean_rank)/(k − 1). The old blended axis saturated at 99.x (ratios-to-best on
near-Bayes-optimal data) — it stays as the DIAGNOSTIC:
- classification = ⅔·Brier% + ⅓·F1% (all "% vs best on task", higher = better)
- blended = HarmonicMean(RegRMSE%, classification) — harmonic on purpose: it tracks the weak leg
- `--metric blended` re-renders the legacy view (writes pareto_blended.png, never the headline)
- x-axis = slowdown (mean fit-time multiple vs fastest model); frontier = up-and-left

**Two charts now (issue #37).** `make_pareto.py` is the INTERNAL north star and is
unchanged. `make_public_pareto.py` is the PUBLISHED one:
- scored against **CatBoost + LightGBM only** — each ChimeraBoost point is never
  scored against its sibling rungs. The internal axis is field-relative, so every
  row moves when an arm is added, and with several of our rungs against two
  competitors most opponents would be our own arms ("wins N% of matchups" would
  largely be us beating ourselves). Competitor-relative is stable: adding a rung
  leaves every other row untouched (pinned in `tests/test_public_winrate.py`).
- meant for the sealed `pub:` suite (`benchmarks/PUBLIC_PLAN.md`). Pointed at any
  other run it prints a NOT-A-PUBLISHABLE-READ banner and stamps the figure —
  a chart on the suites we tune against is in-sample.
- XGBoost and RandomForest are gone: XGB tracks LightGBM, and RF was never in
  the harness (both were chart-only rows on the TabArena figure).

Notes:
- Ship-gating is UNCHANGED (sign tests per /experiment); the win-rate axis is chart legibility only.
- Reference points below PREDATE the issue-#37 timing fix: `fit_time` no longer
  includes predict + metric computation, so every slowdown must be re-read once
  on a fresh run before being quoted. Scoring was 31% of the old "fit time" for
  ChimeraBoost on breast_cancer vs 14% for LightGBM — our predict is absolutely
  slower, so the old convention OVERSTATED our slowdown and the correction moves
  in our favour.
- The input JSON must be a fresh multi-model run (ChimeraBoost + CatBoost + LightGBM + sklearn_HGB
  at minimum) — mixing runs from different machines/fields breaks the %, win-rate, and speed columns.
- `images/pareto.png` + `images/winrate_matrix.png` are committed — commit the refresh after a
  shipped change. (README headline is the TabArena chart, deliberate since fcdc874.)
- Reference points (2026-07-31, run 20260731-142609 — the post-0.27.0 `quality` ladder re-run):
  Ens8 95.7% @ 26.8×, Ens5 82.0% @ 18.7×, ChimeraBoost (the default, quality=3) 71.7% @ 7.1×,
  NoRefit 45.7% @ 5.4×, OneLin 28.9% @ 2.1× — all five on the frontier; CatBoost 41.6% @ 15.3×
  (dominated on both axes), LightGBM 20.8% @ 1.1× (frontier), HGB 13.5% @ 4.7×.
- Win rate is FIELD-RELATIVE (% of dataset×opponent matchups won), so adding or removing arms moves
  every row. Compare only within one run: the older five-arm field read the default at 55.7% @ 5.1×.
- Contracts are pinned in `tests/test_strength_viz.py`.
