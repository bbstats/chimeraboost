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

Notes:
- Ship-gating is UNCHANGED (sign tests per /experiment); the win-rate axis is chart legibility only.
- The input JSON must be a fresh multi-model run (ChimeraBoost + CatBoost + LightGBM + sklearn_HGB
  at minimum) — mixing runs from different machines/fields breaks the %, win-rate, and speed columns.
- `images/pareto.png` + `images/winrate_matrix.png` are committed — commit the refresh after a
  shipped change. (README headline is the TabArena chart, deliberate since fcdc874.)
- Reference points (2026-07-25, run 20260725-213827 — the `quality` ladder co-run): Ens8 94.0% @ 25.7×,
  Ens5 83.7% @ 17.7×, ChimeraBoost (the default, quality=3) 69.9% @ 9.4×, NoRefit 45.6% @ 5.4×,
  OneLin 31.1% @ 2.0× — all five on the frontier; CatBoost 40.9% @ 14.1× (dominated on both axes),
  LightGBM 21.6% @ 1.1× (frontier), HGB 13.3% @ 4.8×.
- Win rate is FIELD-RELATIVE (% of dataset×opponent matchups won), so adding or removing arms moves
  every row. Compare only within one run: the older five-arm field read the default at 55.7% @ 5.1×.
- Contracts are pinned in `tests/test_strength_viz.py`.
