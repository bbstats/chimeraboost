---
name: pareto
description: Regenerate or read the strength vs slowdown Pareto (the north-star chart; skill-score axis)
---

Run `python benchmarks/make_pareto.py` (newest `benchmarks/results/*.json`; pass a path for a
specific run; `--no-image` for the text tables only). It emits `images/pareto.png` (skill-score
axis, two panels) + `images/winrate_matrix.png` (who-beats-whom companion) + phone-readable
tables. Show the tables to the user.

Headline axis (Nathan 2026-08-02): **skill scores, split by task** — Brier skill for
classification, R² for regression. Both are 0 at the no-skill baseline (class marginals / the
target mean) and 1 at perfect.
- Two panels, not one average: BSS and R² sit on different parts of 0..1, and averaging buries
  the classification leg — which is where the field spreads (CatBoost 5th on reg, 7th on clf).
- **Not field-relative.** Adding or dropping an arm moves nobody, unlike win rate.
- **No near-solved exclusion needed** — BSS and R² stay bounded where ratio-to-best explodes
  (the −144% / −8e21% readings). Every dataset in the run is scored.
- **The axis is compressed** — the whole field usually sits within ~0.02 of skill, so both
  panels are truncated dot plots. Read the tick labels, not the visual gaps.
- Needs `class_prior` (clf) + `y_std` (reg) in the run's dataset metadata. `class_prior` landed
  with 0.30.0, so **older runs can only be read with `--metric winrate`.**
- Expected to be superseded by TabArena scores once the wanted runs are uploaded.

Both older axes stay as DIAGNOSTICS:
- `--metric winrate` → `pareto_winrate.png`. % of (dataset × opponent) matchups won, ties ½ each,
  95% bootstrap CI. Ordinal, so it spreads the field more — but field-relative, and it counts
  matchups without regard to margin.
- `--metric blended` → `pareto_blended.png`. classification = ⅔·Brier% + ⅓·F1%;
  blended = HarmonicMean(RegRMSE%, classification), harmonic on purpose so it tracks the weak leg.
- x-axis on every variant = slowdown (mean fit-time multiple vs fastest); frontier = up-and-left.

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
- Ship-gating is UNCHANGED (sign tests per /experiment); the chart axis is legibility only.
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
- Win-rate DIAGNOSTIC reference points (2026-07-31, run 20260731-142609): Ens8 95.7% @ 26.8×,
  Ens5 82.0% @ 18.7×, ChimeraBoost (the default, quality=3) 71.7% @ 7.1×, NoRefit 45.7% @ 5.4×,
  OneLin 28.9% @ 2.1×; CatBoost 41.6% @ 15.3× (dominated), LightGBM 20.8% @ 1.1×, HGB 13.5% @ 4.7×.
- Win rate is FIELD-RELATIVE, so adding or removing arms moves every row. Compare only within one
  run: the older five-arm field read the default at 55.7% @ 5.1×. This is the main reason it lost
  the headline slot — the skill axis has no such coupling.
- Contracts are pinned in `tests/test_strength_viz.py`.
