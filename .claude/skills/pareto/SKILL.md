---
name: pareto
description: Regenerate or read the blended-strength vs slowdown Pareto (the north-star metric and README headline chart)
---

Run `python benchmarks/make_pareto.py` (newest `benchmarks/results/*.json`; pass a path for a
specific run; `--no-image` for the text table only). It emits `images/pareto.png` + a
phone-readable table. Show the table to the user.

Metric (defined in make_pareto.py's docstring, reuses `summarize.aggregate`):
- classification = ⅔·Brier% + ⅓·F1% (all "% vs best on task", higher = better)
- blended = HarmonicMean(RegRMSE%, classification) — harmonic on purpose: it tracks the weak leg
- x-axis = slowdown (mean fit-time multiple vs fastest model); frontier = up-and-left

Notes:
- The input JSON must be a fresh multi-model run (ChimeraBoost + CatBoost + LightGBM + sklearn_HGB
  at minimum) — mixing runs from different machines/fields breaks the % and speed columns.
- `images/pareto.png` is committed and shown in the README — commit the refresh after a shipped change.
- Reference points (2026-07-13): defaults (cross_features auto) 99.4 @ 7.9× — every accuracy column #1,
  CatBoost 98.1 @ 11.8× dominated; `cross_features=False` recovers the fast point 98.7 @ 3.7×.
