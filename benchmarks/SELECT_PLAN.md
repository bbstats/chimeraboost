# SELECT — selection economics and the empty left half of the Pareto

Pre-registered 2026-07-25. North star: `benchmarks/make_pareto.py`
(head-to-head win rate vs mean fit-time multiple). Nothing here touches
TabArena.

## Question

The charted frontier has three points: LightGBM 28.5% @ 1.0x,
ChimeraBoost 55.7% @ 4.9x, Ens8 98.2% @ 23.6x. **Between 1x and 4.9x
there is nothing.** No ChimeraBoost operating point has ever been
measured there.

The cost structure says why that region should be reachable. A default
regression fit is not one model, it is a search:

| leg | site | budget |
|---|---|---|
| const-leaf audition | `sklearn_api.py:1475` | capped at `selection_rounds` (100) |
| linear-leaf audition | `sklearn_api.py:1476` | capped at 100 |
| cross-augmented candidate | `sklearn_api.py:1501` | full if leading, killed at 100 if behind |
| winner refit from round 0 | `sklearn_api.py:1511` | full, only when the audition was capped AND cross lost |

Up to 4 booster fits for regression, up to 3 for classification
(`sklearn_api.py:2101-2181`). Tree growth is 73-92% of a fit on
Grinsztajn (`benchmarks/results/pareto-step0.md`) and **every leg pays it
in full**. LightGBM — the 1.0x denominator — fits once.

So the question is not "can a kernel be made faster" (that program closed;
see GROW_PLAN). It is: **how much of the 4.9x is search, and what does
the search actually buy?**

## Arms (Phase 0)

Grinsztajn, 3 seeds, out-of-box defaults except where stated. All arms in
ONE run so win rates are computed against an identical field.

| arm | config | booster fits |
|---|---|---|
| `ChimeraBoost` | default (control) | 2-4 |
| `ChimeraBoostSel25` | `selection_rounds=25` | 2-4, shorter auditions |
| `ChimeraBoostOne` | `linear_leaves=False`, `cross_features=False` | **1** |
| `ChimeraBoostOneLin` | `linear_leaves=True`, `cross_features=False` | **1** |

`ChimeraBoostOne` / `OneLin` disable selection by making both decisions
statically (`select_ll` needs `linear_leaves is None`, `sklearn_api.py:1436`;
`cross_ok` needs `cross_features is not False`, `:1443`). They are the
speed floor: exactly one fit, no search.

Field for Phase 0: LightGBM, CatBoost, sklearn_HGB. **Ens8 is excluded**
— it is a 23.6x top-right point that dominates wall-clock and does not
affect the chord test below. Absolute win rates in this run are therefore
NOT comparable to the charted 55.7%; every bar is evaluated within-run.

## Pre-registered predictions

- **P1 (cost).** `One` fit time lands at 0.35-0.55x of default
  ChimeraBoost. Rationale: default is ~2-3 effective full fits.
- **P2 (strength).** `One` loses to default on the primary sign test.
  Selection is load-bearing — this is the repo's base rate (B1 pinning
  −0.59%, B2 shared stopping −1.42%). Predicted mean −1% to −3%.
- **P3.** `Sel25` sits within ±0.3% of default at 0.85-0.95x fit.
- **P4 (the real question).** Despite P2, `One` or `OneLin` clears the
  LightGBM→ChimeraBoost chord, because the cost falls faster than the
  strength does.

## Bars — all three must pass for the program to continue

- **Bar A (cost).** Some single-fit arm reaches **≤2.5x** mean fit-time
  multiple on the within-run slowdown axis.
- **Bar B (frontier).** That arm's win rate is **strictly above the chord**
  from LightGBM to default ChimeraBoost evaluated at its own slowdown:
  `wr > wr_lgbm + (wr_default − wr_lgbm) · (s − s_lgbm) / (s_default − s_lgbm)`
  — all four quantities read from this same run. Being merely
  non-dominated is not enough; an interior point is not a frontier point.
- **Bar C (not a mirage).** The arm beats LightGBM on the primary sign
  test (RMSE regression / Brier classification). It must be genuinely
  stronger, not merely faster.

## Kill clause

If no fast arm clears Bar B, the program **closes same-day**. The finding
is then recorded as a real negative — the search is load-bearing, the
1-5x region is genuinely empty, and ChimeraBoost has no cheap operating
point. No retry without a new mechanism.

## If it passes — Phase 1

Canonical run: the charted 5-arm field (incl. Ens8) plus the winning fast
arm, 3 seeds, Grinsztajn. Refresh `images/pareto.png` and
`winrate_matrix.png`. Ship as a **documented preset in `docs/recipes.md`**
— explicitly NOT a default flip. High-card (`--highcard`) is sign-tested
separately before any docs claim, since its cat/prep block is 15-25% of
fit and the search economics differ there.

## Scope

Benchmarks and docs only. No library change, no default change, so no
OpenML gate is owed. If a later idea needs a "use cross features without
auditioning" mode (no such mode exists today — `cross_features=True`
still races, `:1443`), that becomes its own /experiment with a gate.

## Log

- 2026-07-25: pre-registered. Phase 0 pending.
