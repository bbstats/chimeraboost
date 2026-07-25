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

## Phase 0 results (2026-07-25, `results/20260725-150712.json`)

Grinsztajn, 3 seeds, 177 dataset-seed draws, 57 datasets scored
head-to-head. Field as registered (no Ens8), so **absolute win rates are
NOT comparable to the charted 55.7%** — every bar is judged within-run.

| model | win rate | 95% CI | slowdown | frontier |
|---|--:|--:|--:|---|
| ChimeraBoost (default) | 72.2% | [65.5, 78.4] | 5.33x | yes |
| ChimeraBoostSel25 | 70.2% | [64.2, 75.9] | 3.93x | yes |
| CatBoost | 64.9% | [56.7, 73.4] | 14.74x | — |
| **ChimeraBoostOneLin** | **52.0%** | [45.5, 58.6] | **1.98x** | **yes** |
| LightGBM | 35.4% | [28.6, 42.7] | 1.26x | yes |
| ChimeraBoostOne | 31.6% | [24.6, 38.9] | 1.73x | — |
| sklearn_HGB | 23.7% | [15.8, 31.9] | 5.65x | — |

### Verdict: `OneLin` PASSES all three bars — the 1-5x region is reachable

- **Bar A** 1.98x ≤ 2.5x PASS. **Bar B** chord at 1.98x needs 41.8%,
  `OneLin` reaches 52.0% — **+10.2 points above the chord**, PASS.
  **Bar C** vs LightGBM on the primary sign test: **38W-21L-0T, mean
  +0.664%, median +0.133%** (n=57 after the near-solved guard dropped
  `SGEMM_GPU_kernel_performance` and `visualizing_soil`) PASS.
- One booster fit instead of 2-4 costs **20.2 points** of win rate
  (72.2 → 52.0) and returns **2.7x** of fit time (5.33 → 1.98). The
  search is load-bearing, exactly as P2 predicted — but P4 was right that
  cost falls faster than strength.

### `One` is dead — constant leaves are the wrong economy

31.6% @ 1.73x is **dominated by LightGBM itself** (35.4% @ 1.26x: both
stronger and faster). Dropping selection is survivable; dropping linear
leaves with it is not. Predictions P1/P2 held (0.32x of default fit;
default beats it 47W-12L).

### Unregistered second finding: the audition budget is oversized

`Sel25` (auditions cut 100 → 25) is **statistically indistinguishable
from the default** — 20W-23L-**16T**, mean **−0.044%**, median **exactly
+0.000%** — while returning 26% of fit time (5.33x → 3.93x). This is
parity, not a win; the sign-test FAIL is the ship bar for an
*improvement*, which is the wrong instrument for a non-inferiority
question. Recorded as a candidate default change requiring its own full
/experiment (synth → gr + hc → OpenML gate), NOT shipped here.

Note the 24 exact ties in this run (charted runs report 0): `Sel25` and
the default frequently resolve to the identical model, which is itself
the evidence that the extra 75 audition rounds usually change nothing.

## Phase 1 — canonical field (`results/20260725-152037.json`)

`OneLin` holds up against the charted 5-arm field plus itself: **41.4% @
1.93x**, on the frontier, chord needed 34.8% ⇒ **+6.6 points**. Reference
points that run: ChimeraBoost 60.0% @ 5.13x, Ens8 98.6% @ 24.80x,
LightGBM 28.8% @ 1.16x, CatBoost 53.0% @ 13.61x (dominated).

### High-card is where rung 1 is weakest (`results/20260725-154149.json`)

Sign-tested separately as registered, 14 sets, 3 seeds:

- vs the default: **1W-6L-7T, mean −0.228%, median +0.000%** — it loses.
- vs LightGBM: **7W-6L-1T, mean +0.936%** (7W-5L excluding the near-solved
  `hc:cjs`; the full-set bar reads FAIL, the near-solved-excluded bar
  PASS — `compare_runs` flags the disagreement).
- Speed saving shrinks to **1.54x** (vs 2.7x on Grinsztajn): the cat/prep
  block is a fixed 15-25% of fit that skipping auditions cannot touch.

Documented as a caveat rather than hidden — rung 1 is for numeric-heavy
data and sweeps; categorical-dominated data wants rung 2.

## Ladder co-run (`results/20260725-155127.json`) — all five rungs on the frontier

Every rung measured in ONE run against one field, which is what makes the
column comparable. This run regenerated `images/pareto.png`.

| rung | arm | win rate | slowdown | frontier |
|---|---|--:|--:|---|
| 5 max | Ens8 | 94.0% | 24.74x | yes |
| 4 ensemble | Ens5 | 83.7% | 17.07x | yes |
| 3 accurate | Refit | 69.9% | 9.15x | yes |
| 2 balanced | ChimeraBoost | 45.6% | 5.22x | yes |
| 1 fast | OneLin | 31.1% | 1.95x | yes |
| — | CatBoost | 40.9% | 13.93x | **no** |
| — | LightGBM | 21.6% | 1.14x | yes |
| — | sklearn_HGB | 13.3% | 5.39x | no |

- **The frontier is a ChimeraBoost ladder**: five non-dominated rungs from
  1.95x to 24.74x, with LightGBM holding only the extreme-left corner.
  CatBoost is dominated outright — rung 3 is both stronger (69.9 vs 40.9)
  and faster (9.15x vs 13.93x).
- **Rung 4 finally has a number: 17.07x**, not the stale ~43x that
  predated quantized histograms and `selection_rounds=100`.
- Win rate is **field-relative**. Absolute values are lower here than in
  Phase 1 because the field now holds five ChimeraBoost arms, which are
  strong opponents. Only within-run comparisons are meaningful.
- Bars A/B were registered for the FAST rung. Applying the chord to rungs
  3-5 is meaningless — it extrapolates past its own strong anchor and
  demands >100% win rates. Frontier membership is by domination and is
  the valid reading there.

## Shipped

`quality=1..5` (`chimeraboost/sklearn_api.py`), a named operating point
that only pins existing parameters. `quality=None` and `quality=2` are
both bit-identical to the current defaults, and `quality=1` is
bit-identical to the benchmarked `OneLin` arm on both estimators — so the
numbers above are the parameter's own, not a lookalike's. Defaults
unchanged, so no OpenML gate is owed. Docs: `docs/recipes.md`,
`docs/parameters.md`.

## Still open

- **`Sel25`** — auditions at 25 rounds instead of 100 were parity on
  Grinsztajn (20W-23L-16T, mean −0.044%, median exactly +0.000%) for 26%
  less fit time. A default change, so it needs its own /experiment with
  the full gate. The 16 exact ties are the tell: three-quarters of the
  audition budget usually changes nothing.
- **A "use cross features without auditioning" mode.** Rung 1 currently
  drops cross features entirely because `cross_features=True` still races
  (`sklearn_api.py:1443`). A forced-on mode could give a stronger rung 1
  at the same one-fit cost. Untested.

## Log

- 2026-07-25: pre-registered.
- 2026-07-25: Phase 0 PASS on `OneLin`; `One` killed; `Sel25` parity
  finding recorded for a separate /experiment.
- 2026-07-25: Phase 1 PASS, high-card caveat measured, ladder co-run
  charted, `quality=1..5` shipped. Program closed.
