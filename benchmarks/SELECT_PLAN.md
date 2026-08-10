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

## Default flipped to rung 3 (Nathan, 2026-07-25)

> "the DEFAULT should be whichever has best accuracy without ensembling."

That is rung 3 — `refit_full=True`, 69.9% vs rung 2's 45.6%. Re-charted so
the arm labelled `ChimeraBoost` IS the default
(`results/20260725-213827.json`):

| rung | arm | win rate | slowdown | frontier |
|---|---|--:|--:|---|
| 5 max | Ens8 | 94.0% | 25.66x | yes |
| 4 ensemble | Ens5 | 83.7% | 17.67x | yes |
| **3 accurate = DEFAULT** | **ChimeraBoost** | **69.9%** | **9.35x** | **yes** |
| 2 balanced | NoRefit | 45.6% | 5.36x | yes |
| 1 fast | OneLin | 31.1% | 2.00x | yes |
| — | CatBoost | 40.9% | 14.14x | **no** |
| — | LightGBM | 21.6% | 1.12x | yes |
| — | sklearn_HGB | 13.3% | 4.84x | no |

The default now beats CatBoost on **49 of 59 datasets while fitting 1.9x
faster than it**. No new gate is owed: `refit_full` already passed every
registered bar in REFIT_PLAN (gr 48W-11L, hc 10W-3L, OpenML one-shot
26W-8L), and it has never shipped in a release — it sat in
[Unreleased] — so no released behaviour changes under anyone.

**Trap this created.** The rung-1/rung-2 arms only *avoided setting*
`refit_full`; with the default ON they silently inherited it and stopped
being those rungs. `_run_chimera` now takes `refit_full="off"` to force
it, matching the `linear_leaves`/`cross_features` convention. Without
that fix the ladder would have collapsed to three near-identical points
and read as a real result.

**Golden coverage gap (pre-existing, worth knowing).** The
numerical-identity goldens run with `early_stopping=False`, a path where
`refit_full` is a no-op — so they stayed green through a default flip
that changes every ordinary fit. They do not cover this.

## Shipped

`quality=1..5` (`chimeraboost/sklearn_api.py`), a named operating point
that only pins existing parameters. `quality=None` and `quality=3` are
the defaults; `quality=2` reproduces 0.24.0 behaviour; `quality=1` is
bit-identical to the benchmarked `OneLin` arm on both estimators — so the
numbers above are the parameter's own, not a lookalike's. Docs:
`docs/recipes.md`, `docs/parameters.md`.

## `Sel25` — KILLED 2026-07-31, do not retry below k=100 without a new mechanism

Ran the full /experiment gate on flipping the `selection_rounds` default
from 100 to 25. **It fails.** Both arms ran inside ONE benchmark each
(`compare_runs RUN RUN --model ChimeraBoost --model-new ChimeraBoostSel25`),
so the pairing is on identical machine conditions, and every number below is
post-PR-#60 code.

### Tier 1 synth (`results/20260731-163807.json`) — PASS, flat everywhere

136 datasets, 3 seeds. 29W-26L-**81T**, mean −0.066%, median exactly
+0.000%; on Brier 15W-15L-58T, mean −0.050%. `synth_report` finds **no**
slice carrying concentrated harm — worst is func=tree at −0.698%, which is
18 sets with 10 ties and p=0.73, and every factor-OLS |t| < 1. Fit time
−17% (not decision-grade, per CLAUDE.md).

Structural note: 47 of the 48 sets under 2000 rows are **bit-identical**.
Below the row floors (`LINEAR_LEAVES_MIN_SAMPLES=1000`,
`CROSS_MIN_SAMPLES=2000`) no audition runs at all, so the budget is inert.

### Tier 2 decide (`results/20260731-164034.json`) — FAIL on Grinsztajn regression

103 datasets, 7 strata, 3 seeds. Judged on the decision metric (RMSE
regression / Brier classification), per stratum, never pooled:

| stratum | task | W-L-T | p (exact binomial) | median |
|---|---|--:|--:|--:|
| **Grinsztajn** | **regression** | **6-20-10** | **0.009** | **−0.011%** |
| Grinsztajn | classification | 8-10-5 | 0.815 | +0.000% |
| HC | regression | 1-2-3 | 1.000 | +0.000% |
| HC | classification | 4-2-2 | 0.688 | +0.037% |
| gr@sus25 | both | 3-3-6 | 1.000 | +0.000% |
| gr@sus50 | both | 2-4-0 | 0.625 | −0.117% |
| hc@sus25 / hc@sus50 | both | 0-0-7 | 1.000 | +0.000% |
| hc@time | both | 3-1-3 | 1.000 | +0.000% |

The pre-registered bar was non-inferiority on every stratum. Grinsztajn
regression regresses at p=0.009 — the primary decision suite, on its own
decision metric. Kill clause fires.

The loss is **broad, not one dataset**: 20 separate regression sets move
against it, most by ≤0.1%, with four real ones — `Brazilian_houses`
(−4.43% cat / −5.03% num) and `nyc-taxi-green-dec-2016` (−1.70% cat /
−4.18% num). Many small same-signed deltas is what drives p that low.

### Why this is not the 2026-07-25 reading

Phase 0 recorded 20W-23L-16T on Grinsztajn `primary` at a rung-2 default;
the same comparison today reads 15W-28L-16T, and the regression-only cut is
6W-20L. Two things changed: the default is now rung 3, and 0.27.0 made
growth cheaper.

**The mechanism was already characterized in `PARETO_PLAN.md:196-201`**:
the const-vs-linear race "genuinely crosses late on ~1/3 of regression
selections", with no margin rule at k=100 able to separate them, and
`k_ll=500` fixing fidelity only by collapsing the speedup. `k=100` shipped
in July with that mispick tail knowingly waived (its own Grinsztajn sign
test was already 8W-22L-29T). **Going to k=25 auditions even earlier and
widens exactly that tail.** This is not a new failure mode; it is the known
one, further in.

### The refit AMPLIFIES the mispick (`results/20260731-164927.json`)

Unregistered follow-up, added because Phase 0's rung-2 reading was so much
milder. Same suite, same seeds, one run, new arm `ChimeraBoostNoRefitSel25`
(rung 2 + k=25) against `ChimeraBoostNoRefit` (rung 2):

| default the arms sit on | gr regression W-L-T | p | mean |
|---|--:|--:|--:|
| rung 3 (`refit_full="replay"`, today's default) | 6-20-10 | **0.009** | −0.478% |
| rung 2 (`refit_full=False`) | 10-16-10 | 0.327 | **+0.409%** |

**The same knob change is decisively harmful with the refit and not
detectably harmful without it**, and the mean even flips positive. The
mechanism is that rung 3 *replays the audition winner's tree structure over
every training row* — so when a short audition picks the wrong
configuration, the refit propagates that structure instead of diluting it.
The audition decision became higher-stakes the day the default moved to
rung 3.

Read this as directional, not settled: 10W-16L at n=36 with 10 ties is
underpowered, so the honest claim is "the loss does not reproduce without
the refit", not "rung 2 is provably flat". The contrast against p=0.009 is
what carries the weight.

Consequence for the record: **`Sel25` was never really re-tested against the
config it was measured on.** The 07-25 parity finding was a rung-2 result
and is broadly consistent with the rung-2 numbers here. It was invalidated
by the default flip, not by being wrong.

### The speed case was weaker than advertised anyway

| stratum | total fit ratio | median per-dataset ratio |
|---|--:|--:|
| Grinsztajn | 0.911 | 0.793 |
| HC | 0.823 | 0.869 |
| all 103 (speed only) | **0.887** | 0.794 |

Grinsztajn buys **8.9%**, not the 26% Phase 0 quoted — the rung-3 replay
refit is a fixed leg the audition budget cannot touch, and the biggest
datasets (which dominate the sum) save least. For contrast, the original
`None`→100 cut bought **1.50x** on the same suite. The audition saving is
essentially exhausted at 100; 25 buys 9% more while pushing further into a
documented accuracy failure mode. Even at parity that would have failed the
registered ≥10% speed bar.

### Recorded negative

No source change. `selection_rounds` stays 100. Live sub-questions for
anyone revisiting:

- A **per-leg** budget. The mispick is specifically the const-vs-linear
  race, which `PARETO_PLAN` says wants k_ll≈500; the cross race may be fine
  short. A single global k below 100 is closed.
- **Selection fidelity is worth more under rung 3 than it was under rung 2**
  (see the amplification section). Any future audition-cheapening idea has
  to be judged against the *refit* default, and any idea that makes the
  audition more faithful is worth more than its 07-25 valuation.
- Whether the `@sus50` median of −0.117% is real or noise at n=6.
- Cheapest repro: `gr:*/nyc-taxi-green-dec-2016` loses in all four of its
  appearances (−1.70% / −4.18% / −1.92% @sus25 / −2.94% @sus50).

Harness: `ChimeraBoostSel25` and the new `ChimeraBoostNoRefitSel25` arms are
both registered in `run_benchmarks.py`, off by default, so any re-test is
one `--models` flag.

## Still open

- **A "use cross features without auditioning" mode.** Rung 1 currently
  drops cross features entirely because `cross_features=True` still races
  (`sklearn_api.py:1443`). A forced-on mode could give a stronger rung 1
  at the same one-fit cost. Untested.

## E1 — pre-registered 2026-08-10: the rule, not the budget

**Status: pre-registration only. Nothing has run at the decision tier, and the
step below that must run first is an attribution run, not a decide run.**

### Why this reopens a closed knob

The "per-leg budget" recorded above as sel25's live sub-question is **empty**, and
the argument is from source rather than from a run: the augmented candidate wins
20 of 21 selections and `_stop_if_behind` lets a leading augmented fit run to its
own early stop, so shortening the cross leg saves nothing on 20 of 21 datasets;
and lengthening it past `k_ll` reintroduces the asymmetry the symmetric-budget
rule exists to remove (`sklearn_api.py:2004-2011`). Full reasoning in
`PARETO_PLAN.md`, "The dual re-read", D1.

What is *not* empty is the decision rule. `benchmarks/probe_audition_rule.py`
replays rules against the full validation curves in `results/pareto-step0.json`
(zero benchmark cost; it reproduces the recorded 4/12 mispicks at k=100 exactly,
which is its self-check). It finds a dead flat plateau from k=50 to k=200 under
every rule shape tried, near-zero regret where mispicks happen — and one
exception: at **k=25** the shipped best-so-far rule mispicks
`gr:reg_cat/nyc-taxi-green-dec-2016` at 1.452% validation regret, while a
**tail-mean** rule over the same budget reads 4/12 at 0.511%, matching the
shipped rule's fidelity at k=100 for a quarter of the budget.

That dataset is the one this file already names as sel25's cheapest repro. So the
kill has a mechanism now: it was the rule that failed at k=25, not the budget.

### The arm

`selection_rounds=25` with the const-vs-linear decision taken on the **mean of
the last 20 audition rounds** instead of the best value seen. Tie still goes to
constant leaves. The cross race is untouched, and so is every non-regression path.

### Forecast (both axes, recorded before the run)

- **Cost**: −8 to −10% Grinsztajn fit time, i.e. sel25's measured 8.9%. High
  confidence; this is arithmetic on the audition, not a new mechanism.
- **Strength**: flat on Grinsztajn regression, ±0.15% median. This is the whole
  bet, and the honest prior on it is weak — the evidence is one race flipping at
  n=12.
- **Where I expect to be wrong**: that the tail-mean advantage is one dataset's
  noise and does not survive a wider curve corpus. That is what step 1 tests.

### Steps, in order

1. **Widen the curve corpus first — an attribution run, not a decide run.**
   `profile_fit.py --attribution` over more regression datasets, then re-run
   `probe_audition_rule.py`. n=12 races on 4 datasets is a pointer and cannot
   gate anything (`GATE_ROBUSTNESS.md` #2). **Kill here** if tail-mean's edge at
   k=25 does not survive: median regret at k=25 must be no worse than the shipped
   rule at k=100, on at least 30 races. **DONE 2026-08-10 — PASS, see "Step 1
   result" below.**
2. Tier-1 synth screen. Note B1: below 1000/2000 rows no audition runs at all, so
   the small-n slice is structurally inert and must read as exact ties — pass
   `--expect-inert`, and treat engagement there as a bug report.
3. Tier-2 `--decide --seeds 3 --save`, judged at `refit_full=True` (B2 — rung 2
   would flatter it, as it flattered sel25).

### Step 1 result (2026-08-10): 12 → 48 races — bar PASSES; the finding replicates

New corpus: `results/audition-corpus-e1.json` (git-ignored, like step-0) — 12
regression dataset-variants x 3 seeds not already in step-0, deliberately
including the sel25 kill's worst losers (both `Brazilian_houses` variants, the
numeric taxi variant), so the widening is stress-weighted *against* the rule.
Command: `profile_fit.py --attribution --seeds 3 --uncapped-auditions
--datasets ... --out audition-corpus-e1`; replay:
`probe_audition_rule.py --k 25 50 100 200 --corpus
results/audition-corpus-e1.json` (self-check 4/12 intact).

Two instrument traps found and fixed on the way, worth knowing for any future
corpus widening:

- **A capped attribution run is circular for this probe.** Under today's
  `selection_rounds=100` default the recorded curves *stop at the cap*, so the
  replayed "full-curve truth" is the k=100 decision itself — the first attempt
  read a trivially perfect 36/36 agreement at k≥100. `profile_fit.py
  --attribution` now takes `--uncapped-auditions` (`selection_rounds=None`,
  the conditions step-0 was recorded under); the probe's corpus must come from
  such a run.
- **The rung-3 replay refit clobbers audition labels.** It appends a fourth
  booster-fit record that reuses the winner's label with an empty
  `valid_history` (it has no validation loop); the probe's last-wins label
  dict silently dropped 15 of 36 races. The loader now keeps the first fit
  per label — step-0 output byte-identical before/after, self-check still
  4/12.

The registered bar — tail-mean@25 median regret no worse than shipped@100 on
≥30 races — **passes on n=48** (0.000% vs 0.000%). Honest caveat: *every*
rule/k median in the table is 0.000%, so the registered statistic turned out
to be nearly vacuous. The sharper paired reading over the same 48 races:

| candidate | mispicks | mean regret | total regret | worst regret |
|---|--:|--:|--:|--:|
| shipped@25 | 14/48 | 0.299% | 14.33% | 4.33% |
| **tail@25** | **12/48** | **0.262%** | **12.58%** | 4.33% |
| shipped@100 (today) | 12/48 | 0.194% | 9.31% | 3.09% |
| tail@100 | 15/48 | 0.195% | 9.38% | 3.09% |

- **The pointer survived widening.** tail@25 strictly dominates shipped@25:
  it fixes both taxi mispicks (`gr:reg_cat` s2 at 1.452%, and a new one,
  `gr:reg_num` s1 at 0.299%) and pays nothing extra on any other race. The
  forecast's "where I expect to be wrong" (one dataset's noise) did not
  materialize.
- **tail@25 vs shipped@100 is a wash on mispicks** (12 = 12). The mean-regret
  gap (0.262% vs 0.194%) is entirely ONE dataset — `analcatdata_supreme`
  (3,039 train rows, the panel's smallest): its s1 race costs tail@25 4.331%
  while its s0 race costs shipped@100 1.063%, opposite directions on
  different seeds of the same data. Per GATE_ROBUSTNESS #3, a mean moved by
  one dataset is not evidence either way.
- **The tail rule is k=25-specific, not a general upgrade**: at k=100 it is
  *worse* than the shipped rule (15 vs 12 mispicks). E1's arm stays exactly
  as registered — k=25 + tail-mean — and no one should "improve" the k=100
  default rule off this table.

**Consequence: E1 proceeds to step 2.** The arm needs a library knob that does
not exist yet (the const-vs-linear decision is hard-coded best-so-far in
`sklearn_api.py`); implementing it is a source change owned by the /experiment
protocol on its own branch, with the forecast above already registered.

### Barriers this owes an argument about

- **B1** (audition knobs inert on small data) — accepted, not contested; it is
  why the small-n strata are a control here rather than evidence.
- **B2** (the refit amplifies a bad audition) — this arm is *aimed* at that
  finding: it cuts the budget only after making the decision more robust.
- **B12** (price auditions against the short-fit case) — a 25-round audition is
  cheap even against a ~100-round multiclass fit, so this one is cleared rather
  than argued.
- **B14** (no margin rule separates the race at k=100) — not contested. The probe
  extends it: at k=100 *no* rule shape separates them. The claim here is only
  about k=25, where the curves do still differ.

### Kill clause

Registered before the run: fails step 1's bar, or any decisive per-stratum sign
test against it at tier 2, and `selection_rounds` stays 100 permanently — the
budget axis is then closed from below as well as above, and B14 gets a line
saying so.

## Log

- 2026-07-25: pre-registered.
- 2026-07-25: Phase 0 PASS on `OneLin`; `One` killed; `Sel25` parity
  finding recorded for a separate /experiment.
- 2026-07-25: Phase 1 PASS, high-card caveat measured, ladder co-run
  charted, `quality=1..5` shipped. Program closed.
- 2026-07-31: `Sel25` run through the full gate and **KILLED**. Synth PASS
  (flat, no slice harmed); decide tier FAIL — Grinsztajn regression
  6W-20L-10T at p=0.009. Speed case also weaker than recorded (8.9% on
  Grinsztajn, not 26%). No source change. Follow-up found the **refit
  amplifies it**: the same knob reads 10W-16L-10T, p=0.327, mean +0.409% at
  rung 2, so the 07-25 parity finding was a valid rung-2 result invalidated
  by the default flip rather than a bad measurement.
- 2026-08-10: dual re-read of this kill (`PARETO_PLAN.md`, D1). The per-leg
  budget recorded above as the live sub-question is **empty** — argued from
  source and the 20/21 cross flip rate, no run spent. The replacement,
  **E1**, is pre-registered above: the decision RULE at k=25, not the
  budget. New instrument: `benchmarks/probe_audition_rule.py`, which
  reproduces the recorded 4/12 mispicks at k=100 and then explains this
  kill — at k=25 the shipped rule mispicks `nyc-taxi-green-dec-2016`
  (this file's own cheapest repro) at 1.452% validation regret, which the
  rung-3 refit amplifies. Nothing has run at the decision tier.
- 2026-08-10 (later): **E1 step 1 PASS** — corpus widened 12 → 48 races
  (`audition-corpus-e1`, uncapped auditions), self-check intact at 4/12.
  tail@25 replicates its edge (strictly dominates shipped@25, fixes both taxi
  mispicks) and matches shipped@100 on mispicks 12 = 12; the residual
  mean-regret gap is one dataset (`analcatdata_supreme`), swinging both ways
  across seeds. Registered median bar passes but proved nearly vacuous (all
  medians 0.000%) — recorded so the next pre-registration picks a sharper
  statistic. Two instrument fixes: `--uncapped-auditions` on the attribution
  runner (capped curves make the replay circular) and first-fit-per-label in
  the probe loader (the rung-3 replay refit's empty record was clobbering
  audition curves). Next: step 2 tier-1 synth, which first needs the arm
  implemented behind a knob (/experiment territory, separate branch).
