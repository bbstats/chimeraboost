# BREAKTHROUGH — hunting a significant Pareto move (opened 2026-08-01)

Goal set by Nathan: a **significant** strength-vs-slowdown Pareto improvement.
Not a hairline. Honest methods only — TabArena stays sealed, decisions run on
synthetic → Grinsztajn + high-card, per-stratum sign tests.

## Where we actually stand (baseline `results/20260731-142609.json`, 3 seeds)

| model | Reg RMSE% | Bin Brier% | Slowdown |
|---|---|---|---|
| ChimeraBoost | 98.4 | 99.0 | 7.1x |
| ChimeraBoostEns8 | 99.8 | 99.8 | 26.8x |
| CatBoost | 95.2 | 94.4 | 15.3x |
| LightGBM | 93.2 | 94.7 | 1.1x |
| sklearn_HGB | 93.4 | 94.5 | 4.7x |

Head-to-head (primary metric, near-solved excluded, 57 sets): we beat CatBoost
46W-11L (80.7%), LightGBM 56W-1L (98.2%), HGB 55W-2L (96.5%). We **dominate
CatBoost on both axes** — stronger and less than half the fit time.

**Every remaining loss is noisy, low-dimensional binary classification**, gaps
under 1.1% of Brier: bank-marketing (+1.02% vs CatBoost), credit (+0.47%),
heloc (+0.43%), california (+0.28%), default-of-credit (+0.15%),
Diabetes130US (+0.13%), plus electricity vs HGB (+1.07%) and road-safety vs
HGB (+0.32%). Loss rate by suite: clf_num 16.7%, reg_num 3.5%, clf_cat 4.8%,
reg_cat 6.7%. Not a size story (flat across size buckets), not a categorical
story (cat suites lose LESS).

Do not break: Brazilian_houses (−22% to −35% vs the field), covertype (−17% to
−32%), sulfur, pol. Highly structured low-noise sets where ES + refit compound.

---

## FINDING 1 — the gap is RESOLUTION, not calibration. Calibration levers are dead. (2026-08-01)

**Refuted before any code was written**, from the MCB term the harness already
records (CORP isotonic gap = exactly what a perfect recalibration would recover).

Mean miscalibration across the 23 classification sets:

| model | mean MCB | mean Brier | MCB as share of Brier |
|---|---|---|---|
| **ChimeraBoost** | **0.002197** | 0.27089 | 1.07% |
| CatBoost | 0.002310 | 0.27913 | 1.09% |
| LightGBM | 0.002326 | 0.27813 | 1.02% |
| sklearn_HGB | 0.002716 | 0.27989 | 1.19% |

**We are already the best-calibrated model in the field.** And the decisive
control: subtract each model's own MCB — i.e. perfectly recalibrate *both*
sides — and the scoreboard does not move at all.

| opponent | today | both sides perfectly recalibrated |
|---|---|---|
| CatBoost | 17W-6L (73.9%) | 17W-6L (73.9%) |
| LightGBM | 22W-1L (95.7%) | 22W-1L (95.7%) |
| sklearn_HGB | 21W-2L (91.3%) | 21W-2L (91.3%) |

Every matchup keeps its sign. It is true that our MCB exceeds the deficit on 7
of 9 losing matchups — but so does the opponent's, by the same margin, so the
comparison is untouched. The remaining deficit lives in **resolution**
(sharpness/structure), which no calibration map can reach.

⇒ **KILLED**: the *story* that we lose because we are miscalibrated relative to
the field. We are not. Any argument of the form "our gaps are calibration-scale,
therefore a calibration lever closes them" is refuted by the table above.
Script: `scratchpad/calib_headroom.py` (analysis only, no library change).

### Correction (same day) — this answered the wrong question

The counterfactual above recalibrates *both* sides, which is the right way to
ask "are we relatively miscalibrated?" It is the wrong way to ask the shipping
question, which is: **can we recover more of our own remaining 0.00220 while
the opponents stay where they are?** Those have different answers.

Our shipped map is a single scalar T minimizing validation log loss of
`sigmoid(raw/T)` (`_fit_temperature`). That family has **no intercept**, so it
cannot move the operating point of a skewed problem — on an 11%-positive set
the only reachable maps fix p=0.5 at raw=0. Five of the nine losing binary
matchups have gaps below 0.0006, i.e. under a third of our own MCB.

So the reliability lane is narrowed, not closed, and the open question is
specific: does a map *with* an intercept beat a scalar temperature
out-of-sample? Probe: `benchmarks/probe_calibration_map.py` — compares raw /
scalar-T(log loss) / scalar-T(Brier) / Platt / beta / isotonic, all fitted on a
held-out calibration fold and scored on test, over the 7 losing binary sets and
4 controls. Zero fit-time cost if it works, which makes it a pure Pareto move.

### Verdict: the calibration lane is CLOSED. Nothing there, with or without an intercept.

Mean improvement over the shipped scalar temperature (% of Brier, out-of-sample):

| map | losing matchups | controls |
|---|---|---|
| Platt `sigmoid(a·raw+b)` | **−0.027%** (better on 3/7, median −0.001%) | +0.076% (2/4) |
| beta (3-param) | −0.078% (1/7) | +0.060% (2/4) |
| isotonic | −0.587% (0/7) | −1.813% (0/4) |

Two things close this for good:

1. **Adding the intercept buys nothing** — Platt is a wash, beta is worse, and
   isotonic badly overfits the calibration fold.
2. **The `raw` column is the real finding.** Uncalibrated probabilities score
   essentially the same as the shipped temperature everywhere, and on
   bank-marketing and credit they score *better*. Our temperature scaling is
   already a near-no-op because the model comes out of the boosting loop
   calibrated.

And the mechanism argument dissolves on inspection of the suite: **every
Grinsztajn binary set is class-balanced (the `pos%` column is 50.0% on all 11
sets).** The intercept exists to move the operating point of a skewed problem;
the skew is not there. The impl judge's supporting measurement used synthetic
data at 11% positives — a real effect, in a regime our decision suites do not
contain.

⇒ **KILLED without implementation**: Platt, beta, isotonic, Brier-objective
temperature, and the whole reliability lane. Combined with Finding 1, no
calibration lever can move this benchmark.

---

## FINDING 2 — where does CatBoost's remaining edge come from?

Since the features on the losing sets are numeric, CatBoost's ordered target
statistics are irrelevant there. Its remaining distinctive default-ON machinery
is two stochastic regularizers we have never implemented (architecture map
degree-of-freedom #1, "no gain dithering/temperature"):

- `random_strength=1` — Gaussian noise added to each split score before the
  argmax, decaying with round index.
- `bagging_temperature=1` — Bayesian bootstrap per-tree row weights
  `w_i = u_i^(1/T)`, every row kept, mean weight 1.

Hypothesis: our pure greedy argmax over ~p×128 noisy gain estimates suffers a
winner's curse — the selected split is chosen partly for its noise and
underdelivers out of sample — and this is worst exactly on low-signal data.

**Probe (`probe_catboost_ablation.py`): ablate the OPPONENT.** Run CatBoost at
defaults, without `random_strength`, without the bootstrap, and without both,
on the 6 sets where it beats us plus 4 controls where we win comfortably.
Using CatBoost as an oracle for its own mechanism costs one short benchmark.

Pre-registered predictions:
- **Hypothesis right**: disabling both costs CatBoost most of its edge on the
  loss cluster while the controls move little ⇒ build the mechanism.
- **Hypothesis wrong**: CatBoost's Brier barely moves and its edge survives
  ⇒ **do not build it**; the edge lives elsewhere.

### Verdict: HYPOTHESIS WRONG. Mechanism not built.

Mean CatBoost edge over us on the 6-set loss cluster (% of our Brier, positive
= CatBoost ahead), 3 seeds, paired identical splits:

| CatBoost arm | mean edge, loss cluster | leads | mean edge, controls | leads |
|---|---|---|---|---|
| defaults | +0.624% | 5 of 6 | −15.31% | 0 of 4 |
| `random_strength=0` | +0.493% | 3 of 6 | −16.51% | 0 of 4 |
| `bootstrap_type="No"` | +0.558% | 5 of 6 | −14.82% | 0 of 4 |
| both disabled | +0.269% | 3 of 6 | −15.61% | 0 of 4 |

The controls validate the design: CatBoost is 15% behind us there regardless of
arm (covertype −45.8%, pol −8.4%), so the probe is measuring a real, specific
quantity and not a generic shift.

Removing either regularizer alone changes essentially nothing, and the two
ablations disagree about which one even helps. Removing both leaves CatBoost
still ahead. Upper bound on what a perfect port could recover: 0.36% of Brier
averaged over 6 datasets — roughly two matchup flips, about one win-rate point,
and that assumes we reproduce CatBoost's regularizers exactly. Below the bar,
and inside seed noise.

⇒ **KILLED without implementation**: `random_strength`-style split-score
dithering and Bayesian-bootstrap row weights as strength levers. This also
retires the "winner's curse on greedy argmax" story as the explanation for the
clf_num cluster. Probe: `benchmarks/probe_catboost_ablation.py` (resumable,
`--table-only` reprints).

Note the two candidates killed so far were the two that three independent
ideation lenses ranked highest. Cheap probes earned their keep twice.

---

## FINDING 3 — the real blind spot: we have never measured the variant strata against the field

Scanning every recent result file: **no run contains both external competitors
and the `@sus25`/`@sus50`/`@time` variant strata.** The runs that carry those
strata are ChimeraBoost-only A/B arms; the runs that carry competitors are
base-stratum only.

So our competitive standing is unknown in exactly the two regimes that matter
most for the ambition:
- **`@sus25`/`@sus50`** — the small-data regime, where tabular foundation
  models beat GBDTs by the widest margin.
- **`@time`** — distribution shift, which every random split is blind to and
  which no GBDT default addresses.

Chasing the 11 remaining CatBoost losses has a hard ceiling of about +8
win-rate points against one opponent. A whole stratum is a bigger prize, and
we cannot aim at it while blind.

### Result (`results/20260801-013306.json`, 103 datasets, 3 seeds) — THE FINDING

**Our win rate against CatBoost collapses as training data shrinks.** Head-to-head
on the primary metric, per stratum:

| stratum | n | vs CatBoost | vs LightGBM | vs sklearn_HGB |
|---|--:|---|---|---|
| gr:base | 57 | **81%** | 98% | 96% |
| gr:sus25 | 12 | **50%** | 83% | 92% |
| gr:sus50 | 6 | **33%** | 100% | 100% |
| hc:base | 13 | **31%** | 92% | 83% |
| hc:sus25 | 3 | **0%** | 100% | 100% |
| hc:sus50 | 1 | **0%** | 100% | — |
| hc:time | 7 | **43%** | 71% | 100% |

Against LightGBM and HGB we stay dominant everywhere (71-100%). The collapse is
**specific to CatBoost, and specific to small data** — which is exactly the
regime CatBoost's ordered boosting was designed for, and exactly the regime
tabular foundation models win.

Aggregate over all 7 strata: **83.0% win rate on 283 matchups**, against 91.8%
on the base stratum alone. One flip is now worth 0.35 points; 38 matchups sit
inside the 0.25% near-tie band, 15 of them losses.

The headline table also exposes the one column we do not lead:

| | Reg RMSE% | Bin Brier% | **Multi Brier%** | Slowdown |
|---|---|---|---|---|
| ChimeraBoost | **99.3** | **99.7** | **91.1** | **4.8x** |
| CatBoost | 97.5 | 96.7 | **98.8** | 44.6x |
| LightGBM | 94.7 | 96.3 | 97.5 | 1.1x |
| sklearn_HGB | 94.6 | 95.0 | 68.2 | 5.3x |

We beat the field on regression and binary while fitting **9x faster than
CatBoost** — and lose multiclass Brier by 7.7 points.

### The mechanism, from the worst case

`hc:eucalyptus@time` is our worst matchup anywhere (+56% vs CatBoost, +68% vs
LightGBM) and alone drags the hc:time *mean* gap to +4.877% while its *median*
is −1.215%. The per-seed numbers name the failure:

| seed (temporal cut) | ChimeraBoost | CatBoost | LightGBM | sklearn_HGB |
|---|---|---|---|---|
| 0 | 0.530 | 0.543 | 0.545 | 0.648 |
| 1 | 0.770 | 0.484 | 0.430 | 0.880 |
| 2 | **1.030** (log loss 2.99) | 0.464 | 0.409 | **1.104** |

We degrade monotonically across the cuts and end at Brier > 1.0 with log loss
near 3 — **confidently wrong**, not merely weak. sklearn_HGB fails the same way
on the same cuts; CatBoost and LightGBM stay flat. On the base variant of the
same dataset (552 rows, no shift) the gap is only +3.17%, so it is small data
*plus* shift that breaks us, and the failure mode is overconfidence.

Caveat kept honest: eucalyptus has 736 total rows, so each temporal test window
is small and these three numbers carry real variance. It is a pointer, not a
sign test.

⇒ **The program's target is now one regime, not a loss list: small data, and
small data under distribution shift.** Base-stratum Grinsztajn is close to
saturated at 91.8%; the headroom is in the strata nobody had measured.

---

## The leverage arithmetic (from the judge panel, and it reframes the hunt)

The decision run carries 177 matchups after the near-solved filter, so **one
flip is worth 0.56 win-rate points** and the 2-point bar needs 3.5 flips. But
**26 matchups sit inside a 0.25% band** (19 current wins, 7 current losses).

That means a *broad* quarter-percent shift is worth roughly 4-5 points —
secures the 19 near-ties we could lose and flips most of the 7 — while winning
all 11 named CatBoost losses is worth about 6 points against one opponent and
is capped there forever.

**Broad small levers beat targeted ones on this axis.** This is the same
flip-curve fact the refit_full program recorded (+0.25% broad primary ≈ +12pp
win rate). Every candidate should now be judged on how broadly it moves the
metric, not on how many named losses it targets — which is, in hindsight,
exactly why the three killed candidates were doomed: all three were aimed at a
named list of 6-11 datasets.

---

## Candidates (broad by construction) — ALL THREE RESOLVED as of 2026-08-01

| | candidate | outcome |
|---|---|---|
| C1 | bag-member refit | **SHIPPED** opt-in as `refit_members`; all gates passed, and `Ens5RM` dominates `Ens8` on both axes |
| C2 | per-dataset depth race | mechanism transfers, race **NOT BUILT** — priced below the bar by C3's oracle |
| C3 | regressor min-leaf floor | **KILLED**: direction real (p=0.002), magnitude inside noise, ceiling too low |

**No live candidate remains in this program.** What the hunt established is in
Finding 3: the small-data collapse against CatBoost (81% → 50%/33%/0%) is the
largest known headroom, `refit_members` addresses the bagged half of it, and
the single-model half is still open. Three capacity levers were aimed at it
and all three came back too small — capacity is not the mechanism. The next
program needs a different hypothesis, not another capacity knob.

### C1 — BAGREFIT: replay each bag member on the full row set

`_fit_bagged` hands every member an explicit OOB `eval_set`, and
`_refit_on_full` fires only when `auto_split` is True, so **no bag member has
ever run the refit that is worth +2.1 points to the single model.** Each
member's leaf values come from its 0.8n bag. REFIT_PLAN's "bag members have no
data tax" holds for the ensemble (every row is in some member's bag) but not
for any individual member.

Why it should be net positive rather than a diversity loss: the LRE kill
measured that bag lift is *structural* diversity and that leaf-level
resampling buys nothing. Replay keeps each member's structure exactly as its
own bag grew it and only re-estimates leaf values, so it should buy per-member
strength at close to zero diversity cost.

The prize is Pareto, not just strength: Ens8 is 99.8/99.8 at **26.8x**. If
refit members let a smaller bag reach that, the frontier moves a long way.
Probe sweeps K ∈ {2,3,5,8} with two arms — `refit` (same trees, pure
leaf-value effect) and `scaled` (rounds × 1/max_samples, the faithful
refit_full analogue, which also grows a tail) — so a win is attributable.
`benchmarks/probe_bag_member_refit.py`.

#### Probe verdict: CONFIRMED, broadly and unanimously.

10 datasets × 3 seeds, exactly paired (both arms replay the *same* fitted
members, so only the leaf values differ). Improvement over the shipped bag:

| arm | K=2 | K=3 | K=5 | K=8 |
|---|---|---|---|---|
| replay only | **+1.858%** (10/10) | +1.589% (10/10) | +0.920% (9/10) | **+1.023% (10/10)** |
| replay + regrown tail | +2.262% (9/10) | +1.992% (9/10) | +1.389% (9/10) | **+1.527% (9/10)** |

**The mechanism's own prediction held**: the gain is largest at small K
(+1.86% at K=2 falling to +1.02% at K=8), which is what "each member is
individually stronger" implies — with fewer members there is less averaging to
paper over weak leaf values. If this were a diversity loss the sign would go
the other way.

Biggest movers are the structured regression sets (sulfur +4.2%,
Brazilian_houses +2.1%, cpu_act +1.4%) plus electricity (+1.3% Brier); heloc is
the only near-zero. Nothing regresses at K=8 in the replay-only arm.

**And the Pareto question answers yes**: `refit@5` beats `plain@8` on 7 of 10
datasets (mean +1.022%), so a five-member refit bag reaches the eight-member
bag's strength — Ens8 is the 26.8x point on the headline chart.

Cost, measured in the probe: one replay arm is roughly **11% of the member-fit
time** (replaying a structure is far cheaper than growing it), so this is
about +1.0-1.5% strength for ~1.1x on the bagged path.

⇒ **Implemented** as `refit_members=False` (opt-in, byte-identical when off).
`_refit_on_full` gained a `train_frac` override so a member scales its round
budget by `max_samples` rather than `1 - validation_fraction`. Harness arms
`ChimeraBoostEns8RM/Ens5RM/Ens3RM` let both variants run inside ONE benchmark,
so the A/B pairing carries no machine-condition drift (the Sel25 precedent).

Gated off for **multiclass**: there is no replay path there
(`_refit_on_full` rebuilds a vector-leaf model from scratch), so a member
refit would cost a whole extra fit per member instead of a cheap structure
replay — a different trade, and one this evidence does not cover.

896 existing tests pass; 10 new ones in `tests/test_refit_members.py` lock the
contract: default off, off byte-identical, on engages, member structures
preserved and mutually distinct, single-model and explicit-`eval_set` paths
untouched, multiclass unchanged, `get_params`/`clone` roundtrip.

#### Pre-registered gate (stated before the runs)

- **Tier 1 (synth screen)**: `--synth --seeds 3 --models ChimeraBoost
  ChimeraBoostEns8 ChimeraBoostEns8RM --save`, then
  `compare_runs RUN RUN --model ChimeraBoostEns8 --model-new ChimeraBoostEns8RM`.
  Also read the Brier column — the standing rule for any
  classification-touching change.
- **Tier 2 (decide)**: same arm pair on `--decide`, per-stratum sign tests,
  never pooled.
- **Kill rules**: a broad regression in either primary or Brier on any stratum;
  or a fit-time cost materially above the ~1.1x the probe measured.
- **Ship shape if it passes**: opt-in `refit_members=True`, following the
  `refit_full` precedent where the default flip was Nathan's call. It only
  touches the bagged rungs (quality 4/5), so single-model defaults and the
  headline chart are unchanged by construction.

#### Tier 1 (synth screen) — PASS on primary, Brier a wash

`results/20260801-022821.json`, both arms in one benchmark, 136 datasets,
3 seeds.

| judge | result |
|---|---|
| **primary** | **71W-27L-38T, mean +0.738%, median +0.035% — sign test PASS** |
| Brier | 24W-30L-34T, mean **+0.181%**, median +0.000% — sign bar FAIL |

Read honestly: the 34 Brier ties are exactly the 34 multiclass sets the arm
gates off, so the binary Brier split is 24W-30L on 54 sets — a coin flip
(p≈0.5), with a positive mean and a median of exactly zero. That is a wash,
not the broad regression the kill rule names. The aggregate column moves the
right way too: Bin Brier% 98.3 → **98.6**, Reg RMSE% 97.9 → **99.8**.

Fit cost 4.4x → 4.8x = **+9%**, matching the probe's ~11% estimate.

The Brier sign count is recorded as a **watch item** for tier 2 rather than
waved away — the B1 lesson is that a classification-touching change can look
fine on primary and cost Brier, and that the screen predicted it.

#### Tier 2 (decide) — ALL GATES PASS

`results/20260801-023724.json`, all five arms in ONE benchmark so the pairing
carries no machine-condition drift. `ChimeraBoostEns8RM` vs `ChimeraBoostEns8`,
per stratum, never pooled:

| stratum | primary W-L-T | mean | median | Brier W-L | Brier mean |
|---|---|---|---|---|---|
| gr:base | **52W-7L-0T** | +0.500% | +0.175% | **20W-3L** | **+1.503%** |
| gr:sus25 | **12W-0L-0T** | **+1.206%** | +0.304% | 4W-1L | +0.248% |
| gr:sus50 | **6W-0L-0T** | +0.272% | +0.200% | 2W-0L | +0.736% |
| hc:base | 7W-2L-5T | +0.395% | +0.035% | 2W-2L-4T | −0.002% |
| hc:sus25 | 2W-0L-1T | +1.452% | +1.054% | 0W-0L-1T | +0.000% |
| hc:sus50 | 1W-0L-1T | +0.032% | — | — | — |
| hc:time | 3W-2L-2T | +1.020% | +0.000% | 0W-2L-2T | −0.102% |

**Every stratum has a positive mean on the primary metric, and the two
small-data Grinsztajn strata are perfect sweeps (12-0 and 6-0)** — the regime
Finding 3 identified as our weakness. The three strata that miss the sign bar
(hc:base, hc:sus50, hc:time) miss it because the bar counts ties against the
change: hc:base is 7W-2L among decided datasets, hc:sus50 has n=2, hc:time
n=7. None is a loss.

**The tier-1 Brier watch item resolved positively**: on real data gr:base is
20W-3L at +1.503% mean. The only negative anywhere is hc:time Brier at
−0.102% on 4 datasets — inside noise.

Fit cost 4.0x → 4.7x (+17%), a little above the probe's 11% and tier-1's 9%.

#### And the Pareto claim: a 5-member refit bag beats the 8-member plain bag

`ChimeraBoostEns5RM` vs `ChimeraBoostEns8` — **stronger and 20% cheaper**
(3.2x vs 4.0x):

| stratum | W-L-T | mean |
|---|---|---|
| gr:base | 44W-15L | +0.414% |
| gr:sus25 | **11W-1L** | +0.982% |
| gr:sus50 | 4W-2L | +0.021% |
| hc:base | 7W-5L-2T | +0.235% |
| hc:sus25 | **3W-0L** | +1.296% |
| hc:time | 3W-4L | +0.848% (median −0.092%) |

That is the frontier move: five refit members dominate eight plain ones on
both axes. **`Ens3RM` does not** — at 2.1x it is a genuine wash against Ens8
(35W-24L on base but 2W-10L on high-card, 4W-8L on sus25), so the honest claim
stops at five members. The blended-% column flatters Ens3RM (99.4 vs 98.9);
the sign test is the trustworthy reading and it says parity-at-half-cost, not
dominance.

### VERDICT: SHIP (opt-in)

All pre-registered gates pass. Shipping as `refit_members=True`, opt-in,
following the `refit_full` precedent where the default flip was Nathan's call.

Honest scope: this moves the **bagged** rungs (quality 4/5) only. The default
single-model configuration is byte-identical, so `images/pareto.png` and the
README headline are unchanged by construction and need no refresh. What moves
is the high-strength end of the frontier — and Ens8 is the blessed bagged mode.

Open follow-ups, owned rather than implied:
- **Default-flip decision for the bagged path** is Nathan's. The evidence for
  it is strong (positive mean in all 7 strata, +17% fit).
- **Multiclass** is gated off and stays a real gap: multiclass Brier is the one
  column we lose to CatBoost (91.1 vs 98.8, Finding 3). Closing it needs a
  replay path for vector-leaf trees, which does not exist.
- **The small-data deficit is only partly addressed.** `refit_members` helps
  the bagged rungs there (12-0 on sus25); the single-model small-data collapse
  against CatBoost (81% → 50%/33%) is untouched, and remains the largest known
  headroom in the project.

### C2 — the two-way depth race (6 vs 4), whose transfer question is now testable

`A2_PLAN.md` measured this and then shelved it: **+0.398% mean, 16W-4L-16T,
p=0.012, at 1.12–1.24x fit**, on the PMLB tune fold. It was left open with an
explicit caveat — the gain concentrates on datasets under 2,000 rows
(+1.09% to +7.41% there, collapsing to +0.1–0.6% above 4,900), the panel
skewed small, and "the transfer question [is] unresolved". Registered as
"Nathan's call whether that is worth the run."

**The instrument that settles it did not exist then.** The decision tier now
carries `@sus25`/`@sus50` — the same datasets cut to a quarter and a half of
their training rows. That is precisely the small-data regime where depth 4 is
claimed to win, measured on the suites we actually decide on. If depth 4 wins
on `@sus*` and is neutral on base, the mechanism transfers and the race is
worth building; if it is flat on `@sus*` too, the thread closes for good.

Cheap first step, **no library change**: the harness already exposes
`--chimera-depth 4`, and the run above provides a perfectly paired depth-6
baseline on identical splits and seeds. One decide run answers it.

#### Verdict: the mechanism TRANSFERS. The sign flips exactly at data size.

`results/20260801-020408.json` (depth 4) paired against `20260801-013306.json`
(depth 6), same seeds and splits, per stratum:

| stratum | n | depth4 W-L | mean | median | sign test |
|---|--:|---|---|---|---|
| gr:base | 57 | — | **−1.102%** | −0.239% | depth 6 |
| hc:base | 13 | — | −0.331% | +0.000% | depth 6 |
| **gr:sus25** | 12 | **9W-3L** | **+1.265%** | +0.138% | **PASS** |
| gr:sus50 | 6 | 3W-3L | +0.214% | +0.008% | fail |
| hc:sus25 | 3 | 2W-1L | +0.754% | +0.390% | pass (n=3) |
| hc:sus50 | 2 | 1W-0L-1T | +0.078% | — | fail |
| hc:time | 7 | 3W-4L | −0.542% | −0.044% | fail |

**Every small-data stratum has a positive mean; every full-size stratum has a
negative one.** Four independent strata agreeing in direction is the real
signal — the individual sign tests are underpowered at n=2-6.

Read honestly, though, the *magnitude* is modest and concentrated: gr:sus25's
median is only +0.138%, and its +1.265% mean is carried by one dataset,
`cpu_act@sus25` at **+16.41%** (RMSE 3.82 → 3.20). A uniform depth 4 is not
shippable (base loses 1.1%); the registered answer is a per-dataset race, worth
roughly the median.

⇒ A2's transfer question is **ANSWERED**: small data does want less capacity,
and the effect is real but small except where it is enormous. The enormous case
is the interesting one — see below.

**And the race itself is now closed, priced rather than run** (2026-08-01, by
C3 below). A per-dataset depth race and a per-dataset `min_child_weight` are
the same object: a capacity choice with a real direction and a per-dataset
optimum. C3 measured that family's *oracle* — perfect selection, which no
implementation can beat — at a median +0.12% per dataset and +0.29% at quarter
size, against A2's own finding that validation recovers only ~20% of an oracle
like this. So the achievable prize is roughly +0.02–0.06%, against a program
bar of a broad +0.25%. **Do not build the depth race.**

---

## FINDING 4 — the linear-leaf per-leaf guard is far too permissive on small data

`cpu_act@sus25` is both our worst small-data loss (+26.55% vs CatBoost, +41.17%
vs LightGBM) and the dataset depth 4 rescues by +16.41%. That says the failure
is **over-capacity**, and the capacity is not where it first appears.

`tree.py::_linear_leaf_fit` falls back to a constant leaf only when

```
if counts[l] < 2 * d or k == 0:      # d = 1 + k, k = numeric split features
```

so a leaf fits a **(1+k)-parameter ridge on as few as 2 rows per parameter**,
with `linear_lambda` fixed at 1.0 regardless of leaf occupancy. At depth 6, k
can be 6, so d=7 and the guard admits leaves of 14 rows.

The arithmetic lines up with every observation:

| | rows | leaves at d=6 | rows/leaf | rows/parameter |
|---|--:|--:|--:|--:|
| cpu_act (base) | 6,144 | 64 | ~96 | ~13.7 |
| **cpu_act@sus25** | **1,536** | **64** | **~24** | **~3.4** |
| cpu_act@sus25 at depth 4 | 1,536 | 16 | ~96 | ~13.7 |

Full-size cpu_act is only +0.41% behind CatBoost; at a quarter of the rows it
is +26.55% behind; and forcing depth 4 — which restores the same rows-per-leaf
as full size — recovers +16.41%. The gate that decides whether linear leaves
run at all (`LINEAR_LEAVES_MIN_SAMPLES=1000`) counts **total** rows and is
blind to depth, so it cannot see this.

If this is right it is a much better lever than the depth race: it is a guard
correction rather than an extra audition, so it costs **nothing** in fit time,
and it should help wherever `n / 2^depth` is small rather than on one dataset.

Test: `--decide --seeds 3 --models ChimeraBoost --chimera-no-linear-leaves`
(`results/20260801-020906.json`), paired against the depth-6 baseline.

### Verdict: REFUTED. Linear leaves are not the culprit — they are load-bearing.

Forcing constant leaves is worse in every stratum, small data included:

| stratum | noLL W-L | mean |
|---|---|---|
| gr:base | 15W-43L | −0.601% |
| hc:base | 3W-4L-7T | −1.420% |
| gr:sus25 | 4W-7L | −0.453% |
| gr:sus50 | 3W-2L-1T | −0.033% |
| hc:sus25 | 0W-1L-2T | −4.863% |
| hc:time | 1W-2L-4T | −5.369% |

And the diagnostic dataset settles it: on `cpu_act@sus25` forcing constant
leaves is **worse** (RMSE 3.8236 → 3.8639, −1.05%). Linear leaves are *helping*
there, so the per-leaf ridge is not what blows up. `hc:employee_salaries`
depends on them enormously (−16.6% base, −14.6% @sus25, −37.8% @time without).

⇒ **KILLED without shipping**: the per-leaf sample-guard theory, and any
"linear leaves overfit small data" variant. The parameterization I had staged
for it was reverted rather than left in the tree as unused surface.

**What this leaves standing is simpler and better evidenced:** the small-data
deficit is plain *tree* over-capacity, independent of leaf model — which is
exactly what the depth-4 experiment measured directly. cpu_act prefers depth 4
at **both** sizes (+3.27% at full size, +16.41% at sus25), so capacity
preference is a per-dataset property, not a size threshold. That is an argument
for measuring it per dataset (a race) rather than predicting it from n — and it
is also why the historical size-rule attempts failed.

### Deprioritized by cheap analysis: per-leaf capacity (semi-oblivious trees)

The panel's highest-novelty structural idea was to break the oblivious
constraint at the last level, on the theory that one shared split per level
underfits against LightGBM's ~30 conditions per tree. The supporting anecdote
was electricity, where we beat CatBoost by 5.8% yet lose to both leafwise
libraries.

Measuring it directly: **the best oblivious model still loses to a leafwise
one on 2 of 57 datasets** — electricity (+1.07%) and road-safety (+0.32%) —
and those two sets have **zero overlap** with the 11 where CatBoost beats us.
Since ChimeraBoost and CatBoost are both oblivious, that pattern is the only
clean signature of the architectural tax, and it is worth ~1.4% on two
datasets total.

So the oblivious constraint is not what is costing us; it is most of why we
beat CatBoost while fitting in half the time. The panel's most expensive build
(judged feasibility 3.0, the lowest of 15) would chase the smallest target.
Not pursued. Script: `scratchpad/oblivious_tax.py`.

### C3 — the regressor has no effective min-leaf constraint

For squared error the Hessian is exactly 1 per row, so `min_child_weight=1.0`
means "at least one sample" — the weakest possible veto, at every dataset
size. The classifier fades one in via `_auto_min_child_weight`; the regressor
pins 1.0 forever. The PMLB random-search study found `min_child_weight` is the
one knob that transfers. Whether small-data regression wants a real min-leaf
floor has never been tested, and `@sus*` regression is where it would show.

#### Probe design (pre-registered 2026-08-01, before any results)

`benchmarks/probe_reg_mcw.py`. The decide strata carry too few regression
sets at `@sus*` (6-7 in gr:sus25, 3 in gr:sus50) to answer this, so the probe
applies the sus mechanism to **every** decision-suite regression set:

- **Datasets**: all 42 regression sets we decide on — 19 `gr:reg_num`,
  17 `gr:reg_cat`, 6 `hc:` (wine-reviews, colleges, house_prices_nominal,
  black_friday, employee_salaries, Moneyball). No `pub:` (post-hoc only),
  no TabArena in any form.
- **Sizes**: train fraction ∈ {1.00, 0.50, 0.25} using the harness's own
  `_subsample_train` (random_state=0, **test set unchanged**) — exactly the
  `@sus` semantics, learning-curve reading.
- **No extra row cap**: `frac=1.0` is the harness's own size (the 50k `gr:` /
  100k `hc:` builder caps), so the top of every curve is the regime the
  decide gate actually runs. A 20k cap was in the first draft and was cut
  after pricing it: it would have shrunk the full-size arm on 20 of the 42
  sets (`hc:wine-reviews` 75,000 → 15,000 rows) while saving 16 minutes on a
  ~45-minute run — measured, not guessed, from the decide run's own recorded
  fit times (105.3 s for one seed across all 42 sets).
- **Arms**: `min_child_weight` ∈ {1 (shipped), 4, 8, 16, 32} — for squared
  error these ARE min rows per child. Everything else is the out-of-box
  default the harness measures (`n_estimators=2000`,
  `early_stopping_rounds=50`, `random_state=0`). Split 0.25 test at
  `random_state=seed`, seeds 0-2, all arms exactly paired on each split.
- **Primary arm is `mcw=8`**, named here before the run. Four arms × three
  sizes is twelve chances to find a majority in noise, so the other three are
  supporting evidence and every sign test carries a **Holm correction across
  the four arms**.
- **Reading**, all conventions matched to the house tools: seeds averaged on
  the metric before any ratio (never a mean of per-seed percentages);
  wins/losses/**ties** on `compare_runs`' ±1e-9 dead band, so an arm whose
  veto never binds reads as a tie rather than a loss; near-solved excluded on
  the **best** arm in the cell, matching `compare_runs.is_near_solved`;
  per-dataset rows printed before any aggregate. Rounds and fit seconds are
  printed as ratios — a real veto should make fits cheaper.

Pre-registered predictions:

- **C3 right**: `mcw=8` beats `mcw=1` at frac 0.25 on a Holm-corrected sign
  test, each dataset's **own** best mcw rises as its rows shrink (counted
  within datasets, so it cannot be an artefact of which datasets sit in which
  size bucket), and at full size large mcw is neutral-to-harmful (the
  oblivious-veto underfit that made the classifier auto fade to zero above 2k
  rows). Ship shape: a size-adaptive regressor auto analogous to
  `_auto_min_child_weight`, then the standard tier-1 synth + tier-2 decide
  gates.
- **C3 wrong**: flat-to-negative at every size ⇒ the thread closes for good:
  the regressor keeps `min_child_weight=1.0` and the small-data deficit is
  not a min-leaf story. **One caveat is pre-registered against that kill**:
  if rounds and fit seconds are *also* unmoved, the veto never bound at these
  values and the honest finding is "these arms did nothing", not "the
  mechanism is refuted" — that would call for larger arms, not a closed
  thread.

### Verdict: the effect is REAL and DIRECTIONAL, and far too small to ship. KILLED.

378 cells (42 datasets × 3 seeds × 3 sizes), `results/probe-reg-mcw.jsonl`.
Two things are true at once, and the program only cares about the second.

**The direction is confirmed, significantly.** Each dataset's own best `mcw`
rises as its rows shrink on 22 datasets, falls on 5, unchanged on 13 — a sign
test at **p=0.002**. Small-data regression really does want more min-leaf, and
the oracle grows the right way too (median best-arm gain +0.104% at full size
→ +0.290% at quarter size, and +0.577% on the 15 smallest cells).

**The primary pre-registered test fails outright.** `mcw=8` at frac 0.25 is
23W-17L, median **+0.085%**, Holm-adjusted **p=1.000**. No arm at any size
comes close — the best p anywhere is 0.615.

| frac | mcw=4 | mcw=8 (primary) | mcw=16 | mcw=32 |
|---|---|---|---|---|
| 1.00 | +0.013% (21W-19L) | −0.070% (16W-24L) | −0.266% (15W-25L) | −0.136% (15W-25L) |
| 0.50 | +0.012% (21W-19L) | −0.043% (19W-21L) | +0.038% (22W-18L) | −0.156% (20W-20L) |
| 0.25 | +0.046% (21W-19L) | **+0.085% (23W-17L)** | +0.047% (24W-16L) | +0.012% (20W-20L) |

The full-size half of the prediction also holds directionally and weakly: the
two large arms are the worst cells in the table at frac 1.00, which is the
oblivious-veto underfit the classifier auto was built to avoid.

**The pre-registered escape hatch does not apply.** The kill rule was
suspended if rounds and fit seconds were also unmoved, since that would mean
the veto never bound. It bound: rounds run **1.05–1.24×** and fit time
**1.01–1.10×** above the mcw=1 arm. The constraint is changing the trees and
forcing early stopping to run longer — it simply does not pay. So this is a
refutation, not a null from arms that were too small. It is not even a speed
win to trade against.

**And the ship shape is priced dead.** The pre-registered ship was a
size-adaptive auto. The best *fixed* arm captures only 13–31% of the oracle,
so a schedule cannot recover the per-dataset scatter — at frac 0.25 the
winning arm is spread across all five values. The alternative shape, a
per-dataset race, is capped by the oracle itself: median **+0.120%** per
dataset, and A2's config-portfolio measured that validation recovers only
**~20% of an oracle** like this. That lands around +0.02–0.06% against a
program bar of a broad +0.25%.

⇒ **KILLED without shipping**: `min_child_weight` as a small-data regression
lever, in both the fixed and size-adaptive forms. The regressor keeps 1.0.

**What this also settles, by pricing the shape rather than the knob**: C2's
depth race is the same object — a per-dataset capacity choice with a real
direction and a per-dataset optimum. Its own measured prize was a +0.138%
median on `gr:sus25`; this probe independently prices the whole
capacity-racing family at a +0.12–0.29% *oracle*, before the ~80% haircut
that selecting on validation costs. **Racing capacity per dataset is below
the bar as a family**, which is a stronger statement than either measurement
alone and is why C2 should not be built either. (An argument from a priced
ceiling, not a sign test on the depth race — but the ceiling is the binding
constraint, and it binds well under the bar.)

The one dataset that keeps confessing is `cpu_act@0.25`: **+7.77%** from
`mcw=8`, after **+16.41%** from depth 4. Same story as Finding 4 — its
capacity preference is enormous and idiosyncratic. Everything the program has
measured says that is a property of that dataset, not a rule we can infer
from n.

Probe: `benchmarks/probe_reg_mcw.py` (resumable, `--table-only` reprints).

---

## Rules for this program
- Every candidate gets a cheap decisive probe BEFORE library work where one
  exists. Two candidates have already been killed at zero implementation cost.
- Ship gate is unchanged: tier-1 synth screen, then
  `--decide --seeds 3 --save` with per-stratum sign tests against
  `results/20260731-164927.json` (library byte-identical to main).
- Negative results get written down here in the same change that produces them.
