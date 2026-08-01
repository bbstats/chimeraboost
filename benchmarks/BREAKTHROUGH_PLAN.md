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

**Running: the decision tier with the external field on every stratum**
(`--decide --seeds 3 --models ChimeraBoost CatBoost LightGBM sklearn_HGB
--save`) — 103 datasets in 7 strata. This is the measurement that decides where
to aim.

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

## Rules for this program
- Every candidate gets a cheap decisive probe BEFORE library work where one
  exists. Two candidates have already been killed at zero implementation cost.
- Ship gate is unchanged: tier-1 synth screen, then
  `--decide --seeds 3 --save` with per-stratum sign tests against
  `results/20260731-164927.json` (library byte-identical to main).
- Negative results get written down here in the same change that produces them.
