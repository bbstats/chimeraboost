# Shared-tree multi-quantile head

## Campaign 2026-09-23 — the loop's focus (pre-registered)

**Decisions (the maintainer, in chat, 2026-09-23).** The campaign loop moves
from the point models to this head ("shift focus to the multi quantile
'quantiles' model now … We don't have much benching built for it"). On the
proposal: "Just add ngboost, not the rf though. I like crps. Ok yea go ahead
on it." So: NGBoost joins the field, quantile regression forests do not;
**CRPS is the decision score**; the plan below is approved. Verdicts are
logged in `CAMPAIGN_PLAN.md` like every other rung; this section holds the
program.

**Environment (2026-09-23).** NGBoost 0.5.11 installed into
`A:\code\miniconda3` with `--no-deps`, plus sympy 1.14.0 and mpmath 1.3.0:
its declared dependency `lifelines` would have DOWNGRADED pandas 3.0.3 →
2.3.3 (lifelines pins pandas < 3), and lifelines is only imported by
`ngboost.evaluation` (survival plots), never by `NGBRegressor`. numpy
2.4.6, scipy 1.17.1, scikit-learn 1.8.0 and pandas 3.0.3 verified unchanged;
`pip check` reports only the missing lifelines. NGBoost is a benchmark
opponent, never a library dependency.

### Phase 1 — the bench (Q-B1 … Q-B4)

- **Q-B1 (muse): a decision tier for quantiles in `quantile_suite.py`.**
  `--decide` = Grinsztajn regression (36) + high-cardinality regression (6)
  + their `@sus25` / `@sus50` twins + the `@time` twins of the hc
  regressions — the intervals' hardest case, since distribution shift is
  where coverage breaks. The harness's own builders, seeds (`1000 + seed`),
  75/25 split, `_subsample_train` and `_temporal_split`, so the data pairs
  with the point-model decide runs. `--jobs` process pool (one benchmark at
  a time still). New arms: **RigidShift** (one default squared-error fit +
  the empirical quantiles of its validation residuals added to every row —
  the P14 bar the head has never cleared), **ChimeraBoostQuantileCQR**
  (`conformalize=True`), **NGBoost** (`Normal`, the shared early-stopping
  split, quantiles from the fitted distribution). `compare_runs.py --metric
  crps --by-suite` gains a **calibration guard line** (median change in
  |coverage − nominal| at 80% and 90%).
- **Q-B2 (muse): a synthetic screen with known true quantiles**
  (`quantile_synth.py`): location-only, heteroscedastic, skewed,
  heavy-tailed and bimodal noise; n ∈ {1k, 10k}; early-stopped fits (the
  2026-08-30 lesson: fixed-round synthetic fits flatter us); scored as excess
  CRPS over the oracle, so a mechanism shows up without real-data noise.
- **Q-B3 (muse): the quantile Pareto** — CRPS skill (1 − CRPS / CRPS of the
  unconditional grid; 0 = no skill) against fit slowdown, coverage error
  alongside, the partial-coverage rule from #154.
- **Q-B4 (Claude): the baseline** — one `--decide --seeds 3` run of the full
  field on main; the standing quantile BASE, the first quantile chart, facts.

**Q-B1 done (2026-09-23, muse, one pass + review).** `--decide` selects 59
regression keys in 7 strata; the harness's data path (builders at
`1000 + seed`, so rows pair with the point-model decide runs; the
Grinsztajn base keys turned out identical to the 2026-08-30 JSON anyway,
see Q-B4); `--jobs`; arms RigidShift, ChimeraBoostQuantileCQR,
NGBoost; the guard line in `compare_runs --metric crps`. 1160 tests green
outside the sandbox path quirk. Smoke, 3 keys × 1 seed, all 7 arms ran:
fit seconds on cpu_act (6k rows) head 1.26 / per-level 6.18 / LightGBM 3.24
/ CatBoost MQ 25.9 / RigidShift 0.39 / CQR 0.82 / NGBoost 30.8 — CatBoost
and NGBoost are ~75% of the task time, so a full `--decide --seeds 3 --jobs
5` is 1–3 h. Review findings folded into Q-B2's task: NGBoost's
`random_state` does not reach its base learner (a rerun moved its CRPS
~0.4%), and the guard printed "ok" with no data to read. Two scratch
fixtures muse left in the repo root were deleted. The 3-key smoke read is
anecdote, not evidence: NGBoost best CRPS, the head second, CatBoost MQ
third; the head's 90% band covered 0.778.

**Q-B2 done (2026-09-23, muse).** `benchmarks/quantile_synth.py`: six
regimes with exact oracle quantiles (location, hetero, skewed, heavy,
bimodal by CDF inversion, catscale with a 20-level string column carrying
the spread), n ∈ {1k, 10k}, 3 seeds, every arm early-stopped through the
suite's own call, scored as excess CRPS over the oracle; 17 tests pin the
oracles (monotone, 200k-row band coverage within 0.005 of 0.90, bimodal CDF
inversion to 1e-8). Q-B1's two review fixes are in: NGBoost's base learner
seeded (three reruns bit-identical), the guard prints `n/a` with nothing to
read. Smoke (n = 1k, 1 seed — anecdote): the screen discriminates as
designed — RigidShift wins `location` (homoscedastic: one width fits) and
is worst where width must move with x; the head has the lowest excess CRPS
on skewed / heavy / bimodal and under-covers by ~13 points at 90% (median
13.4 against per-level 2.6, RigidShift 1.6, CQR 3.4, CatBoost 19.8) —
the Q1 defect, visible on a known oracle.

**Q-B3 done (2026-09-23, muse).** `make_pareto.py` detects a quantile run
and prints / draws CRPS skill vs slowdown with the 90% coverage error per
model, reusing the #154 frontier and partial-coverage rule; point charts
unchanged. On the 2026-08-30 JSON it reproduces the recorded coverage errors
exactly (head 0.040, per-level 0.011, LightGBM 0.079, CatBoost 0.075) and
puts the head on the frontier: CatBoost MQ 0.5832 skill @ 15.4×, the head
0.5782 @ 1.1×, LightGBM per-level 0.5629 @ 3.1×, our per-level 0.5493 @
4.5×. Branch: 1184 tests green, `ruff check chimeraboost/` clean.

**Code review of Q-B1–Q-B3 (2026-09-23), two findings, fixed by muse before
any baseline ran.** (1) NGBoost stops 50 rounds past its best validation
round and KEEPS those trees; `pred_dist` without `max_iter` used them all,
so NGBoost was scored with a handicap no other arm had. It now predicts
with `max_iter = best_val_loss_itr + 1` (verified against the installed
0.5.11 source: `pred_param` breaks at `i == max_iter`, truthiness-tested).
(2) `--datasets` without `--decide` registered only Grinsztajn, so `hc:` or
`@` keys skipped silently into an empty table; registration now follows the
requested keys and an unknown key exits before any task runs. 66
quantile / compare / chart tests green.

**Q-B4 forecast, written before the baseline runs.** Decide tier, CRPS per
stratum: on gr regression the head keeps August's picture — beats our
per-level models (≥ 30 of 36), ties LightGBM, LOSES to CatBoost MQ (≤ 12 of
36) — beats RigidShift on at least two thirds (the synth screen says the
rigid width only wins where noise is homoscedastic), and is roughly even
with NGBoost (a Normal fit is strong on smooth, near-Gaussian targets and
weak on skew). Coverage at 90%: the head ~0.87, CQR 0.89–0.92, CatBoost
~0.82. hc regression and the time twins: first reads, no forecast beyond
"every arm's coverage drops under the time shift". Synth screen at full
size: the head's excess CRPS lowest on skewed / heavy / bimodal, RigidShift
best on location, the head's 90% coverage error ≥ 8 points at n = 1k and
smaller at n = 10k.

**Q-B4a, the synthetic baseline (2026-09-23,
`results/quantile-synth-20260923-090818.json`, 6 regimes × {1k, 10k} × 3
seeds × 7 arms, ~5 min).** Median over the 12 keys:

| arm | excess CRPS ×1000 | 90% coverage error (points) | fit s |
|:--|--:|--:|--:|
| **head** | **42.9** | 7.3 | 0.37 |
| CatBoost MultiQuantile | 48.3 | 9.2 | 15.70 |
| our per-level (19 models) | 50.3 | **1.4** | 1.76 |
| head + CQR | 52.4 | 2.5 | 0.27 |
| LightGBM per-level | 56.0 | 8.2 | 1.86 |
| RigidShift | 68.2 | **1.4** | 0.19 |
| NGBoost (Normal) | 71.6 | 1.8 | 10.46 |

Read: the head has the lowest excess CRPS overall and on skewed / heavy at
both sizes; RigidShift wins only on `location` (7.9 vs the head's 11.7 at
10k: homoscedastic noise, one width fits); CatBoost edges the head on
`location` / `hetero` at 10k. The Q1 defect is plain: the head's coverage
error is 8.8–17.9 points at n = 1k and 1.8–5.8 at 10k, where per-level,
RigidShift, NGBoost and CQR sit at 0.4–6. **New pointer, `catscale` at 10k:
the head 42.4 against CatBoost 30.9, NGBoost 32.8, LightGBM 34.5** — when a
categorical carries the SPREAD, our ordered target statistic (a per-category
mean) carries nothing about it, while CatBoost's CTRs are computed on the
binarized target and so see the category's distribution. Candidate Q4:
encode each category's spread for the quantile head (a TS of |y − median|,
or TS columns on quantile-binarized targets). Forecast HIT on all three
synthetic counts.

**Q-B4, the real-data baseline (2026-09-23,
`results/quantile-20260923-115727.json`, `--decide --seeds 3 --jobs 5`: 59
keys × 3 seeds × 7 arms, 2 h 49 min, nothing skipped). The standing
quantile BASE.** First, reproducibility: the 36 Grinsztajn base keys came
back bit-identical to the 2026-08-30 run for the head, our per-level models
and CatBoost MQ (108 of 108 records each; LightGBM 94 of 108, most likely
its thread-count summation order, since this run split the cores five
ways). The head has not moved since August, and those keys' builders draw
nothing from the seeded stream.

Grinsztajn regression (36, the one gate-sized stratum). CRPS sign test per
dataset, seeds averaged; the guard's median |coverage − 0.90| in points
(the head's own: 3.43); mean 90% coverage; median fit time as a multiple of
the head's:

| head vs | CRPS W-L | median CRPS change | arm's 90% error | arm's 90% coverage | arm's fit |
|:--|--:|--:|--:|--:|--:|
| CatBoost MultiQuantile | 7-29 | −0.45% | 6.01 | 0.825 | 14.6× |
| RigidShift | 20-16 | +0.41% (CI −2.07..+1.86) | **0.26** | 0.898 | **0.27×** |
| head + CQR | 33-3 | +0.74% | 0.71 | 0.906 | 0.80× |
| LightGBM per-level | 23-13 | +0.43% (CI −0.05..+1.45) | 5.27 | 0.827 | 1.66× |
| our per-level | 32-4 | +1.65% | 0.34 | 0.908 | 2.72× |
| NGBoost (Normal) | 35-1 | +7.57% | 1.35 | 0.903 | 10.96× |

The head covers 0.869. Pointers (strata under 8 datasets): **hc
regression (6)** — the head against CatBoost 1-5, RigidShift 3-3, LightGBM
3-3, per-level 4-2, CQR 4-2, NGBoost 5-1; it covers 0.817 (RigidShift
0.896) and CatBoost fits 154× slower. **Time shift** (the 3 hc `@time` twins
against their own base keys): every arm's 90% coverage drops — the head
0.768 → 0.722, CatBoost 0.744 → 0.644, RigidShift 0.897 → 0.853, CQR 0.905
→ 0.884, NGBoost 0.894 → 0.878. **Small data:** the head covers 0.852 on
the `@sus50` twins (4) and 0.830 on `@sus25` (7), CatBoost 0.699 on
`@sus25`; the hc `@sus25` pair is the head's worst cell at 0.676.

Pareto over all 59 keys (`images/quantile_pareto.png`, CRPS skill @
slowdown, mean |cov90 − 0.90|): frontier **RigidShift 0.5869 @ 1.3× (0.008)
→ the head 0.5916 @ 3.5× (0.063) → CatBoost MQ 0.5982 @ 136× (0.108)**; off
it CQR 0.5846 @ 3.0× (0.012), LightGBM 0.5757 @ 12.6× (0.099), our
per-level 0.5690 @ 13.2× (0.012), NGBoost 0.5559 @ 45.3× (0.019).

Fairness check on the opponents: NGBoost stops on its own (median best
round 663; 15 of 177 fits reach the 2000-round cap, against the head's 26
and CatBoost's 35), so its loss is not a budget artifact.

Forecast: **7 of 9 HIT** — per-level ≥ 30 (32), LightGBM a tie, CatBoost
≤ 12 (7), coverage head 0.869 / CQR 0.906 / CatBoost 0.825 all inside
their forecasts, every arm's coverage drops under the time shift (7 of 7).
**MISS: RigidShift** (20 of 36 against a bar of 24: a tie, not a win) and
**NGBoost** ("roughly even"; the head wins 35-1). The synthetic screen's
regimes make width move with x far more than these targets do: there the
rigid width lost 68.2 to 42.9 excess CRPS, here it ties. The screen stays
a mechanism probe; it does not predict the real-data margin.

**What this settles.**
1. **The head does not clear the rigid bar in aggregate, and the tie hides
   two populations.** One point model plus one validation-residual width
   for every row ties it on CRPS (20-16) at a quarter of its fit time, and
   is calibrated where the head is not (0.26 points against 3.43 at 90%;
   0.896 against 0.817 on hc). But the head reaches the 2000-round cap on
   11 of the 36 datasets (Brazilian_houses and nyc-taxi in both forms, pol,
   SGEMM, superconduct, visualizing_soil, Bike_Sharing, diamonds, houses).
   There it loses CRPS 2-9 to RigidShift (median −5.03%) and 1-10 to
   CatBoost (−3.79%), and its median-level pinball loss runs 1.6× to 1.95×
   RigidShift's on Brazilian_houses, pol and visualizing_soil. On the other
   25 it beats RigidShift 18-7 (+1.10%) and trails CatBoost by a median
   0.26% (6-19). The head is under-fit where the signal needs many rounds,
   at the flat learning rate of 0.1 it keeps pinned
   (`adaptive_learning_rate=False`).
2. **The CRPS is lost at the centre.** Against RigidShift the head loses
   the median level's pinball loss (16-20, −0.45%) and wins the 80% and 90%
   interval scores (26-10, +4.35%; 28-8, +6.03%); against CatBoost it loses
   the median 5-31 and still wins the 90% interval score 25-11. CRPS
   weights the central levels most, so the tails' wins barely move it.
3. **Calibration is a small lever on CRPS.** CQR fixes coverage in every
   stratum (median error ≤ 2.2 points at 90% everywhere but the time twins,
   3.4) and loses CRPS 3-33 to the plain head. Its per-level scale about the
   median cannot move the median itself, and the median loses 2-34 (the
   head +0.67%): that is the 20% calibration fold's data tax, measured
   cleanly. If the tax costs the tails what it costs the median, the
   calibration itself bought about 0.15% on the 90% interval score.
4. **CatBoost MQ's CRPS edge is stable** (the same 7-29 as August, to the
   last digit) at 14.6× the head's fit on Grinsztajn and 154× on hc, with
   the worst calibration in the field; off the capped datasets it is a
   median 0.26%.
5. **NGBoost is no threat on CRPS** and is well calibrated; it stays in the
   field as the distributional reference.

Ledger re-reads on real data (the "Still open" item below): the fit-speed
row stays a MISS, narrowly on wide data — K = 19 early-stopped LightGBM
quantile boosters take 1.66× the head's fit on Grinsztajn and 9.46× on the
six hc sets, against the K/2 = 9.5× target. The CQR coverage row: median
error on Grinsztajn 1.38 / 0.88 / 0.71 points at 50 / 80 / 90%, erring
wide (mean +2.1 / +1.1 / +0.6), as the 2026-07-31 synthetic re-measure
found.

### The gate for Phase 2 changes (pre-registered)

Per stratum, never pooled: on Grinsztajn regression and on hc regression,
**CRPS wins ≥ half + 1 of the decided datasets AND median CRPS change > 0**
(seeds averaged per dataset; near-solved by the house convention). Guard:
the change must not worsen the median coverage error at 80% or at 90% by
more than **1 point** (absolute). Crossing stays exactly 0. Both Pareto axes
read in every verdict. The small-data and time twins are pointers (§2).

**Amended 2026-09-23 (I060), prospectively, before any further library
rung:** a DEFAULT change must also clear a two-sided sign test at
**p < 0.05** on Grinsztajn regression (at least 25 wins of 36 decided,
without ties). Written after Q2's three marginal passes (sign p 0.47 /
0.13 / 0.41), so it is post-hoc, and it only ever makes shipping harder.
Q5 clears it (31W-5L, p ≈ 2e-5). T0 probes keep the original form as a
screen; only a library default needs the significance.

### Phase 2 — the idea queue, ranked (re-ranked 2026-09-23 on Q-B4)

Q-B4 moved the queue: the CRPS is lost at the centre and on the capped
datasets, and calibration is a small lever on it. Q1 was first; the Q-B4 and
Q0 reads moved it down to item 4.

0. **Q0, the T0 probe battery (bench-only, no library change: one decide
   run and one synthetic screen).** Four arms, each answering one question,
   scored against the head and RigidShift:
   (a) **uncapped**, the head at `n_estimators=8000`: is the capped
   datasets' loss truncation?
   (b) **depth 6**, CatBoost's default: is it capacity?
   (c) **recentred**, the head's quantiles shifted so their median equals
   RigidShift's (the squared-error prediction plus the median validation
   residual): does a squared-error centre with the head's shape beat both
   parents?
   (d) **validation-rescaled**, the head's quantiles rescaled per level
   about its median on the shared early-stopping validation rows, so no
   fold is carved: what does calibration alone buy? (Q-B4 says little.)
   The question that pays picks the first library rung: (a) Q3, a rate or
   budget change; (b) a depth default; (c) a location design, the centre
   or the structure from squared error; (d) Q1.
   **DONE 2026-09-23 (I057, `results/quantile-20260923-135654.json`).**
   Against the head on Grinsztajn regression: (a) wins all 13 cap-bound
   sets, median +1.73%, and is bit-identical everywhere else (visualizing_soil
   +22.8%, pol +12.7%, superconduct +7.5%; 1.48× the total fit); (b) wins
   31-5, +0.35%, at 0.87× the fit, but its 90% coverage error worsens 1.06
   points (the limit is 1.00; hc 3.24): it wins the median level 30-6 and
   loses the 90% interval score 14-22, deep leaves overfitting the tails as
   the depth docstring warns; (c) 18-18, guard fails; (d) 21-15, +0.02%,
   and the 90% coverage error falls from 3.43 to 0.50 points (hc 9.24 to
   0.22) for free. (a) and (d) pay, (a) by more per unit of fit cost, so
   Q3 goes next. The synthetic screen disagreed on (d) (worse CRPS on 9 of
   12 keys, mostly skewed noise, where one factor per symmetric pair widens
   the side that was already right); on real data it is neutral.
1. **Q3, the head's learning rate.** Flat 0.1: `adaptive_learning_rate` is
   pinned False in `quantile_api.py` ("Measure before flipping"), never
   measured. Q0 made it first: the cap-bound sets need 3 to 4 times the
   rounds at 0.1 (pol and superconduct stop between 5000 and 7500), and the
   budget is not a library lever (the harness and the head both default to
   2000 rounds). The size fade would LOWER the rate below 15k rows, the
   wrong way for these sets, most of which hold 8k to 16k training rows.
   **T0 first, bench-only:** flat 0.15 and 0.2. In the same run, as its own
   pre-registered question, depth 6 with the validation-row calibration:
   does (d) rescue (b)'s guard?
   **CLOSED 2026-09-23 (I058, `results/quantile-20260923-150433.json`),
   registered as barrier B23.** Neither rate pays: 0.15 goes 17-19
   (−0.04%), 0.2 goes 11-25 (−0.09%), each 7-6 on the 13 cap-bound sets,
   and both lose worst where the head needs rounds most (pol −34% / −74%,
   early stopping firing at round 1548 / 640). Bigger steps turn the
   validation curve early at a worse point; the cap-bound sets need more
   rounds or more capacity per round.
2. **Q5, the next library rung: depth 6 with calibration on the
   early-stopping rows, as the head's default.** The combined probe paid
   in I058: 31-5, +0.33% [CI +0.14..+0.77], the 90% coverage error from
   3.43 to 0.47 points (hc 9.24 to 0.85), at 0.90× the fit. The calibration
   is the head's own CQR factors computed on the rows early stopping
   already holds out, instead of a carved 20% fold, so no rows are spent;
   it turns depth 6's 90% interval score from 14-22 into 22-14. P16's
   objection (that fold also chose the stopping round) meets the measured
   result: the calibrated band covers 0.903 on average over the 59 keys,
   erring wide, so the reuse shows no visible optimism. Design to settle
   at the rung's S0: how `conformalize` carries the new default (a third
   value that the default takes, `True` keeping the carved fold and its
   guarantee, `False` the raw grid), what happens when early stopping is
   off (no rows to calibrate on: the raw grid), and the docs that state
   today's under-coverage. The bench's Depth6ValScaled probe is the exact
   oracle: the new default must reproduce it bit-for-bit, so its gate
   result is already known.
   **PASSED 2026-09-23 (I059, `results/quantile-20260923-185507.json`), PR
   for the maintainer.** `depth=None` resolves to 6 and `conformalize`
   defaults to `"auto"` (True and False keep their meaning; the default
   never raises for calibration). The new default equals the probe on 177
   of 177 fits; against the old head, gr 31W-5L (+0.33%), 90% coverage
   error 3.43 → 0.47 points, hc 9.24 → 0.85, hc `@time` 17.49 → 0.68.
   Against the field on gr: CatBoost MQ 15W-21L (was 7W-29L), RigidShift
   23W-13L (was a tie), LightGBM per-level 26W-10L (was a tie). Every
   point-model pin of the identity snapshot is unchanged.
3. **Q2, CatBoost MultiQuantile's CRPS edge (7W-29L on 2026-08-30 and
   again on Q-B4; 15W-21L against the Q5 default).** Ablate the opponent,
   one knob at a time toward our settings (depth, `l2_leaf_reg`, border
   count, rate, leaf estimation); depth and budget are already measured on
   our side. The next rung once Q5 merges.
   **CLOSED 2026-09-23 (I060, `results/quantile-20260923-195957.json`)
   with a diagnosis, nothing shipped.** Probed from our side, each the Q5
   default with one of CatBoost's settings: 254 bins 18W-13L-5T, rate 0.03
   with 8000 rounds 23W-13L, depth 8 21W-15L (sign p 0.47 / 0.13 / 0.41,
   none clears the amended gate); 8000 rounds 7W-0L on the cap-bound sets
   (a pointer); exact split gain 9W-27L (killed). The edge sits on five
   low-noise sets and has two complementary carriers: resolution (254 bins:
   SGEMM +21%, Brazilian_houses +15%, nyc-taxi +7%, but pol −24%) and
   location (a squared-error centre, RigidShift: Brazilian_houses +31% to
   +33%, pol +35%). B20's June cpu_act reversal does not repeat for the
   head (+4.9% at 254 bins).
3a. **Q6 (added 2026-09-23 from Q2), a validation-chosen head.** Per fit,
   the early-stopping rows choose among the default head, the head at 254
   bins, and the head recentred on a squared-error median. The five
   low-noise sets hold +7% to +35% for whichever candidate suits each; the
   choice must not give it back on the other 31 (the recentred head alone
   lost 9-16 there in Q0). T0 first, bench-only; the cost is about 2.3×
   the fit, so a pass has to earn it on the Pareto too.
   **PASSED 2026-09-23 (I061, `results/quantile-20260923-210350.json`), a
   library candidate.** Against the Q5 default on Grinsztajn: 23W-7L-6T,
   median +1.57%, sign p = 0.005 (clears the amended gate), guard fine; the
   five 5W-0L (+21.1%), the other 31 18W-7L-6T (+0.78%, none worse than
   0.5%); 99% of the per-set best-candidate gain recovered (A2 got 20%,
   B12); picks R 43, B 33, H 32 of 108 fits; 2.35× the head's fit. On the
   59-key chart it sits at the top of the frontier, CRPS skill 0.6006 @
   7.2×, above CatBoost MQ's 0.5982 @ 129×. The recentred candidate alone
   blows up on tie-heavy targets (analcatdata_supreme, CRPS ×5.5e6: the
   head's band collapses and the CQR factor divides by it); validation
   rejected it every time.
3b. **Q7 (added 2026-09-23 from Q6), the audition in the library.** Build
   the three-candidate choice into `ChimeraBoostQuantileRegressor`, with
   the bench arm as its exact oracle. Requirements from Q6: R reachable
   only through the validation choice, a guard on R's calibration for a
   collapsed band, the default head when there are no early-stopping rows,
   and prediction, `staged_predict`, `predict_thresh` and SHAP defined for
   whichever candidate won. Open product question for the maintainer: on
   by default (the house rule makes the default the strongest
   non-ensembling setting, which this is) or opt-in, at 2.35× the fit.
   **PASSED 2026-09-24 (I062, `results/quantile-20260924-005306.json`);
   merged as PR #162 and released in 0.33.0 the same day; on by default (stated with PR #161, not objected
   to).** `audition=True` reproduces the bench arm on 177 of 177 fits;
   against the Q5 default, gr 23W-7L-6T, +1.57%, p = 0.005. The guard
   against R's collapsed band is the validation choice itself (a tie-heavy
   target picks H; a test pins it). Against the field on gr the head now
   beats CatBoost MQ 27W-9L and tops the 59-key chart at 0.6006 @ 7.4×,
   with CatBoost off the frontier. The one flag: hc `@time` (3 sets), 90%
   coverage error 0.68 → 2.12 points.
3c. **Q8 (added 2026-09-25, when the maintainer reopened the quantile
   thread), the head's low-noise losses.** After Q7 the head still lost
   pol, visualizing_soil and SGEMM to NGBoost and RigidShift. Three
   bench-only probes: S, the fixed-width grid (RigidShift's) as a fourth
   audition candidate; S+N, plus a scaled-residual candidate N (a second
   default regressor predicts the centre's absolute error; per-level
   factors from the standardized validation residuals); and the default at
   8000 rounds.
   **PASSED 2026-09-25 for S+N (I070, `results/quantile-20260925-141117.json`).**
   Against the Q7 default: gr 18W-1L-17T, median +2.18%, sign p 7.6e-5,
   90% coverage error 0.41 → 0.32; hc 5W-0L-1T; 1.11× the fit (pol +31%,
   visualizing_soil +17%, the one loss sulfur −0.31%). Picks over 177
   fits: N 80, B 40, H 38, S 13, R 6. S alone (8W-3L-25T, p 0.23) is
   mostly subsumed by N; 8000 rounds (6W-3L-27T, p 0.51, 1.27×) fails.
3d. **Q9, S and N in the library default.**
   **PASSED 2026-09-25 (I071, `results/quantile-20260925-172250.json`);
   merged as PR #177 (a6a90d2) the same day.** The default equals the bench arm on 177 of 177
   fits. Against the single head (Q5): gr 25W-5L-6T, +3.04%, at 2.6× its
   fit. Against the field on gr: CatBoost MQ 29W-7L, RigidShift 35W-1L,
   LightGBM per-level 31W-5L, our per-level 34W-2L, NGBoost 33W-3L; on the
   59-key chart 0.6046 @ 8.3×, CatBoost MQ 0.5982 @ 133×. The flag: hc
   `@time` 90% coverage error 2.12 → 4.69 points; the rise comes from
   Moneyball, where S wins and a fixed width misses the shift (the other
   two sets improve). Open: NGBoost still wins
   visualizing_soil (−34%) and SGEMM (−8%).
3e. **Q10, where NGBoost's remaining lead comes from (read-only).**
   **DIAGNOSED 2026-09-25 (I072).** On visualizing_soil and SGEMM the
   whole lead is the centre: our calibrated grid moved onto RoNGBa's mean
   beats RoNGBa itself, and RoNGBa's grid on our median is worse than ours.
   The centre gap is MAE-shaped (visualizing_soil: RoNGBa's mean has 3.2×
   lower MAE, only 8% lower RMSE). The S/N centre is the point regressor
   without its full refit. pol's small gap is width, not centre.
3f. **Q11, the S/N centre's accuracy (five-set screens, bench-only).**
   **CLOSED 2026-09-25 (I073, I073b), nothing shipped.** 254 bins for the
   centre and spread models match RoNGBa's lead (visualizing_soil +24%,
   SGEMM +7%, at 1.1× the centre's cost) but cost cpu_act 2.5%, and as
   extra candidates beside the 128-bin centre they still do: the finer
   grid's overfit is invisible to the early-stopping rows. 8000 rounds for
   the centre help visualizing_soil only (+5%); the centre hits its cap on
   1 of the 59 decide keys, so it cannot clear the gate. Bagging the centre
   ×5 is the ceiling (visualizing_soil +30%, pol +9%) and is an ensemble,
   so not a default.
3g. **Q12, retraining the winner on all rows (the maintainer's pick
   2026-09-25).** Without an `eval_set` the head's own carve equals the
   suite's shared split bit for bit, so the probe is a retrain added to
   today's fit: every choice and calibration quantity from the held-out
   fit, then (a) the R/S/N centre retrained on all rows, or (b) that plus
   the H/B/R head retrained from scratch at the replay-round rule.
   **PASSED 2026-09-25 (I075, `results/quantile-20260925-202430.json`).**
   Against the default: (a) gr 19W-0L-17T, +1.72%, p 3.8e-6, 1.07× fit;
   (b) gr 35W-1L, +0.91%, p 1.1e-9, hc 6W-0L, 1.28× fit; (b) beats (a)
   20W-1L-15T (p 2.1e-5), so (b). Coverage guard fine (gr 90% error 0.32 →
   0.47 points); the hc `@time` flag shrinks (4.69 → 3.29).
3h. **Q13, `refit_full=True` as the default.**
   **PASSED 2026-09-25 (I076), PR for the maintainer.** Without an
   `eval_set` the default reproduces Q12's arm (b) bit for bit; with one,
   nothing changes. The suite's field arm keeps its `eval_set`, so every
   comparison stays "every model on the same rows" and leaves this gain
   out.
4. **Q1, the narrow-interval defect (P16).** Leaf values are in-sample
   residual quantiles, so intervals over-narrow (0.869 at nominal 0.90 on
   2026-08-30; coverage decays with rounds). Fit leaf quantiles
   out-of-sample. Drafted as P16 in `LEAFTUNE_PLAN.md`, never pre-registered.
   Q-B4: calibration is a small lever on CRPS (point 3 above), so Q1 stays
   for coverage, which is what a user reads off an interval (the head is
   3.4 points short at 90% on Grinsztajn, 9.2 on hc, 17.5 under the time
   shift), and its CRPS bar is the gate as written.
   **Re-read 2026-09-25 (I074):** the coverage case is gone. Calibration
   (Q5) brought the default's median 90% coverage error on Grinsztajn to
   0.32 points. Q1's remaining claim is CRPS through the head candidates
   (H and B, 78 of 177 picks since Q9). It waits on the maintainer's pick
   against the no-`eval_set` refit question (Still open, below).
5. **Q4 (added 2026-09-23 from the synthetic baseline), spread-aware
   categorical encoding.** On `catscale` at 10k the head loses 42.4 to
   CatBoost's 30.9 excess CRPS: our ordered TS is a per-category MEAN, blind
   to a category that sets the spread. Probe first (monkeypatch an extra TS
   of |y − median| per categorical), then the real-data hc regressions.
   **CLOSED 2026-09-25 (I074), nothing to build.** The N candidate (Q9)
   already is the spread-aware encoding: its spread model's ordered TS of
   `|y − centre|` is the per-category spread. On `catscale` the default
   now scores 16.0 at 10k (CatBoost 30.9, NGBoost 32.8) and 79.9 at 1k
   (CatBoost 108.2), picking N on 6 of 6 fits; on the hc regressions
   employee_salaries moved from −5.7% to −1.3% against CatBoost MQ.
6. Later: the leaf refit's cost (~90% of a round). The RigidShift gap
   was Q0's subject: the capped sets explain it, and uncapped rounds and
   depth 6 move the count against RigidShift from 20-16 to 21-15 and
   22-14.

## Status, 2026-08-30

Three things landed around the head. None of them changes how it fits — the
identity snapshot is bit-identical on all 155 configurations.

1. **It can be explained.** `shap_values` with a channel per level, plus
   `kind="width"`, which attributes the width of an interval rather than its
   position. See `docs/quantiles.md`. The kernel is `tree._shap_forest_vec`,
   pinned bit-for-bit against the frozen scalar kernel at K=1.
2. **It can be scored properly.** `quantile_metrics` gained the Winkler
   interval score, PIT, a CRPS skill score and sharpness.
3. **It is finally benchmarked on real data**, by `benchmarks/quantile_suite.py`
   — Grinsztajn regression, against CatBoost `MultiQuantile`, K LightGBM
   quantile boosters and K of our own single-level models. This is the first
   time the head has been scored on anything but synthetic draws and three
   probe datasets, and the first time CatBoost's version of the same idea has
   been run at all. Results below.

**Measured while building the SHAP path, and worth recording:** on the default
19-level grid, roughly 40% of rows have at least one adjacent pair *out of
order in the raw scores* before delivery-time rearrangement; on a 3-level grid
it is about 0%. The delivered crossing rate is still exactly 0 — that is what
the sort is for — but the sort is doing real work on the default grid, not
acting as a no-op. This is why the SHAP path distinguishes raw from delivered
channels rather than pretending the two are interchangeable.

### First real-data result (2026-08-30)

`quantile-20260830-175359.json` — 36 Grinsztajn regression datasets, 3 seeds,
K=19, shared early-stopping split and budget for every arm. Sign tests per
dataset (seeds averaged), never the mean of CRPS, which is dominated by
whichever dataset has the largest target scale.

| head vs | CRPS | interval score (90%) |
|:--|:--|:--|
| our own K per-level models | **32W-4L**, p<0.0001, median +1.65% | **34W-2L**, p<0.0001, +3.46% |
| K LightGBM quantile boosters | 23W-13L, p=0.13, +0.43% — a tie | **31W-5L**, p<0.0001, +5.20% |
| CatBoost MultiQuantile | **7W-29L**, p=0.0003, −0.45% — a loss | **25W-11L**, p=0.029, +0.41% |

Other columns, all 36 datasets:

| arm | coverage at nominal 0.90 (mean abs error) | crossing | median fit vs head |
|:--|:--|:--|:--|
| head | 0.869 (0.040) | **0.0000, on 36/36** | 1.00x |
| our per-level | 0.908 (**0.011**) | 0.163, crosses on 36/36 | 3.41x |
| LightGBM per-level | 0.827 (0.079, worst 0.67) | 0.224, crosses on 36/36 | 1.53x |
| CatBoost MultiQuantile | 0.825 (0.075, worst 0.64) | 0.063, crosses on 36/36 | 8.33x |

**What this settles.**

1. **The shared structure earns its place.** Against our own K independent
   single-level models it wins 32 of 36 on CRPS and 34 of 36 on interval
   score, at a third of the fit time. That was never measured before.
2. **CatBoost's MultiQuantile is genuinely sharper on CRPS** — a significant
   loss for us, and the first time the two have been compared. It costs them a
   median 8.3x our fit time, and their intervals are badly calibrated (worst
   coverage error 0.64 against nominal 0.90), so on the interval score, which
   charges width and miscoverage together, we still win. Recorded as a real
   deficit on the sharpness axis, not explained away.
3. **The LightGBM claim in the docs was synthetic-only and did not survive.**
   `quantile_head.py` measured pinball 1-3% better and 3.0-6.2x faster. On real
   data with both arms early-stopping it is a tie on CRPS (p=0.13) and a median
   1.53x on speed. `docs/quantiles.md` has been corrected. The old numbers were
   not wrong for what they measured; they were measured on fixed-round
   synthetic fits, which flatter us.
4. **Non-crossing is the one uncontested win.** Exactly zero on all 36
   datasets. Every other arm, CatBoost included, crosses on every dataset.

**Next question this raises:** the CRPS gap to CatBoost is a sharpness
deficit, which is the same axis P14 left open against the rigid offset.
Picked up 2026-09-23: the CatBoost gap is Q2, and Q-B4 measured the rigid
offset on real data (a CRPS tie), which made it Q0's subject.

### Still open

- **`adaptive_learning_rate` is pinned False for this head**
  (`quantile_api.py`, "Measure before flipping") and has never been measured
  against pinball. That is a default flip on a strength surface, so it needs
  its own pre-registration and the full `/experiment` protocol. Not attempted
  here. Recorded 2026-08-30.
- RESOLVED 2026-09-25 (Q12 and Q13, I075 and I076): **The head never trains on its own early-stopping fold** (recorded
  2026-09-25, I073b). Without an `eval_set` the head carves
  `validation_fraction` and neither it nor its S/N centre ever sees those
  rows, while `ChimeraBoostRegressor` refits on all rows by default
  (`refit_full`), worth 8.8% RMSE on visualizing_soil and 6.2% on pol. A
  refit of the winner after the audition, keeping the calibration taken
  before it, is the candidate. `quantile_suite.py` cannot measure it: it
  passes the shared split as an `eval_set`, which the head must not train
  on. Needs a no-`eval_set` protocol first. **OPENED 2026-09-25 as Q12**
  (the maintainer picked it over Q1): the head's own carve equals the
  suite's shared split bit for bit, so the probe is a retrain added to
  today's arm (`CAMPAIGN_PLAN.md` I075).
- RESOLVED 2026-09-23 (Q5, I059): `docs/quantiles.md` "How it compares" is
  re-measured against the new default, with the fixed-width baseline and
  NGBoost added.
- RESOLVED 2026-09-23 (Q-B4): the acceptance-ledger rows below (fit-speed
  MISS, CQR coverage FAIL), measured on synthetic data, re-read on real
  data. Fit speed is still a MISS (1.66× on Grinsztajn, 9.46× on hc,
  against 9.5×); CQR's coverage errs wide by a median of at most 1.4 points
  at any level on Grinsztajn, inside the 2-point target (mean +2.1 at 50%).

One booster, an arbitrary tau grid, a K-vector in every leaf. Replaces "fit K
independent quantile boosters" for estimating a whole predictive
distribution.

Run: `python benchmarks/quantile_head.py` (alone). Writes
`benchmarks/results/quantile-head.md`.

## What the split search actually is

Pinball loss has hessian 1 in every channel, so for a fixed candidate split
the exact summed-across-tau gain collapses to a norm:

```
gain = ‖G_L‖²/(n_L+l2) + ‖G_R‖²/(n_R+l2) - ‖G_P‖²/(n_P+l2)
```

By Parseval this equals the sum of projected gains over **any** orthonormal
basis. So projecting the K gradient columns onto one direction is the rank-1
truncation of the exact gain, not a different algorithm, and the only question
is which direction to keep. `exact_splits=True` computes the norm form
directly, at K histogram channels per feature; it is the reference arm.

A second structural fact makes the whole thing cheap. Row i's gradient is
`e(r_i) - taus`, where `r_i` is the row's PIT rank (how many of its own K
estimates the target falls above) and `e(r)` is the step vector that turns on
at r. A row's entire gradient is therefore one integer, so projecting onto `c`
is just scoring that rank by `phi(r) = sum(c[r:])`. The fit never builds an
(n, K) gradient at all: one binary search per row into that row's own sorted
score vector, then a table lookup (`tree._project_pinball`).

It also says what the candidate directions ARE. Taking `c` polynomial in tau
of degree 0, 1, 2 makes phi degree 1, 2, 3 in the rank: location, spread,
skew.

## Decisions, each measured

**Which direction (`split_projection`).** Excess pinball over the oracle,
x1000, 3 seeds, four regimes (see the results file for the current run):

| arm | location | scale | mixed | extreme | TOTAL |
|:--|--:|--:|--:|--:|--:|
| rotate (default) | 13.161 | 21.082 | 30.541 | 55.735 | **120.520** |
| sum | 13.056 | 17.222 | 43.150 | 72.966 | 146.394 |
| gram | 12.715 | 17.573 | 43.264 | 72.996 | 146.548 |
| exact | 15.651 | 20.705 | 28.039 | 48.404 | 112.799 |

- **The literal channel sum is dead.** On a symmetric grid the tau and 1-tau
  pushes are equal and opposite, so a region that is centred correctly but too
  narrow sums to exactly zero. It is competitive only where nothing but the
  centre moves.
- **`rotate` lands within 7% of the exact gain** at one histogram per round
  instead of K.
- **Measuring the direction instead of cycling ("gram") did not pay.** Two
  failure modes, both instructive. Raw gradient energy is dominated by
  per-row Bernoulli noise, which is largest at the median, so the top
  eigenvector collapses onto the location contrast and inherits its blindness
  to spread. Whitening by the no-signal (Brownian-bridge) covariance fixes
  that but is ill-conditioned -- the null's eigenvalues fall like 1/k², so its
  inverse amplifies exactly the high-frequency directions carrying only
  sampling noise, and unrestricted whitening measured four times worse than
  cycling on location-driven data. Restricting the ratio to the Legendre
  subspace makes it well-posed and it still only ties. Kept as an option, not
  the default.

**Cycle weighting.** Two rounds of location per round of spread beat uniform
cycling at both K=3 and K=19, and every schedule that also spent rounds on
skew was worse. `_ROTATE_PATTERN = (0, 0, 1)`.

## Why predictions cannot cross

**Superseded 2026-07-31 — see `LEAFTUNE_PLAN.md` P12 and P14.** The mechanism
described in this section shipped in 0.27.0 and has since been removed; the
history is kept because two of its dead ends are still live traps.

Today: every delivered row is sorted on the way out of the booster, so
`diff(pred, axis=1) >= 0` holds exactly, at every `staged_predict` stage.
Measured crossing rate **0.0000** everywhere, against 0.18-0.21 for K
independent LightGBM boosters. Rearrangement is free in accuracy terms
(Chernozhukov, Fernández-Val & Galichon 2010), and being per-row it is exact
where any training-time construction has to be a bound over all rows at once.

**Trap 1, still live.** "Commit only non-decreasing increments" is sound but far
too strong: a sum of non-decreasing vectors is non-decreasing, so the predicted
interval could never be NARROWER than the pooled one. Forcing it by sorting the
leaf vector is worse still -- sorted values re-enter the accumulated scores and
self-reinforce. Measured: pinball 2.9e7 against an oracle of 0.35, i.e.
divergence. **This is why sorting is safe only at delivery**, where nothing
feeds back into the fit.

**Trap 2, the one that shipped.** The fix for trap 1 was a **narrowing budget**:
`gap[k]` a lower bound on `Q_k(x) - Q_{k-1}(x)` valid for any input row, spent
down each round by the realized minimum over all leaves, with the admissible set
`{v : v_k - v_{k-1} >= -b_k}` projected by PAVA in shifted coordinates. Sound as
a bound and ruinous in practice: charging at the worst-case leaf let one leaf
spend on behalf of every row, so it saturated in tens of rounds and froze the
interval width, leaving bands 2x to 10x too wide. Passing this section's own
acceptance ledger while doing so is the cautionary part -- the ledger compared
against per-level LightGBM and against nominal coverage, and never against a
trivial fixed-width baseline, which would have caught it immediately.

## Conformalization

`conformalize=True` carves a calibration fold **before** the early-stopping
split, so it sees no training, no stopping decision and no model selection.

The correction is a per-level **scale about the predicted median**, not the
usual additive widening. The scores it is applied to are already rearranged, so
every deviation from the median is sign-correct and a non-negative factor cannot
reorder anything; an additive correction that shrinks an interval is a
non-monotone offset vector and has no such property. Scaling also moves in both
directions, which matters more now than it did: the head used to be
systematically over-dispersed (raw coverage 0.844 at nominal 0.70) and the
factors shrank, whereas since the budget was removed the raw grid runs slightly
narrow and they widen.

Fails loudly when the fold cannot support the requested levels
(`ceil((n+1)(1-alpha)) <= n` needs `n >= (1-alpha)/alpha`).

## Acceptance ledger

Target from the build request, against K = 19 LightGBM 4.6.0 quantile
boosters sharing one Dataset, 300 rounds, depth 4, lr 0.1, n = 20 000,
3 seeds.

| criterion | target | measured | verdict |
|:--|:--|:--|:--|
| pinball vs K LightGBM boosters | match within noise | ratio 1.029 (5 features) to 0.996 (128) | **PASS** |
| fit wall clock | >= K/2 = 9.5x faster | 3.4x (5 features) rising to 7.8x (128) | **MISS** |
| crossing rate, no predict-time patch | exactly 0 | 0.0000 at every width; LightGBM 0.18-0.21 | **PASS** |
| CQR coverage at n=10k | within 2 points of nominal | worst 0.73 points | **PASS** |

Re-measured 2026-07-31 after the budget was removed (same protocol, 3 seeds):

| criterion | measured | verdict |
|:--|:--|:--|
| pinball vs K LightGBM boosters | ratio 0.989 (5 features) to 0.969 (128) — now a **win** at every width | **PASS** |
| fit wall clock | 3.0x (5 features) rising to 6.2x (128) | **MISS**, unchanged in kind |
| crossing rate | 0.0000 at every width | **PASS** |
| CQR coverage at n=10k | worst 2.65 points, erring wide | **FAIL** |

The crossing row's original wording ("no predict-time patch") no longer
describes the implementation and is kept only as the historical target. What the
guarantee is worth to a user is unchanged: zero crossings, at every stage.

The CQR row is a real regression against its 2-point target and is recorded as
such. Cause and follow-up in `LEAFTUNE_PLAN.md` P14 — the raw grid now runs
narrow, so the factors widen, and the outer-at-least-inner monotonization pushes
the middle intervals furthest.

**On the speed miss.** The saving is concentrated in the split search, which
runs once per round instead of K times, so the speedup grows with how wide the
data is: 3.4x at 5 features, 4.8x at 32, 6.0x at 64, 7.8x at 128, and still
climbing. It does not reach 9.5x in the tested range because the leaf refit is
irreducibly K quantile selections per leaf and is ~90% of a round at ordinary
widths, while a single LightGBM quantile round is very cheap. Three
optimizations already landed and are in the numbers above: the PIT-rank
projection (no (n, K) gradient, -35% fit), a K-aware serial/parallel dispatch
threshold (the leaf refit does K times the scalar path's per-row work, so the
fork/join break-even arrives at K times fewer rows, +65%), and parallelizing
the refit over (leaf, channel) pairs rather than leaves, since real oblivious
trees are badly unbalanced.

Tried and reverted: gathering leaf residuals channel-major in one pass. It
trades K forward strided reads for a transposing write and measured slower --
the per-channel scan is prefetcher-friendly and the leaf's slice of the score
matrix stays resident across the K passes.
