# REPLAY — structure-transfer full-data refit (pre-registered 2026-07-27)

## Mechanism

`refit_full=True` has been the default since 2026-07-25: after early stopping
picks the tree count, the winner is retrained from scratch on 100% of the rows
so the shipped model finally sees the validation split. It works
(REFIT_PLAN.md: Grinsztajn 48W-11L +2.0%) but it costs a second, longer fit.

Fresh attribution on today's default (`benchmarks/attr_legs.py` over
`profile_fit.py --attribution`, 4 Grinsztajn sets, 2026-07-27) puts that refit
at **37-49% of every default fit** — the single largest leg, now that it has
displaced the auditions:

| dataset | auditions | ES winner | refit_full |
|---|--:|--:|--:|
| cpu_act | 32% | 31% | 37% |
| diamonds | 14% | 41% | 45% |
| MagicTelescope | 11% | 43% | 47% |
| road-safety | 8% | 43% | 49% |

The same profile puts tree GROWTH at 83-85% of a fit. So the refit spends ~85%
of its time rediscovering split structures the winner already found. The claim:
it does not have to. Replay the winner's splits round by round against
gradients computed on all rows, and refit only the leaf values (and the
linear-leaf coefficients). The validation rows still reach the leaf estimates —
which is the mechanism REFIT_PLAN credits for the win — without re-paying for
the split search.

Implemented as `refit_full="replay"` (`tree.replay_oblivious_tree`,
`booster.replay_donor`, `sklearn_api._refit_on_full(replay=True)`). The donor's
preprocessor is reused verbatim: `splits_thr` are bin INDICES, so a re-fitted
binner would silently move every threshold underneath the structures.

Scalar booster only. The multiclass round grows one vector-leaf tree through a
separate loop, so multiclass keeps the from-scratch refit and is expected to be
inert in the A/B — a built-in null slice.

## What replay gives up

Split choice still comes from the 80/85% train subset; only the leaf values see
all rows. So replay should capture most, not all, of the full refit's gain.
Two-sided: structures chosen on a subsample and valued on everything is also a
mild regularizer, so replay beating full on some sets is expected, not a bug.

## Phase 0 probe (`benchmarks/probe_replay_refit.py`, 10 sets x 3 seeds,
shipped defaults, nothing pinned) — RAN 2026-07-27

| | mean gain vs no-refit | fit cost vs no-refit |
|---|--:|--:|
| `refit_full=True` | +3.685% | 1.89x |
| `refit_full="replay"` | +3.412% | 1.37x |

Median capture 94% over the 10 sets; replay >= full on 3/10 (pol 108%,
electricity 101%, heloc 209%). **Replay saves 27% of today's default fit
time.** Weakest capture: elevators 60%, covertype 84% — both large-n numeric,
where split re-selection on the extra rows plausibly does carry real signal.

Cost floor measured (`benchmarks/replay_attr.py`): a replay round costs ~1/3 of
a grow round, not ~1/10, because the grow kernel produces the leaf assignment as
a by-product of its fused quantized level pass while replay pays for assignment
separately. Parallel twins of `_assign_leaves`/`_leaf_values` were tried and
made no measurable difference (1.37x vs 1.41x, noise) — REVERTED, not kept.
Remaining headroom is a fused assign+accumulate kernel; not pursued here.

## Pre-registered predictions

- **Tier 1 (synth screen):** near-flat overall, NOT a broad win — replay is a
  cheaper way to buy the same thing, not a new source of accuracy. Expect a
  small negative mean (giving back part of refit's gain) that is not
  decisive by the sign test. Multiclass slice expected exactly inert
  (falls back to the full refit). Kill if: a large systematic loss (mean worse
  than −1.0%), or losses concentrated in the small-n / steep-learning-curve
  slice that refit_full was shipped to fix — that would mean replay misses the
  mechanism rather than approximating it.
- **Tier 2 (Grinsztajn, HC):** small negative mean on the primary metrics,
  sign test NOT decisive against; fit time down 20-30% on the ChimeraBoost arm.
  The trade, not the sign test, is the verdict.
- **The decision number is the Pareto position, not the sign test.** This
  change deliberately trades a little strength for a lot of speed, so the
  usual "decisive sign test + mean improvement" ship rule does not apply as
  written. What must hold: the replay point beats the CHORD between rung 2
  (no refit) and rung 3 (full refit) at its own cost — i.e. it is a genuine
  frontier point, not an interior one.

## Arms

- BASE: this worktree, defaults, no flag (`refit_full=True`). Library is
  default-path bit-identical to main — 636 tests green including the
  numerical-identity goldens.
- VARIANT: `--chimera-refit-replay` => `refit_full="replay"`.
- Both arms run from the SAME worktree in the same session, so the
  baseline-fingerprint trap cannot bite. `--models ChimeraBoost` on the A/B
  runs: the sign test only ever reads our own arm, and dropping CatBoost cuts
  HC runtime by most of its wall clock. The full field is re-run once, later,
  for the Pareto refresh.

## Ship question reserved for Nathan

Replay is CHEAPER and slightly WEAKER than the current default. His standing
rule is "the default is the best non-ensembling rung, always"
(2026-07-24, restated 2026-07-25), which as written keeps `refit_full=True` as
the default and makes replay a new rung on the `quality` ladder — it lands
between today's rung 2 (no refit, 5.36x) and rung 3 (full refit, 9.35x) but
much closer to rung 3's strength. Presenting the measured trade and letting him
choose; NOT flipping the default unilaterally.

## Results log

- 2026-07-27: pre-registered. Phase 0 probe PASS (94% capture, −27% fit).
### THE TARGET-LEAKAGE BUG (found 2026-07-27, after the first round of runs)

The first implementation built the refit's training matrix with the donor's
`prep_.transform(X)` — the INFERENCE-time encoding. For a training row that
applies target statistics in which a category's mean includes that row's own
label. Ordered TS exists precisely to prevent this, and the replay path went
around it.

`tests/test_chimeraboost.py::test_ordered_ts_resists_leakage` caught it the
moment the default flipped (2500 levels over 5000 rows, category unrelated to
the target):

| `refit_full` | train AUC | test AUC | gap | importance on the noise cat |
|---|--:|--:|--:|--:|
| `False` | 0.9265 | 0.8919 | +0.035 | 2.1% |
| `True` | 0.9217 | 0.8948 | +0.027 | 2.2% |
| `"replay"` **(buggy)** | 0.9709 | **0.8355** | +0.135 | **54.0%** |
| `"replay"` **(fixed)** | 0.9204 | 0.8941 | +0.026 | 3.6% |

Fix: `FeaturePreprocessor.fit_transform` gained a `binner=` argument. Replay
now refits every data-dependent statistic on its own rows exactly as a
from-scratch refit would — categories, gdiff group means, ordered TS — and
adopts ONLY the donor's bin borders, which is the one thing that must not move
(splits are bin indices). `cat_combinations` is pinned to what the donor
actually built rather than re-resolved at the new row count, because it decides
how many TS columns exist and the transferred splits address columns by
position.

Second, quieter bug found at the same time: replayed trees were given ZERO
split gains, so `feature_importances_` summed only the trailing grown trees.
The donor's gains now carry through — the structure is the donor's, so its
gains are the honest attribution.

**All cat-bearing results below are the post-fix re-runs.** Grinsztajn is
unaffected and was re-run anyway to confirm it: **0 of its 24 `_cat` datasets
actually pass `cat_features`**, so no target encoder ever runs there, and the
re-run reproduced the pre-fix numbers exactly.

### Runs

- 2026-07-27 **Tier 1 (synth screen) PASS**: base `replay-synth-base` vs
  variant `replay-synth-var2`, 136 datasets x 3 seeds, `--models ChimeraBoost`.
  **50W-51L-35T, mean +0.074%** (p=1.000) — dead flat, which is exactly the
  pre-registered prediction. Fit 0.886x summed, median 0.826x, faster 109/136.
  Mechanism confirmed: **multiclass exactly 0-0-34 / +0.000%** (the predicted
  null slice — the fallback is wired correctly), **canaries exactly flat**
  (0-0-3), and the one real concession shows where predicted — replay gives up
  split RE-SELECTION, so the deep-interaction slice is negative (depth>=3:
  20-32-25, −0.090%) while shallow is positive (depth<=2: 30-19-10, +0.290%).
  **The buggy run's headline "wins" were the leak**: card>16 read +2.185%
  (p=0.003) and entity cats +1.486% (p=0.001); post-fix they are −0.424% and
  −0.014%. A screen that looks too good on exactly the slice a bug would
  inflate deserves the second look it got.
- 2026-07-27 **Tier 2 Grinsztajn: ACCURACY FLAT, BIG SPEED WIN**. Base
  `replay-gr-base` vs `replay-gr-var2`, 59 datasets x 3 seeds.
  **27W-32L-0T, mean +0.005%, median −0.005%** (near-solved
  SGEMM/visualizing_soil excluded from the mean, as always). Sign-test bar
  FAIL — pre-registered as non-decisive; a 27-32 split at a mean of five
  thousandths of a percent IS parity, which is the result this change wants.
  **Fit time 327.6s → 213.6s = 0.652x (−34.8%), faster on 58/59 datasets**,
  median per-dataset 0.704x, p90 0.867x — the speed win has no tail.
  Bit-for-bit the same verdict as the pre-fix run, as predicted.
- 2026-07-27 **Tier 2 HC: parity, smaller speed win**. Base `replay-hc-base`
  vs `replay-hc-var2`, 14 datasets x 3 seeds. **3W-6L-5T, mean −0.017%,
  median +0.000%** (cjs near-solved, excluded). Fit 0.848x (−15.2%), faster
  12/14. Five ties are the sets that never reach a scalar refit (multiclass
  eucalyptus / okcupid-stem among them), which is also why HC gains less than
  Grinsztajn: replay cannot touch multiclass, and cat preprocessing is a fixed
  cost. **The fix collapsed the variance**: colleges −4.51% → −0.08%,
  wine-reviews −3.42% → −0.01%, house_prices_nominal +3.51% → +0.27%. Those
  large symmetric swings WERE the leak.
- 2026-07-27: 651 tests green (15 new in `tests/test_refit_replay.py`,
  including a dedicated leakage regression test, the multiclass no-op,
  donor-structure reuse, full-data init, and that the shipped model does not
  pickle the donor forest). The numerical-identity goldens are untouched.

## Verdict

**Accuracy is a wash on both decision suites — +0.005% on Grinsztajn and
−0.017% on high-card, both inside noise — while fit time falls 34.8% on
Grinsztajn (58/59 datasets) and 15.2% on high-card.** The pre-registered chord
test is passed trivially: replay sits at rung 3's strength for roughly
two-thirds of rung 3's cost.

**SHIPPED AS THE DEFAULT (Nathan, 2026-07-27):**

> "the speed gains is CERTAINLY worth defaulting with, @ quality = 3 and below
> (or whatever our default is). but - maybe we leave it off of 4/5?"

`refit_full` now defaults to `"replay"`; `True` remains available for the
from-scratch refit. **Rungs 4/5 needed no change and got none** — they pin only
`n_ensembles`, and `refit_full` is already a deliberate no-op inside bagged
members (their OOB rows are an eval set), so replay cannot reach them. Rungs
1/2 already set `refit_full=False`. The only rung where it was ever live is
rung 3, which IS the default, so the flip is one value in two constructors plus
the rung-3 recipe.

This satisfies the standing "default = best non-ensembling rung" rule rather
than bending it: at +0.005% / −0.017% the two settings are tied on strength,
and the tie-break goes to the one that is a third cheaper.

Defaults moved, so the Pareto chart is refreshed.

## Post-hoc validation (2026-07-27, after #41/#42 merged)

None of this can change the ship — it is confirmation on suites that took no
part in the decision, plus two questions the decision suites cannot answer.

**Public validation suite (22 datasets, n >= 50K, 3 seeds).** Never sealed,
never blocking, zero overlap with the decision suites, and heavy on real
high-cardinality categoricals — i.e. the regime where the leakage bug lived.
`pub-scratch` vs `pub-replay`: **6W-6L-10T, mean −0.114%, median +0.000%.**
All NINE multiclass datasets tied EXACTLY, which re-confirms the multiclass
no-op on real large data rather than on synthetics. That is also why the
whole-suite speed number looks modest (0.896x): those nine untouched datasets
are 71% of the suite's compute. **On the 13 datasets replay actually touches:
0.626x (−37.4%), faster on 13/13.** Mean is carried by two sets (rossmann
−1.63%, BNP_Paribas −0.93%) against six wins.

**PMLB (25 datasets, 3 seeds).** The suite most likely to find a weakness:
small-data heavy, and REFIT_PLAN's gains concentrated in steep learning
curves. It did not. **9W-5L-11T, mean −0.043%, median +0.000%** — more wins
than losses. The one alarming cell, `pm:holdout/dis` at −3.00%, is seed noise:
per-seed deltas −0.088 / +0.025 / −0.012 on a tiny, severely imbalanced
binary set. Speed 0.962x summed, median 0.876x — small data saves least.

**Backward compatibility.** Models fitted and pickled by the PRE-#41 library
(566573a) load under the new code and predict **bit-identically**, for
regression, binary and multiclass, despite their boosters predating the
`replay_donor` attribute entirely (predict never reads it). An unpickled
estimator still carries `refit_full=True`, so re-fitting an old model keeps
the old behaviour rather than silently switching.

**Size scaling (`benchmarks/replay_scaling.py`).** Every suite caps dataset
size, so the headline was really "−35% at up to ~50K rows". Measured directly
at 50K / 200K / 500K on mixed numeric+categorical data:

| task | 50K | 200K | 500K |
|---|--:|--:|--:|
| regression | 27.5% | 28.9% | 28.4% |
| binary | 27.0% | 29.0% | 27.2% |

**The saving is FLAT at 27-29% across a 10x row range.** Combined with PMLB's
~12% on few-thousand-row sets, the shape is: the saving climbs out of small
data and then plateaus near 30%. It does NOT keep growing — do not promise
more at giant n. (`scaling_giant.py` cannot measure this: it fits a fixed tree
count with early stopping OFF, the one path on which no refit happens at all.)

**Four independent suites now agree accuracy is a wash**: Grinsztajn +0.005%,
high-card −0.017%, PMLB −0.043%, public −0.114%, all with medians of
essentially zero, and two of the four played no part in the decision.

## Open / stackable

- **`Sel25`** (auditions 100 -> 25) still stacks on top of this: it was
  measured at −26% fit back when auditions were ~35% of a fit, but the
  `refit_full` default flip has since diluted it to about **12%**
  (auditions are now 8-32% of a fit, mean ~16% — see the attribution table
  above). Worth doing, no longer headline-sized.
- **Fused assign+accumulate replay kernel.** A replay round currently costs
  ~1/3 of a grow round rather than the ~1/10 the work would suggest, because
  the grow kernel gets the leaf assignment free as a by-product of its fused
  quantized level pass while replay pays for assignment, leaf-value scatter
  and the `F` update as three separate passes over the rows. Fusing them is
  the remaining headroom. Parallel twins of `_assign_leaves`/`_leaf_values`
  were tried first and made NO measurable difference (1.37x vs 1.41x, noise)
  — reverted; do not retry that specific idea.
- **Multiclass replay.** The vector-leaf round would need its own replay
  path. HC's multiclass sets are exactly where the tie count comes from.
