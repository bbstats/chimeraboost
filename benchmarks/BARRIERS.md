# Barriers — ideas this project has already closed

Companion to `GATE_ROBUSTNESS.md`. That file is about **readings** that fool us.
This one is about **ideas**: each entry is a closure we paid for, stated so that
the next proposal in its family can be recognised before it spends a run.

The scarce resource is benchmark wall-clock — `--decide` is one run at a time and
takes hours. An idea barred here is not forbidden; it is *known to owe an
argument*. Clearing a barrier means saying which specific finding below is wrong
or does not apply, in the plan file, before the run.

`python benchmarks/barrier_check.py "<idea in a sentence>"` matches an idea
against the tags here and prints what it must clear. Run it before tier 1.

Entries are the machine-readable source for that script: `### B<n> — <title>`
followed by a `tags:` line. Keep that shape when adding one.

---

### B1 — Audition knobs are structurally inert below the audition thresholds
tags: audition, selection_rounds, small-data, sus25, sus50, linear_leaves, cross_features, budget

Below `LINEAR_LEAVES_MIN_SAMPLES=1000` and `CROSS_MIN_SAMPLES=2000` no audition
runs at all, so no audition knob can move anything there. 47 of 48 synth sets
under 2000 rows were bit-identical across the sel25 arms.

Consequence: never read a small-data stratum as evidence about an audition knob,
and never propose an audition change *aimed* at small data.

*Incident*: Sel25, 2026-07-31 (`SELECT_PLAN.md`).

### B2 — The rung-3 refit amplifies a bad audition, so audition changes must be judged at `refit_full=True`
tags: audition, selection_rounds, refit, refit_full, replay, quality, budget, rung

Same knob, same suite, same seeds: at rung 3 sel25 read 6W-20L-10T, p=0.009, mean
−0.478%; at rung 2 (`refit_full=False`) it read 10W-16L-10T, p=0.327, mean
**+0.409%**. Rung 3 replays the audition winner's structure over all rows, so a
wrong pick propagates instead of being diluted.

Consequence: selection **fidelity** is worth more than it was under rung 2. Judge
every audition-cheapening idea at the shipped default, and re-value
fidelity-improving ideas upward.

*Incident*: Sel25, 2026-07-31 (`SELECT_PLAN.md`, results `20260731-164927.json`).

### B3 — Partial CatBoost mechanism ports each regress somewhere
tags: catboost, categorical, cat, encoding, onehot, one-hot, combinations, target-statistics, port, leaf-estimation

Seven CatBoost-inspired levers, seven kills. Each partial port helps some
categorical sets and regresses another badly, while the win it chases is already
banked in the current defaults (the all-categorical `cat_combinations` auto-rule,
the binary `linear_leaves` default, plain boosting with size-adaptive
`min_child_weight`).

The gap looks like an emergent property of CatBoost's *integrated*
ordered-boosting-on-permutations machinery, not any single transplantable part.

*Incident*: research cascade, `benchmarks/research/SUMMARY.md`.

### B4 — Ordered boosting is closed
tags: ordered, ordered-boosting, permutation, catboost, small-data

CatBoost never runs ordered boosting — it is `Plain` at every dataset size. The
hypothesis family this project carried for months is retired, and our own dormant
`ordered_boosting` flag with it. Its one size-dependent default is the learning
rate, which is worth 57% of its small-data edge and which we shipped.

*Incident*: SMALLDATA, 2026-08-01 (`SMALLDATA_PLAN.md`, `probe_catboost_ablation.py`).

### B5 — A shrinkage estimator cannot correct a common in-sample/out-of-sample bias
tags: shrinkage, pooling, pooled, blend, calibration, quantile, coverage, band, leaf-values

Shrinkage only corrects the component of the error that **varies across** the
units being shrunk. P15 shrank each leaf's quantile spread toward the pooled band
to fix under-coverage; train coverage sat at the nominal 0.80 at every pseudo-mass
while test coverage never arrived, because the pooled band is computed from
in-sample residuals too and carries the same bias.

Consequence: diagnose whether the defect is a common or a varying component
before designing any correction that rearranges in-sample quantities.

*Incident*: P15 quantile spread smoothing, 2026-07-31 (PR #63, `LEAFTUNE_PLAN.md`).

### B6 — Broad hyperparameter tuning buys nothing that generalises
tags: tuning, hyperparameter, search, random-search, reg_lambda, depth, grid, sweep, defaults

A PMLB random search over the whole space found essentially nothing that
transferred; `min_child_weight` was the one exception, and it shipped. LEAFTUNE
then reproduced the same result on `reg_lambda` specifically against an exact
cross-validated grid: 3 wins, 3 losses, 2 ties of 8 datasets, median exactly
0.000%, none near-solved.

Consequence: the defaults are Grinsztajn-tuned and near a good optimum. A tuning
proposal needs a mechanism story, not a wider grid.

*Incident*: PMLB random-search study; LEAFTUNE P-series, 2026-07-28.

### B7 — Shadow CV: do not rebuild it
tags: cv, cross-validation, folds, shared-structure, out-of-fold, oof, reuse

Built end to end 2026-07-28 and thrown away. Three independent reasons: shared
structure diverges at a 0.70 median so there is nothing to share; sharing it leaks
(out-of-fold log loss 0.376 on pure noise, below the 0.693 floor); and an exact
lockstep tier costs ×1.04 against the naive loop, so even done right it buys
nothing.

*Source*: the program record. It was built and then deleted, so no code or plan
file survives in the repo to re-check these numbers against.

### B8 — Subsample is a dead axis, on both speed and strength
tags: subsample, bagging-fraction, row-sampling, mvs, regularization, stochastic

Killed 2026-07-17 on speed, re-tested 2026-07-30 after the MVS numba work made
`subsample<1` fit 1.32–1.44× faster, and killed again on strength: 0.8 read
primary 63W-67L median −0.000% and **Brier 31W-55L median −0.730%**; 0.7 was
worse. The implicit-regularisation mechanism was absent, not merely small — the
`noise_level` OLS coefficient came back at t = −0.03.

*Incident*: `SUBSAMPLE_PLAN.md`, both dates.

### B9 — Bagging ships only output-identical engineering or more data per member
tags: bagging, ensemble, n_ensembles, members, oob, quality-ladder

The bagging program is closed with `Ens8` as the blessed mode. Two-member bagging
is net negative (2 members lose to 1). Everything that changed member *behaviour*
without giving members more data was killed; what shipped was engineering that
left predictions identical, plus `refit_members`, which reclaims the per-member
data tax.

*Incident*: `BAGGING_PLAN.md`; `refit_members`, 2026-08-01.

### B10 — The remaining grow-kernel objects are below their ceilings
tags: kernel, numba, grow, histogram, scatter, scan, speed, micro-optimisation, dtype

Phase-0 measured every candidate's ceiling as a share of estimator fit before any
was written: multiclass copy trims 0.9%, int32 leaves ~1–2%, uint8 bins ≤2–3% at
small n and ~0 at n≥50K — all killed at ceiling. The one double-digit object left
is the fused scatter+scan split (36.5–57.6% of fit), and it is **FP-drift class**:
it cannot be made bit-identical, so it cannot pass the numerical-identity goldens
that every kernel change here has had to pass.

The one lever that was implemented anyway (L-ridge, a row-major restructure) came
back at ×0.49–0.67 at n≥37.5K and was reverted — the ridge is accumulator-bound,
not gather-bound.

*Incident*: `GROW_PLAN.md`, Phase 0 and Phase-1 verdicts, 2026-07-18.

### B11 — Isotonic-in-sample is a broken stopping rule
tags: calibration, isotonic, early-stopping, es, stopping, brier

Calibration-aware early stopping was killed as a default. The reason is
structural and is the part worth carrying: an isotonic map fitted on the same
rows the stopping decision then scores cannot report honestly on them, so the
stopping curve it produces is optimistic exactly where it is being read.

*Source*: the program record — the shipped calibration is temperature scaling
(`sklearn_api.py`), and no isotonic stopping path exists in the tree today, so
this entry is history rather than something to inspect. Anyone reopening it
should re-derive the numbers rather than quote them.

### B12 — Config portfolios die on cost, not on headroom
tags: portfolio, config, race, multi-config, audition, selection, oracle

A2 passed every strength bar and failed on cost. The test-set oracle ceiling was
+2.587% and validation selection was real (21W-9L-6T, p=0.043) — but it recovers
only **20% of that ceiling**, and racing four configurations projected to
**1.90×** fit time against a 1.35× bar.

The reason generalises past A2: **a k=100 audition is nearly a whole fit whenever
the full fit is short**, which is common (mean 323–382 rounds; multiclass runs
~100). Any proposal that adds auditions must price them against the *short*-fit
case, not the average.

*Incident*: A2 Phase 0, 2026-07-25 (`A2_PLAN.md`).

### B13 — Replay is exact and cheap, and misreads the axis that matters
tags: replay, leaftune, tune_leaves, refit, structure-transfer, sweep

Replay round-trips bit-identically and costs a median 5.2× less per configuration
than re-growing. But its grid agreed with an exact grid's chosen cell on only 2 of
8 datasets, and on the one dataset where the parameter genuinely mattered
(`gr:reg_cat/Brazilian_houses`, +4.4% for exact tuning) replay picked the wrong
end and gave up **6.8%** on test.

Consequence: replay is a screening instrument, not a selection instrument. Its
out-of-fold regret diagnostic *did* detect its own failure — keep that diagnostic
in any future version.

*Incident*: LEAFTUNE, 2026-07-28.

### B14 — The audition budget axis is closed from both ends; no decision rule buys back k<100
tags: audition, const-vs-linear, linear_leaves, margin, race, selection_rounds, early-exit, budget, tail-mean, rule

The const-versus-linear validation race genuinely crosses late on about a third of
regression selections, and on the step-0 curves the two arms' overlap is total —
no margin or early-exit rule at k=100 separates them. `k_ll=500` restores fidelity
but collapses the 1.50× audition speedup to a projected 1.11×.

From below (E1, 2026-08-10): a better pick rule cannot buy back a shorter
budget, because pick fidelity was never the harm. Shipped@25 mispicks barely
more than shipped@100 (14 vs 12 of 48 real races) while Sel25's decide-tier
kill was a rout (6W-20L, p=0.009) — the damage lives in the truncation itself
(cross race included), not in the const-vs-linear pick. Confirmed on the synth
screen: k=25 + tail-mean reproduces plain Sel25's losing signature vs the
default (regression 7W-17L, mean −0.326%, concentrated in shallow/linear/
cross-scope slices) while the rule itself flips only 2 picks in 136 datasets,
both losses. The *per-leg* budget direction is empty too, argued from source
(`PARETO_PLAN.md` D1): the augmented candidate wins 20 of 21 selections and a
leading augmented fit already runs to its own early stop. `selection_rounds`
stays 100.

*Incident*: `PARETO_PLAN.md`, "Known residual"; Sel25 kill 2026-07-31; E1 kill
2026-08-10 (`SELECT_PLAN.md` E1).

### B15 — Histogram sibling-subtraction is closed while histograms stay cache-resident
tags: histogram, subtraction, sibling, scatter, scan, quantize, grow-kernel, hist, parent-minus-sibling, speed

Deriving the sibling histogram as parent − child looked like the last
double-digit fit-speed object twice, and both doors are shut. In the float
domain it is FP-drift class (`GROW_PLAN.md`). In the quantized integer domain
it is exact — and it was built and measured 2026-07-18 at **0.49–0.57×, a ~2×
regression**, with its correctness oracle green. Quantization and subtraction
are SUBSTITUTES, not complements: the packed int64 histogram slice (64 KB per
feature at 128 bins) is L2-resident, so the random read-modify-writes that
subtraction saves are cheaper than the 50/50 `leaf & 1` branch mispredicts it
adds.

Consequence: do not propose histogram subtraction in any domain while the
histogram working set stays in cache. Reopen only if that premise breaks:
`max_bins` ≫ 128, much deeper trees, or a GPU backend.

*Incident*: `QUANT_PLAN.md:263-272` (its SUBTRACT_PLAN.md was never committed);
corroborated `GPU_PLAN.md:56`. Nearly re-proposed 2026-08-16 during campaign
intake — this entry exists so that cannot happen again.

### B16 — Pre-screening the cross-feature candidate block trades picks for time, at every k
tags: cross, cross_features, cross-column, candidate, screen, prescreen, top-k, prune, residual, correlation, split-gain, importance, augmented, diff, prod, gdiff

The cross-audition leg is 40–58% of the default's fit where it engages, and
carrying ~42 candidate columns into the augmented fit looks like obvious waste.
It is not waste that can be safely removed by ranking the candidates first.
Measured 2026-08-16 (`cross_top_columns`, synth, both arms in one run): ranking
candidates by the best single-split variance reduction they achieve on the base
fit's validation residuals and keeping the top k cost regression **4W-14L at
k=6** (p=0.031, mean −0.574%) and **6W-12L at k=12**, while saving only −13.2%
and −11.6% of engaged fit respectively.

Two facts make this general rather than a bad choice of k. The saving is nearly
flat in k (doubling the block back cost 1.6 points of speed), so the time comes
off the tail of the ranking, not the head — while the harm barely moved, so it
comes from misranking near the head. And the statistic is not the weak link
either: plain |correlation| was tried first and is strictly worse, blind by
construction to comparison interactions (the residual a staircase leaves around
an `x_i < x_j` boundary is not linear in `x_i - x_j`), which is half of what
cross features are for. A screen good enough to rank these columns is a screen
that already knows the answer the augmented fit is being run to find.

Consequence: do not propose ranking-then-trimming the cross candidate set.
Cheapening this leg has to come from making the augmented fit cheaper per
column, not from carrying fewer columns. B14 separately closes the round-budget
axis, so both "do less of the audition" doors are now shut.

*Incident*: `CAMPAIGN_PLAN.md` F1, entries I005–I008 (2026-08-16); runs
`results/campaign-f1s2-20260816.json`, `results/campaign-f1s2b-20260816.json`.

---

### B17 — Sub-gate CV-averaged selection races at chance
tags: cv, race, subgate, audition, small-data, cross-features

Below the 2000-row gate the single validation split is too small to referee
cross features (B1) — but averaging the referee over 3 stratified folds
does not repair it either: on the synth screen the CV-averaged race picked
the augmented model on 10 sub-gate multiclass sets and was right on
exactly 5 (engaging median −0.22%), while each engaging set paid 3-7x the
base fit (B12: short fits run ~full rounds, so 6 fold-fits cost ~5 full
fits). ~110-row validation slices stay noisy even averaged; a cheaper or
cleverer referee doesn't fix a signal that thin. Worse than imprecise, the
referee is unstable: one set's decision flips between two identical runs
(aug-pick in one, decline in the rerun). The one bright thread (a +17.76%
win where interaction depth runs high; OLS t+2.90 on interaction_depth) is
a dataset-selection question for a future family, not a reprieve:
precision over the engaging slice was 50%.

Consequence: do not propose re-refereeing below-gate selection with CV
averaging (or more folds thereof). If the geometry-prize thread is ever
taken up, it starts as a dataset-gating claim with its own S1, not as F2.

*Incident*: `CAMPAIGN_PLAN.md` F2, I016 (S1 thin pass, +1.20% on one real
set) → I017 (S2 5/5 flat, killed); screen `results/f2s2-20260918.json`.

---

### B18 — The ordered-TS train/test asymmetry is real, and fixing it at transform moves one dataset
tags: target-statistics, ordered-ts, encoder, transform, smoothing, shrinkage, rare, high-cardinality, count, prior, train-test, mismatch

`fit_transform` encodes a training row from a permutation prefix (expected
weight on the category mean 0.54 at m = 5, exactly 0 for a singleton);
`transform` encodes every other row from full totals (0.83, and 0.5). Measured
2026-09-21: on high-card columns a held-out row of a rare category (m ≤ 5)
reads an encoding 1.9–5.2× more spread than a training row of the same
stratum, and the target's slope on it is 0.63–0.70 of the training slope —
the trees over-trust rare categories. In big categories it runs the other
way (1.10–1.29). The mean never moves (≤ 0.03 SD), so a first-moment check
misses all of it.

Two transform-side fixes were tried, both zero-library-change and both
pre-registered. Matching the weight for every category: 7W-5L of 12 against a
bar of 8 — it gains where rare held-out rows are common (sf-police +0.31%,
3/3 seeds) and costs a little where they are not (kick 0/3). Matching it for
rare categories only, tested with the training rows subsampled so every
category gets rarer: 7W-5L again, and without sf-police the other three sets
read −0.03% (4W-5L). sf-police itself is 9 of 9 fits, +0.29% growing to
+0.53% — because 27–41% of its held-out rows sit in categories of five or
fewer training rows. No other set in the high-card suite has such a column.

Consequence: do not propose re-weighting, re-smoothing or count-matching the
transform side of the target encoder. The change would re-score every
categorical model and every categorical golden for one dataset in fourteen.
The door that stays open is a different mechanism — letting the trees SEE the
count, which is how CatBoost lives with the same asymmetry — and it goes
through the CatBoost ablation (shortlist R3) and B3 first. `cat_smoothing`
is a separate, already-closed axis: it moves both sides together.

*Incident*: `CAMPAIGN_PLAN.md` F6, I024 (S1, uniform weight) → I025 (S1b,
rare-only, killed); `benchmarks/probe_ts_mismatch.py`; results
`probe-ts-mismatch-20260921.json`, `probe-ts-mismatch-s1b-20260921.json`.

---

### B19 — The early-stopping round is not a free axis: smoothing the argmin buys nothing, stopping earlier costs strength
tags: early-stopping, early stopping, stopping rule, argmin, smoothing, smoothed, moving average, tail averaging, polyak, ema, best iteration, validation curve, round, patience, 1-se, one standard error, tolerance, earlier stop, fewer rounds

Production stops at the raw argmin of the held-out validation curve
(patience 50). Probed 2026-07-13 (`benchmarks/probe_tail_averaging.py`,
`results/probe-tailavg.jsonl`; 11 Grinsztajn sets × 3 seeds, one fit each
with `early_stopping=False`, staged test predictions): the argmin of a
9-round moving average of the curve is within ±0.03% of the raw stop on 9
sets of 11, −0.58% on sulfur, and +5.24% on Brazilian_houses — one seed
where the raw stop landed on a validation spike. The regression mean of
+0.67% is that single row; the binary mean is −0.008%. Polyak-style tail
averaging of the last K rounds' predictions (K = 5/10/20) and the symmetric
window around the stop read ≤ 0 on every set and every K (regression −0.2
to −1.2%, binary −0.02 to −0.23%).

The other direction, stopping EARLIER for cost, was probed 2026-09-22
(`benchmarks/probe_es_one_se.py`, 21 Grinsztajn sets × 3 seeds, same
protocol): the earliest round within one noise unit of the minimum, and
within 0.1% / 0.5% of it. Every rule loses in proportion to the rounds it
drops — 0.5% tolerance gives back 0.25–0.66% of RMSE / Brier on 20 of 21
sets for a 16–22% round saving, 0.1% gives back ~0.09% for 5–6%, and the
noise-unit rule barely moves the pick (ratio 0.99) and still reads 5W-15L.
The validation argmin already sits on the flat part of the test curve;
everything before it is uphill.

Consequence: do not propose picking the boosting round by a smoothed
validation curve, by averaging predictions across rounds near the stop, by
any other post-hoc reweighting of the trajectory the argmin already
approximates, or by an earlier-within-tolerance rule for cost. The one
pathology the smoothing grazed (a spike-sited stop on Brazilian_houses) is
the validation-noise problem the cross-feature race addresses at the
source. The round count is a strength axis, not a free cost axis; a cheaper
fit has to come from cheaper rounds.

*Incident*: the July half was recorded in session memory only, re-proposed
by the 2026-09-21 beam refill as shortlist R2, and caught by a grep rather
than by `barrier_check.py`, which is why this entry exists; the 1-SE half
is `CAMPAIGN_PLAN.md` I032.

---

### B20 — Finer bins (max_bins 128 → 254) sharpen a few high-signal sets and overfit the rest
tags: max_bins, bins, bin count, border count, border_count, 254, 255, 256, bin resolution, finer bins, more bins, histogram resolution, max_bin

`max_bins=128` is half of every opponent's default (CatBoost 254 borders,
LightGBM and HGB 255, XGBoost 256), so "give the splits more resolution"
keeps looking like free strength. It was tested 2026-06-01 on three suites
and rejected: on Grinsztajn it read net positive (regression 20W-4L) with
the Brier gain almost all `electricity`; on an independent OpenML panel it
was an overall wash (+0.03%) with regression REVERSED (1W-4L, `cpu_act`
−30%, and −17% on Grinsztajn's `cpu_act`, a reproducible overfit); on a
fresh 17-set binary batch a wash (median +0.25%, `madelon` −10%, fine bins
fitting noise). It also broke a sub-1k-row MAE test and cost 30–50% fit at
the time.

Re-read 2026-09-22 (I047, `benchmarks/probe_catboost_split_score.py`,
8 Grinsztajn binary keys × 3 seeds, the harness's own splits): 254 bins
up on 7 of 8, `electricity` +7.97% on all three seeds and hitting the
2000-round cap, the other six gains +0.09% to +0.88%, fit ×1.12 median
(quantized histograms made the cost smaller; the strength picture did not
change). Same shape as June: one fine-grained set carries it.

Consequence: do not propose a larger default bin count, or a probe arm
whose case rests on "the opponents use ~255", without a mechanism that
answers the June regression reversal (a size- or noise-aware bin rule
would be such a mechanism; a flat increase is not). The other direction,
whether 128 bins is too fine for small or noisy numeric data, was asked at
I048 and the answer is no (below).

*Incident*: the June verdict lived only in session memory
(`project_algorithm_history`), so `barrier_check.py` could not find it; the
2026-09-22 S2 task added a `chimera_bins254` arm on the strength of
"LightGBM at 255 bins beats us on california" and re-derived the June
result. `CAMPAIGN_PLAN.md` I047.

Re-read the other way 2026-09-22 (I048, `benchmarks/probe_small_reg_loss.py`):
64 bins on the eight gr regression small-data keys helps `cpu_act@sus25`
+6.6% on average by a coin flip per seed (+17.8 / +16.0 / −15.7%) and loses
on 5 of the other 7 keys, the parent `cpu_act` by −7.6%. Coarser is not a
small-data rule either: 128 stays, in both directions.

---

### B21 — A regression min-leaf floor is directionally right on small data and too small to ship
tags: min_child_weight, mcw, min leaf, min-leaf, minimum leaf, leaf size, min_data_in_leaf, min_samples_leaf, sample floor, regressor, small data, small-data regression, tiny leaves, sparse leaves

For squared error the Hessian is one per row, so the regressor's
`min_child_weight=1.0` lets a leaf hold a single row, where LightGBM
requires 20. Probed 2026-08-01 (`benchmarks/probe_reg_mcw.py`,
`BREAKTHROUGH_PLAN.md` C3; all 42 decision-suite regression sets × 3
seeds × train fraction 1.0 / 0.5 / 0.25, mcw ∈ {1, 4, 8, 16, 32}): each
dataset's own best mcw rises as rows shrink (22 up, 5 down, p = 0.002),
but the pre-registered primary arm (mcw = 8 at a quarter of the rows) reads
23W-17L, median +0.085%, Holm p = 1.000, and the best fixed arm captures
13–31% of the per-dataset oracle, so no size schedule recovers it. The
veto bound (rounds ×1.05–1.24): a refutation, not an inert arm. On an
oblivious tree the floor vetoes a whole level, not the one sparse leaf.

Consequence: do not propose a minimum-leaf-size default for the regressor,
fixed or size-adaptive, as the answer to a small-data regression loss.
`cpu_act@sus25`, the largest such loss on the suite, was in that panel:
mcw = 8 took its RMSE from 3.82 to 3.53 against LightGBM's 2.71.

*Incident*: the 2026-09-22 S3 probe (`CAMPAIGN_PLAN.md` I048) reached for
"LightGBM's min_data_in_leaf = 20" as the explanation of `cpu_act@sus25`
before a grep found C3, which `barrier_check.py` could not see.

---

## Adding an entry

An entry earns its place when a closure is **paid for and general** — a measured
kill whose reason will recur. A one-off negative goes in its plan file and the
algorithm history, not here. Keep the `### B<n> — title` / `tags:` shape so
`barrier_check.py` keeps working, and state the incident that proved it.
