# ChimeraBoost: a whitepaper

*Lightning-fast gradient boosting, near-CatBoost quality, all in Python.*

**Version 0.32.0 · September 2026 · Apache-2.0 · [github.com/bbstats/chimeraboost](https://github.com/bbstats/chimeraboost)**

## Abstract

ChimeraBoost is a gradient-boosted decision tree library written entirely in Python, on
a dependency surface of exactly four packages: numpy, numba, scipy, and scikit-learn.
No C++ extension, no GPU, no pandas requirement. On TabArena, the community tabular
benchmark that this project treats as a sealed holdout — reported, never tuned against —
ChimeraBoost's default configuration is the second-strongest gradient-boosting library
on the board, behind only CatBoost and more than a hundred Elo points clear of XGBoost
and LightGBM defaults. It reaches that quality while training in under half of
CatBoost's median time on the same hardware, and it ships capabilities the incumbents
do not: a structurally non-crossing predictive-distribution head, exact SHAP
attributions with no extra install, and a self-tuning pipeline whose honest advice on
hyperparameters is *mostly, don't*.

This paper lays out the evidence first, the practitioner-facing benefits second, and
the design that produces them third — with the limitations stated as plainly as the
wins.

## 1. What it is

ChimeraBoost is a scikit-learn-compatible GBDT with a deliberately small public API:
seven names, of which three are estimators — `ChimeraBoostRegressor`,
`ChimeraBoostClassifier`, and `ChimeraBoostQuantileRegressor`. It covers regression
(squared error, absolute error, Huber, Poisson, Gamma, Tweedie, custom objectives),
binary and multiclass classification, and joint estimation of a full grid of
conditional quantiles from a single booster.

The core design is CatBoost's oblivious tree: every node at a given depth shares one
(feature, threshold) pair, so a depth-*d* tree is just *d* splits and a leaf index is a
*d*-bit number. Most other ideas are borrowed too, and credited — ordered target
statistics for categoricals, histogram split finding, minimum-variance sampling,
quantized gradient histograms, vector leaves, linear leaves, exact TreeSHAP. The
library's [attribution page](attribution.md) separates what was borrowed, what was
adapted, and the two contributions that are original.

One integer, `quality=1..5`, selects a point on the speed/accuracy frontier; the
default is 3. The library is roughly 11,700 lines across 13 modules, held to a CI-
enforced complexity ceiling, and covered by 601 test functions including bit-identity
goldens for every performance refactor.

## 2. The evidence

### 2.1 TabArena: a sealed holdout

[TabArena](https://tabarena.ai) is a living tabular-ML leaderboard maintained by its
own authors (Erickson et al., NeurIPS 2025): 51 curated classification and regression
datasets, every model run by the maintainers in a pipeline they optimize, ranked by
Elo. ChimeraBoost's development process treats it as a **sealed holdout**: results are
reported, and are never — in aggregate or per-task — allowed to influence a change to
the source. Development decisions run on separate suites (§5), so the number below is
an out-of-sample read in the strictest sense available to us.

As of 2026-09-01, among gradient-boosting libraries at **default settings**:

| model (default config) | Elo | median train s/1K | median predict s/1K |
|---|---|---|---|
| CatBoost | 1379 | 5.88 | 0.025 |
| **ChimeraBoost** | **1315** | **2.65** | **0.052** |
| XGBoost | 1214 | 1.94 | 0.123 |
| EBM | 1204 | 6.67 | 0.014 |
| LightGBM | 1187 | 1.96 | 0.142 |

Two readings of that table. On strength, ChimeraBoost's default is second only to
CatBoost, and the gap *down* to third place (101 Elo) is half again the gap up to first
(64). On cost, it trains in 45% of CatBoost's median time per thousand rows and
predicts two to three times faster than XGBoost and LightGBM defaults — though CatBoost
remains the faster predictor. ChimeraBoost's tuned-and-ensembled entry scores Elo 1383,
nudging past CatBoost's default, on a search space that has not been refreshed since
June. (For calibration: the top of the overall board belongs to tabular foundation
models at Elo ≈ 1790 — a different trade-off class entirely, involving pretrained
networks orders of magnitude heavier than any GBDT.)

### 2.2 The public suite: quality per unit of compute

![Strength versus fit-time slowdown on the public suite](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png)

The published chart runs on 22 independently audited public datasets that overlap with
nothing the defaults were tuned on. On its first full read (2026-07-27, three seeds,
21 of 22 datasets scored):

| model | avg rank | 95% CI | median slowdown |
|---|---|---|---|
| ChimeraBoost `quality=4` (ensemble) | **1.73** | 1.46–1.99 | 16.4× |
| CatBoost | 1.92 | 1.53–2.31 | 52.0× |
| ChimeraBoost default | 1.97 | 1.67–2.28 | 7.2× |
| LightGBM | 2.23 | 1.74–2.70 | 1.0× |

Rank is per three-way contest (each ChimeraBoost rung against CatBoost and LightGBM),
so 2.0 is the middle of the field. The defensible sentence is a cost claim: **the
default is within noise of CatBoost at under a third of CatBoost's median fit time**,
and the ensemble rung tops the table at still less than a third of CatBoost's cost.

## 3. What that buys a practitioner

### 3.1 Data goes in as it is

Categorical columns are handled natively by ordered target statistics — CatBoost's
scheme, averaged over four permutations — addressed by column index or DataFrame name.
Missing values get their own histogram bin at fit and at predict, numeric and
categorical alike. There is no one-hot step, no imputation step, no label encoder, and
no pandas dependency: DataFrames are consumed through their own `to_numpy`.

### 3.2 Mostly, you don't tune it

The library audits itself during the fit. Linear leaves versus constant leaves, and
automatic cross features versus plain, are raced on a shared budget with only the
winner continuing to full early stopping. The learning rate adapts to dataset size —
an idea recovered by diffing CatBoost's resolved parameters across dataset sizes, which
revealed that its one size-dependent default is the learning rate. Early stopping is
on by default, and the stopped model is then re-fit on all rows by structure replay
(§4). What remains is one integer:

| `quality` | profile | fit time (× fastest library) |
|---|---|---|
| 1 | fast | 2.7× |
| 2 | balanced | 5.3× |
| 3 | accurate (default) | 6.9× |
| 4 | ensemble of 5 | 18.1× |
| 5 | ensemble of 8 | 26.0× |

Every rung buys accuracy for its extra time; each sits on the measured frontier.

### 3.3 Uncertainty is a first-class output

`ChimeraBoostQuantileRegressor` fits a whole predictive distribution — 19 quantile
levels by default — in **one booster**, with one tree structure per round and a
K-vector in every leaf. Off that single fitted grid it answers, at no extra cost:
central intervals, the median, the tau-integrated mean, the CDF at any threshold,
inverse-transform samples for downstream simulation, and `predict_thresh(X, t)` — the
probability that the outcome exceeds a threshold, per row.

Three properties distinguish it from stacking up per-level quantile models:

**Predictions never cross.** Every row's grid is monotone by construction — sorting a
crossing quantile curve never increases pinball loss (Chernozhukov et al., 2010), so
the guarantee is free. Measured across 36 real regression datasets, LightGBM's
per-level boosters reverse 22% of adjacent level pairs on average and cross on every
single dataset; CatBoost's shared `MultiQuantile` head also crosses on every dataset.
ChimeraBoost's crossing rate is exactly zero on all 36.

**The intervals are honest, and can be guaranteed.** The raw grid's nominal 80% and
90% intervals deliver 77% and 87% observed coverage — closer to nominal than LightGBM
per-level (72%/83%) or CatBoost MultiQuantile (73%/83%). `conformalize=True` upgrades
that to distribution-free marginal coverage, measured within 2.7 points of nominal at
n = 10,000. An out-of-fold study on seven real datasets (5-fold, 2026-09-01) found the
derived heads calibrated as claimed — exceedance probabilities from `predict_thresh`
land within an expected calibration error of 0.03–0.05 of reality, crossing rate zero
everywhere — with the raw grid's known narrowness confined to the outermost tail cells,
and conformalization repairing exactly that: 90% coverage of 0.902–0.905 on the three
worst datasets, at about one point of CRPS skill.

**You can explain the width.** SHAP attributions come per level — and, uniquely to a
shared-structure head, for the *width* of an interval: which features make this row's
prediction uncertain, as opposed to higher or lower. No stack of independent per-level
models can produce that decomposition, because it requires both interval edges to come
from one model.

Head-to-head on those 36 datasets, all arms early-stopping on the same split: against
19 independent LightGBM quantile boosters, CRPS is a statistical tie (23W–13L) but the
interval score — the proper rule that prices coverage and width together — is 31W–5L in
ChimeraBoost's favor, at 1.5× less fit time. Against CatBoost's MultiQuantile the
honest result is split and reported as such in §6.

### 3.4 Explanations and calibrated probabilities, built in

`shap_values` computes **exact** interventional TreeSHAP — no `shap` package, no
sampling approximation, roughly 3 ms per row at depth 6. Oblivious trees are why
exactness is affordable: a depth-6 tree involves at most six players, so all 64
coalitions are simply enumerated. Attributions remain exact through per-leaf linear
models, and work for multiclass and quantile heads. Classifier probabilities are
temperature-scaled on the early-stopping fold; the scaling is monotone, so accuracy and
AUC are untouched while calibration improves, and the fitted `temperature_` is exposed.

### 3.5 Deployment is four packages

The wheel depends on numpy, numba, scikit-learn, and scipy — nothing else. Predict
latency at TabArena's scale is 0.05 s per thousand rows. The one honest tax is numba's
JIT: the first fit on a fresh install pays roughly 10 s of compilation (0.8 s warm,
0.27 s steady-state), and upgrading the package re-pays it. The library ships a
`chimeraboost-warmup` command and an environment-variable hook to pre-compile in the
background; the [deployment guide](deployment.md) documents the trap rather than hiding
it.

## 4. How it works

**Oblivious trees** are the speed and much of the regularization. Because a leaf is a
bit pattern of *d* shared comparisons, prediction is a handful of comparisons and one
table lookup per tree, executed in a single numba kernel parallelized over *samples* —
each row walks the entire forest while its features stay in cache. The constraint that
one (feature, threshold) serves a whole depth is also a strong prior against
overfitting — and, on high-signal low-noise data, a real handicap against leaf-wise
learners; §6 owns this.

**Histograms, quantized.** Split finding runs on binned features (128 bins by
default, missing values binned separately) with gradients and hessians quantized to
15-bit integers (after Shi, Ke et al., NeurIPS 2022). Integer histograms fit 20–25%
faster at the same measured accuracy; leaf values are always computed from exact float
gradients, and results are deterministic for a fixed seed.

**Vector leaves** give multiclass one tree per round instead of one per class — fits
2.5× faster with Brier score *improved* 5% at F1 parity — and give the quantile head
its K-vector leaf.

**Replay refit** is one of the library's two original contributions. Early stopping
needs a validation fold, so the stopped model has seen only part of the data; a
from-scratch refit on all rows used to cost 37–49% of every default fit. Instead,
ChimeraBoost replays the winning structure — the recorded splits, round by round —
against gradients computed on all rows, refitting leaf values only. Growing trees is
83–85% of a fit; replay skips it. Result: total fit time down 34.8% on the 59-dataset
decision suite, faster on 58 of 59, with accuracy a wash across four independent
suites.

**A gradient-matrix-free quantile head** is the other. The K-channel pinball gradient
collapses per row to a single projected value in one pass, so the (n × K) gradient
matrix is never materialized and a 19-level head costs about what a 1-level head costs
per round — within 7% of the exact summed-gain answer. The projection matters: naively
summing gradients across a symmetric grid is blind to spread, because the pushes from
level τ and level 1−τ cancel exactly for an interval that is centered but too narrow.

**Bagging without replacement.** `n_ensembles` fits members on 80% subsamples drawn
without replacement — measured better than the bootstrap on accuracy *and* fit time —
in parallel worker processes sharing one thread budget. `refit_members=True` replays
each member on its full data: five refit members beat eight plain ones on both accuracy
and speed.

## 5. The measurement discipline

Numbers are only as good as the process that produced them, so the process is part of
the product. Five benchmark suites, each with one job: a synthetic screen for
mechanism; a decision suite (Grinsztajn et al. 2022 plus a frozen high-cardinality
set) where changes ship or die by per-stratum sign tests; a tuning suite (PMLB) that
is the *only* data hyperparameters may see; a 22-dataset audited public suite for
published evidence; and TabArena, sealed. A suite that tunes cannot also validate, and
the leaderboard we quote never feeds back into the code. Datasets enter suites on data
properties alone — row counts, cardinality, missingness — never on results, or the
suite would be cherry-picked from birth.

The same discipline applies retroactively: when the quantile head's early speed claims
(3–6× versus LightGBM) turned out to be an artifact of fixed-round fits on synthetic
data, the docs were corrected in place to the real-data figure (1.5×), and the
correction is documented on the page it amends.

## 6. Limitations

- **CatBoost's MultiQuantile is sharper on CRPS** — it wins 29 of 36 datasets against
  our head. That is a real deficit, not noise. It comes at a median 8.3× our fit time
  and with much worse interval calibration (its worst coverage error is 0.64 against a
  nominal 0.90; ours is 0.10), which is why the interval score favors us 25W–11L — but
  a user who cares only about CRPS and not about cost should know this.
- **No GPU**, by design; the pure-Python constraint is the project's foundation.
- **JIT cold start**: ~10 s of one-time compilation per fresh install or upgrade,
  mitigated but not eliminated by the warmup tooling.
- **The oblivious tax**: on high-signal, low-noise data, leaf-wise learners can carve
  sharper structure than symmetric trees. The regularization that wins on noisy
  real-world tabular data is a handicap on nearly-deterministic problems.
- **Training is not the fastest**: LightGBM and XGBoost defaults train faster (1.9–2.0
  vs 2.65 s per thousand rows on TabArena's hardware); the claim is quality per unit of
  compute, not minimum compute.
- **The API is 0.x beta**; names are stable in practice but not yet guaranteed.

## 7. Getting started

```
pip install chimeraboost
```

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostQuantileRegressor

clf = ChimeraBoostClassifier()                     # or quality=4 for the ensemble
clf.fit(X_train, y_train, cat_features=["city"])   # categoricals by name, NaNs fine
proba = clf.predict_proba(X_test)                  # temperature-calibrated

qr = ChimeraBoostQuantileRegressor(conformalize=True)
qr.fit(X_train, y_num, cat_features=["city"])
lo, hi = qr.predict(X_test, kind="interval", alpha=0.1).T   # guaranteed 90% band
p_over = qr.predict_thresh(X_test, 100.0)                   # P(y > 100) per row
```

Documentation: [bbstats.github.io/chimeraboost](https://bbstats.github.io/chimeraboost/).
Benchmark harness, raw results, and every study cited here:
[github.com/bbstats/chimeraboost](https://github.com/bbstats/chimeraboost) under
`benchmarks/`.

## Sources and further reading

TabArena figures are the official leaderboard read of 2026-09-01
(`benchmarks/tabarena/LEADERBOARD_SNAPSHOT.md`); public-suite figures are the audited
read of 2026-07-27 (`benchmarks/PUBLIC_PLAN.md`); quantile comparisons are
`benchmarks/quantile_suite.py` over 36 Grinsztajn regression datasets, three seeds
([full tables](quantiles.md#how-it-compares)); the out-of-fold calibration study is
`benchmarks/quantile_oof_calibration.py` (2026-09-01); replay-refit and split-search
results are recorded in `docs/attribution.md`. The research ChimeraBoost builds on —
CatBoost, XGBoost, LightGBM, MVS, quantized GBDT training, SketchBoost/GBDT-MO,
linear-leaf trees, TreeSHAP, OpenFE — is cited per-feature on the
[attribution page](attribution.md).
