# Shared-tree multi-quantile head

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

**Next question this raises** (not started): the CRPS gap to CatBoost is a
sharpness deficit, which is the same axis P14 left open against the rigid
offset. Worth a pre-registration of its own.

### Still open

- **`adaptive_learning_rate` is pinned False for this head**
  (`quantile_api.py`, "Measure before flipping") and has never been measured
  against pinball. That is a default flip on a strength surface, so it needs
  its own pre-registration and the full `/experiment` protocol. Not attempted
  here. Recorded 2026-08-30.
- The acceptance-ledger rows below (fit-speed MISS, CQR coverage FAIL) were
  measured on synthetic data. `quantile_suite.py` is the instrument for
  re-reading them on real data; doing so is a separate piece of work.

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

## 2026-09-01 — Out-of-fold calibration study (issue #106 follow-up)

`benchmarks/quantile_oof_calibration.py`: 7 Grinsztajn reg_num sets (<=20k
rows), 5-fold OOF, estimator defaults. Measures what real data can measure:
skill vs the fold-marginal forecast, 20-cell occupancy of the grid (empirical
KL vs the claimed 5% per cell), interval coverage, and predict_thresh
reliability (ECE + Brier skill at the marginal 10/30/50/70/90% thresholds).

| dataset | CRPSskill | medSkill | meanR2 | cov90 | tails lo/hi | cellKL | ECE | BrierSk |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| abalone | 0.358 | 0.364 | 0.541 | 0.857 | .073/.071 | 0.0126 | 0.047 | 0.353 |
| Bike_Sharing_Demand | 0.528 | 0.527 | 0.699 | 0.863 | .068/.069 | 0.0080 | 0.031 | 0.574 |
| elevators | 0.641 | 0.623 | 0.891 | 0.829 | .087/.083 | 0.0268 | 0.051 | 0.498 |
| houses | 0.654 | 0.666 | 0.844 | 0.828 | .087/.085 | 0.0270 | 0.029 | 0.667 |
| diamonds | 0.778 | 0.790 | 0.946 | 0.881 | .059/.060 | 0.0022 | 0.038 | 0.765 |
| MiamiHousing2016 | 0.761 | 0.760 | 0.931 | 0.800 | .097/.103 | 0.0475 | 0.037 | 0.770 |
| medical_charges | 0.883 | 0.891 | 0.978 | 0.887 | .056/.057 | 0.0015 | 0.042 | 0.870 |

Crossing rate exactly 0 on every set. Interior cells near-flat everywhere; the
raw grid's known narrowness shows only in the two tail cells (5.6-10.3% vs the
nominal 5%). conformalize=True re-run on the 3 worst (MiamiHousing2016,
houses, elevators): cov90 0.902/0.902/0.905, every tail cell 0.045-0.050,
cellKL <= 0.0014, at ~1 point of CRPS skill. Verdict: grid, derived heads and
predict_thresh are calibrated OOF; tails need conformalize=True when coverage
is the contract. No open items.
