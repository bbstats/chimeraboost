# Predictive distributions

`ChimeraBoostQuantileRegressor` estimates a whole grid of conditional quantiles from a
single booster. One tree structure per round serves every level, and each leaf holds a
K-vector with one entry per level. CatBoost's `MultiQuantile` loss works the same way;
one booster for the whole grid is not a design we came up with.

```python
import numpy as np
from chimeraboost import ChimeraBoostQuantileRegressor
from chimeraboost import quantile_metrics as qm

model = ChimeraBoostQuantileRegressor(random_state=0).fit(X_train, y_train)

Q = model.predict(X_test)                            # (n_samples, 19)
model.quantiles_                                     # the level for each column
lo, hi = model.predict(X_test, kind="interval", alpha=0.1).T   # central 90%
mean = model.predict(X_test, kind="mean")            # tau-integrated point prediction

print(qm.format_report(model.report(X_test, y_test)))
print(np.mean((y_test >= lo) & (y_test <= hi)))      # realized 90% coverage
```

The default grid is 0.05, 0.10, ... 0.95. Pass your own with `quantiles=[...]`,
ascending, unique, and strictly inside (0, 1). Column `k` of `predict` is level `k`.
`kind="interval"` reads its two levels straight off the grid and raises if they are not
on it, so fit the levels you intend to use:

```python
model = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.5, 0.9],
                                      random_state=0).fit(X_train, y_train)
lo, med, hi = model.predict(X_test).T
```

More worked snippets are in [Recipes](recipes.md#quantile-regression).

## Reading the distribution other ways

`predict` will answer four more questions off the same fitted grid, at no extra cost:

```python
model.predict(X, kind="median")                          # (n,) the centre
model.predict(X, kind="cdf", thresholds=[0.0, 10.0])     # (n, 2) P(y <= t)
model.predict(X, kind="sample", n_samples=500, random_state=0)   # (n, 500)
model.predict_thresh(X, 10.0)                            # (n,) P(y > 10)
```

`kind="cdf"` inverts the grid, and `kind="sample"` draws from it by inverse transform —
useful for feeding a downstream simulation. Both interpolate between fitted levels and
clamp outside the outermost ones, because a finite grid says nothing about the tails
beyond it. `kind="interval"` still refuses levels you did not fit: reading a fitted
curve at a point is a different thing from claiming a level was fitted when it was not.

`predict_thresh` is the exceedance view of the same inversion: `direction="greater"`
(the default) returns `P(y > t)`, `"less"` returns `P(y <= t)`. Thresholds may be a
scalar, a 1-D list applied to every row, or a 2-D `(n, T)` array read row against row —
1-D always means shared, per-row always means 2-D, regardless of length. The clamp
carries over: on the default grid no probability reads below 0.05 or above 0.95,
because the model never estimated those tails.

Both CDF readers need a grid dense enough to interpolate honestly. If any gap between
adjacent fitted levels exceeds 0.2 — `quantiles=[0.1, 0.5, 0.9]`, say — they warn,
because the probabilities would be mostly interpolation between distant levels rather
than estimates. A sparse grid is fine for the intervals it was fitted for; fit the
19-level default when you want probabilities.

## Predictions never cross

The 30% quantile is never returned above the 70%. Every row is sorted on its way out, so
`np.diff(Q, axis=1) >= 0` holds exactly, including at every intermediate stage of
`staged_predict`. Sorting is not a compromise: rearranging a crossing quantile curve
never increases pinball loss at any level, for any row (Chernozhukov, Fernández-Val &
Galichon 2010), so the guarantee is free.

Independently fitted per-level models have no such property. Across the 36 real
datasets in `benchmarks/quantile_suite.py`, LightGBM's per-level boosters reverse 22% of
adjacent pairs on average and cross on every single dataset; CatBoost's own shared head
crosses on every dataset too. Ours is exactly zero on all 36.

The band is free to be much *narrower* than the pooled one where the data is quiet — it
tracks the local spread rather than a global floor.

## Interval calibration

The intervals are calibrated by default. After the fit, each level is rescaled about the
predicted median by a conformal factor (Romano, Patterson & Candès 2019), computed on the
rows early stopping already held out: your `eval_set`, or the `validation_fraction` fold
the model carves for itself. No extra rows are spent. Across the 36 datasets in
`benchmarks/quantile_suite.py`, a nominal 80% interval covers 81% on average and a
nominal 90% one covers 90%. LightGBM per-level (72% and 83%) and CatBoost
`MultiQuantile` (73% and 83%) both fall short.

```python
model = ChimeraBoostQuantileRegressor().fit(X, y)
print(model.conformal_scale_)      # one factor per level; above 1 widened the fit
```

This default is `conformalize="auto"`. It carries no formal guarantee, because the rows
it calibrates on also chose the stopping round. When you need one, `conformalize=True`
holds out `calibration_fraction` of the rows **before** the early-stopping split, so that
fold influences neither the fit nor the stopping point. On exchangeable data this gives
distribution-free marginal coverage. Measured coverage lands within about 2 percentage
points of nominal at n = 10,000, erring on the wide side for the outer intervals, since
conformal prediction is conservative by construction. The price is the rows in that
fold: on the 36 datasets the default scores a better CRPS on 35.

`conformalize=True` raises rather than guessing when its fold is too small to certify the
levels you asked for. A 90% interval needs at least 9 calibration rows, and a 99% one
needs 99. The default never raises for this: when its rows cannot certify the grid, or
when there are none (early stopping off and no `eval_set`), it returns the raw grid.

The raw grid, which `conformalize=False` also returns, runs narrow. Leaf values are the
residual quantiles of the rows in that leaf, measured on those same rows, which is
optimistic. The deeper trees of the default narrow it further, and the calibration
repairs that.

## Three candidates, one kept

By default the fit builds three candidates and keeps the one with the best CRPS on the
rows early stopping held out:

- the head as configured;
- the same head with 254 histogram bins, which resolves very fine-grained signal;
- the head's shape moved onto the median of an ordinary squared-error
  `ChimeraBoostRegressor`, which places the centre better on low-noise targets.

```python
model = ChimeraBoostQuantileRegressor().fit(X, y)
model.audition_["selected"]     # "head", "bins" or "recentred"
model.audition_["crps"]         # each candidate's score on the held-out rows
```

Each candidate is calibrated the same way before it is scored. Every prediction method
serves the winner, and so does `shap_values`: for the recentred candidate the
squared-error model's attributions are added to every level, so the attributions still
add up to the prediction.

On the 36 Grinsztajn regression datasets the choice beats a single head on 23 and loses
on 7 (on the other 6 it keeps the single head), with a median CRPS gain of 1.6% where the
two differ. The gain is largest on low-noise targets: 7% to 30% on five of them. It
costs about 2.35 times the fit time of a single head. The choice needs held-out rows, so
with early stopping off and no `eval_set`, or with `conformalize` set to `True` or
`False`, one head is fitted. `audition=False` fits one head, as earlier releases did.

## Scoring

`chimeraboost.quantile_metrics` scores a predicted grid.

```python
from chimeraboost import quantile_metrics as qm

print(qm.format_report(model.report(X_test, y_test)))
```

| function | answers |
|:--|:--|
| `pinball_loss` | Is each level in the right place? (one value per level) |
| `crps` | Is the distribution as a whole right? |
| `quantile_skill_score` | Is it right by a useful margin? 1 perfect, 0 no better than ignoring every feature. |
| `interval_coverage` | Do the intervals hold what they claim? Coverage and width. |
| `interval_score` | Coverage and width in one number — the proper rule that trades them off. |
| `sharpness` | Width alone, for comparing two equally calibrated models. |
| `pit_values` / `pit_histogram` | *Where* is the model wrong? |
| `crossing_rate` | What fraction of adjacent pairs is out of order? |

### Reading CRPS

CRPS is the one number for "is this predictive distribution any good". Lower is better,
and only the true conditional distribution reaches the minimum. Exactly three things
make it worse, and the score alone will not tell you which:

```python
from scipy.stats import norm
taus = np.round(np.arange(0.1, 0.91, 0.1), 2)
y = rng.standard_normal(20_000)                     # the truth is standard normal
grid = lambda loc, scale: np.tile(loc + scale * norm.ppf(taus), (len(y), 1))
```

| forecast | CRPS |
|:--|--:|
| right centre, right width | **0.3075** |
| right centre, 3× too narrow | 0.3451 |
| right centre, 3× too wide | 0.4484 |
| centre off by 1, right width | 0.4537 |

Note the third and fourth rows especially. Being **too narrow** is punished as well as
being too wide, which is what stops CRPS being gamed by shrinking the band — the same
property `interval_score` has. When CRPS says something is off, use
`interval_coverage` and `sharpness`, or the PIT histogram below, to find out which of
the three it was.

**On the factor of two.** `crps` returns the mean pinball loss over the grid, which is
half the textbook value; the convention matches the early-stopping metric. This never
changes which model wins — every score is on the same scale, and switching conventions
multiplies them all by two. In the table above, `convention="full"` reads 0.6150,
0.6902, 0.8969, 0.9073: the same ordering, the same relative gaps. Use it only when
quoting a number next to another library, since `properscoring` and `scoringrules` both
report the doubled value.

Two models are only comparable on the same tau grid, since the grid bounds how well the
integral is approximated.

Coverage on its own is not a score: an infinitely wide interval covers everything.
`interval_score` is the Winkler score, which charges the width plus a penalty for every
outcome that falls outside, and so cannot be gamed in either direction.

The PIT histogram is the instrument for the under-dispersion described above. It asks
where in the predicted grid each outcome actually landed: flat means calibrated, a U
means the bands are too narrow, a hump means too wide.

```python
freq, edges = qm.pit_histogram(y_test, model.predict(X_test), model.quantiles_)
```

`predict(kind="mean")` integrates the quantile function over tau, by the trapezoid rule
across the grid with the edge levels extended flat to 0 and 1. That flat extension
assumes nothing about tails the model never estimated.

## Explaining a predicted distribution

`shap_values` gives exact TreeSHAP attributions with a channel per level:

```python
phi = model.shap_values(X_test)                 # (n, n_features, n_quantiles)
model.shap_importances(X_test, n_features=5)    # averaged over the grid
model.shap_values(X_test, quantile=0.95)        # (n, n_features), one level
```

The one no per-level approach can give you is the attribution of interval **width** —
which features make a particular row's prediction more *uncertain*, as opposed to
higher or lower:

```python
w = model.shap_values(X_test, kind="width", alpha=0.1)   # (n, n_features)
```

Shapley values are linear in the value function, so the difference between two levels'
attributions is exactly the attribution of their difference. On heteroscedastic data
the feature driving the spread tops this ranking while barely appearing in the median's.

`kind="mean"` does the same for the tau-integrated point prediction.

### Averaging across rows

One wrinkle, and only if you aggregate attributions yourself. Predictions are
rearranged on the way out, which relabels a row's levels, so each row is explained
against its own reordering and `expected_value_` has one row per sample. Averaging
that across rows mixes rows that were reordered differently.

`shap_importances` already handles this — it explains the levels *before*
rearrangement, where every row is on the same footing. If you are building your own
global summary or a beeswarm plot, ask for the same thing:

```python
phi = model.shap_values(X_test, space="raw")   # one shared (n_quantiles,) baseline
```

Per-prediction explanations need none of this; the default is what you want.

## How it compares

Measured on 36 Grinsztajn regression datasets, 3 seeds, every arm sharing one
early-stopping split and round budget, apart from NGBoost's own cap described below
(`benchmarks/quantile_suite.py`). Win-loss is per
dataset. "Interval score" is the Winkler score of the 90% interval, which charges width
and miscoverage together. Fit time is each arm's median against ours (which includes the
three-candidate choice), in the same run.

| against | CRPS | interval score | crossing | fit time |
|:--|:--|:--|:--|:--|
| 19 `loss="Quantile"` models | **34W-2L** | **33W-3L** | 0.00 vs 0.16 | 1.5x ours |
| 19 LightGBM quantile boosters | **30W-6L** | **30W-6L** | 0.00 vs 0.22 | 0.9x ours |
| CatBoost `MultiQuantile` | **27W-9L** | **26W-10L** | 0.00 vs 0.06 | 6.7x ours |
| one squared-error model, fixed width | **29W-7L** | **28W-8L** | none on either side | 0.15x ours |
| NGBoost with the RoNGBa settings | **33W-3L** | **30W-6L** | none on either side | 1.4x ours |

**CatBoost's shared head no longer leads on CRPS.** We win 27 of the 36 datasets, by a
median of 0.29%, in about a seventh of its fit time. Its 90% intervals cover 83% on
average against our 90%, so on the interval score, which prices coverage and width
together, the lead is wider: 26 of 36, by a median of 2.3%.

The fixed-width row is the simplest honest baseline: fit an ordinary squared-error model,
take the quantiles of its validation residuals, and add the same offsets to every
prediction, so every row gets the same interval. It fits in about a seventh of our time
and its intervals are well calibrated. We beat it on 29 of 36 datasets on CRPS, by a
median of 1.9%, and on 28 of 36 on the interval score, because our widths follow the
data. If one width for every row suits your problem and fit time matters most, it is a
reasonable choice.

Against a stack of independent per-level models, ours or LightGBM's, the shared structure
clearly pays in accuracy, and its levels never cross. Fit time is now similar: our own 19
per-level models take 1.5 times as long, LightGBM's 19 boosters about 0.9 times.

The NGBoost row uses the settings from Ren, Sun and Wu (2019), called RoNGBa: trees of
up to 31 leaves, a learning rate of 0.04 and at most 500 rounds, with the round count
chosen on the validation rows. They beat NGBoost's stock settings on 23 of the 36
datasets and fit about four times faster. Two of its 177 fits here broke down. In each,
one training row sat about 30 standard deviations from the mean and got a leaf of its
own in the first tree, and the one or two test rows that landed in that leaf were given
a spread tens of millions of times too wide. The win counts above don't change because
of it, but it would swamp any average over datasets.

Earlier versions of this page claimed 3.0x-6.2x the speed of LightGBM and 1-3% better
pinball. Those numbers came from fixed-round fits on synthetic data
(`benchmarks/quantile_head.py`), which flatters us. The table above is the real-data
measurement, with every arm early-stopping.

Those are averages over the whole grid; a single level can trade more, because every
level shares one tree structure per round. On data whose signal takes many rounds to
resolve, the median column of the default 19-level grid has measured up to 18% worse
than a dedicated `quantiles=[0.5]` fit, with a 3-level grid recovering most of the gap
— so fit only the levels you need when per-level accuracy matters more than the full
distribution.

## Tuning

Two defaults are set for this head rather than inherited. `depth` is 6: deeper trees
place the centre of the distribution better but narrow its tails, and the default
calibration repairs the tails. Earlier releases used `depth=4` with no calibration; pass
`depth=4, conformalize=False` to reproduce them exactly. `min_child_weight` follows the
most extreme level on the grid, so a leaf estimating the 5% quantile keeps at least
about 20 rows.

When fit time matters more than the last 1.6%, `audition=False` fits a single head in
less than half the time.

`split_projection` chooses how the K gradient columns collapse into the single vector
the tree grower accepts. Leave it alone unless you are exploring: `"rotate"` measured
best, `"sum"` is blind to changes in spread, and `"gram"` measured no better than
`"rotate"`. `exact_splits=True` scores the exact gain summed across levels instead of
the projection, at the cost of K histogram channels per feature. It is not more accurate
in practice: on the 36 Grinsztajn regression datasets it scored a worse CRPS than the
default on 27, so leave it off.

`benchmarks/QUANTILE_PLAN.md` records why each of those defaults is what it is.
