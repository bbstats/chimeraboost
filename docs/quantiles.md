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

Read the intervals with this in mind: **the raw grid runs slightly narrow.** Leaf values
are the residual quantiles of the rows in that leaf, measured on those same rows, which
is optimistic. Across the 36 datasets in `benchmarks/quantile_suite.py` a nominal 80%
interval delivers 77% coverage on average, and a nominal 90% delivers 87%. That is
closer to nominal than either LightGBM per-level (72% and 83%) or CatBoost
`MultiQuantile` (73% and 83%) manages — but it is still narrow, and a raw interval is
not a coverage guarantee.

`conformalize=True` turns it into one:

```python
model = ChimeraBoostQuantileRegressor(conformalize=True).fit(X, y)
print(model.conformal_scale_)      # one factor per level; above 1 widened the fit
```

This holds out `calibration_fraction` of the rows **before** the early-stopping split,
so that fold influences neither the fit nor the stopping point, then rescales each level
about the predicted median by a conformal factor (Romano, Patterson & Candès 2019). On
exchangeable data this gives distribution-free marginal coverage. Measured coverage lands
within 2.7 percentage points of nominal at n = 10,000, erring on the wide side — conformal
prediction is conservative by construction, so over-coverage is the expected direction.

Use it whenever you need the interval to mean what it says. It costs one extra held-out
fold and no extra fitting.

It raises rather than guessing when the calibration fold is too small to certify the
levels you asked for. A 90% interval needs at least 9 calibration rows, and a 99% one
needs 99.

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

Measured on 36 Grinsztajn regression datasets, 3 seeds, all four arms sharing one
early-stopping split and budget (`benchmarks/quantile_suite.py`). Win-loss is per
dataset; "interval score" is the Winkler score, which charges width and miscoverage
together.

| against | CRPS | interval score | crossing | median fit time |
|:--|:--|:--|:--|:--|
| 19 `loss="Quantile"` models | **32W-4L** | **34W-2L** | 0.00 vs 0.16 | **3.4x faster** |
| 19 LightGBM quantile boosters | 23W-13L (a tie) | **31W-5L** | 0.00 vs 0.22 | **1.5x faster** |
| CatBoost `MultiQuantile` | 7W-**29L** | **25W-11L** | 0.00 vs 0.06 | **8.3x faster** |

Read that honestly. **CatBoost's shared head is sharper than ours on CRPS** — it wins 29
of 36 datasets — and that is a real deficit, not a rounding error. It costs a median 8.3x
our fit time to get there, and its intervals are much worse calibrated (its worst
coverage error is 0.64 against a nominal 0.90, ours 0.10), so on the interval score,
which prices coverage and width together, we come out ahead.

Against a stack of independent per-level models — ours or LightGBM's — the shared
structure clearly pays: better or equal accuracy, faster, and the only arm here whose
levels never cross.

Earlier versions of this page claimed 3.0x-6.2x the speed of LightGBM and 1-3% better
pinball. Those numbers came from fixed-round fits on synthetic data
(`benchmarks/quantile_head.py`), which flatters us; on real data with both sides
early-stopping, the accuracy is a tie and the speed edge is 1.5x.

Those are averages over the whole grid; a single level can trade more, because every
level shares one tree structure per round. On data whose signal takes many rounds to
resolve, the median column of the default 19-level grid has measured up to 18% worse
than a dedicated `quantiles=[0.5]` fit, with a 3-level grid recovering most of the gap
— so fit only the levels you need when per-level accuracy matters more than the full
distribution.

## Tuning

Two defaults are set for this head rather than inherited. `depth` is 4, because deep
leaves overfit tail quantiles, and `min_child_weight` follows the most extreme level on
the grid, so a leaf estimating the 5% quantile keeps at least about 20 rows.

`split_projection` chooses how the K gradient columns collapse into the single vector
the tree grower accepts. Leave it alone unless you are exploring: `"rotate"` measured
best, `"sum"` is blind to changes in spread, and `"gram"` measured no better than
`"rotate"`. `exact_splits=True` scores the exact gain summed across levels, which is
slightly more accurate at the cost of K histogram channels per feature.

`benchmarks/QUANTILE_PLAN.md` records why each of those defaults is what it is.
