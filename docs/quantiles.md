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
```

`kind="cdf"` inverts the grid, and `kind="sample"` draws from it by inverse transform —
useful for feeding a downstream simulation. Both interpolate between fitted levels and
clamp outside the outermost ones, because a finite grid says nothing about the tails
beyond it. `kind="interval"` still refuses levels you did not fit: reading a fitted
curve at a point is a different thing from claiming a level was fitted when it was not.

## Predictions never cross

The 30% quantile is never returned above the 70%. Every row is sorted on its way out, so
`np.diff(Q, axis=1) >= 0` holds exactly, including at every intermediate stage of
`staged_predict`. Sorting is not a compromise: rearranging a crossing quantile curve
never increases pinball loss at any level, for any row (Chernozhukov, Fernández-Val &
Galichon 2010), so the guarantee is free.

Independently fitted per-level models have no such property. On the benchmark in
`benchmarks/quantile_head.py`, 18 to 21% of adjacent quantile pairs come out reversed.

The band is free to be much *narrower* than the pooled one where the data is quiet — it
tracks the local spread rather than a global floor.

## Interval calibration

Read the intervals with this in mind: **the raw grid runs slightly narrow.** Leaf values
are the residual quantiles of the rows in that leaf, measured on those same rows, which
is optimistic. On the datasets in `benchmarks/probe_quantile_band.py` a nominal 80%
interval delivers about 72 to 76% coverage. Pinball loss is what the model optimizes and
it is good — better than one dedicated LightGBM booster per level — but a raw interval
is not a coverage guarantee.

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

`crps` is the mean pinball loss over the grid. The textbook CRPS is twice that, but the
factor is constant, so comparisons are unaffected and this convention matches the
early-stopping metric. Pass `convention="full"` when comparing against another library —
`properscoring` and `scoringrules` both report the doubled value.

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

### What the sort does to the attribution

Predictions are rearranged on the way out (see above), and that rearrangement is a
per-row permutation. It cannot be folded into the Shapley game: sorting acts on the
summed forest score, so pushing it inside the coalition enumeration would make the game
range over every input feature instead of the handful one tree touches — the exact
blow-up oblivious trees avoid. So there are two honest views, and `space` picks one:

| `space` | explains | baseline |
|:--|:--|:--|
| `"raw"` (default) | the per-level scores the booster accumulated, before rearrangement | one `(n_quantiles,)` vector |
| `"delivered"` | what `predict` returns, rearrangement and conformalization included | per-row `(n, n_quantiles)` |

Aggregate in `"raw"` — global importances, beeswarm plots — because every row is then
measuring the same game. Use `"delivered"` to explain a single prediction in the levels
you actually read off.

The two agree exactly whenever a row's raw grid was already ordered. On a 3-level grid
that is essentially every row; on the default 19-level grid roughly 40% of rows cross at
least one adjacent pair, so the choice is not cosmetic there.

## What it costs

The split search runs once per round instead of once per level, so the saving grows with
how wide the data is: roughly 3.0x the fit speed of 19 independent LightGBM quantile
boosters at 5 features, 3.6x at 32, and 6.2x at 128. Accuracy is not traded for it —
pinball loss comes out 1 to 3% *better* than those per-level models at every width
measured, with the margin widening on wide data.

That is an average over the whole grid; a single level can trade more, because every
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
