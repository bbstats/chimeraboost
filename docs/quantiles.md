# Predictive distributions

`ChimeraBoostQuantileRegressor` estimates a whole grid of conditional quantiles from a
single booster. One tree structure per round serves every level, and each leaf holds a
vector with one entry per level. CatBoost's `MultiQuantile` loss works the same way;
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

## Predictions never cross

The 30% quantile is never returned above the 70%. Every row is sorted on its way out, so
`np.diff(Q, axis=1) >= 0` holds exactly, including at every intermediate stage of
`staged_predict`. Sorting is not a compromise: rearranging a crossing quantile curve
never increases pinball loss at any level, for any row (Chernozhukov, Fernández-Val &
Galichon 2010), so the guarantee is free.

Independently fitted per-level models have no such property: in our benchmarks, 18 to
21% of their adjacent quantile pairs come out reversed.

Ordering does not mean a fixed width. Where the data is quiet the band comes out much
*narrower* than one built from the unconditional quantiles of `y`, because it tracks
local spread rather than a global floor.

## Interval calibration

Read the intervals with this in mind: **the raw grid runs slightly narrow.** Leaf values
are the residual quantiles of the rows in that leaf, measured on those same rows, which
is optimistic. Across our benchmark datasets a nominal 80% interval delivers about 72 to
76% coverage. Pinball loss is what the model optimizes and it is good — better than one
dedicated LightGBM booster per level — but a raw interval is not a coverage guarantee.

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
| `interval_coverage` | Do the intervals hold what they claim? Coverage and width. |
| `crossing_rate` | What fraction of adjacent pairs is out of order? |

`crps` is the mean pinball loss over the grid. The textbook CRPS is twice that, but the
factor is constant, so comparisons are unaffected and this convention matches the
early-stopping metric.

`predict(kind="mean")` integrates the quantile function over tau, by the trapezoid rule
across the grid with the edge levels extended flat to 0 and 1. That flat extension
assumes nothing about tails the model never estimated.

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

`split_projection` controls how a split's gain is scored across the quantile levels.
Leave it at the default `"rotate"`: `"sum"` is blind to changes in spread, and `"gram"`
scores no better. `exact_splits=True` scores the gain exactly across every level instead
of through a projection — more faithful, but fits get much slower and use much more
memory, both growing with the number of levels. It is a reference setting, not something
to switch on for routine use.
