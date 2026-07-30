# Predictive distributions

`ChimeraBoostQuantileRegressor` estimates a whole grid of conditional quantiles from a
single booster. One tree structure per round serves every level, and each leaf holds a
K-vector with one entry per level.

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

The 30% quantile is never returned above the 70%. This is built into the model rather
than repaired afterwards: it starts from the sorted global quantiles, and every leaf
vector is projected onto a set of increments that cannot reorder anything. So
`np.diff(Q, axis=1) >= 0` holds exactly, including at every intermediate stage of
`staged_predict`.

Independently fitted per-level models have no such property. On the benchmark in
`benchmarks/quantile_head.py`, 18 to 21% of adjacent quantile pairs come out reversed.

Intervals can still be *narrower* than the pooled one where the data is quiet, so that
is not given up to get the guarantee.

## Interval calibration

Boosting under-disperses quantiles. Each round's step is scaled by the learning rate, so
the grid never fully contracts and intervals come out too wide. `conformalize=True`
fixes it:

```python
model = ChimeraBoostQuantileRegressor(conformalize=True).fit(X, y)
print(model.conformal_scale_)      # one factor per level; below 1 means the fit was too wide
```

This holds out `calibration_fraction` of the rows **before** the early-stopping split,
so that fold influences neither the fit nor the stopping point, then rescales each level
about the predicted median by a conformal factor (Romano, Patterson & Candès 2019). On
exchangeable data this gives distribution-free marginal coverage. The worst measured
error against nominal is 0.7 percentage points at n = 10,000.

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
how wide the data is: roughly 3.4x the fit speed of 19 independent boosters at 5
features, 4.8x at 32, and 7.8x at 128. Accuracy stays within 3% of per-level models on
pinball loss throughout, and comes out better on wide data.

That 3% is an average over the whole grid; a single level can trade more, because every
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
