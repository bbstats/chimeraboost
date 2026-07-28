# Predictive distributions

`ChimeraBoostQuantileRegressor` estimates a whole grid of conditional
quantiles from a single booster. One tree structure per round serves every
level; each leaf holds a K-vector, one entry per level.

```python
from chimeraboost import ChimeraBoostQuantileRegressor

model = ChimeraBoostQuantileRegressor().fit(X_train, y_train)

Q = model.predict(X_test)                            # (n_samples, 19)
lo, hi = model.predict(X_test, kind="interval", alpha=0.1).T
mean = model.predict(X_test, kind="mean")
```

The default grid is 0.05, 0.10, ... 0.95. Pass your own with
`quantiles=[...]` — ascending, unique, strictly inside (0, 1). Column `k` of
`predict` is level `k`.

## Predictions never cross

The 30% quantile is never returned above the 70%. This is structural, not a
repair applied afterwards: the model starts from the sorted global quantiles
and every leaf vector is projected onto a set of increments that cannot
reorder anything, so `np.diff(Q, axis=1) >= 0` holds exactly — including at
every intermediate stage of `staged_predict`.

Independently fitted per-level models have no such property. On the benchmark
in `benchmarks/quantile_head.py`, 18–21% of adjacent quantile pairs come out
reversed.

Intervals can still be *narrower* than the pooled one where the data is
quiet — that is not given up to get the guarantee.

## Interval calibration

Boosting under-disperses quantiles: each round's step is scaled by the
learning rate, so the grid never fully contracts and intervals come out too
wide. `conformalize=True` fixes it:

```python
model = ChimeraBoostQuantileRegressor(conformalize=True).fit(X, y)
```

This holds out `calibration_fraction` of the rows **before** the
early-stopping split, so that fold influences neither the fit nor the
stopping point, then rescales each level about the predicted median by a
conformal factor (Romano, Patterson & Candès 2019). On exchangeable data this
gives distribution-free marginal coverage. Measured worst error against
nominal: 0.7 percentage points at n = 10 000.

It raises rather than guessing if the calibration fold is too small to certify
the levels you asked for — a 90% interval needs at least 9 calibration rows,
and a 99% one needs 99.

## Scoring

`chimeraboost.quantile_metrics` scores a predicted grid.

```python
from chimeraboost import quantile_metrics as qm

print(qm.format_report(model.report(X_test, y_test)))
```

| function | answers |
|:--|:--|
| `pinball_loss` | is each level in the right place? (one value per level) |
| `crps` | is the distribution as a whole right? |
| `interval_coverage` | do the intervals hold what they claim — coverage *and* width |
| `crossing_rate` | fraction of adjacent pairs out of order |

`crps` is the mean pinball loss over the grid. The textbook CRPS is twice
that; the factor is constant, so comparisons are unaffected, and this
convention matches the early-stopping metric.

`predict(kind="mean")` integrates the quantile function over tau: trapezoid
across the grid, with the edge levels extended flat to 0 and 1. That flat
extension assumes nothing about tails the model never estimated.

## What it costs

The split search runs once per round instead of once per level, so the saving
grows with how wide the data is — roughly 3.4× the fit speed of 19
independent boosters at 5 features, 4.8× at 32, and 7.8× at 128. Accuracy is
within 3% of per-level models on pinball loss throughout, and better on wide
data.

## Tuning

Defaults are set for the head, not inherited: `depth` is 4 (deep leaves
overfit tail quantiles) and `min_child_weight` follows the most extreme level
on the grid, so a leaf estimating the 5% quantile keeps at least ~20 rows.

`split_projection` chooses how the K gradient columns collapse into the single
vector the tree grower accepts. Leave it alone unless you are exploring:
`"rotate"` measured best, `"sum"` is blind to changes in spread, and `"gram"`
measured no better than `"rotate"`. `exact_splits=True` scores the exact
summed-across-level gain — slightly more accurate, at K histogram channels per
feature.

`chimeraboost/benchmarks/QUANTILE_PLAN.md` records why each of those defaults
is what it is.
