# API reference

The full public API is seven names, all importable from the top-level package:

```python
from chimeraboost import (
    ChimeraBoostRegressor,
    ChimeraBoostClassifier,
    ChimeraBoostQuantileRegressor,
    CustomObjective,
    metrics,
    quantile_metrics,
    warmup,
)
```

| | What it is |
|---|---|
| [`ChimeraBoostRegressor`](regressor.md) | Gradient boosted oblivious trees for regression. Squared-error, absolute-error, quantile, Huber, and the log-link losses. |
| [`ChimeraBoostClassifier`](classifier.md) | Gradient boosted oblivious trees for classification. Binary and multiclass, with calibrated probabilities. |
| [`ChimeraBoostQuantileRegressor`](quantile-regressor.md) | A whole grid of conditional quantiles from one booster, with levels that cannot cross. |
| [`CustomObjective`](custom-objective.md) | Base class for writing your own regression loss. |
| [`metrics`](metrics.md) | Scoring a fitted regressor or classifier: error, skill, calibration. Behind `model.report()`. |
| [`quantile_metrics`](quantile-metrics.md) | Scoring a predicted quantile grid: pinball loss, CRPS, coverage, interval score, PIT. |
| [`warmup`](warmup.md) | Pre-compile the numba kernels so the first `fit` or `predict` is not slow. |

All three estimators are scikit-learn compatible: `fit`, then `predict` or
`predict_proba`. For worked examples see [Recipes](../recipes.md), and for defaults and
guidance on every option see [Parameters](../parameters.md).
