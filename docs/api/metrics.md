# metrics

Scoring for a fitted regressor or classifier: error, skill against a no-skill forecast,
and — for classification — how far the probabilities are from calibrated. This is what
`ChimeraBoostRegressor.report` and `ChimeraBoostClassifier.report` return.

Definitions match `benchmarks/run_benchmarks.py`, so a number here is comparable with
the project's own [benchmark tables](../benchmarks.md). For a predicted quantile grid,
see [`quantile_metrics`](quantile-metrics.md) instead.

::: chimeraboost.metrics
    options:
      show_root_heading: false
      show_root_toc_entry: false
