# How ChimeraBoost is benchmarked

Defaults are tuned against benchmarks, so it is worth knowing which ones do what. We run
five suites, each with a single job, and only one of them can decide whether a change
ships.

| suite | what it is | what it is for |
|---|---|---|
| SynthGen | synthetic generators | a first screen: does the idea do anything at all? |
| Grinsztajn + high-card | real datasets, one low-cardinality and one high — the Grinsztajn et al. 2022 tabular benchmark plus a frozen high-cardinality set | the decision. A change ships or dies here. |
| PMLB | the Penn Machine Learning Benchmarks, split into tune and holdout folds | hyperparameter tuning only |
| Public | independently audited larger datasets | published evidence, including the chart below |
| TabArena | the community leaderboard | a sealed read, reported and never tuned against |

Keeping these apart is the point. A suite you tune against cannot also tell you whether
you have overfit it, so the two suites that pick the defaults are not the ones we quote,
and the leaderboard we quote never feeds back into the code.

## The published chart

![strength vs speed](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png)

Average rank across the public suite against median fit time, both measured only against
CatBoost and LightGBM. The honest summary is that the default sits within noise of
CatBoost at about a seventh of its median fit time: average ranks of 1.90 and 1.88 with
overlapping intervals, at 7.1x versus 53.1x slowdown.

Datasets are picked on data properties alone: row counts, cardinality, missingness, task
type. No benchmark result is allowed to influence which datasets are in a suite, or it
would be cherry-picked from birth.

## Running them yourself

Run either one, then open an issue or PR with the JSON.

```
pip install -e ".[bench,competitors]"

python benchmarks/run_benchmarks.py --synth --seeds 3 --save     # quick, synthetic
python benchmarks/run_benchmarks.py --decide --seeds 3 --save    # slower, 103 real datasets
```

Each writes `benchmarks/results/<timestamp>.json`:

```json
{
  "provenance": {"chimeraboost": "0.30.0", "platform": "Linux-6.1", "cpu_count": 12,
                 "libraries": {"catboost": "1.2.10", "lightgbm": "4.6.0"}},
  "records": [
    {"dataset": "diabetes", "model": "ChimeraBoost", "seed": 0,
     "metrics": {"primary": -59.82, "rmse": 59.82}, "fit_time": 0.23}
  ]
}
```

That file is all we need. The first run downloads a few gigabytes into
`benchmarks/data_cache/`; set `BENCH_DATA_HOME` to put it elsewhere.
