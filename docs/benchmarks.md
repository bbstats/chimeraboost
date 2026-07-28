# How ChimeraBoost is benchmarked

Defaults are tuned against benchmarks, so it is worth knowing which ones do what. We run
five suites, each with a single job, and only one of them can decide whether a change
ships.

| suite | what it is | what it is for |
|---|---|---|
| SynthGen | synthetic generators | a first screen: does the idea do anything at all? |
| Grinsztajn + high-card | real datasets, one low-cardinality and one high | the decision. A change ships or dies here. |
| PMLB | curated sets with tune and holdout folds | hyperparameter tuning only |
| Public | independently audited larger datasets | published evidence, including the chart below |
| TabArena | the community leaderboard | a sealed read, reported and never tuned against |

Keeping these apart is the point. A suite you tune against cannot also tell you whether
you have overfit it, so the two suites that pick the defaults are not the ones we quote,
and the leaderboard we quote never feeds back into the code.

## The published chart

![strength vs speed](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png)

Average rank across the public suite against median fit time, both measured only against
CatBoost and LightGBM. The honest summary is that the default sits within noise of
CatBoost at under a third of its median fit time.

Datasets are picked on data properties alone: row counts, cardinality, missingness, task
type. No benchmark result is allowed to influence which datasets are in a suite, or it
would be cherry-picked from birth.

Reproduce the chart with:

```
python benchmarks/run_benchmarks.py --public --seeds 3 --save \
    --models ChimeraBoost ChimeraBoostEns5 CatBoost LightGBM
python benchmarks/make_public_pareto.py benchmarks/results/<stamp>.json
```

`benchmarks/PUBLIC_PLAN.md` records every dataset and why it is in the suite.
