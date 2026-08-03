<!-- Hand-written docs homepage. README.md is the GitHub and PyPI front page and is
     maintained separately: it carries GitHub-only material (badges, the docs link,
     the full credit list) that has its own home on this site. Change the pitch in
     one and check the other. -->

# ChimeraBoost

*pronounced kai-MEER-uh-boost*

Gradient boosting on oblivious trees, written in Python with numba kernels. On the public
benchmark suite the defaults land within noise of CatBoost at about a seventh of its
median fit time.

[![Average rank against fit-time slowdown on the public benchmark suite: ChimeraBoost sits close to CatBoost's accuracy at a fraction of its fit time.](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png)](benchmarks.md)

## Install

```bash
pip install chimeraboost
```

Python 3.9 or newer. Four dependencies: numpy, numba, scikit-learn, scipy.

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

clf = ChimeraBoostClassifier(random_state=0)
clf.fit(df, y, cat_features=["city", "device_type"])    # strings, no encoding step
proba = clf.predict_proba(X_test)

reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
preds = reg.predict(X_test)
```

There is nothing else to set. The fit early-stops on a holdout it splits off itself,
probabilities come out calibrated, and categorical columns are encoded for you.
[Getting started](getting-started.md) runs both of these end to end.

## What you get

- Regression, binary and multiclass classification, and quantile regression.
- Categorical columns passed as strings, encoded with ordered target statistics — no
  one-hot or `LabelEncoder` step, and no imputation for missing values.
- Losses: squared error, absolute error, quantile, Huber, Poisson, Gamma, Tweedie, or
  [one you write yourself](recipes.md#custom-objectives-and-metrics).
- Early stopping and probability calibration on by default.
- Exact SHAP attributions with no extra package ([regression and binary
  classification](shap.md); multiclass is not supported yet).
- A whole predictive distribution from a single fit, with non-crossing quantiles and
  conformal prediction intervals ([quantiles](quantiles.md)).
- Bagging as a first-class setting (`n_ensembles`), and a `quality` dial from 1 (fastest)
  to 5 (strongest), with 3 as the default.

It is all Python. You can read every part of it and change any of it; there is no C or
C++ build step and no compiled extension to fight.

## Where to go next

- [Getting started](getting-started.md) — install and your first models.
- [Recipes](recipes.md) — categoricals, quantiles, bagging, custom losses, persistence.
- [How it works](concepts.md) — oblivious trees, categorical encoding, linear leaves,
  calibration.
- [Parameters](parameters.md) — every option, with defaults and guidance.
- [Deployment](deployment.md) — compile cost, threads, latency, model size.
- [FAQ](faq.md) — common questions.
- [How it is benchmarked](benchmarks.md) — which suites decide what, and the chart above.

## Credit where it is due

Almost every idea in ChimeraBoost is someone else's: gradient boosting itself, oblivious
trees, ordered target statistics, histogram split finding, TreeSHAP, temperature scaling,
conformalized quantile regression, and more. This is a from-scratch Python implementation
of published work. [Where the ideas come from](attribution.md) maps it out feature by
feature, and names the short list of what actually originated here. If a credit is missing
or wrong, please open an issue.

ChimeraBoost is Apache-2.0 licensed. Source and issues:
[github.com/bbstats/chimeraboost](https://github.com/bbstats/chimeraboost).
