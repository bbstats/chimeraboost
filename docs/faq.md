# FAQ

## Does it use the GPU?

No.

## How does it compare to CatBoost, LightGBM, and XGBoost?

On the public benchmark suite the default lands within noise of CatBoost's accuracy at
about a seventh of its median fit time, and ahead of LightGBM. See
[How ChimeraBoost is benchmarked](benchmarks.md) for the chart and the method.

That chart does not include XGBoost. The comparison that does is the public TabArena
leaderboard, where all four libraries run under one protocol: the ChimeraBoost default
scores above the XGBoost and LightGBM defaults and below the CatBoost default.

For more accuracy, `n_ensembles=8` bags the model, and adding `refit_members=True` is
the strongest setting available. Both cost several times the default's fit time — see
[Bagging](recipes.md#bagging).

Things the other three have and this does not: GPU training, monotone constraints, SHAP
for multiclass, and `scipy.sparse` input.

## Do I need to one-hot encode categoricals or impute missing values?

No.

Pass your categorical columns to `fit(..., cat_features=[...])`, by integer position or
by column name. NaNs route to their own bin at fit and predict time, so no imputation
is needed. pandas nullable dtypes (`Int64`, `Float64`, `boolean`) and `pd.NA` are
accepted and treated as missing.

## How can I make inference faster?

If your serving data is already validated, skip scikit-learn's finiteness check with
`assume_finite`:

```python
import sklearn
with sklearn.config_context(assume_finite=True):
    preds = model.predict(X)        # finiteness scan skipped
```

## Why is the very first fit or predict slow?

That is numba compiling the kernels, once, cached on disk per user. Run
`chimeraboost-warmup` after installing to pay it up front, or call
`chimeraboost.warmup()` or set `CHIMERABOOST_WARMUP=1` to pay it at import time. See
[Deployment](deployment.md) for timings and patterns for short-lived workers.

## Why did it get slow again after I upgraded?

Numba stamps each cached kernel with its source file's timestamp and size, so
installing a new ChimeraBoost version invalidates the cache and the next run
recompiles. Put `chimeraboost-warmup` in the same script as your `pip install -U`.

## Why oblivious (symmetric) trees?

They make prediction extremely fast and provide strong built-in regularization, at some
cost to per-tree sharpness. The design is CatBoost's, not ours. See
[How it works](concepts.md#oblivious-trees).

## Does SHAP support multiclass?

Not yet — `shap_values` raises `NotImplementedError` for three or more classes.
Regression and binary classification are supported; for multiclass,
`feature_importances_` still gives a global ranking.

## How do I save and load a model?

A fitted estimator pickles like any scikit-learn object. Pickles are not guaranteed to
load across ChimeraBoost versions, so store the version next to the model and re-fit
after upgrading:

```python
import chimeraboost
import joblib

joblib.dump({"model": model, "version": chimeraboost.__version__}, "model.joblib")

bundle = joblib.load("model.joblib")
model = bundle["model"]
```

## What exactly does it depend on?

NumPy, numba, scikit-learn, and SciPy. pandas is not one of them, but you can still pass
pandas (or polars) DataFrames straight to `fit` and `predict` — whichever you have
installed works.

## How do I tune it?

Mostly, you don't. The defaults are benchmark-tuned, and in our experiments broad
hyperparameter search bought little that generalized. Two settings address specific
situations rather than general tuning: `n_ensembles=8` with `refit_members=True` is the
maximum-accuracy mode, at several times the fit cost, and `depth=8` to `10` suits large,
interaction-heavy regression. [Parameters](parameters.md) documents every option.

## Is the API stable?

ChimeraBoost is beta (0.x). Breaking API or behavior changes bump the minor version and
are recorded in the
[CHANGELOG](https://github.com/bbstats/chimeraboost/blob/main/CHANGELOG.md); patch
releases are fixes only. Pickled models are not guaranteed to load across versions — see
[How do I save and load a model?](#how-do-i-save-and-load-a-model) for the version stamp
to store with them.
