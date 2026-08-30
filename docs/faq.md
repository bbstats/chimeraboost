# FAQ

## Does it use the GPU?

No.

## How does it compare to CatBoost, LightGBM, and XGBoost?

On defaults, it scores around LightGBM and XGBoost or better, and it is much faster
than CatBoost. Setting `n_ensembles=8` bags the model, which is the strongest setting
short of adding `refit_members=True`, and beats CatBoost on accuracy at well under half its fit cost on
high-cardinality categorical data.

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

Yes. `shap_values` returns `(n_samples, n_features, n_classes)`, attributing each
class's raw softmax score. Quantile models are covered too, with a channel per
level. See [SHAP explanations](shap.md).

## How do I save and load a model?

A fitted estimator pickles like any scikit-learn object:

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

## What exactly does it depend on?

NumPy, numba, scikit-learn, and SciPy. pandas is not required. DataFrames are consumed
through their own conversion methods, so passing them works whenever you have pandas
(or polars) installed yourself.

## How do I tune it?

Mostly, you don't. The defaults are benchmark-tuned, and in our experiments broad
hyperparameter search bought little that generalized. Two settings address specific
situations rather than general tuning: `n_ensembles=8` with `refit_members=True` is the
maximum-accuracy mode, at several times the fit cost, and `depth=8` to `10` suits large, interaction-heavy
regression. [Parameters](parameters.md) documents every option.

## Is the API stable?

ChimeraBoost is beta (0.x). Breaking API or behavior changes bump the minor version and
are recorded in the
[CHANGELOG](https://github.com/bbstats/chimeraboost/blob/main/CHANGELOG.md); patch
releases are fixes only. Pickled models are not guaranteed to load across versions, so
store the version next to the model and re-fit after upgrading.
