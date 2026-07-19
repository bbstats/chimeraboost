# FAQ

## Does it use the GPU?

No.

## How does it compare to CatBoost, LightGBM, and XGBoost?

On defaults: roughly around LightGBM/XGBoost or better, and significantly faster than CatBoost.
Setting n_ensembles=8 bags the model — on the Grinsztajn benchmark suite this is the
strongest model on every accuracy column, ahead of CatBoost at well under half its fit
cost on high-cardinality data. Note that since TabArena uses ensembling, this improvement
does not show up on those leaderboard results.

## Do I need to one-hot encode categoricals or impute missing values?

No.

Pass your categorical columns to `fit(..., cat_features=[...])`, by integer position or by column name.
NaNs route to a dedicated bin at fit and predict time, so no imputation is needed. pandas nullable
dtypes (`Int64`/`Float64`/`boolean`) and their `pd.NA` are accepted and treated as missing.


## How can I make inference faster?

If you have already validated your serving data and want to skip it,
use scikit's 'assume_finite'.

```python
import sklearn
with sklearn.config_context(assume_finite=True):
    preds = model.predict(X)        # finiteness scan skipped
```


## Why is the very first fit or predict slow?

That is numba compiling the kernels (one-time per process, disk-cached per
user). Call `chimeraboost.warmup()` — or set `CHIMERABOOST_WARMUP=1` — to
pay it at import time instead of on the first call. See
[Deployment](deployment.md) for numbers and short-lived-worker patterns.

## Why oblivious (symmetric) trees?

They make prediction extremely fast and provide strong built-in regularization, at some
cost to per-tree sharpness. See [How it works](concepts.md#oblivious-trees).

## Does SHAP support multiclass?

Not yet.

## How do I save and load a model?

A fitted estimator pickles like any scikit-learn object:

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

## What exactly does it depend on?

NumPy, numba, scikit-learn, SciPy, and pandas.

## How do I tune it?

Mostly, you don't: the defaults are benchmark-tuned, and in our experiments broad
hyperparameter search bought little that generalized. Two settings address specific
situations rather than general tuning: `n_ensembles=8` is the benchmarked
maximum-accuracy mode (at several times the fit cost), and `depth=8–10` suits large,
interaction-heavy regression. [Parameters](parameters.md) documents every knob.

## Is the API stable?

ChimeraBoost is beta (0.x). Breaking API or behavior changes bump the minor version
and are recorded in the
[CHANGELOG](https://github.com/bbstats/chimeraboost/blob/main/CHANGELOG.md); patch
releases are fixes only. Pickled models are not guaranteed to load across versions —
store the version next to the model and re-fit after upgrading.
