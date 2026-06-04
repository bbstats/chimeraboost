# FAQ

## Does it use the GPU?

No.

## How does it compare to CatBoost, LightGBM, and XGBoost?

On defaults: roughly around LightGBM/XGBoost or better, and significantly faster than CatBoost.
Setting n_ensembles=10 automatically ensembles the model through bagging, and gets accuracy
and speed roughly = to CatBoost. Note that since TabArena uses ensembling, this improvement
does not show up on those pareto chart results.

## Do I need to one-hot encode categoricals or impute missing values?

No.

Pass your categorical columns to `fit(..., cat_features=[...])`, by integer position or by column name.
NaNs route to a dedicated bin at fit and predict time, so no imputation is needed.


## How can I make inference faster?

If you have already validated your serving data and want to skip it,
use scikit's 'assume_finite'.

```python
import sklearn
with sklearn.config_context(assume_finite=True):
    preds = model.predict(X)        # finiteness scan skipped
```


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

First reach for `depth` (raise to 8–10 for large, interaction-heavy regression) or
`n_ensembles` (variance reduction) first. See [Parameters](parameters.md).
