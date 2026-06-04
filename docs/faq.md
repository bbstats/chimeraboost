# FAQ

## Does it use the GPU?

No. ChimeraBoost is CPU-only, compiled with numba and parallelized across cores. There
is no CUDA dependency and no GPU build.

## How does it compare to CatBoost, LightGBM, and XGBoost?

It targets the same problems with a different trade-off: roughly CatBoost-class accuracy
at a fraction of the training time, and ahead of XGBoost and LightGBM defaults on both
accuracy and speed on TabArena-Lite. The design borrows oblivious trees and ordered
target statistics from CatBoost and histogram split-finding from LightGBM, implemented
in pure Python. It is not trying to win the >10M-row distributed regime those libraries
also serve.

## Do I need to one-hot encode categoricals or impute missing values?

No to both. Pass categorical column indices to `fit(..., cat_features=[...])` and they
are handled with ordered target statistics. NaNs route to a dedicated bin at fit and
predict time, so no imputation is needed.

## Does column order matter when I predict on a DataFrame?

Yes. If you fit on a DataFrame (pandas or polars), the column **names and order** are
recorded, and `predict` raises if the columns you pass don't match — reordered or renamed
columns would otherwise produce silently-wrong predictions, since the model consumes
columns positionally. Pass the same columns in the same order (or plain arrays on both
sides). This mirrors scikit-learn's behavior.

## What input does it reject, and how?

Malformed input fails fast with a clear message rather than a cryptic crash or a silently
broken model. A non-numeric column passed without `cat_features` names the offending
column and points you at `cat_features`; out-of-range `cat_features` indices, bad
`eval_set` shapes, non-finite or negative `sample_weight`, and out-of-range
hyperparameters (e.g. negative `learning_rate`, `depth` outside `[1, 16]`) all raise
`ValueError`. `inf` is rejected at both fit and predict; `NaN` is accepted as missing.

## How do I make inference as fast as possible?

Prediction is already fast (oblivious trees, a single fused forest pass). The one
per-row check on the hot path is a finiteness scan that rejects `inf`. If you have
already validated your serving data and want to skip it, use scikit-learn's global
config — the same switch its own estimators honor:

```python
import sklearn
with sklearn.config_context(assume_finite=True):
    preds = model.predict(X)        # finiteness scan skipped
```

## Is it deterministic?

Yes, given a fixed `random_state` and `thread_count`. With multiple threads, the order
of floating-point reductions can vary across runs or machines, producing tiny numerical
differences; pin `thread_count` for bit-stable results.

## How large a dataset can it handle?

It is built for in-memory, cache-resident data and is comfortable into the hundreds of
thousands to low millions of rows. It is not a distributed or out-of-core system.

## Can I get probability estimates?

Yes — `predict_proba` returns probabilities that are temperature-scaled on the
validation split for calibration. See [calibration](concepts.md#probability-calibration).

## Why oblivious (symmetric) trees?

They make prediction extremely fast and provide strong built-in regularization, at some
cost to per-tree sharpness. See [How it works](concepts.md#oblivious-trees).

## Does SHAP support multiclass?

Not yet. `shap_values` works for regression and binary classification; multiclass raises
`NotImplementedError`.

## How do I save and load a model?

A fitted estimator pickles like any scikit-learn object:

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

## What exactly does it depend on?

NumPy, numba, scikit-learn, SciPy, and pandas. No C or C++ extensions, and no build
toolchain — the whole library is Python.

## How do I tune it?

Most problems need no tuning. When they do, reach for `depth` (raise to 8–10 for large,
interaction-heavy regression) and `n_ensembles` (variance reduction) first. See
[Parameters](parameters.md).
