# Recipes

Imports used throughout:

```python
import numpy as np
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
```

## Regression and classification

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
y_pred = reg.predict(X_test)

clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
labels = clf.predict(X_test)            # original label values
proba = clf.predict_proba(X_test)       # columns follow clf.classes_
```

A plain `fit(X, y)` early-stops on an internal holdout — see [Early stopping](#early-stopping).

## Categorical features

Pass your categoricals as `cat_features`, by integer position or — for a DataFrame — by
column name (or a mix of both). They are encoded with ordered target statistics
(CatBoost-style), so there is no one-hot or `LabelEncoder` step. Categorical columns can be
strings or objects; the rest of the matrix stays numeric.

```python
# columns 0 and 3 are categorical (e.g. "city", "device_type")
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X, y, cat_features=[0, 3])

# equivalently, by name when X is a DataFrame
clf.fit(df, y, cat_features=["city", "device_type"])
```

`cat_combinations` adds all pairwise category-by-category features. They help when the
target depends on categorical interactions but can crowd out numerics on mixed data, so
the default (`None`) turns them on automatically only when the data is entirely
categorical. Force them with `cat_combinations=True` (e.g. on mixed data where you know
the interactions matter) or disable with `False`.

## Missing values

NaNs route to a dedicated histogram bin — no imputation needed. This works for both
numeric and categorical columns, at fit and at predict time.

```python
X[mask] = np.nan
reg = ChimeraBoostRegressor(random_state=0).fit(X, y)   # handled directly
```

## Quantile regression

Set `loss="Quantile"` and the level `alpha`. For a prediction interval, fit one model
per quantile:

```python
lo = ChimeraBoostRegressor(loss="Quantile", alpha=0.05, random_state=0).fit(X_train, y_train)
md = ChimeraBoostRegressor(loss="Quantile", alpha=0.50, random_state=0).fit(X_train, y_train)
hi = ChimeraBoostRegressor(loss="Quantile", alpha=0.95, random_state=0).fit(X_train, y_train)

lower, median, upper = lo.predict(X_test), md.predict(X_test), hi.predict(X_test)
```

`loss="MAE"` gives median regression; `loss="RMSE"` (default) is squared error.

Quantile models default to a shallower tree (`depth=4`) than the squared-error
default (`depth=6`): an extreme conditional quantile is estimated from the points in
each leaf, so deep, sparse leaves overfit the tails and the predicted quantiles
collapse toward the median on held-out data. Predictions also include a
split-conformal correction (`quantile_offset_`) fitted on the early-stopping
validation split, which restores near-nominal marginal coverage at the tails.
With `early_stopping=False` and no `eval_set` there is no split to calibrate on,
and the raw (typically under-dispersed) quantiles are returned.

## Counts, positive targets, zero-inflated targets

The log-link losses keep predictions positive and match the noise model:

```python
counts = ChimeraBoostRegressor(loss="Poisson").fit(X, y_counts)        # y >= 0
costs = ChimeraBoostRegressor(loss="Gamma").fit(X, y_positive)         # y > 0
claims = ChimeraBoostRegressor(loss="Tweedie",
                               tweedie_variance_power=1.5).fit(X, y)   # y >= 0, exact zeros
```

`loss="Huber"` (transition `delta`, in y units) is squared error that tolerates
outliers.

## Custom objectives and metrics

Subclass `CustomObjective` with the gradient/hessian of your loss on the raw
score; pass an instance as `loss`. `eval_metric` swaps the early-stopping
metric on either estimator:

```python
from chimeraboost import ChimeraBoostRegressor, CustomObjective

class LogCosh(CustomObjective):          # smooth MAE
    def grad_hess(self, y, raw):
        t = np.tanh(raw - y)
        return t, 1.0 - t**2 + 1e-6
    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.logaddexp(raw - y, y - raw) - np.log(2),
                                weights=sample_weight))

model = ChimeraBoostRegressor(loss=LogCosh()).fit(X_train, y_train)

def mae(y_true, y_pred):                 # early-stop on MAE instead of RMSE
    return float(np.mean(np.abs(y_true - y_pred)))

model = ChimeraBoostRegressor(eval_metric=mae).fit(X_train, y_train)
```

A metric where larger is better declares it: `mae.greater_is_better = True`-style
attribute (then `validation_history_` records negated values). Define custom
objectives at module level — bagged members fit in worker processes and must
pickle the loss.

## Multiclass classification

No configuration needed — the classifier switches to softmax when it sees 3 or more
classes, and `classes_` preserves your original labels.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X, y)   # 3+ classes
proba = clf.predict_proba(X_test)        # shape (n_samples, n_classes)
```

`linear_leaves` and `shap_values` are binary/regression only; multiclass uses constant
leaves and raises `NotImplementedError` from `shap_values`.

## Sample weights

```python
w = np.where(y_train == 1, 5.0, 1.0)     # upweight the positive class
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, sample_weight=w)
```

Weights are normalized to mean 1 internally and apply to training only; the
early-stopping metric stays unweighted.

## Bagging

`n_ensembles` trains that many models on random row samples and averages them —
regressors average predictions, classifiers soft-vote calibrated probabilities.
Each member trains on `max_samples` (default 0.8) of the rows drawn without
replacement — measurably stronger and faster than the classic bootstrap —
and early-stops on its own unsampled rows.

```python
reg = ChimeraBoostRegressor(n_ensembles=8, random_state=0).fit(X_train, y_train)
```

Recommended size is `n_ensembles=8` (benchmarked stronger than 5 at similar
cost). Avoid `n_ensembles=2`: two members measure worse than one model.

Inside a bag, parameters left on auto resolve to tuned member defaults —
currently `learning_rate=0.15` and `colsample=0.85` — because averaging
tolerates coarser, cheaper members. The fit warns once when this happens
(a filterable `UserWarning`), `member_params_` records what was applied,
and passing explicit values disables it.

Members fit in parallel worker processes by default, splitting the thread
budget so a bagged fit uses the same cores a single fit would; pass
`ensemble_n_jobs=1` to fit them sequentially instead.

## Full-data refit

`refit_full=True` is the cheaper accuracy lever: after early stopping has
chosen the tree budget on the automatic validation split, the winning
configuration is retrained on 100% of the rows at that budget, so the final
model does not pay the 20% holdout data tax. About 2x fit time for a broad
accuracy gain — largest on small or high-signal data. It composes with
everything above but is a no-op inside bagged members (their held-out rows
already serve as an external eval set).

```python
reg = ChimeraBoostRegressor(refit_full=True, random_state=0).fit(X_train, y_train)
```

`feature_importances_` and `shap_values` average across the bag automatically.

## Early stopping

Early stopping is on by default. With no `eval_set`, the estimator holds out a
validation split (`validation_fraction=0.2`, stratified for classifiers), stops after a
plateau, and keeps the best round.

```python
# default: automatic internal holdout
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
print(m.best_iteration_)

# explicit validation set (overrides the internal split)
m = ChimeraBoostRegressor(random_state=0)
m.fit(X_train, y_train, eval_set=(X_val, y_val))

# grouped split: keep each group entirely in train or validation
m.fit(X_train, y_train, groups=subject_ids)

# fixed number of trees, no stopping
m = ChimeraBoostRegressor(early_stopping=False, n_estimators=500, random_state=0)
m.fit(X_train, y_train)
```

After fitting, `validation_history_` holds the per-round validation loss, and the
regressor's `staged_predict(X)` yields the prediction after each successive tree
(not defined for a bagged ensemble).

## Calibrated probabilities

`predict_proba` is temperature-scaled on the validation split to minimize log loss. The
scaling is monotonic, so `predict()`, AUC, and accuracy are unchanged while the
probabilities themselves are better calibrated.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
proba = clf.predict_proba(X_test)        # already calibrated
print(clf.temperature_)                  # > 1 means raw scores were over-confident
```

## Feature importance

`feature_importances_` is total split gain per input column, normalized to sum to 1
(averaged across the bag when `n_ensembles > 1`).

```python
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
for j in np.argsort(m.feature_importances_)[::-1][:5]:
    print(f"feature {j}: {m.feature_importances_[j]:.3f}")
```

Gain reflects what the trees split on, not how much each feature moves a given
prediction, and it ignores the per-leaf linear models. For a faithful decomposition of
the output, use [SHAP](shap.md).

## Cross-validation and hyperparameter search

The estimators are standard scikit-learn objects:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

scores = cross_val_score(
    ChimeraBoostRegressor(random_state=0), X, y, cv=5,
    scoring="neg_root_mean_squared_error",
)

search = GridSearchCV(
    ChimeraBoostRegressor(random_state=0),
    {"depth": [6, 8, 10], "l2_leaf_reg": [1.0, 3.0]},
    cv=5,
)
search.fit(X, y)
print(search.best_params_)
```

To pass `cat_features` through a search, set it on the constructor —
`ChimeraBoostClassifier(cat_features=["city", "brand"])` — so the meta-estimator
carries it (a fit-only kwarg can't be).

## Save and load a model

A fitted estimator pickles like any scikit-learn object:

```python
import joblib

joblib.dump(reg, "model.joblib")
reg = joblib.load("model.joblib")
```

## Interaction-heavy regression

The default `depth=6` is conservative to protect small data. On large, interaction-heavy
problems, raise `depth` to give each tree more interaction capacity:

```python
reg = ChimeraBoostRegressor(depth=10, random_state=0).fit(X_train, y_train)
```

Per-leaf linear models add local slope inside each leaf (on by default for binary
classification; the regression default picks the better of linear and constant on the
validation split — force one variant to skip the double fit):

```python
reg = ChimeraBoostRegressor(linear_leaves=True, random_state=0).fit(X_train, y_train)
```

## Reproducibility and threads

```python
m = ChimeraBoostRegressor(
    random_state=0,        # deterministic for a fixed thread count
    thread_count=4,        # numba threads; None or -1 uses all cores
).fit(X_train, y_train)
```
