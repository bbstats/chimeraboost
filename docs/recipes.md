# Recipes

Worked examples for common tasks. Every snippet assumes:

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

Pass the column indices of your categoricals as `cat_features`. They are encoded with
ordered target statistics (CatBoost-style), so there is no one-hot or `LabelEncoder`
step. Categorical columns can be strings or objects; the rest of the matrix stays numeric.

```python
# columns 0 and 3 are categorical (e.g. "city", "device_type")
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X, y, cat_features=[0, 3])
```

For mostly-categorical data, `cat_combinations=True` adds all pairwise category-by-category
features. It can crowd out numerics on mixed data, so it is off by default.

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
collapse toward the median on held-out data. If your intervals still look too narrow,
go shallower (`depth=3`); if they look too wide, raise `depth` and add more `iterations`.
As with any tree-based quantile model, held-out coverage is approximate, not exact.

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

`n_ensembles` trains that many models on bootstrap resamples and averages them —
regressors average predictions, classifiers soft-vote calibrated probabilities.

```python
reg = ChimeraBoostRegressor(n_ensembles=10, random_state=0).fit(X_train, y_train)

# fit members in parallel processes
reg = ChimeraBoostRegressor(n_ensembles=10, ensemble_n_jobs=-1, random_state=0)
reg.fit(X_train, y_train)
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
m = ChimeraBoostRegressor(early_stopping=False, iterations=500, random_state=0)
m.fit(X_train, y_train)
```

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

To pass `cat_features` through a search, set it once on the estimator via a small
wrapper or use a `Pipeline` whose final step is the booster.

## Save and load a model

A fitted estimator pickles like any scikit-learn object:

```python
import joblib

joblib.dump(reg, "model.joblib")
reg = joblib.load("model.joblib")
```

## Interaction-heavy regression

The default `depth=6` is conservative to protect small data. On large, interaction-heavy
problems, raising depth is the biggest single lever:

```python
reg = ChimeraBoostRegressor(depth=10, random_state=0).fit(X_train, y_train)
```

Per-leaf linear models add local slope inside each leaf (on by default for binary
classification, opt-in for regression):

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
