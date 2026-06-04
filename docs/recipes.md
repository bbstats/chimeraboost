# Recipes

Copy-paste solutions for common tasks. Every snippet assumes:

```python
import numpy as np
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
```

All estimators are scikit-learn compatible, so they drop into `Pipeline`,
`cross_val_score`, `GridSearchCV`, and friends.

---

## Regression and classification

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
y_pred = reg.predict(X_test)

clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
labels = clf.predict(X_test)            # original label values
proba = clf.predict_proba(X_test)       # calibrated probabilities, columns = clf.classes_
```

A plain `fit(X, y)` already early-stops on an internal holdout — see
[Early stopping](#early-stopping). Pass `random_state` for reproducibility.

---

## Categorical features

Pass the **column indices** of your categoricals as `cat_features`. ChimeraBoost
encodes them with ordered target statistics (CatBoost-style) — no one-hot, no
manual `LabelEncoder`. Categorical columns may be strings/objects; the rest of
the matrix stays numeric.

```python
# columns 0 and 3 are categorical (e.g. "city", "device_type")
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X, y, cat_features=[0, 3])
```

!!! tip "Pairwise combinations"
    For predominantly-categorical data, `cat_combinations=True` adds all
    pairwise category×category features (helps datasets like `car`; can crowd
    out numerics on mixed data, so it is off by default).

---

## Quantile regression

Set `loss="Quantile"` and the target quantile `alpha`. To get a **prediction
interval**, fit one model per quantile:

```python
lo = ChimeraBoostRegressor(loss="Quantile", alpha=0.05, random_state=0).fit(X_train, y_train)
md = ChimeraBoostRegressor(loss="Quantile", alpha=0.50, random_state=0).fit(X_train, y_train)
hi = ChimeraBoostRegressor(loss="Quantile", alpha=0.95, random_state=0).fit(X_train, y_train)

lower, median, upper = lo.predict(X_test), md.predict(X_test), hi.predict(X_test)
# [lower, upper] is a 90% predictive interval
```

`loss="MAE"` (median regression) and the default `loss="RMSE"` are also available.

---

## Multiclass classification

Nothing to configure — the classifier switches to softmax multiclass automatically
when it sees 3+ classes, and `classes_` preserves your original label values.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X, y)   # y has 3+ classes
proba = clf.predict_proba(X_test)        # shape (n_samples, n_classes)
```

!!! note
    `linear_leaves` and `shap_values` are binary/regression only; multiclass
    uses constant leaves and raises `NotImplementedError` for SHAP.

---

## Sample weights

```python
w = np.where(y_train == 1, 5.0, 1.0)     # e.g. upweight the positive class
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, sample_weight=w)
```

Weights are normalized to mean 1 internally, so the gradient scale matches the
unweighted case. They apply to training only; the early-stopping metric is always
unweighted.

---

## Bagging (ensembles)

`n_ensembles` trains that many models on bootstrap resamples and averages them —
regressors average predictions, classifiers soft-vote calibrated probabilities.
This is a first-class feature, not an afterthought.

```python
reg = ChimeraBoostRegressor(n_ensembles=10, random_state=0).fit(X_train, y_train)

# train the members in parallel processes
reg = ChimeraBoostRegressor(n_ensembles=10, ensemble_n_jobs=-1, random_state=0)
reg.fit(X_train, y_train)
```

`feature_importances_` and `shap_values` are averaged across the bag automatically.

---

## Early stopping

Early stopping is **on by default**. With no `eval_set`, the estimator carves an
internal validation split (`validation_fraction=0.2`, stratified for classifiers),
stops after patience plateaus, and keeps the best iteration.

```python
# default: automatic internal holdout
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
print(m.best_iteration_)

# explicit validation set (overrides the internal split)
m = ChimeraBoostRegressor(random_state=0)
m.fit(X_train, y_train, eval_set=(X_val, y_val))

# grouped split: keep each group entirely in train OR val (no leakage)
m.fit(X_train, y_train, groups=subject_ids)

# turn it off: build a fixed number of trees
m = ChimeraBoostRegressor(early_stopping=False, iterations=500, random_state=0)
m.fit(X_train, y_train)
```

---

## Calibrated probabilities

`predict_proba` is temperature-scaled on the validation split to minimize log
loss. The scaling is monotonic, so `predict()` and ranking metrics (AUC, accuracy)
are unchanged while the probabilities themselves become better calibrated.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
proba = clf.predict_proba(X_test)        # already calibrated
print(clf.temperature_)                  # >1 = scores were over-confident
```

---

## Feature importances

`feature_importances_` is total split **gain** per original input column,
normalized to sum to 1 (aggregated across the bag when `n_ensembles > 1`).

```python
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
for j in np.argsort(m.feature_importances_)[::-1][:5]:
    print(f"feature {j}: {m.feature_importances_[j]:.3f}")
```

!!! tip "Want faithful contributions instead of split counts?"
    Gain importance reflects what the trees *built on*, not what moved each
    prediction, and it ignores the per-leaf linear models. For an exact,
    consistent decomposition of the actual output, use [SHAP](#shap-feature-attributions).

---

## SHAP feature attributions

`shap_values(X)` returns **exact** interventional TreeSHAP — computed, not
sampled — in the user's original feature space. For every row the contributions
satisfy the Shapley efficiency identity to floating-point tolerance:

```text
phi.sum(axis=1) + expected_value_  ==  prediction
```

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)

phi = reg.shap_values(X_test)            # shape (n_samples, n_features)
base = reg.expected_value_               # set as an attribute by the call above

# global importance: mean absolute contribution
for j in np.argsort(np.abs(phi).mean(0))[::-1][:5]:
    print(f"feature {j}: mean|SHAP| = {np.abs(phi[:, j]).mean():.4f}")

# explain a single prediction
i = 0
print("baseline:", base)
for j in np.argsort(np.abs(phi[i]))[::-1][:5]:
    print(f"  feature {j}: {phi[i, j]:+.4f}")
print("reconstruction:", phi[i].sum() + base, " vs prediction:", reg.predict(X_test)[i])
```

- **Regression** explains the predicted target; **binary classification** explains
  the pre-temperature **log-odds** of the positive class.
- The **linear-leaf slopes are included exactly**, so SHAP explains the actual
  model rather than just its split structure.
- Bagged models (`n_ensembles > 1`) return attributions averaged across members.
- The reference distribution defaults to a sample of the training data; override
  it with `X_background=...`. Cost is linear in the background size.

```python
phi = clf.shap_values(X_test, X_background=X_train[:200])
```

!!! warning "Binary / regression only"
    Multiclass SHAP is not supported yet and raises `NotImplementedError`.

---

## Interaction-heavy regression

The default `depth=6` is deliberately conservative to protect small datasets. On
large, interaction-heavy regression problems, raising depth is the single biggest
lever:

```python
reg = ChimeraBoostRegressor(depth=10, random_state=0).fit(X_train, y_train)
```

The optional per-leaf **linear models** add local slope inside each leaf (a Brier
win for binary; on by default for binary classification, opt-in for regression):

```python
reg = ChimeraBoostRegressor(linear_leaves=True, random_state=0).fit(X_train, y_train)
```

---

## Reproducibility and threads

```python
m = ChimeraBoostRegressor(
    random_state=0,        # deterministic given the same thread count
    thread_count=4,        # numba threads; None/-1 = all cores
).fit(X_train, y_train)
```
