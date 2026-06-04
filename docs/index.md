# ChimeraBoost

**What if CatBoost was slightly worse, 12× faster, and all in Python?**

ChimeraBoost is an exceedingly opinionated gradient-boosted decision tree library
built on **oblivious (symmetric) trees**, accelerated with [numba](https://numba.pydata.org/),
and depending on nothing heavier than NumPy, scikit-learn, SciPy, and pandas. No
C++, no build toolchain — you can read and modify every line in Python.

```python
pip install chimeraboost
```

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# classification
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, cat_features=[0, 1])
proba = clf.predict_proba(X_test)

# regression
reg = ChimeraBoostRegressor(random_state=0)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

A bare `fit(X, y)` already does the sensible thing: it carves an internal
validation split, early-stops on it, and predicts from the best iteration.

## What you get

- **Regression, quantile regression, binary and multiclass classification** under one API.
- **First-class categorical handling** — ordered target statistics (CatBoost-style), no manual encoding.
- **Sample weights**, **bagging** (`n_ensembles`), and **grouped early-stopping splits**.
- **Well-calibrated probabilities** out of the box (temperature scaling on `predict_proba`).
- **Exact SHAP attributions** (`model.shap_values(X)`) — interventional TreeSHAP computed
  exactly, not sampled, including the linear-leaf slope terms.
- A scikit-learn-compatible estimator interface (`fit`/`predict`/`predict_proba`,
  `feature_importances_`, full `check_estimator` compliance).

## Where to go next

- **[Recipes](recipes.md)** — copy-paste examples for every task: categoricals, quantile
  regression, multiclass, bagging, early stopping, calibration, importances, and SHAP.

## Design in one paragraph

Every node at a given depth splits on the *same* `(feature, threshold)`, so a depth-`d`
tree is just `d` splits and a leaf is a `d`-bit number. That symmetry is the source of
both the speed (prediction is a handful of comparisons plus an array lookup, vectorized
over the whole forest in one numba pass) and a large part of the regularization (only
`d` splits per tree, shared across the level). On top of that sit ordered target
statistics for categoricals, a leave-one-out leaf correction, optional per-leaf linear
models, and temperature-scaled probabilities.
