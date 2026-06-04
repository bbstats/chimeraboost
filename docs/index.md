# ChimeraBoost

Gradient boosting on oblivious (symmetric) decision trees, written in Python with a
[numba](https://numba.pydata.org/) backend. It depends only on NumPy, scikit-learn,
SciPy, and pandas — no C++ extensions and no build step, so you can read and modify
every line.

> *What if CatBoost was slightly worse, 12× faster, and all in Python?*

## Install

```bash
pip install chimeraboost
```

Python 3.9 or newer.

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, cat_features=[0, 1])
proba = clf.predict_proba(X_test)

reg = ChimeraBoostRegressor(random_state=0)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

`fit(X, y)` holds out an internal validation split, early-stops on it, and predicts
from the best round. Categorical columns are passed by index (`cat_features=`); NaNs
route to a dedicated bin, so no imputation is needed.

## What it does

- Regression (squared error, absolute error, quantile), binary and multiclass classification.
- Categorical features via ordered target statistics — pass column indices, no manual encoding.
- Sample weights, bagging (`n_ensembles`), and grouped validation splits.
- Calibrated probabilities (`predict_proba` is temperature-scaled on the validation split).
- Exact SHAP attributions ([`shap_values`](shap.md)), including the per-leaf linear models.
- A scikit-learn estimator API that drops into `Pipeline`, `GridSearchCV`, and `cross_val_score`.

## Benchmarks

[![TabArena-Lite Elo vs training time](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png){ width="560" }](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png)

On TabArena-Lite, ChimeraBoost sits on the accuracy-vs-training-time Pareto frontier:
ahead of XGBoost and LightGBM defaults on both axes, and within reach of CatBoost's
accuracy at a fraction of its training time.

## Documentation

- [Recipes](recipes.md) — worked examples for every task.
- [Parameters](parameters.md) — what each option does and when to change it.
- [SHAP](shap.md) — exact feature attributions.
- [API reference](api.md) — classes, methods, and signatures.

## How the trees work

Every node at a given depth splits on the same `(feature, threshold)`, so a depth-`d`
tree is `d` splits and a leaf is a `d`-bit number. That symmetry makes prediction a
handful of comparisons plus an array lookup (vectorized across the whole forest in one
numba pass) and provides much of the regularization, since a tree has only `d` splits
shared across its level. Categoricals use ordered target statistics, leaves can carry
small linear models, and probabilities are temperature-scaled after fitting.
