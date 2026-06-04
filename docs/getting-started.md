# Getting started

## Install

```bash
pip install chimeraboost
```

(Python 3.9 or newer)

## Regression

```pycon
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import root_mean_squared_error
>>> from chimeraboost import ChimeraBoostRegressor

>>> X, y = load_diabetes(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=0)

>>> reg = ChimeraBoostRegressor(random_state=0)
>>> reg.fit(X_train, y_train)
ChimeraBoostRegressor(random_state=0)

>>> preds = reg.predict(X_test)
>>> round(root_mean_squared_error(y_test, preds), 2)
56.82
```

Number of trees selected:

```pycon
>>> reg.best_iteration_
29
```

## Classification

`predict_proba` returns calibrated probabilities; columns follow `clf.classes_`.

```pycon
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.metrics import roc_auc_score
>>> from chimeraboost import ChimeraBoostClassifier

>>> X, y = load_breast_cancer(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=0, stratify=y)

>>> clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> round(roc_auc_score(y_test, proba[:, 1]), 3)
0.985
```

The probabilities are temperature-scaled on the validation split:

```pycon
>>> round(clf.temperature_, 3)
0.525
```

## Which features mattered

`feature_importances_` is a quick global ranking by split gain:

```pycon
>>> import numpy as np
>>> imp = reg.feature_importances_
>>> [(int(j), round(float(imp[j]), 3)) for j in np.argsort(imp)[::-1][:3]]
[(8, 0.402), (2, 0.235), (3, 0.108)]
```

For a faithful, per-prediction explanation, use SHAP. The contributions plus the
baseline reconstruct each prediction exactly:

```pycon
>>> phi = reg.shap_values(X_test)
>>> phi.shape
(89, 10)
>>> round(phi[0].sum() + reg.expected_value_, 4), round(reg.predict(X_test)[0], 4)
(276.5232, 276.5232)
```

## Next

- [Recipes](recipes.md) — categoricals, quantile regression, bagging, persistence, and more.
- [How it works](concepts.md) — oblivious trees, categorical encoding, linear leaves, calibration.
- [Parameters](parameters.md) — what each option does and when to change it.
- [SHAP](shap.md) — exact feature attributions in depth.
