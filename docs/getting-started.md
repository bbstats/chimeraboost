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
60.57
```

Number of trees selected:

```pycon
>>> reg.best_iteration_
52
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
0.987
```

The probabilities are temperature-scaled on the validation split:

```pycon
>>> round(clf.temperature_, 3)
0.426
```

## Which features mattered

`feature_importances_` is a quick global ranking by split gain:

```pycon
>>> import numpy as np
>>> imp = reg.feature_importances_
>>> [(int(j), round(float(imp[j]), 3)) for j in np.argsort(imp)[::-1][:3]]
[(8, 0.419), (2, 0.225), (3, 0.117)]
```

For a faithful, per-prediction explanation, use SHAP. With the default
identity-link regressor losses, the contributions plus the baseline reconstruct
each prediction exactly (see [SHAP](shap.md) for the raw-score caveat on
transformed losses):

```pycon
>>> phi = reg.shap_values(X_test)
>>> phi.shape
(89, 10)
>>> round(phi[0].sum() + reg.expected_value_, 4), round(reg.predict(X_test)[0], 4)
(245.7708, 245.7708)
```

`shap_importances` is the global SHAP ranking in one call — mean absolute
contribution per feature, sorted. DataFrame column names are picked up
automatically; `prettified=True` returns a dict:

```pycon
>>> {f: round(v, 2) for f, v in
...  reg.shap_importances(X_test, n_features=3, prettified=True).items()}
{8: 24.72, 2: 19.92, 3: 10.31}
```

## How did it do?

`report` scores a fitted model without wiring up sklearn by hand. Every headline
number comes with a skill score against the no-skill forecast, because an RMSE of
60.6 means nothing on its own and an R2 of 0.28 does:

```pycon
>>> from chimeraboost import metrics
>>> print(metrics.format_report(reg.report(X_test, y_test)))
rows scored: 89
RMSE (lower better)                                 60.568497
MAE (lower better)                                  47.204309
R2 skill vs the mean (1 perfect, 0 no better)        0.284595
```

A classifier's `report` adds log loss, the Brier score and its skill, accuracy, F1,
and how far the probabilities are from calibrated:

```
rows scored: 114
log loss (lower better)                              0.311271
Brier score (lower better)                           0.072318
Brier skill vs the prior (1 perfect, 0 no better)    0.844741
accuracy                                             0.964912
F1 macro                                             0.962312
miscalibration (0 is perfectly calibrated)           0.011351
```

`calibration_mcb` is 0 when the probabilities are already perfectly calibrated and
higher when a monotone rescaling would improve them — it is the number temperature
scaling exists to keep small.

Quantile models have their own [`report`](quantiles.md#scoring).

## Next

- [Recipes](recipes.md): categoricals, quantile regression, bagging, persistence, and more.
- [How it works](concepts.md): oblivious trees, categorical encoding, linear leaves, calibration.
- [Parameters](parameters.md): what each option does and when to change it.
- [SHAP](shap.md): exact feature attributions in depth.
- [Predictive distributions](quantiles.md): quantiles, intervals, and calibration.
