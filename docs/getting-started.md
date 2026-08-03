# Getting started

## Install

```bash
pip install chimeraboost
```

Python 3.9 or newer. The only dependencies are numpy, numba, scikit-learn and scipy.

Optional, once: run `chimeraboost-warmup` to compile the numba kernels now, so the
cold-compile wait (about ten seconds) does not land on your first real fit. Re-run it
after every upgrade — see [Deployment](deployment.md).

## Regression

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from chimeraboost import ChimeraBoostRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

reg = ChimeraBoostRegressor(random_state=0)
reg.fit(X_train, y_train)

preds = reg.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, preds)))   # ~60.6
```

You never chose a number of trees. `fit` holds back a slice of the training rows, stops
adding trees once the score on that slice stops improving, and keeps the best round:

```python
print(reg.best_iteration_)      # 52 trees kept on this data
```

To pass your own validation set instead, or to turn stopping off and fit a fixed number
of trees, see [Early stopping](recipes.md#early-stopping).

## Classification

`predict_proba` returns [calibrated](recipes.md#calibrated-probabilities) probabilities;
columns follow `clf.classes_`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from chimeraboost import ChimeraBoostClassifier

Xc, yc = load_breast_cancer(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=0, stratify=yc)

clf = ChimeraBoostClassifier(random_state=0).fit(Xc_train, yc_train)
proba = clf.predict_proba(Xc_test)
print(roc_auc_score(yc_test, proba[:, 1]))      # ~0.987
```

## Categorical columns

Name your categorical columns and pass the DataFrame as it is — no one-hot, no
`LabelEncoder`, and missing values need no imputation:

```python
model = ChimeraBoostClassifier(random_state=0)
model.fit(df, labels, cat_features=["city", "device_type"])   # strings stay strings
```

Integer column positions work too, and are the only option for a plain numpy array:
`cat_features=[0, 3]`. See [Categorical features](recipes.md#categorical-features).

## Which features mattered

`feature_importances_` is a quick global ranking by split gain — one entry per input
column, summing to 1:

```python
order = np.argsort(reg.feature_importances_)[::-1]
print(order[:3])        # the three columns that earned the most gain
```

For a per-prediction explanation, use SHAP. A row's contributions plus the baseline add
back up to that prediction (see [SHAP](shap.md)):

```python
phi = reg.shap_values(X_test)
print(phi.shape)        # (89, 10) -- one row per prediction, one column per feature

recon = phi.sum(axis=1) + reg.expected_value_
print(np.allclose(recon, reg.predict(X_test)))      # True
```

## Next

- [Recipes](recipes.md): categoricals, quantile regression, bagging, persistence, and more.
- [How it works](concepts.md): oblivious trees, categorical encoding, linear leaves, calibration.
- [Parameters](parameters.md): what each option does and when to change it.
- [SHAP](shap.md): exact feature attributions in depth.
