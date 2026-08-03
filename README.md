# chimeraboost
*pronounced kai-MEER-uh-boost*

### Lightning-fast Gradient Boosting, near-<a href="https://github.com/user-attachments/assets/3a63ecb0-2f18-41bd-8119-ce4e0ca9a86a" />Catboost quality</a>, all in Python

📖 **Documentation:** [bbstats.github.io/chimeraboost](https://bbstats.github.io/chimeraboost/)

<center>
<img width="532" height="428" alt="image" src="https://github.com/user-attachments/assets/1778b56f-e05e-4dfb-be93-1f0b66f15d98" />

</center>

## Installation

```
pip install chimeraboost
```

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# quality picks the speed/accuracy trade-off: 1 is fastest - 5 is strongest,
# default is 3.
clf = ChimeraBoostClassifier(quality=4)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

reg = ChimeraBoostRegressor()
reg.fit(X, y)
```

## What?

* Regression, quantile regression, binary and multiclass classification
* Fast training and inference, all in python
* Extremely high quality predictions

## How?
* Bagging as a first-class feature (using `n_ensembles`)
* Automatic early stopping
* Automatic linear-leaf auditioning
* numba is very fast

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/public_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png" width="500" alt="Average rank vs fit-time slowdown on the public suite" /></a></p>

## Why?

* I want to be able to modify my GBDT library at will
* I know Python and I don't know C

## Documentation

* [Getting started](https://bbstats.github.io/chimeraboost/getting-started/): install and first models
* [Recipes](https://bbstats.github.io/chimeraboost/recipes/): categoricals, quantiles, bagging, custom losses, and more
* [Parameters](https://bbstats.github.io/chimeraboost/parameters/): every option, with defaults and guidance
* [FAQ](https://bbstats.github.io/chimeraboost/faq/): common questions

## Inspirations / Citations

* **CatBoost**, Prokhorenkova et al., *NeurIPS* 2018. Ordered boosting, ordered target statistics, oblivious trees.
* **XGBoost**, Chen & Guestrin, *KDD* 2016. Regularized objective, Newton leaf estimation, column subsampling.
* **LightGBM**, Ke et al., *NeurIPS* 2017. Histogram-based split finding.
* **Linear-leaf trees**, Shi et al., *IJCAI* 2019 (arXiv:1802.05640). Piece-wise-linear regression trees (`linear_leaves`).
* **TreeSHAP**, Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017). Exact additive feature attributions (`shap_values`).
* **OpenFE**, Zhang et al., *ICML* 2023 (arXiv:2211.12507). Automated pairwise feature generation (`cross_features`).
* **Conformalized quantile regression**, Romano, Patterson & Candès, *NeurIPS* 2019. Distribution-free interval calibration (`conformalize`).
