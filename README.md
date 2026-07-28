# chimeraboost

### Lightning fast, near-CatBoost quality, all in Python

📖 **Documentation:** [bbstats.github.io/chimeraboost](https://bbstats.github.io/chimeraboost/)

<center>
<img width="500" height="500" alt="chimeraboost logo" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />
</center>

## Install

```
pip install chimeraboost && chimeraboost-warmup
```

`chimeraboost-warmup` compiles the numba kernels once and caches them, so the first
`fit` is not several seconds slower than the rest. Re-run it after every upgrade, which
resets the cache. See
[Deployment](https://bbstats.github.io/chimeraboost/deployment/).

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# classification. quality picks the speed/accuracy trade-off: 1 fastest .. 5 strongest,
# defaulting to 3.
clf = ChimeraBoostClassifier(quality=5)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

# regression (RMSE, MAE, Quantile, Huber, Poisson, Gamma, Tweedie, or your own)
reg = ChimeraBoostRegressor(loss="Quantile", alpha=0.9)
reg.fit(X, y)
```

## What it is

An opinionated GBDT library that only depends on common Python libraries
(NumPy, numba, scikit-learn, SciPy):

* Regression, quantile regression, binary and multiclass classification
* A whole predictive distribution from one booster, with quantiles that cannot cross
  (`ChimeraBoostQuantileRegressor`)
* Categorical features handled natively (CatBoost-style ordered target statistics)
* Missing values handled directly, no imputation
* Automatic early stopping, with optional grouped splitting for the validation set
* Bagging as a first-class feature (`n_ensembles`)
* **Exact SHAP** explanations (`model.shap_values(X)`). The oblivious tree structure
  makes interventional TreeSHAP cheap enough to compute exactly, with no sampling.

On the [TabArena](https://tabarena.ai/) benchmark, the default model scores above
XGBoost and LightGBM while training faster than either. CatBoost scores higher and
takes considerably longer.

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/tabarena_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png" width="500" alt="TabArena-Lite Elo vs speed Pareto" /></a></p>

<sub>TabArena is a benchmark we never tune against. A second read on an independently
audited 22-dataset suite, including how it is weighted and where it disagrees with us,
is in <a href="https://github.com/bbstats/chimeraboost/blob/main/docs/benchmarks.md">docs/benchmarks.md</a>.</sub>

## Documentation

* [Getting started](https://bbstats.github.io/chimeraboost/getting-started/): install and first models
* [Recipes](https://bbstats.github.io/chimeraboost/recipes/): categoricals, quantiles, bagging, custom losses, and more
* [Parameters](https://bbstats.github.io/chimeraboost/parameters/): every option, with defaults and guidance
* [FAQ](https://bbstats.github.io/chimeraboost/faq/): common questions

## Why?

* I want to be able to modify my GBDT library at will
* I know Python and I don't know C

## Inspirations / Citations

* **CatBoost**, Prokhorenkova et al., *NeurIPS* 2018. Ordered boosting, ordered target statistics, oblivious trees.
* **XGBoost**, Chen & Guestrin, *KDD* 2016. Regularized objective, Newton leaf estimation, column subsampling.
* **LightGBM**, Ke et al., *NeurIPS* 2017. Histogram-based split finding.
* **Linear-leaf trees**, Shi et al., *IJCAI* 2019 (arXiv:1802.05640). Piece-wise-linear regression trees (`linear_leaves`).
* **TreeSHAP**, Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017). Exact additive feature attributions (`shap_values`).
* **OpenFE**, Zhang et al., *ICML* 2023 (arXiv:2211.12507). Automated pairwise feature generation (`cross_features`).
* **Conformalized quantile regression**, Romano, Patterson & Candès, *NeurIPS* 2019. Distribution-free interval calibration (`conformalize`).
* **TabArena**, Erickson et al., *NeurIPS* 2025 (arXiv:2506.16791). The tabular benchmark used for evaluation.
