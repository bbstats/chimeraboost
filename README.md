# chimeraboost
*pronounced kai-MEER-uh-boost*

### Lightning-fast gradient boosting, near-[CatBoost quality](https://bbstats.github.io/chimeraboost/benchmarks/), all in Python

📖 **Documentation:** [bbstats.github.io/chimeraboost](https://bbstats.github.io/chimeraboost/)

<center>
<img width="532" height="428" alt="Accuracy against fit time: ChimeraBoost sits close to CatBoost at a fraction of its fit cost" src="https://github.com/user-attachments/assets/1778b56f-e05e-4dfb-be93-1f0b66f15d98" />

</center>

## Installation

```
pip install chimeraboost
```

## Quickstart

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# classification. quality picks the speed/accuracy trade-off: 1 fastest .. 5 strongest,
# default is 3.
clf = ChimeraBoostClassifier(quality=4)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

# regression (RMSE, MAE, Quantile, Huber, Poisson, Gamma, Tweedie, or your own)
reg = ChimeraBoostRegressor()
reg.fit(X, y)
```

## What?

* Regression, quantile regression, binary and multiclass classification
* Categorical columns handled directly — no one-hot, no label encoding
* Near-CatBoost accuracy on public benchmarks, at a fraction of the fit time (chart below)

## How?
* Bagging as a first-class feature (`n_ensembles`)
* Early stopping and probability calibration on by default
* Fits with and without linear leaves, and keeps whichever validates better
* numba is very fast

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/public_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png" width="500" alt="Average rank vs fit-time slowdown on the public suite" /></a></p>

## Why?

* It is all Python, so you can read and change any part of it
* No C or C++ build step, and no compiled extension to fight

## Documentation

* [Getting started](https://bbstats.github.io/chimeraboost/getting-started/): install and first models
* [Recipes](https://bbstats.github.io/chimeraboost/recipes/): categoricals, quantiles, bagging, custom losses, and more
* [Parameters](https://bbstats.github.io/chimeraboost/parameters/): every option, with defaults and guidance
* [FAQ](https://bbstats.github.io/chimeraboost/faq/): common questions
* [Where the ideas come from](https://bbstats.github.io/chimeraboost/attribution/): what is borrowed, what is ours

## Credit where it is due

Almost every idea in ChimeraBoost is someone else's. It is a from-scratch Python
implementation of published work — the algorithms below were invented by the people
listed next to them, not by us. If a credit is missing or wrong, please open an issue.

**What we did build**, so it is not a mystery: the **structure-transfer refit**, where
early stopping picks the tree count and those trees are then replayed over all your data
instead of being regrown, so the held-out rows still reach the model. Growing trees is
about 85% of a fit, and a from-scratch refit re-pays for structure it already has. This
does not: fit time falls 34.8% on the Grinsztajn suite, faster on 58 of 59 datasets, at
unchanged accuracy. It is on by default. Also ours: a multi-quantile split search that
never builds the per-level gradient matrix, and a set of small size-dependent defaults.
**[Where the ideas come from](https://bbstats.github.io/chimeraboost/attribution/)** maps
all of it out feature by feature — including which parts of those are themselves borrowed.

* **Gradient boosting**, Friedman, *Annals of Statistics* 2001, and *Stochastic Gradient Boosting*, *Computational Statistics & Data Analysis* 2002. The algorithm itself, shrinkage, row subsampling, and the terminal-node override used for MAE and quantile losses.
* **CatBoost**, Prokhorenkova et al., *NeurIPS* 2018. Oblivious trees (the tree type itself from Kohavi & Li, *IJCAI* 1995), ordered target statistics, feature combinations, ordered boosting, and the size-dependent automatic learning rate (`adaptive_learning_rate`).
* **XGBoost**, Chen & Guestrin, *KDD* 2016. Regularized objective, second-order split gain, Newton leaf estimation, `min_child_weight`.
* **LightGBM**, Ke et al., *NeurIPS* 2017. Histogram-based split finding, which that paper itself treats as prior art (McRank, Li et al. 2007; pGBRT, Tyree et al. 2011).
* **Minimum Variance Sampling**, Ibragimov & Gusev, *NeurIPS* 2019. Gradient-weighted row sampling (`subsample`).
* **Quantized GBDT training**, Shi, Ke et al., *NeurIPS* 2022. Integer gradient histograms (`quantize_gradients`).
* **SketchBoost**, Iosipoi & Vakhrushev, *NeurIPS* 2022, and **GBDT-MO**, Zhang & Jung. Vector leaves and the projected gradient used for multiclass and multi-quantile splits.
* **Linear-leaf trees**, Shi, Li & Li, *IJCAI* 2019 (arXiv:1802.05640). Piece-wise-linear regression trees (`linear_leaves`); model trees back to Quinlan's M5, 1992.
* **TreeSHAP**, Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017). Exact additive feature attributions (`shap_values`).
* **OpenFE**, Zhang et al., *ICML* 2023 (arXiv:2211.12507). Automated pairwise feature generation (`cross_features`).
* **Quantile regression** — Koenker & Bassett, *Econometrica* 1978; one model for every level, Meinshausen, *JMLR* 2006; the shared-split scoring of **generalized random forests**, Athey, Tibshirani & Wager, *AoS* 2019; the location/spread contrasts it projects on (**L-moments**, Hosking, *JRSS-B* 1990) and cycling through them (**gamboostLSS**, Mayr et al., *JRSS-C* 2012); non-crossing by **monotone rearrangement**, Chernozhukov, Fernández-Val & Galichon, *Econometrica* 2010.
* **Conformal prediction**, Papadopoulos et al. 2002 and Vovk et al. 2005; **conformalized quantile regression**, Romano, Patterson & Candès, *NeurIPS* 2019. Distribution-free interval calibration (`conformalize`).
* **Temperature scaling**, Guo et al., *ICML* 2017 (Platt 1999 lineage). Probability calibration.
* **Bagging and out-of-bag estimation**, Breiman 1996; **subagging**, Bühlmann & Yu, *Annals of Statistics* 2002; column subsampling from *Random Forests* 2001 and random subspaces, Ho, *IEEE TPAMI* 1998. The `n_ensembles` path and `colsample`.
* **AutoGluon**, Erickson et al. 2020. Refitting the selected model on all the data, and named quality presets. The other half of our replay refit — refreshing leaf values on a tree structure you already have — is XGBoost's `refresh` updater and LightGBM's `Booster.refit()`. Putting the two together is the part that is ours.
* **scikit-learn**, Buitinck et al., *ECML-PKDD* 2013, and **numba**, Lam, Pitrou & Seibert, 2015. The API conventions and the compiler this is all built on.
