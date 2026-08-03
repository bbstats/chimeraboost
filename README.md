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
* [Where the ideas come from](https://bbstats.github.io/chimeraboost/attribution/): what is borrowed, what is ours

## Credit where it is due

Almost every idea in ChimeraBoost is someone else's. It is a from-scratch Python
implementation of published work — the algorithms below were invented by the people
listed next to them, not by us. **[Where the ideas come from](https://bbstats.github.io/chimeraboost/attribution/)**
maps this out feature by feature, including the short list of what actually originated
here. If a credit is missing or wrong, please open an issue.

* **Gradient boosting**, Friedman, *Annals of Statistics* 2001, and *Stochastic Gradient Boosting*, *Computational Statistics & Data Analysis* 2002. The algorithm itself, shrinkage, row subsampling, and the terminal-node override used for MAE and quantile losses.
* **CatBoost**, Prokhorenkova et al., *NeurIPS* 2018. Oblivious trees (the tree type itself from Kohavi & Li, *IJCAI* 1995), ordered target statistics, feature combinations, ordered boosting, and the size-dependent automatic learning rate (`adaptive_learning_rate`).
* **XGBoost**, Chen & Guestrin, *KDD* 2016. Regularized objective, second-order split gain, Newton leaf estimation, `min_child_weight`.
* **Column subsampling** (`colsample`), Breiman, *Random Forests*, 2001; random subspaces, Ho, *IEEE TPAMI* 1998. The XGBoost paper credits these rather than claiming it, and so do we.
* **LightGBM**, Ke et al., *NeurIPS* 2017. Histogram-based split finding, which that paper itself treats as prior art (McRank, Li et al. 2007; pGBRT, Tyree et al. 2011).
* **Minimum Variance Sampling**, Ibragimov & Gusev, *NeurIPS* 2019. Gradient-weighted row sampling (`subsample`).
* **Quantized GBDT training**, Shi, Ke et al., *NeurIPS* 2022. Integer gradient histograms (`quantize_gradients`).
* **SketchBoost**, Iosipoi & Vakhrushev, *NeurIPS* 2022, and **GBDT-MO**, Zhang & Jung. Vector leaves and the projected gradient used for multiclass and multi-quantile splits.
* **Linear-leaf trees**, Shi, Li & Li, *IJCAI* 2019 (arXiv:1802.05640). Piece-wise-linear regression trees (`linear_leaves`); model trees back to Quinlan's M5, 1992.
* **TreeSHAP**, Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017). Exact additive feature attributions (`shap_values`).
* **OpenFE**, Zhang et al., *ICML* 2023 (arXiv:2211.12507). Automated pairwise feature generation (`cross_features`).
* **Pinball loss**, Koenker & Bassett, *Econometrica* 1978, and **monotone rearrangement**, Chernozhukov, Fernández-Val & Galichon, *Econometrica* 2010. Quantile regression and its non-crossing guarantee.
* **Quantile regression forests**, Meinshausen, *JMLR* 2006, and **generalized random forests**, Athey, Tibshirani & Wager, *Annals of Statistics* 2019. One fitted model serving every quantile level, and labelling each row by the grid bucket it falls in.
* **L-moments**, Hosking, *JRSS-B* 1990, and **gamboostLSS**, Mayr et al., *JRSS-C* 2012. The shifted-Legendre location/spread/skew contrasts the quantile split search projects on, and cycling updates through them.
* **Conformal prediction**, Papadopoulos et al. 2002 and Vovk et al. 2005; **conformalized quantile regression**, Romano, Patterson & Candès, *NeurIPS* 2019. Distribution-free interval calibration (`conformalize`).
* **Temperature scaling**, Guo et al., *ICML* 2017 (Platt 1999 lineage). Probability calibration.
* **Bagging and out-of-bag estimation**, Breiman 1996; **subagging**, Bühlmann & Yu, *Annals of Statistics* 2002. The `n_ensembles` path.
* **AutoGluon**, Erickson et al. 2020. Refitting the selected model on all the data, and named quality presets.
* **scikit-learn**, Buitinck et al., *ECML-PKDD* 2013, and **numba**, Lam, Pitrou & Seibert, 2015. The API conventions and the compiler this is all built on.
