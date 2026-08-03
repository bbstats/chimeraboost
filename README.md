# chimeraboost
*pronounced kai-MEER-uh-boost*

### Lightning-fast Gradient Boosting, near-<a href="https://github.com/user-attachments/assets/3a63ecb0-2f18-41bd-8119-ce4e0ca9a86a" />Catboost quality</a>, all in Python

📖 **Documentation:** [bbstats.github.io/chimeraboost](https://bbstats.github.io/chimeraboost/)

<center>
<img width="532" height="428" alt="image" src="https://github.com/user-attachments/assets/1778b56f-e05e-4dfb-be93-1f0b66f15d98" />



**#2 On TabArena GBDT Elo (Defaults)**
<img width="1312" height="464" alt="image" src="https://github.com/user-attachments/assets/e1d26498-88f4-4ad9-bc76-05b7778054e5" />
<br><br>
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
* Replay mechanism for faster refitting after early stopping
* Gradient-matrix free multi-quantile split search
* numba is very fast

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/public_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/public_pareto.png" width="500" alt="Average rank vs fit-time slowdown on the public suite" /></a></p>

## Why?

* I want to be able to modify my GBDT library at will
* I know Python and I don't know C

## Documentation

* [Getting started](https://bbstats.github.io/chimeraboost/getting-started/)
* [Recipes](https://bbstats.github.io/chimeraboost/recipes/)
* [Parameters](https://bbstats.github.io/chimeraboost/parameters/)
* [FAQ](https://bbstats.github.io/chimeraboost/faq/)
* [Where the ideas come from](https://bbstats.github.io/chimeraboost/attribution/)

## Benchmarking

Want to help? Run either one, then open an issue or PR with the JSON.

```
pip install -e ".[bench,competitors]"

python benchmarks/run_benchmarks.py --synth --seeds 3 --save     # quick, synthetic
python benchmarks/run_benchmarks.py --decide --seeds 3 --save    # slower, 103 real datasets
```

Each writes `benchmarks/results/<timestamp>.json`:

```json
{
  "provenance": {"chimeraboost": "0.30.0", "platform": "Linux-6.1", "cpu_count": 12,
                 "libraries": {"catboost": "1.2.10", "lightgbm": "4.6.0"}},
  "records": [
    {"dataset": "diabetes", "model": "ChimeraBoost", "seed": 0,
     "metrics": {"primary": -59.82, "rmse": 59.82}, "fit_time": 0.23}
  ]
}
```


## Inspirations / Citations

Most ideas in ChimeraBoost were someone else's.

* **CatBoost**, Prokhorenkova et al., *NeurIPS* 2018. Oblivious trees (the tree type itself from Kohavi & Li, *IJCAI* 1995), ordered target statistics, feature combinations, ordered boosting, and the size-dependent automatic learning rate (`adaptive_learning_rate`).
* **XGBoost**, Chen & Guestrin, *KDD* 2016. Regularized objective, second-order split gain, Newton leaf estimation, `min_child_weight`.
* **LightGBM**, Ke et al., *NeurIPS* 2017. Histogram-based split finding, which that paper itself treats as prior art (McRank, Li et al. 2007; pGBRT, Tyree et al. 2011).
* **Minimum Variance Sampling**, Ibragimov & Gusev, *NeurIPS* 2019. Gradient-weighted row sampling (`subsample`).
* **Quantized GBDT training**, Shi, Ke et al., *NeurIPS* 2022. Integer gradient histograms (`quantize_gradients`).
* **SketchBoost**, Iosipoi & Vakhrushev, *NeurIPS* 2022, and **GBDT-MO**, Zhang & Jung. Vector leaves and the projected gradient used for multiclass and multi-quantile splits.
* **Linear-leaf trees**, Shi, Li & Li, *IJCAI* 2019 (arXiv:1802.05640). Piece-wise-linear regression trees (`linear_leaves`); model trees back to Quinlan's M5, 1992.
* **TreeSHAP**, Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017). Exact additive feature attributions (`shap_values`).
* **OpenFE**, Zhang et al., *ICML* 2023 (arXiv:2211.12507). Automated pairwise feature generation (`cross_features`).
* **Conformal prediction**, Papadopoulos et al. 2002 and Vovk et al. 2005; **conformalized quantile regression**, Romano, Patterson & Candès, *NeurIPS* 2019. Distribution-free interval calibration (`conformalize`).
* **Temperature scaling**, Guo et al., *ICML* 2017 (Platt 1999 lineage). Probability calibration.
* **Bagging and out-of-bag estimation**, Breiman 1996; **subagging**, Bühlmann & Yu, *Annals of Statistics* 2002; column subsampling from *Random Forests* 2001 and random subspaces, Ho, *IEEE TPAMI* 1998. The `n_ensembles` path and `colsample`.
