# chimeraboost
### What if CatBoost was slightly worse, 12× faster, and all in Python?

> ⚠️ **Project is in active development:** breaking changes should be expected.

<center>
<img width="500" height="500" alt="chimeraboost logo" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />
</center>

* **Installation**

```
pip install chimeraboost
```

* **Sample code:**

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# classification
clf = ChimeraBoostClassifier(early_stopping=True, n_ensembles=2)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

# regression (RMSE, MAE, or Quantile)
reg = ChimeraBoostRegressor(loss="Quantile", alpha=0.9, early_stopping=True, n_ensembles=10)
reg.fit(X, y)
```

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/tabarena_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png" width="500" alt="TabArena-Lite Elo vs speed Pareto" /></a></p>

* **What?**
    * Exceedingly opinionated GBDT library that only depends on common Python libraries
        * Categorical features (catboost-like processing) and sample weights
        * Bagging as a first-class feature (`n_ensembles`)
        * Automatic early stopping, with optional grouped splitting for the validation set
    * Supports regression, quantile regression, binary and multiclass classification
    * **Exact SHAP** explanations (`model.shap_values(X)`) — interventional TreeSHAP
      computed exactly (not sampled) thanks to the oblivious tree structure, with
      the linear-leaf slopes included

* **Tuning tips**
    * Interaction-heavy regression: raise `depth` to 8–10 (default 6 is conservative to protect small data).

* **Inspirations / Citations**
    * **XGBoost** — Chen & Guestrin, *KDD* 2016 — regularized objective, Newton leaf estimation, column subsampling
    * **LightGBM** — Ke et al., *NeurIPS* 2017 — histogram-based split finding
    * **CatBoost** — Prokhorenkova et al., *NeurIPS* 2018 — ordered boosting, ordered target statistics, oblivious trees
    * **Linear-leaf trees** — Shi et al., *IJCAI* 2019 (arXiv:1802.05640) — piece-wise-linear regression trees (the `linear_leaves` default for binary)
    * **TreeSHAP** — Lundberg et al., *Nature Machine Intelligence* 2020 (orig. SHAP, *NeurIPS* 2017) — exact additive feature attributions (`shap_values`)
    * **Hierarchical shrinkage** — Agarwal et al., *ICML* 2022 (arXiv:2202.00858) — the optional `hs_lambda` leaf regularizer
    * **TabArena** — Erickson et al., *NeurIPS* 2025 (arXiv:2506.16791) — the benchmark

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C
