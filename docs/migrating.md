# Coming from XGBoost, LightGBM or CatBoost

The estimators are plain scikit-learn estimators: `fit`, `predict`,
`predict_proba`. There is no `DMatrix`, `Dataset` or `Pool` — pass numpy arrays
or pandas/polars DataFrames straight in. For most models the whole port is:

```python
from chimeraboost import ChimeraBoostClassifier

clf = ChimeraBoostClassifier(random_state=0)
clf.fit(df_train, y_train, cat_features=["city", "device_type"])
proba = clf.predict_proba(df_test)
```

No encoder, no imputer, no early-stopping callback: NaNs route to their own bin,
categoricals go in as strings, and early stopping is already on.

## Parameter map

| XGBoost | LightGBM | CatBoost | ChimeraBoost | Notes |
|---|---|---|---|---|
| `n_estimators` | `n_estimators` | `iterations` | `n_estimators` | Default 2000. It is a cap, not a target — early stopping picks the round. |
| `learning_rate` | `learning_rate` | `learning_rate` | `learning_rate` | Default `None`: 0.1 at about 15,000 training rows or more, fading to 0.07 at 5,000 or fewer. |
| `max_depth` | `num_leaves`, `max_depth` | `depth` | `depth` | Trees are oblivious, so depth is the only shape knob: a depth-d tree has 2^d leaves and no `num_leaves` equivalent. Default 6 (4 for `loss="Quantile"`), maximum 16. |
| `reg_lambda` | `reg_lambda` / `lambda_l2` | `l2_leaf_reg` | `l2_leaf_reg` | Same meaning, default 1.0. |
| `min_child_weight` | `min_child_weight` | `min_data_in_leaf` | `min_child_weight` | Hessian mass on each side of a split, as in XGBoost and LightGBM — not a row count like CatBoost. Regressor default 1.0; the classifier's `None` adapts to dataset size. |
| `subsample` | `subsample` / `bagging_fraction` | `subsample` | `subsample` | Applied every tree, so there is nothing to set like `subsample_freq` / `bagging_freq`. Below 1.0 rows are drawn by Minimum Variance Sampling, not uniformly. |
| `colsample_bytree` | `colsample_bytree` / `feature_fraction` | `rsm` | `colsample` | Feature fraction per tree. `None` (the default) means all columns for a single model, and 0.85 for members inside a bag. |
| `max_bin` | `max_bin` | `border_count` | `max_bins` | Note the plural. Default 128. |
| `objective` | `objective` | `loss_function` | `loss` (regressor only) | Names follow CatBoost: `"RMSE"`, `"MAE"`, `"Quantile"` (level in `alpha`), `"Huber"` (`delta`), `"Poisson"`, `"Gamma"`, `"Tweedie"`. The classifier has no `loss` — it uses log loss for two classes and softmax for three or more. |
| `eval_metric` | `eval_metric` | `eval_metric` | `eval_metric` | A callable `metric(y_true, y_pred[, sample_weight]) -> float`. Metric-name strings are not accepted. |
| `early_stopping_rounds` + `eval_set` | `callbacks=[lgb.early_stopping(...)]` | `early_stopping_rounds` / `od_type` | `early_stopping`, `early_stopping_rounds`, `validation_fraction` | On by default, on an automatic 20% holdout (stratified for classifiers). Patience defaults to 50. Pass `eval_set=(X_val, y_val)` to `fit` to choose the holdout yourself, or `early_stopping=False` to build a fixed `n_estimators` trees. |
| `enable_categorical` + `category` dtype | `categorical_feature` | `cat_features` | `cat_features` | Integer positions and/or column names, given to `fit` or to the constructor. String and object columns go in raw. |
| `sample_weight` | `sample_weight` | `sample_weight` | `sample_weight` | A `fit` argument, same meaning. |
| `n_jobs` / `nthread` | `n_jobs` / `num_threads` | `thread_count` | `thread_count` | `None` or `-1` uses all cores. |
| `random_state` | `random_state` | `random_seed` | `random_state` | Deterministic for a fixed `thread_count`. |
| `verbosity` | `verbose` | `verbose` | `verbose` | Boolean. Prints per-round train and validation scores. |
| `DMatrix` | `Dataset` | `Pool` | — | Not needed. Arrays and DataFrames go straight to `fit`. |

Everything else is in [Parameters](parameters.md).

## Different by design

- **Oblivious trees,** the design CatBoost uses. Every node at the same depth
  takes the same split, so a tree is a decision table: strongly regularized,
  very fast to evaluate, less sharp per tree. Depth replaces leaf-count tuning
  entirely. See [How it works](concepts.md#oblivious-trees).
- **Early stopping is on by default,** and once the automatic holdout has picked
  the round, the model is retrained on 100% of the rows so it does not pay for
  the holdout. Passing your own `eval_set` leaves your split alone and skips
  that refit. See [Early stopping](recipes.md#early-stopping) and
  [Full-data refit](recipes.md#full-data-refit).
- **`predict_proba` is calibrated,** by temperature scaling fitted on that same
  holdout. `predict` is unchanged by it. See
  [Calibrated probabilities](recipes.md#calibrated-probabilities).
- **The learning rate adapts to your data size** when you leave it at `None`,
  because small data wants a lower rate. Set it explicitly to override.
- **Categoricals are always ordered target statistics** (CatBoost's scheme), at
  every cardinality — there is no one-hot fallback for small categories and no
  `one_hot_max_size` to set. See
  [Categorical features](concepts.md#categorical-features).
- **`quality=1..5` replaces most tuning.** One integer moves you along the
  speed/accuracy curve; `quality=3` is the defaults. See
  [the quality ladder](recipes.md#speed-and-accuracy-the-quality-ladder).

## Not supported

- **GPU training.** CPU only — no `task_type="GPU"`, `device="cuda"` or
  `device_type="gpu"`.
- **Monotone and interaction constraints.** No `monotone_constraints` or
  `interaction_constraints`.
- **L1 leaf regularization.** No `reg_alpha` / `lambda_l1`; `l2_leaf_reg` is the
  only leaf penalty.
- **Class weights.** No `class_weight`, `scale_pos_weight` or
  `auto_class_weights`. Use `sample_weight` for the same effect.
- **Continued training.** No `warm_start`, `init_model` or `xgb_model`; each
  `fit` starts from scratch.
- **Ranking objectives.** No `rank:pairwise`, `lambdarank` or `YetiRank`.
- **Text and embedding features.** No `text_features` / `embedding_features` —
  vectorize those yourself.
- **SHAP for multiclass.** `shap_values` raises `NotImplementedError` above two
  classes. `feature_importances_` still works. See [SHAP](shap.md).
- **Per-iteration classifier predictions.** The regressor and the quantile head
  have `staged_predict` (single models only); there is no
  `staged_predict_proba`, and `predict` has no `iteration_range` /
  `ntree_limit`.
- **Sparse matrices and multi-output `y`.** Both are rejected with an error;
  pass a dense 2-D `X` and a 1-D `y`.
- **A native model format.** No `save_model` / `.cbm` / JSON dump — pickle the
  estimator instead, and see [FAQ](faq.md) for the version caveat.
