# Parameters

For more detail, see [API reference](api.md).

## Core boosting

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | `2000` | Maximum boosting rounds (trees). |
| `learning_rate` | `None` (auto) | Per-tree shrinkage. `None` resolves to 0.1 with early stopping. Lower trades more trees for slightly better fit. |
| `depth` | `None`→auto | Tree depth (a depth-d tree is d splits). Defaults to to 6 for squared/absolute error and 4 for `loss="Quantile"` (deep leaves overfit the tail quantile). Conservative by default; raise to 8–10 for large, interaction-heavy regression. |
| `l2_leaf_reg` | `1.0` | L2 penalty on leaf values. Higher is smoother. |
| `min_child_weight` | `1.0` (reg) / `None`→auto (clf) | Minimum hessian mass on each side of a split. The classifier's `None` is size-adaptive: gradates off when below 500 rows, gradates off when above 2000. |
| `leaf_estimation_iterations` | `1` (reg) / `3` (clf) | Extra Newton refinement steps per leaf. Likely more important to tune in classification tasks. |

## Binning

| Parameter | Default | Effect |
|---|---|---|
| `max_bins` | `128` | Histogram bins per numeric feature. Raising it can improve fit in some scenarios. |

## Row and column sampling

| Parameter | Default | Effect |
|---|---|---|
| `subsample` | `1.0` | Row fraction per tree. Below 1.0, uses Minimum Variance Sampling (gradient-weighted, unbiased). |
| `colsample` | `None` | Feature fraction eligible per tree. `None` = 1.0 for a single model, 0.85 for members inside `n_ensembles > 1` (see recipes: bagging). |

## Categorical features

| Parameter | Default | Effect |
|---|---|---|
| `cat_smoothing` | `1.0` | Prior strength for ordered target statistics; higher shrinks rare categories toward the global mean. Must be `> 0`. |
| `cat_n_permutations` | `4` | Random orderings averaged by the ordered target encoder. |
| `cat_combinations` | `None`→auto | Add all pairwise category-by-category features. `None` turns them on automatically only when the data is entirely categorical (where they help without crowding out numeric splits); set `True`/`False` to force it. Auto is skipped for very wide all-categorical data (a resource guard against the `C(n_cat, 2)` blow-up) — pass `True` there if you want them anyway. |

Which columns are categorical can be passed either to `fit(..., cat_features=[...])` or as the
`cat_features` to your ChimeraBoostRegressor/ChimeraBoostClassifier arguments depending on your use case.
Columns may be named by integer position or by column name (resolved against the DataFrame), or a
mix — e.g. `cat_features=["city", "brand"]` or `cat_features=[0, 3]`.

## Loss (regressor only)

| Parameter | Default | Effect |
|---|---|---|
| `loss` | `"RMSE"` | `"RMSE"`, `"MAE"` (median), or `"Quantile"`. |
| `alpha` | `0.5` | Quantile level for `loss="Quantile"`. |

The classifier picks its loss automatically: binary logloss for 2 classes, softmax for 3+.

## Leaf models

| Parameter | Default | Effect |
|---|---|---|
| `linear_leaves` | `None` → auto | Fit a ridge linear model per leaf over the numeric split features instead of a constant. Binary classification: on by default. Regression: the default fits both variants and keeps the validation winner (~2× fit; `True`/`False` skips the double fit). Falls back to constant below ~1000 rows. Not available with MAE/Quantile or multiclass. |
| `linear_lambda` | `1.0` | Ridge penalty on per-leaf slopes; larger is closer to a constant. |

## Cross features

| Parameter | Default | Effect |
|---|---|---|
| `cross_features` | `None` → auto | Refit with difference and product columns for the pairs of the base fit's top-6 numeric features and keep whichever model has the lower validation loss (`cross_features_selected_`, `cross_pairs_` record the outcome). Oblivious trees can only staircase a numeric interaction like `x_i < x_j`; a cross column makes it one split. Large wins on interaction-heavy data (coordinates, prices, physical units). Auto applies to RMSE regression and binary classification with ≥ 2000 rows and ≥ 2 numeric features, and skips everything else (multiclass included). `False` turns it off; explicit `True` raises for multiclass. |
| `selection_rounds` | `100` | Round budget for the internal selection fits (the linear-leaves double fit and the pre-cross base fit). Candidates are judged on their best validation loss within the budget; only the winner continues to full early stopping (~1.5× faster fits). An audition that early-stops before the budget already is the full fit. `None` runs every variant to full early stopping; the audition can occasionally pick a different variant than full runs would. |

## Ordered boosting

| Parameter | Default | Effect |
|---|---|---|
| `ordered_boosting` | `False` | Leave-one-out leaf training step. Off by default; mutually exclusive with `leaf_estimation_iterations` in the booster. |

## Early stopping

| Parameter | Default | Effect |
|---|---|---|
| `early_stopping` | `True` | Hold out a validation split and stop on a plateau. Set `False` to build a fixed `n_estimators` trees. |
| `early_stopping_rounds` | `None`→`50` | Patience when early stopping is active. `50` is the sweet spot across the Grinsztajn suite; raising it to `100`–`300` helps only large, high-signal datasets (e.g. covertype, electricity, pol) and costs ~25–35% more trees, so it is not worth it as a default — bump it yourself for that kind of data. |
| `validation_fraction` | `0.2` | Held-out fraction (stratified for classifiers). Ignored when `eval_set` is passed to `fit`. |

See [Recipes → early stopping](recipes.md#early-stopping) for `eval_set` and `groups`.

## Bagging

| Parameter | Default | Effect |
|---|---|---|
| `n_ensembles` | `None` | `None`/`1` is a single model; `≥2` averages members fit on bootstrap resamples. Reduces variance. |
| `ensemble_n_jobs` | `-1` | Worker processes fitting members concurrently, each on an equal share of the thread budget (same total cores, identical models, 1.2–2x faster wall-clock). `1` fits members sequentially. |

## System

| Parameter | Default | Effect |
|---|---|---|
| `thread_count` | `None` | numba threads. `None`/`-1` uses all cores. Affects determinism of floating-point reductions. |
| `random_state` | `None` | Seed (deterministic for a fixed `thread_count`). |
| `verbose` | `False` | Print per-round metrics. |

## `fit()` arguments

| Argument | Effect |
|---|---|
| `cat_features` | Columns to treat as categorical, by integer position and/or column name. |
| `eval_set` | `(X_val, y_val)` validation set; overrides the internal split. |
| `groups` | Group labels; keeps each group entirely in train or validation when auto-splitting. |
| `sample_weight` | Per-sample training weights (normalized to mean 1). |

## Fitted attributes

| Attribute | Meaning |
|---|---|
| `feature_importances_` | Split-gain importance per input feature, summing to 1. |
| `best_iteration_` | Trees kept after early stopping. |
| `classes_` *(classifier)* | Label values, in `predict_proba` column order. |
| `temperature_` *(classifier)* | Calibration temperature; > 1 means scores were over-confident. |
| `quantile_offset_` *(regressor)* | Split-conformal correction added to `loss="Quantile"` predictions, fitted on the validation split; 0.0 for other losses or without one. |
| `expected_value_` | SHAP baseline; set after `shap_values` (see [SHAP](shap.md)). |
| `estimators_` | Fitted members when `n_ensembles > 1`, else `None`. |
