# Parameters

Every constructor option for `ChimeraBoostRegressor` and `ChimeraBoostClassifier`,
grouped by what it controls, with both defaults shown where they differ. The defaults
are the benchmarked configuration, so most problems need no tuning. The **Tune?**
column flags the few worth reaching for; in practice that is usually just `depth`,
`n_ensembles`, and `random_state`.

For exact types and the auto-generated reference, see the [API reference](api.md).

## Core boosting

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `iterations` | `2000` | rarely | Maximum boosting rounds. With early stopping on, an upper bound rather than a target. |
| `learning_rate` | `None` (auto) | rarely | Per-tree shrinkage. `None` resolves to 0.1 with early stopping. Lower trades more trees for slightly better fit. |
| `depth` | `None`→auto | **yes** | Tree depth (a depth-d tree is d splits). The regressor's `None` resolves to 6 for squared/absolute error and 4 for `loss="Quantile"` (deep leaves overfit the tail quantile). Conservative by default; raise to 8–10 for large, interaction-heavy regression. |
| `l2_leaf_reg` | `1.0` | rarely | L2 penalty on leaf values. Higher is smoother. |
| `min_child_weight` | `1.0` (reg) / `None`→auto (clf) | rarely | Minimum hessian mass on each side of a split. The classifier's `None` is size-adaptive: full veto below ~500 rows, off above ~2000. |
| `leaf_estimation_iterations` | `1` (reg) / `3` (clf) | rarely | Extra Newton refinement steps per leaf. Helps logloss; little effect on squared error. |

## Binning

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `max_bins` | `128` | no | Histogram bins per numeric feature. Raising it overfits noise and slows builds without generalizing. |

## Row and column sampling

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `subsample` | `1.0` | optional | Row fraction per tree. Below 1.0, uses Minimum Variance Sampling (gradient-weighted, unbiased). |
| `colsample` | `1.0` | optional | Feature fraction eligible per tree. |

## Categorical features

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `cat_smoothing` | `1.0` | rarely | Prior strength for ordered target statistics; higher shrinks rare categories toward the global mean. Must be `> 0` (a Bayesian pseudocount; `0` is rejected). |
| `cat_n_permutations` | `4` | no | Random orderings averaged by the ordered target encoder. |
| `cat_combinations` | `False` | optional | Add all pairwise category-by-category features. Helps mostly-categorical data, can crowd out numerics on mixed data. |

Which columns are categorical can be passed either to `fit(..., cat_features=[...])` or as the
`cat_features` constructor argument. The constructor form lets `GridSearchCV`/`Pipeline` carry
it; a value passed to `fit` overrides the constructor one.

## Loss (regressor only)

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `loss` | `"RMSE"` | task | `"RMSE"`, `"MAE"` (median), or `"Quantile"`. |
| `alpha` | `0.5` | task | Quantile level for `loss="Quantile"`. |

The classifier picks its loss automatically: binary logloss for 2 classes, softmax for 3+.

## Leaf models

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `linear_leaves` | `False` (reg) / `None`→auto (clf) | optional | Fit a ridge linear model per leaf over the numeric split features instead of a constant. On by default for binary classification; falls back to constant below ~1000 rows. Not available with MAE/Quantile or multiclass. |
| `linear_lambda` | `1.0` | optional | Ridge penalty on per-leaf slopes; larger is closer to a constant. |
| `hs_lambda` | `0.0` | optional | Hierarchical shrinkage: above 0, leaf values are pulled toward their ancestors, hardest for deep or low-mass leaves. |

## Ordered boosting

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `ordered_boosting` | `False` | no | Leave-one-out leaf training step. Off by default; mutually exclusive with `leaf_estimation_iterations` in the booster. Tends to *hurt* accuracy here (the plain Newton path plus leaf refinement wins broadly), so leave it off unless you have a specific reason. |

## Early stopping

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `early_stopping` | `True` | rarely | Hold out a validation split and stop on a plateau. Set `False` to build a fixed `iterations` trees. |
| `early_stopping_rounds` | `None`→`50` | rarely | Patience when early stopping is active. |
| `validation_fraction` | `0.2` | rarely | Held-out fraction (stratified for classifiers). Ignored when `eval_set` is passed to `fit`. |

See [Recipes → early stopping](recipes.md#early-stopping) for `eval_set` and `groups`.

## Bagging

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `n_ensembles` | `None` | **yes** | `None`/`1` is a single model; `≥2` averages members fit on bootstrap resamples. Reduces variance. |
| `ensemble_n_jobs` | `1` | optional | Processes used to fit members; `-1` uses all cores. |

## System

| Parameter | Default | Tune? | Effect |
|---|---|:--:|---|
| `thread_count` | `None` | optional | numba threads. `None`/`-1` uses all cores. Affects determinism of floating-point reductions. |
| `random_state` | `None` | **yes** | Seed (deterministic for a fixed `thread_count`). |
| `verbose` | `False` | optional | Print per-round metrics. |

## `fit()` arguments

| Argument | Effect |
|---|---|
| `cat_features` | Column indices to treat as categorical. |
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
| `expected_value_` | SHAP baseline; set after `shap_values` (see [SHAP](shap.md)). |
| `estimators_` | Fitted members when `n_ensembles > 1`, else `None`. |
