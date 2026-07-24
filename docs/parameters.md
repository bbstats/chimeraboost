# Parameters

For more detail, see [API reference](api.md).

## Core boosting

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | `2000` | Maximum boosting rounds (trees). |
| `learning_rate` | `None` (auto) | Per-tree shrinkage. `None` resolves to 0.1 with early stopping. Lower trades more trees for slightly better fit. |
| `depth` | `None`→auto (reg) / `6` (clf) | Tree depth (a depth-d tree is d splits). The regressor's `None` resolves to 6 for `"RMSE"`/`"MAE"` and 4 for `loss="Quantile"` (deep leaves overfit the tail quantile). Conservative by default; raise to 8–10 for large, interaction-heavy regression. |
| `l2_leaf_reg` | `1.0` | L2 penalty on leaf values. Higher is smoother. |
| `min_child_weight` | `1.0` (reg) / `None`→auto (clf) | Minimum hessian mass on each side of a split. The classifier's `None` is size-adaptive: the full veto (1.0) below ~500 training rows, fading linearly to 0 above ~2000 — oblivious trees underfit large data under a fixed veto, while small data still needs one. |
| `leaf_estimation_iterations` | `1` (reg) / `None`→`3` (clf) | Extra Newton refinement steps per leaf. The classifier's `None` is an auto default resolving to 3 (it helps small/categorical-heavy binary fits). Applies to the plain constant-leaf path only: inert while linear leaves are active (default-on for binary ≥ ~1000 rows — the per-leaf ridge already fits the second-order-optimal leaf), and not implemented for multiclass or `loss="MAE"`/`"Quantile"`. An explicitly-set value that will be ignored on the path warns. |

## Binning

| Parameter | Default | Effect |
|---|---|---|
| `max_bins` | `128` | Histogram bins per numeric feature. Raising it can improve fit in some scenarios. |
| `quantize_gradients` | `True` | Split search on ~15-bit quantized grad/hess packed into integer histograms: ~20-25% faster fits, benchmark-flat accuracy. Leaf values always use exact float gradients. Deterministic per `random_state`. `False` = exact float64 histograms. |

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
| `loss` | `"RMSE"` | `"RMSE"`, `"MAE"` (median), `"Quantile"`, `"Huber"`, or the log-link losses `"Poisson"` (counts), `"Gamma"` (positive, right-skewed), `"Tweedie"` (non-negative with exact zeros) — predictions are `exp(raw) > 0`. Or a custom objective instance: subclass `chimeraboost.CustomObjective`, implement `grad_hess(y, raw)` and `eval(y, raw, sample_weight=None)`, optionally override `init`/`transform`. |
| `alpha` | `0.5` | Quantile level for `loss="Quantile"`. |
| `delta` | `1.0` | Huber transition point, in y units (quadratic within, linear beyond; fixed, so scale it to the data). |
| `tweedie_variance_power` | `1.5` | Tweedie variance power, strictly between 1 (Poisson) and 2 (Gamma). |

The classifier picks its loss automatically: binary logloss for 2 classes, softmax for 3+.

## Validation metric (both estimators)

| Parameter | Default | Effect |
|---|---|---|
| `eval_metric` | `None` | Callable `metric(y_true, y_pred[, sample_weight]) -> float` scored on the validation set each round; drives early stopping and the internal selections instead of the training loss. `y_pred` is the prediction (probabilities for the classifier; multiclass metrics get one-hot `y_true` and the probability matrix). Lower is better unless the callable has a `greater_is_better = True` attribute (`validation_history_` then records negated values). Gradients still come from the training loss. |

## Leaf models

| Parameter | Default | Effect |
|---|---|---|
| `linear_leaves` | `None` → auto | Fit a ridge linear model per leaf over the numeric split features instead of a constant. Binary classification: on by default. Regression: the default fits both variants and keeps the validation winner (~2× fit; `True`/`False` skips the double fit). Falls back to constant below ~1000 rows. Not available with MAE/Quantile or multiclass. |
| `linear_lambda` | `1.0` | Ridge penalty on per-leaf slopes; larger is closer to a constant. |

## Cross features

| Parameter | Default | Effect |
|---|---|---|
| `cross_features` | `None` → auto | Refit with cross columns for the base fit's top features and keep whichever model has the lower validation loss (`cross_features_selected_`, `cross_pairs_` record the outcome): difference and product columns for the top-6 numeric pairs, plus group-centered columns `x_i − mean(x_i \| category)` for the top-4 numerics × top-3 categoricals. Oblivious trees can only staircase "x_i < x_j" or "above this row's category baseline"; a cross column makes each one split. Large wins on interaction-heavy data (coordinates, prices, physical units) and entity-heavy categorical data. Auto applies to RMSE regression and to binary and multiclass classification with ≥ 2000 rows and ≥ 2 numeric features — or ≥ 1 numeric plus categoricals — (multiclass judges on softmax log loss), and skips everything else. `False` turns it off. |
| `selection_rounds` | `100` | Round budget for the internal selection fits (the linear-leaves double fit and the pre-cross base fit). Candidates are judged on their best validation loss within the budget; only the winner continues to full early stopping (~1.5× faster fits). An audition that early-stops before the budget already is the full fit. `None` runs every variant to full early stopping; the audition can occasionally pick a different variant than full runs would. |

## Ordered boosting

| Parameter | Default | Effect |
|---|---|---|
| `ordered_boosting` | `False` | Leave-one-out leaf training step. Off by default; mutually exclusive with `leaf_estimation_iterations` in the booster. Ignored while linear leaves are active (the linear-leaf update owns the training step — set `linear_leaves=False` to use it) and with `loss="MAE"`/`"Quantile"`; supported for multiclass. |

## Early stopping

| Parameter | Default | Effect |
|---|---|---|
| `early_stopping` | `True` | Hold out a validation split and stop on a plateau. Set `False` to build a fixed `n_estimators` trees. |
| `early_stopping_rounds` | `None`→`50` | Patience when early stopping is active. `50` is the sweet spot across the Grinsztajn suite; raising it to `100`–`300` helps only large, high-signal datasets (e.g. covertype, electricity, pol) and costs ~25–35% more trees, so it is not worth it as a default — bump it yourself for that kind of data. |
| `validation_fraction` | `0.2` | Held-out fraction (stratified for classifiers). Ignored when `eval_set` is passed to `fit`. |
| `refit_full` | `False` | Accuracy mode: after early stopping has chosen the tree budget on the automatic split, retrain the winning configuration on 100% of the rows at that budget (rounds scaled by the train-size ratio). Roughly doubles fit time; broad accuracy gain everywhere the automatic split ran. No effect with an explicit `eval_set`, `early_stopping=False`, `loss="Quantile"`, or inside bagged members. |

See [Recipes → early stopping](recipes.md#early-stopping) for `eval_set` and `groups`.

## Bagging

| Parameter | Default | Effect |
|---|---|---|
| `n_ensembles` | `None` | `None`/`1` is a single model; `≥2` averages members fit on random row subsamples (see `max_samples`). Reduces variance. |
| `ensemble_n_jobs` | `-1` | Worker processes fitting members concurrently, each on an equal share of the thread budget (same total cores, identical models, 1.2–2x faster wall-clock). `1` fits members sequentially. |
| `max_samples` | `0.8` | Fraction of rows each member trains on, drawn without replacement. Beats the classic bootstrap on accuracy and fit time; `1.0` restores the full-size with-replacement bootstrap. |

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
| `callbacks` | A callable or list of callables `cb(iteration, train_loss, val_loss, model)` invoked each boosting round; returning `True` requests an early stop. |

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
| `validation_history_` | Per-round validation loss recorded during fit; empty without a validation split, a list of member histories when bagged. |
| `linear_leaves_selected_` *(regressor)* | Leaf variant the auto-audition kept (`True` = linear leaves won on validation). |
| `cross_features_selected_` / `cross_pairs_` | Whether the cross-feature refit won, and the feature pairs it used. |
| `member_params_` | Member hyperparameters applied when `n_ensembles > 1` auto-resolves them (see recipes: bagging). |
