# Parameters

Every constructor option, with its default and what it does. Full signatures live in
the [API reference](api/index.md); worked examples live in [Recipes](recipes.md).

## Speed and accuracy

| Parameter | Default | Effect |
|---|---|---|
| `quality` | `None` | A named setting on the speed/accuracy curve, from `1` (fastest) to `5` (strongest). It only sets the parameters listed in the ladder table; `None` and `3` are both exactly the shipped defaults. Where it collides with a parameter you set yourself, `quality` wins and warns. See the User Guide: [the quality ladder](recipes.md#speed-and-accuracy-the-quality-ladder). |

## Core boosting

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | `2000` | Maximum boosting rounds (trees). |
| `learning_rate` | `None` (auto) | Per-tree shrinkage. With early stopping, `None` resolves to 0.1 on data of about 15,000 training rows or more, and fades to 0.07 at 5,000 rows or fewer — see `adaptive_learning_rate`. Lower values trade more trees for a slightly better fit. |
| `adaptive_learning_rate` | `True` | Let the auto `learning_rate` depend on how much data it has, as CatBoost's automatic rate does: a linear fade from 0.07 at 5,000 training rows or fewer up to a flat 0.1 at 15,000 or more. Small data is where a lower rate pays, and it is also where the extra trees are cheap. At 15,000 training rows or more it changes nothing; `False` uses a flat 0.1 at every size. Only consulted when `learning_rate` is `None` and early stopping is on, so three paths are unaffected: bagged fits (`n_ensembles >= 2`), whose members already carry an explicit member learning rate, fits with `early_stopping=False`, and `ChimeraBoostQuantileRegressor`. |
| `depth` | `None`→auto (reg) / `6` (clf) | Tree depth (a depth-d tree makes d splits). The regressor's `None` resolves to 6 for `"RMSE"` and `"MAE"`, and to 4 for `loss="Quantile"`, where deep leaves overfit the tail. Conservative by default; raise to 8 or 10 for large, interaction-heavy regression. See the User Guide: [interaction-heavy regression](recipes.md#interaction-heavy-regression). |
| `l2_leaf_reg` | `1.0` | L2 penalty on leaf values. Higher is smoother. |
| `min_child_weight` | `1.0` (reg) / `None`→auto (clf) | Minimum hessian mass on each side of a split. The classifier's `None` adapts to dataset size: the full veto (1.0) below about 500 training rows, fading linearly to 0 above about 2000. Small data needs the veto, and oblivious trees underfit large data when it stays on. |
| `leaf_estimation_iterations` | `1` (reg) / `None`→`3` (clf) | Extra Newton refinement steps per leaf. The classifier's `None` resolves to 3, which helps small and categorical-heavy binary fits. This applies to the constant-leaf path only. It is ignored while linear leaves are active, since the per-leaf ridge already fits the optimal leaf value, and it is unavailable for multiclass, `loss="MAE"`, and `loss="Quantile"`. Setting it explicitly on a path that ignores it warns. |

## Binning

| Parameter | Default | Effect |
|---|---|---|
| `max_bins` | `128` | Histogram bins per numeric feature. Raising it can improve the fit on some data. |
| `quantize_gradients` | `True` | Fits are 20 to 25% faster at the same accuracy, because the split search runs on 15-bit quantized gradients and hessians packed into integer histograms — the quantized-training technique of Shi, Ke et al. Leaf values always use exact float gradients, and results stay deterministic for a fixed `random_state`. `False` uses exact float64 histograms. |

## Row and column sampling

| Parameter | Default | Effect |
|---|---|---|
| `subsample` | `1.0` | Row fraction per tree. Below 1.0, rows are drawn by Minimum Variance Sampling (Ibragimov & Gusev; gradient-weighted and unbiased) rather than uniformly. |
| `colsample` | `None` | Feature fraction eligible per tree. `None` means 1.0 for a single model and 0.85 for members inside a bag. See the User Guide: [bagging](recipes.md#bagging). |

## Categorical features

Tell the model which columns are categorical either at fit time,
`fit(..., cat_features=[...])`, or on the constructor,
`ChimeraBoostClassifier(cat_features=[...])`. Use the constructor when a
meta-estimator such as `GridSearchCV` needs to carry the setting. Columns may be named
by integer position or by column name (resolved against the DataFrame), or a mix of
both: `cat_features=["city", "brand"]` or `cat_features=[0, 3]`.

| Parameter | Default | Effect |
|---|---|---|
| `cat_smoothing` | `1.0` | Prior strength for ordered target statistics. Higher shrinks rare categories toward the global mean. Must be `> 0`. |
| `cat_n_permutations` | `4` | Random orderings averaged by the ordered target encoder. |
| `cat_combinations` | `None`→auto | Add all pairwise category-by-category features. `None` turns them on only when every column is categorical, where they help without crowding out numeric splits. Auto is skipped for very wide all-categorical data, since the number of pairs grows quadratically; pass `True` there if you want them anyway. See the User Guide: [categorical features](recipes.md#categorical-features). |

## Loss (regressor only)

| Parameter | Default | Effect |
|---|---|---|
| `loss` | `"RMSE"` | One of `"RMSE"`, `"MAE"` (median), `"Quantile"`, `"Huber"`, or the log-link losses `"Poisson"` (counts), `"Gamma"` (positive, right-skewed), and `"Tweedie"` (non-negative with exact zeros), whose predictions are `exp(raw) > 0`. You can also pass a custom objective instance: subclass `chimeraboost.CustomObjective`, implement `grad_hess(y, raw)` and `eval(y, raw, sample_weight=None)`, and optionally override `init` and `transform`. See the User Guide: [custom objectives](recipes.md#custom-objectives-and-metrics). |
| `alpha` | `0.5` | Quantile level for `loss="Quantile"`. See the User Guide: [quantile regression](recipes.md#quantile-regression). |
| `delta` | `1.0` | Huber transition point, in units of y (quadratic within, linear beyond). It is fixed, so scale it to your data. |
| `tweedie_variance_power` | `1.5` | Tweedie variance power, strictly between 1 (Poisson) and 2 (Gamma). |

The classifier picks its loss automatically: binary log loss for 2 classes, softmax
for 3 or more.

## Quantile head (`ChimeraBoostQuantileRegressor` only)

A whole grid of conditional quantiles from one booster. See the User Guide:
[predictive distributions](quantiles.md).

| Parameter | Default | Effect |
|---|---|---|
| `quantiles` | `0.05` to `0.95` in steps of `0.05` | The levels to estimate: ascending, unique, strictly inside (0, 1). Column `k` of `predict` is level `k`. |
| `conformalize` | `False` | Calibrate the intervals on a fold held out before the early-stopping split. Raises if that fold is too small to certify the levels you asked for. |
| `calibration_fraction` | `0.2` | Rows reserved for calibration. Ignored unless `conformalize=True`. |
| `split_projection` | `"rotate"` | How the split search scores gain across the quantile levels. Leave it at `"rotate"` unless you are experimenting: `"sum"` adds the levels up and is blind to changes in spread, and `"gram"` costs more without fitting better. |
| `exact_splits` | `False` | Score splits on the exact gain summed across every level instead of a projection. More faithful, but the split search then builds one histogram per quantile level instead of one in total, so fits are much slower and use more memory as the grid grows. A check on the projection, not a routine setting. |

Two defaults are set for this head rather than inherited: `depth` is 4, because deep
leaves overfit tail quantiles, and `min_child_weight` follows the most extreme level on
your grid, so a leaf estimating the 5% quantile keeps roughly 20 rows. `loss`,
`linear_leaves`, `cross_features`, `ordered_boosting`, `refit_full`, `n_ensembles`, and
`quality` do not apply here.

## Validation metric (both estimators)

| Parameter | Default | Effect |
|---|---|---|
| `eval_metric` | `None` | A callable `metric(y_true, y_pred[, sample_weight]) -> float` scored on the validation set each round. It drives early stopping and the internal selections in place of the training loss. `y_pred` is the prediction (probabilities for the classifier; multiclass metrics receive one-hot `y_true` and the probability matrix). Lower is better unless the callable carries a `greater_is_better = True` attribute, in which case `validation_history_` records negated values. Gradients still come from the training loss. See the User Guide: [custom metrics](recipes.md#custom-objectives-and-metrics). |

## Leaf models

| Parameter | Default | Effect |
|---|---|---|
| `linear_leaves` | `None`→auto | Fit a ridge linear model per leaf over the numeric split features instead of a constant. On by default for binary classification. For `loss="RMSE"` regression the default fits both variants and keeps the validation winner, at roughly twice the fit time; `True` or `False` skips that double fit. That double fit needs a validation split and about 1000 rows — below that, and under the other regression losses, you get constant leaves unless you set `linear_leaves=True` yourself. Unavailable with MAE, Quantile, and multiclass. See the User Guide: [linear leaves](recipes.md#interaction-heavy-regression). |
| `linear_lambda` | `1.0` | Ridge penalty on the per-leaf slopes. Larger values approach a constant leaf. |

## Cross features

| Parameter | Default | Effect |
|---|---|---|
| `cross_features` | `None`→auto | Refit with interaction columns built from the base fit's top features, then keep whichever model has the lower validation loss. Gains are largest on interaction-heavy data (coordinates, prices, physical units) and on entity-heavy categorical data. Auto applies to RMSE regression and to binary and multiclass classification with at least 2000 rows and either 2 numeric features or 1 numeric feature plus categoricals, and skips everything else; `False` turns it off. `cross_features_selected_` and `cross_pairs_` record the outcome. See [How it works](concepts.md#cross-features) for the columns it builds. |
| `selection_rounds` | `100` | Round budget for the internal selection fits (the linear-leaves double fit and the pre-cross base fit). Candidates are judged on their best validation loss within the budget, and only the winner continues to full early stopping, which makes fits about 1.5x faster. An audition that early-stops before the budget is already the full fit. `None` runs every variant to full early stopping; the short audition can occasionally pick a different variant than full runs would. |

## Ordered boosting

| Parameter | Default | Effect |
|---|---|---|
| `ordered_boosting` | `False` | A leave-one-out training step: each row's update uses its leaf's totals with that row's own contribution removed, which curbs the self-reinforcement of plain boosting (CatBoost's ordered boosting, done cheaply). It measured no better than the plain update, so leave it off unless you want to try it on your own data. Mutually exclusive with `leaf_estimation_iterations`. Ignored while linear leaves are active, since the linear-leaf update owns the training step (set `linear_leaves=False` to use it), and ignored with `loss="MAE"` and `loss="Quantile"`. Supported for multiclass. |

## Early stopping

| Parameter | Default | Effect |
|---|---|---|
| `early_stopping` | `True` | Hold out a validation split and stop on a plateau. `False` builds a fixed `n_estimators` trees. |
| `early_stopping_rounds` | `None`→`50` | Patience when early stopping is active. 50 is the best general-purpose value; raise it to 100 or 300 for large, high-signal datasets, which is the only place the extra trees pay for themselves. |
| `validation_fraction` | `0.2` | Held-out fraction (stratified for classifiers). Ignored when `eval_set` is passed to `fit`. |
| `refit_full` | `"replay"` | After early stopping has chosen the tree budget on the automatic split, retrain the winning configuration on all of the rows at that budget (rounds scaled by the train-size ratio), so the final model is not left trained on only part of your data. This is the strongest single-model setting available. The default `"replay"` reuses the winner's tree structures and refits only the leaf values, which reaches the same accuracy for about a third less fit time than a from-scratch refit. `True` grows the whole model again, which is worth it where feature interactions run deep. `False` (or `quality=2`) skips the refit. Multiclass always refits from scratch. No effect with an explicit `eval_set`, with `early_stopping=False`, with `loss="Quantile"`, or inside bagged members. See the User Guide: [full-data refit](recipes.md#full-data-refit). |

See the User Guide: [early stopping](recipes.md#early-stopping) for `eval_set` and
`groups`.

## Bagging

| Parameter | Default | Effect |
|---|---|---|
| `n_ensembles` | `None` | `None` or `1` is a single model. `2` or more averages members fit on random row subsamples, which reduces variance. Two sizes are worth knowing: `n_ensembles=8` with `refit_members=True` is the strongest setting there is, and `n_ensembles=5` with `refit_members=True` is the better deal when fit time matters — it beats a plain 8-member bag on accuracy while fitting about 20% faster. `quality=5` sets 8 members but leaves `refit_members` alone, so set it yourself. See the User Guide: [bagging](recipes.md#bagging). |
| `ensemble_n_jobs` | `-1` | Worker processes fitting members concurrently, each on an equal share of the thread budget. Same total cores, identical models, 1.2 to 2x faster wall clock. `1` fits members sequentially. |
| `max_samples` | `0.8` | Fraction of rows each member trains on, drawn without replacement. This beats the classic bootstrap on both accuracy and fit time; `1.0` restores the full-size with-replacement bootstrap. |
| `refit_members` | `False` | After a member early-stops on its out-of-bag rows, replay its own tree structure against gradients from every row, so its leaf values stop being estimated from `max_samples` of the data. The splits are untouched, which is where a bag's diversity actually lives. Costs about 10 to 17% more fit time and improved accuracy at every dataset size measured; `n_ensembles` gives the two sizes to pair it with. Regression and binary classification only, and ignored with an explicit `eval_set` or `loss="Quantile"`. |

## System

| Parameter | Default | Effect |
|---|---|---|
| `thread_count` | `None` | CPU threads used for fitting and prediction. `None` or `-1` uses all cores. Results can shift slightly between different thread counts, because the summation order changes, so keep it fixed along with `random_state` if you need runs to match exactly. See [Deployment: thread control](deployment.md#thread-control). |
| `random_state` | `None` | Seed (deterministic for a fixed `thread_count`). |
| `verbose` | `False` | Print per-round metrics. |

## `fit()` arguments

| Argument | Effect |
|---|---|
| `cat_features` | Columns to treat as categorical, by integer position and/or column name. |
| `eval_set` | `(X_val, y_val)` validation set; overrides the internal split. |
| `groups` | Group labels. Keeps each group entirely in train or validation when auto-splitting. |
| `sample_weight` | Per-sample training weights (normalized to mean 1). |
| `callbacks` | A callable or list of callables `cb(iteration, train_loss, val_loss, model)` invoked each boosting round. Returning `True` requests an early stop. |

## Fitted attributes

| Attribute | Meaning |
|---|---|
| `feature_importances_` | Split-gain importance per input feature, summing to 1. |
| `best_iteration_` | Trees kept after early stopping. |
| `classes_` *(classifier)* | Label values, in `predict_proba` column order. |
| `temperature_` *(classifier)* | Calibration temperature. Above 1 means the raw scores were over-confident. |
| `quantile_offset_` *(regressor)* | Split-conformal correction added to `loss="Quantile"` predictions, fitted on the validation split. 0.0 for other losses or without a split. |
| `expected_value_` | SHAP baseline; set after `shap_values` (see [SHAP](shap.md)). |
| `estimators_` | Fitted members when `n_ensembles > 1`, otherwise `None`. |
| `validation_history_` | Per-round validation loss recorded during fit. Empty without a validation split, and a list of member histories when bagged. |
| `linear_leaves_selected_` *(regressor)* | Which leaf variant the audition kept (`True` means linear leaves won on validation). |
| `cross_features_selected_` / `cross_pairs_` | Whether the cross-feature refit won, and the feature pairs it used. |
| `member_params_` | Member hyperparameters applied when a bagged fit auto-resolves them. |
