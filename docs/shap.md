# SHAP explanations

`model.shap_values(X)` returns exact SHAP feature attributions: an additive
decomposition of each prediction into per-feature contributions. It is built in — no
separate `shap` install, and no sampling approximation to configure.

```python
from chimeraboost import ChimeraBoostRegressor

reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
phi = reg.shap_values(X_test)        # (n_samples, n_features)
base = reg.expected_value_           # baseline, set by the call above
```

`phi` has one column per column you passed to `fit`, in the same order.

## What the numbers mean

`phi[i, j]` is feature `j`'s signed contribution to the raw score of row `i`, measured
against `expected_value_` (the mean raw score over the background):

- Regressor with `loss="RMSE"`, `"MAE"`, `"Huber"` or `"Quantile"`: contributions to
  the predicted target.
- Regressor with a log-link loss (`"Poisson"`, `"Gamma"`, `"Tweedie"`): contributions
  to `log(prediction)`, because those losses predict `exp(raw score)`. The same applies
  to a custom objective with a non-identity `transform`.
- Binary classifier: contributions to the log-odds of the positive class, before the
  temperature-scaling step that calibrates `predict_proba` (see
  [Calibrated probabilities](recipes.md#calibrated-probabilities)). Probabilities are a
  nonlinear squash of the margin, so the attribution lives in margin space, as it does
  in the wider SHAP ecosystem.

Per-leaf linear models are included exactly. A leaf that predicts
`intercept + slope·(x − center)` folds its slope into the attribution, so
`shap_values` explains the fitted model rather than only its split structure.

## Contributions sum to the prediction

Contributions plus the baseline reconstruct the prediction, to floating-point
tolerance:

```python
i = 0
recon = phi[i].sum() + base
assert abs(recon - reg.predict(X_test)[i]) < 1e-6   # holds to ~1e-14
```

This is the Shapley efficiency property, and it is what lets `shap_values` stand in as
the model's own accounting of a prediction. Gain importance
(`feature_importances_`) has no such guarantee: it measures which features were split
on, ignores the per-leaf linear models, and does not decompose any individual
prediction.

## Cost

Milliseconds per row, growing with the number of trees kept, the tree depth, and the
size of the [background](#background-distribution). A 100-tree depth-6 model with the
default 200-row background costs roughly 5 ms per row on a desktop CPU; a few hundred
trees cost proportionally more.

## Global importance

Average the absolute contributions for a prediction-faithful global ranking:

```python
import numpy as np
global_importance = np.abs(phi).mean(axis=0)
for j in np.argsort(global_importance)[::-1][:10]:
    print(f"feature {j}: {global_importance[j]:.4f}")
```

## Explaining one prediction

```python
i = 0
print(f"baseline: {base:.3f}")
for j in np.argsort(np.abs(phi[i]))[::-1][:5]:
    direction = "up" if phi[i, j] > 0 else "down"
    print(f"  feature {j}: {phi[i, j]:+.3f} ({direction})")
print(f"  prediction: {phi[i].sum() + base:.3f}")
```

## Plotting with the `shap` package

ChimeraBoost draws nothing itself and does not depend on the `shap` package. What it
returns is the layout that package expects for a single-output model — values of shape
`(n_samples, n_features)` plus a scalar base value — so if you have `shap` installed,
wrap the two and use its plots:

```python
import shap                                          # pip install shap

explanation = shap.Explanation(values=phi, base_values=base, data=X_test)
shap.plots.beeswarm(explanation)
shap.plots.waterfall(explanation[0])
```

Add `feature_names=list_of_names` if `X_test` is a plain array rather than a DataFrame.

## Background distribution

SHAP attributions are defined against a reference: how a feature moves the prediction
away from a typical input. That reference is the background, which defaults to a sample
of the training data captured at fit. Override it to explain against a specific cohort:

```python
phi = reg.shap_values(X_test, X_background=X_reference)
```

`expected_value_` is the mean prediction over whichever background is used. Cost scales
linearly with background size, so a larger cohort is a slower explanation.

## Bagged models

When `n_ensembles > 1`, attributions are averaged across members. For regression this
is exact, since the bag prediction is the members' mean and Shapley values are linear.
For classification it is an additive surrogate for the soft-voted probability.

## Limits

- Binary classification and regression only. Multiclass raises `NotImplementedError`.
- Attributions live in raw-score / log-odds space, rather than probability space.
- They explain this model's behavior. They are not causal effects.

## Compared with `feature_importances_`

| | `feature_importances_` | `shap_values` |
|---|---|---|
| Measures | total split gain | contribution to each prediction |
| Granularity | global only | per-prediction and global |
| Includes linear leaves | no | yes |
| Reconstructs the output | no | yes |
| Cost | free (tracked at fit) | milliseconds per row |

Use gain for a free global glance, and SHAP for a faithful or per-prediction
explanation.

## Why the exact computation is cheap

This is TreeSHAP (Lundberg et al.), in its interventional formulation, integrated over
a [background distribution](#background-distribution). TreeSHAP is exact on trees of
any shape, so exactness is not special here — what oblivious trees buy is that the
exact computation is cheap and short. A depth-`D` tree splits on at most `D` distinct
features, so the coalition game has at most `D` players and every one of its `2**D`
coalitions can simply be enumerated in a numba kernel (64 evaluations per tree at
depth 6), with no clever bookkeeping.
