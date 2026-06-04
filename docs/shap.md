# SHAP explanations

`model.shap_values(X)` returns exact SHAP feature attributions: an additive
decomposition of each prediction into per-feature contributions.

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
phi = reg.shap_values(X_test)        # (n_samples, n_features)
base = reg.expected_value_           # baseline, set by the call above
```

## Exact, not sampled

Most SHAP tooling approximates by sampling feature coalitions, because exact
computation is expensive on trees of arbitrary shape. ChimeraBoost computes it exactly.
Oblivious trees split on the same feature at every node of a level, so a depth-`D` tree
involves at most `D` distinct features. The coalition game therefore has at most `D`
players, and all `2**D` or fewer coalitions are enumerated directly in a numba kernel
(64 evaluations per tree at depth 6). This is the interventional formulation of
TreeSHAP, integrated over a background distribution.

## Efficiency

Contributions plus the baseline reconstruct the prediction, to floating-point tolerance:

```python
i = 0
recon = phi[i].sum() + base
assert abs(recon - reg.predict(X_test)[i]) < 1e-6   # holds to ~1e-14
```

This is the Shapley efficiency property, and it is what lets `shap_values` stand in as
the model's own accounting of a prediction. Gain importance (`feature_importances_`)
has no such guarantee: it measures which features were split on, ignores the per-leaf
linear models, and does not decompose any individual prediction.

## What the numbers mean

`phi[i, j]` is feature `j`'s signed contribution to the raw score of row `i`, measured
against `expected_value_` (the mean raw score over the background):

- Regressor: contributions to the predicted target.
- Binary classifier: contributions to the pre-temperature log-odds of the positive
  class. Probabilities are a nonlinear squash of the margin, so the attribution is in
  margin space, as in the wider SHAP ecosystem.

Per-leaf linear models are included exactly. A leaf that predicts
`intercept + slope·(x − center)` folds its slope into the attribution, so `shap_values`
explains the fitted model rather than only its split structure.

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

## Background distribution

SHAP attributions are defined against a reference: how a feature moves the prediction
away from a typical input. That reference is the background, which defaults to a sample
of the training data captured at fit. Override it to explain against a specific cohort:

```python
phi = clf.shap_values(X_test, X_background=X_reference)
```

`expected_value_` is the mean prediction over whichever background is used. Cost scales
linearly with background size; the default sample keeps it around 3 ms per row at depth
6 with 200 background rows.

## Bagged models

When `n_ensembles > 1`, attributions are averaged across members. For regression this
is exact, since the bag prediction is the members' mean and Shapley values are linear.
For classification it is an additive surrogate for the soft-voted probability.

## Limits

- Binary and regression only. Multiclass raises `NotImplementedError`.
- Attributions are in raw-score / log-odds space, not probability space.
- They explain this model's behavior; they are not causal effects.

## Versus `feature_importances_`

| | `feature_importances_` | `shap_values` |
|---|---|---|
| Measures | total split gain | contribution to each prediction |
| Granularity | global only | per-prediction and global |
| Includes linear leaves | no | yes |
| Reconstructs the output | no | yes |
| Cost | free (tracked at fit) | milliseconds per row |

Use gain for a free global glance; use SHAP for a faithful or per-prediction explanation.
