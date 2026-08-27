# SHAP explanations

`model.shap_values(X)` returns exact SHAP feature attributions in the model's
additive-score space: an additive decomposition of each prediction before any
final nonlinear link.

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
phi = reg.shap_values(X_test)        # (n_samples, n_features)
base = reg.expected_value_           # baseline, set by the call above
```

## Why it can be exact

This is TreeSHAP (Lundberg et al.), in its interventional formulation, integrated over a
background distribution. TreeSHAP is exact on trees of any shape, so exactness is not
special here — what oblivious trees buy is that the exact computation is cheap and
short. A depth-`D` tree splits on at most `D` distinct features, so the coalition game
has at most `D` players and every one of its `2**D` coalitions can simply be enumerated
in a numba kernel (64 evaluations per tree at depth 6), with no clever bookkeeping. The
practical difference for you is that it is built in: no separate `shap` install, and no
sampling approximation to configure.

## Efficiency

For the default identity-link regressor losses, contributions plus the baseline
reconstruct the prediction, to floating-point tolerance:

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

## What the numbers mean

`phi[i, j]` is feature `j`'s signed contribution to the model's additive score
for row `i`, measured against `expected_value_` (the mean additive score over the
background):

- Regressor with the identity-link losses (`"RMSE"`, `"MAE"`, `"Quantile"`,
  `"Huber"`): contributions to the predicted target. For `loss="Quantile"`, the
  fitted conformal offset is folded into `expected_value_`, so `phi.sum() +
  expected_value_` still reconstructs `predict(X)`.
- Regressor with the log-link losses (`"Poisson"`, `"Gamma"`, `"Tweedie"`) or a
  custom loss with a non-identity `transform`: contributions to the raw score
  before the link. In those cases `phi.sum() + expected_value_` reconstructs
  `predict_raw(X)`, and `predict(X)` is the linked version of that raw score.
- Binary classifier: contributions to the pre-temperature log-odds of the positive
  class. Probabilities are a nonlinear squash of the margin, so the attribution lives
  in margin space, as it does in the wider SHAP ecosystem.

Per-leaf linear models are included exactly. A leaf that predicts
`intercept + slope·(x − center)` folds its slope into the attribution, so
`shap_values` explains the fitted model rather than only its split structure.

## Global importance

`shap_importances` averages the absolute contributions for a
prediction-faithful global ranking — this is `mean(abs(shap_values(X)))`,
sorted descending:

```python
reg.shap_importances(X_test, n_features=10)
# structured array of (feature, importance) rows

reg.shap_importances(X_test, n_features=10, prettified=True)
# {feature: importance} dict, CatBoost-style
```

`feature` holds names when the model was fit on a DataFrame (or when you pass
`feature_names=...`), otherwise column indices. The SHAP pass is cached by
data content, so repeated calls on the same X — with any formatting options —
compute it once; the cache is dropped on refit.

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

`expected_value_` is the mean additive score over whichever background is used.
For the default identity-link regressor losses, that is also the mean prediction.
Cost scales linearly with background size; the default sample keeps it around 3 ms
per row at depth 6 with 200 background rows.

## Bagged models

When `n_ensembles > 1`, attributions are averaged across members. For regression this
is exact, since the bag prediction is the members' mean and Shapley values are linear.
For classification it is an additive surrogate for the soft-voted probability.

## Limits

- Binary classification and regression only. Multiclass raises `NotImplementedError`.
- Attributions live in additive-score / log-odds space, rather than probability space.
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
