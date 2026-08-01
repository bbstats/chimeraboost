# Recipes

Imports used throughout:

```python
import numpy as np
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
```

## Regression and classification

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
y_pred = reg.predict(X_test)

clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
labels = clf.predict(X_test)            # original label values
proba = clf.predict_proba(X_test)       # columns follow clf.classes_
```

A plain `fit(X, y)` early-stops on an internal holdout. See [Early
stopping](#early-stopping).

## Speed and accuracy: the `quality` ladder

`quality` picks a setting between `1` (fastest) and `5` (strongest). It is shorthand:
each rung just sets the parameters in the last column of the table, and `quality=3` is
exactly the shipped defaults.

```python
reg = ChimeraBoostRegressor(quality=1, random_state=0).fit(X_train, y_train)
```

| `quality` | name | fit time | sets |
|---|---|--:|---|
| `1` | fast | **1.9x** | `linear_leaves=True`, `cross_features=False`, `refit_full=False` |
| `2` | balanced | 5.3x | `refit_full=False` |
| `3` | accurate *(= default)* | 6.9x | nothing (these are the defaults) |
| `4` | ensemble | 18.1x | `n_ensembles=5` |
| `5` | max | 26.0x | `n_ensembles=8` |

Fit times are multiples of the fastest gradient boosting library measured on the same
data, averaged over a benchmark suite. Every rung buys accuracy for its extra time, so
none of them is a bad deal; pick the one that fits your time budget.

The default is rung 3, the strongest setting that does not build an ensemble. Rung 2 is
the same model without the [full-data refit](#full-data-refit). It saves less time than
the gap suggests, because the default now does that refit by replaying tree structures
instead of re-growing them.

**What rung 1 gives up.** By default ChimeraBoost auditions its own configuration:
constant leaves against linear leaves, plain features against cross features. That
costs two to four boosting fits. `quality=1` pins both decisions and fits once.
Reach for it on numeric-heavy data and large parameter sweeps. Prefer `quality=2`
when categorical columns dominate, where the saving is smaller (categorical
preprocessing is a fixed cost that skipping auditions cannot touch) and the accuracy
loss is larger.

**Rungs 4 and 5 do not build on rung 3.** `refit_full` does nothing inside bagged
members, since their out-of-bag rows already act as an eval set, so the ensemble
rungs start from the plain defaults.

If you set `quality` alongside a parameter it controls, `quality` wins and warns.
Drop it to set those parameters yourself.

## Categorical features

Pass your categoricals as `cat_features`, by integer position or, for a DataFrame, by
column name (or a mix of both). They are encoded with ordered target statistics
(CatBoost-style), so there is no one-hot or `LabelEncoder` step. Categorical columns
can hold strings or objects while the rest of the matrix stays numeric.

```python
# columns 0 and 3 are categorical (e.g. "city", "device_type")
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X, y, cat_features=[0, 3])

# equivalently, by name when X is a DataFrame
clf.fit(df, y, cat_features=["city", "device_type"])
```

`cat_combinations` adds all pairwise category-by-category features. They help when the
target depends on categorical interactions, but they can crowd out numeric splits on
mixed data, so the default (`None`) turns them on only when every column is
categorical. Set `True` to force them on (for example on mixed data where you know the
interactions matter) or `False` to turn them off.

## Missing values

NaNs route to their own histogram bin, so no imputation is needed. This works for
numeric and categorical columns alike, at fit and at predict time.

```python
X[mask] = np.nan
reg = ChimeraBoostRegressor(random_state=0).fit(X, y)   # handled directly
```

## Quantile regression

### A whole grid at once

[`ChimeraBoostQuantileRegressor`](quantiles.md) fits every level from one booster and
guarantees the levels never cross. This is the recommended way to get prediction
intervals.

```python
from chimeraboost import ChimeraBoostQuantileRegressor
from chimeraboost import quantile_metrics as qm

model = ChimeraBoostQuantileRegressor(random_state=0).fit(X_train, y_train)

Q = model.predict(X_test)                # (n_samples, 19), column k is model.quantiles_[k]
median = Q[:, 9]                         # the 0.50 column of the default grid
assert np.all(np.diff(Q, axis=1) >= 0)   # holds by construction

lo, hi = model.predict(X_test, kind="interval", alpha=0.1).T   # central 90%
point = model.predict(X_test, kind="mean")                     # tau-integrated mean

print(qm.format_report(model.report(X_test, y_test)))  # CRPS, pinball, coverage, width
```

The default grid is `0.05, 0.10, ... 0.95`. `kind="interval"` reads the two levels off
that grid and raises if `alpha` asks for levels it was not fitted for, so pass your own
grid when you want a specific interval:

```python
model = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.5, 0.9],
                                      random_state=0).fit(X_train, y_train)
lo, med, hi = model.predict(X_test).T
```

Raw intervals come out too wide, because boosting shrinks every round's step and the
grid never fully contracts. `conformalize=True` calibrates them:

```python
model = ChimeraBoostQuantileRegressor(quantiles=[0.1, 0.5, 0.9], conformalize=True,
                                      random_state=0).fit(X_train, y_train)
lo, med, hi = model.predict(X_test).T
print(model.conformal_scale_)                              # < 1 means the fit was too wide
print(np.mean((y_test >= lo) & (y_test <= hi)))            # ~0.80, the nominal level
```

The calibration fold is carved out before the early-stopping split, so it influences
neither the fit nor the stopping point. See [Interval
calibration](quantiles.md#interval-calibration) for what it guarantees.

### One level at a time

Set `loss="Quantile"` and the level `alpha` on the ordinary regressor. For a prediction
interval that way, fit one model per quantile — note that nothing stops these three from
crossing each other:

```python
lo = ChimeraBoostRegressor(loss="Quantile", alpha=0.05, random_state=0).fit(X_train, y_train)
md = ChimeraBoostRegressor(loss="Quantile", alpha=0.50, random_state=0).fit(X_train, y_train)
hi = ChimeraBoostRegressor(loss="Quantile", alpha=0.95, random_state=0).fit(X_train, y_train)

lower, median, upper = lo.predict(X_test), md.predict(X_test), hi.predict(X_test)
```

`loss="MAE"` gives median regression; `loss="RMSE"` (the default) is squared error.

Quantile models default to `depth=4` rather than the squared-error default of `6`.
An extreme conditional quantile is estimated from the points inside each leaf, so
deep, sparse leaves overfit the tails and the predicted quantiles collapse toward the
median on held-out data. Predictions also carry a split-conformal correction
(`quantile_offset_`) fitted on the early-stopping validation split, which brings
coverage at the tails back near its nominal level. With `early_stopping=False` and no
`eval_set` there is no split to calibrate on, and the raw quantiles are returned;
these are usually too narrow.

## Counts, positive targets, zero-inflated targets

The log-link losses keep predictions positive and match the noise model:

```python
counts = ChimeraBoostRegressor(loss="Poisson").fit(X, y_counts)        # y >= 0
costs = ChimeraBoostRegressor(loss="Gamma").fit(X, y_positive)         # y > 0
claims = ChimeraBoostRegressor(loss="Tweedie",
                               tweedie_variance_power=1.5).fit(X, y)   # y >= 0, exact zeros
```

`loss="Huber"` is squared error that tolerates outliers, switching to absolute error
beyond `delta` (measured in units of y).

## Custom objectives and metrics

Subclass `CustomObjective` with the gradient and hessian of your loss on the raw
score, then pass an instance as `loss`. `eval_metric` swaps the early-stopping metric
on either estimator:

```python
from chimeraboost import ChimeraBoostRegressor, CustomObjective

class LogCosh(CustomObjective):          # smooth MAE
    def grad_hess(self, y, raw):
        t = np.tanh(raw - y)
        return t, 1.0 - t**2 + 1e-6
    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.logaddexp(raw - y, y - raw) - np.log(2),
                                weights=sample_weight))

model = ChimeraBoostRegressor(loss=LogCosh()).fit(X_train, y_train)

def mae(y_true, y_pred):                 # early-stop on MAE instead of RMSE
    return float(np.mean(np.abs(y_true - y_pred)))

model = ChimeraBoostRegressor(eval_metric=mae).fit(X_train, y_train)
```

If larger values of your metric are better, set `mae.greater_is_better = True` on the
callable; `validation_history_` then records negated values. Define custom objectives
at module level, because bagged members fit in worker processes and the loss has to
pickle.

## Multiclass classification

Nothing to configure. The classifier switches to softmax when it sees 3 or more
classes, and `classes_` preserves your original labels.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X, y)   # 3+ classes
proba = clf.predict_proba(X_test)        # shape (n_samples, n_classes)
```

`linear_leaves` and `shap_values` cover binary classification and regression only.
Multiclass uses constant leaves, and `shap_values` raises `NotImplementedError`.

## Sample weights

```python
w = np.where(y_train == 1, 5.0, 1.0)     # upweight the positive class
clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, sample_weight=w)
```

Weights are normalized to mean 1 internally and apply to training only. The
early-stopping metric stays unweighted.

## Bagging

`n_ensembles` trains that many models on random row samples and averages them.
Regressors average predictions; classifiers soft-vote calibrated probabilities. Each
member trains on `max_samples` (default 0.8) of the rows drawn without replacement,
which beats the classic bootstrap on both accuracy and fit time, and early-stops on
its own unsampled rows. With `groups`, each member draws whole groups instead of
rows and early-stops on the groups it never saw.

```python
reg = ChimeraBoostRegressor(n_ensembles=8, random_state=0).fit(X_train, y_train)
```

Use `n_ensembles=8`: it is stronger than 5 at similar cost. Avoid `n_ensembles=2`,
which scores worse than a single model. If you want the most accuracy per unit of
fit time, turn on [`refit_members`](#reclaiming-the-per-member-data-tax) as well —
five refit members beat eight plain ones on both accuracy and speed.

Inside a bag, parameters left on auto resolve to member defaults tuned for averaging,
currently `learning_rate=0.15` and `colsample=0.85`, because averaging tolerates
coarser and cheaper members. The fit warns once when this happens (a filterable
`UserWarning`), `member_params_` records what was applied, and passing explicit values
disables it.

Members fit in parallel worker processes by default, splitting the thread budget so a
bagged fit uses the same cores a single fit would. Pass `ensemble_n_jobs=1` to fit
them sequentially.

`feature_importances_` and `shap_values` average across the bag automatically.

### Reclaiming the per-member data tax

Each member's leaf values are estimated from only the rows it sampled — the
out-of-bag rows it early-stopped on never inform them. `refit_members=True`
replays each member's own tree structure against gradients from every row once
early stopping is finished:

```python
reg = ChimeraBoostRegressor(n_ensembles=8, refit_members=True,
                            random_state=0).fit(X_train, y_train)
```

Only the leaf values change; the splits stay exactly as each member's own
sample grew them, which is where a bag's diversity actually lives. Expect
roughly 10-17% more fit time, the higher end on larger suites. Because each
member is individually stronger, you can also spend the gain on fewer members:
`n_ensembles=5, refit_members=True` beat a plain eight-member bag on accuracy
while fitting about 20% faster. Regression and binary classification only —
multiclass members would need a full refit rather than a cheap replay, so the
flag is ignored there.

## Full-data refit

Once early stopping has chosen the tree budget on the automatic validation split, the
winning configuration is retrained on all of the rows at that budget, so the final
model is not left trained on only 80% of your data. This is the strongest
single-model setting available, with the largest gains on small or high-signal data.
It does nothing inside bagged members, whose held-out rows already serve as an eval
set, so `n_ensembles` builds on the non-refit model instead of stacking with this.

`refit_full` takes three values:

| Value | What it does |
|---|---|
| `"replay"` *(default)* | Reuse the winner's tree structures and refit only the leaf values. |
| `True` | Grow the whole model again from scratch. |
| `False` | Skip the refit. Same as `quality=2`. |

Growing trees is most of a fit, and it is a search. A from-scratch refit spends that
search rediscovering structures the early-stopping winner already found. `"replay"`
walks the winner's splits round by round against gradients computed on all the rows and
refits only the leaf values, so every leaf value still sees the held-out rows while the
split search is paid for once. That costs about a third of what `True` costs, for the
same accuracy, which is why it is the default.

What `"replay"` gives up is that the splits themselves still come from the training
subset. That matters most where feature interactions run deep, so reach for `True`
there. Multiclass ignores `"replay"` and always refits from scratch.

```python
# skip the refit entirely when fit time matters more than the last bit of accuracy
reg = ChimeraBoostRegressor(refit_full=False, random_state=0).fit(X_train, y_train)
# equivalently
reg = ChimeraBoostRegressor(quality=2, random_state=0).fit(X_train, y_train)
```

## Early stopping

Early stopping is on by default. With no `eval_set`, the estimator holds out a
validation split (`validation_fraction=0.2`, stratified for classifiers), stops after
a plateau, and keeps the best round.

```python
# default: automatic internal holdout
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
print(m.best_iteration_)

# explicit validation set (overrides the internal split)
m = ChimeraBoostRegressor(random_state=0)
m.fit(X_train, y_train, eval_set=(X_val, y_val))

# grouped split: keep each group entirely in train or validation
m.fit(X_train, y_train, groups=subject_ids)

# fixed number of trees, no stopping
m = ChimeraBoostRegressor(early_stopping=False, n_estimators=500, random_state=0)
m.fit(X_train, y_train)
```

After fitting, `validation_history_` holds the per-round validation loss, and the
regressor's `staged_predict(X)` yields the prediction after each successive tree. It
is not defined for a bagged ensemble.

## Calibrated probabilities

`predict_proba` is temperature-scaled on the validation split to minimize log loss.
The scaling is monotonic, so `predict()`, AUC, and accuracy are unchanged while the
probabilities themselves become better calibrated.

```python
clf = ChimeraBoostClassifier(random_state=0).fit(X_train, y_train)
proba = clf.predict_proba(X_test)        # already calibrated
print(clf.temperature_)                  # > 1 means raw scores were over-confident
```

## Feature importance

`feature_importances_` is total split gain per input column, normalized to sum to 1
(averaged across the bag when `n_ensembles > 1`).

```python
m = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
for j in np.argsort(m.feature_importances_)[::-1][:5]:
    print(f"feature {j}: {m.feature_importances_[j]:.3f}")
```

Gain tells you what the trees split on. It says nothing about how much a feature moves
any given prediction, and it ignores the per-leaf linear models. For a faithful
decomposition of the output, use [SHAP](shap.md).

## Cross-validation and hyperparameter search

The estimators are standard scikit-learn objects:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

scores = cross_val_score(
    ChimeraBoostRegressor(random_state=0), X, y, cv=5,
    scoring="neg_root_mean_squared_error",
)

search = GridSearchCV(
    ChimeraBoostRegressor(random_state=0),
    {"depth": [6, 8, 10], "l2_leaf_reg": [1.0, 3.0]},
    cv=5,
)
search.fit(X, y)
print(search.best_params_)
```

To pass `cat_features` through a search, set it on the constructor,
`ChimeraBoostClassifier(cat_features=["city", "brand"])`, so the meta-estimator
carries it. A fit-only keyword argument cannot survive the clone.

## Save and load a model

A fitted estimator pickles like any scikit-learn object:

```python
import joblib

joblib.dump(reg, "model.joblib")
reg = joblib.load("model.joblib")
```

## Interaction-heavy regression

The default `depth=6` is conservative to protect small data. On large,
interaction-heavy problems, raise `depth` to give each tree more room:

```python
reg = ChimeraBoostRegressor(depth=10, random_state=0).fit(X_train, y_train)
```

Per-leaf linear models add local slope inside each leaf. They are on by default for
binary classification; for regression the default fits both variants and keeps the one
with the lower validation loss. Set the parameter explicitly to skip that double fit:

```python
reg = ChimeraBoostRegressor(linear_leaves=True, random_state=0).fit(X_train, y_train)
```

## Reproducibility and threads

```python
m = ChimeraBoostRegressor(
    random_state=0,        # deterministic for a fixed thread count
    thread_count=4,        # numba threads; None or -1 uses all cores
).fit(X_train, y_train)
```
