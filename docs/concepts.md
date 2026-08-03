# How it works

## Oblivious trees

The tree type is CatBoost's, and so is much of what follows on this page — see
[where the ideas come from](attribution.md).

Every node at a given depth splits on the same `(feature, threshold)`. A depth-`d` tree
is therefore `d` splits, and a sample's leaf is a `d`-bit number: bit `k` is 1 when the
sample exceeds the threshold at level `k`. Two things follow:

- **Speed.** Prediction is `d` comparisons and one array lookup. The whole forest is
  evaluated in a single numba pass parallelized over samples, so each sample loads its
  feature values once and walks every tree while they stay in cache.
- **Regularization.** A tree has only `d` splits, shared across its level, which limits
  how sharply it can carve up the input space.

The trade-off is sharpness. A leaf-wise tree can isolate a local region in fewer
splits, which matters on clean, high-signal data. Raising `depth` and enabling
[linear leaves](#leaf-values-and-linear-leaves) recover most of that.

## Histogram binning

Numeric features are bucketed into at most `max_bins` (default 128) bins once, up
front. Splits are searched over bin edges rather than raw values, which is what makes
the histogram pass fast — the approach LightGBM introduced. Missing values route to
their own bin, so NaNs are handled directly at fit and predict time, with no imputation.

## Categorical features

Categoricals are encoded with **ordered target statistics**, the CatBoost approach.
Each category is replaced by a running estimate of the target computed under a random
ordering of the rows, so a row never sees its own label. Several orderings
(`cat_n_permutations`, default 4) are averaged to cut variance, and rare categories are
shrunk toward the global mean by `cat_smoothing`. Pass the columns (by integer position
or column name) to `fit(..., cat_features=[...])`; everything else is automatic.

`cat_combinations` additionally builds all pairwise category-by-category features. The
default (`None`) turns them on automatically when every column is categorical.

## Cross features

An oblivious tree can only split on one column at a time, so a rule like "x₁ is
greater than x₂" has to be approximated by a staircase of axis-aligned splits, which
costs depth and generalizes poorly. Adding the column `x₁ − x₂` turns that rule into a
single split.

`cross_features` builds those columns automatically, following OpenFE's approach to
automated feature generation: differences and products for the top numeric pairs, plus
group-centered columns `x − mean(x | category)` that express "above this row's category
baseline". The model then fits with and without them and
keeps whichever scores better on validation, so the extra columns cannot hurt beyond
their fit-time cost. The payoff is largest on data with real geometric or arithmetic
structure, such as coordinates, prices, and physical units.

## Leaf values and linear leaves

By default each leaf predicts a single constant value. With `linear_leaves`, a leaf
instead fits a small ridge **linear model** over the numeric features the tree split
on, adding local slope where a constant underfits smooth structure — piece-wise-linear
trees, from Shi et al. and the older model-tree literature. Leaves with too few
rows fall back to constant behavior, which limits overfitting on small datasets. Linear
leaves are on by default for binary classification. The regression default fits both
variants and keeps whichever reaches the lower validation loss; set `True` or `False`
to skip the double fit.

## Probability calibration

After fitting, the classifier applies temperature scaling (Guo et al.): it scales its
raw scores by a single temperature chosen on the validation split to minimize log loss.
The scaling is monotonic, so AUC and
accuracy are unchanged while the probabilities from `predict_proba` become better
calibrated. The fitted value is exposed as `temperature_`.

## Bagging and subsampling

`n_ensembles` trains independent members on random row subsamples (`max_samples`,
default 0.8, drawn without replacement) and averages them to reduce variance:
predictions for regression, calibrated probabilities for classification.

Within a single model, `subsample < 1.0` uses Minimum Variance Sampling (Ibragimov &
Gusev, the row sampler CatBoost uses). Rows are drawn with probability tied to gradient
magnitude and reweighted to stay unbiased, which concentrates effort on the rows that
still carry signal.

## SHAP

See [SHAP](shap.md).
