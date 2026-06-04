# How it works

## Oblivious trees

Every node at a given depth splits on the same `(feature, threshold)`. A depth-`d` tree
is therefore `d` splits, and a sample's leaf is a `d`-bit number: bit `k` is 1 when the
sample exceeds the threshold at level `k`. This leads to:

- **Speed.** Prediction is `d` comparisons and one array lookup. The whole forest is
  evaluated in a single numba pass parallelized over samples, so each sample loads its
  feature values once and walks every tree while they stay in cache.
- **Regularization.** A tree has only `d` splits, shared across its level, which limits
  how sharply it can carve the input space.

The trade-off is sharpness: a leaf-wise tree can isolate a local region in fewer splits
than an oblivious tree, which matters on clean, high-signal data. Raising `depth` and
enabling [linear leaves](#linear-leaves) recover most of that.

## Histogram binning

Numeric features are bucketed into at most `max_bins` (default 128) bins once, up front.
Splits are searched over bin edges rather than raw values, which is what makes the
histogram pass fast. Missing values route to a dedicated bin, so NaNs are handled
directly at fit and predict time — no imputation.

## Categorical features

Categoricals are encoded with **ordered target statistics**, i.e. the CatBoost approach: each
category is replaced by a running estimate of the target computed under a random ordering
of the rows, so a row never sees its own label. Several orderings
(`cat_n_permutations`, default 4) are averaged to cut variance, and rare categories are
shrunk toward the global mean by `cat_smoothing`. Pass the columns (by integer position or
column name) to `fit(..., cat_features=[...])`; everything else is automatic.

`cat_combinations=True` additionally builds all pairwise category-by-category features.

## Leaf values and linear leaves

By default each leaf predicts a single constant value. With `linear_leaves`, a leaf
instead fits a small ridge **linear model** over the numeric features the tree split on,
adding local slope where a constant underfits smooth structure. Leaves with too few rows
fall back to norma behavior to reduce overfitting small datasets. Linear leaves are on by
default for binary classification, but off for regression.

`hs_lambda` (hierarchical shrinkage) optionally pulls each leaf value toward its
ancestors' value.

## Probability calibration

After fitting, the classifier scales its raw scores by a single temperature chosen on
the validation split to minimize log loss. The scaling is monotonic, so AUC and accuracy
are unchanged while probabilities (i.e. with `predict_proba`) becomes better calibrated.
The fitted value is exposed as `temperature_`.

## Bagging and subsampling

`n_ensembles` trains independent members on bootstrap resamples and averages them —
predictions for regression, calibrated probabilities for classification — to reduce
variance. Within a single model, `subsample < 1.0` uses Minimum Variance Sampling:
rows are drawn with probability tied to gradient magnitude and reweighted to stay
unbiased, concentrating effort on the rows that still carry signal.

## SHAP

See [SHAP](shap.md) for details.
