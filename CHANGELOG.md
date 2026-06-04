# Changelog

All notable changes to ChimeraBoost are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Added
- **Exact SHAP feature attributions** (`model.shap_values(X)`). Interventional
  TreeSHAP computed exactly — not approximated — by exploiting the oblivious tree
  structure: a depth-D tree touches at most D distinct features, so the Shapley
  coalition game is enumerated directly (≤2**D subsets) rather than sampled. The
  attributions satisfy Shapley efficiency to floating-point tolerance
  (`phi.sum(1) + expected_value_ == prediction`), are reported in the user's
  original feature space (categorical combos / multi-target encodings fold into
  one player), and **include the linear-leaf slope terms exactly** — so they
  faithfully explain the actual model rather than just its split structure (which
  is all gain importance sees). Regression explains the target; binary
  classification explains the pre-temperature log-odds. Averaged across the bag
  when `n_ensembles > 1`. Multiclass is not supported yet.
- **Linear-leaf models** (`linear_leaves`, default-on for binary classification).
  Each leaf fits a ridge model over its numeric split features instead of a
  constant, adding local slope where step leaves underfit; `linear_lambda` sets
  the ridge penalty. Leaves with too few rows fall back to a constant. Not
  available with MAE/Quantile loss or multiclass.
- **Hierarchical shrinkage** (`hs_lambda`). Above 0, leaf values are recursively
  shrunk toward their ancestors — hardest for deep or low-mass leaves — at no
  inference cost.
- **`cat_features` as a constructor argument**, so `GridSearchCV`/`Pipeline` can
  carry it; a value passed to `fit` still overrides it.
- **Input and hyperparameter validation.** Malformed constructor params (e.g.
  non-positive `iterations`/`depth`, `depth` capped at 16 to avoid OOM, `lr > 0`,
  non-negative regularizers, `subsample`/`colsample` in `(0, 1]`,
  `cat_smoothing > 0`, known `loss`/`alpha`), `sample_weight` values (finite,
  non-negative, positive sum), `cat_features` indices, and `eval_set` shape now
  raise clear errors instead of crashing cryptically or silently misbehaving.
- **Predict-time feature-name enforcement.** Reordered or renamed DataFrame
  columns at `predict` now raise instead of silently producing wrong predictions.

### Changed
- **Regressor `depth` default is loss-adaptive.** `None` resolves to 6 for
  RMSE/MAE (behavior unchanged — predictions are bit-identical) and to 4 for
  `loss="Quantile"`, where deep leaves overfit the extreme-quantile tails.

### Fixed
- **Quantile under-dispersion.** Held-out coverage of extreme quantiles collapsed
  toward the median as depth grew; the loss-adaptive shallower default restores
  both coverage and the pinball objective.
- **`cat_smoothing=0` is now rejected** with a clear error (previously a cryptic
  `ZeroDivisionError` from a 0/0 in the ordered target encoder).
- **pyarrow-backed DataFrames** no longer pollute captured feature names; masked
  arrays are rejected at `fit`; `inf` is rejected at `predict` (mirroring `fit`),
  with the O(n) scan skippable via scikit-learn's `assume_finite` for serving.

## [0.10.0] - 2026-06-02
### Changed
- **Out-of-the-box defaults now early-stop.** Both estimators default to
  `early_stopping=True`, `iterations=2000` (was 500), and `validation_fraction=0.2`
  (was 0.1). A plain `model.fit(X, y)` now carves an internal stratified holdout,
  early-stops on it (patience 50), and uses the best iteration — instead of
  building a fixed 500 trees with no stopping (which could overfit). This makes
  the **out-of-box defaults match the benchmarked/Pareto configuration exactly**.
  Pass `early_stopping=False` for the old fixed-iteration behavior; an explicit
  `eval_set` still overrides the internal split.
- **Benchmarks measure default behavior.** The ChimeraBoost benchmark runner now
  calls the bare default estimator (no external `eval_set`), so it performs its
  own internal early-stopping split exactly like a user's `.fit(X, y)`. The
  published Pareto/summary/slowdown images are regenerated from this run.

### Fixed
- Early stopping degrades gracefully on tiny data: when the training set is too
  small to carve a valid (stratified) validation split, `early_stopping` is
  silently disabled for that fit instead of raising — so `early_stopping=True`
  is safe as the new default even on very small or few-member-class datasets.

## [0.9.2] - 2026-06-02
### Performance
- Vectorized categorical encoding (`factorize`, `_codes_for_transform`) via pandas,
  replacing per-element Python loops. ~3.4× faster on the encoding step and
  ~15% faster end-to-end fit on categorical-heavy datasets (e.g. adult), with
  **bit-identical** output. Numeric-only datasets are unaffected. Adds `pandas`
  as a dependency.

### Changed
- **Default `l2_leaf_reg` lowered 3.0 → 1.0.** Lifts Grinsztajn binary Brier
  95.7% → 97.2% of best (+1.5pp), pulling the classification leg even with
  LightGBM, with RMSE and F1 flat (all 24 regression deltas <0.2% noise).
- **Classifier `min_child_weight` is now size-adaptive by default** (`None` → auto:
  full veto ~1 below ~500 training rows, fading to 0 above ~2000). The old flat
  `mcw=1` silently capped oblivious classification tree depth (~4.9 of 6),
  under-fitting larger data; the new default lifts binary Brier broadly (18W/0L on
  the Grinsztajn suite, +1.6pp, reaching the speed/accuracy Pareto frontier) while
  the size ramp protects small datasets (validated on an independent OpenML set).
  Root-caused by matching a stripped-down CatBoost: the gap was our min-leaf veto,
  not the oblivious tree structure. Regression is unaffected (a no-op in [0,1]
  post empty-child-exemption); explicit `min_child_weight` values are still honored.

### Added
- **Input validation** across both estimators: clear, actionable errors instead
  of cryptic numpy/numba tracebacks for predict-before-fit (`NotFittedError`),
  feature-count mismatch at predict time, and 1-D / empty / mismatched-length /
  complex / sparse / non-finite inputs and `y=None`.
- `n_features_in_` and (for DataFrame input) `feature_names_in_` attributes.
- A column-vector `y` of shape `(n, 1)` is now raveled with a
  `DataConversionWarning`; a continuous target passed to the classifier raises.
- **scikit-learn `check_estimator` compliance** for both estimators, with a
  single documented deviation: `sample_weight` reweights the loss but is not
  bit-exactly equivalent to integer row repetition. Other intentional deviations:
  NaN-in-X accepted as missing, dense-only input, and the `cat_features` /
  `eval_set` fit kwargs.

### Docs
- README "Tuning tips": interaction-heavy regression (e.g. `pol`) benefits from
  `depth=8–10` — at `depth=10` ChimeraBoost is best-in-field on `pol` (+12% vs
  CatBoost/LightGBM/sklearn). The `depth=6` default stays conservative for
  small-data safety.

## [0.9.1] - 2026-06-01
### Changed
- Tidied the README and benchmark tables; moved the "near-solved excluded from
  RMSE" note into a proper footnote and added the blended-strength Pareto image.
- Corrected the CatBoost speed claim to ~5x (geomean on the 59-dataset
  Grinsztajn 2022 benchmark); the old ~30x was from the categorical-heavy
  OpenML suite.

## [0.9.0] - 2026-06-01
### Fixed
- **Oblivious depth cap:** empty (pure) children are now exempt from the
  `min_child_weight` veto, so `depth` is a real lever again. Regression RMSE
  rose from 95.7% to 98.0% of best on the Grinsztajn suite (now beats sklearn),
  with a broad 26W/6L per-dataset sign test, and fits got faster.
### Changed
- Classifier defaults: `ordered_boosting=False`, `leaf_estimation_iterations=3`.
- Regressor default: `ordered_boosting=False`.
- Benchmarks: blended-strength Pareto, near-solved RMSE guard, `/bench` command.

## [0.8.0]
### Added
- First-class bagging (`n_ensembles`) and the Brier benchmark metric.
