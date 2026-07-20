# Changelog

All notable changes to ChimeraBoost are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Fixed
- **Early stopping now truncates to the best iteration when `n_estimators`
  runs out before patience fires** (previously every tree built past the
  best round was kept, contradicting the documented `best_iteration_`
  contract; on budget-exhausted fits this cost ~5% validation RMSE in an
  adversarial repro). Fits where patience fires, or with early stopping
  off, or stopped by a callback, are unchanged.
- **Thread hygiene**: `thread_count` now applies to predict as well as fit,
  and the process-global numba thread setting is restored afterwards in both
  — previously one `fit(thread_count=1)` silently capped every later numba
  call in the process, and predict ignored the setting entirely. A
  `thread_count` equal to the ambient numba count is applied for free; a
  differing one is switched and restored per call (usually cheap, up to
  ~1 ms observed in some process states as the omp layer re-teams), so
  prefer setting the ambient count (`NUMBA_NUM_THREADS` or
  `numba.set_num_threads`) in latency-sensitive serving.
- **Pickles no longer carry the packed-forest predict cache** (roughly 2x
  payload when a model was pickled after predicting); it rebuilds lazily on
  the first predict after load. Loaded models predict bit-identically.
- **Loud failures instead of silent misbehavior**:
  - a classifier `eval_set` containing labels absent from `y` now raises
    (previously binary counted them as the negative class and multiclass
    silently remapped them onto a neighboring class, or crashed with a bare
    `IndexError` past the top);
  - an automatic early-stopping split that strands a class entirely in the
    validation side (rare class, or a class confined to one group with
    `groups`) now raises instead of silently training a model that can
    never predict that class;
  - `fit(X, y, w)` — weights accidentally bound to the third positional
    argument `cat_features` — now raises an error that names
    `sample_weight=w` instead of a cryptic numpy complaint;
  - explicitly-set `ordered_boosting` / `leaf_estimation_iterations` warn
    when the chosen path ignores them (linear-leaves active, multiclass,
    or MAE/Quantile losses); defaults stay quiet and the precedence is
    documented in Parameters.
- `cat_features` accepts a numpy integer array (previously crashed with
  "ambiguous truth value"); classifier `depth=None` resolves to the
  documented default 6 (previously crashed with `TypeError` in the booster).
- Bagged members fitted from a DataFrame now retain the parent's feature
  names, so bagged predicts no longer emit a spurious "fitted without
  feature names" warning on every call and the column-order guard applies
  at the member level too. The bagged-mode member-defaults notice is now a
  filterable `UserWarning` instead of bare stdout.

## [0.18.0] - 2026-07-18
### Changed
- **Quantized-gradient histograms are now the default**
  (`quantize_gradients=True`): the split search runs on ~15-bit quantized
  gradient/hessian pairs packed one-integer-per-sample into integer
  histograms (the LightGBM-4 quantized-training idea, adapted). Summed
  single-model fit time fell 26% on the Grinsztajn suite and 20% on the
  high-cardinality suite (bagged Ens8 arm likewise -19%), at benchmark-flat
  accuracy: even per-dataset sign tests on both decision suites and the
  OpenML gate, with exact ties on many datasets — leaf values are always
  computed from the unquantized float gradients, so rounding noise only
  ever touches split selection. Deterministic for a fixed `random_state`
  (counter-based stochastic rounding). `quantize_gradients=False` restores
  the exact float64 histograms (and bit-identical pre-flip models).
- **Tree growth runs one fused kernel launch per level** — split search,
  leaf descend, and the next level's occupied-leaf list in a single numba
  call, replacing a second kernel launch plus a `bincount`/`flatnonzero`
  pair per level. Models are bit-identical (numerical-identity goldens,
  exact-equality kernel oracles, and full decision-suite ties: 73/73
  datasets, single and Ens8 arms). Small fits gain the most — the removed
  per-level fixed cost was 8-15% of fit time on 10-40K-row data; suite-
  summed fit time moves ~1% (large-data fits are scatter-bound).

## [0.17.0] - 2026-07-17
### Added
- **Cross features now cover multiclass classification.** The same
  validation-selected difference/product columns that regression and binary
  classification get (top numeric pairs of the base fit, raced audition at
  `selection_rounds`, judged on softmax validation log loss). Auto-on under
  the same gates (≥ 2000 rows, ≥ 2 numeric features); explicit
  `cross_features=True` no longer raises on multiclass. Costs up to ~2x fit
  time on eligible multiclass data when the refit runs; regression and
  binary paths are bit-identical to 0.16.1 (verified: every untouched
  dataset in the synthetic screen, both decision suites, and the OpenML
  gate tied exactly, per-arm).

## [0.16.1] - 2026-07-17
### Changed
- **Faster fits at identical output: internal model selection no longer
  recomputes preprocessing.** The variant auditions, the cross-feature
  candidate, and the winner refit inside one `fit` previously each reran the
  full preprocessing pipeline (target encoding, quantile borders, binning) on
  the same data; they now share it, and the cross-feature candidate only
  computes its added columns. Models are bit-identical to before (verified
  by the numerical-identity test goldens and exact prediction matches on
  benchmark data). Biggest on categorical-heavy data, where repeated target
  encoding was the waste (17–32% faster single-model fits measured there);
  bagged ensembles save it in every member.

## [0.16.0] - 2026-07-17
### Changed
- **Bagged members now train on 80% row subsamples drawn without replacement**
  (new `max_samples` parameter, default `0.8`; `1.0` restores the classic
  full-size bootstrap). A bootstrap gives each member only ~63% unique rows
  at 100% of the compute; 80% without replacement is more effective data and
  less work. Decision suites vs the previous bootstrap: Grinsztajn 54W-5L
  +0.94% with a 23-0 Brier sweep at 0.87x fit time; high-card Brier 8-0 at
  0.73x; OpenML gate 22W-5L +0.68%.
- **Tuned bagged-member defaults** (opt-in `n_ensembles` mode): inside a bag,
  parameters left on auto now resolve to member-tuned values —
  `learning_rate` `None` → 0.15 and `colsample` `None` → 0.85 (single-model
  resolution unchanged: 0.1 / 1.0). PMLB-tuned, holdout-confirmed, and
  validated on both decision suites (54W-17L, +0.28% pooled vs the previous
  bagged defaults at par fit cost; OpenML gate 18W-9L). The fit prints a
  one-line notice when member defaults activate, `member_params_` records
  them, and explicit values always win. Recommended ensemble size is now
  `n_ensembles=8` (measured stronger than 5 at similar cost; 2 remains
  anti-recommended). `colsample`'s default changed from `1.0` to `None`
  (identical behavior for single models).
- **Bagged ensembles fit their members in parallel by default**
  (`ensemble_n_jobs` default `1` → `-1`): members fit across
  `min(n_ensembles, thread budget)` worker processes, each on an equal share
  of the budget, so a bagged fit uses the same cores a single fit would.
  Models are identical to the sequential fit (verified exactly on 73
  benchmark datasets); wall-clock is 1.2–2x faster on a free box. Pass
  `ensemble_n_jobs=1` for the old sequential behavior.
### Fixed
- `ensemble_n_jobs=-1` previously gave every member worker the full thread
  budget (oversubscribing cores by the worker count); the budget is now
  divided across workers.

## [0.15.0] - 2026-07-16
### Added
- **`selection_rounds` (default `100`): raced internal selections, ~1.5x
  faster fits at the same accuracy.** The constant/linear-leaf variants and
  the pre-cross base fit now run as capped auditions instead of to full early
  stopping; candidates are judged on their best validation loss within the
  shared budget, only the winner continues, and an audition that early-stops
  before the cap is reused as the finished fit (that path is bit-identical to
  the old behavior). Grinsztajn suite fit time 351→235 s (accuracy columns
  flat: RMSE 99.4/F1 99.8/Brier 99.1 vs best, calibration slightly better);
  headline slowdown 7.9x→6.0x at unchanged blended strength; high-card suite
  1.11x faster, columns flat. Trade-off (measured, accepted): on a minority of
  regression datasets the 100-round audition can pick the leaf variant a full
  run would have rejected, typically costing 0.5–1.5% there (cpu_act −1.4% is
  the worst observed on real data). `selection_rounds=None` restores the old
  run-everything-to-full-early-stopping behavior.
- **SynthGen decision suite (benchmarks only, no library change):**
  `benchmarks/synthgen/` generates unlimited SCM-prior synthetic datasets
  (TabPFN/TabICLv2/Mitra recipe family, numpy-only, deterministic per key)
  with observable marginals bootstrapped from 1,644 harvested public OpenML
  dataset profiles (TabArena's 51 members excluded at the source — the sealed
  holdout stays untouched in every form) and exact Bayes floors in the
  per-dataset meta. Frozen suites `syn:v1` (smoke 6 / screen 182 / full 242
  datasets) run via `run_benchmarks.py --synth`; `synth_report.py` attributes
  A/B deltas to generative factors (interaction depth, cats, noise,
  saturation…); `synthgen/backtest.py` scores the suite against ledger
  verdicts before it gates anything. ~10% of ids are saturated
  kr-vs-kp-style canaries where the baseline sits at the ceiling, so
  complexity-adding flags that "win" there are exposed as variance injection.
  `compare_runs.py` gains `--model`; the summary caption now names the suite
  it aggregated instead of always saying Grinsztajn. **Validated 2026-07-14:
  8/9 ledger arms agree** (removing cross_features: −3.3% exactly on the
  pre-registered interaction slice; forced cat_combinations: mixed on ordinary
  data, +27% on the car-analog cat-interaction sets). Known v1 biases queued
  for v2: targets slightly shallow (depth-4 arm disagrees), no entity-effect
  categoricals, no cat-bearing verified-at-ceiling canaries in the screen.
  **v2 (2026-07-15, re-frozen as `syn:v2`, smoke 6 / screen 136 / full 211):**
  deeper interaction prior and wider in-degree; ~40% of categorical columns
  are latent *entities* (Zipf-frequency levels with per-level target effects
  and singleton rare levels — the mechanism behind ordered target statistics;
  the CatBoost high-card realism check flipped to PASS, winrate 0.71 vs 0.60);
  screen n-mix stratified (n<2000 share ≤ 35%); canary status *earned* by a
  freeze-time at-ceiling fit check across the harness's 3 seed-splits
  (`suites.CANARIES`; the loose single-seed criterion admitted sets whose
  residual headroom forced cat_combinations then captured — tightening cut
  the list 18 → 8). Gate re-passed 7/9 with the canary slice exactly flat;
  depth4 flipped from its v1 wrong-sign win to a mean-negative not-win.
- **`cross_features` (default `None` = on where applicable):
  validation-selected numeric interaction columns.** For RMSE regression and
  binary classification with ≥ 2000 rows and ≥ 2 numeric features, the
  estimator refits with difference and product columns for the pairs of the
  base fit's top-6 numeric features and keeps whichever model reaches the
  lower validation loss (`cross_features_selected_` / `cross_pairs_` record
  the outcome; predict and SHAP take the original columns unchanged;
  multiclass, MAE/Quantile, and small data are skipped — bit-identical to
  the previous release there; `cross_features=False` restores the old
  behavior everywhere). Oblivious trees can only approximate a numeric
  interaction such as `x_i < x_j` with a depth-limited staircase — one
  shared split per level — so boundaries between features cost many levels
  and many trees; a cross column makes them a single split.
  Full-suite A/B (Grinsztajn 59, 3 seeds): **51W/8L, mean +1.5%**, with the
  historical regression trouble spots largely resolved (sulfur +13%,
  Brazilian_houses +7–8%, pol +6.4%, nyc-taxi +0.7/+3.3%, covertype F1
  +3.1%); independent OpenML one-shot 8W/4L on the datasets where the refit
  engages (guards keep everything else bit-identical). With this default the
  blended-strength Pareto reads 99.4 vs CatBoost's 98.1 with every accuracy
  column ranked first. Costs ~2.2× fit time in aggregate where it engages
  (absolute: 160 s → 348 s across the whole 59-dataset suite).
### Changed
- **Small-data fit is 1.2–1.35× faster** (2k×30 rows: regressor 196→146 ms =
  1.34×, classifier 85→71 ms = 1.20×; single tree build 347→220 µs = 1.58×),
  bit-identical — predictions verified exactly equal across 17 fit
  configurations (reg/clf/multiclass, categoricals, subsample, colsample,
  min_child_weight, depth 8), golden suite untouched. One fused kernel
  (`_build_and_split`) replaces the per-level histogram-build + best-split
  pair: one parallel launch instead of two, empty leaf rows skipped (zeroing
  and scanning), only each feature's actual `n_bins_` zeroed/scanned, and the
  split scan transposed (leaf-outer/bin-inner) so it streams each histogram
  row sequentially with the per-leaf parent term computed once. Sample
  descent runs serially below 32k rows (the parallel fork/join costs more
  than the pass). Large-n is unaffected (20k and 200k A/B at parity). The
  original kernels remain as the exact-equality oracle
  (`tests/test_tree_kernels.py`). This targets the TabArena-scale regime
  where per-level fixed cost, not sample count, dominates fit time.

## [0.14.2] - 2026-07-13
### Changed
- **Predict is 1.35–1.6× faster end-to-end** (2M×30 batch, 200 trees;
  default binary 1.35×, constant-leaf regressor/classifier 1.63×), from two
  bit-identical changes — predictions are unchanged to the last bit,
  verified by exact-equality kernel tests and the golden suite:
  - The fused forest kernels now consume the binner's row-major output
    directly (`_predict_forest_rm`/`_predict_forest_linear_rm`): each
    sample's bins sit in one or two cache lines for the whole forest walk,
    and the per-predict feature-major transpose copy is gone. Fit-side
    kernels keep the feature-major layout (histograms want it).
  - `FeaturePreprocessor` no longer gathers the numeric block with a
    whole-matrix fancy-index copy when every column is numeric (the
    no-categoricals case) — that copy was ~18% of end-to-end predict on
    large batches.
  Measured against the field (fit 200k / predict 2M / 200 trees, 12
  threads): default binary predict 1.26 Mrows/s — 1.30× LightGBM, 3.0×
  sklearn-HGB; constant-leaf paths ~2.6 Mrows/s, on par with XGBoost.
  CatBoost's SIMD-fused C++ inference remains ~10× faster.

## [0.14.1] - 2026-07-09
### Changed
- **Regressor `linear_leaves` default `False` → `None` (validation-selected).**
  Fixed linear leaves were a regression wash with casualties on breadth
  benchmarks (16W/12L): real wins (pol −6.4%, abalone −3.1%) but real losses
  (visualizing_soil −4.7%). The new default fits both variants and keeps the
  one with the lower validation loss on the already-held-out early-stopping
  split — the same post-fit-decision pattern as temperature scaling and the
  conformal quantile offset. Gates: Grinsztajn 36-set breadth 20W/9T/7L
  (−0.58% mean RMSE) vs constant and 12W/19T/5L (−0.32%) vs always-linear,
  dodging every fixed-linear casualty; independent OpenML+PMLB one-shot
  8W/7T/1L (−0.81%). Costs ~2× fit time when selection runs (RMSE loss, a
  validation split, ≥1000 rows); pass `linear_leaves=True/False` to force a
  variant and skip the double fit. `linear_leaves_selected_` records the
  choice.

### Added
- **`chimeraboost.warmup()`** — pre-compiles (or loads from the on-disk cache)
  every numba kernel on the default fit and predict paths via three tiny
  synthetic fits. A fresh process pays the JIT inside its first `fit`
  (~5–15 s cold) and first `predict` (~0.2–2 s) — irrelevant for long-lived
  processes, dominant for fleets of short-lived workers (benchmark harnesses,
  serverless inference, ray/spark tasks) fitting small data. Calling
  `warmup()` at startup, outside anything timed or billed, restores
  steady-state speed: on a 2K-row task, first-fit wall time inside the timed
  section drops 9.3 s → 0.10 s and first-predict 1.8 → 0.001 s per 1K rows.
  This is the fix for the inflated ChimeraBoost train/predict times on the
  TabArena leaderboard, whose cluster re-times every fold in a fresh worker
  process (our identical run measured 0.6 s/1K train, 0.068 s/1K predict —
  faster at predict than every other tree model on the board).
  Setting `CHIMERABOOST_WARMUP=1` runs it automatically at import — no code
  changes needed in worker fleets. `warmup(background=True)` (or
  `CHIMERABOOST_WARMUP=background`) instead compiles in a daemon thread so
  the JIT overlaps the caller's own startup work, for deployments with real
  setup between import and first fit; a fit issued mid-compile just waits on
  numba's per-kernel locks, never slower than compiling inline.
### Added
- **Conformal quantile calibration.** `loss="Quantile"` predictions now include
  a split-conformal offset (`quantile_offset_`) fitted on the early-stopping
  validation split — the regression analog of the classifier's temperature
  scaling. Boosting under-disperses quantiles (each round's per-leaf quantile
  step is shrunk by the learning rate, so the tails converge slowly and early
  stopping cuts them short); the conformal order statistic of the validation
  residuals is both the coverage-restoring shift (distribution-free, Romano et
  al. 2019) and the pinball-optimal constant correction, so calibration and
  accuracy improve together. Measured at α=0.1/0.9 across four datasets:
  tail coverage 0.12–0.23 → 0.08–0.11 and 0.80–0.90 → 0.88–0.91 (nominal
  0.1/0.9), test pinball loss improved or flat everywhere. RMSE/MAE fits and
  quantile fits without a validation split are bit-identical to before
  (offset 0.0). SHAP additivity and `staged_predict` fold the offset in.

### Fixed
- **`feature_importances_` no longer counts trees discarded by early
  stopping.** Gains were accumulated as trees were built, but the truncation
  at the best iteration never subtracted the dead trees (up to `patience` of
  them). Importances are now computed from the retained trees only.
  Predictions are unaffected.
- **Core booster default aligned with the sklearn wrappers.** `_BaseBooster`
  defaulted `ordered_boosting=True` while `ChimeraBoostRegressor`/`Classifier`
  default `False`; anyone driving `GradientBoosting`/`MulticlassBoosting`
  directly silently got a different algorithm. The core now defaults `False`
  too. (The sklearn wrappers always passed it explicitly — no change there.)

### Changed
- **Column subsampling now skips masked features when building histograms**
  (`_best_split` already honored the mask; the histogram kernel scanned every
  feature anyway). Bit-identical trees; fits with `colsample<1` get the
  proportional histogram work back — measured 1.44× end-to-end on a
  histogram-dominated regression fit at `colsample=0.4` (less where other
  kernels dominate, e.g. binary with linear leaves).
- **MAE/Quantile leaf correction groups samples with one stable argsort**
  instead of an n_leaves-pass boolean scan. Exactly the same values reach the
  quantile estimator in the same order — predictions bit-identical.
- **Linear-leaf fitting is now parallel — binary classification fits 1.4–1.8×
  faster** (5k rows 1.4×, 50k 1.8×, 200k 1.6×; regression with
  `linear_leaves=True` benefits equally). The two remaining serial kernels
  (`_linear_leaf_fit`, `_linear_predict`) were ~half of binary fit time; they
  are now `parallel=True`. Bit-identical predictions: a stable counting sort
  groups samples by leaf so every leaf's normal equations accumulate in the
  exact float-add order the serial code used, and per-sample prediction is
  embarrassingly parallel. Thread-count invariance preserved. Trade-off:
  first-fit JIT in a fresh environment grows ~2s (parallel compilation is
  costlier); the on-disk kernel cache still makes this once per environment.

## [0.13.1] - 2026-07-06
### Changed
- **Faster cold start.** The single `np.linalg.solve` call in the linear-leaf
  fit kernel is replaced with a hand-rolled LU solver (partial pivoting) that
  runs inside numba without pulling in the LAPACK bindings. Those bindings were
  the dominant cost of the first `fit()` in a fresh environment; eliminating
  them cuts first-fit JIT time by ~25% on dev hardware. Fixed-seed predictions
  may differ from 0.13.0 at the ~1e-15 level (solver elimination order); tree
  structures are unchanged.

### Fixed
- **pandas nullable dtypes no longer crash.** Columns of dtype `Int64`/`Float64`/
  `boolean` (and the `string` dtype) carry missing values as `pd.NA`, which used
  to fail the float cast with a cryptic `float() argument must be ... not
  'NAType'`. `pd.NA` is now mapped to `np.nan` and routed to the missing bin, at
  both fit and predict.
- **`inf` is now rejected when `cat_features` is set.** The infinity check
  previously skipped the whole matrix for categorical fits, silently routing an
  `inf` in a numeric column to the missing bin. It now checks the numeric columns
  at fit and predict, matching the no-`cat_features` behavior.

## [0.13.0] - 2026-06-15
### Changed
- **Faster inference (~1.9×) and fit (~1.4×).** Predict-time bin assignment and
  the per-level leaf descent during tree building are now parallel numba kernels
  instead of allocation-heavy NumPy. Output is bit-identical; large-batch
  `predict`/`predict_proba` throughput roughly doubles (now on par with
  LightGBM) and fitting on large data is ~1.4× faster.

### Removed
- **Eight default-off experimental flags retired** after the research cascade
  found each either null or net-negative: `hs_lambda`, `adaptive_leaf_shrinkage`,
  `adaptive_leaf_estimation`, `ordered_leaf_estimation`, `forest_leaf_refit`
  (+`forest_refit_iterations`), `onehot_low_card` (+`onehot_max_card`),
  `cat_combinations_selective` (+`cat_combinations_max_pairs`), and
  `cat_aware_binning` (+`cat_max_bins`). The constructor drops from 36 to 24
  parameters. All shipped defaults (`cat_combinations` auto-rule, `linear_leaves`,
  `leaf_estimation_iterations`, ordered boosting) are unchanged — predictions for
  any model not setting a removed flag are identical.

## [0.12.0] - 2026-06-09
### Changed
- **`cat_combinations` default is now adaptive** (`None`). Pairwise
  category-by-category features are enabled automatically when the data is
  entirely categorical — where they capture interactions without crowding out
  numeric splits — and stay off otherwise. This closes the long-standing gap on
  all-categorical datasets (e.g. the `car` multiclass set) out of the box. Set
  `True`/`False` to force it; auto is skipped for very wide all-categorical data
  as a resource guard against the `C(n_cat, 2)` blow-up.

### Added
- **`validation_history_`** property on both estimators — the full per-round
  validation-loss curve from a single fit (length = rounds run; with
  `early_stopping=False` it runs to the horizon, never truncated). Makes
  per-iteration capture first-class.
- **`callbacks=`** fit hook — `cb(iteration, train_loss, val_loss, model)` called
  each round; returning `True` requests an early stop. (Not supported with bagging.)
- **Opt-in research flags** (all default-off, byte-identical no-ops unless set).
  Each was validated through an efficient paired-curve benchmark cascade; none
  improved the blended defaults broadly (the defaults are already at a good
  optimum — see `benchmarks/research/SUMMARY.md`), so they ship as documented
  opt-ins for data that matches their narrow sweet-spot: `onehot_low_card`
  (one-hot low-cardinality categoricals), `cat_aware_binning` (larger bin budget
  for target-encoded categoricals — both help all-categorical sets like
  `car`/`splice`), `cat_combinations_selective` (mutual-info-selected combos on
  mixed data), `forest_leaf_refit` (post-fit joint ridge over all leaves),
  `ordered_leaf_estimation` (ordered boosting + leaf refinement together),
  `adaptive_leaf_estimation` (size-scheduled Newton steps), and
  `adaptive_leaf_shrinkage` (mass-dependent per-leaf shrinkage).
- **Research cascade harness** under `benchmarks/research/` — a reusable,
  download-once, paired-validation-curve engine for evaluating ideas efficiently
  without ever touching the sealed TabArena holdout.

## [0.11.0] - 2026-06-04
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
- **`cat_features` by column name.** Categoricals can now be marked by DataFrame
  column name as well as integer position, or a mix — e.g.
  `cat_features=["city", "brand"]`. Names are resolved against the DataFrame at fit.
- **Input and hyperparameter validation.** Malformed constructor params (e.g.
  non-positive `n_estimators`/`depth`, `depth` capped at 16 to avoid OOM, `lr > 0`,
  non-negative regularizers, `subsample`/`colsample` in `(0, 1]`,
  `cat_smoothing > 0`, known `loss`/`alpha`), `sample_weight` values (finite,
  non-negative, positive sum), `cat_features` indices, and `eval_set` shape now
  raise clear errors instead of crashing cryptically or silently misbehaving.
- **Predict-time feature-name enforcement.** Reordered or renamed DataFrame
  columns at `predict` now raise instead of silently producing wrong predictions.

### Changed
- **Renamed `iterations` → `n_estimators`** (BREAKING), matching the
  LightGBM/XGBoost convention for the number of boosting rounds (trees). Update
  any code that passed `iterations=...`.
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
