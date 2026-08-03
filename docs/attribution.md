# Where the ideas come from

Almost everything ChimeraBoost does was invented by someone else. The gradient boosting
algorithm, the oblivious tree, the way categoricals are encoded, the histogram split
search, the leaf formula, the calibration step, the conformal intervals — all published
work by other people, listed below with the source.

This page exists so nobody has to guess which parts are borrowed. If a credit here is
wrong, missing, or too generous to us, please open an issue; being corrected on this is
the point of writing it down.

## Borrowed

Taken from published work or from another library. The implementation is ours, written
from the paper or the described behaviour; the idea is not.

| What | Where it came from |
|---|---|
| Gradient boosting, shrinkage, the terminal-node value override used for MAE and quantile losses | Friedman, *Greedy Function Approximation*, Annals of Statistics 2001; *Stochastic Gradient Boosting*, 2002 |
| Oblivious (symmetric) trees — one shared split per level | CatBoost, Prokhorenkova et al., NeurIPS 2018. The tree type itself goes back to Kohavi & Li, IJCAI 1995 |
| Ordered target statistics for categoricals, multi-permutation averaging, feature combinations, `leaf_estimation_iterations` | CatBoost, Prokhorenkova et al., NeurIPS 2018 |
| Smoothing a category's target estimate toward the prior | Micci-Barreca, SIGKDD Explorations 2001 |
| Histogram-based split finding on pre-binned features | LightGBM, Ke et al., NeurIPS 2017; earlier in McRank (Li et al. 2007) and pGBRT (Tyree et al. 2011) |
| Second-order split gain `G²/(H+λ)`, Newton leaf values, `min_child_weight` | XGBoost, Chen & Guestrin, KDD 2016 |
| Minimum Variance Sampling — the gradient-weighted row draw behind `subsample` | Ibragimov & Gusev, NeurIPS 2019 (CatBoost's MVS) |
| Quantized gradient histograms (`quantize_gradients`) | Shi, Ke et al., *Quantized Training of Gradient Boosting Decision Trees*, NeurIPS 2022 (LightGBM) |
| Vector leaves for multiclass, and the random-projection gradient sketch that scores their splits | SketchBoost, Iosipoi & Vakhrushev, NeurIPS 2022; GBDT-MO, Zhang & Jung, IEEE TNNLS 2021 |
| Linear (ridge) leaves | Shi, Li & Li, IJCAI 2019; model trees back to Quinlan's M5, 1992 |
| Exact TreeSHAP attributions, interventional formulation | Lundberg & Lee, NeurIPS 2017; Lundberg et al., Nature Machine Intelligence 2020 |
| Automatic pairwise feature generation (`cross_features`) | OpenFE, Zhang et al., ICML 2023 |
| Pinball loss | Koenker & Bassett, Econometrica 1978 |
| Non-crossing quantiles by monotone rearrangement | Chernozhukov, Fernández-Val & Galichon, Econometrica 2010 |
| Split-conformal calibration | Papadopoulos et al. 2002; Vovk, Gammerman & Shafer 2005; Lei et al., JASA 2018 |
| Conformalized quantile regression (`conformalize`) | Romano, Patterson & Candès, NeurIPS 2019 |
| Temperature scaling of classifier probabilities | Guo, Pleiss, Sun & Weinberger, ICML 2017; Platt 1999 |
| Bagging, subagging, out-of-bag estimation | Breiman 1996; Bühlmann & Yu, Annals of Statistics 2002 |
| Refitting the selected model on all the data; named quality presets | AutoGluon, Erickson et al. 2020; the CV-then-refit convention in scikit-learn |
| Refreshing leaf values on a fixed tree structure — the mechanism `refit_full="replay"` uses | XGBoost's `refresh` updater; LightGBM's `Booster.refit()` |
| Racing candidate configurations under a shared budget and killing the loser early | Hoeffding races, Maron & Moore 1994; Karnin et al., ICML 2013; Jamieson & Talwalkar, AISTATS 2016 |
| CRPS as mean pinball loss over a quantile grid | Gneiting & Raftery, JASA 2007 |
| The estimator API — `fit`/`predict`, `get_params`, validation and error conventions | scikit-learn, Buitinck et al., ECML-PKDD 2013 |
| SplitMix64, the counter-based generator behind stochastic rounding | Steele, Lea & Flood, OOPSLA 2014 |
| Synthetic data drawn from structural-causal-model priors (the `--synth` screen) | the prior-sampling approach of TabPFN, Hollmann et al., ICLR 2023, and its descendants |

## Adapted

A borrowed idea, changed enough here that the difference is worth stating. The credit
still belongs upstream.

| What | Borrowed from | What is different here |
|---|---|---|
| `adaptive_learning_rate` | CatBoost's automatic learning rate, its one size-dependent default | We found it by diffing CatBoost's resolved parameters across dataset sizes, then measured how much of its small-data advantage the schedule accounts for. The fade curve is ours; the idea that the rate should depend on `n` is entirely theirs |
| `min_child_weight` on oblivious trees | XGBoost's `min_child_weight` | One sparse child vetoes the whole level, because the split is shared. Empty children are exempt so pure leaves do not cap depth |
| `refit_members` | The fixed-structure leaf refresh above | Applied per bag member, to recover the data each member gives up to its own out-of-bag split |
| `cat_combinations` default | CatBoost's feature combinations | The rule that turns them on automatically for all-categorical data is ours |
| Interval calibration | Conformalized quantile regression | Applied as a per-level multiplicative scale around the predicted median rather than the usual additive widening, so that calibrating cannot reorder the quantile grid. Scaled conformity scores are a known variant of CQR, not something we invented |
| The multi-quantile head | CatBoost's `MultiQuantile` loss — one model, vector leaves, all levels at once | The split search is ours; see below |
| Bag member draws | Subagging and the cluster bootstrap | Group-disjoint draws, so grouped data keeps a usable out-of-bag set |

## Ours

Written here, by Nathan Walker and Claude. It is a short list, and most of it is
engineering or small default-setting rather than new learning theory.

**The multi-quantile split search.** A row's entire K-channel pinball gradient is
determined by one integer — where its label falls in the predicted quantile grid — so
the split search never builds the `(n, K)` gradient matrix at all. Gains are scored by
projecting onto one direction, which for the pinball loss is provably the rank-1
truncation of the exact gain rather than a heuristic, and the direction cycles through
low-order polynomial contrasts so that changes in spread are visible and not just
changes in location. SketchBoost's random projections are the nearest relative we know
of; the rank-lookup form and the contrast schedule we have not found elsewhere.

**Structure-transfer refit.** Reusing a donor model's tree structures and refitting only
the leaves is the borrowed part. Using it to give a fitted model the data its own
early-stopping split consumed, by default, is the composition we added.

**Small default policies.** Fading `min_child_weight` out as the training set grows;
defaulting quantile models to shallower trees because tail estimates need more rows per
leaf; deriving a leaf-size floor from the most extreme quantile level; injecting a donor
row when a bag member's draw misses a class entirely. Each is a few lines, and each
exists because a benchmark caught the failure it prevents.

**Equivalence-preserving engineering.** The fused per-level kernel, the bit-identical
splice that adds cross features to an already-fitted preprocessor, the audited fast path
for factorizing numeric categories, and the numerical-identity tests that hold all of it
in place. Kernel fusion is not new; what is ours is the discipline that every fast path
must produce bit-identical output to the slow one it replaces, enforced by tests.

**A cold-start notice.** Detecting that numba is about to spend fifteen seconds
compiling, and saying so, instead of appearing to hang.

**Two benchmarking methods.** Diffing an opponent's resolved parameters across dataset
sizes to find which of its defaults depend on `n`; and forcing an opponent to adopt our
setting to measure what its own choice was worth. Ablation is ordinary science — running
it on the opponent rather than on ourselves is the part we have not seen done elsewhere.

## Benchmarks

The evaluation leans on other people's work too:

- **Grinsztajn, Oyallon & Varoquaux**, NeurIPS 2022 — the tabular benchmark that decides
  whether a change ships here.
- **PMLB**, Olson et al. 2017 and Romano et al. 2021 — the Penn Machine Learning
  Benchmarks, used only for hyperparameter tuning.
- **TabArena** — the community leaderboard, run by its maintainers on their own defaults.
  It is read and reported, never tuned against.
- **OpenML** and **Hugging Face** host most of the datasets.

## Built on

numba (Lam, Pitrou & Seibert, LLVM-HPC 2015), NumPy, SciPy, and scikit-learn. There is
no ChimeraBoost without them.
