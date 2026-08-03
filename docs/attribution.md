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
| Ordered target statistics for categoricals, multi-permutation averaging, ordered boosting, feature combinations, `leaf_estimation_iterations` | CatBoost, Prokhorenkova et al., NeurIPS 2018 |
| Smoothing a category's target estimate toward the prior | Micci-Barreca, SIGKDD Explorations 2001 |
| Histogram-based split finding on pre-binned features | LightGBM, Ke et al., NeurIPS 2017; earlier in McRank (Li et al. 2007) and pGBRT (Tyree et al. 2011) |
| Second-order split gain `G²/(H+λ)`, Newton leaf values, `min_child_weight` | XGBoost, Chen & Guestrin, KDD 2016 |
| Column subsampling (`colsample`) | Breiman, *Random Forests*, 2001; random subspaces, Ho, IEEE TPAMI 1998 — the sources the XGBoost paper itself names for it |
| Minimum Variance Sampling — the gradient-weighted row draw behind `subsample` | Ibragimov & Gusev, NeurIPS 2019 (CatBoost's MVS) |
| Quantized gradient histograms (`quantize_gradients`) | Shi, Ke et al., *Quantized Training of Gradient Boosting Decision Trees*, NeurIPS 2022 (LightGBM) |
| Vector leaves for multiclass, and the random-projection gradient sketch that scores their splits | SketchBoost, Iosipoi & Vakhrushev, NeurIPS 2022; GBDT-MO, Zhang & Jung, IEEE TNNLS 2021 |
| Linear (ridge) leaves | Shi, Li & Li, IJCAI 2019; model trees back to Quinlan's M5, 1992 |
| Exact TreeSHAP attributions, interventional formulation | Lundberg & Lee, NeurIPS 2017; Lundberg et al., Nature Machine Intelligence 2020 |
| Automatic pairwise feature generation (`cross_features`) | OpenFE, Zhang et al., ICML 2023 |
| Pinball loss | Koenker & Bassett, Econometrica 1978 |
| One fitted model serving every quantile level | Quantile regression forests, Meinshausen, JMLR 2006 |
| Labelling each row by which quantile-grid bucket its target falls in, and scoring one shared split across all levels on those labels | Generalized random forests, Athey, Tibshirani & Wager, Annals of Statistics 2019 (the `grf` quantile forest) |
| Shifted-Legendre contrasts in tau as location, spread and skew | L-moments, Hosking, JRSS-B 1990 |
| Cycling boosting updates through location and spread, and the greedy variant that instead picks the strongest component each round | gamboostLSS, Mayr, Fenske, Hofner, Kneib & Schmid, JRSS-C 2012; the non-cyclical variant of Thomas et al., Statistics and Computing 2018 |
| Making a tree's split criterion see spread and not only location | Distributional regression forests, Schlosser, Hothorn, Stauffer & Zeileis, AoAS 2019 |
| Non-crossing quantiles by monotone rearrangement | Chernozhukov, Fernández-Val & Galichon, Econometrica 2010 |
| Split-conformal calibration | Papadopoulos et al. 2002; Vovk, Gammerman & Shafer 2005; Lei et al., JASA 2018 |
| Conformalized quantile regression (`conformalize`) | Romano, Patterson & Candès, NeurIPS 2019 |
| Temperature scaling of classifier probabilities | Guo, Pleiss, Sun & Weinberger, ICML 2017; Platt 1999 |
| Bagging, subagging, out-of-bag estimation | Breiman 1996; Bühlmann & Yu, Annals of Statistics 2002 |
| Refitting the selected model on all the data; named quality presets | AutoGluon, Erickson et al. 2020; the CV-then-refit convention in scikit-learn |
| Refreshing leaf values on a fixed tree structure — the mechanism `refit_full="replay"` uses | XGBoost's `refresh` updater; LightGBM's `Booster.refit()` |
| Racing candidate configurations under a shared budget and killing the loser early | Hoeffding races, Maron & Moore, NeurIPS 1993; Karnin et al., ICML 2013; Jamieson & Talwalkar, AISTATS 2016 |
| CRPS as mean pinball loss over a quantile grid | Gneiting & Raftery, JASA 2007 |
| Huber loss for outlier-robust regression | Huber, Annals of Mathematical Statistics 1964; as a boosting loss, Friedman 2001 |
| Log-link Poisson, Gamma and Tweedie losses | generalized linear models, Nelder & Wedderburn, JRSS-A 1972; the compound Poisson-gamma family, Tweedie 1984 and Jorgensen 1987 |
| Missing values routed to their own bin, so no imputation is needed | XGBoost's sparsity-aware split finding, Chen & Guestrin, KDD 2016; LightGBM's dedicated missing bin |
| Greedy bin construction that isolates heavy values | LightGBM's `GreedyFindBin` |
| Split-gain feature importance (`feature_importances_`) | Breiman, Friedman, Olshen & Stone, *CART* 1984; relative influence for boosted ensembles, Friedman 2001 |
| Early stopping on a held-out split with a patience window | Prechelt 1998, and the automatic-internal-holdout convention of scikit-learn, LightGBM and XGBoost |
| Compensated summation in the group-mean kernel | Kahan, CACM 1965 |
| The estimator API — `fit`/`predict`, `get_params`, validation and error conventions | scikit-learn, Buitinck et al., ECML-PKDD 2013 |
| SplitMix64, the counter-based generator behind stochastic rounding | Steele, Lea & Flood, OOPSLA 2014 |
| Synthetic data drawn from structural-causal-model priors (the `--synth` screen) | the prior-sampling approach of TabPFN, Hollmann et al., ICLR 2023, and its descendants |

## Adapted

A borrowed idea, changed enough here that the difference is worth stating. The credit
still belongs upstream.

| What | Borrowed from | What is different here |
|---|---|---|
| `adaptive_learning_rate` | CatBoost's automatic learning rate — the size-dependent default that showed up when we diffed its resolved parameters | The fade curve is ours; that the rate should depend on `n` at all is theirs |
| `min_child_weight` on oblivious trees | XGBoost's `min_child_weight` | One sparse child vetoes the whole level, because the split is shared. Empty children are exempt so pure leaves do not cap depth |
| `refit_members` | The fixed-structure leaf refresh above | Applied per bag member, to recover the data each member gives up to its own out-of-bag split |
| `cat_combinations` default | CatBoost's feature combinations | The rule that turns them on automatically for all-categorical data is ours |
| Interval calibration | Conformalized quantile regression | Applied as a per-level multiplicative scale around the predicted median rather than the usual additive widening, so that calibrating cannot reorder the quantile grid. Scaled conformity scores are a known variant of CQR, not something we invented |
| The multi-quantile head | CatBoost's `MultiQuantile` loss and Meinshausen's quantile regression forests — one model, all levels at once | Vector leaves hold an exact empirical quantile per level, and the split search scores gains on a projection rather than on class indicators. See below for what is and is not ours in it |
| Scoring one shared split across all quantile levels | `grf`'s quantile forest, which relabels each row by the grid bucket its target falls in | We score that shared split with a projected second-order gain instead of a multiclass split rule, and never materialise the per-level gradient matrix on the default settings |
| Rotating the split direction through location and spread contrasts | gamboostLSS's cyclical fitting, on the shifted-Legendre (L-moment) basis | Ours is the measured ratio — two location rounds per spread round — and restricting the greedy `"gram"` variant to that subspace against a no-signal whitener |
| Reclaiming the early-stopping split | Refitting on all the data once the round count is known: AutoGluon's `refit_full`, and the same pattern packaged elsewhere for LightGBM and XGBoost | We do it as a fixed-structure leaf refresh rather than a from-scratch regrow, and it is on by default rather than an extra step the user asks for |
| `ordered_boosting` | CatBoost's ordered boosting | CatBoost estimates gradients from a cascade of permutation-ordered supporting models. Ours is a leave-one-out leaf step: a row's update uses its leaf's totals with its own contribution removed. Same goal, far cheaper, weaker guarantee. Off by default — and CatBoost itself resolves to plain boosting at every size we measured |
| Bag member draws | Subagging and the cluster bootstrap | Group-disjoint draws, so grouped data keeps a usable out-of-bag set |

## Ours

Written here, by Nathan Walker and Claude. This list got considerably shorter once we
went looking for prior art properly, which is the honest outcome: nearly everything we
thought of as ours turned out to have a name and a paper. What survives is engineering
and small default-setting, not new learning theory.

**Not building the per-level gradient matrix.** On the default settings the quantile
split search collapses each row's whole K-channel pinball gradient to one number in a
single pass over that row's own scores, so the `(n, K)` gradient is never materialised
and a K-level head costs roughly what a one-level head costs per round. It is built on
the `"gram"` and `exact_splits` arms, which need it. The representation this scores —
one shared split across every level — is `grf`'s, and the projection machinery is
SketchBoost's; what we have not found elsewhere is scoring that shared split with a
projected second-order gain that never forms the matrix.

**Small default policies.** The classifier fades `min_child_weight` out as the training
set grows, because on an oblivious tree a single sparse child vetoes the whole level.
Quantile losses default to depth 4 rather than 6, on measurement, for the standard reason
that tail estimates need more rows per leaf. The multi-quantile head derives its leaf-size
floor from the most extreme level on the grid, wiring in the usual rule that a tau
quantile needs on the order of `1/tau` rows to be estimable, so the floor tracks whatever
grid you ask for. A bag member whose draw collapses to a single class gets one donor row,
which is a crash guard rather than the stratified draw the imbalanced-bagging literature
would reach for. Setting a default by a formula in `n` is itself an old idea — `mtry` is
the textbook case — so what is ours here is which curve, not the shape of the answer.

**Equivalence-preserving engineering.** The fused per-level kernel, the bit-identical
splice that adds cross features to an already-fitted preprocessor, and the audited fast
path for factorizing numeric categories. Checking an optimized path against a reference
implementation is standard practice in numerical libraries, not a house invention; what
is ours is only these particular equivalences and the tests that pin them.

**A cold-start notice.** Detecting that numba is about to spend fifteen seconds
compiling, and saying so, instead of appearing to hang. Small, but we have not seen
another library do it.

## Benchmarks

The evaluation leans on other people's work too:

- **Grinsztajn, Oyallon & Varoquaux**, NeurIPS 2022 — the tabular benchmark that decides
  whether a change ships here.
- **PMLB**, Olson et al. 2017 and Romano et al. 2021 — the Penn Machine Learning
  Benchmarks, used only for hyperparameter tuning.
- **TabArena** — the community leaderboard, run by its maintainers on their own defaults.
  It is read and reported, never tuned against.
- **OpenML** and **Hugging Face** host most of the datasets.
- Scoring uses the **Brier** score (Brier 1950) in **Murphy**'s (1973) skill-score form,
  alongside R² and CRPS.

Two things the harness does that are worth stating plainly, with no claim that they are
new: it diffs a competitor's resolved parameters across dataset sizes to find which of
its defaults depend on `n`, and it forces a competitor to adopt our setting to measure
what its own choice was worth.

## Built on

numba (Lam, Pitrou & Seibert, LLVM-HPC 2015), NumPy, SciPy, and scikit-learn. There is
no ChimeraBoost without them.
