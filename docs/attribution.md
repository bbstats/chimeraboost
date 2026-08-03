# Where the ideas come from

Almost everything ChimeraBoost does was invented by someone else. The gradient boosting
algorithm, the oblivious tree, the way categoricals are encoded, the histogram split
search, the leaf formula, the calibration step, the conformal intervals — all published
work by other people, listed below with the source.

This page exists so nobody has to guess which parts are borrowed. If a credit here is
wrong, missing, or too generous to us, please open an issue; being corrected on this is
the point of writing it down.

## Borrowed

| What | Borrowed from |
|---|---|
| Gradient boosting, shrinkage, the terminal-node value override used for MAE and quantile losses | Friedman, *Greedy Function Approximation*, Annals of Statistics 2001; *Stochastic Gradient Boosting*, 2002 |
| Oblivious (symmetric) trees — one shared split per level | CatBoost, Prokhorenkova et al., NeurIPS 2018. The tree type itself goes back to Kohavi & Li, IJCAI 1995 |
| Ordered target statistics for categoricals, multi-permutation averaging, ordered boosting, feature combinations, `leaf_estimation_iterations` | CatBoost, Prokhorenkova et al., NeurIPS 2018 |
| Smoothing a category's target estimate toward the prior | Micci-Barreca, SIGKDD Explorations 2001 |
| Histogram-based split finding on pre-binned features | LightGBM, Ke et al., NeurIPS 2017; earlier in McRank (Li et al. 2007) and pGBRT (Tyree et al. 2011) |
| Second-order split gain `G²/(H+λ)`, Newton leaf values, `min_child_weight` | XGBoost, Chen & Guestrin, KDD 2016 |
| Minimum Variance Sampling — the gradient-weighted row draw behind `subsample` | Ibragimov & Gusev, NeurIPS 2019 (CatBoost's MVS) |
| Quantized gradient histograms (`quantize_gradients`) | Shi, Ke et al., *Quantized Training of Gradient Boosting Decision Trees*, NeurIPS 2022 (LightGBM) |
| Vector leaves for multiclass, and the random-projection gradient sketch that scores their splits | SketchBoost, Iosipoi & Vakhrushev, NeurIPS 2022; GBDT-MO, Zhang & Jung, IEEE TNNLS 2021 |
| Linear (ridge) leaves | Shi, Li & Li, IJCAI 2019; model trees back to Quinlan's M5, 1992 |
| Exact TreeSHAP attributions, interventional formulation | Lundberg & Lee, NeurIPS 2017; Lundberg et al., Nature Machine Intelligence 2020 |
| Automatic pairwise feature generation (`cross_features`) | OpenFE, Zhang et al., ICML 2023 |
| Quantile regression, and one fitted model serving every level at once | Koenker & Bassett, Econometrica 1978; quantile regression forests, Meinshausen, JMLR 2006; CatBoost's `MultiQuantile` |
| The pieces our multi-quantile split search is built from: one shared split scored across every level, location/spread/skew contrasts in tau, and cycling through them or greedily picking the strongest each round | `grf`, Athey, Tibshirani & Wager, Annals of Statistics 2019; L-moments, Hosking, JRSS-B 1990; gamboostLSS, Mayr et al., JRSS-C 2012, and the non-cyclical variant of Thomas et al. 2018 |
| Non-crossing quantiles by monotone rearrangement | Chernozhukov, Fernández-Val & Galichon, Econometrica 2010 |
| Split-conformal calibration, and its quantile form (`conformalize`) | Vovk, Gammerman & Shafer 2005; Lei et al., JASA 2018; conformalized quantile regression, Romano, Patterson & Candès, NeurIPS 2019 |
| Temperature scaling of classifier probabilities | Guo, Pleiss, Sun & Weinberger, ICML 2017; Platt 1999 |
| Bagging, out-of-bag estimation, and column subsampling (`colsample`) | Breiman 1996 and 2001; subagging, Bühlmann & Yu, Annals of Statistics 2002; random subspaces, Ho, IEEE TPAMI 1998 — the sources the XGBoost paper itself names for `colsample` |
| Refitting the selected model on all the data, and refreshing leaf values on a fixed tree structure — the two halves of `refit_full` | AutoGluon, Erickson et al. 2020; XGBoost's `refresh` updater; LightGBM's `Booster.refit()` |
| Racing candidate configurations under a shared budget and killing the loser early | Hoeffding races, Maron & Moore, NeurIPS 1993; Karnin et al., ICML 2013; Jamieson & Talwalkar, AISTATS 2016 |
| Missing values routed to their own bin, so no imputation is needed | XGBoost's sparsity-aware split finding, Chen & Guestrin, KDD 2016; LightGBM's dedicated missing bin |
| Greedy bin construction that isolates heavy values | LightGBM's `GreedyFindBin` |
| The estimator API — `fit`/`predict`, `get_params`, validation and error conventions | scikit-learn, Buitinck et al., ECML-PKDD 2013 |
| Synthetic data drawn from structural-causal-model priors (the `--synth` screen) | the prior-sampling approach of TabPFN, Hollmann et al., ICLR 2023, and its descendants |

Not listed: the textbook. Huber loss, the GLM log links behind Poisson/Gamma/Tweedie,
CRPS, split-gain importance, early stopping with a patience window, compensated
summation. These belong to everybody, and citing a 1964 paper because the library
offers `loss="Huber"` would make the credits above look like padding rather than
the real debts they are.

## Adapted

A borrowed idea, changed enough here that the difference is worth stating. The credit
still belongs upstream.

| What | Borrowed from | Our version |
|---|---|---|
| `adaptive_learning_rate` | CatBoost's automatic learning rate — the size-dependent default that showed up when we diffed its resolved parameters | The fade curve is ours; that the rate should depend on `n` at all is theirs |
| `min_child_weight` on oblivious trees | XGBoost's `min_child_weight` | One sparse child vetoes the whole level, because the split is shared. Empty children are exempt so pure leaves do not cap depth |
| `cat_combinations` default | CatBoost's feature combinations | The rule that turns them on automatically for all-categorical data is ours |
| Interval calibration | Conformalized quantile regression | Applied as a per-level multiplicative scale around the predicted median rather than the usual additive widening, so that calibrating cannot reorder the quantile grid. Scaled conformity scores are a known variant of CQR, not something we invented |
| The multi-quantile head | CatBoost's `MultiQuantile`, Meinshausen's quantile regression forests, and `grf`'s shared-split scoring | Vector leaves hold an exact empirical quantile per level, and the split search scores a projected second-order gain instead of a multiclass split rule. The contrast schedule — two location rounds per spread round — is measured here, on a basis that is Hosking's |
| `ordered_boosting` | CatBoost's ordered boosting | CatBoost estimates gradients from a cascade of permutation-ordered supporting models. Ours is a leave-one-out leaf step: a row's update uses its leaf's totals with its own contribution removed. Same goal, far cheaper, weaker guarantee. Off by default — and CatBoost itself resolves to plain boosting at every size we measured |
| Bag member draws | Subagging and the cluster bootstrap | Group-disjoint draws, so grouped data keeps a usable out-of-bag set |

## Ours

Written here, by Nathan Walker and Claude. It is mostly engineering rather than new
learning theory, and two of these carry measured results.

### 1. Structure-transfer refit — `refit_full="replay"`, on by default

**The problem.** Early stopping holds rows back to choose the tree count, and those rows
never reach the model you ship. The standard fix is to retrain on all the data once the
count is known. It works — and it costs a second full fit. On our own profile that refit
was 37 to 49% of every default fit, the single largest thing the library did.

**The observation.** Growing trees is 83 to 85% of a fit. So a from-scratch refit spends
most of its time rediscovering split structures the first model already found. It does
not have to.

**What it does instead.** Replay the winner's splits round by round against gradients
computed on all the rows, and refit only the leaf values. The held-out rows still reach
the leaf estimates — which is where the accuracy came from — without paying for the split
search a second time.

**What it bought.** On Grinsztajn, fit time down **34.8%** and faster on **58 of 59**
datasets; on the high-cardinality suite, down 15.2% and faster on 12 of 14 — less,
because replay does not cover multiclass and categorical preprocessing is a fixed cost
either way. Accuracy is a wash on four independent suites, every median essentially zero:
Grinsztajn +0.005%, high-card −0.017%, PMLB −0.043%, public −0.114%. Two of those four
played no part in the decision. `refit_members` applies the same trick per bag member, to
reclaim what each member gives up to its own out-of-bag split.

**What is borrowed.** Both halves. Refreshing leaf values on a fixed structure is
XGBoost's `refresh` updater and LightGBM's `refit()`. Retraining on all the data after
early stopping is AutoGluon's `refit_full`, and is packaged elsewhere for LightGBM and
XGBoost. Using the first as the implementation of the second, by default, is the part we
put together.

**One thing worth recording.** The first cut of this leaked the target. Split thresholds
are bin *indices*, so the donor's preprocessor has to be reused verbatim — a re-fitted
binner silently moves every threshold underneath the structures it is replaying. Our own
leakage regression test caught it, and the buggy run's apparent wins turned out to be the
leak. The headline number above is from the fixed version.

### 2. Gradient-matrix free quantile split search

Each row's whole K-channel pinball gradient collapses to one number in a single pass over
that row's own scores, so on the default settings the `(n, K)` gradient is never
materialised and a K-level head costs roughly what a one-level head costs per round. The
`"gram"` and `exact_splits` arms do build it, because they need it.

It costs accuracy, and the cost is small: the default projection lands within **7%** of
the exact summed-across-levels gain while building one histogram per round instead of one
per level. Scoring the levels by simply adding them up — the obvious thing — is much
worse, and for a reason worth knowing: on a symmetric grid the pushes from tau and
1 − tau are equal and opposite, so an interval centred correctly but too narrow sums to
exactly zero. Adding them up is blind to spread.

The representation being scored — one shared split across every level — is `grf`'s, and
the projection machinery is SketchBoost's. What we have not found elsewhere is scoring
that shared split with a projected second-order gain that never forms the matrix.

### 3. Equivalence-preserving engineering

The fused per-level kernel, the bit-identical splice that adds cross features to an
already-fitted preprocessor, and the audited fast path for factorizing numeric
categories. Checking an optimized path against a reference implementation is standard
practice in numerical libraries, not a house invention; ours are these particular
equivalences and the tests that pin them.

## Benchmarks

The evaluation leans on other people's work too:

- **Grinsztajn, Oyallon & Varoquaux**, NeurIPS 2022 — the tabular benchmark that decides
  whether a change ships here.
- **PMLB**, Olson et al. 2017 and Romano et al. 2021 — the Penn Machine Learning
  Benchmarks, used only for hyperparameter tuning.
- **TabArena** — the community leaderboard, run by its maintainers on their own defaults.
  It is read and reported, never tuned against.
- **OpenML** and **Hugging Face** host most of the datasets.

Two things the harness does that are worth stating plainly, with no claim that they are
new: it diffs a competitor's resolved parameters across dataset sizes to find which of
its defaults depend on `n`, and it forces a competitor to adopt our setting to measure
what its own choice was worth.

## Built on

numba (Lam, Pitrou & Seibert, LLVM-HPC 2015), NumPy, SciPy, and scikit-learn. There is
no ChimeraBoost without them.
