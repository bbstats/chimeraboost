# Changelog

All notable changes to ChimeraBoost are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Changed
- **Multiclass fits are ~40% faster; predictions are bit-identical.** The
  softmax that turns raw scores into probabilities every boosting round was
  the largest single cost in a multiclass fit — 45% of it on a three-class,
  38K-row dataset — and almost none of that was the arithmetic. It ran two
  numpy reductions across the class axis, so for three classes numpy paid its
  per-row reduction machinery 38,000 times to compare three numbers; the
  exponential itself was only a fifth of the cost. It is now one fused numba
  kernel per row, the same treatment the binary logistic path already had.
  33× faster on the softmax itself, which is 37–44% off end-to-end fit time
  across the multiclass datasets measured, and nothing at all on binary or
  regression data, where this code never ran. Above seven classes the original
  numpy path is kept: numpy sums a short axis left to right but switches to
  pairwise blocking from eight elements, and past that point a fused kernel
  would no longer agree with it in the last bit. Bit-identity is a gate here,
  not a hope — the exact-output snapshot passes 89 of 89 configurations.
- **String categorical columns factorize ~40% faster; predictions are
  bit-identical.** Columns whose entries are not all numbers — every string
  categorical — fell through to a per-row Python dict loop, which on a
  17-string-column dataset ran 1.6 million `dict.get` calls and took 18% of
  the profiled fit. The loop's dictionary is now driven from C rather than
  from Python bytecode: `map(dict.setdefault, values, count())` assigns each
  new category the row index of its first appearance, and one scatter turns
  those indices into codes, with no sort. Keeping the same dictionary is
  deliberate — it is what defines which values count as the same category,
  including across types (`True`, `1` and `1.0` are one category; `"1.5"` and
  `1.5` are two), so equivalence is structural rather than re-audited. 0.57×
  the old loop over 35 real string columns; 4–7% off end-to-end fit time on
  string-heavy datasets, nothing on datasets without string categoricals.
  Anything the vectorized path cannot express — a value that raises on
  comparison, an unhashable one — still takes the original loop.

### Added
- **`cross_features="always"` (regressor): cross features without the
  audition.** The default path auditions plain features against
  cross-augmented ones, and the augmented candidate wins 20 of 21 of those
  races — so at a one-fit operating point the race is mostly a fee for a
  known answer. The forced mode skips it: a short importance probe (~25
  rounds, a few percent of fit time) ranks features, then one full fit keeps
  a narrower top-4 cross block unconditionally. Same applicability gates as
  the raced mode (2000-row floor, RMSE-only, enough numeric/categorical
  columns), inert wherever they fail; the classifier rejects the value. The
  raced default path is byte-identical to before.

### Changed
- **`quality=1` (regressor) now pins `cross_features="always"`.** The fast
  rung previously dropped cross features entirely because keeping them
  meant paying for the race. Forced-on passed the full decision gate: 2.7×
  within-run fit time (between rungs 1 and 2, at 0.57× of rung 2), 7.9
  points above the LightGBM-to-default frontier chord, beats the old rung 1
  wherever it engages (28W–8L on Grinsztajn, engaged median +0.5%) and
  still beats LightGBM head-to-head (43W–16L). The classifier's rung 1 is
  unchanged. Evidence: benchmarks/SELECT_PLAN.md E2.

## [0.30.0] - 2026-08-02
### Changed
- **The automatic `learning_rate` now depends on how much data it has, and this
  is the default (`adaptive_learning_rate=True`).** Our rate was a flat 0.1 at
  every dataset size. CatBoost's is not, and asking it directly — fit at eight
  row counts and diff the resolved parameters — showed that of its 43 settled
  parameters, **exactly one varies with dataset size, and it is the learning
  rate**: a clean power law running from 0.026 at 200 rows to 0.064 at 60,000.
  Denying CatBoost that schedule costs it 57% of its edge over us at a quarter
  of the rows, which makes it the measured mechanism behind a win rate that had
  been collapsing as data shrank.

  Sweeping our own rate found the knee at 0.07, so `None` now resolves to a
  linear fade: 0.07 at 5,000 training rows or fewer, rising to the historical
  0.1 at 15,000 or more. **Above 15,000 rows this is a no-op and the model is
  byte-identical to 0.29.0** — only small-data fits move. It applies only when
  early stopping is on, since without it the rate already scales with the round
  budget, and an explicit `learning_rate` still overrides everything.

  On the decision suites the mean is positive in six of seven strata, with
  sign-test passes at a quarter of the rows on both Grinsztajn (9W-3L) and
  high-card (3W-0L), and no losses at all on high-card at full size (6W-0L).
  Against CatBoost on the two strata that named the problem, our win rate moves
  from 50% to 67% and from 33% to 67% — the first movement on that number. The
  gains are individually small (medians of +0.13% to +0.31%) and cost 1.09x to
  1.31x fit time on the sizes the fade touches.

  The seventh stratum, temporal splits, initially read as a loss and held the
  default back. Re-run over six rolling origins instead of the three the suite
  happens to use, it is 21 wins and 21 losses — a coin flip.

  Pass `adaptive_learning_rate=False` for the pre-0.30.0 flat 0.1 everywhere.
  Three paths are unaffected regardless: bagged fits (`n_ensembles >= 2`), whose
  members already carry an explicit member learning rate and so never consult
  the auto rule; fits with `early_stopping=False`, where the rate already scales
  with the round budget; and `ChimeraBoostQuantileRegressor`, deliberately — the
  evidence here is squared error and Brier score, and the fade has never been
  measured against pinball loss.

  Also lands as a side effect: **CatBoost does not run ordered boosting** — it
  resolves `boosting_type=Plain` at every dataset size, which retires a
  hypothesis this project had carried for months.

## [0.29.0] - 2026-08-01
### Added
- **`refit_members` (opt-in, bagged path): each member now reclaims the rows it
  never learned from.** A bag member trains on `max_samples` of the rows and
  early-stops on its out-of-bag complement, so the full-data refit that helps a
  single model never fired for it — every member's leaf values were estimated
  from 80% of the data. With `refit_members=True` each member replays its own
  tree structure against gradients from every row once early stopping is done.

  Only the leaf values move: the splits stay exactly as each member's own
  sample grew them, which is where a bag's diversity actually lives. Measured
  over 10 datasets and 3 seeds, an 8-member bag improves on 10 of 10 datasets
  (mean +1.02% on the primary metric), and the gain is largest for small bags
  (+1.86% at 2 members) — which is what "each member is individually stronger"
  predicts, since fewer members mean less averaging to hide weak leaf values.
  A 5-member refit bag matches or beats a plain 8-member bag on 7 of 10.
  Costs about 10% more fit time.

  On the decision suites an 8-member bag improves in **every stratum**, with
  perfect sweeps on the small-data ones — Grinsztajn 52W-7L (+0.500%),
  Grinsztajn at a quarter of the rows **12W-0L** (+1.206%), at half the rows
  **6W-0L**, high-card at a quarter of the rows 2W-0L (+1.452%). Brier moves
  with it (Grinsztajn 20W-3L, +1.503%). Fit cost rises about 17%.

  A **5-member bag with `refit_members` beats a plain 8-member bag on both
  axes** — stronger in five of seven strata and 20% faster — so the cheapest
  way to buy accuracy from bagging is now fewer, better members.

  Regression and binary classification only; multiclass members would need a
  full refit rather than a cheap structure replay, so the flag is ignored
  there. Default `False`, and byte-identical to the previous release when off:
  single-model behaviour, the default configuration and the headline chart are
  all untouched.

### Docs
- **Corrected the bagging guidance, which `refit_members` made wrong.** The FAQ
  called `n_ensembles=8` "the strongest setting available" and "the
  maximum-accuracy mode"; both are false now that adding `refit_members`
  improves on it in every stratum. The recommended default is unchanged.
- `refit_members` is now documented on both estimator classes, so it appears in
  the published API reference alongside the other bagging parameters, and the
  quoted fit cost is stated as 10-17% (the 10% figure came from the smaller
  10-dataset probe; the decision suites measured 17%).

## [0.28.0] - 2026-07-31
### Fixed
- **`ChimeraBoostQuantileRegressor` returned prediction intervals 2x to 10x
  wider than they should have been, and they got worse the longer you
  trained.** The non-crossing guarantee was enforced during the fit by a global
  "narrowing budget" charged at the worst-case leaf each round, so a single
  aggressively narrowing leaf spent budget on behalf of every row. It ran out
  within tens of rounds and froze the interval width from then on: on `cpu_act`
  the width was identical at rounds 1000, 2000 and 3000, while coverage climbed
  from 0.86 to 0.99 against a nominal 0.80 as the centre kept sharpening under
  a band that could no longer follow.

  Ordering is now imposed per row on the delivered predictions by monotone
  rearrangement. That is exact for each row rather than a bound that has to
  hold for every possible row at once, and it costs no accuracy — sorting a
  crossing quantile curve never increases pinball loss at any level
  (Chernozhukov, Fernández-Val & Galichon 2010). Predictions still cannot
  cross, at every stage of `staged_predict`, and `crossing_rate` is still
  exactly zero.

  Measured on three Grinsztajn regression sets, 3 seeds: pinball loss improves
  by 80%, 24% and 24%, and the band now tracks local spread instead of freezing.
  Against 19 independent LightGBM quantile boosters the head moves from roughly
  matching them to beating them at every width tested (pinball ratio 0.968 to
  0.989 from 5 to 128 features), at 3.0x to 6.2x their fit speed.

  One caveat worth knowing: the raw grid now errs slightly *narrow* rather than
  very wide — a nominal 80% interval delivers about 72 to 76% coverage, because
  leaf values are in-sample residual quantiles. Use `conformalize=True` when you
  need the interval to carry a coverage guarantee; it lands within 2.7
  percentage points of nominal. See `docs/quantiles.md`.

### Removed
- The `gap_` attribute on `MultiQuantileBoosting`, which reported the narrowing
  budget's remaining margin. The budget no longer exists, and an attribute
  whose documented guarantee is no longer true is worse than no attribute.
  Nothing on `ChimeraBoostQuantileRegressor` exposed it.

### Notes
- Models pickled with 0.27.0 or earlier load and predict identically: their
  stored leaf values already satisfied the old constraint, so rearranging them
  changes nothing.
- The scalar `MAE` and `Quantile` losses share these leaf kernels at `K = 1` and
  are bit-identical across this change, confirmed by the identity snapshot
  (81 of 89 arrays unchanged; the 8 that moved are the two multi-quantile
  configurations in full).

## [0.27.0] - 2026-07-31
### Changed
- **A pass of seven output-identical speedups: MAE and quantile fits about
  twice as fast, categorical prediction about 25% faster, categorical fits
  about 11%.** Each one removed work that was provably redundant rather than
  changing any arithmetic:
  - `MAE` and `Quantile` re-sorted every leaf's residuals and called
    `np.quantile` from Python once per leaf per boosting round. Both now
    dispatch to the multi-quantile head's compiled kernels with `K = 1` —
    kernels that were already pinned bit-identical to exactly those NumPy
    calls, and were sitting unused on the scalar path.
  - Categorical factorization ran a per-row Python loop on every fit and
    every predict batch, which was the dominant cost of predicting with a
    categorical model. All-numeric columns now take a vectorized path whose
    equivalence is *audited* per column rather than assumed: numeric strings,
    the string `"nan"`, integers past 2^53 and `Decimal` drift all fail the
    audit and fall back to the original loop, which is kept verbatim as both
    the fallback and the test oracle.
  - `RMSE`, `MAE`, `Quantile`, `Huber` and `MultiQuantile` allocated a fresh
    vector of ones for their constant hessian every round; they now share one
    cached buffer, which pickles drop.
  - `feature_importances_` became a single `bincount` instead of a nested
    Python loop over trees, levels and splits — it is re-read once per
    audition on the selection path and once per bagged member.
  - Early-stopping evaluation walked the validation set single-threaded every
    round; above 32 768 rows it now descends in parallel.
  - Leaf-value updates gather into a fused kernel instead of building a
    full-length temporary each round.

  Bit-identical — every model, every prediction exactly unchanged (89/89
  identity-snapshot arrays, including the weighted, subsampled, categorical
  and multi-quantile configurations).
- **`subsample < 1` is 22-31% cheaper to fit.** MVS row sampling picks a
  threshold from the gradient magnitudes once per boosting round, and it was
  single-threaded NumPy allocating about ten full-length temporaries each
  round while the tree build beside it ran on every core — at 200k rows it
  cost 61% of the fit. The threshold search and the importance-weight pass are
  now numba kernels; the sort and its sum deliberately stay in NumPy, because
  those are the only two pairwise-summed steps and porting them would move the
  threshold by an ulp and change which rows get sampled. Bit-identical — every
  model, every prediction exactly unchanged (89/89 identity-snapshot arrays,
  which now include a subsampled multiclass config). Measured on 12 threads:
  regression at 200k x 20 1.38-1.44x, at 2M x 20 1.34x, binary 1.32x,
  multiclass 1.08x. Defaults are untouched: `subsample` defaults to 1.0, which
  never enters this path.
- **Numeric binning fits its borders in parallel across features.** Every
  column's borders were already computed independently; wide-enough fits
  (200k+ cells) now farm the columns out to a thread pool running the same
  per-column code, sized by the numba thread count so `thread_count` and
  bagged-worker budgets are honored. Bit-identical — every border, bin
  center and downstream model is exactly unchanged (84/84 identity-snapshot
  arrays). Measured at 200k rows x 30 features on 12 threads: dense borders
  4.3x faster, zero-inflated 3.2x, sample-weighted 5.5x; a 50k x 200 matrix
  2.7x. Small fits keep the serial loop.

## [0.26.0] - 2026-07-30
### Added
- **`ChimeraBoostQuantileRegressor`: a whole predictive distribution from one
  booster, with predictions that cannot cross.** One tree structure per round
  serves every level in `quantiles` (default 0.05 ... 0.95), each leaf holding
  a K-vector whose entries are the exact per-level empirical quantiles of the
  leaf's residuals. `predict` returns the grid, or a central interval, or the
  mean by integrating the quantile function.

  The ordering guarantee is structural rather than a repair applied at predict
  time: the model starts from the sorted global quantiles and every leaf
  vector is projected onto increments that cannot reorder anything, so
  `diff(Q, axis=1) >= 0` holds exactly, at every intermediate `staged_predict`
  stage too. Measured crossing rate 0.0000 against 0.18-0.21 for 19
  independently fitted LightGBM quantile boosters. Intervals can still be
  narrower than the pooled one where the data is quiet — a narrowing budget
  buys that back, which a plain monotone-increment construction cannot express
  at all.

  The split search runs once per round instead of once per level, so the
  saving grows with data width: **3.4x the fit speed of 19 independent
  boosters at 5 features, 4.8x at 32, 7.8x at 128**, with pinball loss within
  3% averaged over the grid (a single level can trade more against a dedicated
  fit — see the quantiles guide) and better on wide data. `conformalize=True` calibrates the
  intervals on a fold held out before the early-stopping split (worst coverage
  error 0.7 points at n = 10 000). New `chimeraboost.quantile_metrics` scores
  a predicted grid: per-level pinball, CRPS, and coverage with width.
  Full record in `benchmarks/QUANTILE_PLAN.md`; defaults elsewhere are
  untouched.

### Fixed
- **`max_bins` below 16 no longer crashes on a dominated column.** The greedy
  border pass floored the light region's bin budget at `max_bins // 16`, which
  is zero under 16; a column whose mass sits overwhelmingly on one value then
  divided by zero. `max_bins` of 3, 4 and 6 all failed outright on a 90%-zeros
  column. Budgets of 16 and up keep their exact allocation.
- **A bagged classifier survives a one-row member draw.** The rare-class guard
  added below overwrote a random drawn row with a donor of the missing class;
  when the draw held exactly one row that replaced the only row, so the member
  still saw a single class and raised "Need at least 2 classes" anyway. Tiny-`n`
  bags with a small `max_samples` now grow the draw to two rows instead.
- **A column dominated by its minimum value no longer loses its bins.** When
  one value holds more than an even bin's share of the mass (sparse count
  features: mostly zeros), the quantile borders collapsed onto that value and
  the whole column silently binned as constant. Colliding quantile levels now
  trigger a greedy border pass that isolates heavy values in bins of their own
  and spreads the remaining budget over the rest by mass. Columns without
  collisions keep bit-identical borders.
- **`eval_set` targets are validated like training targets.** A NaN/inf in the
  validation `y` (or a value outside the loss's domain, e.g. a zero with
  `loss="Gamma"`, or a custom `eval_metric` returning NaN) made every
  validation score NaN, and early stopping silently kept a one-tree model.
  Both are errors now, raised with the cause named.
- **A reordered or renamed `eval_set` DataFrame raises at fit**, matching the
  predict-time guard. It was consumed positionally and silently corrupted
  early stopping, temperature scaling, and the conformal offset. The
  `shap_values` background matrix gets the same check.
- **Zero-weight rows can no longer steer the post-fit calibrations.** The
  classifier's temperature and the quantile regressor's conformal offset now
  honor validation-row weights, as the `sample_weight` contract promises.
- **`conformalize=True` on asymmetric quantile grids no longer breaks the
  non-crossing guarantee.** Unpaired levels kept scale 1.0 while their
  neighbors shrank and could be jumped; they now interpolate their factor from
  the paired levels. A grid with no symmetric pair at all raises instead of
  silently skipping calibration.
- **A bagged binary fit survives a bootstrap member missing the rare class**
  (one row of the missing class is injected) instead of crashing with "Need
  at least 2 classes".
- **`ordered_boosting=True` with `l2_leaf_reg=0` no longer crashes** on
  singleton leaves (ZeroDivisionError in the leave-one-out step).
- **Refitting on a plain array clears the previous fit's feature names**, so
  the column-order guard no longer misfires against stale names.
- **The multi-quantile split search weights the hessian by `sample_weight`**,
  matching the scalar path; weighted fits previously optimized a different
  objective in the structure than in the leaves.
- **`groups` is honored under bagging:** a member's out-of-bag early-stopping
  rows now exclude every group present in its training sample (falling back
  to the member's group-aware auto-split when none remain). Previously the
  group boundary was silently ignored for `n_ensembles > 1`.
- **``mkdocs build --strict`` failed on two pre-existing warnings.** The
  ``ChimeraBoostQuantileRegressor`` docstring closed its ``Parameters`` section
  with a free prose paragraph, which griffe parsed as three malformed parameters
  (``Other``, ``defaults``, ``grid``) and rendered as garbage; it is a ``Notes``
  section now. And ``docs/benchmarks.md`` pointed at ``../images/public_pareto.png``,
  outside the docs tree, so the chart did not render on the published page.

### Changed
- **Small-batch `predict` is up to ~1.4x faster.** The serial/parallel kernel
  dispatch threshold assumed the parallel forest walk overtakes serial at about
  5 rows; re-measuring both kernels on the same packed forest puts the crossover
  between 32 and 64, so every 5-to-32-row predict was paying thread fork/join
  for nothing. The threshold moves from 4 to 32. The two kernels are
  bit-identical, so predictions are unchanged. `warmup()` now derives its
  parallel-batch row count from the threshold rather than hardcoding it, so a
  future change cannot silently leave the parallel kernel uncompiled.
- **Binning is up to ~4x faster on zero-inflated columns and ~3x faster on
  dense ones.** The greedy border pass walked every distinct value in a Python
  loop; it is now a numba kernel, its per-value mass is read off the sort's run
  lengths instead of a `searchsorted` + `np.add.at` pass, and the quantile
  probe partitions the already-sorted copy in place. Borders are bit-identical
  throughout (pinned against transcriptions of the old code). 200k x 30
  zero-inflated: 1.34 s -> 0.33 s; dense: 0.29 s -> 0.10 s.
- **Bagged members draw whole groups when `groups` is passed.** The
  group-disjoint out-of-bag eval set introduced above was correct but dead in
  practice: a typical 80% row draw touches essentially every group, so it came
  back empty and every grouped member fell back to its auto-split. Drawing
  `max_samples` of the *groups* (a cluster bootstrap at 1.0) always holds at
  least one group out, so members early-stop on groups they never saw and no
  longer carve a validation slice out of their own sample. Held-out-group
  strength is unchanged on a 24-config synthetic panel (11W-13L, median
  +0.05%) with ~20% faster bagged fits. `groups=None` draws are byte-identical
  to before.
- **``refit_full`` now defaults to ``"replay"``: the same full-data refit for
  about two thirds of the fit time.** Refitting the early-stopping winner on
  100% of the rows has been on by default since 0.25.0, and fresh attribution
  puts that second, from-scratch fit at 37-49% of every default fit. But
  growing trees is 83-85% of a fit and is a SEARCH, and the refit already
  knows the structures it is rediscovering. ``"replay"`` replays the winner's
  splits round by round against gradients computed on all rows and refits only
  the leaf values (and the linear-leaf coefficients), so the held-out rows
  still shape every leaf value without the split search being paid for twice.

  Measured against ``refit_full=True`` at 3 seeds, accuracy is a wash on both
  decision suites while fit time falls sharply: **Grinsztajn (59 datasets)
  27W-32L, mean +0.005%, median -0.005%, fit time -34.8% and faster on 58 of
  59**; high-cardinality (14 datasets) 3W-6L-5T, mean -0.017%, median +0.000%,
  fit time -15.2%.

  ``refit_full=True`` still selects the from-scratch refit. Scalar boosters
  only: multiclass grows one vector-leaf tree per round through a separate
  loop and keeps the from-scratch refit, so ``"replay"`` is an exact no-op
  there. ``quality=1``/``2`` already disable refitting, and ``quality=4``/``5``
  are unaffected because ``refit_full`` is a no-op inside bagged members — so
  the only rung this moves is ``3``, the default. Like ``refit_full=True`` it
  does nothing with an explicit ``eval_set``, ``early_stopping=False``, or
  ``loss="Quantile"``. Evidence: ``benchmarks/REPLAY_PLAN.md``.

### Docs
- **README and user docs rewritten for readability.** README restructured into
  Install / Quickstart / What it is / Documentation / Why / Citations, with the
  TabArena chart captioned in words. Across the docs, benchmark-report content
  (win-loss records, suite names, seed counts) and version-history asides give
  way to direct guidance, since that history lives here. ``parameters.md`` cells
  are shortened and carry scikit-learn style "See the User Guide" links into
  ``recipes.md``; the estimator docstrings gained the matching "Read more in the
  User Guide" line.
- **API reference split into one page per public name** (an ``api/index.md``
  overview plus a page each for the three estimators, ``CustomObjective``,
  ``quantile_metrics`` and ``warmup``), replacing a single page that had grown
  to 197 KB. The overview keeps the old ``/api/`` URL. ``navigation.indexes`` is
  on so the section header links to it.
- **``docs/benchmarks.md`` is now in the site nav**, under Reference. It was
  reachable only through a GitHub link from the README.
- New "Cross features" section in ``concepts.md`` explaining why an oblivious
  tree needs an ``x1 - x2`` column to express a comparison between two features.
  That rationale previously existed only inside a parameters table cell.

## [0.25.0] - 2026-07-26
### Added
- **``quality=1..5``: named operating points on the speed/accuracy curve.**
  ``1`` fast, ``2`` balanced (0.24.0's defaults), ``3`` accurate (the
  current defaults), ``4`` ensemble, ``5`` max. Each rung only pins
  parameters that already exist, so nothing new is computed;
  ``quality=None``/``quality=3`` are the defaults and ``quality=2``
  reproduces 0.24.0 behaviour. Measured on Grinsztajn (3 seeds, one
  co-run so the column is comparable), **all five rungs sit on the
  accuracy/speed Pareto frontier**: 31.1% of head-to-head matchups won at
  1.95x fit time, 45.6% at 5.22x, 69.9% at 9.15x, 83.7% at 17.07x, 94.0%
  at 24.74x — against LightGBM's 21.6% at 1.14x and CatBoost's 40.9% at
  13.93x, which is dominated by rung 3 on both axes. Rung 1 buys out the
  configuration search (the default auditions constant-vs-linear leaves
  and plain-vs-cross features, costing two to four boosting fits) and
  fits once; it still beats LightGBM 38W-21L on Grinsztajn, but on
  high-cardinality categorical data it loses to the default (1W-6L-7T)
  and saves only 1.5x, so prefer ``quality=2`` there. Rungs 4 and 5 build
  on the defaults, not on rung 3 — ``refit_full`` is a no-op inside bagged
  members. Where ``quality`` collides with a parameter you set yourself it
  wins and warns. Evidence: benchmarks/SELECT_PLAN.md.
- **``chimeraboost-warmup`` command** (also ``python -m chimeraboost``) —
  compiles the numba kernels and caches them on disk, so the wait lands where
  you chose rather than inside the first ``fit``. Run it after installing,
  and after every upgrade: numba stamps each cache entry with its source
  file's mtime and size, so a new version silently resets the cache. This is
  the closest thing to compiling at install time that numba supports —
  ahead-of-time compilation (``numba.pycc``) requires ``numpy.distutils``,
  removed in numpy 1.26, cannot compile ``prange``, and would make the wheel
  platform-specific; and a pre-built cache cannot be shipped in a wheel
  because its key includes the building machine's exact CPU model and
  feature set as well as the installed file's timestamp.
- **``warmup(shap=True)`` / ``chimeraboost-warmup --shap``** — the SHAP
  kernels are ~3.7 s of compile that most callers never reach, so they stay
  opt-in; without this the cost landed on the first ``shap_values`` call.
- **A one-line notice on a cold first fit.** A silent 15-second first call is
  indistinguishable from a hang. It prints to stderr only when the on-disk
  cache really is cold — roughly once per installed version, not once per
  session — and ``CHIMERABOOST_NO_NOTICE=1`` silences it, as does having
  ``CHIMERABOOST_WARMUP`` set.
- **``benchmarks/cold_start.py``** — a reproducible cold/warm/steady-state
  measurement, including per-kernel compile cost. The numbers in
  docs/deployment.md now come from it; previously they came from scratchpad
  scripts that were never committed and had gone stale by six releases.

### Fixed
- **``warmup()`` missed the small-batch constant-leaf predictor.** A warmed
  serving process still stalled ~0.31 s on its first single-row ``predict``
  for a model without linear leaves — exactly the case warmup exists to
  prevent. Now 0.2 ms. The guard test that should have caught this asserted
  against a hand-written list of 11 kernels and, run in a full suite, passed
  only because an earlier test compiled one of them as a side effect; it now
  enumerates every numba kernel in the package and fails on any that warmup
  leaves uncompiled without a written reason.

### Changed
- **Vector-leaf multiclass.** ``MulticlassBoosting`` now grows ONE oblivious
  tree per round with K-vector Newton leaves (shared structure, per-class
  values) instead of K per-class trees; splits are scored on a centered
  Rademacher projection of the K gradient columns (a 1-d Newton sketch —
  SketchBoost-style), reusing the scalar split kernels, quantized path
  included. Multiclass models get better-calibrated and faster: synth
  screen Brier 23W-8L +5.0% at F1 parity with fits 2.5× faster; OpenML
  gate 10W-0L across every multiclass set; high-card multiclass fits 1.6×
  faster at parity, and the 8-member ensemble now leads CatBoost on both
  hc multiclass quality columns. Regression/binary paths are bit-identical
  (Grinsztajn 59/59 exact ties). Models pickled on ≤0.24.0 still load and
  predict via a legacy per-class-forest path. Evidence:
  benchmarks/A1_PLAN.md.

- **``refit_full`` (default ``True``): full-data refit of the early-stopped
  winner.** The automatic early-stopping split holds out 20% of the training
  rows that the final model otherwise never learns from; the winning
  configuration (selected linear-leaf variant, cross pairs, resolved
  learning rate) is retrained on 100% of the rows at the selected budget,
  rounds scaled by the train-size ratio. Temperature scaling transfers;
  ``validation_history_`` keeps the early-stopped fit's curve.
  **On by default because it is the strongest single-model setting
  measured** — Grinsztajn 48W-11L +2.0% (Brier 18W-5L +2.4%), high-card
  10W-3L (Brier 8W-0L), OpenML one-shot gate 26W-8L +1.3%, synth screen
  95W-39L +1.0%; on the ladder co-run it wins 69.9% of head-to-head
  matchups where the non-refit model wins 45.6%. **It costs about one extra
  fit** (5.2x → 9.2x slowdown), so upgrading from 0.24.0 makes a default fit
  both slower and more accurate, with different predictions. Pass
  ``refit_full=False`` or ``quality=2`` for the 0.24.0 behaviour.
  No-ops (bit-identical) with an explicit ``eval_set``,
  ``early_stopping=False``, ``loss="Quantile"`` (its conformal offset needs
  a genuine holdout), and inside bagged members (their OOB rows are already
  an external eval set — so ``n_ensembles`` builds on the non-refit model
  rather than stacking with this). Evidence: benchmarks/REFIT_PLAN.md,
  benchmarks/SELECT_PLAN.md.

## [0.24.0] - 2026-07-24
### Added
- **Custom objectives.** The regressor's ``loss`` accepts a user objective
  instance: subclass ``chimeraboost.CustomObjective``, implement
  ``grad_hess(y, raw)`` and ``eval(y, raw, sample_weight=None)``, optionally
  override ``init`` / ``transform``. Flows through the full stack —
  quantized histograms, subsampling, bagging, pickling.
- **Four new built-in regression losses on that hook:** ``"Huber"``
  (``delta``), and the log-link family ``"Poisson"``, ``"Gamma"``,
  ``"Tweedie"`` (``tweedie_variance_power``), whose predictions are
  ``exp(raw) > 0``. y-domain violations raise clear errors at fit.
- **``eval_metric``** on both estimators: a callable
  ``metric(y_true, y_pred[, sample_weight])`` scored on the validation set
  each round, driving early stopping and the internal model selections
  instead of the training loss; ``greater_is_better = True`` attribute
  supported (``validation_history_`` records negated values).
  All defaults are bit-identical: string losses take the exact old path and
  ``eval_metric=None`` changes nothing (numerical-identity goldens green).

## [0.23.0] - 2026-07-23
### Changed
- **Categorical combinations are pairs, not concatenated strings.** A combo
  category is now the pair of its parents' canonical categories. The old
  "a_x_b" string building aliased distinct pairs whose values contain the
  delimiter, turned real missing values into the literal string "nan"
  (bypassing the ``__nan__`` missing sentinel), and split int/float
  spellings of one value ("1" vs "1.0"). Grouping — and therefore the model
  — is unchanged on data without those edge cases; models pickled with the
  old string-keyed maps keep predicting through the legacy path. Combo
  fits also skip the per-row string building.
- **Cross-feature columns are written straight into their output block**
  (ufunc ``out=``), removing a per-column temporary and the final
  ``column_stack`` copy at fit and predict. Bit-identical.
- **Tiny predict batches (≤4 rows) take serial kernel twins.** The binning
  and forest-walk kernels are parallel; on a 1-row call the OpenMP
  fork/join (~20 µs on 12 threads) costs more than the whole walk, and the
  parallel kernels only overtake serial around 5 rows. Measured numeric
  1-row predict: 56 µs → 36 µs. Bit-identical (the fit path already uses
  the same serial-twin dispatch); ``warmup()`` compiles both sides.
### Fixed
- **Grouped classification's automatic early-stopping split honors
  `random_state`.** It used an unshuffled `StratifiedGroupKFold`, so the
  holdout was always the same first fold and the seed was inert — unlike
  every other split branch. A seed now shuffles fold selection;
  `random_state=None` keeps the historical deterministic split.
- **Bool columns are no longer named in the "add these to cat_features"
  error guidance** (they cast to 0/1 cleanly; the guidance was wrong).

## [0.22.0] - 2026-07-23
### Changed
- **Cross-feature columns no longer re-cast their parent columns per pair.**
  `_cross_block` reads numeric parents from the float64 numeric block the
  transform already builds (one cast per input column instead of one per
  cross pair). On object-dtype input (categoricals present) those per-pair
  element-wise casts dominated cross-feature predict cost: 4k-row predict
  on a cross-selected model with 2 categoricals 13.6 ms → 8.6 ms, 1-row
  predict 0.35 ms → 0.27 ms; fit improves too (same code path). Bit-identical.
- **Predict converts the input once, not twice.** The predict-time inf
  check's model-array conversion is now reused for the prediction itself
  instead of being discarded and redone; on DataFrame input this removes a
  full second `to_numpy` materialization per predict call. Bit-identical.
- **Bagged predict switches the numba thread count once per call, not once
  per member** (only observable with an explicit `thread_count`; each
  switch charges an ~1 ms OpenMP re-team on the next parallel region).
- **Multiclass predict skips a dead full-matrix init**, and `shap_values`
  on linear-leaf models reuses the packed forest predict already caches
  instead of repacking on every call.
- **`warmup()` now covers the weighted ordered-TS kernel and the gdiff
  group-sum kernel** (a weighted categorical regression fit plus a direct
  kernel call). Both sit on default paths — `sample_weight` fits, and
  ≥2000-row categorical fits (auto cross features) — but were not compiled
  by warmup, so short-lived workers still hit a JIT stall there.
### Fixed
- **`leaf_estimation_iterations > 1` no longer corrupts MAE/Quantile
  models.** These losses set each leaf to the exact minimizer (median /
  alpha-quantile); the extra Newton refinement steps then ran anyway,
  adding sign-gradient steps on top and degrading the model — while the
  sklearn layer warned the parameter "will have no effect". The warning is
  now true: refinement is skipped when leaves are loss-corrected. Defaults
  (`leaf_estimation_iterations=1`) are unaffected.
- **A depth-0 early exit now keeps the best validation prefix.** When
  boosting stopped because a round produced no legal split, the
  "truncate to the best validation iteration" step was skipped, silently
  keeping trees grown after the validation optimum. That exit now truncates
  exactly like patience and budget exhaustion do (scalar and multiclass).

## [0.21.0] - 2026-07-22
### Changed
- **pandas is no longer a dependency.** The categorical machinery
  (factorize, predict-time code mapping, gdiff group means) is now
  numpy/numba only, bit-identical to the pandas implementation (the gdiff
  group sums replicate pandas' Kahan-compensated groupby exactly).
  DataFrames still work as input — they are consumed through their own
  `to_numpy`/`columns` attributes, so users who pass frames already have
  pandas (or polars) installed. Installs pull four packages instead of five.
- **Bagged predict no longer redoes the categorical transform per member.**
  Members factorize each categorical (and combo) column once per call
  through a shared cache and remap only the unique values through their own
  fit-time maps; the input conversion and validation also run once instead
  of once per member. Predictions are bit-identical. 50k rows,
  `n_ensembles=8`, 4 string categoricals: binary predict 1.21 s → 0.40 s,
  multiclass 1.78 s → 0.74 s; single-model predict is unchanged at big
  batches and ~9× faster at 1-row calls (0.19 ms vs 1.7 ms — the per-call
  pandas overhead dominated small batches).
### Fixed
- **Bagged members no longer keep their fit-time thread cap at predict.** A
  parallel `n_ensembles` fit divides the thread budget across workers
  (budget/K threads per member), but the members retained that sliver
  permanently — and they predict sequentially, so an 8-member bag walked each
  forest on 1–2 threads while the rest of the machine idled. Members now get
  the parent's `thread_count` back after fitting. Predictions are
  bit-identical (per-row tree walks don't depend on thread count); bagged
  predict is ~3–5× faster on an 8-core box (50k rows, `n_ensembles=8`:
  numeric 0.80 s → 0.24 s, with categoricals 2.31 s → 1.60 s). The remaining
  categorical predict gap (each member redoes the string→code mapping) is a
  known follow-up.

## [0.20.0] - 2026-07-20
### Added
- **Group-centered categorical crosses.** The `cross_features` race now also
  auditions `gdiff` columns — a numeric column minus the fit-time mean of
  that column within the row's category (`x_i - mean(x_i | c_j)`), for the
  top-4 numeric × top-3 categorical features by base-fit importance — so
  "above this row's own category's baseline" becomes one split instead of a
  per-category staircase. Group means are target-free (the same map serves
  fit and predict; unseen categories fall back to the global mean) and
  weight-aware (zero-weight rows never shape a mean). Auto engagement now
  also covers datasets with one numeric plus categorical features. On the
  high-cardinality suite: +0.25% mean primary (employee_salaries +2.5%,
  okcupid-stem +0.5%, black_friday +0.3%), losses all ≤ 0.12%, at 1.32×
  suite fit; Grinsztajn is bit-identical by construction (its loaders pass
  no categorical features); the independent OpenML gate came out flat
  (+0.003%, bank-marketing +1.0%).
### Changed
- Fit no longer evaluates the training loss every round for the internal
  selection-audition callbacks that never read it; user callbacks still
  receive the documented value. Predictions are bit-identical; a default
  50k-row binary fit is ~15% faster and the Grinsztajn suite fit sum ~3%
  faster.

## [0.19.0] - 2026-07-20
### Added
- With `verbose=True`, fit now prints a notice when `early_stopping=True`
  silently holds out `validation_fraction` of the training rows as the
  validation set (the default path when no `eval_set` is passed), so the
  effective training size is visible. Default `verbose=False` is unchanged.
### Changed
- **Classifier `leaf_estimation_iterations` default is now `None` (auto)
  instead of a concrete `3`; it resolves to 3 and fits bit-identically.** The
  effective value is unchanged — an A/B across the synthetic, Grinsztajn,
  highcard and OpenML suites confirmed `3` is correct: it helps small and
  categorical-heavy binary fits (e.g. `credit-g`, `kc2` on the independent
  gate) and is provably inert everywhere linear leaves take over (Grinsztajn
  came out 59/59 bit-identical) or for multiclass. `None` simply stops the API
  from advertising a refinement count that is dead for multiclass and shadowed
  for large binary. The regressor's honest `1` is unchanged.
### Fixed
- The inert-knob warnings for `leaf_estimation_iterations` now fire for any
  **explicitly-set** value `> 1` that will be ignored on the path about to run
  (linear leaves active, or multiclass), while the auto default stays quiet —
  previously the former default `3` was silently exempted, so an explicit `3`
  meant to request refinement was never flagged.
- **`sample_weight` is now honored everywhere, not just in the gradient
  step.** Rows are weighted in the ordered-target categorical encoder (prior
  and per-category statistics), in the quantile bin borders, and in the
  early-stopping validation metric on an automatically split (or bagged
  out-of-bag) holdout. Previously a `sample_weight=0` row still shaped the
  categorical encodings and bin edges, and — worst — still scored the
  early-stopping metric with full weight: in an adversarial repro (zeroed-out
  rows carrying garbage targets) the reported validation RMSE was ~2000
  instead of ~0.1, which stopped training at the wrong iteration and could
  degrade the model by an order of magnitude. Uniform weights collapse to the
  unweighted path, so `sample_weight=None` and any constant weight vector stay
  bitwise-identical to before. An explicitly passed `eval_set` is still scored
  unweighted (it carries no weights); a third `(X, y, sample_weight)` element
  is now accepted for callers that want to supply them.

## [0.18.1] - 2026-07-20
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
