# REFRESH_PLAN — `refresh(X, y)` on an opt-in stored training set (issue #131)

Started 2026-09-24. Program file for GitHub issue #131 (the maintainer's spec).
Design read from the code by a planning pass on 2026-09-24; file:line refs are
at main 3e02ab1.

## What refresh is

Tree structure, `lr_`, binner borders, count/cross selections and
`cat_combinations` stay pinned; leaf values, linear-leaf coefficients, ordered
target statistics and gdiff group means are recomputed by replaying every tree
on (stored rows + new rows). Mechanism = the existing structure-transfer refit
(`replay_donor`, booster.py:934-1011, tree.py:2126-2169), fed from stored rows
instead of raw X.

## Findings that shape the design

- Invariant holds: replaying a fitted model's own trees on its own training
  rows reproduces it bit for bit, for `refit_full` "replay", True and False, on
  the regressor and the binary classifier (diff/prod/gdiff crosses, count
  column, linear leaves, weights, NaNs). Grown and replayed leaves share the
  float-gradient kernels (quantization touches split search only,
  tree.py:2433-2437, 2501-2507); replay consumes the random stream in the same
  order (booster.py:901-903, 1002-1003).
- The rows to store are the rows the FINAL booster's leaves came from: all
  rows after a refit (default), the 80% split in splitter order when the refit
  is skipped (`refit_full=False`, quality 1-2, `loss="Quantile"`,
  sklearn_api.py:890), X as given with an explicit `eval_set` or no early
  stopping.
- Storage (REVISED 2026-09-24, see Log): numeric columns as BINS of the
  pinned binner (uint16 today, uint8 when every column has <= 256 bins), except
  the parents of any cross column (diff, prod or gdiff), which are kept as RAW
  float64 because crosses are computed from raw values and gdiff group means
  are refit on replay rows (preprocessing.py:420-465, 506-510, 685);
  categoricals as int32 codes plus each column's categories in code order (new
  rows append unseen categories in first-appearance order); y as the booster
  saw it; raw weights. Row order is the booster's order (first-appearance
  codes, positional TS permutations). Count, cross and TS columns are not
  stored: they are rebuilt.
- The ONE preprocessing path: the store is turned back into a raw-equivalent X
  and fed to the existing replay refit unchanged. A bin maps back to a value
  that bins identically, exactly: the binner puts v in the bin equal to the
  number of borders <= v, and non-finite v in the missing slot
  (binning.py:82-93), so bin 0 -> just below the first border, bin k (1..m) ->
  border k-1, the missing slot -> NaN. Linear leaves read the binned matrix
  (booster.py:888), not raw values, so they are unaffected; only the cross
  parents need their raw values.
- The spec's 10-20x storage saving is optimistic for numeric data: about 3x
  when crosses engage (up to ~6 parent columns stay float64), about 6.5x when
  they do not; far more for object-dtype categoricals.
- Exact reproduction needs an integer `random_state` (TS permutations are
  redrawn from `random_state + t`).

## Slice 1 (regressor + binary classifier, single model)

- `store_training_data=False` (last ctor param, `_SKLEARN_ONLY`, validated in
  `_check_flag_params`); `n_samples_trained_`; private `_training_data_`
  (a `TrainingRows`), not a public `X_train_` (in sklearn that means raw X).
- `refresh(X, y, sample_weight=None) -> self`; zero rows allowed; replays the
  FITTED configuration (ignores later `set_params`).
- Raises at fit: bagging (`n_ensembles > 1`, quality 4/5), multiclass,
  `loss="Quantile"`, `random_effects=True`. Raises at refresh: no store,
  unseen class label, weighted fit refreshed without weights.
- `temperature_` (binary) stays frozen.
- Files: a new `chimeraboost/training_rows.py` (`TrainingRows`: capture from
  raw rows with a fitted preprocessor, append new rows, rebuild a
  raw-equivalent X), booster.py (`replay_kwargs()` only), sklearn_api.py
  (capture at the end of both `_fit_single`, `refresh` via a shared
  `_refresh_single` that rebuilds X and runs the existing replay refit).
- Tests (`tests/test_refresh.py`): (a) zero new rows = identity, bit for bit,
  over refit modes, eval_set, no ES, 256+ category column, all cross kinds,
  linear leaves, weights, NaNs, ordered boosting, MAE/Poisson/custom
  objective, subsample/colsample, binary; (b) refresh = a manual replay refit
  on the concatenated raw rows; (c) nothing pinned moves, `n_samples_trained_`
  grows; (c2) refresh(A) then refresh(B) = refresh(A+B); (d) pickle round
  trip; (e) every error.
- Risk: a future preprocessing input that reads raw numeric values beyond
  binning (as crosses do) would silently lose exactness for non-parent
  columns. The refresh tests (a)/(b) over every column kind are the guard;
  `training_rows.py` says so where it decides which columns stay raw.

## Later slices (one GitHub issue each when slice 1 ships)

2 bagging (per-member stores) · 3 multiclass (vector replay; keep multiclass
`refit_full` as is) · 4 quantiles (`loss="Quantile"`, the multi-quantile head)
· 5 random effects · 6 reservoir `store_training_data=int` · 7 `strict_ids`.

## Decisions and open questions

Slice 1 (stated to the maintainer 2026-09-24 as "unless you say otherwise",
no objection when he merged #169; revisit if he says so):
1. DECIDED: private `_training_data_` + public `n_samples_trained_`, not the
   spec's public `X_train_` / `y_train_` / `sample_weight_` (in sklearn
   `X_train_` means raw X; the store is bins, codes and a few raw columns).
2. DECIDED: a weighted fit refreshed without `sample_weight` raises.
3. DECIDED: `temperature_` stays frozen on refresh.

Still open, for their slices:
4. (slice 2) bag members: every member gets all new rows, or a member-seeded
   draw?
5. (slice 6) reservoir rows up-weighted by seen/capacity, or plain rows?

## Log

- 2026-09-24: design pass; slice 1 started as campaign rung I066 (branch
  `campaign/issue131-refresh-slice1`), in two muse passes: internals, then
  the public API and its tests. Docs by Claude.
- 2026-09-24: pass 1a built the design as first written (a stored-rows twin
  of `fit_transform`: `fit_transform_stored`, `Binner.transform_block`, a
  shared `_gdiff_means`, `stored=` plumbing through the booster). Every
  equality was exact (binned matrix, fitted state, leaf values), but it added
  ~550 library lines, ~400 of them a second implementation of the
  preprocessing that would have to be kept in step by hand. DISCARDED at
  review for the reconstruct design above: the same storage, one code path,
  no booster plumbing. The twin's diff and tests are kept outside the repo
  only as a reference.
- 2026-09-24: slice 1 done (I066): pass 1a' `training_rows.py` + `replay_kwargs`
  (198 lines), pass 1b the public API (291 lines); zero-row refresh exact in
  15 configurations, chained refreshes compose, identity snapshot unchanged.
  Smoke: a 60% model refreshed with 30% more rows recovered ~67% of a full
  90% refit's RMSE gain. Awaiting the maintainer's merge.
- 2026-09-24 (maintainer follow-up on PR #170): at scale, a daily update
  (Zurich delays, 3.3M-row model, 50k new rows) refreshes in 70 s against
  a 232 s full refit and a 216 s original fit, same RMSE (3.04146 vs
  3.04145); a predict pass over the store is 9.0 s, so a refresh is ~8
  predict passes, not the spec's ~1. Most of it was the linear-leaf fit,
  now faster and bit-identical (PR #170, commit 435e866): 9.9 -> 7.8 ms per
  tree at 517k rows. Lead for more: gather uint16 bins instead of float64
  design values in `_linear_leaf_fit` (the gather is ~76 MB per call).
- 2026-09-25: the maintainer left PR #170 open ("not sure I want it
  implemented"). Slice 1 is PARKED; slices 2-7 are not started. If #170 is
  closed, salvage the bit-identical replay-kernel speedup (435e866) as its
  own PR: it speeds up every linear-leaf fit.
- 2026-09-26: un-parked. The maintainer: "let's move forward with 170,
  but clean up the merge conflicts then i'll merge the PR". Main merged
  into the branch (conflicts only in CHANGELOG and two plan files; no
  code overlapped), so the salvage note above no longer applies. Slices
  2-7 are not started.
- 2026-09-26: **slice 1 SHIPPED**, merged as PR #170 (fa577cc).
