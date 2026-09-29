# MISSING_PLAN — missing values: imputation vs native handling

Opened 2026-09-29. Goal: better predictions on data with gaps. Question: does
MICE-style imputation (miceforest / missForest) beat improving ChimeraBoost's
own NaN handling?

## Where we start
- NaN rides in a reserved top bin (`binning.py:5-13`), so it always goes with
  the largest values at every split. No learned direction, no indicators.
- Learned default direction killed 2026-09-22 (I045, `CAMPAIGN_PLAN.md:377`):
  zero Grinsztajn sets have numeric NaN, so nothing could show a gain.

## Literature (2026-09-29 survey)
- Prediction with boosted trees: native NaN handling + missing indicators
  matches or beats impute-then-predict (Perez-Lebel 2022 GigaScience; Le Morvan
  & Varoquaux 2025 ICLR "diminishing returns"; Le Morvan 2021 NeurIPS).
- Imputation quality without torch: HyperImpute (ICML 2022), UnmaskingTrees
  (TMLR 2025) edge missForest; mice_cart best on distributional fidelity
  (Grzesiak/Näf 2025). GAIN/MIWAE/diffusion lose. TabImpute (TabPFN) leads but
  needs torch — out.

## Steps
1. Test bed `@mcar/@mar/@mnar` (`--miss`, `VARIANTS.md`) — **done 2026-09-29**.
2. Bake-off arms — **built 2026-09-29, run pending on Nathan's machine**
   (HuggingFace is blocked from the cloud container). `benchmarks/missing_arms.py`,
   runners `CBMissInd CBMissMIA CBMeanImp CBMissForest CBMiceForest CBMice
   CBMaskedImp` (off by default; `tests/test_missing_arms.py`). Each is a
   train-fitted transform, then the default ChimeraBoost; imputation time
   counts as fit time. MIA is emulated by a mirrored column with NaN below the
   minimum, so no kernel change is needed to test it.
   - Barrier check 2026-09-29: none of the 23 closures in `BARRIERS.md`
     concern missing values.
   - **Forecast (pre-registered 2026-09-29):**
     - `CBMissInd` is an exact tie with the baseline. The NaN top bin already
       gives the "missing vs not" split, so the indicator column is redundant.
     - `CBMissMIA` gains on `@mnar`, a little on `@mar`, and is flat on
       `@mcar`, at about 1.3x fit time.
     - Model imputers gain on `@mcar`/`@mar` (+0.5 to +2% RMSE on correlated
       sets, 0 on sets with independent features) and are 3x (CBMice,
       CBMaskedImp) to 30x (CBMissForest) slower.
     - `CBMaskedImp` matches `CBMice` accuracy at lower cost.
     - The HC real-NaN sets are flat for every arm.
   - **Smoke run 2026-09-29** (one stand-in dataset: Friedman plus 5 noisy
     copies, 10k rows, 1 seed; not evidence, only a check that everything
     runs). RMSE:

     | arm | @mcar | @mar | @mnar | fit s (mcar) |
     |---|---|---|---|---|
     | baseline | 1.939 | 1.235 | 1.357 | 0.7 |
     | CBMissInd | 1.939 | 1.235 | 1.357 | 0.9 |
     | CBMissMIA | 1.941 | 1.227 | 1.347 | 1.3 |
     | CBMeanImp | 1.935 | 1.236 | 1.415 | 1.1 |
     | CBMissForest | 1.912 | 1.219 | 1.371 | 29.8 |
     | CBMiceForest | 1.977 | 1.244 | 1.377 | 7.7 |
     | CBMice | 1.889 | 1.208 | 1.353 | 4.0 |
     | CBMaskedImp | 1.908 | 1.202 | 1.345 | 3.1 |
   - **Trap: miceforest 6.0.5 breaks on LightGBM >= 4.6.** Installing
     `lightgbm<4.6` would also move the LightGBM competitor, so run
     `CBMiceForest` from a separate environment.
   - **Trap: miceforest's new-data path.** `complete_data()` defaults to
     iteration -1, which on new data returns iteration 0, the random initial
     fill. The arm passes the iteration explicitly, and a test covers it.
   - **Screen:**
     `python benchmarks/run_benchmarks.py --grinsztajn --highcard --miss --no-variants --seeds 1 --models ChimeraBoost CBMissMIA CBMeanImp CBMissForest CBMice CBMaskedImp --save`
     (add `CBMissInd` only to confirm the tie). Read each arm against the
     baseline with
     `compare_runs.py RUN.json RUN.json --model ChimeraBoost --model-new <arm> --by-suite`.
     Survivors go to 3 seeds.
3. Ship rule: beats baseline on the `@miss` strata, no loss on the real-NaN
   HC sets or the plain strata; cost counts (Pareto). If imputers tie MIA +
   indicators, ship the cheap one and close this file.
