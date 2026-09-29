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
2. Bake-off, downstream accuracy, sign-tested per stratum
   (`compare_runs.py --by-suite`) — **open, next**. Arms:
   - baseline: current top-bin NaN
   - + missing-indicator columns
   - MIA: per-split NaN-left / NaN-right (revisits I045)
   - mean impute + indicators
   - miceforest + indicators (benchmark-only dep)
   - ChimeraBoost-MICE (IterativeImputer with our regressor) + indicators
   - masked single-model imputer: one ChimeraBoost trained on randomly
     re-masked copies, predicting any cell from (row, column id)
3. Ship rule: beats baseline on the `@miss` strata, no loss on the real-NaN
   HC sets or the plain strata; cost counts (Pareto). If imputers tie MIA +
   indicators, ship the cheap one and close this file.
