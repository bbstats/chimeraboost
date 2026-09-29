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
2. Bake-off arms — **built 2026-09-29; adversarially reviewed and reworked the
   same day; run pending on Nathan's machine** (HuggingFace is blocked from the
   cloud container). `benchmarks/missing_arms.py`, runners `CBMissInd CBMissMIA
   CBMeanImp CBMissForest CBMiceForest CBMice CBMaskedImp` (off by default;
   `tests/test_missing_arms.py`). Each is a train-fitted transform, then the
   same ChimeraBoost config as the baseline; imputation time counts as fit time.
   - **Design (after review): augment, never replace.** Every arm keeps the
     original columns (NaN intact) and appends one filled copy per NaN-bearing
     column, so all arms have equal width and differ only in where the copy puts
     missing rows: just below the minimum (MIA: the original routes NaN high, the
     copy low, so each tree level picks — learned direction without a kernel
     change), the mean, or a model's prediction. No indicators: the original's
     top-bin split already isolates missing rows (`CBMissInd` tied exactly).
   - Barrier check 2026-09-29: none of the 23 closures in `BARRIERS.md`
     concern missing values.
   - Test-bed deviation: one rate (30%), not the 20% and 40% first planned —
     halves the cost; revisit only if the screen finds a rate-sensitive arm.
   - **Forecast, pre-registered 2026-09-29 before any run** (the scored one):
     - Indicators and MIA gain on `@mnar`, a little on `@mar`; flat on `@mcar`.
     - The imputers roughly tie MIA on accuracy and are 2–20x slower.
     - `CBMaskedImp` lands between mean fill and missForest.
   - Post-smoke expectation (written after the smoke runs — NOT a
     pre-registration, not scored): model imputers beat MIA on all three
     mechanisms; `CBMaskedImp` ≈ `CBMice` at lower cost; HC flat.
   - **Smoke runs 2026-09-29** (one stand-in dataset: Friedman + 5 noisy
     copies, 10k rows, 1 seed; not evidence, only a check that everything
     runs). RMSE, lower is better:

     | arm | @mcar | @mar | @mnar | fit s (mcar) |
     |---|---|---|---|---|
     | baseline | 1.939 | 1.235 | 1.357 | 0.7 |
     | first design (replace + indicators): | | | | |
     | CBMissInd | 1.939 | 1.235 | 1.357 | 0.9 |
     | CBMissMIA | 1.941 | 1.227 | 1.347 | 1.3 |
     | CBMeanImp | 1.935 | 1.236 | **1.415** | 1.1 |
     | CBMissForest | 1.912 | 1.219 | **1.371** | 29.8 |
     | CBMiceForest | 1.977 | 1.244 | 1.377 | 7.7 |
     | CBMice | 1.889 | 1.208 | 1.353 | 4.0 |
     | CBMaskedImp | 1.908 | 1.202 | 1.345 | 3.1 |
     | augment design (current): | | | | |
     | CBMissMIA | 1.929 | 1.229 | 1.345 | 0.8 |
     | CBMeanImp | 1.941 | 1.226 | 1.339 | 0.8 |
     | CBMissForest | 1.924 | 1.210 | 1.323 | 26.9 |
     | CBMiceForest | 1.983 | 1.249 | 1.332 | 8.9 |
     | CBMice | 1.900 | 1.209 | 1.321 | 3.9 |
     | CBMaskedImp | 1.911 | 1.203 | 1.307 | 3.3 |

     Bold: the first design lost to the baseline under MNAR — consistent with
     replacing the column throwing away the "value > t or missing" single
     split. Under the augment design every imputer beats the baseline there
     (one stand-in dataset; a mechanism check, not a result).
   - **Review fixes 2026-09-29** (besides the design):
     - `CBMaskedImp` hides only the target cell. The first version also masked
       random cells, including never-missing columns (under MAR the very
       drivers of the gaps), and trained on ~51% missing against ~30% at use.
     - `CBMaskedImp` memory: training pairs capped by a 25M-cell budget,
       prediction chunked. Uncapped it reached 2–3 GB per job on Bioresponse
       (419 cols), Mercedes (359), topo_2_1 (255), with 5 jobs in parallel.
     - Arms take the baseline's `--chimera-*` config (they silently ran the
       function defaults; identical today by luck — now a test).
     - MIA copy just below the minimum (`np.nextafter`), not a full range
       below: same tree splits, no distortion of linear leaves/cross features.
     - All-missing columns skipped; no `keep_empty_features` (sklearn ≥ 1.0).
   - **Trap: miceforest.** 6.0.5 calls a private LightGBM method that changed
     in 4.6, so every fit raises — the harness now refuses `CBMiceForest` up
     front when LightGBM ≥ 4.6. It also pulls pandas ≥ 2.1 + pyarrow. Run it in
     its own venv on A: (recipe below). Its new-data path returned the random
     initial fill unless the iteration is passed explicitly (fixed, tested).
   - **Rival strength:** screen missForest is reduced-cost (50 trees, ≤ 10k rows
     per tree); miceforest runs 3 iterations with mean matching off (its best
     point-fill setting). Any "beats missForest/miceforest" claim needs a
     full-strength rerun first — open, 2026-09-29.
   - **Recipe (PowerShell, repo root).** Screen, seed 0:
     `python benchmarks/run_benchmarks.py --miss-only --seeds 1 --models ChimeraBoost CBMissMIA CBMeanImp CBMissForest CBMice CBMaskedImp LightGBM --save`
     (`--miss-only` = the twins + HC sets with numeric NaN in training; LightGBM
     learns NaN direction natively — a reference, not an arm.) miceforest
     head-to-head, own environment:
     `python -m venv A:\code\venv-miceforest`, then
     `A:\code\venv-miceforest\Scripts\python.exe -m pip install -e . miceforest "lightgbm<4.6"`, then
     `A:\code\venv-miceforest\Scripts\python.exe benchmarks/run_benchmarks.py --miss-only --seeds 1 --models ChimeraBoost CBMiceForest CBMaskedImp --save`.
     Read: `compare_runs.py RUN.json RUN.json --model ChimeraBoost --model-new <arm> --by-suite`,
     and head-to-head `--model CBMissMIA --model-new <imputer>`.
   - **Screen rule:** an arm survives if it beats the baseline on ≥ 1 of
     `@mcar/@mar/@mnar` at sign-test p < 0.10 and loses no stratum at p < 0.10.
   - **Confirm rule:** survivors + baseline + `CBMissMIA` on FRESH seeds
     (`--seed-start 1 --seeds 3`; the screen's seed 0 picked the winners, so
     reusing it would be a winner's curse). A win needs p < 0.05 on the stratum
     it won in the screen, no stratum worse at p < 0.10, and a median gap ≥ 0
     on the HC real-NaN sets.
3. Ship rule:
   - MIA best, or tied with the best imputer head-to-head → build the learned
     NaN direction natively, NaN-gated (NaN-free data bit-identical, goldens
     green); check native vs emulated on the confirmed strata.
   - An imputer beats MIA head-to-head → an opt-in estimator option; HC evidence
     decides any default.
   - Either way the headline Pareto cannot move: its suites carry no numeric
     NaN except HC. Say so in the verdict.

## Follow-ups (dated)
- 2026-09-29: `CBMaskedImp` one-hot "which column" is weak on wide data (an
  oblivious tree level must be spent per column). Idea: slot-aligned inputs —
  for target j, feed its top-K correlated columns in fixed slots (sign-flipped)
  plus their |corr|, so one model learns a column-agnostic rule. Only if the
  screen shows the arm losing on the wide sets.
- 2026-09-29: full-strength missForest/miceforest rerun before any "beats"
  claim (see Rival strength).
