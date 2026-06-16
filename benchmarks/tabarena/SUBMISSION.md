# TabArena submission — ChimeraBoost (default entry)

Ready-to-paste drafts for the two-step community submission. Step A = model
integration issue (tabarena.ai/code). Step B = results PR
(tabarena.ai/community-results). Submitting the **default** row first; the tuned
row follows after the ~2–3-week full-tuned run.

Status of the technical gates (all green as of 2026-06-15):
- ✅ Public `AbstractModel` impl: [`chimeraboost_tabarena_model.py`](chimeraboost_tabarena_model.py); library at https://github.com/bbstats/chimeraboost (`pip install chimeraboost`).
- ✅ Passes the default model unit test: [`model_unittest.py`](model_unittest.py) → 3/3 (binary/multiclass/regression) via AutoGluon `FitHelper`.
- ✅ Default hyperparameters + HPO search space specified (below).
- ✅ Promising results on TabArena-Lite (Elo 1211) and full TabArena (Elo 1220).

---

## Step A — Model integration issue

**Title:** Add model: ChimeraBoost (pure-Python CatBoost-inspired oblivious GBDT)

**1. Public model implementation.**
ChimeraBoost is a standalone gradient-boosting model (not an ensembling pipeline):
oblivious/symmetric trees, histogram split-finding, ordered target-statistics for
categoricals — implemented in pure Python + numba, no heavy deps. Library:
https://github.com/bbstats/chimeraboost (`pip install chimeraboost`). AutoGluon
`AbstractModel` wrapper: `benchmarks/tabarena/chimeraboost_tabarena_model.py`
(`ag_key="CHIMERA"`, `ag_name="ChimeraBoost"`). Passes the default unit test
(`model_unittest.py`, `FitHelper.fit_and_validate_dataset`, 3/3 problem types).

**2. Preprocessing and hyperparameters.**
- Preprocessing: AutoGluon `LabelEncoderFeatureGenerator` for categoricals; the
  categorical column indices are passed to ChimeraBoost as `cat_features` so it
  applies its native ordered-target-statistics encoding (numeric features passed
  through; NaN routed to a missing bin).
- Default hyperparameters: `n_estimators=500`, `early_stopping=True` (internal
  validation split), `thread_count=-1`, `random_state=0`; all other knobs at
  library defaults (depth=6, learning_rate auto, l2_leaf_reg=1.0, max_bins=128…).
- HPO search space (200 random configs, TabArena convention):
  `learning_rate` LogReal(0.03,0.3), `depth` Int(4,8), `l2_leaf_reg`
  LogReal(0.1,10), `min_child_weight` Real(0,8), `subsample` Real(0.5,1),
  `colsample` Real(0.5,1), `leaf_estimation_iterations` Int(1,5), `max_bins`
  Cat(128,254), `linear_leaves` Cat(F,T), `linear_lambda` LogReal(0.1,10),
  `cat_smoothing` LogReal(0.1,10), `cat_n_permutations` Int(1,8),
  `ordered_boosting` Cat(F,T). (`n_estimators` fixed at 1500 + early stopping.)

**3. Model verification.**
TabArena-Lite default = Elo 1211; full TabArena default = Elo 1220 (rank 39/70),
beating default XGBoost (1210) and LightGBM (1184), below CatBoost-default
(1359), at the fastest tree-model train time on the board (0.59 s/1K). I am the
original author of ChimeraBoost.

**4. Maintenance commitment.**
Preferred contact: GitHub @bbstats (issues on bbstats/chimeraboost). Pure-Python
+ numba, no GPU/foundation-model deps → low version-conflict surface.

---

## Step B — Results PR (default entry)

**Contents:**
- (a) Results artifacts for ChimeraBoost (default), produced by the provided
  TabArena code: `run_chimera_full.py` → `run_chimera_full_eval.py`
  (full repetitions, all folds/repeats, 51 tasks, 816 fits, 0 failures).
- (b) Reproducible pipeline: this repo's `benchmarks/tabarena/` (wrapper + run +
  eval scripts) + the model-integration issue from Step A.
- (c) Model description: https://github.com/bbstats/chimeraboost (README +
  `docs/`).
- (d) Attestation (verbatim):
  > I confirm that these results were produced using the attached modeling
  > pipeline and to the best of my knowledge, I have used the test data
  > appropriately and have not manipulated the results.
- (e) Verification requested: **yes** (re-run a sample of folds for the main
  leaderboard).

**Not in this PR (follow-up):** the tuned entry — pending the full 201-config
tuned run on the fixed search space (`run_chimera_tuned.py`, fresh
`chimera_tuned_full` dir).
