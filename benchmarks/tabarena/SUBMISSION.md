# TabArena submission — ChimeraBoost (default entry)

How models are *actually* added (per the OrionMSP PR #303 to autogluon/tabarena):
a single **`[New Model]` pull request** to a fork of `github.com/autogluon/tabarena`
that (1) adds the model package, (2) carries a concise description + confirmation
statement, and (3) posts results in the PR. A maintainer reviews and verifies.
We submit the **default** row first; the tuned row follows the full-tuned run.

## Technical gates (all green as of 2026-06-15)
- ✅ Public `AbstractModel` impl — library https://github.com/bbstats/chimeraboost (`pip install chimeraboost`); wrapper [`chimeraboost_tabarena_model.py`](chimeraboost_tabarena_model.py).
- ✅ Passes the default model unit test — [`model_unittest.py`](model_unittest.py) → 3/3 (binary/multiclass/regression) via AutoGluon `FitHelper`.
- ✅ Default hyperparameters + HPO search space specified (in the package below).
- ✅ Results: TabArena-Lite Elo 1211; full TabArena Elo 1220 (rank 39/70), 0 failures.

## The PR contents
1. **Model package** — `upstream_pr/chimeraboost/` (`model.py`, `hpo.py`, `info.py`, `__init__.py`), copied to `tabarena/tabarena/models/chimeraboost/` in your fork. Plus the public-class registration. See [`upstream_pr/REGISTER.md`](upstream_pr/REGISTER.md).
2. **Results** — full-default leaderboard (Elo 1220) posted as a PR comment + a link to the raw results zip (host the `chimera_full` artifacts somewhere public, e.g. a release asset).

---

## PR description (paste into the PR body)

**Title:** `[New Model] ChimeraBoost`

ChimeraBoost is a pure-Python, CatBoost-inspired oblivious gradient-boosting
library (numba-backed; no GPU / heavy deps). It supports binary, multiclass,
regression, and quantile regression. This PR adds it to TabArena as a CPU config
model (default entry; a tuned entry will follow).

**Notes**
- Standalone model (not an ensembling pipeline). Native categorical handling via
  ordered target statistics; categoricals are passed through AutoGluon's
  `LabelEncoderFeatureGenerator` then marked as `cat_features`.
- Passes the default model unit test (`FitHelper.fit_and_validate_dataset`) on
  toy binary / multiclass / regression.
- Default config: `n_estimators=500`, `early_stopping=True`, `thread_count=-1`,
  `random_state=0`; all other knobs at library defaults.

**Misc**
- Codebase: https://github.com/bbstats/chimeraboost
- PyPI: https://pypi.org/project/chimeraboost
- Docs: https://bbstats.github.io/chimeraboost
- Inspirations: CatBoost (ordered TS, oblivious trees), XGBoost, LightGBM (see repo README).

By submitting this pull request, I confirm that you can use, modify, copy, and
redistribute this contribution, under the terms of your choice.

---

## Results to post as a PR comment

Full TabArena (all folds/repeats), default config, produced by the provided
TabArena code (`run_chimera_full.py` → `run_chimera_full_eval.py`):

| Model (default) | Elo | Train s/1K |
|---|---|---|
| CatBoost (default) | 1359 | 6.83 |
| **ChimeraBoost (default)** | **1220** (+47/−58) | **0.59** |
| XGBoost (default) | 1210 | 1.94 |
| LightGBM (default) | 1184 | 1.96 |

51 tasks, 816 fits, 0 failures. ChimeraBoost beats default XGBoost and LightGBM,
trails CatBoost, at the fastest tree-model train time on the board. Raw results:
`<link to chimera_full artifacts zip>`. Request: please verify (re-run a sample
of folds) for the main leaderboard.

**Follow-up (not in this PR):** the tuned entry — pending the full 201-config
tuned run on the fixed search space (`run_chimera_tuned.py`, fresh
`chimera_tuned_full` dir; ~2–3 weeks of pausable compute).
