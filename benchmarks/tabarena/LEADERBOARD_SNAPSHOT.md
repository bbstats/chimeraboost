# TabArena leaderboard snapshot — 2026-09-01

Report-only record of the official leaderboard, fetched from the Space's data file:
`huggingface.co/spaces/TabArena/leaderboard` →
`data/entrants_models/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv`.
The Space's data was last updated with "ChimeraBoost 0.30.0" (~2026-08-10). 69 rows on
the board; the top entries are tabular foundation models (Elo up to ~1790).

This supersedes the 2026-07-23 read (Elo 1278, rank 31/68) recorded in
`docs/PROJECT_STATUS.md`. Per the vow: this file is a citation for published prose
(README, whitepaper) and must never influence a source change.

## ChimeraBoost rows

| config | Elo | 95% CI | board row | train s/1K | predict s/1K |
|---|---|---|---|---|---|
| default | 1315 | +42/−62 | 31 | 2.65 | 0.052 |
| tuned | 1352 | +46/−55 | 26 | 518.43 | 0.050 |
| tuned + ensembled | 1383 | +46/−59 | 21 | 518.43 | 0.469 |

## GBDT-family comparison (default configs)

| model | Elo | train s/1K | predict s/1K |
|---|---|---|---|
| CatBoost (default) | 1379 | 5.88 | 0.025 |
| **ChimeraBoost (default)** | **1315** | **2.65** | **0.052** |
| XGBoost (default) | 1214 | 1.94 | 0.123 |
| EBM (default) | 1204 | 6.67 | 0.014 |
| LightGBM (default) | 1187 | 1.96 | 0.142 |

→ **#2 GBDT on defaults, behind CatBoost, 101 Elo clear of third.** ChimeraBoost's
tuned+ensembled row (1383) edges CatBoost's default (1379); note the tuned rows run the
stale June `hpo.py` search space (see `UPSTREAM.md`).

Other tuned GBDT rows for context: LightGBM tuned+ens 1435, CatBoost tuned+ens 1423,
CatBoost tuned 1410, LightGBM tuned 1387, XGBoost tuned+ens 1377, XGBoost tuned 1351.

Timing caveat: leaderboard times are the maintainers' cluster and include their
pipeline; compare only within the board, never against our local `fit_time` numbers
(`TIMING_FINDINGS.md`).
