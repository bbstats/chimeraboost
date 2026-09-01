# TabArena leaderboard snapshot — 2026-09-01

Report-only record of the official leaderboard, fetched from the Space's data files:
`huggingface.co/spaces/TabArena/leaderboard` →
`data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv`.
This is the **site's default view** (`imputation: yes` — models with missing task
results included via imputed scores; 79 rows). The Space's data was last updated with
"ChimeraBoost 0.30.0" (~2026-08-10). Top of the board is tabular foundation models
(EXAONE-Tabular default, Elo 1755).

This supersedes the 2026-07-23 read (Elo 1278, rank 31/68) recorded in
`docs/PROJECT_STATUS.md`. Per the vow: this file is a citation for published prose
(README, whitepaper) and must never influence a source change.

## ChimeraBoost rows

| config | Elo | 95% CI | board row | train s/1K | predict s/1K |
|---|---|---|---|---|---|
| default | 1297 | +42/−57 | 34 | 2.65 | 0.052 |
| tuned | 1333 | +47/−52 | 29 | 518.43 | 0.050 |
| tuned + ensembled | 1360 | +45/−55 | 21 | 518.43 | 0.469 |

All ChimeraBoost rows: Imputed 0.0% (no missing task results).

## GBDT-family comparison (default configs)

| model | Elo | train s/1K | predict s/1K |
|---|---|---|---|
| CatBoost (default) | 1357 | 5.88 | 0.025 |
| **ChimeraBoost (default)** | **1297** | **2.65** | **0.052** |
| XGBoost (default) | 1208 | 1.94 | 0.123 |
| EBM (default) | 1191 | 6.67 | 0.014 |
| LightGBM (default) | 1181 | 1.96 | 0.142 |

→ **#2 GBDT on defaults, behind CatBoost; the 89-Elo gap down to XGBoost is bigger
than the 60-Elo gap up to CatBoost.** ChimeraBoost's tuned+ensembled row (1360) edges
CatBoost's default (1357); note the tuned rows run the stale June `hpo.py` search
space (see `UPSTREAM.md`).

Other tuned GBDT rows for context: LightGBM tuned+ens 1410, CatBoost tuned+ens 1398,
CatBoost tuned 1386, LightGBM tuned 1366, XGBoost tuned+ens 1357, XGBoost tuned 1334.

## The imputation_no slice (same date, for reference)

The sibling `imputation_no` slice drops the ten entrants that needed imputed scores
(69 rows); with the smaller pool every Elo reads higher, ordering unchanged:
ChimeraBoost default 1315 / tuned 1352 / tuned+ens 1383; CatBoost default 1379,
XGBoost default 1214, EBM 1204, LightGBM default 1187. Timing columns are identical
across slices. Quote the imputation_yes numbers: they are what the site shows by
default.

Timing caveat: leaderboard times are the maintainers' cluster and include their
pipeline; compare only within the board, never against our local `fit_time` numbers
(`TIMING_FINDINGS.md`).
