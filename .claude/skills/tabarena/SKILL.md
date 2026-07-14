---
name: tabarena
description: Re-run and evaluate ChimeraBoost on TabArena (sealed holdout, report-only) — env setup, run/eval recipe, cache gotchas, Windows fixes
---

**THE VOW: TabArena is a sealed holdout.** Report-only — no result from it (aggregate or per-task)
may influence a source change. Report the aggregate Elo + rank only, never per-task breakdowns.
Compare default-vs-default (tuned/ensembled entries are a different weight class).

## Environment (everything lives off C: — only ~4 GB free)
Set in every shell before any uv/python:
```powershell
$env:UV_CACHE_DIR='A:\code\uv_cache'; $env:UV_PYTHON_INSTALL_DIR='A:\code\uv_python'
$env:TMP='A:\code\tmp'; $env:TEMP='A:\code\tmp'; $env:HF_HOME='A:\code\hf'
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'
```
Venv: `A:\code\tabarena\.venv` (py3.11; AutoGluon needs ≤3.12). Verify it's alive before reinstalling:
`A:\code\tabarena\.venv\Scripts\python.exe -c "import tabarena, bencheval, autogluon, chimeraboost"`.
If broken, the install recipe is in memory `project_tabarena_elo.md` (`uv sync` FAILS — use the
uv-pip editable-install block there). chimeraboost is installed editable → source changes auto-apply.

## Run + eval
Canonical scripts live in repo `benchmarks/tabarena/`, synced to and **run from**
`A:\code\tabarena\examples\benchmarking\custom_tabarena_model\` (wrapper imported by filename)
with the venv python:
- Lite: `run_chimera_lite.py` (`--limit N` = smoke test first) → `run_chimera_eval.py`
- Full (the public leaderboard protocol): `run_chimera_full.py` → `run_chimera_full_eval.py`
- Tuned: `run_chimera_tuned.py` (+ its eval) — full-tuned is ~2–3 weeks on this box
Outputs: `A:\code\tabarena_out\...`; leaderboards under `A:\code\tabarena_out\evals\`.

## Gotchas (each cost real time once)
- **Precache gotcha**: a run silently LOADS cached results ("Loading cache exists=True") instead of
  refitting. To force fresh fits, delete `A:\code\tabarena_out\<dir>\data\ChimeraBoost*_BAG_L1` first.
- **Before every eval**: clear the processed cache `~/.cache/tabarena/artifacts/ChimeraBoost*`
  (lite/full/tuned share method names). First eval also downloads the ~30-method baseline pool (slow once).
- **Eval is single-threaded and I/O-bound** (`to_results()` re-scores every prediction pickle serially)
  — launch it unbuffered in the background and walk away; don't kill it because it looks stuck.
- **OpenML flakiness**: download calls can "succeed" without writing files. Success criterion is the
  file on disk, never the API return. Precache/audit scripts live in the A: example dir; gate a run
  on all 5 files per task existing (datasplits.arff, dataset .pq, description/qualities/features.xml).
- **Windows**: shim `os.sched_getaffinity` if a script lacks it; write md/csv with `encoding="utf-8"`.
- Runs resume from per-(config, task, fold) cache — to pause, kill the process and relaunch the same
  script later.
- Preserve the old raw run dir (rename aside) before a re-run, for before/after leaderboard diffs.

Current standings + PR #358 state: memory `project_tabarena_elo.md`.
