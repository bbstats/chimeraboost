# TabArena-Lite integration

Scripts to run ChimeraBoost on [TabArena](https://tabarena.ai) (Lite protocol)
and produce an Elo leaderboard entry. **TabArena is a sealed holdout: results
are report-only and must never feed back into source or default choices.**

## Files

| File | Purpose |
|---|---|
| `chimeraboost_tabarena_model.py` | AutoGluon `AbstractModel` wrapper (native `cat_features` passed through) + config generators. Must stay a separate file — TabArena pickles the class. |
| `run_chimera_lite.py` / `run_chimera_eval.py` | Default-config entry (1 config, 51 tasks) + Elo eval. |
| `run_chimera_e10.py` / `run_chimera_e10_eval.py` | `n_ensembles=10` variant as a separate entry. |
| `run_chimera_tuned_lite.py` / `run_chimera_tuned_lite_eval.py` | Tuned entry: default + 200 random configs (TabArena convention). Search space = core knobs + the default-off research flags (see `benchmarks/research/SUMMARY.md`). |
| `run_chimera_tuned.py` / `run_chimera_tuned_eval.py` | Full-repetitions variant of the tuned run (all folds/repeats). ~30x Lite cost; unused so far. |

## Environment (Windows box; keep everything off C:!)

Venv: `A:\code\tabarena\.venv\Scripts\python.exe` (Python 3.11, editable
installs of tabarena + bencheval + this repo). Before running, set:

```powershell
$env:TMP='A:\code\tmp'; $env:TEMP='A:\code\tmp'
$env:HF_HOME='A:\code\hf'; $env:PYTHONIOENCODING='utf-8'
```

OpenML caches to `A:\code\openml` (set in-script). Outputs land in
`A:\code\tabarena_out\`; leaderboards in `A:\code\tabarena_out\evals\`.

Run from this directory (the wrapper module is imported from cwd):

```powershell
& A:\code\tabarena\.venv\Scripts\python.exe run_chimera_tuned_lite.py --limit 2   # smoke
& A:\code\tabarena\.venv\Scripts\python.exe run_chimera_tuned_lite.py             # full
& A:\code\tabarena\.venv\Scripts\python.exe run_chimera_tuned_lite_eval.py        # Elo
```

## Gotchas

* **Result caching**: a re-run silently loads cached results
  (`Loading cache exists=True`) instead of refitting. To force fresh fits,
  delete `A:\code\tabarena_out\<run>\data\<Method>_*` and the processed cache
  `~/.cache/tabarena/artifacts/<Method>` before evaluating.
* **OpenML outages**: pre-cache all task files and verify they exist on disk
  (the API can "succeed" without writing); see the precache scripts in
  `A:\code\tabarena\examples\benchmarking\custom_tabarena_model\`.
* Windows: `os.sched_getaffinity` shim + UTF-8 output are handled in the eval
  scripts.
