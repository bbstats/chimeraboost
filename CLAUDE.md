# ChimeraBoost — working notes for Claude

Pure-Python oblivious-tree GBDT (numpy + numba + sklearn only).

## Hard constraints
- **Pure Python, no heavy deps.** No torch/onnx/foundation-model anything. Filter every idea through this first.
- **TabArena (Lite and Full) is a SEALED HOLDOUT.** Report-only. Its results — aggregate or per-task — must never influence a source change. Decisions run on synthetic → dev panel → Grinsztajn, gated by an independent OpenML one-shot. PMLB is the HP-tuning suite only.
- **North star:** blended strength vs slowdown Pareto (`benchmarks/make_pareto.py`, `/pareto` skill). Ship only what pushes the frontier. (Elo is a person's name — never "ELO".)
- **Always print the aggregate results table after every benchmark run**, unprompted.

## Benchmarks
- Decision suite: `python benchmarks/run_benchmarks.py --grinsztajn --seeds 3 --save` → `benchmarks/results/<stamp>.json` (gitignored). Independent gate: `--openml`. HP tuning: `--pmlb --pmlb-fold tune|holdout`.
- Mechanism probe (tier 1): `--synth` = frozen SynthGen prior-sampled suite (`benchmarks/synthgen/`, NO TabArena in any form incl. metadata). Attribute deltas: `benchmarks/synth_report.py BASE NEW`. Any generator change ⇒ VERSION bump + re-freeze + `benchmarks/synthgen/backtest.py` re-validation (gate ≥7/9 vs ledger).
- Sign-test two runs: `python benchmarks/compare_runs.py BASE.json NEW.json [--model ChimeraBoost]`.
- Progress / latest table: `python benchmarks/bench_status.py` (the `/bench` skill).
- **One benchmark at a time** — never two concurrently (core contention corrupts timings).
- Full protocol for shipping a change: `/experiment` skill.

## Machine quirks (learned the hard way — trust these)
- Run script **files**, not `python -c "..."` (quoting breaks on this box).
- Terminal stdout garbles under batched tool calls; trust file-based reads over what scrolled by.
- A/B trap: with `pip install -e .`, any `python script.py` resolves chimeraboost to **this repo** regardless of CWD. For worktree A/Bs set `PYTHONPATH=<worktree>` and print `chimeraboost.__file__` in the run.
- Writing a keys/datasets file from Python on Windows gets `\r\n` → `--datasets` silently matches nothing; `tr -d '\r'` first.
- C: has ~4 GB free. Big envs/caches/outputs go to `A:\code` (see `/tabarena` skill for the env-var block).
- HuggingFace 401-rate-limits anonymous bursts; Grinsztajn CSVs are cached in `benchmarks/data_cache/` (auto). Flaky network → just relaunch.
- `gh` is unauthenticated here; get a token via `git credential fill` → `GH_TOKEN` (see `/release` skill).
- Watch CHANGELOG section headers in merge conflicts — a merge once clobbered a version header.

## Layout
- `chimeraboost/` library · `tests/` (395+, incl. numerical-identity goldens — bit-identical refactors must keep them green)
- `benchmarks/` harness + analysis scripts · `benchmarks/tabarena/` sealed-holdout runners · `benchmarks/research/` cascade engine
- `images/` committed charts (pareto.png is the README headline — refresh after shipping)
- `docs/` user docs — keep terse, no slop, no tuning-priority claims (defaults are Grinsztajn-tuned)

## Skills
`/bench` progress+table · `/experiment` A/B gate protocol · `/pareto` refresh headline chart · `/tabarena` holdout run recipe · `/release` cut a release
