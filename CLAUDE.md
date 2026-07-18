# ChimeraBoost — working notes for Claude

Pure-Python oblivious-tree GBDT (numpy + numba + sklearn only).

## Speak English in chat (Nathan's standing complaint — honor it)
- Benchmark-speak belongs in files and tables, not in replies. Lead every
  report with the plain-English takeaway ("depth-4 stopped beating the suite —
  it now loses slightly on average"), then the numbers as support.
- Unpack project shorthand on first use in a message. Not "canary slice
  +0.000%@3 clean" — say "the three canary datasets, where any win would mean
  the suite rewards overfitting, came out exactly flat: what we want."
- No stat fragments as sentences ("W61-L57 mean −0.113%"), no arrow chains,
  no @-counts without saying what's counted. Numbers ride inside sentences
  with their referents, or live in a table whose meaning the prose states.
- Self-check before sending: would this sentence survive being read aloud to
  someone who didn't watch the run? If not, rewrite it.
- Docs, verdict files, and memory stay terse — this rule is about talking to
  a human.

## Hard constraints
- **Pure Python, no heavy deps.** No torch/onnx/foundation-model anything. Filter every idea through this first.
- **TabArena (Lite and Full) is a SEALED HOLDOUT.** Report-only. Its results — aggregate or per-task — must never influence a source change. Decisions run on synthetic → dev panel → Grinsztajn, gated by an independent OpenML one-shot. PMLB is the HP-tuning suite only.
- **North star:** strength vs slowdown Pareto (`benchmarks/make_pareto.py`, `/pareto` skill). Headline axis since 2026-07-18 = head-to-head win rate (% of dataset×opponent matchups won, primary metric RMSE reg / Brier clf); blended-% stays the weak-leg diagnostic; ship-gating (sign tests) unchanged. Ship only what pushes the frontier. (Elo is a person's name — never "ELO".)
- **Always print the aggregate results table after every benchmark run**, unprompted.

## Benchmarks
- Decision suites (run both, sign-test separately): `--grinsztajn` (low/no-card, no multiclass) and `--highcard` (real high-cardinality cats + multiclass — the regime Grinsztajn is blind to; see `benchmarks/HIGHCARD_PLAN.md`, `benchmarks/hc_gap.py`). `python benchmarks/run_benchmarks.py --grinsztajn --seeds 3 --save` → `benchmarks/results/<stamp>.json` (gitignored). Independent gate: `--openml`. HP tuning: `--pmlb --pmlb-fold tune|holdout`.
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

## Output language (final step, non-negotiable)
You may think, draft, or reason internally in whatever form is most efficient for you — including shorthand, symbols, or non-English fragments. That is fine and expected.

However, the **final response shown to the user must be complete, fluent, natural English** — no shorthand, no untranslated fragments, no mixed-language output.

Before emitting your final answer, do a silent self-check: "Is every word of this response fluent, grammatical English a non-technical reader could parse?" If not, rewrite it in English before sending. This check applies to the very last thing you output, not to intermediate reasoning. Applies regardless of which model is running this session.
