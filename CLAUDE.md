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
- **TabArena is the ONE sealed holdout** (changed 2026-07-27). Report-only: its results — aggregate or per-task — must never influence a source change. The full run is executed by its authors on their defaults, days of turnaround, and is what the README figure shows. Decisions run on synthetic → Grinsztajn + high-card. The `pub:` public suite (`benchmarks/PUBLIC_PLAN.md`) is **no longer sealed** — it is post-hoc validation, read it freely, and it **never blocks a ship**; it may raise a question but Grinsztajn + high-card still answer it. PMLB is the HP-tuning suite only. A published chart must never run on a suite we tune against; that would be in-sample.
- **North star:** strength vs slowdown Pareto (`benchmarks/make_pareto.py`, `/pareto` skill). Headline axis since 2026-07-18 = head-to-head win rate (% of dataset×opponent matchups won, primary metric RMSE reg / Brier clf); blended-% stays the weak-leg diagnostic; ship-gating (sign tests) unchanged. Ship only what pushes the frontier. (Elo is a person's name — never "ELO".)
- **Always print the aggregate results table after every benchmark run**, unprompted.

## Benchmarks
- Decision tier is ONE command: `python benchmarks/run_benchmarks.py --decide --seeds 3 --save` → Grinsztajn + high-card + their variant families, reported and sign-tested **per stratum, never pooled** (`compare_runs.py BASE NEW --by-suite`). Pooling would count a variant's rows twice, since a variant is a derived view of its parent. Variant families: `@sus25`/`@sus50` (25%/50% of training rows, test set unchanged — the small-data regime) and `@time` (temporal splits on audited datasets — distribution shift, the regime every random split is blind to). See `benchmarks/VARIANTS.md`. `--list-datasets` previews a run without downloading. **The OpenML one-shot gate (`--openml`) was RETIRED 2026-07-27** — eight of its 29 datasets were exact-name Grinsztajn members, so it partly re-scored the very data it was meant to check independently; the `oml:` keys survive only as a registry for the research scripts. Post-hoc validation is `--public` (22 audited datasets, zero overlap with anything we tune on), never a blocker. HP tuning: `--pmlb --pmlb-fold tune|holdout` (dormant — the one study on it found broad tuning buys nothing that generalises).
- **Scoring is head-to-head win rate** (RMSE regression / Brier classification, near-solved excluded), with a bootstrap CI and a **median** gap. Never the mean of relative gaps — that is what produced the −144% and −8e21% readings.
- `fit_time` excludes prediction and metric computation; runs are stamped `timing="fit_only"`. Older JSONs are not speed-comparable and the tools warn on a mix.
- Mechanism probe (tier 1): `--synth` = frozen SynthGen prior-sampled suite (`benchmarks/synthgen/`, NO TabArena in any form incl. metadata). Attribute deltas: `benchmarks/synth_report.py BASE NEW`. Any generator change ⇒ VERSION bump + re-freeze + `benchmarks/synthgen/backtest.py` re-validation (gate ≥7/9 vs ledger).
- Sign-test two runs: `python benchmarks/compare_runs.py BASE.json NEW.json [--model ChimeraBoost]`.
- Progress / latest table: `python benchmarks/bench_status.py` (the `/bench` skill).
- **One benchmark at a time** — never two concurrently (core contention corrupts timings).
- Full protocol for shipping a change: `/experiment` skill.

## Close the loop — no loose ends
A record must never outlive the thread it describes. Close it out in the SAME
change that resolves the thread, not in a later sweep.
- **Plan files**: when a program ends, the commit that records the verdict also
  clears that file's open items. No `*_PLAN.md` may sit saying "pending",
  "deferred", "not yet run", or "Nathan's call" once the answer is known — mark
  it resolved, killed, or shipped, with the date.
- **Branches**: delete a branch when it merges. After a stacked-PR merge,
  confirm the commits actually reached `main` (PR #56 was silently orphaned
  that way and had to be recovered as PR #60).
- **Prose follows the code**: replacing a chart, flag, or default means
  grepping README, `docs/`, and image alt text for descriptions of the old one
  in the same change. A chart axis renamed in code and not in prose is a lie
  with a long half-life.
- **Orphans**: delete a file in the change that stops referencing it.
- **"Later"**: gets a GitHub issue or a dated plan-file line. Never an unowned
  comment or an unwritten intention.

## This is a Windows box — write Windows commands
Windows 11, and the primary shell is **Windows PowerShell 5.1** (`powershell.exe`), not
bash. Commands written for Linux fail here, usually with a confusing parser error
rather than a clear one. Rules:
- **No `&&` and no `||`.** They are parse errors in PowerShell 5.1. To run B only if
  A succeeded, write `A; if ($?) { B }`. To run both regardless, write `A; B`.
- No `?:` ternary, no `??`, no `?.` — all PowerShell 7 syntax that 5.1 rejects.
- No Unix-isms: use `Get-Content -TotalCount N` for `head`, `-Tail N` for `tail`,
  `2>$null` for `2>/dev/null`, `Remove-Item -Recurse -Force` for `rm -rf`,
  `New-Item -ItemType Directory -Force` for `mkdir -p`, `$env:VAR = 'x'; cmd` for
  `VAR=x cmd` (PowerShell has no inline environment-variable prefix).
- Paths are `C:\Users\Nathan\...` and `A:\code\...`. Quote any path with spaces.
- Multi-line strings (commit messages) go in a single-quoted here-string `@'...'@`
  with the closing `'@` at column 0.
- A Git Bash tool is also available and *does* accept `&&`, forward slashes and
  `/dev/null` — but it is a separate shell with its own syntax. Pick one per
  command and stay inside its rules; never mix the two dialects in one line.
- Anything written to a file for later shell consumption picks up `\r\n` line
  endings — see the `tr -d '\r'` quirk below.

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
