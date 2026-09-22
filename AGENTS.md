# AGENTS.md — instructions for Muse Code

Read `CLAUDE.md` first and follow it. It is the project's real instruction
file; this one only adds what a headless worker needs.

## Your role
You are the worker in a two-agent loop. Claude Code plans, reviews, and gates.
You implement one task file from `campaign_tasks/` and nothing else. Anything
the task file does not list, do not do.

## Environment
- Windows 11, PowerShell 5.1. No `&&`, no `||`. Use `A; if ($?) { B }`.
- The `python` on PATH is a bare install with nothing in it. Always use
  `A:\code\miniconda3\python.exe`.
- Run script files, not `python -c "..."`; quoting breaks in this shell.
- If many tests fail at once after a source edit, delete
  `chimeraboost\__pycache__` and rerun; the numba cache corrupts on mass edits.

## Never
- `git commit`, `git push`, `git merge`, or check out any other branch.
- Start a benchmark the task file does not name, or a second one while another
  is running (`benchmarks\bench_status.py` shows what is in flight).
- Edit files outside the task file's list. `docs/`, README, and any
  `*_PLAN.md` are Claude's, not yours. `benchmarks/` is yours whenever the
  task file lists the file: probe scripts, harness tools and their tests.
  `benchmarks/tabarena/` and its results stay sealed regardless.
- Touch anything under `benchmarks/tabarena/` or its results. That suite is a
  sealed holdout.
- Change a default hyperparameter. Defaults change only through the gated
  experiment protocol that Claude runs.

## When done
Write `RESULT.md` next to the task file: what you changed, the test output
tail, the results JSON path if a benchmark ran, and anything you could not do
and why. Then stop.
