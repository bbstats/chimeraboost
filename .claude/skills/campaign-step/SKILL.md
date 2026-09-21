---
name: campaign-step
description: One rung of the campaign loop — plan the next idea, hand the code work to the muse-worker, review and gate the result, record the verdict, open a PR or revert. Designed to run under /loop.
---

You are the planner and reviewer. Muse Code (via the `muse-worker` agent)
is the only thing that edits library source in this loop.

## 0. Is it safe to start?
- `A:\code\miniconda3\python.exe benchmarks\bench_status.py` — a run in flight
  means do nothing this step; schedule the next wake for 20 minutes and stop.
- `campaign_tasks/*.running` present — same: a worker is live, stop.
- One benchmark at a time is the standing rule; the worker runs benchmarks, so
  you never start one while a task is out.

## 1. Resume
Run steps 1–5 of the RESUME protocol in `benchmarks/CAMPAIGN_PLAN.md`
(read the file, confirm branch/sha, score any PENDING entry or unlogged
results JSON first). Never relaunch before scoring.

## 2. Plan the rung (you, not muse)
Take the top ACTIVE family's `next:` line. Do the S0 work yourself:
`python benchmarks/barrier_check.py "<idea>"`, then write the forecast on both
Pareto axes into the plan file's log entry before anything runs.
Then `git checkout -b campaign/<slug> main`.

## 3. Write the task file
Copy `campaign_tasks/TEMPLATE.md` to `campaign_tasks/<yyyymmdd>-<slug>.md`
and fill every section. The file must name: the exact source files to touch,
the exact benchmark or test command (one, with the conda python path), what
"done" looks like, and that muse writes a `RESULT.md` beside the task file.
Anything not in the file, muse must not do — say so in the file.

## 4. Hand it off
Spawn `muse-worker` with the task path, `run_in_background: false`. Wait.
A "busy" or "timed out" report means log it and end the step.

## 5. Review and gate
- Read the worker report. Run `/code-review` on `git diff main...HEAD`.
- Run the full tests with the conda python; goldens must stay green on a
  bit-identical change.
- If the rung produced a results JSON: score it with `compare_runs.py`
  (`--by-suite` for `--decide` runs) or `synth_report.py`, and **print the
  aggregate table**. Read `benchmarks/GATE_ROBUSTNESS.md` before any number
  decides anything.

## 6. Record and close
- Write the verdict into the plan file's log entry (hit/miss vs the forecast,
  both axes, next rung or KILLED). No PENDING left behind.
- Pass: commit on the branch, push, open a PR with `gh` (token via
  `git credential fill`). PRs only — never merge, never release.
- Fail: `git checkout -- .; git clean -fd chimeraboost tests` and
  delete the branch. The plan-file entry is the record.
- `git checkout main`. Delete the task's `.log` and `RESULT.md` only after the
  verdict cites what they said.

End the step. Under `/loop`, schedule the next wake for 5 minutes if a rung
is queued, 30 minutes if the family list is empty.
