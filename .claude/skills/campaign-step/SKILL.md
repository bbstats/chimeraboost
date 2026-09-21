---
name: campaign-step
description: One rung of the campaign loop — plan the next idea, hand the code work to Muse Code (headless), review and gate the result, record the verdict, open a PR or revert. Designed to run under /loop.
---

You are the planner and reviewer. Muse Code (headless, launched by you in
step 4) is the only thing that edits library source in this loop. You never
edit library source yourself, and you never judge a rung before step 5.

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
Launch muse yourself, from the main session (a subagent spawn is blocked by
the permission classifier here). The launch is a single PowerShell command
that begins with `muse exec` and contains nothing else — no `cd`, no
`Start-Process`, no variables, no second statement; that exact prefix is the
pre-authorized form and anything wrapped around it gets blocked.

1. `git rev-parse --short HEAD; git status --porcelain` — record both.
2. Touch `<task>.running`.
3. Run, with the tool's own run_in_background option and a 90-minute timeout:
   ```
   muse exec --prompt-file <task> --trust-workspace --disable-approval --user-input-auto-resolve --disable-web-tools --max-model-steps 200 --json *> <task>.log
   ```
   `--trust-workspace` is what makes muse read `AGENTS.md`. Never pass
   `--yolo` or `--disable-sandbox`; the sandbox stays on. Muse's sandbox runs
   as a different Windows user, so it will report a "dubious ownership" git
   quirk and use a one-shot `-c safe.directory=` flag; that is expected.
4. Wait for the task notification. Remove `<task>.running`. Record the exit
   code (0 done, 1 failed or step cap, 2 usage error). Still running after
   90 minutes: kill it, remove the marker, log "timed out", end the step.
5. Collect: `git status --porcelain`, `git diff --stat`, the `RESULT.md` the
   task told muse to write, the last 40 lines of `<task>.log`, and the test
   command the task names run with `A:\code\miniconda3\python.exe`.

## 5. Review and gate
- Read what you collected in step 4. Run `/code-review` on `git diff main...HEAD`.
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
