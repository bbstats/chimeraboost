# Task: <slug>

Read `CLAUDE.md` first and follow it. Work only on the current git branch.

## Goal
<one paragraph: the change, the mechanism, the expected effect>

## Files you may edit
- chimeraboost/<file>.py
- tests/<file>.py (only if the goal says tests change)
Nothing else. Do not touch benchmarks/, docs/, README, or plan files.

## Steps
1. <concrete edit>
2. Run tests: `A:\code\miniconda3\python.exe -m pytest tests/<file>.py -q`
3. <the ONE benchmark command, if this rung has one, exactly as given>
   `A:\code\miniconda3\python.exe benchmarks\run_benchmarks.py --synth --seeds 3 --save --models ChimeraBoost`

## Done means
- tests above pass
- <numerical-identity expectation: bit-identical / drift allowed and quantified>
- `RESULT.md` written next to this file: what changed, test output tail, the
  results JSON path if a benchmark ran, anything you could not do and why.

## Never
- commit, push, merge, or switch branches
- run a second benchmark or any benchmark not listed above
- edit files outside the list
- use `python` from PATH (it has nothing installed); use the conda path above
