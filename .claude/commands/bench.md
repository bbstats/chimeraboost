---
description: Report current benchmark run progress, or the most recent results table
---

Run `python benchmarks/bench_status.py` from the repo root and show the user its
output verbatim in a code block.

- If a run is in progress, the output is a live progress block (percent, tasks
  done, ETA). Relay it as-is; add one short sentence on what experiment is
  running and what comes next in the autonomous plan.
- If no run is in progress, the output is the latest completed results table.
  Relay it, and if you are mid-experiment, remind the user which experiment that
  result belongs to and what is running/next.

Keep it phone-friendly: the table or progress block first, then at most two
lines of commentary. Do not re-run the benchmark.
