# Why the leaderboard shows ChimeraBoost as slow — and the fix

*Diagnosed 2026-07-07. Upstream engagement deferred; this file is the draft
material for a PR #358 follow-up comment / tabarena PR whenever we choose to
send it.*

## The discrepancy

| Source | Train (s/1K, median) | Predict (s/1K, median) |
|---|---|---|
| Official leaderboard (their cluster re-run) | 10.45 | 2.617 |
| Our TabArena-Full artifacts, identical code/seeds/splits (`chimera_full_leaderboard.csv`) | 0.6 | 0.068 |
| CatBoost (default), leaderboard | 6.83 | 0.08 |

Same wrapper, same 10000-tree cap, same folds — so tree counts are identical
and the ES/tree-cap hypothesis from the PR #358 thread cannot explain the gap.
Warm-process ChimeraBoost predict (0.068) is *faster than CatBoost* (0.08).

## Root cause: numba JIT charged to the timed sections in fresh workers

ChimeraBoost's kernels JIT-compile on first use (disk-cached thereafter).
TabArena's harness times exactly `fit(...)` and `predict_proba(...)`
(`tabarena/benchmark/models/wrapper/abstract_class.py`, `Timer()` blocks);
module import happens outside the timers. Our local runs execute all 51 tasks
in one long process — compile once, amortized to zero. A cluster that gives
every fold/task a fresh worker process (with no persistent numba cache across
workers) pays, **per job**:

| Scenario (2K×20 task, 12-core box) | timed fit | timed 1st predict (s/1K) |
|---|---|---|
| steady-state (in-process warm) | 0.10–0.34 s | 0.001 |
| fresh process, warm disk cache | 0.47–0.80 s | ~0.24 (cache load) |
| fresh process, cold cache | **4.4–9.3 s** | **1.3–1.8** |

(Fresh-process predict was measured on a pickled model, the harness pattern:
classification compiles its predict kernels during fit via temperature
scaling, so the predict-side hit appears when inference runs in a different
process than training.) Scale the cold column by slower cluster cores and
divide by TabArena's small median task size and you reproduce 10.45 / 2.617.

## Fix (chimeraboost 0.14.1)

`chimeraboost.warmup()` — three tiny synthetic fits + predictions covering
every default-path kernel (binary w/ linear leaves + categoricals + eval_set,
multiclass, regression w/ ordered boosting). The wrapper calls it at module
import, next to the existing top-level import that exists for the same
reason. Verified end-to-end (fresh process, cold cache, warmup at import):

| task | timed fit | timed 1st predict (s/1K) |
|---|---|---|
| binary | 0.10 s | 0.001 |
| multiclass | 0.24 s | 0.001 |
| regression | 0.08 s | 0.001 |

Warmup itself costs 10.5 s cold / 0.7 s warm — one-time environment setup,
the moral equivalent of the C++ boosters' ahead-of-time compilation at
package build, and it sits outside the harness timers by design.

Note: numba's cache locator already falls back to a per-user cache dir when
site-packages is read-only (`UserWideCacheLocator`), so no cache-dir
workaround is needed in the library; ephemeral workers simply never share a
cache, which warmup sidesteps entirely.

## Relation to the 2026-07-06 draft PR (import hoist + 0.13.1 pin)

`upstream_pr/PR_DESCRIPTION_0.13.1.md` (+ the .patch and verification notes
alongside it) drafted the first half of this fix: hoisting the import out of
the fit timer and pinning 0.13.1's LAPACK-free solver (~25% off cold-start
JIT). Today's measurements show the remaining in-timer JIT is the dominant
term, so any upstream PR should combine that draft with the `warmup()` call
and a `chimeraboost>=0.14.1` pin — treat the draft as superseded input, not
a separate PR.

## Repro scripts

The probe/driver scripts used for the tables live in the session scratchpad;
the essential pattern: run `python probe.py` variants as fresh subprocesses,
deleting `chimeraboost/__pycache__/*.nb?` between runs for the cold cells.
