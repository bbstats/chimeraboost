# ChimeraBoost: bump to 0.13.1 + hoist import out of the fit timer

Follow-up to #358 (ChimeraBoost model, now merged). Two small changes to the
`chimeraboost` model package.

## What changed

**1. `model.py` — hoist the `chimeraboost` import to module top.**
`_fit` used to lazy-import `ChimeraBoostClassifier` / `ChimeraBoostRegressor`
on the first call. Because the import fires *inside* `_fit`, it charged
`chimeraboost`'s import cost (numba + numpy module load, ~1.4s on my dev
hardware) to every job's reported train time. Moving it to module top runs it
before the benchmark's fit timer starts. The upstream AutoGluon template
lazy-imports so its registry works without optional deps installed — that
concern doesn't apply to a dedicated model file that already hard-depends on
`chimeraboost` via `pip_extra`.

**2. `info.py` — bump the pin `chimeraboost>=0.13.0` → `>=0.13.1`.**
0.13.1 replaces the single `np.linalg.solve` in the linear-leaf fit kernel
with a hand-rolled LU solver, so the first `fit()` in a fresh worker no longer
JIT-compiles numba's LAPACK bindings. On ephemeral TabArena workers (no
persistent numba on-disk cache) that JIT lands in the reported train time, so
this is a measurable cold-start win — ~25% off first-fit JIT on dev hardware —
with predictions unchanged except at the ~1e-15 solver-elimination-order level
(tree structures identical). Release notes:
https://github.com/bbstats/chimeraboost/blob/main/CHANGELOG.md

## Not changed

No behavior, config, search-space, or metadata changes. The `MethodMetadata`
fields the maintainers set on merge (`verified`, `cache_type`/`cache_kwargs`,
`suite`) are untouched.

## Verification (dev hardware)

- Fresh-cache compile profile of a first fit: **zero `linalg.*` numba compiles**
  post-0.13.1 (`tree._solve_small` compiles in ~0.36s); pre-0.13.0 showed
  `linalg.solve_impl` ~2.4s + `linalg.oneD_impl` ~1.7s + the LAPACK helper
  cluster.
- `import chimeraboost` in a fresh process: ~1.4s — the cost the lazy import
  was billing to train time.
