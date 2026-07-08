# Draft: follow-up comment for PR #358 (or a new issue) — ready to paste

Status: DRAFT, not posted. Update the 0.14.1 link once the release is on PyPI.

---

Hi @LennartPurucker — following up on ChimeraBoost's train/predict times on the
leaderboard (10.45 / 2.617 s per 1K samples), which I promised to investigate.

I believe it's a measurement artifact of fresh-worker execution rather than the
model. Re-running the identical config/seeds/splits in a single long-lived
process gives **~0.6 s/1K train and ~0.068 s/1K predict** (predict faster than
CatBoost's 0.08). Since the splits are identical, the tree counts are too — so
it isn't the early-stopping / tree-cap behavior we discussed.

Root cause: ChimeraBoost's hot loops are numba-JIT'd — compiled on first use,
disk-cached thereafter. A process that fits all tasks in sequence pays the
compile once and it amortizes away; a harness that gives each fold a fresh
worker (no shared numba cache) pays it **inside the `Timer()` blocks** every
time. Measured on my machine:

- cold-cache fit: +4.4–9.3 s of compile on a 2K-row task;
- first predict in a fresh process (pickled model): 1.3–1.8 s/1K
  (~0.24 even with a warm disk cache — loading the cache isn't free either).

Scale by your cluster's cores and divide by TabArena's small median task and
that reproduces the leaderboard numbers.

Fix: chimeraboost 0.14.1 adds `chimeraboost.warmup()` — a few tiny synthetic
fits that compile every default-path kernel, bit-identical predictions. Called
once outside the timed sections, timed fit drops 9.3 → 0.10 s and timed first
predict 1.8 → 0.001 s/1K on a stone-cold cache. Warmup itself costs ~10 s cold
/ ~0.7 s warm — one-time environment setup, the moral equivalent of the C++
boosters' ahead-of-time compilation at package build.

Question: where would you like this to live so the leaderboard timings reflect
steady-state behavior, and do you consider it fair?

1. the model wrapper calls `chimeraboost.warmup()` at module import (module
   import is already outside the fit timer);
2. the environment sets `CHIMERABOOST_WARMUP=1` (0.14.1 runs warmup at import
   when it's set) — explicit in the worker config rather than implicit in the
   wrapper;
3. something else you'd prefer for reproducibility.

Happy to open a small PR against whichever option you prefer (wrapper change +
pin bump to `chimeraboost>=0.14.1`), and grateful if you could re-time
afterward. Thanks!
