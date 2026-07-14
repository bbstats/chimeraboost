---
name: experiment
description: Run the validated A/B experiment protocol for a proposed library change — benchmark, sign-test, gate, ship-or-revert
---

The validated 3-tier methodology (it shipped mcw-auto, linear-leaves, cross_features; skipping tiers shipped nothing):

1. **Mechanism probe** (cheap): synthetic or a ≤6-dataset dev panel, or an external-augmentation
   probe script under `benchmarks/` that needs zero library changes (cf. `probe_cross_features.py`).
   Kill here if the mechanism story doesn't show up.
2. **Full Grinsztajn A/B** (the decision suite):
   - Baseline: reuse the newest clean `benchmarks/results/*.json` if the field/seeds match, else run one.
   - `python benchmarks/run_benchmarks.py --grinsztajn --seeds 3 --save` (flags for the variant:
     see `--chimera-*` args in run_benchmarks.py).
   - **Sequential only** — never two benchmarks at once. Progress: `python benchmarks/bench_status.py`.
   - Compare: `python benchmarks/compare_runs.py BASE.json NEW.json` (per-dataset sign test).
3. **Independent one-shot gate**: `--openml` (never re-run until it passes — it's one-shot to stay independent).
   PMLB (`--pmlb --pmlb-fold tune`) is only for HP tuning, with `holdout` as its confirm fold.

**Always print the aggregate table after every run** (bench_status or summarize output), unprompted.

Ship rules:
- Decisive sign test + mean improvement on Grinsztajn AND a non-negative OpenML gate.
- Brier gains ship even at small F1 cost. Large speed regressions need explicit user sign-off
  (user accepted 7.9× for cross_features: "as long as we are Pareto and all python").
- Near-solved guards (`summarize.NEAR_SOLVED_NRMSE`, Brier `skip_best_below`) exist because
  %-vs-best explodes when best→0 — don't chase wins/losses on near-solved sets.
- Bit-identical refactors: goldens + numerical-identity tests must pass exactly; keep old kernels
  as oracle tests when replacing kernels.

A/B trap (cost an hour once): editable install means `python script.py` runs **repo** code from any
CWD. For worktree baselines set `PYTHONPATH=<worktree>` and print `chimeraboost.__file__` in both arms.

After a ship: update CHANGELOG [Unreleased], regenerate the Pareto (`/pareto`), and record the
verdict (win or kill — kills are valuable) in memory's algorithm history.

TabArena is NEVER part of this loop — it's a sealed holdout, re-read only after shipping (`/tabarena`).
