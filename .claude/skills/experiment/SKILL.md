---
name: experiment
description: Run the validated A/B experiment protocol for a proposed library change — benchmark, sign-test, gate, ship-or-revert
---

The validated 3-tier methodology (it shipped mcw-auto, linear-leaves, cross_features; skipping tiers shipped nothing):

1. **Mechanism probe** (cheap): the SynthGen screen —
   `python benchmarks/run_benchmarks.py --synth --seeds 3 --save` (182 frozen prior-sampled
   datasets, ~30 min) vs the newest synth baseline, then
   `python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost` and
   `python benchmarks/synth_report.py BASE.json NEW.json` — the factor table must show the
   effect concentrated in the slice the mechanism predicts (validated 8/9 vs the ledger,
   2026-07-14; e.g. removing cross_features = −3.3% exactly on the interaction-depth≥2 numeric
   slice). Kill here if the mechanism story doesn't show up. Fall back to a ≤6-dataset dev
   panel or a zero-library-change probe script (cf. `probe_cross_features.py`) only where no
   recipe factor can express the idea. Known v1 biases (don't over-read): targets run slightly
   shallow (depth-4 arm disagrees), synthetic cats lack entity effects (CatBoost's high-card
   moat absent), mcw large-n slice leans positive — see `benchmarks/synthgen/README.md`.
2. **Decision-suite A/B — Grinsztajn + HC (run BOTH, sign-test SEPARATELY):**
   - Grinsztajn (the low/no-card, 0-multiclass suite):
     `python benchmarks/run_benchmarks.py --grinsztajn --seeds 3 --save`.
   - HC (the real high-cardinality suite: entity cats, high card, multiclass — the
     regime Grinsztajn is blind to; `benchmarks/HIGHCARD_PLAN.md`):
     `python benchmarks/run_benchmarks.py --highcard --seeds 3 --save`.
     Confirmed 2026-07-15 to faithfully express the CatBoost high-card Brier moat
     the synth entity prior predicted (86–88% CB Brier winrate; `hc_gap.py`).
   - Baselines: reuse the newest clean `*.json` per suite if field/seeds match, else run one.
     Variant flags: see `--chimera-*` args in run_benchmarks.py.
   - **Sequential only** — never two benchmarks at once (HC's CatBoost fits run
     50–240 s on card 7k–15k). Progress: `python benchmarks/bench_status.py`.
   - Compare each suite: `python benchmarks/compare_runs.py BASE.json NEW.json`
     (per-dataset sign test). **Report the two sign tests separately**, then a
     pooled union verdict. A change that wins on only ONE suite needs a mechanism
     story for why (e.g. a high-card lever helps HC but is inert on Grinsztajn).
     Exact ship-rule weighting Grinsztajn vs HC = Nathan's call at first live use
     (not hardcoded). HC multiclass Brier/F1 columns are report-only — the blended
     north star (make_pareto) is unchanged.
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
