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
2. **Decision tier — ONE run, reported per stratum:**
   `python benchmarks/run_benchmarks.py --decide --seeds 3 --save`
   runs Grinsztajn + HC together (73 datasets) plus their variant families
   (~23 more; `--no-variants` to skip). Strata:
   - **Grinsztajn** — low/no-card, zero multiclass. Note its loaders pass NO
     `cat_features`, so cat levers cannot show up here.
   - **HC** — real high-cardinality cats + multiclass, the regime Grinsztajn is
     blind to (`benchmarks/HIGHCARD_PLAN.md`). Confirmed 2026-07-15 to express
     the CatBoost high-card Brier moat (86–88% CB Brier winrate; `hc_gap.py`).
   - **`@sus25` / `@sus50`** — 25%/50% of the training rows, test set unchanged.
     The small-data regime; a twin reads as a point on its parent's learning curve.
   - **`@time`** — temporal splits on 7 audited HC datasets (`VARIANTS.md`).
     Distribution shift — the only regime probing "train on the past, predict the
     future", which every other split here is blind to.
   - Baselines: reuse the newest clean `*.json` if field/seeds match, else run one.
     Variant flags: see `--chimera-*` args. `--list-datasets` shows what a run
     will cover without downloading anything.
   - **Sequential only** — never two benchmarks at once (HC's CatBoost fits run
     50–240 s on card 7k–15k). Progress: `python benchmarks/bench_status.py`.
   - Sign-test: `python benchmarks/compare_runs.py BASE.json NEW.json --by-suite`
     → one INDEPENDENT test per stratum. **Never pool them.** The suites answer
     different questions, and a variant reuses its parent's rows, so a pooled bar
     would count the same data twice and be a weaker test wearing the same name.
     A change that wins on only ONE stratum needs a mechanism story for why (e.g.
     a high-card lever helps HC but is inert on Grinsztajn). Exact ship-rule
     weighting across strata = Nathan's call at first live use (not hardcoded).
     HC multiclass Brier/F1 columns are report-only.
3. **Independent one-shot gate**: `--openml` (never re-run until it passes — it's one-shot to stay independent).
   PMLB (`--pmlb --pmlb-fold tune`) is only for HP tuning, with `holdout` as its confirm fold.

**Always print the aggregate table after every run** (bench_status or summarize output), unprompted.

Ship rules:
- Decisive sign test + a non-negative MEDIAN improvement on Grinsztajn AND a
  non-negative OpenML gate. (Median, not mean: the mean of relative gaps is the
  statistic that produced this project's −144% and −8e21% readings. The run
  summary now reports head-to-head win rate with a bootstrap CI plus a median
  gap, scored on Brier for classification like every other decision — it used to
  print an unguarded mean on F1, disagreeing with both the chart and the gate.)
- Brier gains ship even at small F1 cost. Large speed regressions need explicit user sign-off
  (user accepted 7.9× for cross_features: "as long as we are Pareto and all python").
- Near-solved guards (`summarize.NEAR_SOLVED_NRMSE`, Brier `skip_best_below`) exist because
  %-vs-best explodes when best→0 — don't chase wins/losses on near-solved sets.
  `compare_runs.py` applies the same guard to its MEAN (and prints a median beside it);
  excluded sets are named in the output. Its sign test and PASS/FAIL bar still count every
  dataset, so the guard can't flip a verdict recorded in an older plan file — `--keep-near-solved`
  reproduces pre-fix numbers when auditing one. Every suite has such sets (gr: visualizing_soil,
  SGEMM; hc: cjs; oml: mushroom, nursery), so read the median whenever the mean looks wild.
- Bit-identical refactors: goldens + numerical-identity tests must pass exactly; keep old kernels
  as oracle tests when replacing kernels.

A/B trap (cost an hour once): editable install means `python script.py` runs **repo** code from any
CWD. For worktree baselines set `PYTHONPATH=<worktree>` and print `chimeraboost.__file__` in both arms.

After a ship: update CHANGELOG [Unreleased], regenerate the Pareto (`/pareto`), and record the
verdict (win or kill — kills are valuable) in memory's algorithm history.

Sealed holdouts are NEVER part of this loop: TabArena (`/tabarena`) and the
`pub:` public suite (`benchmarks/PUBLIC_PLAN.md`) are report-only. Never read
either — aggregate or per-task — to justify a source change.

Speed note: `fit_time` excludes prediction and metric computation as of issue
#37, and runs are stamped `timing="fit_only"`. Older result JSONs charged
scoring to fit, so their Speed columns are not comparable; `compare_runs` and
`summarize` warn on a mix.
