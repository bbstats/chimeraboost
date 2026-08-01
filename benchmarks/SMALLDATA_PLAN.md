# SMALLDATA — closing the single-model small-data gap vs CatBoost (opened 2026-08-01)

Successor to `BREAKTHROUGH_PLAN.md`, which closed with all three of its
candidates resolved and one thread explicitly left open:

> The single-model small-data collapse against CatBoost (81% → 50%/33%/0%) is
> untouched and remains the largest known headroom.

`refit_members` addressed the **bagged** half. This program is the single-model
half.

## What the predecessor already ruled out — do not re-run these

| lever | outcome |
|---|---|
| calibration (temperature, Platt, beta, isotonic) | KILLED — we are the best-calibrated model in the field; the deficit is resolution |
| CatBoost's `random_strength` / `bagging_temperature` | KILLED by opponent ablation — perfect port worth ≈1 win-rate point |
| depth race (6 vs 4) | priced dead — capacity-racing family oracle is +0.12–0.29% before an ~80% validation haircut |
| regressor `min_child_weight` floor | KILLED — direction real (p=0.002), magnitude inside noise |
| linear-leaf per-leaf guard | REFUTED — linear leaves are load-bearing, including on small data |
| semi-oblivious last level | deprioritized — the oblivious tax is worth ~1.4% on 2 of 57 datasets |

**Three capacity levers all came back too small. Capacity is not the
mechanism.** This program is required to start from a different axis.

---

## FINDING 5 — CatBoost's ONLY size-dependent default is the learning rate, and ours is size-blind

A win rate that collapses as rows shrink is the signature of a mechanism that
*switches on* at small n. So rather than guess which of CatBoost's mechanisms
that is, ask it: fit at a range of row counts and diff `get_all_params()`.

Of CatBoost's 43 resolved parameters, **exactly one varies with dataset size**,
and it is the learning rate (`scratchpad/catboost_size_defaults.py`, harness
budget `n_estimators=2000`, `early_stopping_rounds=50`):

| train rows | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 20,000 | 60,000 |
|---|---|---|---|---|---|---|---|---|
| CatBoost regression | 0.0259 | 0.0299 | 0.0334 | 0.0372 | 0.0430 | 0.0479 | 0.0534 | 0.0635 |
| CatBoost binary | 0.0158 | 0.0198 | 0.0234 | 0.0278 | 0.0349 | 0.0414 | 0.0491 | 0.0644 |
| **ChimeraBoost (any size)** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** |

Two things fall out, and the second is the one that matters:

1. **`boosting_type` does not vary.** Ordered boosting is *not* what switches on
   at small n — it was the obvious hypothesis and it is dead before any run.
2. **CatBoost's rate is a clean power law in n** — regression
   `0.0259·(n/200)^0.157`, binary `0.0158·(n/200)^0.247`, both reproducing all
   eight measured points to under 1%. It is below our flat 0.1 at *every* size
   in our suites, but the mismatch widens from ~1.6x at 60k rows to **4x
   (regression) and 6.4x (binary) at 200 rows** — which is the shape of the
   collapse curve.

`_auto_learning_rate` returns a flat 0.1 whenever early stopping is on, and its
docstring justifies it as converging "in ~half the trees of a smaller rate with
no measured accuracy cost". That was measured at full size. **Nobody has tested
it at a quarter size.**

### Why this is a different axis, not another capacity knob

Depth, `min_child_weight` and the leaf guard all restrict what a single tree can
express. The learning rate does not restrict the model class at all — early
stopping is free to buy back any lost capacity by growing more rounds. What it
changes is the **step size along the boosting path, and therefore how finely
early stopping can choose where to stop**. On small data the validation curve is
noisy and shallow; at 0.1 each step is a coarse jump, so the best reachable
point on the path can sit well off the true optimum. That is an optimisation and
model-selection story, and it predicts things the capacity story does not.

### Why it clears the leverage arithmetic

The bar from the predecessor: judge candidates on how **broadly** they move the
metric, never on how many named losses they target — a broad +0.25% is worth
4–5 win-rate points, while sweeping every named CatBoost loss is capped at ~6.
A default learning rate is as broad as a change gets: one default, every
dataset, no per-dataset audition, and therefore none of the ~80% haircut that
sank the capacity-racing family.

### The cost, stated up front

Lower rate ⇒ more rounds ⇒ slower fit, and the north star is strength *vs
slowdown*. The probe must therefore report round and fit-time ratios, not just
strength. The reason this may still be a Pareto move: the rate would only drop
where rows are few, and small datasets are exactly where fits are cheapest in
absolute seconds. A size-adaptive rate concentrates its cost where cost is
least. If the fit-time ratio is bad even there, this dies like the others.

---

## C4 — probe design (pre-registered 2026-08-01, before any results)

`benchmarks/probe_learning_rate.py`. Pilot first: if a quarter-size effect is
not visible on the datasets most likely to show it, the thread closes cheap.

- **Datasets (12)**: the six binary sets where CatBoost beats us
  (bank-marketing, credit, heloc, california, default-of-credit-card-clients,
  Diabetes130US), two binary controls where we win comfortably (pol,
  MagicTelescope), and four regression sets (cpu_act — which has confessed a
  huge capacity preference twice — plus sulfur, houses, superconduct). Decision
  suites only: no `pub:`, no TabArena in any form.
- **Sizes**: train fraction ∈ {1.00, 0.50, 0.25} via the harness's own
  `_subsample_train` (random_state=0, **test set unchanged**) — the `@sus`
  semantics, read as a learning curve. Rows capped at 20,000 for the pilot,
  which is stated rather than hidden: it shrinks the full-size arm on the
  largest sets, so the pilot's frac=1.00 column is *not* the decide regime.
- **Our arms**: `lr ∈ {0.1 (shipped), 0.05, 0.03}` plus `cb`, the CatBoost
  power law above evaluated at each cell's own row count. Everything else is
  out-of-box default (`n_estimators=2000`, `early_stopping_rounds=50`,
  `random_state=0`), fitted with **no explicit eval_set** so the internal split
  and `refit_full` run exactly as a user gets them.
- **Opponent arms**: CatBoost at its default auto rate, and CatBoost **forced to
  our 0.1**. This is the "ablate the opponent" method that earned its keep
  twice: if denying CatBoost its rate schedule erases its small-data edge, the
  mechanism is confirmed from its side at the cost of one benchmark.
- **Primary arm, named before the run**: **`cb` at frac 0.25** — the
  size-adaptive shape that would actually ship. The fixed rates are supporting
  evidence, and every sign test carries a **Holm correction across the four
  arms**, because four arms times three sizes is twelve chances to find a
  majority in noise.
- **Reading**, matched to the house tools: seeds averaged on the metric before
  any ratio; wins/losses/ties on `compare_runs`' ±1e-9 dead band; near-solved
  excluded on the best arm in the cell; per-dataset rows printed before any
  aggregate; rounds and fit seconds printed as ratios.
- **Metric**: RMSE for regression, Brier for binary — the house primary.

### Pre-registered predictions

- **C4 right**: the `cb` arm beats shipped 0.1 at frac 0.25 on a Holm-corrected
  sign test; the advantage grows monotonically as rows shrink; and the CatBoost
  ablation shows its small-data edge shrinking when forced to 0.1. Ship shape: a
  size-adaptive `_auto_learning_rate`, then the standard tier-1 synth and
  tier-2 decide gates.
- **C4 wrong**: flat-to-negative at every size ⇒ the thread closes and the flat
  0.1 is vindicated at small size as well as large. **One caveat pre-registered
  against that kill**: if the round counts do not move either, the arms did not
  bind and the honest finding is "these rates did nothing", not a refutation.
- **Pareto kill, stated before the numbers**: even a strength win dies if the
  fit-time ratio at the sizes where it wins is worse than the strength gain buys
  on the frontier. Strength alone is not the bar; the chart is.

---

## Rules for this program
- Inherited unchanged from the predecessor: every candidate gets a cheap
  decisive probe BEFORE library work; negative results are written down here in
  the same change that produces them; the ship gate is tier-1 synth then
  `--decide --seeds 3 --save` with per-stratum sign tests, never pooled.
- TabArena stays sealed and report-only.
