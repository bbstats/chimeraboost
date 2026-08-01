# A2 — config-portfolio audition on the early-stopping split

Pre-registered before any result was read (the A1_PLAN / REFIT_PLAN convention:
design, predictions and kill bars first, evidence appended after).

## Why

ChimeraBoost already selects *structure* on the early-stopping split — constant
vs linear leaves, plain vs cross-augmented, and since 0.15 it does so through a
cheap capped audition (`selection_rounds=100`). That pattern has shipped five
times. What it has never been pointed at is *capacity*: `depth`, row subsampling
and friends are fixed global constants chosen once on Grinsztajn.

The external evidence for making capacity per-dataset is the strongest of the
remaining program items: TabRepo (2311.02971) reports AutoGluon-1.0's portfolio
of a handful of configurations reaching ~75% win rate against tuned single
configs, and Better-by-Default (2407.04491) finds the best default *differs by
task type* — regression wants deeper trees than classification, and row
subsampling belongs on. A portfolio raced on our existing audition machinery is
the GBDT-side analog of the in-context adaptation that makes tabular foundation
models strong on unseen data, which is what this whole program is chasing.

**Honest prior against it.** Our own PMLB random-search study found that broad
hyperparameter tuning buys essentially nothing that transfers: the search winner
*anti*-generalized (−1.87% out of sample) and only `min_child_weight` carried
transferable signal — and the classifier already makes that one size-adaptive.
A2 is not refuted by that result, because A2 does not claim a better global
default; it claims that picking per dataset on held-out validation data beats
any fixed choice. But the study says the headroom is probably thin, so Phase 0
exists to measure the headroom before a line of library source changes.

## Instrument — and a protocol call

A2 is a hyperparameter question, so Phase 0 runs on the **PMLB tuning fold**
(`pm:tune/*`, 13 sets: 5 regression, 3 binary, 5 multiclass) — the one suite
this project tunes hyperparameters against. Grinsztajn, the high-card suite and
the OpenML gate are **not** touched in Phase 0; measuring accuracy headroom on
a decision suite and then designing to it would be tuning on the instrument that
later has to judge the result. TabArena stays sealed as always.

If Phase 0 passes, the change goes through the normal `/experiment` protocol:
synthetic screen (tier 1) → Grinsztajn + high-card, sign-tested separately
(tier 2) → OpenML one-shot gate.

Phase 0 mirrors the decision-suite protocol exactly: same loaders, same
`train_test_split(test_size=0.25, random_state=seed)`, same
`rng = default_rng(1000 + seed)`, same shared budget (2000 rounds, patience 50),
and `fit(X, y)` with **no** explicit eval_set so every fit uses the estimator's
own internal early-stopping split — out-of-box default behavior.

## The portfolio

Four configurations, everything not named left at the shipped default:

| id | config | why it is in the portfolio |
|---|---|---|
| `default` | `depth=6` | the shipped default, the incumbent |
| `d4` | `depth=4` | shallower for small/noisy targets |
| `d8` | `depth=8` | Better-by-Default: regression wants more capacity |
| `sub08` | `depth=6, subsample=0.8` | Better-by-Default: row subsampling on |

Each is fit to full early stopping so its whole validation curve is recorded;
the race is then simulated **offline** from those curves at several budgets.
No library source changes in Phase 0.

## What Phase 0 measures

1. **Test oracle headroom** — per-cell best config by *test* metric vs the
   fixed default. The ceiling: nothing downstream can beat this.
2. **Validation-selected headroom** — pick by lowest full-early-stopping
   validation loss (ties to `default`), then read that pick's *test* metric.
   This is what A2 could actually deliver, and it is the number that matters.
3. **Race fidelity at k** — would judging on only the first k rounds of each
   curve pick the same config as the full fits, for k in {50, 100, 200, 500}?
   Plus the test regret conceded when it mispicks.
4. **Cost** — measured fit seconds per config, and the projected fit-time
   multiple of a k-round audition per extra portfolio member.

## Pre-registered kill bars

All four must hold for A2 to proceed to a library implementation.

- **Bar A — capturable headroom.** Validation-selected config beats the fixed
  default by **≥ +0.30% mean** on the primary metric across cells, and wins in
  **≥ 55%** of non-tied cells. (Calibration: the program's own estimate is that
  ~+0.25% broad ≈ +12pp of head-to-head win rate, so +0.30% is a real Pareto
  move rather than noise.)
- **Bar B — oracle sanity.** Test-oracle headroom **≥ +0.50%**. If even the
  cheating oracle cannot clear this, there is nothing worth racing and A2 dies
  regardless of Bar A.
- **Bar C — race fidelity.** At k=100 the raced pick reproduces the full-fit
  validation pick on **≥ 75%** of cells with **≤ 0.15% mean test regret**. This
  is the known trap: the constant-vs-linear race already mispicks about a third
  of regression selections because those curves genuinely cross late.
- **Bar D — cost.** Projected default fit-time multiple **≤ 1.35×**. The Pareto
  point must move up more than it moves right; at today's 4.9× slowdown, 1.35×
  lands near 6.6× and the strength gain has to pay for that.

Failing Bar A or B is a **KILL** and gets written into the ledger as such —
seven of the last seven categorical levers died this way and the record is the
deliverable either way.

## Predictions (stated before the run)

- Test-oracle headroom clears Bar B, mostly on the multiclass sets (5 of 13),
  where the shipped defaults have had the least tuning attention.
- Validation selection recovers well under half of the oracle — small PMLB
  validation splits are noisy, and the oracle is fit on the test metric.
- Bar A is the coin flip and the one most likely to kill A2.
- Race fidelity at k=100 is *better* here than for constant-vs-linear, because
  depth changes the curve's whole trajectory rather than its late shape.

## Phase 0 RESULTS (2026-07-25) — pre-registered A2 KILLED on cost

Run: `python benchmarks/a2_phase0.py --seeds 3 --out a2-phase0`, 13 PMLB tune
sets × 3 seeds × 4 configs = 156 fits. Full record in
`benchmarks/results/a2-phase0.{json,md}` (gitignored).

**Measurement guard applied.** `pm:tune/nursery` is solved to numerical zero
(Brier ~1e-49 for every configuration) and manufactured a −8×10²¹% "mean
delta" on the first pass. It is dropped by summarize.py's own near-solved
convention (classification best Brier < 1e-3; regression best NRMSE < 0.02),
leaving 36 of 39 cells. Everything below is on the retained cells.

| bar | quantity | value | verdict |
|---|---|--:|---|
| B | test-oracle headroom (ceiling) | +2.587% | **PASS** (≥ +0.50%) |
| A | validation-selected headroom, mean | +0.522% | **PASS** (≥ +0.30%) |
| A | validation-selected win rate | 70.0% (21W-9L-6T) | **PASS** (≥ 55%) |
| C | race fidelity @ k=100 | 83.3%, 0.119% regret | **PASS** (≥75%, ≤0.15%) |
| D | projected fit-time multiple | **1.90×** | **FAIL** (≤ 1.35×) |

Validation selection is real but weak: it recovers only **20% of the oracle
ceiling**, and the sign test over cells is 21W-9L-6T, p=0.043. Median gain
+0.358%.

**Why it dies: three extra auditions cost too much.** A k=100 audition is
nearly a whole fit whenever the full fit is short, which is common here (mean
323–382 rounds, and multiclass fits run ~100 rounds on the decision panel).
Racing four configurations lands at 1.90×, far outside the bar.

### What survives (POST-HOC — proves nothing until re-validated)

Same curves, cheaper portfolios. Selection bias applies: these subsets were
chosen knowing this panel, and seven-way subset search makes a p=0.012 worth
roughly 0.08 after a Bonferroni correction.

| portfolio | mean % | median % | W-L-T | p | cost k=50 | cost k=100 |
|---|--:|--:|--:|--:|--:|--:|
| **default+d4** | +0.398% | +0.000% | 16W-4L-16T | 0.012 | **1.12×** | **1.24×** |
| default+d4+sub08 | +0.844% | +0.230% | 20W-9L-7T | 0.061 | 1.29× | 1.58× |
| default+sub08 | +0.681% | +0.000% | 11W-10L-15T | 1.000 | 1.17× | 1.34× |
| default+d8 | −0.125% | +0.000% | 6W-4L-26T | 0.754 | 1.16× | 1.32× |

A two-way depth race (6 vs 4) clears every bar including cost. `d8` is simply
bad here (12W-24L on its own) and belongs in no portfolio.

### The size caveat that decides whether this transfers

The gain is concentrated on small data, which is exactly where this project
has been burned before ("small panels invert knob verdicts"):

| n_train band | datasets | mean % vs default |
|---|---|--:|
| under 2,000 | SWD, LEV, contraceptive, yeast, segmentation | +1.09% to +7.41% |
| 2,000–5,000 | hypothyroid, churn, texture | −5.52% to +0.22% |
| over 4,900 | wind, puma8NH, coil2000, house_8L | +0.12% to +0.64% |

On the larger half of the panel the gain collapses to roughly +0.1% to +0.6%,
and the Grinsztajn and high-card suites are larger still. Two losses are
genuine, not near-solved artifacts (hypothyroid Brier 0.026–0.039, texture
0.014–0.023) — validation genuinely mispicks there. Note also that relative
Brier deltas amplify small absolute gaps on easy multiclass sets, so the
sign test is the trustworthy reading and the mean is not.

### Verdict

**A2 as pre-registered is KILLED at Phase 0** — Bar D fired and the plan
required all four bars. The honest residual is narrower than the idea that
was registered: not a config portfolio, but a **two-way depth race (6 vs 4)
at k=50–100**, worth about +0.4% on small-to-mid data for 1.12–1.24× fit,
with the transfer question unresolved because this panel skews small.

That variant is a legitimate Phase 1 candidate but it was selected post hoc,
so it is not evidence yet. It only becomes real if it survives the untouched
instruments: synth screen → Grinsztajn + high-card (sign-tested separately)
→ OpenML one-shot gate. **Nathan's call whether that is worth the run.**

### RESOLVED 2026-08-01 — the transfer question is answered, and the race is dead

Both halves closed in `BREAKTHROUGH_PLAN.md` (C2 and C3):

- **Transfer: confirmed.** The `@sus25`/`@sus50` strata, which did not exist
  when this was shelved, measure exactly the small-data regime depth 4 was
  claimed to win. Every small-data stratum has a positive mean for depth 4 and
  every full-size stratum a negative one, `gr:sus25` 9W-3L. So small data does
  want less capacity, and a uniform depth 4 is still not shippable.
- **The race itself: NOT WORTH THE RUN, and priced rather than guessed.** The
  registered answer was a per-dataset depth race. C3 measured the *oracle* of
  the capacity-racing family — perfect per-dataset selection, which no
  implementation can beat — at a median +0.12% per dataset and +0.29% at
  quarter size. Against this plan's own Phase 0 finding that validation
  recovers only ~20% of an oracle, the achievable prize is ~+0.02-0.06%,
  against a program bar of a broad +0.25%.

⇒ **Do not build the depth race.** The thread is closed; no run is pending.

## Phases

- [x] **Phase 0** — offline headroom + race simulation on PMLB tune
      (`benchmarks/a2_phase0.py`). Bars A–D. No source change.
      **Done 2026-07-25: A2 killed on cost; two-way depth race survives.**
- [~] **Phase 1 — NEVER RUN, moot.** Phase 0 killed the program on cost
      (validation recovers only ~20% of the oracle gain), so the gate that
      would have opened this phase never opened.
- [~] **Phase 2 — NEVER RUN, moot** (same reason). Note for anyone
      reopening this: the OpenML one-shot gate named here was RETIRED
      2026-07-27, so a revived A2 would run synth → Grinsztajn + high-card
      per stratum, with `--public` as post-hoc validation.
