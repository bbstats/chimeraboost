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

## Phases

- [ ] **Phase 0** — offline headroom + race simulation on PMLB tune
      (`benchmarks/a2_phase0.py`). Bars A–D. No source change.
- [ ] **Phase 1** — if Phase 0 passes: implement the config race inside the
      existing `selection_rounds` audition, default-off flag, byte-identical
      when off, numerical-identity goldens green.
- [ ] **Phase 2** — `/experiment`: synth screen → Grinsztajn + high-card →
      OpenML one-shot gate. Default flip is Nathan's call, as always.
