# REPLAY — structure-transfer full-data refit (pre-registered 2026-07-27)

## Mechanism

`refit_full=True` has been the default since 2026-07-25: after early stopping
picks the tree count, the winner is retrained from scratch on 100% of the rows
so the shipped model finally sees the validation split. It works
(REFIT_PLAN.md: Grinsztajn 48W-11L +2.0%) but it costs a second, longer fit.

Fresh attribution on today's default (`benchmarks/attr_legs.py` over
`profile_fit.py --attribution`, 4 Grinsztajn sets, 2026-07-27) puts that refit
at **37-49% of every default fit** — the single largest leg, now that it has
displaced the auditions:

| dataset | auditions | ES winner | refit_full |
|---|--:|--:|--:|
| cpu_act | 32% | 31% | 37% |
| diamonds | 14% | 41% | 45% |
| MagicTelescope | 11% | 43% | 47% |
| road-safety | 8% | 43% | 49% |

The same profile puts tree GROWTH at 83-85% of a fit. So the refit spends ~85%
of its time rediscovering split structures the winner already found. The claim:
it does not have to. Replay the winner's splits round by round against
gradients computed on all rows, and refit only the leaf values (and the
linear-leaf coefficients). The validation rows still reach the leaf estimates —
which is the mechanism REFIT_PLAN credits for the win — without re-paying for
the split search.

Implemented as `refit_full="replay"` (`tree.replay_oblivious_tree`,
`booster.replay_donor`, `sklearn_api._refit_on_full(replay=True)`). The donor's
preprocessor is reused verbatim: `splits_thr` are bin INDICES, so a re-fitted
binner would silently move every threshold underneath the structures.

Scalar booster only. The multiclass round grows one vector-leaf tree through a
separate loop, so multiclass keeps the from-scratch refit and is expected to be
inert in the A/B — a built-in null slice.

## What replay gives up

Split choice still comes from the 80/85% train subset; only the leaf values see
all rows. So replay should capture most, not all, of the full refit's gain.
Two-sided: structures chosen on a subsample and valued on everything is also a
mild regularizer, so replay beating full on some sets is expected, not a bug.

## Phase 0 probe (`benchmarks/probe_replay_refit.py`, 10 sets x 3 seeds,
shipped defaults, nothing pinned) — RAN 2026-07-27

| | mean gain vs no-refit | fit cost vs no-refit |
|---|--:|--:|
| `refit_full=True` | +3.685% | 1.89x |
| `refit_full="replay"` | +3.412% | 1.37x |

Median capture 94% over the 10 sets; replay >= full on 3/10 (pol 108%,
electricity 101%, heloc 209%). **Replay saves 27% of today's default fit
time.** Weakest capture: elevators 60%, covertype 84% — both large-n numeric,
where split re-selection on the extra rows plausibly does carry real signal.

Cost floor measured (`benchmarks/replay_attr.py`): a replay round costs ~1/3 of
a grow round, not ~1/10, because the grow kernel produces the leaf assignment as
a by-product of its fused quantized level pass while replay pays for assignment
separately. Parallel twins of `_assign_leaves`/`_leaf_values` were tried and
made no measurable difference (1.37x vs 1.41x, noise) — REVERTED, not kept.
Remaining headroom is a fused assign+accumulate kernel; not pursued here.

## Pre-registered predictions

- **Tier 1 (synth screen):** near-flat overall, NOT a broad win — replay is a
  cheaper way to buy the same thing, not a new source of accuracy. Expect a
  small negative mean (giving back part of refit's gain) that is not
  decisive by the sign test. Multiclass slice expected exactly inert
  (falls back to the full refit). Kill if: a large systematic loss (mean worse
  than −1.0%), or losses concentrated in the small-n / steep-learning-curve
  slice that refit_full was shipped to fix — that would mean replay misses the
  mechanism rather than approximating it.
- **Tier 2 (Grinsztajn, HC):** small negative mean on the primary metrics,
  sign test NOT decisive against; fit time down 20-30% on the ChimeraBoost arm.
  The trade, not the sign test, is the verdict.
- **The decision number is the Pareto position, not the sign test.** This
  change deliberately trades a little strength for a lot of speed, so the
  usual "decisive sign test + mean improvement" ship rule does not apply as
  written. What must hold: the replay point beats the CHORD between rung 2
  (no refit) and rung 3 (full refit) at its own cost — i.e. it is a genuine
  frontier point, not an interior one.

## Arms

- BASE: this worktree, defaults, no flag (`refit_full=True`). Library is
  default-path bit-identical to main — 636 tests green including the
  numerical-identity goldens.
- VARIANT: `--chimera-refit-replay` => `refit_full="replay"`.
- Both arms run from the SAME worktree in the same session, so the
  baseline-fingerprint trap cannot bite. `--models ChimeraBoost` on the A/B
  runs: the sign test only ever reads our own arm, and dropping CatBoost cuts
  HC runtime by most of its wall clock. The full field is re-run once, later,
  for the Pareto refresh.

## Ship question reserved for Nathan

Replay is CHEAPER and slightly WEAKER than the current default. His standing
rule is "the default is the best non-ensembling rung, always"
(2026-07-24, restated 2026-07-25), which as written keeps `refit_full=True` as
the default and makes replay a new rung on the `quality` ladder — it lands
between today's rung 2 (no refit, 5.36x) and rung 3 (full refit, 9.35x) but
much closer to rung 3's strength. Presenting the measured trade and letting him
choose; NOT flipping the default unilaterally.

## Results log

- 2026-07-27: pre-registered. Phase 0 probe PASS (94% capture, −27% fit).
- 2026-07-27 **Tier 1 (synth screen) PASS**: base `replay-synth-base` vs
  variant `replay-synth-var`, 136 datasets x 3 seeds, `--models ChimeraBoost`.
  **60W-41L-35T, mean +0.552%** (p=0.073) — better than the predicted small
  negative. Fit time 0.871x summed, median 0.800x per dataset, faster on
  118/136.
  Mechanism confirmed on three counts: **multiclass exactly 0-0-34 / +0.000%**
  (the predicted null slice — the fallback is wired correctly), **canaries
  exactly flat** (0-0-3, +0.000%), and the concession shows up where predicted
  — replay gives up split RE-SELECTION, and the deep-interaction slice is dead
  even (depth>=3: 26-26-25, +0.285%, p=1.00) while shallow gains (depth<=2:
  34-15-10, +0.901%, p=0.009). Replay is ahead on high-card cats (card>16
  +2.185% p=0.003; entity cats +1.486% p=0.001) — structures chosen on a
  subsample and valued on everything is a mild regularizer, and the donor's
  ordered-TS encoder is reused rather than re-permuted. No kill condition
  triggered (no large negative; small-n slice +0.679%, not a loss cluster).
- 2026-07-27 **Tier 2 Grinsztajn: ACCURACY FLAT, BIG SPEED WIN**. Base
  `replay-gr-base` vs `replay-gr-var`, 59 datasets x 3 seeds.
  **27W-32L-0T, mean +0.005%, median −0.005%** (near-solved
  SGEMM/visualizing_soil excluded from the mean, as always). Sign-test bar
  FAIL — pre-registered as non-decisive; a 27-32 split at a mean of five
  thousandths of a percent IS parity, which is the result this change wants.
  **Fit time 327.6s → 214.9s = 0.656x (−34.4%), faster on 59/59 datasets**,
  median per-dataset 0.703x, p90 0.866x — the speed win has no tail.
- 2026-07-27 **Tier 2 HC: parity, smaller speed win**. Base `replay-hc-base`
  vs `replay-hc-var`, 14 datasets x 3 seeds. **4W-5L-5T, mean −0.256%,
  median +0.000%** (cjs near-solved, excluded). Fit time 0.830x (−17.0%),
  faster on 12/14. Five ties are the sets that never reach a scalar refit
  (multiclass eucalyptus / okcupid-stem among them), which is also why HC
  gains less than Grinsztajn: replay cannot touch multiclass, and cat
  preprocessing is a fixed cost. Largest moves both ways and roughly
  symmetric: house_prices_nominal +3.51% and Moneyball +3.10% against
  colleges −4.51% and wine-reviews −3.42%.
- 2026-07-27: 650 tests green (14 new in `tests/test_refit_replay.py`,
  including the multiclass no-op, donor-structure reuse, full-data init, and
  that the shipped model does not pickle the donor forest). Default paths
  bit-identical — the numerical-identity goldens are untouched.

## Verdict

**Accuracy is a wash on both decision suites; fit time falls 34% on
Grinsztajn (59/59 datasets) and 17% on high-card.** The pre-registered chord
test is passed trivially: replay sits at rung 3's strength for roughly
two-thirds of rung 3's cost, so it beats any interpolation between rungs 2
and 3 by a wide margin.

Shipped as an OPTION, `refit_full="replay"`. **Defaults are unchanged**, so
the README Pareto chart does not move and no chart refresh is owed.

The default question is Nathan's, per the reserved decision above. The
argument each way, stated plainly: his standing rule is that the default is
the strongest non-ensembling setting, and replay is a hair behind on
high-card (−0.256%) though a hair ahead on Grinsztajn (+0.005%) — so on a
strict reading of "strongest", `True` stays. But the two are tied inside
noise, and replay is a third cheaper, so if the rule is read as "best
accuracy per unit cost at equal accuracy", replay wins. A default flip would
move the charted default from 9.35x to roughly 6.5x at essentially unchanged
strength, which is the largest single move on the frontier available right
now.

## Open / stackable

- **`Sel25`** (auditions 100 -> 25) still stacks on top of this: it was
  measured at −26% fit back when auditions were ~35% of a fit, but the
  `refit_full` default flip has since diluted it to about **12%**
  (auditions are now 8-32% of a fit, mean ~16% — see the attribution table
  above). Worth doing, no longer headline-sized.
- **Fused assign+accumulate replay kernel.** A replay round currently costs
  ~1/3 of a grow round rather than the ~1/10 the work would suggest, because
  the grow kernel gets the leaf assignment free as a by-product of its fused
  quantized level pass while replay pays for assignment, leaf-value scatter
  and the `F` update as three separate passes over the rows. Fusing them is
  the remaining headroom. Parallel twins of `_assign_leaves`/`_leaf_values`
  were tried first and made NO measurable difference (1.37x vs 1.41x, noise)
  — reverted; do not retry that specific idea.
- **Multiclass replay.** The vector-leaf round would need its own replay
  path. HC's multiclass sets are exactly where the tie count comes from.
