# LEAFTUNE — post-structure hyperparameter tuning by replay

**Status: probes done 2026-07-28. Verdict — do NOT build `tune_leaves` as a
`reg_lambda` sweeper.** The mechanism works and is fast; the thing it was going
to sweep turns out not to be worth sweeping. Nothing shipped. Results below.

## Verdict

1. **The mechanism is sound.** Replay is exact (bit-identical round trip) and
   honest (the random-label canary rises, and sits within 1.6 standard errors of
   the log-2 floor across three seeds).
2. **The mechanism is fast.** A replay sweep costs a median of 5.2x less per
   configuration than re-growing (range 3.4x to 7.2x), and that is a floor —
   preprocessing and leaf routing are still repaid per configuration, which the
   Phase-0 routing cache would remove.
3. **`reg_lambda` is flat.** Against an exact cross-validated grid, tuning it on
   top of the round count won 3, lost 3 and tied 2 of 8 datasets, median exactly
   0.000%. None were near-solved. This reproduces the PMLB random-search finding.
4. **Replay's fidelity is anti-correlated with what matters.** The replay grid
   agreed with the exact grid's chosen cell on only 2 of 8 datasets. That is
   harmless while the surface is flat — median test difference +0.127%, in
   replay's favour. But on the one dataset where lambda genuinely mattered
   (`gr:reg_cat/Brazilian_houses`, +4.4% for exact tuning) replay picked the
   wrong end of the axis and gave up **6.8%** on test. Its out-of-fold regret
   was 0.0044, 2.4x the next largest, so the regret diagnostic did detect its
   own failure — keep that diagnostic in any future version.
5. **Out-of-fold round selection does not beat single-split early stopping.**
   Five folds versus the shipped default: 4 wins, 4 losses, median −0.050%, at a
   median 8.7x the cost. No detectable benefit for 5x to 35x the compute.

**Measurement caveat, stated because it bounds every number above:** one seed
per dataset. `gr:reg_num/cpu_act` alone swings between 2.11 and 2.96 RMSE across
four split seeds, so effects smaller than a few percent are under the noise
floor there. The P4 cpu_act row (−10.26%) was traced to exactly this and is not
a real effect. Conclusions that survive are the flat ones and the speed ratio.

## What is actually tunable at replay (P5, measured not assumed)

The replay round loop IS the grow loop with one call swapped, so a parameter is
genuinely leaf-stage only if changing it leaves ROUTING identical (every row in
the same leaf of every tree) while leaf VALUES change. Measured on a numeric
dataset, a mixed-categorical one (`oml:adult`) and an all-categorical one
(`oml:kr-vs-kp`):

| parameter | verdict |
|---|---|
| `l2_leaf_reg` | tunable — but P1 says the axis is flat |
| `learning_rate` | tunable mechanically; see the caveat below |
| `leaf_estimation_iterations` | tunable, untested for value |
| `ordered_boosting` | tunable, untested for value |
| `subsample` | tunable mechanically (it only zeroes gradients) |
| leaf-loss swap (`Huber`, `Quantile`, ...) | tunable, clean on all three |
| `sample_weight` | tunable **only without categorical columns** |
| `min_child_weight` | **inert** — silently does nothing |
| `colsample` | **inert** |
| `depth` | **inert** |
| `max_bins` | **inert** |
| `quantize_gradients` | **inert** |
| `cat_smoothing` | **moves routing** on categorical data |
| `cat_n_permutations` | changes values, not routing; a preprocessing knob |

Three things this changes about the original spec.

1. The structural rejects (`depth`, `min_child_weight`, `colsample`, binning)
   do need a hard error, but not for the reason the spec gave. They do not
   re-raise leakage — they are consumed only inside `build_oblivious_tree`,
   which replay bypasses, so they run without error and do **nothing**. A sweep
   over them would score a constant and return an arbitrary winner.
2. **`sample_weight` is not safely leaf-stage on categorical data.** Weights
   feed `prep_.fit_transform`, so changing them changes the target encoding,
   which re-bins, which moves rows between leaves. The inherited structure stops
   meaning what it meant. Same for `cat_smoothing`. Nothing raises.
3. `learning_rate` and `subsample` are mechanically clean — routing never moves.
   The spec's instinct to treat `learning_rate` as structural still stands, but
   the reason is semantic, not mechanical: it changes every subsequent round's
   margins and therefore its gradients, so the replayed trajectory drifts away
   from the structures being replayed. Untested.

## The leaf-loss swap (P6–P9): killed, on a structural argument

The quantile loss swap looked like the promising axis. It is not, and the reason
is not an implementation detail — it is a tension between two conditions that
cannot both hold.

**The transfer condition.** Under a location-shift model `Y = f(X) + e` with the
noise independent of `X`, every conditional quantile is the same `f` plus a
constant. The optimal partition for the mean IS the optimal partition for every
quantile, so replay is exact in the population limit. Once the spread varies —
`Y = f(X) + sigma(X)*e` — the tau-quantile is `f(X) + sigma(X)*q_tau(e)`, and the
squared-error split search is blind to `sigma`, because it chases variation in
the mean.

**The need condition.** But if the data really is homoscedastic, you do not need
conditional quantile regression at all: the conditional quantiles are the mean
function plus a constant, obtainable from one squared-error fit plus the marginal
residual quantile.

**So the regime where replay transfers is the regime where nothing needed to
transfer.** Conditional quantile regression earns its keep exactly where replay
breaks. Measured, not argued:

| probe | finding |
|---|---|
| P6, 5 real datasets | replay beat native quantile fitting on all 5, median +7.5%, crossing cut to ~1/5 |
| P7, heteroscedastic synthetic | replay's damage concentrates in the TAILS and grows with heteroscedasticity: tails −0.9% homoscedastic, −9.6% shared-scale, −14.2% hidden-scale, against a hidden-scale median of only −3.2% |
| P8, signal sweep | the "replay wins by variance reduction" story is REFUTED — the margin does not grow as signal weakens, it shrinks |
| P9, rigid offset on the same real data | a rigid location shift (one squared-error fit + marginal residual quantile) scores median +5.45% against native, versus replay's +6.70%, with exactly zero crossing and **10x faster** |

So on the datasets replay won, it bought about 1.3 points of pinball over the
dumbest possible quantile model, at ten times the cost — and on data where the
spread genuinely varies, it gives up 10% to 17% in the tails.

**Caveat that limits every "vs native" number above — RESOLVED by P10, and the
answer was not the one expected.** The native arm was a booster-level quantile
fit pinned at depth 4, 300 rounds, `min_child_weight` 1, where the shipped
`ChimeraBoostQuantileRegressor` raises `min_child_weight` via
`_auto_min_child_weight` and defaults to 2000 rounds with early stopping. That
handicap was real but small, and fixing it did not change the verdict.

## P10–P13: the rigid offset holds up, and the shipped quantile head does not

**P10.** Rebuilt the native arm with the shipped treatment, and gave the rigid
offset the fix it had been denied (residual quantiles from a held-out fold
rather than in-sample, which understates spread). Median pinball against P9's
pinned native arm, 3 seeds, the same 5 Grinsztajn regression sets:

| arm | median vs P9 native | median coverage | seconds |
|---|---|---|---|
| native d4 (P9's arm) | +0.00% | 0.759 | 1.6 |
| native + `min_child_weight` floor | +0.66% | 0.765 | 1.5 |
| native + floor + early stopping | +2.03% | 0.731 | 3.3 |
| shipped `ChimeraBoostQuantileRegressor` | −16.70% | 0.950 | 0.4 |
| rigid offset (P9's) | +5.45% | 0.759 | 0.1 |
| **rigid offset, held-out residuals + ES** | **+8.78%** | **0.793** | **0.3** |

Fair configuration recovered ~2 points of a ~5.5 point gap. The honest offset
got *better*, not worse, and leads the properly configured native arm on all 5
datasets. Crucially it is not buying the win with narrow bands: at 0.793 against
a nominal 0.800 it is the best-calibrated arm in the table, while the tuned
native arms sit near 0.73 and are the ones running too narrow.

*Harness bug, for the record:* P10's "wins vs best offset" summary column
compares each arm against `min(offset, offset OOF+ES)` with a strict `<`, so
the winning offset arm ties itself and prints 0. Column is meaningless; the
per-dataset tables above it are correct.

**P11 — it is not a knob.** Every configuration explanation for the shipped
head's showing is dead. Grid size (3 vs its own default 19), both alternative
projections, `exact_splits`, and P10's fit protocol all leave coverage at 0.92
to 0.998 against a nominal 0.80. Best of them buys under 3% of pinball.

**P12 — the mechanism.** The band freezes by round ~50 and never moves again
out to 3000 rounds, with the narrowing budget exhausted to its floor everywhere:

| dataset | marginal band at start | final band | calibrated band | ratio | budget spent |
|---|---|---|---|---|---|
| Brazilian_houses | 2.1355 | 0.6551 | 0.0656 | 9.98x | yes |
| cpu_act | 25.0000 | 13.2029 | 4.5880 | 2.88x | yes |
| elevators | 0.0140 | 0.0096 | 0.0046 | 2.09x | yes |

`cpu_act` width by round: 13.166 @50, 13.191 @100, 13.202 @300, 13.2029 @1000,
@2000, @3000. `gap_` reads 2.5e-08 against `gap_floor_` 2.5e-08.

The cause is that the budget is charged at the **worst-case leaf**: each round
subtracts `np.diff(tree.values, axis=1).min(axis=0)`, the smallest gap
increment over ALL leaves, so one aggressively-narrowing leaf spends budget on
behalf of every row including rows that never reach it. At depth 4 that is 16
leaves competing to be the worst, every round; it saturates in tens of rounds.
Valid as a bound on the guarantee (no possible input row can cross), far more
conservative than any individual row needs.

Signature to recognise it by: on Brazilian_houses coverage is 0.817 at round 1
(near nominal, since the band still sits at the marginal spread) and 0.995 from
round 50 onward. Coverage getting WORSE with training means the centre keeps
sharpening while the width cannot follow.

**P13 — conformalization does not rescue it.** `conformalize=True` is the
feature's own remedy and it fixes exactly the half that does not matter:

| arm | median vs rigid offset | median coverage | beats offset |
|---|---|---|---|
| rigid offset | +0.00% | 0.793 | — |
| head K=3 | −41.52% | 0.988 | 0 of 3 |
| head K=3 conformalized | −25.22% | 0.793 | 0 of 3 |
| head K=19 (true default grid) | −74.99% | 0.996 | 0 of 3 |
| head K=19 conformalized | −20.04% | 0.827 | 0 of 3 |

Coverage snaps back to nominal; pinball stays 12% to 111% behind the trivial
baseline, losing on 3 of 3. A single global scale per level cannot recover the
conditional information the frozen band discarded.

**Conclusion.** The zero crossing rate is not free. On these datasets it costs
intervals 2x to 10x too wide, and the head loses to a one-model rigid residual
band on every dataset tried, conformalized or not, at higher cost. The
non-crossing guarantee should be re-derived per-row rather than as a worst-case
bound over all leaves and rounds, or made optional. **Not yet shipped as a fix
— no library code touched, and any change needs the `/experiment` protocol.**

*Separate, minor, confirmed:* `_auto_min_child_weight` returns 11 for the grid
[0.1, 0.5, 0.9] where 10 is intended — `1.0 - 0.9` is slightly under 0.1 in
binary and the reciprocal rounds past the integer. The 19-level default grid
(20), [0.25, 0.75] (4) and [0.01, 0.99] (100) are all correct. Fix is to round
the reciprocal before `ceil`.

## P14 — the fix, pre-registered 2026-07-31 (branch `fix-quantile-interval-width`)

Written **before** the fix was measured; the base numbers below are from the
shipped code, re-measured today with `benchmarks/probe_quantile_band.py --tag
base` (which rebuilds P10-P13's dead harness) and they reproduce P12/P13.

**Change.** Delete the training-time narrowing budget entirely and enforce
non-crossing **per row at delivery** by monotone rearrangement — sort each
predicted K-vector. Chernozhukov, Fernández-Val & Galichon (2010): rearranging a
crossing quantile curve never increases pinball loss at any level, for any row.
The guarantee moves from a worst-case bound over all leaves and rounds to an
exact per-row operation, which is what P12 said it should be.

Why this is safe here, verified in code rather than assumed:

- Pinball gradients are per-channel independent (`losses.MultiQuantile.grad_hess`
  reads only `F[:, k]` against `y`), so quantiles crossing *during training*
  corrupt no learning signal. Only delivered predictions need ordering.
- Training `F`/`Fv` are accumulated in place and never read back from a predict
  path, so a delivery-time sort cannot feed back into the fit. **This is the
  distinction that matters:** PR #48 recorded that sorting *leaf vectors during
  training* diverged to pinball 2.9e7 against an oracle of 0.35. That failure was
  a feedback loop — sorted values re-entered `F` and self-reinforced. Sorting
  delivered output has no such path, and `F` must never be sorted in place.

**Base measurements to beat** (K=3 grid, central interval 0.10-0.90, nominal
coverage 0.80, 3 seeds, `results/quantile-band-base.md`):

| dataset | head pinball | head coverage | head width | offset pinball | offset width |
|---|--:|--:|--:|--:|--:|
| Brazilian_houses | 0.02992 | 0.9976 | 0.65019 | 0.00452 | 0.02747 |
| cpu_act | 0.74380 | 0.9821 | 12.38850 | 0.55148 | 4.74665 |
| elevators | 0.00067 | 0.9494 | 0.00971 | 0.00048 | 0.00467 |

Head loses to the rigid offset on 0 of 3 at K=3 and 0 of 3 at K=19. The freeze is
visible directly: cpu_act width 12.912 at rounds 1000, 2000 and 3000 alike, with
coverage climbing 0.861 → 0.987 as the centre sharpens under a fixed band.

**Bars.** All five, or the change does not ship:

1. **Coverage** without `conformalize` inside [0.75, 0.88] at nominal 0.80 on
   3 of 3 (base: 0.949-0.998).
2. **Pinball beats the shipped head** on 3 of 3.
3. **Pinball beats or matches the rigid offset** on 3 of 3 — the ship/kill line,
   and the comparison the shipped head fails 0 of 3. "Matches" means within 2%.
4. **Crossing rate exactly 0.0** in every arm, every path, every staged stage.
5. **The freeze is dead**: central-interval width at round 3000 differs from
   round 50 by more than 1%, and coverage does not rise monotonically with
   rounds.

Plus the unchanged obligations: the bit-identity snapshot must show every array
identical except the two `mq3` families (the shared scalar MAE/Quantile leaf
kernels route through the same code at K=1, where the projection was the
identity — so any scalar diff is a bug, not a consequence), and PR #48's own
ledger in `QUANTILE_PLAN.md` must still pass (within 3% of per-level LightGBM,
CQR worst coverage error ≤ 2 pp).

**Kill clause.** If bar 3 fails, the approach is falsified and reverted; record
the negative here. The decision suites do not score quantiles and this touches
`MultiQuantileBoosting` only, so no `--decide` run is owed unless the identity
snapshot moves a scalar array.

### Result: the mechanism is confirmed and fixed, but bar 3 FAILS — not shipped

Same probe, `--tag fix`, same seeds and splits (`results/quantile-band-fix.md`).
The band is no longer frozen and every absolute measure improves by a lot. The
head still does not beat the trivial baseline, so the kill clause fires.

| bar | verdict | evidence |
|---|---|---|
| 1 coverage in [0.75, 0.88] | **FAIL** (1 of 3) | 0.758 / 0.719 / 0.725 — now *under*-covers |
| 2 pinball beats the shipped head | **PASS** (3 of 3) | 80%, 24%, 24% better |
| 3 pinball beats/matches the offset | **FAIL** (0 of 3) | −34.3% / −3.0% / −5.9% |
| 4 crossing rate exactly 0 | **PASS** | every arm, every staged stage |
| 5 freeze is dead | **PASS** | see below |

K=3, nominal 0.80, 3 seeds:

| dataset | pinball base → fix | offset | coverage base → fix | width base → fix |
|---|--:|--:|--:|--:|
| Brazilian_houses | 0.02992 → 0.00607 | 0.00452 | 0.998 → 0.758 | 0.650 → 0.049 |
| cpu_act | 0.74380 → 0.56782 | 0.55148 | 0.982 → 0.719 | 12.389 → 4.894 |
| elevators | 0.00067 → 0.00051 | 0.00048 | 0.949 → 0.725 | 0.0097 → 0.0044 |

**The freeze is gone and its signature reversed.** cpu_act width now runs
13.08 → 8.73 → 6.01 → 5.03 → 4.54 → 4.27 across rounds 50 → 3000, where the base
was flat at 12.91 from round 1000 onward. Coverage now *falls* with training
(0.815 → 0.705) instead of climbing away from nominal (0.861 → 0.987). P12's
diagnosis is confirmed by construction: remove the worst-case-leaf charge and
the width moves again.

**Against the external reference the change is a clear win.** The PR #48 ledger
(`benchmarks/quantile_head.py`, 3 seeds, `results/quantile-head.md`) improves
from parity to a win at every width — pinball ratio against 19 independent
LightGBM quantile boosters is 0.989 / 0.984 / 0.981 / 0.981 / 0.968 / 0.969 at 5
to 128 features, so the head is now *better* than the per-level baseline it was
only supposed to match, at 3.0x-6.2x their fit time, with crossing 0.0000
against their 0.18-0.21.

#### Why bar 3 still fails: the error flipped sign

The band no longer freezes too wide; it now over-narrows. Leaf values are the
**in-sample** residual quantiles of the rows in that leaf, which are
optimistically tight, and with the budget gone nothing damps that. The freeze
table shows coverage decaying monotonically with rounds on all three datasets
and still falling at round 3000. Early stopping halts it at the *pinball*
optimum, which on these datasets sits below nominal coverage — hence bar 1
failing on the low side.

That also trips the ledger's conformal bar: worst conformalized coverage error
is **2.65 pp against a 2 pp target**. The sign is informative — CQR now has to
*widen* rather than shrink, and the "outer factor at least the inner factor"
monotonization pushes the middle intervals up, so the largest errors sit at the
0.20-0.80 and 0.25-0.75 pairs rather than in the tails.

So the head's remaining deficit is variance, not bias: it estimates a
conditional band per leaf where the offset estimates one marginal band from the
whole calibration fold, and on these three datasets that extra freedom does not
pay for itself. Consistent with P7/P8, which found the rigid arm wins wherever
the conditional spread is close to constant.

#### Status: recorded, not shipped

Branch `fix-quantile-interval-width`, committed and left unmerged. The kill
clause as written says revert, and it is not overridden here — but note the
asymmetry before acting on it: this branch is better than `main` on every
measured axis (pinball on 3 of 3, coverage nearer nominal, LightGBM ratio,
freeze), so reverting restores a strictly worse model. The bar it misses is
"beat a one-line baseline", which `main` misses by 10x more.

The live follow-up is a **new mechanism and needs its own pre-registration**:
shrink each leaf's quantile step toward the pooled band, or fit leaf residual
quantiles out-of-sample, so the band stops over-narrowing. Both address the
in-sample bias directly rather than re-capping the width.

Reusable facts from this pass:

- `tree._project_pinball` read the projected pinball gradient out of a suffix
  table by **binary-searching the row's PIT rank**, which is valid only while
  every row of `F` is sorted. That precondition was supplied by the very budget
  being removed. It is now an explicit per-channel scan, and
  `test_projected_gradient_makes_no_ordering_assumption` pins it against the
  dense definition on a deliberately crossing `F`. Any future change that
  relaxes an ordering invariant should grep for readers of it first.
- Bit-identity held exactly where predicted: 81 of 89 arrays unchanged, the 8
  that moved being the two `mq3` families in full (`pred`, `n_trees`,
  `valid_hist`, `imp` — the tree structures change, not just the output).
  `mae`, `mae_w_sub`, `quantile` and `quantile_w` share the same leaf kernels at
  K=1 and are untouched, which is what licenses skipping a `--decide` run.

## Where to go instead

The two untested axes are **re-purposing, not tuning**, and neither is a search
over a flat surface: the leaf-loss swap (fit structure under squared error,
refit leaves for quantile / Huber / Poisson — the quantile leaf kernels already
exist) and sample or class weights applied at the leaf statistics. Both change
what the model is *for* at replay cost, which is the shape the 5.2x actually
suits. Leaf damping is a near-relative of `reg_lambda` and should be expected to
be flat too.

---

**Original plan follows.**

## Idea

Fit k fold models once. Then sweep leaf-stage hyperparameters by **replaying**
each fold model's tree structures against its own training rows at the new
config — no split search, no re-growing. Fold i's structure never saw fold i, so
the replayed out-of-fold predictions stay honest cross-validation.

Deployed estimator = "structure at the base config, leaves at the winning
config", which is exactly the thing the out-of-fold grid scored.

## What we already have (do not rebuild)

- **The replay engine is shipped.** `tree.replay_oblivious_tree` refits one
  tree's leaf values on new data from a donor structure, and
  `booster.GradientBoosting(replay_donor=(trees, prep))` drives it round by
  round, re-deriving gradients from replayed margins each round. So
  `GradientBoosting(l2_leaf_reg=NEW, replay_donor=(m.trees_, m.prep_)).fit(D)`
  *is* a full leaf-stage replay at `NEW`. Phase 1 of the original spec is done.
- **`staged_predict_raw`** gives the whole M axis from one pass, so every replay
  emits its full out-of-fold-versus-round curve for free.
- **Quantization does not interfere.** `quantize_gradients` affects the split
  search only; leaf values come from `_leaf_values` on float gradients in both
  the grown and the replayed path. A base-config replay should therefore be
  bit-identical to the donor.

## What shadow CV already settled (see memory `project_shadow_cv`)

- Sharing one structure across folds **leaks** — 99 of 100 cells optimistic,
  median bias −14.8%, and on random labels it reported out-of-fold log loss
  0.376, below the 0.693 floor. **This design does not do that** and must keep
  not doing it: every fold's donor is that fold's own model.
- An exact lockstep k+1 fit is only ×1.04 faster than a plain loop. So the fold
  models come from a **thin ordinary loop**. No lockstep machinery.

## Probes, in dependency order

**P0 — correctness.** (a) Replaying a fold model's structures on its own
training rows at the base config reproduces that model bit-identically.
(b) Random-label canary: no `(config, M)` cell may beat chance out of fold.

**P1 — headroom. This is the go/no-go.** Does leaf-stage tuning win anything at
all? Run an *exact* cross-validated grid over `(l2_leaf_reg, M)`, pick the
winner, score it on a held-out test set, and compare against the base config.
The PMLB random-search study found broad tuning buys almost nothing that
generalises, and `l2_leaf_reg` was specifically among the knobs that did not
transfer. If the grid is flat, the tool has nothing to select and the project
stops here.

**P2 — fidelity.** Does the replay grid choose the same `(l2_leaf_reg, M)` as
the exact grid, and when it disagrees, what does the disagreement cost on test?
This bounds what the shortcut is worth.

**P3 — speed.** Wall-clock of the replay sweep against the exact grid. Note the
honest prior: the shipped full-data replay refit saved 34.8% of Grinsztajn fit
time, not an order of magnitude, because preprocessing and the per-round
gradient work survive. Expect single-digit multiples, and measure whether
hoisting preprocessing across configs widens it.

## Scope

Sweepable: `l2_leaf_reg`, leaf damping, sample/class weights at the leaf
statistics, leaf-loss swap, and `M` (free on every replay).
Structural, rejected: `depth`, `learning_rate`, `min_child_weight`,
`subsample`, `colsample`, binning.

## Protocol notes

- Pin `learning_rate` explicitly in every arm — `None` re-resolves from the row
  count and would move between arms.
- No early stopping inside the arms; grow all M rounds and read the curve.
- Decision suites are Grinsztajn plus high-card. TabArena is not involved at any
  point, in any form.
