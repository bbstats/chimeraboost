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
bound over all leaves and rounds, or made optional. **SHIPPED as exactly that
fix — PR #62, released in 0.28.0**: ordering is now imposed per row on the
delivered predictions by monotone rearrangement, the narrowing budget is gone,
and `crossing_rate` is still exactly zero. Pinball loss improved 80%/24%/24%
on the three sets measured.

*Separate, minor, confirmed — since FIXED:* `_auto_min_child_weight` returned
11 for the grid [0.1, 0.5, 0.9] where 10 is intended, because `1.0 - 0.9` is
slightly under 0.1 in binary and the reciprocal rounds past the integer. The
19-level default grid (20), [0.25, 0.75] (4) and [0.01, 0.99] (100) were all
correct. The reciprocal is now snapped to a whole number when it is within
rounding noise of one (`chimeraboost/quantile_api.py:58-66`); genuine
fractions such as the 3.33 of a [0.3, 0.7] grid are far outside the tolerance
and still round up.

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

### Result: mechanism confirmed and fixed; bars 1 and 3 FAIL; shipped anyway (see Status)

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

#### Status: SHIPPED (Nathan, 2026-07-31) — bar 3 reclassified as a research target

> "if we are winning we shouldn't kill it"

The kill clause is deliberately overridden, and the reasoning is recorded here
because overriding a pre-registered bar is exactly the move that needs an audit
trail.

Bar 3 asked the head to beat a rigid location shift. That is the right question
for *"is a conditional quantile model worth building at all"* — the question
P9-P13 were asking — but it is the wrong question for *"should this change
replace what users currently have."* On the shipping question the change wins
every comparison there is: against the shipped head (pinball better on 3 of 3,
by 80% / 24% / 24%), against the external per-level reference (LightGBM, now
beaten at all six widths where it previously only matched), and on the defect
itself (the freeze is gone). The baseline it still loses to, `main` loses to by
roughly 10x more, so honouring the kill clause would have restored a strictly
worse model in the name of a bar neither version clears.

The bar is therefore reclassified: **beating the rigid offset stays an open
research target for the head, not a gate on individual improvements to it.**
Bars 1 and 3 remain FAILED and are not to be quietly restated as passes.

Shipped with the raw-grid under-coverage documented rather than hidden
(`docs/quantiles.md`, CHANGELOG): a nominal 80% interval delivers about 72-76%,
and `conformalize=True` is the supported route to a coverage guarantee. Note
this is a *change in the direction* of the miscalibration, not its removal —
users reading a raw interval as a guarantee were previously safe-but-useless and
are now optimistic, which is the more dangerous failure and the reason it is
called out in both places.

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

## P15 — spread smoothing, pre-registered 2026-07-31 (branch `quantile-spread-smoothing`)

Written **before** anything was measured. This is the follow-up P14 called for:
attack the in-sample bias in the leaf values rather than re-capping the width.

**The defect.** A leaf's value at level tau is the empirical tau-quantile of the
residuals of the rows *in that leaf*, measured on those same rows. That is
optimistically tight, and the smaller the leaf the tighter it is: an m-row leaf
cannot resolve a 0.95 quantile at all once m is small, so the estimate collapses
toward the middle of the leaf's residuals. Nothing damps it, so the band narrows
with every round and coverage decays monotonically (P14 result table).

**Change.** Per round, after the leaf refit and **before** the values enter `F`,
shrink each leaf's *spread* toward the pooled band by an amount that depends on
how much data the leaf actually has. With `v_l` the leaf's K-vector, `p` the
pooled band (the same quantile computation over all rows, one leaf), `c(.)` the
interpolated tau=0.5 point of a vector, and `n_l` the leaf's row mass:

```
lambda_l = k0 / (n_l + k0)
v'_l[k]  = c(v_l) + (1 - lambda_l) * (v_l[k] - c(v_l)) + lambda_l * (p[k] - c(p))
```

Four properties, each chosen against a recorded failure:

- **Convex, never a floor.** The band stays free to be *narrower* than pooled,
  which trap 1 in `QUANTILE_PLAN.md` says any pooled-band mechanism must
  preserve — a sum of clipped-to-monotone vectors can never narrow, and that is
  what diverged in PR #48. Nothing is sorted and `F` is never touched.
- **Location-preserving**, so it is exactly the identity at K=1 (the deviations
  from the centre are all zero). The scalar MAE/Quantile paths share these leaf
  kernels at K=1; this keeps them provably unreachable.
- **Adaptive, not a knob.** P11 established that no knob on this head moved
  coverage. `lambda_l` is a function of leaf mass, so small leaves — precisely
  the ones whose tail quantiles are unresolvable — shrink hardest, while a leaf
  with plenty of rows keeps its own spread.
- **The blend is NumPy on `tree.values` in the booster**, not a kernel change,
  so the four leaf kernels and their bit-exact oracles are untouched and no new
  warmup entry is owed.

`k0` is argued, not tuned: `_auto_min_child_weight(taus)` = ceil(1 / min(tau_0,
1 - tau_K)), the constant this head already uses for "rows needed before the
extreme channel means anything" (20 on the default grid). A leaf sitting at the
minimum admissible mass gets lambda = 1/2. The probe sweeps k0 at half and twice
that value as **diagnosis of the mechanism's shape**, not to pick a default.

**Arms** (`probe_quantile_band.py --tag p15`, same 3 datasets, seeds 0-2, K3 and
K19 grids as P14): `head` (the shipped PR #62 head, unchanged), `head+sm` at the
argued k0, `head+sm-half` and `head+sm2` for the sweep, `head+sm+CQR`, and the
`offset` baseline. Freeze section gains a `head+sm` row.

**Bars.** Ship gates, judged at the argued default `head+sm`, all required for
the default to flip on:

1. **Coverage** without `conformalize` inside [0.75, 0.88] at nominal 0.80 on
   3 of 3 (fix: 0.758 / 0.719 / 0.725 — P14's failed bar 1).
2. **The decay is dead**: freeze-section coverage at round 3000 is no more than
   1 pp below its round-300 value on 3 of 3, and width at 3000 still differs
   from round 50 by more than 1% (the P14 freeze stays dead — this must not be
   bought by re-freezing the band).
3. **No pinball give-back**: within 2% of the shipped PR #62 head on every
   dataset, both grids. This is the "compare against what users have" gate that
   P14's status entry established as the right shipping question.
4. **Crossing rate exactly 0.0** in every arm, every path, every staged stage.
5. **The conformal ledger is restored**: worst conformalized coverage error
   ≤ 2 pp (fix: 2.65 pp, FAIL).
6. **Unchanged obligations**: the identity snapshot moves only the two `mq3`
   families and every scalar family is byte-identical (which again licenses
   skipping `--decide`); the full suite green, including
   `test_narrow_regions_are_actually_narrower` (tight/wide ratio < 0.35 — the
   test that bounds how hard this may pull toward a global band).

**Research target, not a gate.** Pinball beats or matches the rigid offset on
3 of 3, currently −34.3% / −3.0% / −5.9%. Per the P14 status entry this is an
open research target for the head and never a gate on improvements to it; bars 1
and 3 of P14 remain recorded as FAILED. Report the movement either way.

**Kill clause.** If any ship gate fails at every probed k0, mechanism A is
falsified for this defect. Record the negative here and **remove the parameter**
— no dead knobs. The fallback, which would need its own pre-registration, is
mechanism B: fit the leaf residual quantiles out-of-sample on a dedicated third
fold. B was not built first because it needs a fold that is neither the
calibration nor the early-stopping fold (reusing the ES fold would let stopping
score rows that set the leaf values), it cuts rows-per-leaf on a head whose
remaining deficit P14 diagnosed as *variance*, and it breaks the recorded
invariant that leaf values see exactly the rows the split search saw
(`booster.py`, the MVS row-selection comment).

### Result: FALSIFIED at tier 1. Kill clause honoured, parameter removed.

**Sweep widened before measuring** (recorded because it changes the
pre-registration): the plan above swept k0 at half and twice the argued 20.
Before running anything, that was judged too narrow to be diagnostic. With 16
leaves over 20 000 rows a leaf holds ~1250 rows, so `lambda = k0 / (n_l + k0)`
is ~0.016 at k0 = 20 — if the optimism is driven by the split *selection*
rather than by within-leaf sample size, a sweep spanning a factor of four could
not tell a wrong shape from a too-small constant. The sweep was therefore
widened to four orders of magnitude, k0 in {20, 100, 500, 2000, 10 000,
100 000}, up to lambda ~0.99 (the fully-pooled limit). No measurement had been
taken at that point.

Mechanism screen, synthetic heteroscedastic data (8 features, conditional
spread 0.2 against 2.0, 4000 train / 4000 test, 300 rounds, 3 seeds, grid
(0.1, 0.5, 0.9), nominal coverage 0.80):

| k0 | lambda at a 250-row leaf | test coverage | **train coverage** | width |
|--:|--:|--:|--:|--:|
| 0 (off) | 0.00 | 0.7555 | 0.8004 | 2.697 |
| 20 | 0.07 | 0.7586 | 0.8013 | 2.685 |
| 100 | 0.29 | 0.7674 | 0.8021 | 2.707 |
| 500 | 0.67 | 0.7702 | 0.8029 | 2.703 |
| 2 000 | 0.89 | 0.7772 | 0.8023 | 2.581 |
| 10 000 | 0.98 | 0.7888 | 0.8037 | 2.333 |
| 100 000 | 1.00 | 0.7882 | 0.8019 | 3.055 |

**The train-coverage column is the whole finding.** It sits at nominal 0.80 at
every k0, including fully pooled, while test coverage never gets there. The
deficit is entirely the gap between in-sample and out-of-sample residuals — and
the pooled band is computed from *in-sample* residuals too, so it carries the
same bias. Shrinking toward a target that shares the error cannot remove it.
The mechanism only ever redistributes spread *between* leaves; it has no route
to the common component, which is all of it. A homoscedastic control moved even
less (0.7576 to 0.7690 across the same sweep), as expected when there is no
between-leaf variation to redistribute.

What the residual gain at extreme k0 costs, same data:

| k0 | test coverage | pinball | tight/wide width ratio (true 0.10) |
|--:|--:|--:|--:|
| 0 (off) | 0.7555 | 0.28507 | 0.129 |
| 20 | 0.7586 | 0.28454 | 0.127 |
| 500 | 0.7702 | 0.28592 | 0.137 |
| 2 000 | 0.7772 | 0.28733 | 0.166 |
| 10 000 | 0.7888 | 0.30820 | **0.420** |
| 100 000 | 0.7882 | 0.33404 | **0.906** |

The three points of coverage available are bought by pooling the band across
leaves — the ratio walks from 0.129 (near the true 0.10) to 0.906 (a flat
global band), which is the head ceasing to be conditional at all. That breaks
ship gate 6 (`test_narrow_regions_are_actually_narrower`, ratio < 0.35) at the
same k0 where gate 3 breaks too (pinball +8.1% at k0 = 10 000, +17.2% at
100 000, against a 2% allowance). Coverage never reaches the gate-1 window on
this data even in the fully-pooled limit. **Every k0 that helps coverage fails
two other gates; no k0 passes all six.** Killed per the clause; the
`spread_smoothing` parameter is removed rather than left in as a dead knob.

Cost of the negative: three short screens, no `--decide` run, no probe run. The
tier-1 screen did its job — the story did not show up, so nothing downstream
was spent on it.

**Generalizable.** A shrinkage estimator can only correct the part of the error
that *varies across* the units being shrunk. Diagnose which component you have
before designing the correction: here, printing in-sample alongside
out-of-sample coverage answered it in one table and would have falsified the
design on paper. In-sample-vs-out-of-sample gaps need an out-of-sample estimate;
there is no arrangement of in-sample quantities that fixes them.

Note what this implies about the *product* question, as distinct from the
research one: a uniform in-sample/out-of-sample gap is exactly what a global
multiplicative correction from a held-out fold repairs, and that is
`conformalize=True`, already shipped and already documented as the route to a
coverage guarantee. The remaining open item for the raw grid is pinball against
the rigid offset, which is a sharpness question, not a calibration one.

## P16 — out-of-sample leaf quantiles (pre-registration DRAFT, not started)

The fallback P15 named, now with P15's evidence behind it: the defect is that
leaf values are residual quantiles of the rows that chose the split, so the
only fix is to estimate them on rows that did not. Structure from one set of
rows, leaf values from another — "honest" trees in the Athey-Imbens sense,
applied to the leaf quantile rather than the leaf mean.

Not started, and it needs a real pre-registration before it is. The three
things that pre-registration has to answer, all of them recorded costs rather
than open questions:

1. **Which rows.** Neither existing fold is available: the calibration fold
   must stay pristine for conformalization, and using the early-stopping fold
   would let stopping score the rows that set the leaf values. So it needs a
   third split, or a per-round row partition (cheaper, and closer to what the
   MVS draw already does).
2. **Variance.** P14 diagnosed the head's remaining deficit as variance, not
   bias. Halving the rows per leaf to buy honesty makes that worse; the K = 19
   grid, whose extreme channels are the noisiest, is where this will show
   first. Whether the bias removed exceeds the variance added is the whole
   experiment, and it is genuinely uncertain.
3. **A recorded invariant breaks.** `booster.py` states that leaf values see
   exactly the rows the split search saw (the MVS row-selection comment). This
   mechanism contradicts it deliberately, so the comment changes with the code
   and the reason lands in the same commit.

Bars would carry over from P15 unchanged (coverage window, no pinball
give-back against the shipped head, crossing zero, the conformal ledger, the
identity-snapshot scope), with the rigid-offset comparison still a research
target and not a gate.

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
