# SUBSAMPLE re-test — should `subsample<1` become the default post-MVS-speedup?

Date: 2026-07-30. Branch: `mvs-numba-2026-07-30` (= origin/main + ddc2da7, bit-identical).
Protocol: `/experiment` (Tier 1 synth screen → Tier 2 decision run, per-stratum, never pooled).
**Pre-registered before any run launched.**

## Motivation

The 2026-07-17 "subsample = dead axis" verdict (BAGGING_PLAN.md:522-529) had two legs:
1.19–1.21× SLOWER (pure MVS overhead) and Brier −0.65% (measured only inside small
bagged members). Primary deltas were mildly positive (+0.05%/+0.11%). A2_PLAN.md:151-152
had `default+sub08` as a coin flip (11W-10L-15T, p=1.000). ddc2da7 removed the cost leg:
subsample<1 fits are now 1.32–1.44× FASTER than full fits (12 threads, CHANGELOG
[Unreleased]). Untested cell: single-model, quality=3 default, post-speedup.

## Value selection rule (pre-registered)

Screen 0.8 and 0.7 at Tier 1; carry exactly ONE to Tier 2: the value with the better
synth primary-metric median, subject to no broad Brier regression. Tie / both-flat →
0.8 (historical sub08 arm, milder importance weights, direct comparability with A2).
The decision suite never sees more than one arm.

## Mechanism predictions (pre-registered)

1. **Speed:** median fit-time ratio ≤ 0.85 on n≥10k slices; little/no speedup on
   smallest-n (serial `_SMALL_N` dispatch, sampling barely engages).
2. **Strength:** neutral overall median; gains (if any) concentrated in noisy/large-n
   slices (MVS as implicit regularization); losses confined to smallest-n slices.
3. **Brier:** no broad classification regression. Known risk: MVS importance weights
   (≤ 1/subsample) widen gradient dynamic range under default-on quantize_gradients
   (QUANT_PLAN.md:57). Canary slices flat.

## Kill rules

**Tier 1 kill (for BOTH values):** (a) no speedup on n≥10k slices (median fit ratio
> 0.95), OR (b) negative strength median with losses NOT confined to small-n, OR
(c) broad Brier regression. Known synth v1 biases (shallow targets, no entity effects
in cats) — don't over-read cat slices.

**Tier 2 ship gate:** decisive sign test (> half wins) AND non-negative MEDIAN on
Grinsztajn AND on high-card, each stratum independent, never pooled.

**Small-data strata position (pre-registered):** @sus25/@sus50 losses are PREDICTED by
the mechanism (row sampling shrinks effective n where n is already small) and do not by
themselves block a ship if the Grinsztajn and HC gates pass — but a sus median worse
than −0.2% or a decisively negative sus sign test escalates to Nathan (stratum weighting
is his call). Recorded fallback if sus is decisively negative: n-adaptive default
(subsample=1.0 below a row threshold) as a FUTURE experiment, not a patch to this one.
@time is a read-out.

**Bagged position (pre-registered):** if the single-model gate passes, bagged members
PIN at subsample=1.0 unless an Ens8 synth check shows Brier flat AND member speedup
≥ ~1.15× (prior: Brier −0.65% in members, multiclass member speedup only 1.08×).

## Stamp → arm ledger

Appended the moment each run prints its stamp — the saved JSON does NOT record the arm.

| stamp | arm | flags | tier |
|---|---|---|---|
| 20260730-223839 | BASE-SYNTH | `--synth --seeds 3 --models ChimeraBoost` | 1 |
| 20260730-223952 | SUB08-SYNTH | + `--chimera-subsample 0.8` | 1 |
| 20260730-224101 | SUB07-SYNTH | + `--chimera-subsample 0.7` | 1 |
| (never ran) | BASE-DECIDE | killed at Tier 1 | 2 |
| (never ran) | SUB-DECIDE | killed at Tier 1 | 2 |

## Results — Tier 1 (2026-07-30)

Synth screen, 136 datasets, 3 seeds, ChimeraBoost-only, branch mvs-numba-2026-07-30.

| arm | primary W-L-T | primary median | Brier W-L-T (86 clf) | Brier median |
|---|---|---|---|---|
| 0.8 vs base | 63-67-6 | −0.000% | 31-55-0 | −0.730% |
| 0.7 vs base | 59-71-6 | −0.080% | 33-53-0 | −0.351% |

Factor report (0.8 arm): noise_level coefficient t = −0.03 (dead flat — the
implicit-regularization mechanism did NOT show), n>=2000 slice 43-43-2 (dead even —
no large-n concentration), canaries flat (0-1-2, −0.027%). Every sign test FAIL.

## VERDICT: KILLED at Tier 1 (kill rule c, both values)

Broad Brier regression on both arms (55/86 and 53/86 classification losses, decisive),
primary flat-to-negative, and the pre-registered mechanism (gains concentrated in
noisy/large-n slices) is absent — the noise coefficient is zero. The decision tier
never ran; the decision suite saw no subsample arm.

**What this adds over the 2026-07-17 verdict:** the Brier regression reproduces on
SINGLE models at the quality=3 default — it was never a small-bagged-member artifact.
With the cost leg removed (ddc2da7: subsample<1 fits 1.32–1.44× faster), subsample<1
still loses on strength. The dead axis is dead on strength grounds alone; the speedup
ships regardless (bit-identical, benefits opt-in users).

**Open thread (future experiment, not this one):** whether the Brier hit is MVS itself
or the MVS-weight × quantize_gradients dynamic-range interaction (QUANT_PLAN.md:57).
Diagnosing needs a quantize-off arm (harness has no off switch today).
