# Pareto program — same accuracy, ~half the fit time (+ multiclass parity)

Self-sufficient handoff (the HIGHCARD_PLAN.md convention): everything needed to
implement this without its authoring session.

## Why (state of the world, 2026-07-15)

The north star is blended strength vs slowdown (`benchmarks/make_pareto.py`).
Today's default point is **99.4 blended @ 7.9x** on Grinsztajn (CatBoost 98.1 @
11.8x, dominated); the pre-cross_features point was 98.7 @ 3.7x. The 2026-07-15
hc-suite baseline reads ChimeraBoost 2.9x vs CatBoost 111.6x with a small real
accuracy deficit there (Brier winrate 86-88% CatBoost, multiclass swept 4/4).

**Where the 7.9x comes from (verified in source, 2026-07-15):** the default fit
is not one boosting run. `sklearn_api.py`:

- Regressor (`~line 1028`): `linear_leaves=None` fits **constant AND linear**
  variants to full early stopping, keeps the val winner; then cross_features
  (`~line 1051`) fits a **third** cross-augmented model and keeps the val
  winner. Default regression fit = **up to 3 full boosting runs**, 2 discarded.
- Binary classifier (`~line 1446`): base fit + cross-augmented fit = **2 runs**.
- Multiclass: 1 run — but it is locked out of BOTH accuracy levers
  (cross_features raises, `~line 1418`; linear_leaves falls back to constant),
  which is consistent with CatBoost sweeping the hc multiclass sets 4/4.

So the biggest available Pareto move is not a new accuracy mechanism; it is
**making the selection protocol cheap** (Track 1, frontier moves LEFT at pinned
accuracy), plus **giving multiclass the levers the other tasks already have**
(Track 2, accuracy on the suite that can now see it).

Hard constraints (unchanged): pure Python (numpy/numba/sklearn); TabArena sealed
in every form; one benchmark at a time; script files, never `python -c`;
decisions on Grinsztajn + hc (sign-tested separately) gated by the OpenML
one-shot; bit-identical refactors must keep the numerical-identity goldens green.

## Track 1 — fit-speed at pinned accuracy (the headline)

### Step 0 — attribution (no library change, ~1 session)

Instrument/extend `benchmarks/profile_fit.py` to answer, per task type and
dataset size on ~6 Grinsztajn sets + 3 hc sets:

1. Wall-time split across the selection fits (const / linear / augmented) vs
   binning vs TS encoding vs tree growth vs leaf estimation.
2. How often each selection actually flips the winner (`linear_leaves_selected_`,
   `cross_features_selected_` are already recorded on the estimator) — the
   Grinsztajn/hc flip RATES bound how much a cheap selector can lose.
3. Confirm the expected shape: regression fit-time ~= 3x binary per tree budget.

Deliverable: a table in this file; pre-registers which sub-lever runs first.

### Step 1 — zero-risk reuse (bit-identical, ship on goldens + timing)

The three fits rebuild shared state from scratch. Reuse across fits within one
`.fit()` call:

- Bin edges + binned base-feature matrix (the augmented fit re-bins every base
  column identically; only the cross columns are new).
- Ordered-TS encodings for cat columns (identical across variants).
- The internal ES split (already shared? verify — same seed implies yes).

Acceptance: goldens green (bit-identical predictions), fit-time table before/
after, no API change. This alone should reclaim a large slice of the 2-3x.

### Step 2 — cheap selection (behavior-changing, full /experiment protocol)

Replace "run every variant to full early stopping" with a raced selection:

- **Race design (preferred):** run variants in lockstep to a small shared budget
  (e.g. first ~25% of `n_estimators` or until val curves separate by a margin
  rule), kill the losers, continue ONLY the winner to full early stopping.
  Tie-handling and margin need tuning on the synth screen first (tier 1), then
  Grinsztajn + hc.
- **Fallback design:** truncated selection fits (fixed small budget) + one full
  refit of the winner. Costs one extra partial fit but is simpler to reason
  about; the winner's full refit keeps final-model quality identical to today
  whenever the truncated selector picks the same variant.
- Judge: accuracy columns must hold (sign tests not unfavorable on EITHER
  decision suite; blended within noise) while the Speed column drops. A
  selector that flips choices on >~10% of sets is suspect — check step 0's
  flip-rate table for how much headroom exists.

Target: default regression ~3 fits -> ~1.3-1.5 effective; binary ~2 -> ~1.2.
Plausible frontier move: 7.9x -> ~4-5x at the same 99.4 blended. That strictly
dominates today's point and re-widens the moat over CatBoost on the chart.

### Step 3 — kernel profiling (opportunistic, bit-identical)

The 2026-07-13 predict work (row-major kernels) took predict to ~LightGBM
parity; fit kernels never got the same pass. If step 0 shows a hot kernel
(histogram build / split scan), give it the row-major/numba treatment. Ship on
goldens + timing only.

## Track 2 — multiclass parity (accuracy; needs Track 1's step 0 numbers first)

Multiclass is the only task type with no accuracy lever, and the hc suite now
makes it decision-visible (4 sets; Grinsztajn has 0; PMLB has 10 for tuning
sanity). CatBoost swept all 4 hc multiclass sets (Brier AND F1).

- **M1 cross_features for multiclass:** the mechanism (staircased numeric
  interactions) is loss-agnostic; the block is only wired for RMSE/binary.
  Port = allow cross columns under the softmax booster + the same val-selection.
  Cost: doubles multiclass fit -> mitigate with Track 1 step 2's cheap selector
  (do NOT ship a 2x multiclass slowdown; sequence after Track 1).
- **M2 linear leaves for multiclass:** K-output leaf ridge is a bigger lift and
  the binary evidence is weaker — hold unless M1 lands and the hc multiclass
  gap persists.
- Screen on synth multiclass slices (tier 1), decide on hc multiclass + PMLB
  multiclass sanity, gate on OpenML one-shot as usual.

## Explicitly NOT in this program

- No new cat-handling levers (cat_smoothing re-killed on hc 2026-07-15; lei,
  one-hot, combos, binning ports all dead — see memory + PAYOFF.md).
- No ensemble defaults (Grinsztajn-killed on merit; opt-in `n_ensembles` ships).
- No giant-data fit scaling work (documented non-goal).
- No north-star formula change (multiclass stays out of blended — Nathan's
  standing decision; revisit only after M1 exists).
- No TabArena anything (sealed; re-read only after ships, `/tabarena`).

## Decision points reserved for Nathan

- Step 2 selector: accept a selection rule that can (rarely) pick a different
  variant than today's full-fit rule? (Bounded by step 0's flip-rate data.)
- Track 2 sequencing: is a multiclass accuracy win worth a multiclass fit-cost
  increase before the cheap selector lands?
- Any default flip ships only with the usual explicit speed-trade sign-off.

## Acceptance

- [ ] Step 0 attribution table committed here (time split + selection flip rates)
- [ ] Step 1 reuse shipped bit-identical (goldens green) with measured fit-time wins
- [ ] Step 2 raced selection through /experiment (both suites + gate) or killed with data
- [ ] Pareto chart refreshed; frontier point dominates 99.4 @ 7.9x or program verdict says why not
- [ ] M1 multiclass cross_features screened + decided on hc, or explicitly deferred
- [ ] Memory + CLAUDE.md updated with verdicts (wins AND kills)
