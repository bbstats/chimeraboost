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

### Step 0 RESULTS (2026-07-16)

Measured: `python benchmarks/profile_fit.py --attribution --seeds 3 --out
pareto-step0` (full records incl. per-variant val curves in
`benchmarks/results/pareto-step0.{json,md}`, gitignored). Default estimator,
same loaders/splits as the decision suites, warm JIT.

Time split (secs are means over 3 seeds; phases are % of estimator fit):

| dataset | task | n_train | fit_s | const/base | linear | cross | grow% | prep%* | other% |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gr:reg_num/cpu_act | reg | 6144 | 0.8 | 0.2 | 0.2 | 0.3 | 82.9 | 5.5 | 11.6 |
| gr:reg_num/diamonds | reg | 37500 | 1.9 | 0.5 | 0.4 | 0.9 | 86.9 | 4.4 | 8.7 |
| gr:reg_cat/nyc-taxi | reg | 37500 | 5.5 | 1.4 | 1.2 | 2.9 | 92.1 | 1.0 | 6.9 |
| gr:clf_num/MagicTelescope | bin | 10032 | 0.6 | 0.2 | — | 0.3 | 73.6 | 6.1 | 20.3 |
| gr:clf_num/Higgs | bin | 37500 | 1.7 | 0.7 | — | 1.0 | 77.1 | 8.8 | 14.1 |
| gr:clf_cat/road-safety | bin | 37500 | 2.6 | 1.2 | — | 1.4 | 83.7 | 2.9 | 13.4 |
| hc:kick | bin | 54737 | 2.2 | 0.7 | — | 1.3 | 51.2 | 32.9 | 15.9 |
| hc:wine-reviews | reg | 75000 | 2.0 | 1.0 | 1.0 | 0** | 66.6 | 21.2 | 12.2 |
| hc:okcupid-stem | multi | 38091 | 1.8 | 1.7 | — | — | 48.6 | 18.6 | 32.8 |

\* prep = ordered-TS encode + binning + cat-code mapping + val-set transform,
summed across all variant fits. \** wine-reviews has 1 numeric column → no
cross pairs. okcupid's other% is softmax grad/hess + per-round val eval.

Rounds actually run per fit (min/mean/max over the panel): const 338/601/977,
linear 220/391/608, cross 152/320/786, binary base 114/247/454, multiclass
93/99/105.

**Selection flip rates:** linear_leaves selected 8/12 regression fits;
cross_features selected 20/21 eligible fits. The extra fits are not
tie-breakers — the augmented model nearly always IS the shipped model, so
"run every variant to full ES" mostly buys confirmation, not models.

**Race preview** (offline, from the recorded val curves: would truncating both
variants at k rounds pick the same winner as today's full fits?):

| selection | k=50 | k=100 | k=200 | k=500 |
|---|---|---|---|---|
| linear-vs-const | 8/12 (max regret 0.51%) | 8/12 (0.51%) | 8/12 (0.51%) | 12/12 |
| cross-vs-plain | 20/21 (4.25%) | 21/21 | 21/21 | 21/21 |

Regret = full-run best-val loss conceded on a mispick, % of the better
variant's best val loss (pre-cross for the ll stage; suite impact TBD).

**Projected fit speedup** (from the recorded per-round costs; assumes cost
linear in rounds and truncated picks match today's — validated above):

| design | k=100 | k=200 | k=500 |
|---|--:|--:|--:|
| fallback: all selection fits @k, cross full, refit plain winner only if cross loses | 1.38x | 1.20x | 0.99x |
| race: kill loser @k, winner continues to its own ES | 1.27x | 1.16x | 1.04x |

(Time-weighted panel totals; per-dataset fallback@k=100 ranges 1.28–1.69x on
Grinsztajn, 1.00–1.40x on hc; multiclass untouched at 1.00x.)

**Pre-registration (what the data says runs first):**

1. **Step 2 goes first**, with the FALLBACK design at k≈100 rounds (5% of the
   2000 budget), not the race design — the flipped 20/21 cross rate means the
   fallback's "never run a plain variant to full ES unless cross loses"
   exploits the structure harder (1.38x vs 1.27x projected). Known risks to
   screen on synth tier 1 before /experiment: (a) cross pairs will now come
   from a 100-round fit's feature_importances_ — measure pair overlap vs
   full-fit pairs first; (b) the ll stage mispicks 4/12 at k=100 (≤0.51% val
   regret pre-cross) — check it stays noise-level through the cross stage.
2. **Step 1 (reuse) is demoted to opportunistic hygiene.** Measured redundant
   prep across variant fits is ~3–6% of fit on Grinsztajn, ~10–17% on hc —
   real but NOT "a large slice of the 2-3x" (that expectation is refuted:
   73–92% of fit is tree growth, which is per-variant, not shared state).
3. **Step 3 (grow kernel) is the only lever bigger than selection** — after
   step 2 lands, profile inside build_oblivious_tree.
4. Track 2 costing note: the multiclass base fit is short (~100 rounds) but
   ~33% of its time is non-kernel overhead; M1's extra cross fit under the
   k=100 selector adds roughly one short fit, not a 2x.

Shape check from the plan: regression = 3 full boosting runs and binary = 2
confirmed; but per-tree cost is ~2-3x HIGHER for binary (3.6 vs 1.2–1.7
ms/tree), so regression wall time is NOT 3x binary — the two roughly cancel.

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

### Step 2 RESULTS (2026-07-16) — SHIPPED AS DEFAULT (Nathan's sign-off, same day)

Nathan approved the default flip (selection_rounds=100), explicitly waiving
the strict non-negative-gate rule (the burned gate's mean decomposes to one
sklearn pure-linear toy + the since-fixed asymmetric-race bug) and accepting
the ll-mispick tail. `selection_rounds=None` = old behavior; ablation arm
`--chimera-full-selection`. pareto.png refreshed: **99.4 @ 6.0x**, every
accuracy column #1, CatBoost dominated. Evidence below is the record.

Implemented as `selection_rounds` (opt-in, default None = old behavior; harness
`--chimera-selection-rounds`). Final design after two protocol-driven fixes:
auditions capped at k=100; an audition that early-stops before the cap is
reused as the full fit (no refit); the cross decision is a SYMMETRIC race
(both candidates judged on their first k rounds; trailing aug killed at k) —
the asymmetric full-vs-capped comparison first implemented let a worse aug
model steal selections (caught by the OpenML gate on synthetic_reg, −12.8%).

Evidence (all at k=100, 3 seeds, controls tied canonical baselines 59/59+14/14):
- Synth screen: 22W/9L/180T, +0.010% — accuracy flat, guards bit-identical.
- Grinsztajn: fit **351→235 s = 1.50x**; headline **Speed 7.9x→6.0x at
  blended ~99.4** (RMSE 99.5→99.4, F1 99.8 flat, Brier 99.2→99.1, Calib
  better). Sign test 8W/22L/29T mean −0.087% — loss-leaning, all deltas small
  (worst real: cpu_act −1.44%).
- HC: fit 1.11x, slowdown 2.9x→2.4x, 3W/1L/10T −0.068%, columns flat ±0.2pp.
- OpenML gate (**burned pre-fix**): 4W/6L/26T mean −0.396%, dominated by the
  since-fixed compound bug case. Post-fix the same set still loses ~−10.6% to
  the residual ll-audition mispick, so a re-run would still be negative-mean
  on the strict reading.

Known residual (characterized, no cheap fix): the const-vs-linear race
genuinely crosses late on ~1/3 of regression selections; no k=100 margin rule
separates them (measured on step-0 curves — overlap is total), and k_ll=500
fixes fidelity but collapses the speedup to a projected 1.11x. Real-data cost
~0.5–1.5%/set on the susceptible minority; adversarial pure-linear synthetics
can lose ~10%.

Nathan's decision points: (a) ship opt-in only (protocol-clean; headline
unchanged), (b) flip default to 100 — requires waiving the strict
non-negative-gate rule (gate mean excl. the sklearn linear toy ≈ −0.06% =
noise; 26/36 exact ties) and accepting the ll-mispick tail, (c) kill.

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

- [x] Step 0 attribution table committed here (time split + selection flip rates) — 2026-07-16
- [x] Step 1 reuse — DEMOTED by step 0 (3–17% upside), then SHIPPED anyway as
      B-prep 2026-07-17 (bagging program; intra-fit prep reuse, bit-identical,
      hc fits ~12% cheaper; released 0.16.1). Closed.
- [x] Step 2 raced selection through /experiment (both suites + gate) — 2026-07-16, SHIPPED as default (Nathan's sign-off); evidence in "Step 2 RESULTS"
- [x] Pareto chart refreshed — 2026-07-16: **99.4 @ 6.0x strictly dominates 99.4 @ 7.9x**
- [x] M1 multiclass cross_features — **SHIPPED 2026-07-17, released 0.17.0**
      (full record: M1_PLAN.md; all three registered bars passed; TabArena
      Lite re-read flat at 1267)
- [x] Step 3 grow kernels — **own pre-registered program opened 2026-07-18:
      GROW_PLAN.md** (Nathan approved as next; Phase 0 profiling first);
      **CLOSED same day: acceptance NOT met** (bit-identical levers sum to
      ~1% suite-level; small-fit wins are real but suite sums are large-n
      weighted). Full attribution table + per-lever verdicts in GROW_PLAN.md.
      Settled: the fused scatter+scan (37-58% of fit) is the ONLY remaining
      double-digit fit-side object, and it is Phase-2/FP-drift class.
      L-pytree (single-launch level kernel), identity-certified 73/73,
      was subsequently MERGED and released in 0.18.0 — it is the live fit
      path (`_build_split_descend`). Nothing is left on a side branch.
- [x] Memory + CLAUDE.md updated with verdicts (wins AND kills) — ongoing per
      program close; algorithm-history memory current through M1

---

## The dual re-read (2026-08-10) — reopening Track 1

**Why this section exists.** The north star is a frontier with two coordinates,
but every kill in this project's ledger was recorded against one of them. A
change that is flat on strength and cheaper pushes the frontier exactly as far as
one that is cost-neutral and stronger, and we have never gone back to read the
"flat" verdicts that way. This is a one-pass re-read of the recorded kills asking,
for each, what it says on the axis it was **not** judged on. It cost no benchmark
time. The rule it comes from is now in the `/experiment` skill: every verdict
states both axes.

Two of the kills turn out to have been read on both axes already and are properly
closed — subsample (faster after the MVS numba work, but Brier 31W-55L median
−0.730%, so the trade is measured and bad) and the grow kernels (bit-identical by
construction, so strength cannot move; killed at their measured ceilings). The
rest divide as follows.

### D1 — The audition: per-leg budget is empty, the decision RULE is not

Sel25 died on strength (gr regression 6W-20L-10T, p=0.009 at the rung-3 default)
while measuring **8.9%** of total Grinsztajn fit on the cost axis, and its own
record named a **per-leg** budget as the way back in — long `k` for the
const-vs-linear race, short for the cross race. Two cheap checks, both costing no
benchmark time, say that direction is empty and name a better one.

**The per-leg budget is empty, from source plus the recorded flip rate.** The
cross race is not symmetric with the const-vs-linear race. `_stop_if_behind` kills
a *trailing* augmented fit at the budget but lets a *leading* one run to its own
early stop — and the augmented candidate wins 20 of 21 selections (step-0
pre-registration above). So on 20 of 21 datasets the augmented fit runs full
whatever `k_cf` is, and shortening it saves almost nothing. Lengthening it is
barred in the other direction: the cross comparison comes from
`min(aug.valid_history_[:k]) < base_best`, where `base_best` is the *capped*
audition's best, so raising `k_cf` above `k_ll` reintroduces exactly the bias the
symmetric-budget rule was written to remove (`sklearn_api.py:2004-2011`). A
per-leg budget can therefore only make the cheap leg cheaper, which it is not.

**The rule shape is where the room is** — `benchmarks/probe_audition_rule.py`,
which replays decision rules against the full validation curves recorded in
`results/pareto-step0.json`. It reproduces the recorded 4/12 mispicks at k=100
exactly, so the replay matches the estimator. Findings:

| k | shipped rule | worst validation regret | dataset carrying it |
|--:|---|--:|---|
| 25 | 5/12 mispicks | **1.452%** | `gr:reg_cat/nyc-taxi-green-dec-2016` |
| 50–200 | 4/12 | 0.511% | `gr:reg_num/cpu_act` |
| 300 | 3/12 | 0.511% | |
| 500 | 0/12 | 0.000% | |

Three things fall out.

1. **The k=100 plateau is real and flat.** From 50 to 200 nothing changes at all —
   not the mispick count, not the regret, and not under any of the four rule
   shapes tried (best-so-far, last value, tail mean, slope-extrapolated). At
   k=100 the information simply is not in the curves, which generalises B14 from
   margin rules to every rule of this family.
2. **The mispicks cost almost nothing where they happen.** Median regret is
   0.000% and the worst single case is 0.511%. The 4/12 rate reads like a
   fidelity problem and mostly is not one.
3. **The exception is exactly the dataset that killed sel25.** At k=25 the worst
   regret triples to 1.452% on `nyc-taxi-green-dec-2016` — the dataset sel25's
   own record names as its cheapest repro, losing in all four of its appearances
   (−1.70% / −4.18% / −1.92% @sus25 / −2.94% @sus50). The offline replay fingers
   the same dataset the decide-tier run did, and supplies the mechanism: a
   short-audition mispick that the rung-3 refit then amplifies (B2).

And the opening: at k=25 the **tail-mean** rule reads 4/12 at 0.511% worst regret
— the same fidelity as the shipped rule at k=100, at a quarter of the budget. If
that holds up, sel25's 8.9% is available after all, and it was the rule rather
than the budget that was wrong. One race carries the whole difference at n=12, so
this is a pointer and nothing more (`GATE_ROBUSTNESS.md` #2).

Queued as E1 in `SELECT_PLAN.md`, **with a curve-corpus widening first** — raising
n from 12 costs one attribution run, not a decide run, and this is far too thin to
gate on. Barriers it owes an argument about: B1, B2, B12, B14.

### D2 — Cheapening the audition revives everything killed on audition cost

A2 passed every strength bar and failed only on cost: oracle ceiling +2.587%,
validation-selected +0.522% at a 70% win rate (21W-9L-6T, p=0.043), and a
projected **1.90×** fit multiple against a 1.35× bar. Its stated reason —
"a k=100 audition is nearly a whole fit whenever the full fit is short" — is a
statement about audition *pricing*, not about config portfolios.

So D1 and A2 are coupled, and the coupling is the point: the audition cost model
is a shared denominator under several dead ideas. If D1 lands, A2's kill is worth
re-deriving at the new price rather than re-running blind. **Do not re-run A2
before D1 reports** — that is the ordering the dual read buys us.

### D3 — The patience default's rationale is stale below 20k rows

`early_stopping_rounds=50` was chosen because "50 beat 10 on 25 of 34 datasets…
because lr=0.1 keeps improving past a 10-round plateau" (`sklearn_api.py`), and
the synth backtest records patience 300 as flat. So the axis reads: 10 loses, 50
ships, 300 is flat.

Since 0.30.0 the small-data learning rate is no longer 0.1 — it fades to 0.07
below 20k rows. A lower rate improves for *longer*, so the measurement behind the
default no longer describes the regime where the rate now differs. This is not a
speed lever (it points at more patience, not less); it is an unmeasured default
whose stated justification has expired. Open question, not a queued run.

### D4 — Replay is a screening instrument in search of a use

LEAFTUNE killed `tune_leaves`, but the mechanism it built round-trips
bit-identically and costs a median **5.2× less per configuration** than
re-growing. Its fidelity failure (agreed with an exact grid on 2 of 8 datasets;
gave up 6.8% on the one set where the parameter mattered) makes it unfit to
*select*, and it does not touch the shipped fit path either way — so this is
research-tooling speed, not product Pareto. Worth remembering the next time a
study needs a wide sweep; not worth a program.

### What this leaves

D1 is the only queued run. D2 is a scheduling constraint on a future one, D3 an
open question with no owner, D4 a note. The barriers each of these must clear are
in `benchmarks/BARRIERS.md`.
