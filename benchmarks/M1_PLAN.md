# M1 — multiclass cross features (bring the selection machinery to softmax)

Self-sufficient handoff (BAGGING_PLAN.md convention). Approved by Nathan
2026-07-17 as the next program. Goal: extend the shipped numeric cross-feature
machinery (pair proposal from base-fit importances + raced validation
selection) to multiclass classification — today the multiclass path has ZERO
selection machinery and multiclass is CatBoost's last crown (hc baseline:
CatBoost wins all 4 multiclass sets vs the single model, 100% multiclass Brier
winrate per HIGHCARD_PLAN). Everything below was registered BEFORE any
library change or benchmark run.

## State of the world (2026-07-17, canonical hc run `20260717-115025`)

hc multiclass, F1 macro (single model vs CatBoost):

| set | Chimera | CatBoost | delta | CB fit/ours |
|---|--:|--:|--:|--:|
| okcupid-stem | 0.5798 | 0.5816 | −0.0018 | 47x |
| Traffic_violations | 0.8524 | 0.8557 | −0.0033 | 39x |
| cjs | 0.9992 | 1.0000 | −0.0008 (near-ceiling) | 105x |
| eucalyptus | 0.6592 | 0.6732 | −0.0140 | 73x |

Ens8 closes some of it (eucalyptus 0.6871 BEATS CatBoost) but the single
model loses all four. Multiclass F1/Brier are report-only columns — the
blended north star (reg + binary) is untouched by anything M1 does, on both
suites. The win condition is the multiclass slices themselves plus "CatBoost's
last crown" narrowing; the headline gr chart cannot move by construction.

## Mechanism, and why it should transfer

Oblivious trees approximate numeric interactions with a depth-limited
staircase; a diff column makes the `x_i < x_j` boundary one split, a prod
column captures multiplicative structure. This shipped for regression +
binary (2026-07-13, Pareto 99.4 @ 7.9x; synth v1 ablation: removing it =
−0.94% overall, **−3.30% on the interaction-depth≥2 numeric slice**). The
argument is about feature-space geometry, not the loss: a softmax class
margin wants the same boundaries. Nothing in the multiclass objective breaks
the mechanism; the per-class trees just share the augmented columns.

Code facts (why this is cheap to build):

- `cross_pairs` already lives on `_BaseBooster` and flows into
  `FeaturePreprocessor` — `MulticlassBoosting` can already CONSUME cross
  columns; nothing proposes or selects them (`sklearn_api.py` raises
  `NotImplementedError` on explicit `cross_features=True`, auto silently
  skips).
- `feature_importances_` already aggregates rounds-of-K-trees → pair
  proposal (`_cross_candidate_pairs`) works unchanged.
- The raced-selection callbacks (`_stop_after`, `_stop_if_behind`) run on
  `valid_history_`, which the multiclass fit records (softmax logloss).
- Missing pieces: `MulticlassBoosting.fit` has no `prep_cache` (B-prep's
  intra-fit prep reuse is scalar-only — `_prep_matrices` lives on
  `GradientBoosting`), and the classifier fit has no multiclass selection
  block.

## Treatment surface (measured before design lock — the honest part)

Eligibility = the SAME gates as binary, kept at parity by design:
post-ES-split train n ≥ `CROSS_MIN_SAMPLES`(2000) and ≥ 2 numeric columns.
Surveyed with `m1_baseline_facts.py` (synth) and `m1_hc_eligibility.py` (hc):

- **Synth screen (tier 1): 34 multiclass sets of 136; 15 eligible.** The 3
  multiclass canaries (017/117/317) are all INeligible → they stay pure
  canaries. The 19 ineligible multiclass + all 102 reg/binary sets must be
  EXACT TIES (untouched code paths).
- **hc (tier 2): 1 of 4 eligible — okcupid-stem only** (50,789 rows but just
  2 numeric columns → exactly 1 pair → 2 cross columns; weak surface).
  Traffic_violations has 1 numeric column; cjs misses the row gate (1,677
  post-split vs 2000); eucalyptus has 441. The biggest CatBoost gap
  (eucalyptus) is out of M1's reach — recorded, not hidden.
- **Grinsztajn: zero multiclass — a pure identity canary for M1.**
- **OpenML gate: 9 multiclass of 29; 4 eligible** (optdigits, satimage,
  pendigits, letter — all-numeric, n_fit 3.3k–12k). vehicle/segment fail the
  row gate; car/splice/nursery are all-categorical. The other 25 sets must
  exact-tie.
- PMLB has 10 multiclass sets but is HP-tuning only — NOT used (M1 tunes no
  HPs).

So the real-data decision surface is thin: okcupid-stem + the gate's 4. The
decision weight deliberately sits on the tier-1 eligible slice (15 sets,
mechanism-attributed) with the gate as the independent real-data check —
the same evidence shape that shipped the original cross_features, minus a
Grinsztajn read that cannot exist for multiclass.

## Pre-registered design (parity, no new knobs)

1. **Selection block for multiclass** in `ChimeraBoostClassifier.fit`,
   mirroring binary exactly: base `MulticlassBoosting` audition capped at
   `selection_rounds` (cap armed only when a cross race will follow); pairs
   from `_cross_candidate_pairs` on the base fit's importances (same
   `CROSS_TOP_M=6`, pooled-over-classes importances); augmented candidate
   raced with `_stop_if_behind` on softmax val logloss; winner by best val
   within the shared budget; full refit of the base variant only if its
   audition was truncated by the cap. `selection_rounds=None` = both to full
   ES, judged on best val (the binary semantics).
2. **Auto-default parity:** `cross_features=None` → auto-on for multiclass
   under the same gates; explicit `True` honored (the `NotImplementedError`
   is removed); explicit `False` off. `linear_leaves` multiclass raise stays.
3. **Prep reuse extension (bit-identical engineering):** hoist
   `_prep_matrices` so `MulticlassBoosting.fit` accepts `prep_cache` — the
   audition, augmented candidate, and refit share one prep (multiclass prep
   is K per-class TS encodings; without reuse the selection would recompute
   it up to 3x on exactly the expensive sets). `from_base_with_cross` is
   per-column and target-count-agnostic → the splice works for K TS blocks;
   new tests assert cache-hit + splice bit-identity for multiclass.
4. **Bagged mode:** members inherit the machinery per-member, stock (B1
   verdict: per-member selection is load-bearing diversity — no pinning, no
   budget caps).
5. **No harness change needed for the A/B** (defaults flow through; the
   explicit-on guard + comments in `run_benchmarks.py` update at ship time,
   with docs).
6. Watch item (fix only if the smoke shows it): with callbacks present and
   verbose off, `MulticlassBoosting.fit` evals full softmax train loss every
   round (booster.py ~L687) — the internal stop callbacks never read it. Any
   fix must be output-identical.

Out of scope (registered): multiclass linear_leaves (own program),
threshold changes (`CROSS_MIN_SAMPLES`, `CROSS_TOP_M`), per-class pair
proposal, decision-suite composition changes (no adding multiclass sets to
hc mid-program — that is suite-shopping).

## Pre-registered predictions

- Tier-1 eligible slice: net positive, concentrated on the
  interaction-depth≥2 numeric-heavy sets (the multiclass analog of the v1
  −3.30% ablation slice), attenuated vs binary (K margins share the
  columns); expected mean +0.3–1.5% on F1-primary.
- All ineligible multiclass sets + all reg/binary sets + all canaries:
  exact ties everywhere, all tiers. Any broken tie on an untouched path =
  implementation bug → the run is void as evidence; fix and re-screen.
- okcupid-stem: small positive or tie (1 pair only).
- Gate eligible 4 (digit/letter/satellite images): plausibly positive —
  pixel/coordinate interactions are real structure.
- Speed: eligible multiclass fits gain ≤ ~2.2x (the audition adds up to two
  100-round races + at most one refit; prep reuse bounds the prep share).
  hc slowdown column drift ≤ ~+0.2x (okcupid is 1/14 of the speed mean);
  gr chart bit-unchanged.

## Kill bars (registered before any run)

- **Tier 1:** eligible slice must be net-positive on F1-primary (wins >
  losses AND mean > 0) AND the multiclass Brier read (standing B1 lesson:
  read Brier at tier 1) must not be negative beyond noise. Anything less =
  KILL — a thin real-data surface does not get shipped on hope.
- **Tier 2:** Grinsztajn 59/59 exact ties (identity gate). hc: 13 non-okcupid
  sets exact ties; okcupid-stem not a clear loss (primary + Brier over 3
  seeds). A clear okcupid loss = KILL even with a green screen (it is the
  only real hc evidence).
- **Gate (one-shot, runs last):** eligible multiclass sets pooled primary
  non-negative; every other set exact ties. Pooled negative = KILL.
- Speed alone cannot kill inside the predicted envelope; > ~2.5x on eligible
  sets = stop and investigate before tier 2.

## Protocol (per /experiment, adapted to a multiclass-only surface)

1. Tier-1 synth screen, single-model arm: NEW `--synth --seeds 3 --models
   ChimeraBoost --save` vs the newest clean BASE containing that arm
   (`20260717-103015` if it has it — single-arm outputs are bit-identical
   across B-samp/B-prep ships — else one fresh BASE from main). Read:
   `compare_runs.py` overall + eligible/ineligible multiclass slices +
   Brier + `synth_report.py` factor attribution.
2. Tier 2 (sequential, one benchmark at a time):
   - hc 5-arm canonical run vs BASE `20260717-155202` (real read + LightGBM
     cross-run canary + fresh hc table).
   - Grinsztajn single-arm identity run vs `20260717-153114` (59/59 ties;
     deviation from the 5-arm precedent registered here: zero treatment
     surface → competitor arms buy nothing).
3. OpenML gate, one-shot: fresh BASE from a main worktree (PYTHONPATH set,
   `chimeraboost.__file__` printed both arms) vs NEW, arms ChimeraBoost +
   LightGBM, seeds 3.
4. Ship: docs (parameters/FAQ), harness comment+guard update, CHANGELOG
   [Unreleased], verdict here + memory. No TabArena. hc pareto table refresh
   is report-only; README headline chart does not change (gr identity).

Baseline-reuse note: B-prep shipped bit-identical and B-samp touched only
bagged sampling, so single-model arms are comparable across every 2026-07-17
run; any doubt → run a fresh BASE. Aggregate table printed after every run,
per standing rule.

## Implementation log (2026-07-17, branch m1-multiclass-cross)

Shape exactly as registered: `_prep_matrices` hoisted to `_BaseBooster`
(generalized to a list of encode targets — `[y]` scalar, K one-hot columns
multiclass); `MulticlassBoosting.fit` gains `prep_cache`; the classifier's
binary selection block unified over both tasks via a local `_make` factory
(binary constructs `GradientBoosting(loss="Logloss")`, multiclass
`MulticlassBoosting`; same `fast` gate, raced budget, refit rule; the
`NotImplementedError` removed). 454 tests green (16 new: multiclass
selection semantics, raced-budget equivalences, booster cross_pairs
roundtrip, multiclass prep-cache identity, preps-once) incl. all goldens —
the golden panel runs `early_stopping=False`, so no golden can see the new
path; reg/binary bit-identity verified by the suite.

Smoke (`m1_smoke.py`, 1 seed, not decision-grade): okcupid-stem SELECTS its
1 pair (fit 1.9s→4.1s = 2.1x; holdout F1 −0.009/logloss −0.002 = seed
noise, the protocol will judge); syn 531/663 audition and REJECT (metrics
exact ties, as designed). Envelope note, investigated per the registered
bar: 663 hit 3.0x fit (vs ~2.5x registered) — its natural fit is short
(0.35s) so the fixed 100-round challenger race dominates, and the augmented
matrix is ~1.8x wider (30 cross columns on 36 numerics). Structural, same
trade shape binary accepted at ship; absolute cost trivial on synth
(CatBoost is 50–70x there). Revisit only if the screen's fit-cost pattern
reads worse. Bagged multiclass members select per-member in parallel
workers (`m1_bagged_smoke.py`).

## Tier-1 screen (2026-07-17; BASE `20260717-103015` vs NEW `20260717-192856`)

**Identity: PERFECT.** 19/19 ineligible-multiclass + 102/102 reg/binary +
4/4 canaries exact ties, on both primary and Brier — the change touches
exactly the registered surface. Overall: 7W-7L-122T +0.067%.

**Eligible slice (the treatment): knife-edge on the registered bar.**
Primary 7W-7L-1T, mean **+0.610%**; Brier 8W-6L-1T, mean **+0.277%**.
The bar demands wins > losses AND mean > 0: the seventh "loss"
(syn:v2/531) is **−0.00003 absolute F1** (−0.003% relative) — an effective
tie counted as a loss by compare_runs' 1e-9 threshold — while win
magnitudes run ~3x loss magnitudes (+0.0240/+0.0141/+0.0109 vs
−0.0102/−0.0045/−0.0020 absolute). Attribution leans the mechanism's way
without significance: depth≥3 6W-3L vs depth≤2 1W-4L; func=tree 3W-0L +
func=product 1W-0L vs func=neural 2W-6L (crosses help piecewise/product
structure, dilute smooth neural warps); n coef t=+1.27; entity/mixed-cat
sets 0W-4L. All p ≥ 0.25.

**Registered deviation (recorded BEFORE the run):** neither shipping past
a missed bar nor killing on a −3e-5 technicality; the screen-tier remedy
is POWER, not judgment. Extend the eligible 15 to **6 seeds, both arms
fresh** (BASE from a main worktree via PYTHONPATH, path printed; seeds
0–2 must bit-reproduce the prior runs as a validity canary). Verdict
instrument unchanged and pre-stated: **wins > losses AND mean > 0 on the
15-set slice at 6 seeds, Brier not negative beyond noise; miss = KILL.**
No other knob, threshold, or slice changes; one extension only, no
further re-rolls.

**6-seed extension result (BASE `20260717-193421` worktree@main, NEW
`20260717-193523`; validity canaries EXACT — both arms bit-reproduce
seeds 0-2, 45/45 pairs each): PASS on the pre-stated instrument.**
Primary **11W-4L, mean +0.686%** (pooled (set,seed) pairs 35W-24L);
Brier **9W-6L, mean +0.719%** (pooled 40W-19L). The 3-seed coin flip was
seed noise: 130 flipped worst-loss→win, 531's −3e-5 pseudo-loss→win, 765
loss→win. Remaining losses small (worst −1.03%). Tier 1 **PASS** →
tier 2.

## Acceptance checklist

- [x] Implementation on branch `m1-multiclass-cross`: selection block +
      prep_cache hoist + tests — **DONE 2026-07-17** (454 green incl.
      goldens; golden panel runs early_stopping=False so no golden crosses
      the gates; see implementation log)
- [ ] Tier-1 screen vs kill bar — verdict recorded here
- [ ] Tier-2 hc + gr identity — verdict recorded here
- [ ] OpenML one-shot gate — verdict recorded here
- [ ] Ship or revert; docs + CHANGELOG + memory + harness comments; hc table
      refreshed (report-only)
