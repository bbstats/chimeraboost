# Bagging program — make `n_ensembles` a Pareto-frontier point (opt-in accuracy mode)

Self-sufficient handoff (HIGHCARD_PLAN.md convention). Goal: a tuned, turnkey
`n_ensembles=5` operating point that lands ON the pareto.png frontier as
`ChimeraBoostEns5` — strictly stronger than the 99.4 default, fast enough to
recommend honestly. The single-model default is untouched; this supersedes
PARETO_PLAN.md's "No ensemble defaults" line only as OPT-IN-MODE work (still no
default flip).

## Why this is winnable (state of the world, 2026-07-16)

- Default point **99.4 blended @ 6.0x**; CatBoost 98.1 @ 11.8x (dominated).
- **The strength is already proven.** 2026-07-15 Brier program (PAYOFF.md L1):
  Ens5 primary **43W-16L +1.89% decisive** on Grinsztajn — killed ONLY on
  speed (43.1x). That reading predates the selection_rounds=100 default
  (shipped 2026-07-16, 1.5x member fits), so today's true baseline is
  unmeasured — plausibly ~29x. The OOB-ES fix (178ce96, 2026-06-02) is
  ALREADY in that reading.
- **Ens2 is NET NEGATIVE** (Brier 7W-16L, primary 23W-36L): 2 members lose to
  1. Any ship documents K=2 as anti-recommended; the blessed mode is K=5.
- **Brier is the bag's weak leg** (Ens5 Brier 12W-11L coin flip while primary
  was decisive) — and Brier is 2/3 of the classification blend. Suspect:
  mean-of-member-calibrated-probabilities miscalibration. Fixable (Phase 2).
- 2026-07-16 correlation study (`benchmarks/tree_correlation_study.py`):
  cross-member correlation of round-i trees is 0.93–0.98 at round 0, >0.9
  through ~round 15–20, ~0 by round 50–100 (decay rate tracks signal
  strength). ZERO exact structural collisions → no free dedup; the early
  near-collinear block is pure compute redundancy; the zero-corr tail is
  where bagging earns. Member ES variance is huge (133–816 trees, cpu_act).
- **TabArena: OUT OF SCOPE.** Sealed as always, and their protocol already
  ensembles configs, so bagging is redundant there by construction (the e10
  entry exists as the record). Nothing here runs or reads TabArena.

## Cost structure of a bagged fit today (why ~29x, where it goes)

Each member independently: (a) reruns the FULL variant selection (const/linear
audition + cross audition @ selection_rounds=100) even though selection flip
rates are near-deterministic (PARETO_PLAN step 0: ll 8/12, cross 20/21) —
(K−1) redundant auditions; (b) trains on n bootstrap rows (vs 0.8n single) —
~1.25x per-tree; (c) evals every round on ~0.37n OOB rows (vs 0.2n single);
(d) early-stops with huge variance (133–816 trees on one dataset) — some
members plausibly overbuild badly. Members run sequentially
(ensemble_n_jobs=1 in the harness arm).

## Acceptance targets

- **Strength:** Ens5 blended ≥ single + 0.2 on Grinsztajn (≥ ~99.6); #1 on
  every accuracy column; Brier NOT below single on either decision suite.
- **Speed:** Ens5 slowdown ≤ ~12x (≤ 2x the single point; under CatBoost's
  11.8x is the headline sentence).
- `n_ensembles=None` path bit-identical throughout (goldens green).
- hc suite sign tests not unfavorable; OpenML one-shot gate per ship.
- **Kill rule:** if after B1–B3 the point is still >20x or <+0.15 blended,
  program stops, verdicts recorded here + memory.

## Phase 0 — re-baseline + attribution (no source change, ~1 session)

1. Place today's point: `--grinsztajn --models ChimeraBoost ChimeraBoostEns5
   CatBoost LightGBM sklearn_HGB --seeds 3 --save`, then `--highcard` same
   arms. Aggregate table printed per standing rule; pareto text table with
   the Ens5 row is the baseline-of-record.
2. Extend `profile_fit.py` to bagged fits: per-member audition vs winner-fit
   cost, OOB-eval overhead, member round-count distribution, and per-member
   SELECTION AGREEMENT (how often members pick different linear/cross
   variants — bounds B1's risk at zero cost).
3. Brier diagnosis on 3–4 clf sets: member-vs-bag Brier, reliability of
   mean-of-calibrated-probs vs margin-averaging + single temperature refit.
4. Deliverable: tables in this file; pre-register B1–B4 order by measured
   headroom.

### Phase 0 RESULTS — Grinsztajn baseline-of-record (2026-07-16)

Run: `20260716-182842.json` (3 seeds, 59 sets, arms ChimeraBoost /
ChimeraBoostEns5 / CatBoost / LightGBM / sklearn_HGB). Pareto table (all %
vs best IN THIS RUN — Ens5 raises the yardstick, so the single-model row
reads lower than the headline 99.4; the default did not change):

| Model | Blended | Slowdown | RMSE% | Brier% | F1% | Frontier |
|---|--:|--:|--:|--:|--:|---|
| ChimeraBoostEns5 | **99.5** | 34.3x | 99.9 | 98.7 | 99.7 | yes |
| ChimeraBoost | 98.1 | 6.3x | 97.5 | 98.4 | 99.7 | yes |
| CatBoost | 96.9 | 12.6x | 96.9 | 95.7 | 99.4 | no |
| sklearn_HGB | 96.0 | 4.5x | 95.0 | 95.9 | 99.2 | yes |
| LightGBM | 95.9 | 1.0x | 94.7 | 96.1 | 99.3 | yes |

- Ens5 vs single: primary metric 45/59 wins, mean +2.16% — the strength gap
  is real and LARGE, dominated by regression (RMSE% 99.9 vs 97.5; e.g.
  visualizing_soil 0.0512 vs 0.0596, nyc-taxi 0.3675 vs 0.3722).
- Brier 98.7 vs 98.4 — NOT the weak leg on this run (contra the 2026-07-15
  read); confirm with a proper sign test in the Brier diagnosis step.
- Speed: Ens5/single ratio **5.37x** (34.3x absolute vs the 6.3x default
  point). Program needs ~3x off the ratio to hit the ≤~12x ceiling.
- Members often stop EARLIER than the single model (OOB stopping):
  house_sales trees ~228 vs 292, particulate ~193 vs 268, topo_2_1 36 vs
  112 — yet topo_2_1 still costs 4.15x, pointing at per-member fixed costs
  (auditions + prep), exactly B1's target.

### Phase 0 RESULTS — hc baseline-of-record (2026-07-16)

Run: `20260716-185101.json` (3 seeds, 14 sets, same arms). Blended uses the
6 reg + 4 binary sets; 4 multiclass ride along per standing formula.

| Model | Blended | Slowdown | RMSE% | Brier% | F1% | Frontier |
|---|--:|--:|--:|--:|--:|---|
| ChimeraBoostEns5 | **98.9** | 22.1x | 98.7 | 99.1 | 99.2 | yes |
| CatBoost | 98.5 | 114.5x | 97.4 | 99.4 | 100.0 | no |
| ChimeraBoost | 97.6 | 2.6x | 96.1 | 98.9 | 99.4 | yes |
| LightGBM | 96.4 | 1.0x | 94.4 | 98.0 | 99.4 | yes |
| sklearn_HGB | 94.2 | 4.6x | 92.3 | 94.3 | 100.0 | no |

- Ens5 beats CatBoost on blended ON THE CATBOOST-FAVORED SUITE (98.9 vs
  98.5 at 22.1x vs 114.5x) — via regression; CatBoost keeps the clf crown
  (Brier 99.4, F1 100.0) and the multiclass sets.
- Ens5 vs single: primary 9/14 wins, +1.21%. One real regression:
  colleges 0.1494 vs 0.1453 WORSE, with members building 769 trees vs the
  single model's 233 — OOB stopping runs LONG on some hc sets (opposite of
  Grinsztajn's early stops). Attribution must explain this.
- **Speed ratio on hc is 10.0x** (vs 5.37x on Grinsztajn) — K x prep (TS
  encoding on high-card cats was 21–33% of fit in PARETO_PLAN step 0) is
  the prime suspect; raises shared-prep/binning priority alongside B1.

### Phase 0 RESULTS — Brier diagnosis (2026-07-16)

Within-run sign tests, Ens5 vs single (`compare_runs.py --model
ChimeraBoost --model-new ChimeraBoostEns5 --metric brier`):

- Grinsztajn: **14W/9L of 23, mean +0.305% — PASS.**
- hc: **5W/3L of 8, mean +10.8% — PASS** (mean inflated by cjs +84% on a
  near-zero-Brier set; median-ish read is mildly positive).

The 2026-07-15 "Brier weak leg" is NOT present in today's baseline (the
OOB-ES fix predated that kill, but selection_rounds + this fresh measurement
read clean). **B6 recalibration demoted from "the Brier fix" to
opportunistic upside** — run it only if it's cheap after Phase 1.

Pooled decision read (Nathan's dataset-count-weighted rule): 54/73 wins,
weighted mean +1.98% — unambiguous GO on strength.

### Phase 0 RESULTS — bagged fit attribution (2026-07-16)

`python benchmarks/profile_fit.py --bag-attribution --seeds 2 --out
bagging-phase0` (full records `benchmarks/results/bagging-phase0.{json,md}`,
gitignored). K=5 vs single on the same split; shares are % of BAG fit;
select% = all booster fits except each member's final one (what B1 removes,
minus one audition kept).

| dataset | task | single_s | bag_s | ratio | select% | grow% | prep% | eval% |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| gr:cpu_act | reg | 0.4 | 2.9 | 6.5x | 26.7 | 79.9 | 8.2 | 1.7 |
| gr:nyc-taxi | reg | 2.7 | 13.5 | 5.0x | 14.6 | 89.6 | 2.2 | 1.6 |
| gr:MagicTelescope | bin | 0.5 | 2.4 | 5.1x | 24.5 | 60.3 | 8.4 | 1.7 |
| gr:road-safety | bin | 2.2 | 11.3 | 5.0x | 18.8 | 70.0 | 3.9 | 1.3 |
| hc:kick | bin | 2.1 | 15.6 | 7.4x | 45.5 | 41.4 | 25.4 | 10.5 |
| hc:wine-reviews | reg | 1.6 | 11.0 | 6.8x | 41.6 | 52.6 | 23.8 | 15.2 |
| hc:colleges | reg | 0.7 | 6.0 | 8.7x | 33.1 | 71.4 | 12.3 | 8.2 |
| hc:okcupid-stem | multi | 1.8 | 9.7 | 5.5x | 0.0 | 38.8 | 17.0 | 5.3 |

Rounds, members vs single (the colleges anomaly, EXPLAINED):

| dataset | single | member min/mean/max | fits/member |
|---|--:|---|--:|
| gr:cpu_act | 340 | 285/433/674 | 3.1 |
| gr:nyc-taxi | 680 | 412/607/801 | 3.0 |
| gr:MagicTelescope | 234 | 153/188/232 | 2.0 |
| gr:road-safety | 370 | 219/297/400 | 2.0 |
| hc:kick | 140 | 126/161/210 | 2.4 |
| hc:wine-reviews | 423 | 323/482/606 | 3.0 |
| hc:colleges | 310 | **281/738/1377** | 3.6 |
| hc:okcupid-stem | 102 | 77/86/96 | 1.0 |

**Findings:**
1. **Selection redundancy is the single biggest lever and it is WORSE on hc**
   (15–27% of bag fit on Grinsztajn, 33–46% on hc — members re-audition on
   expensive TS-encoded data). Multiclass has no selection (0%), which is
   why its ratio is the mildest.
2. **The hc excess ratio decomposes cleanly:** selection (33–46%) + K x prep
   (12–25% vs 2–8% on Grinsztajn) + OOB-eval overhead (5–15% vs ~1.6%).
3. **colleges = OOB long-stop pathology:** members average 738 rounds
   (max 1377) vs the single model's 310 — 2.4x the trees AND worse accuracy
   (the Phase-0 hc baseline regression). OOB stopping can run members far
   past useful on small noisy high-card sets. B2 must fix this specifically.
4. Elsewhere member rounds are comparable to or BELOW single — the round
   budget is not globally bloated; it is variance + tail pathology.

**Selection agreement across members** (decoded post-hoc from the same run by
`benchmarks/bagging_b1_agreement.py`, 2026-07-16): members 2..K disagree with
member 1 on **16/80 selection decisions (20%)** — all of it on three sets
(cpu_act ll+cf, colleges cf, kick cf); the other five are unanimous.
Disagreement = near-tie audition margins flipped by bootstrap noise, so the
pinned variant should cost ~nothing where it "mispins" — that is B1's risk
hypothesis, and the tier-2 suites are its judge (pinning also removes
selection diversity, which could in principle have been earning variance
reduction).

| dataset | ll votes (per seed) | cf votes (per seed) | disagree vs m1 |
|---|---|---|--:|
| gr:clf_cat/road-safety | ----- ----- | YYYYY YYYYY | 0/8 |
| gr:clf_num/MagicTelescope | ----- ----- | YYYYY YYYYY | 0/8 |
| gr:reg_cat/nyc-taxi | YYYYY YYYYY | YYYYY YYYYY | 0/16 |
| gr:reg_num/cpu_act | YNNNY YYNYY | NYYYY YYYYY | 8/16 |
| hc:colleges | YYYYY YYYYY | NYYNY NNNYN | 4/16 |
| hc:kick | ----- ----- | YYYNN YNYYN | 4/8 |
| hc:okcupid-stem | ----- ----- | ----- ----- | - |
| hc:wine-reviews | YYYYY YYYYY | ----- ----- | 0/8 |

**Pre-registered Phase 1 order (locked by this data):**
1. **B1 shared selection** — removes most of select%; projected ratio
   ~6.2x -> ~4.3x panel-wide, biggest on hc.
2. **B2 stopping design** — must kill the colleges long-stop; include the
   (c) eval-row subsampling variant (hc eval overhead up to 15%).
3. **B-prep (promoted from Phase 3): share bin edges across members** — TS
   encodings stay per-member (they are the diversity), but binning is
   shareable; targets the hc prep slice specifically.
4. **B3 member HPs** (PMLB tune fold) — strength-first per Nathan's K rule.
5. **B4 parallel members** — orthogonal wall-clock lever, measured last.

**Projected path to the ≤~12x target:** B1+B2+B-prep plausibly take the
ratio to ~3.3–4x (absolute ~21–25x on Grinsztajn); B4's concurrent members
at split threads close the rest (sublinear numba thread scaling means ~2x
wall-clock recovery is realistic). If B4 under-delivers, the target needs
B3 to cut rounds via coarser learning rates. Phase 0 COMPLETE.

## Phase 1 — fit-cost levers (the frontier move)

Ordered by expected headroom; re-order only on Phase 0 numbers.

- **B1 Share the variant selection across members.** Audition once (member 1,
  or a pre-pass on full X), pin the winning variant
  (`linear_leaves`/`cross_features` explicit) for members 2..K. Removes
  (K−1)x audition cost — for regression that is ~2 extra 100-round fits per
  member. Risk: loses per-member selection diversity; Phase 0's agreement
  table bounds it. Behavior-changing → full /experiment.
- **B2 Tame the ES round budget.** A/B three designs: (a) shared explicit
  20% eval split for all members (cleaner stopping, enables B7; known
  data-tax risk — cf. the 2026-07-08 GES kill — members currently train on
  full-n bootstraps and would drop to 0.8n); (b) cap members 2..K at
  ~1.3x member 1's best_iteration_; (c) keep OOB but subsample the eval rows
  to 0.2n (pure eval-overhead cut, no stopping change). The 133–816 spread
  says the mean round count can likely drop a lot — but the variance might
  itself be useful diversity: measure, don't assume.
- **B3 Bagged-mode member defaults (the "really good default params" ask).**
  Tiny mechanism-driven grid on the PMLB TUNE fold only (broad search is a
  known anti-generalizer — see the random-search study): learning_rate
  x{1, 1.5, 2} (averaging tolerates coarser steps → fewer rounds → direct
  fit-time cut), colsample {0.7, 0.85, 1.0} and subsample {0.7, 0.85, 1.0}
  (RF-style decorrelation, also cheaper trees), K ∈ {3, 5, 8}. Objective:
  bag blended vs bag fit cost. Top-2 configs → PMLB holdout fold → winner
  through the decision suites. Ship shape: adaptive member defaults applied
  when `n_ensembles>1` (precedent: auto cat_combinations); explicit user
  params always win.
- **B4 Parallel members (wall-clock lever).** K members at threads/K each,
  concurrently, likely beat sequential full-thread fits (numba thread
  scaling is sublinear). Same core budget = honest. Measure on the bench
  box under harness pinning; if real, bagged mode auto-sets ensemble_n_jobs
  and the harness arm uses it (Nathan signs off on chart legitimacy).
- **B5 Shared trunk (PARKED).** Fit rounds 0..m once, fork members after.
  The correlation study caps the win: trunk ~15–20 rounds of 130–800+ on
  strong-signal sets (a few %), bigger only on small noisy sets. Needs
  shared binning + full-data trunk semantics. Build only if B1–B3 land and
  attribution still shows redundant-prefix cost worth it.

### B1 /experiment log (2026-07-16)

**Implementation** (branch bagging-b1, d6e63a5): member 1 auditions, members
2..K pinned — explicit `linear_leaves`, `cross_features=False`, or member 1's
exact pairs via a `_pinned_cross_pairs` fast path (fits the augmented model
directly, zero audition fits). `n_ensembles=None` untouched; 435 tests green
incl. goldens + 2 new pin tests. Panel smoke (contended, indicative):
bag/single ratio cpu_act 6.5→2.6, colleges 8.7→4.6, kick 7.4→4.7,
wine-reviews 6.8→4.5.

**Tier-1 synth screen, full B1** (BASE `20260716-202944` vs NEW
`20260716-203751`, single-model arm 136/136 exact ties = clean canary):

- Ens5 arm: **11W-33L-92T, mean −0.205% (p=0.001)** — the 92 ties are sets
  where every member would have picked the pin anyway; among changed sets
  the pin systematically loses.
- Slices: **regression −0.587% (6W-23L, p=0.002)**, crossfeat-scope −0.501%;
  binary FLAT (+0.005%); multiclass all ties (no selection — expected).
- Loss tail is real, not near-tie noise: worst sets −6.9%, −5.5%, −2.8%.
- Speed mechanism confirmed: Ens5-vs-single cost 5.86x → 4.86x on the screen.
- Read: per-member selection is not pure redundancy on regression — member-
  adaptive variants/pairs are a diversity mechanism the average exploits.
  The 2026-07-15 screen-reversal lesson cuts both ways, but a p=0.001
  negative doesn't go to tier 2 unmodified.

**Iteration (screens are ~10 min — iterate here):** B1-ll isolation variant
(pin linear_leaves only; members keep their own cross race + pairs; binary
arm becomes pre-B1 = built-in canary). If regression damage persists → the
ll pin is the culprit; if it vanishes → the cross pin is.

**B1-ll isolation screen** (BASE `20260716-202944` vs `20260716-204930`):
binary 0-0-54 all ties (canary clean). Regression 9W-16L, **−0.208%,
p=0.23** vs full-B1's −0.587% (p=0.002) → BOTH components cost strength;
cross pinning is the bigger culprit (~−0.38%), ll pinning the smaller
(~−0.21%, weak signal). Member-adaptive selection = real diversity on
regression, in both the variant choice and the pair choice.

**D2 ship-candidate (screening now):** binary keeps the FULL pin (screened
flat at +0.005%, and binary carried the worst select% waste — kick 45.5%);
regression members keep their OWN selection but audition at HALF budget
(`selection_rounds` capped at 50 inside the bag; step-0 race study: k=50
agrees with k=100 on 28/33 decisions, and per-member mispicks average out
across the bag). Single-model default untouched. Fallback if D2's regression
slice regresses: binary-pin-only (already validated by the full-B1 screen's
flat binary slice; regression reverts to stock members).

**D2 screen** (BASE `20260716-202944` vs `20260716-205947`): **PASS —
strength dead flat.** 20W-19L-97T, mean +0.000%; regression 15W-9L −0.004%
(damage gone); binary 5W-10L +0.005% (same benign pin churn as full-B1);
multiclass all ties. 435 tests green.

**D2 clean-box smoke** (b1_smoke.py, real sets): the speed win is
binary-concentrated — kick bag fit 15.6s→11.6s (**26% faster**, ratio
7.4x→4.6x); regression modest (cpu_act ratio 6.5→5.2, wine-reviews/colleges
~flat) because regression members still run 3 booster fits with per-fit
prep — only audition ROUNDS halved. Member variant flags on cpu_act:
(T,F),(T,T),(F,T),(F,T),(T,T) — diversity confirmed alive at k=50. The
remaining regression cost is B2/B-prep/B4 territory by construction.

**D2 tier 2 — KILLED on the Brier leg** (gr `20260716-210952`, hc
`20260716-212356`, canaries 59/59 + 14/14 exact ties):

- Grinsztajn primary: 22W-31L-6T, −0.014% = neutral; pareto **99.5 blended
  @ 29.9x** (Ens5/single ratio 5.4x→4.3x, −20%). Looked like a clean ship…
- …but Grinsztajn **Brier 5W-18L, −0.394%** (p≈0.01): the binary pin forces
  all K members into member 1's model family (naturally ~40% would be base
  models on flippy sets) — averaged probabilities lose sharpness. F1 stays
  flat; Brier is 2/3 of the clf blend. hc: primary −0.017% flat, Brier
  −0.145% same signature (few applicable binary sets).
- **The tier-1 screen HAD this signal and it was under-read**: D2 synth
  binary Brier 4W-12L −0.338% (p=0.077) vs −0.394% real — synth predicted
  the real number almost exactly. PROTOCOL LESSON, now standing: **read the
  Brier metric at tier 1 for any classification-touching change.**

**D3 (final B1 shape): no pins anywhere.** All members keep their own full
selection machinery; the bag caps members' `selection_rounds` at 50
(single-model default untouched). Keeps every diversity channel (variant
decisions, pairs, calibration); only the audition/race budget halves —
regression already screened flat under exactly this treatment (B1-ll ruled
the k=50 cap harmless there: regression slice of D2 == D3 regression
treatment). Library diff vs main is 9 lines; all pin machinery removed
(git history keeps it).

**D3 screen (BASE `20260716-202944` vs `20260716-212906`): PASS on BOTH
metrics.** Primary 23W-17L-96T +0.018% (regression −0.004%, binary +0.050%,
multiclass ties); **Brier 10W-7L +0.005%, flat** (D2 was 4W-12L −0.338%).

**D3 tier 2, Grinsztajn** (`20260716-213641`, canary 59/59 ties): primary
25W-27L-7T +0.004% neutral; blended 99.5 @ 32.5x. But Brier: 10W-13L with
an asymmetric magnitude tail — losses pol −9.5% (near-solved inflation,
Brier 0.016→0.018), **road-safety −2.2% (real)**, california −1.2%; wins
cap at +0.3%. Within-run Ens5-vs-single Brier: 13W-10L signs PASS but mean
flipped +0.305% → −0.245%. The k=50 cap on the BINARY audition race has a
small real mispick tail (step-0 race data predicted it: cross @k=50 had a
1-in-21 mispick with 4.25% regret). Regression slice: **14W-16L-6T +0.011%,
worst set −0.63% — clean.**

**D3 tier 2, hc** (`20260716-215133`, canary 14/14 ties): Ens5 primary
3 non-ties of 14 — colleges −0.36%, employee_salaries +0.11%, kick −0.04%;
mean −0.021%. Flat; the kick delta is the binary cap, removed in B1-final.

**B1-FINAL: regression-only member cap.** Regression bag members audition
at `selection_rounds=50` (validated flat: 3 synth screens + Grinsztajn
regression slice); classifier members fully stock — bit-identical to
baseline, restoring the Phase-0 Ens5 Brier edge by construction. Library
diff vs main: 13 lines in `_fit_bagged._fit_one`. Modest speed (regression
audition rounds halved; binary reverts): the B1 lesson is that most of the
"redundant" selection cost is load-bearing diversity, and only the
regression audition-budget slice was safely removable.

**B1-final confirmation** (gr `20260716-215636`, hc `20260716-221126`;
canaries 59/59 + 14/14 exact ties; predictions verified exactly):

- Grinsztajn: Ens5 primary +0.006%, **Brier +0.000% (all 23 binary sets
  exact ties — the Phase-0 Brier edge restored by construction)**; pareto
  **99.6 blended @ 33.5x** (Ens5 Brier% 98.9 > single 98.6 again).
- hc: primary −0.018% (colleges −0.36% / employee_salaries +0.11%, rest
  ties incl. all clf), Brier +0.000%; pareto **99.3 @ 21.0x**.
- Raw summed Ens5 fit time (`fit_time_delta.py`, LightGBM-drift-free):
  **gr 3823s→3519s = −7.9%** (reg_cat −18.3%, reg_num −12.7%, clf ~0);
  **hc 825s→760s = −7.8%**.
- Pooled (73 sets, dataset-count-weighted): strength-neutral, Brier
  untouched, ~8% cheaper bagged fits. OpenML one-shot gate in flight
  (BASE from worktree @ main, PYTHONPATH verified via
  print_chimera_path.py).

**OpenML one-shot gate (BASE `b1base/20260716-221826` vs
`20260716-223137`): NEGATIVE — B1 KILLED.** Canary 29/29 ties, Brier 22/22
ties, but Ens5 primary **0W-5L-24T, mean −0.056%**: ailerons −0.33%,
cpu_act −0.20%, elevators −0.26%, house_16H −0.28%, wine_quality −0.57%.
Small magnitudes, uniform direction — and wine_quality/house_16H are the
same repeat losers the gr regression slice contained. The independent gate
reads the k=50 cap as a small SYSTEMATIC regression cost that the balanced
suite averaged away.

### B1 VERDICT (2026-07-16): KILL — nothing ships. Selection is load-bearing.

Every variant died on a measured mechanism: **full pin** (regression
−0.59% p=0.002 — variant/pair diversity is real), **binary pin** (Ens5
Brier edge erased, −0.39% — model-family diversity keeps averaged
probabilities sharp), **uniform k=50 cap** (binary race mispick tail,
road-safety Brier −2.2%), **regression-only k=50 cap** (OpenML gate 0W-5L,
−0.06% systematic). Per Nathan's standing rule (bagged mode is the accuracy
play; fit cost only breaks ties) an ~8% fit saving does not buy a
measurable strength nibble. Library reverted to stock; branch bagging-b1
holds the full iteration history. The select% slice of the Phase-0
decomposition is hereby re-labelled: NOT waste — working diversity.
Speed program continues on levers that cannot touch strength: B2 (stopping;
the colleges long-stop), B-prep (shared binning — output-identical possible),
B4 (parallel members — output-identical by construction).

### B2 /experiment log (2026-07-16, branch bagging-b2)

Screens vs BASE `20260716-202944` (same as B1's; library was stock).

- **B2c (OOB eval subsampled to 0.2n): KILLED at screen**
  (`20260716-224958`). Regression 13W-35L **−0.161% p=0.002**; Brier
  −0.229%. Damage sits on n≥2000 (+0.075% on n<2000) → NOT tiny-eval
  noise; suspect the subsampled eval also degrades members' variant-
  selection races (B1's lesson recurring: anything that touches selection
  quality costs strength).
- **B2 b+c composite** (`20260716-225744`): regression −0.249% p=0.013,
  **Brier 30W-58L −0.391% p=0.004** — (b)'s implied marginal is ADDED
  Brier damage. Mechanism: anchoring members 2..K to 1.3x member 1's stop
  truncates members whose legitimate stop is later; under-trained members
  blur the averaged probabilities. The plan's "variance might itself be
  useful diversity" caution is measuring TRUE.
- **B2b solo (`20260716-230417`): KILLED at screen.** Primary fine
  (+0.182%, 70W-55L) but **Brier 25W-62L −0.237% p=0.000** — binary
  −0.276% p=0.009 AND multiclass −0.175% p=0.005. Truncated members lose
  probability sharpness while F1 stays flat — the same failure axis as the
  D2 binary pin. (Without B1's read-Brier-at-tier-1 lesson this would have
  sailed through the primary read.) The member round-count VARIANCE is
  functioning ensemble machinery; the colleges long-stop needs a different
  fix (note for B3: coarser member learning rates may shorten it honestly).
- **B2a (shared 20% split, `20260716-231147`): KILLED at screen —
  decisively.** Primary 35W-98L **−1.42% p=0.000** (regression −3.33%!),
  Brier −3.39% p=0.000 across the board. The registered prediction (data
  tax + correlated stopping/selection both cost) confirmed at 5-10x the
  expected magnitude; the 2026-07-08 GES-kill precedent held exactly.

### B2 VERDICT (2026-07-16): KILL — all three pre-registered stopping designs dead at the screen. Library stock.

The member stopping VARIANCE (like B1's selection diversity) is functioning
ensemble machinery: capping it costs Brier (b), noising its signal costs
regression strength (c), and centralizing it costs everything (a). The
colleges long-stop pathology remains OPEN — carried to B3 as a
member-defaults question (coarser member learning rate shortens rounds
honestly instead of truncating them). OOB-eval overhead (5-15% hc) stays
as-is: the eval rows are doing selection work, not just stopping work.

### B4 /experiment log (2026-07-16, branch bagging-b4; REORDERED ahead of B-prep/B3)

Reorder rationale: B1+B2 killed every strength-touching lever; B4 is the
only one that provably cannot change model outputs (same members,
different scheduling), and it carries the largest projected win. B-prep
changes bin edges (bootstrap-specific → shared) = another strength gamble,
so it queues behind.

- **Clean-box measurement** (`b4_parallel_timing.py`, 12 cores, members at
  12/5 threads): speedups cpu_act 1.47x, diamonds 1.58x, kick 1.87x,
  wine-reviews 1.97x, colleges 1.22x, nyc-taxi 1.21x — predictions
  IDENTICAL (allclose 1e-9) in every case. Weak cases = member-length
  imbalance (the longest member dominates wall-clock; colleges' 1377-round
  member). Worker-count sweep (`b4_worker_count.py`): W=5 best on the good
  sets; cold-executor caveat — first parallel fit in a process pays worker
  spawn/import, so a one-off fit on a long-member set can be ~0.9-1.0x;
  repeated fits (and the harness) amortize.
- **Ship shape:** `ensemble_n_jobs` default 1 → **-1**: W = min(K, budget)
  workers, each at budget/W threads, where budget = thread_count or numba
  threads — a bagged fit uses the same cores a single fit would. (Also
  fixes the pre-existing -1 oversubscription bug: abs(-1)=1 gave every
  member the FULL budget.) Sequential = explicit ensemble_n_jobs=1.
  Harness Ens5 arm measures the shipped config (Nathan's chart-legitimacy
  pre-answer). New test: budget division + identity.
- **Tier-2 identity gate PASSED** (gr `20260716-233233`, hc
  `20260716-234740`): Ens5 primary **59/59 + 14/14 EXACT ties** vs the
  Phase-0 baselines — models bit-identical under parallel scheduling and
  divided threads, across regression/binary/multiclass. Raw Ens5 wall-clock
  −5.3% (gr) / −8.1% (hc) under harness jobs=5 pinning (budget=2 → W=2×1t;
  the clean-box 1.2–2x is what full-budget users get). The slowdown COLUMN
  read 37.2x only because LightGBM itself ran 11.7% faster in the 3-arm run
  (less box contention than the 5-arm baseline) — cross-run slowdown
  columns are polluted by arm-set composition; **the canonical pareto
  refresh at program close must re-run the full 5-arm baseline set.**

### B4 VERDICT (2026-07-17): SHIP. `ensemble_n_jobs` default → -1.

Output-identical by construction and verified exactly on 73 datasets (the
OpenML accuracy gate is vacuous for a scheduling change — no strength
surface exists; timing is the only change). Ships with the thread-budget
division (also fixes the old -1 oversubscription bug), harness arm on the
shipped config, terse docs, CHANGELOG [Unreleased].

### B3 /experiment log (2026-07-17, branch bagging-b3)

B3 is now load-bearing for the ≤~12x target (B1/B2 killed; B4's chart gain
is ~6-8% under harness pinning). Tooling: `--lr` / `--chimera-subsample` /
new `--chimera-colsample` now reach the bagged arms' members (d91f161).

- **Tune-fold baseline** (`20260716-235522`, 13 tune sets, 3 seeds):
  default Ens5 beats single 11/13 +0.50% at 6.27x.
- **One-factor grid launched** (each vs that baseline, member params only,
  K=5 unless noted): lr {0.15, 0.2} (averaging tolerates coarser steps →
  fewer rounds → direct fit cut, and the honest colleges fix), subsample
  {0.85, 0.7} and colsample {0.85, 0.7} (RF-style decorrelation, cheaper
  trees), K {3, 8} (strength decides K per Nathan). Judge: bag blended +
  Brier (tier-1 lesson) vs bag fit cost; top-2 composites → PMLB holdout →
  decision suites → OpenML gate. Ship shape if a config wins: adaptive
  member defaults when `n_ensembles>1`, SUPER visible per Nathan
  (always-on fit notice + attribute + docs section).

**One-factor grid results** (tune fold, 13 sets, vs baseline
`20260716-235522`; mean deltas outlier-prone on near-solved PMLB sets —
signs weighted over means):

| config | primary Δ | Brier Δ (sign) | Ens fit time |
|---|--:|--:|--:|
| lr 0.15 | −0.33% | +0.65% (PASS) | 0.91x |
| lr 0.20 | −0.17% | +0.33% | **0.79x** |
| subsample 0.85 | +0.05% | −0.65% | 1.19x (SLOWER) |
| subsample 0.70 | +0.11% | −0.64% | 1.21x (SLOWER) |
| colsample 0.85 | −0.03% | −0.30% | 0.90x |
| colsample 0.70 | −0.20% | +1.24% (PASS) | **0.77x** |
| K=3 | −0.13% | (noisy) | 0.65x |
| K=8 | **+0.15% (PASS)** | (sign PASS) | 1.43x |

Reads: **subsample = dead axis** (slower AND Brier down — MVS overhead
inside small bagged members). lr 0.2 and colsample 0.7 are the real speed
axes (~40% member-cost cut combined, small primary dips). **K=8 > K=5 on
strength** (primary + Brier signs) at 1.43x — with B4's parallel members
K scales sublinearly.

**Composites, tune fold** (vs baseline; fit secs are the Ens arm total,
base 322.8s):

| composite | fit | primary | Brier signs |
|---|--:|---|---|
| C1 lr.2+cs.7 K=5 | 208.5s (0.65x) | 4W-8L −0.26% | 5W-3L |
| C2 lr.2+cs.7 K=8 | **297.7s (0.92x)** | 6W-6L +0.01% | 6W-2L |
| C3 lr.15+cs.85 K=8 | 367.8s (1.14x) | **8W-4L +0.17%** | **7W-1L** |

C1 DROPPED (pays for speed with primary — the exact profile that dies at
gates). **Top-2 = C2 (8 members cheaper than today's 5, strength par) and
C3 (strength winner at +14% cost).** → PMLB holdout fold confirm
(+ its own default-Ens5 baseline), then suites, then gate.

**Holdout confirm (12 sets, baseline `b3-ho-base` 823.7s): both
generalize.** C2: primary 6W-5L +0.39%, Brier 4W-4L, fit 679.3s
(**0.82x**). C3: primary 5W-6L +0.10%, Brier 6W-2L, fit 847.6s (1.03x).
Pooled 25 sets: primary near-tie (C2 +0.19% 12W-11L, C3 +0.14% 13W-10L);
**Brier pooled favors C3 (13W-3L vs 10W-6L); cost favors C2 (0.87x vs
1.09x).** PMLB cannot separate them → DEVIATION from the "one winner"
pre-registration, recorded here: BOTH go to the decision suites (~2h),
because C3's Brier edge is exactly the late-kill profile (B1 lesson) and
the suites are the real judge. Suite runs use LightGBM as the cross-run
exact-tie canary (the single arm inherits the config flags in these runs,
so it is an lr/cs ablation, not a canary).

**Decision suites (2026-07-17; all four LightGBM canaries 73/73+73/73
exact ties):**

| config | gr primary | gr Brier | hc primary | hc Brier | fit vs Ens5 base |
|---|---|---|---|---|--:|
| C2 (gr `092230`, hc `093414`) | 33W-26L −0.02% | 12W-11L −0.05% | 9W-3L +0.42% | 4W-4L | 0.75x gr / 0.95x hc |
| C3 (gr `093733`, hc `095309`) | **43W-16L +0.22%** | **14W-9L +0.19%** | **11W-1L +0.54%** | **6W-2L** | 1.01x gr / 1.10x hc |

**C3 WINS by the strength-first rule** — pooled 54W-17L +0.28% (sign
p≈1e-5), positive on both suites separately AND on both Brier legs, at
par cost. Note C3's gr read (43W-16L +0.22%) is the same shape as Ens5's
original strength case vs single — this is a real upgrade, not noise.
C2 recorded as the budget alternative (0.75x gr at par strength).
→ OpenML one-shot gate in flight (Ens5-default vs Ens8-C3, both configs
expressed via flags on identical code, LightGBM canary). If it passes:
ship = blessed K=8 + adaptive member defaults (lr 0.15 auto when
learning_rate=None; colsample default moves to None so bagged members
resolve 0.85 while explicit user values always win), SUPER-visible per
Nathan.

**OpenML one-shot gate (base `20260717-095807` vs new `20260717-101033`,
LightGBM canary 29/29 ties): PASS.** Primary **18W-9L-2T +0.18%**; Brier
**18W-4L** (mean −21% is 100% the `mushroom` artifact — an exactly-solved
set, Brier ~0.0000 both arms, −486% relative on a <1e-4 absolute delta;
trimmed mean ≈ +0.8%).

### B3 VERDICT (2026-07-17): SHIP — Ens8-C3 becomes the blessed bagged mode.

Shipped (branch bagging-b3-ship): bagged-member auto defaults
`learning_rate None→0.15`, `colsample None→0.85` (colsample constructor
default 1.0→None; single-model resolution None→1.0 is bit-identical,
goldens green); always-on one-line fit notice + `member_params_` attribute
+ docs (recipes bagging section: recommended K=8, K=2 anti-recommended);
harness ChimeraBoostEns8 arm; --chimera-colsample default None so harness
arms measure the shipped config. 437 tests green. The old grid/composite
runs used explicit flags at the same values, so all B3 evidence transfers
to the shipped shape exactly.

### B-samp (queued behind B3): member sample size — subagging (lit-reviewed 2026-07-17)

Nathan's observation: sklearn BaggingRegressor exposes `max_samples`; we
hardcode a full-size with-replacement bootstrap (= its default). Literature:

- **Subagging** (Bühlmann & Yu 2002; also Grandvalet 2004, Buja & Stuetzle
  2006, Friedman & Hall 2007): sampling ~half WITHOUT replacement ≈ full
  bootstrap accuracy at a fraction of the compute. Mechanism: a bootstrap's
  effective sample size is ~n/2 (duplicate multiplicities are just integer
  weights), so 0.5-without-replacement is its statistical twin and
  0.632-without-replacement (same unique-row exposure, ESS 0.632n > 0.5n)
  is if anything slightly data-richer per member.
- **Martínez-Muñoz & Suárez 2010** (Pattern Recognition): optimal m/n is
  problem-dependent and usually SMALLER than the standard choices; the
  performance transition sits where samples hold ~half the distinct
  instances (≈69% with replacement / 50% without); OOB error selects a
  near-optimal ratio per dataset — machinery we already have.
- **Random Patches** (Louppe & Geurts, ECML 2012): joint row+column
  subsampling matches full-data ensembles at much lower cost — supports
  composing max_samples with C2/C3's colsample members.
- Transfer caveats: (1) that literature bags UNSTABLE weak learners; our
  members are bias-optimized GBDTs with early stopping — data cuts may hit
  member bias, not just variance; (2) B2a's kill is CONFOUNDED evidence
  (its members had ~0.51n unique rows AND correlated stopping — can't
  attribute the −1.4% between the two); (3) no literature found on
  calibration/Brier under subagging — our screens' Brier read covers it.

**Pre-registered design (after B3 resolves):** arms max_samples ∈
{0.632 no-replacement (compute-free twin of today), 0.5 no-replacement
(the literature's half-subagging, ~50% row cut)}; OOB = unsampled rows
(machinery unchanged, OOB grows to 0.37n/0.5n); everything else stock.
Screen both (primary + Brier), winner composes with the B3 winner → tier
2 → gate. If it ships, expose as a real `max_samples` param. Thresholds
note: size-triggered auto rules (LL/CROSS minimums, auto-mcw) see the
smaller member n — watch the screen's small-n slices.

**B-samp screens (2026-07-17, branch bagging-bsamp; BASE = shipped Ens8,
`20260717-103015`):**

- **0.632 no-replacement (`103856`): KILLED.** Fit 0.51x (!) but primary
  63W-69L −0.45% with a heavy tail (−16.5/−13.6/−11.2%) concentrated on
  n≥2000 — NOT the threshold story (n<2000 was +0.24%). The registered
  strong-learner caveat confirmed: boosted members convert rows to
  accuracy; the literature's weak-learner equivalence does not transfer at
  0.632. 0.5 skipped as strictly dominated.
- **0.8 no-replacement (`104515`): DOUBLE WIN — the best screen of the
  program.** Primary **92W-41L +1.20% p=0.000** (regression 42W-6L
  +2.72%!), Brier signs favorable both task types (binary 35W-19L +0.35%
  p=0.04; multiclass 23W-11L, mean is near-zero-set artifact), fit
  **0.65x**. Mechanism (clear in hindsight): a bootstrap gives members
  0.632n unique rows at n rows of compute (ESS ~0.5n); 0.8-no-replacement
  gives 0.8n unique rows at 0.8n compute — MORE effective data AND less
  work. Strong-learner members monetize the extra data. Tail note:
  syn 569 loses under every row-cut (−10% here). Not tuning further at
  the screen (anti-generalization risk; 0.8 is mechanism-round). → Tier 2
  vs the shipped-Ens8 baselines (C3 suite runs are bit-equivalent) in
  flight.

**B-samp 0.8 tier 2 (gr `20260717-105157`, hc `110629`; canaries 59/59 +
14/14 exact ties): the strongest tier-2 result of the program.**

- Grinsztajn: primary **54W-5L +0.94%** (p≈1e-11), **Brier 23W-0L +2.53%
  — a perfect sweep**; fit 0.874x.
- hc: primary 7W-5L-2T +0.05% (flat-positive), **Brier 8W-0L** (mean
  +13.2% is near-zero-set inflated; the 8/8 signs are the read); fit
  **0.725x**.
- Pooled: **61W-10L-2T primary; 31W-0L Brier.** Stronger AND faster on
  both suites. → OpenML one-shot gate in flight (base = `101033`, the
  shipped-Ens8 gate run; one new run from the branch).

**OpenML one-shot gate (`111013` vs `101033`, canary 29/29): PASS —
primary 22W-5L +0.68%; Brier 14W-8L** (mean −3.9% is entirely near-solved
artifacts: mushroom −102% at Brier 0.0000, nursery at 0.0002, car/kr-vs-kp
absolute deltas <0.007; trimmed mean positive).

### B-samp VERDICT (2026-07-17): SHIP — `max_samples=0.8` without replacement.

Shipped: `max_samples` parameter (default 0.8 subagging; 1.0 restores the
classic bootstrap), docs, CHANGELOG, 438 tests green. The 0.632 kill +
0.8 win pin the design law of this program: **for strong boosted members,
effective data per member beats sampling diversity at the margin** — the
opposite of the weak-learner bagging intuition the plan started from.

### B-prep /experiment log (2026-07-17, branch bagging-bprep)

**Design pivot, recorded before any run.** The plan's registered shape —
share bin edges ACROSS members (bootstrap-specific → full-X borders) — is
behavior-changing: member border jitter is a (weak) diversity channel, and
B1/B2 killed four straight levers on exactly that class of risk, for a
measured ceiling of only ~1-3% (Phase-0 bin% 2-9% of bag fit, ÷ fits/member,
× (K−1)/K, × the numeric-column fraction). Meanwhile the Phase-0 phase data
exposes a strictly better target: **every booster fit inside one sklearn fit
recomputes identical prep** — the const/linear auditions, the cross-augmented
candidate, and the winner refit (fits/member 2.0–3.6) each rerun the full
TS-encode + border-learn + bin pipeline on the same rows with the same
random_state. Deduplicating THAT is output-identical by construction (every
prep artifact is per-column; appending cross columns leaves base columns
untouched), captures most of the same prep slice, and also speeds the
SINGLE-model points on both suites. Cross-member border sharing: **SKIPPED
(measured-ceiling + diversity-risk grounds), recorded here.**

**Implementation** (42debf7): booster fits within one sklearn fit share a
`prep_cache` keyed by `cross_pairs`; repeat fits reuse the cached
prep+matrices outright; the cross-augmented fit splices freshly binned cross
columns into the base binned matrix
(`FeaturePreprocessor.from_base_with_cross`). 445 tests green incl. all
goldens; 7 new tests (splice bit-identity, cache-hit identity,
prep-runs-once counters).

**Clean-box smoke** (`bprep_smoke.py`, BASE = main worktree via PYTHONPATH,
paths verified): prediction fingerprints EXACT to 9 decimals on all 5 panel
sets, single AND Ens8. Fit time: hc singles −17/−32/−21%
(kick/wine-reviews/colleges — the TS re-encode was the waste), gr singles
−4/−6%, Ens8 bags −4 to −13% (parallel members amortize wall-clock).

**Tier-2 identity gate + canonical timing** (gr `20260717-153114`, hc
`155202`, 5 arms, 3 seeds): **identity PERFECT — 73/73 datasets exact ties**
(gr 59/59 single + 59/59 Ens8 + 23/23 Brier; hc 14/14 + 14/14 + 8/8).
Raw summed fit time (`fit_time_delta.py`): **gr single 0.976 / Ens8 0.977;
hc single 0.881 / Ens8 0.886**. Honesty note: LightGBM itself drifted
0.941 (gr) / 0.912 (hc) between these runs (box conditions), so the
suite-level net-of-drift read is ~flat on gr and ~−6% normalized on hc; the
CONTROLLED measurement is the clean-box smoke (−17/−32/−21% on the TS-heavy
hc singles), and the suite pools in multiclass sets (no selection → no
reuse win) which dilutes it. OpenML gate: **vacuous by the B4 precedent**
(bit-identical output ⇒ no strength surface; timing is the only change).

### B-prep VERDICT (2026-07-17): SHIP — intra-fit prep reuse, bit-identical.

Ships (branch bagging-bprep → main): `prep_cache` sharing across the booster
fits of one sklearn fit + `FeaturePreprocessor.from_base_with_cross` splice
for the cross-augmented candidate. 445 tests green incl. goldens; 73/73
suite identity; hc fits ~12% cheaper raw (single AND every bagged member),
gr ~2%. The registered cross-member border-sharing shape is retired
permanently: measured ceiling ~1-3%, and it is the diversity-risk class
this program killed four times. Prep redundancy INSIDE a fit was the real
slice, and it was free.

## Phase 2 — strength levers (make it goated)

- **B6 Bag-level recalibration (the Brier fix).** Average raw margins across
  members and fit ONE temperature on pooled OOB predictions, vs today's
  mean of per-member calibrated probabilities. Targets the proven weak leg
  (2/3 of the clf blend). Cheap; decide on the suites like everything else.
- **B7 ISLE-style post-hoc tree reweighting.** Flatten the bag to K×T trees;
  nonnegative lasso/ridge over per-tree contribution vectors on held-out
  data; prune zeroed trees. The correlation study predicts it collapses the
  near-collinear early block (~1 round-0 tree at weight ~1 instead of K at
  1/K). Leakage note: member k's OOB rows are in-bag for other members —
  needs B2(a)'s shared holdout (or a dedicated split). Expected: small
  strength gain + predict compression; kill if dev-panel strength is flat.
- **B8 Conformal/quantile offsets under reweighting.** `quantile_offset_` is
  per-member post-fit; any reweighting/flattening must recompute it. Listed
  so it is not forgotten (regression Quantile/MAE losses).

### B7 probe — pre-registered design (2026-07-17, run now per Nathan)

Zero-library-change probe, regression first (the bag's strength lives there
and flattening is clean; clf adds per-member temperatures — only explored if
regression shows signal). Per dataset × 3 seeds: 75/25 train/test, then
train → 80% member-fit / 20% reweight split. Arms: **bag100** (Ens8 on full
train = the shipped mode), **bag80** (Ens8 on the 80% — the fair
data-tax baseline), **rw** (bag80's flattened K×T trees reweighted by
nonnegative LassoCV fit on the 20% split; intercept absorbs the inits).
Identity guard: flattened member sums must reproduce member.predict.
**Kill bar (registered before the run): B7 dies unless rw beats bag100 on
a majority of (set, seed) pairs** — beating only bag80 is not shippable
because shipping pays the 20% data tax. Record tree-compression
(% zero-weight trees) either way; B8 (quantile offsets) stays moot unless
this survives. Panel: cpu_act, house_sales, wine_quality (a gr repeat
loser), hc wine-reviews, hc colleges (the long-stop set — reweighting is
the one remaining lever that could tame its 1377-round members post hoc).

### B7 VERDICT (2026-07-17): KILL — 0/15 against the kill bar; uniform averaging wins outright.

Probe run (`b7_reweight_probe.py`, results `benchmarks/results/b7_probe.json`,
identity guard passed on every member): reweighting beat the shipped bag100
on **0/15 (set, seed) pairs** (registered bar: majority), mean RMSE
**−6.30% vs bag100** — and, decisively, **−2.30% vs bag80 on the SAME
training data**: the lasso hurts before the data tax is even counted. The
only near-parity was wine-reviews (rw ≈ bag80, still behind bag100). The
lasso pruned 88–99% of trees (mean 93%), confirming the correlation study's
near-collinear structure — but the sparse solution the small holdout
supports is strictly worse than uniform 1/K averaging. Even as a
predict-compression story it is dead: −6.3% RMSE for the pruning is far
worse than just using the single model. Consistent with every kill in this
program: the ensemble's uniform structure IS the machinery; replacing any
part of it with holdout-optimized selection loses. **B8 moot. Program has
no open items left.**

## Phase 3 — predict-side flat forest (optional engineering; NOT the Pareto axis)

Slowdown on the chart is FIT time; this phase is for the predict story
(currently K transforms + K forest walks per predict). Shared bin edges
across members (bin once on full X) → single transform + one packed
multi-member forest walk on numeric data; TS-encoded cats keep per-member
transforms. Composes with B7's pruning. New path gets its own goldens.
Do last; skip freely.

## Protocol (unchanged, stated for the handoff)

- Every behavior-changing lever: /experiment — synth screen (tier 1) →
  Grinsztajn + hc, sign-tested separately (`compare_runs.py --model
  ChimeraBoostEns5`) → OpenML one-shot gate. PMLB tune fold ONLY for B3.
- One benchmark at a time. Script files, never `python -c`. Worktree A/Bs
  with PYTHONPATH + `chimeraboost.__file__` printed. No TabArena, ever.
- Every verdict (win or kill) recorded here; memory + CLAUDE.md updated at
  program close.

## Decision points — ANSWERED (Nathan, 2026-07-16)

- **B2(a) data tax: YES, judge by results.** A/B all three stopping designs;
  the suite outcome is the only judge, even if the winner trains members on
  80% of the data.
- **B4 chart timing: parallel-member wall-clock is legitimate** for the
  headline Ens5 point IF bagged mode ships with it auto-enabled — the chart
  measures the shipped config (same core budget as every other model).
- **Adaptive member defaults: auto-adjust approved, but make it SUPER
  visible.** Not verbose-gated: an always-on one-line notice at fit when
  bagged-mode defaults activate, an estimator attribute exposing the
  effective member params, and a dedicated docs section listing exactly what
  changes and why. Terse but unmissable.
- **Suite split: weighted average of the two suites.** Operationalize as a
  dataset-count-weighted pooled decision across Grinsztajn + hc (each
  dataset one vote in the pooled sign test / mean delta); both suites still
  reported separately. (Exact weights = dataset counts unless Nathan revises.)
- **Final K: strength decides.** (Nathan, 2026-07-16.) The bagged mode is the
  accuracy play — "if all you wanted was speed, you could skip the bagging."
  B3 picks the strongest K on the suites; fit cost only breaks ties, subject
  to the program's ≤~12x acceptance ceiling.

## Program close (2026-07-17) — status accounting

**Shipped (all unreleased, on main):** B4 parallel members
(`ensemble_n_jobs=-1`), B3 member defaults + blessed K=8, B-samp
`max_samples=0.8` subagging. **Killed with full records:** B1 (4
variants), B2 (3 designs), B-samp 0.632/0.5. **The blessed bagged mode
is now: `n_ensembles=8`, members at lr 0.15 / colsample 0.85, 0.8
no-replacement samples, parallel workers.**

- **Strength: targets EXCEEDED.** Cumulative over the Phase-0 Ens5
  baseline: +0.28% (B3) then +0.94% gr (B-samp) primary, with perfect
  tier-2 Brier sweeps (23-0, 8-0) — on top of Ens5's original +2.16% over
  the single model. Canonical 5-arm close-out runs in flight for the
  final pareto numbers.
- **Speed: the ≤~12x ceiling was NOT met.** Projected ~30x absolute on gr
  (34.3 baseline × B3 ~1.01 × B-samp 0.874; B4's chart gain is small
  under harness pinning). Every fit-cost lever that touched the ensemble
  mechanism's diversity or data was killed by measurement; what shipped
  is what survived. Remaining known headroom: B-prep shared binning
  (hc-heavy, est. single-digit % pooled) — OPEN, queued for a future
  session. **Kill-rule review (>20x) = Nathan's call**: the program
  delivered a much stronger, modestly cheaper frontier point rather than
  a cheap one.
- **Phase 2 verdicts:** B6 recalibration — **SKIP, documented**: the leg
  it targeted is now the bag's strongest (tier-2 Brier sweeps); nothing
  to fix. B7 reweighting — queued as a zero-library-change OOB-masked
  probe (design + downgraded outlook recorded above with Nathan); B8
  moot unless B7 ships. B5 shared trunk stays parked.

## Acceptance checklist

- [x] Phase 0 baseline-of-record tables committed here (Ens5 point + attribution + Brier diagnosis) — 2026-07-16, Phase 0 COMPLETE
- [x] B1 shared selection through /experiment — **KILLED 2026-07-16** (all 4 variants; selection = load-bearing diversity; OpenML gate 0W-5L on the final k=50-cap shape; library stock)
- [x] B2 ES-budget design picked by A/B, through /experiment — **ALL THREE KILLED at screen 2026-07-16** (b: Brier p=0.000; c: regression p=0.002; a: −1.42% p=0.000); stopping variance = working machinery; colleges long-stop carried to B3
- [x] B3 bagged-mode defaults tuned on PMLB, validated holdout, through the suites — **SHIPPED 2026-07-17** (Ens8-C3: lr .15 + colsample .85 members, K=8 blessed; suites 54W-17L +0.28% pooled; gate 18W-9L PASS)
- [x] B6 recalibration decided — **SKIP 2026-07-17**: the Brier leg swept 23-0/8-0 at tier 2; no weak leg exists to fix
- [x] B7 reweighting decided — **KILLED 2026-07-17** (probe: 0/15 vs the registered bar, −2.3% even against the same-data bag; 93% pruning confirms collinearity but the sparse solution loses; B8 moot)
- [x] pareto.png shows the blessed bagged point on the frontier — **DONE 2026-07-17**: canonical 5-arm runs (gr `20260717-112941`, hc `115025`, single-arm canary 59/59 ties vs Phase 0); **Ens8 sweeps Grinsztajn 100.0/100.0/100.0/100.0 @ 30.1x** (CatBoost off the frontier); **hc 99.6 @ 14.5x vs CatBoost 98.6 @ 118.7x**; chart refreshed, README examples fixed (n_ensembles 2→8), FAQ updated. NOTE: the Ens8 arm raises the in-chart yardstick, so the single-model row reads 97.0 — the default did not change.
- [x] Verdicts → memory (CLAUDE.md unchanged: no protocol changes needed)
- [x] B-samp `max_samples=0.8` — **SHIPPED 2026-07-17** (added post-plan; Nathan's idea, lit-validated)
- [x] B-prep — **SHIPPED 2026-07-17 as intra-fit prep reuse** (design pivot: cross-member border sharing SKIPPED on measured-ceiling + diversity-risk grounds; the real slice was prep recomputation ACROSS booster fits within one sklearn fit; bit-identical, 73/73 suite identity ties, hc fits ~12% cheaper raw)
