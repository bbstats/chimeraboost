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

## Acceptance checklist

- [x] Phase 0 baseline-of-record tables committed here (Ens5 point + attribution + Brier diagnosis) — 2026-07-16, Phase 0 COMPLETE
- [ ] B1 shared selection through /experiment
- [ ] B2 ES-budget design picked by A/B, through /experiment
- [ ] B3 bagged-mode defaults tuned on PMLB, validated holdout, through the suites
- [ ] B6 recalibration decided (Brier ≥ single on both suites, or documented kill)
- [ ] B7 reweighting decided (ship or clean kill)
- [ ] pareto.png shows ChimeraBoostEns5 on the frontier; README + docs updated (terse)
- [ ] Verdicts → memory + CLAUDE.md
