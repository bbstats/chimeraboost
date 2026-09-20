# Random effects, slice 1: group intercepts for regression (#109)

Status: SHIPPED 2026-09-20 (verdict below). Refinement stays (ablation);
random-split validation confirmed.

## What the issue asks

Issue #109 ("Random effects?") proposes semi-parametric GBDT, `y = F(X) + Zb
+ eps`: trees learn the global non-linear part while a mixed-model solver fits
group intercepts (later: slopes), alternating until convergence. GPBoost/MERF
are the named precedents. Maintainer scoping (2026-09-19): slice 1 is
**random intercepts, regression only**, behind an **opt-in flag on the
existing `groups=` fit parameter**, gated by a **dedicated grouped-data
benchmark** (the decision suites have no group columns, so `--decide` cannot
see this feature).

## Design (pre-registered)

- API: `ChimeraBoostRegressor(random_effects=True)` + the existing
  `fit(X, y, groups=...)`. `predict(X, groups=...)` adds the fitted intercept
  for each row's group; unseen groups get exactly 0 (the `F(X)`-only
  fallback the issue specifies). Default `False`: predictions bit-identical
  to today, enforced by a test.
- Algorithm: F-first plus a top-up plus one refinement (amended
  2026-09-20: see below). The standard pipeline fits trees on raw y with
  the group column dropped; b solves on the residuals; then the full-data
  refit -- which runs anyway -- trains on the group-adjusted target, and b
  re-solves once more. No booster changes at all: two O(N) solves around
  the existing refit.

## Amendment 2026-09-20: F-first plus top-up (gate finding)

In-loop backfitting (trees on demeaned targets from round 0) was built,
gated, and FALSIFIED the same day. On hc:house_prices_nominal it ran all
2000 rounds without early stopping ever firing, at 9x the plain fit's time
and worse accuracy (35210 vs 33569). Mechanism: b absorbs group means from
round 0, so the trees fit demeaned targets containing no between-group
X-signal to learn; residuals then genuinely show huge between-variation and
REML correctly answers "shrink less" -- a self-consistent bad equilibrium
where the two smoothers split the work wrong. Re-estimating the ratio each
refresh (the first attempted fix) converged in 456 rounds but worse still
(43331, ratio falling to 0.16): the alternation must start from
F-fits-raw-y (MERF order), never from demeaned. So slice 1 keeps no in-loop
machinery at all. Verified on the pathology case before committing:
post-only gives trees=387, ratio=16, b_sd=2.6k, test=33498 -- matching
plain for ~zero extra cost. The full coordinate descent (refit F on y-b,
re-solve b) is slice-2 material, only if the gate shows post-only leaving
accuracy behind the joint optimum anywhere it matters. All booster.py
changes reverted; solver, API surface, and gate stand. The in-loop gate
numbers stay on file as the falsified baseline.

### Follow-up same day: one refinement round (gated, kept iff it wins)

Pure post-only gated 9W-9L vs ChimeraCat on synth (in-loop had 17W-1L):
where groups are strong, trees fit cleaner on group-adjusted targets, and
post-only's F fits group noise as if it were signal. The fix keeps the good
equilibrium (F learns X on raw y first) and adds exactly one refinement:
b solves against the winner, the already-scheduled full-data refit trains
on the adjusted target, b re-solves. If the gate shows refinement ~= top-up
alone, it comes back out (simplest wins); if it recovers the synth margins
while holding the real sets, it stays. No explicit-eval/ES-off path
refines (no refit runs there) -- those stay pure post-only.

### Follow-up same day: random-split validation (gate finding)

The refinement gate showed RE fitting ~1/4 the trees of ChimeraCat on synth
(48 vs 252 on gsyn:base): the group-split holdout withholds whole unseen
groups whose means F cannot predict, flooring the val curve and blinding
early stopping. Since the winner is a plain fit (no b exists during
training), it validates on a random split like one; groups enter only at
the solve. Forecast: synth tree counts normalize toward plain's and the
synth-vs-Cat gap recovers; real sets (informative val either way) hold.

### Superseded same-day note: ratio adaptation (kept for the record)

As first attempted, the ratio came from one REML on the raw grouped target.
The gate exposed the failure: on hc:house_prices_nominal (location signal
in raw y, house features correlated with neighborhood) the fit ran all 2000
rounds without early stopping ever firing, at 9x the plain fit's time and
worse accuracy. Diagnosis: the raw-y ratio (0.47, almost no shrinkage) stays
correct only while F knows nothing; as the trees learn X-signal the residual
between-group variation collapses and the ratio belongs ~10x higher. Fixed
too low, b re-absorbs group-predictable X-signal every refresh while the
trees re-learn it -- backfitting thrash, converging glacially if at all
(synth data never shows this: X is independent of groups there, so b steals
nothing). Forcing x10/x100 shrinkage confirmed it: convergence in 1094/123
rounds at better accuracy. So the loop now re-estimates the ratio on the
current residuals at every refresh -- the full coordinate descent the issue
blueprint describes ("alternate until convergence" with a REML solver), not
just the b-step. Cost: one O(G) REML per refresh, ~1s on the largest gate
set. Re-forecast: house converges normally; synth margins hold (the search
starts at the upfront value and adapts from there).
- RMSE only in slice 1 (`loss="RMSE"`; every other loss raises a clear
  error with the flag on). Single-model path only (`n_ensembles > 1` +
  the flag raises, same as `callbacks` today). `groups=None` + flag on
  raises: silence would fit a model that ignores the flag.
- Trees fit exactly as a plain model would (raw y, standard early
  stopping, standard refit, random auto-split); the group column is
  dropped, never encoded. `b` solves twice: pre-refit (so the refit
  trains on the group-adjusted target -- the refinement) and once more
  against the final model on the full-row residuals, so the shipped
  intercepts cover every group in X, including auto-split val groups
  the trees never trained on. No-refit paths stay pure post-only.

## Forecast (both axes, scored at verdict)

- Strength: large win on test rows from **seen** groups when the
  intra-class correlation is high (shrinkage beats both no-pooling and
  complete-pooling there — that is the mechanism), roughly flat on
  **unseen** groups (the model degrades to `F(X)`, trained on a
  group-adjusted target). The gate reports seen/unseen RMSE separately;
  a win concentrated anywhere else is a failed mechanism story.
- Fit time: ~+0-2% on grouped fits (one predict pass + one 1-D REML + one
  groupby after the fit), zero when the flag is off. No extra fits anywhere
  (B12 priced: the top-up rides after the single fit, never as a second
  fit).

## Barriers owed (written before any run)

`barrier_check.py` matched nothing on the idea sentence; manual read of
BARRIERS.md owes three arguments:

- B3 (partial CatBoost ports regress somewhere): this touches no
  categorical machinery and ships default-off, so the default path cannot
  regress — the bit-identical-off test is the proof, not a benchmark.
  CatBoost enters only as an arm in the new suite, not as transplanted code.
- B5 (shrinkage cannot correct a common in-sample bias): group intercepts
  genuinely vary across groups, which is exactly the varying component
  empirical Bayes shrinks; the common component stays in `F(X)`. And the
  referee cannot leak: auto-split validation groups are unseen by
  construction (`GroupShuffleSplit`), scored `F(X)`-only like any unseen
  group at predict time.
- B12 (portfolios die on cost): no auditions, no second fits; the cost is
  the forecasted +5–15% inside one fit.

GATE_ROBUSTNESS.md must be (re-)read before any number here decides the
ship — in particular the seen/unseen split guards the "how many
independent things" question.

## Gate: `benchmarks/grouped_suite.py` (to build, quantile_suite.py shape)

- Synthetic grouped generator with known truth: `y = f(X) + b_g + eps`,
  factors = group count, size skew, intra-class correlation, and whether
  `X` correlates with group membership. Truth-known, so the mechanism
  (shrinkage ≈ optimal pooling) is checkable, not just the RMSE.
- Real grouped sets: registry datasets with a natural cluster column
  (panel/store/subject IDs — the issue's examples), picked by rule, named
  in the verdict. Seen-group and unseen-group test RMSE reported
  separately on each.
- Arms: ChimeraBoost+RE vs ChimeraBoost with the group column as a plain
  categorical (ordered target stats — the in-house baseline the new
  machinery must justify itself against) vs LightGBM/CatBoost on the same
  column vs GPBoost if it installs (benchmarks may use anything; the
  pure-Python constraint is the library's, not the harness's).
- Harness-shaped JSON so `compare_runs.py` sign-tests it unchanged.

## Tests (slice 1)

- EB solver: closed form vs brute-force normal equations; limits (one-row
  group shrinks to ~prior, huge group tends to its mean); variance-ratio
  REML recovers a known ratio on synthetic draws.
- Integration: fit/predict round-trip with groups; unseen-group fallback
  exactly 0; determinism under seed; flag-off bit-identical to main;
  error paths (flag without groups, non-RMSE loss, bagged path).
- Leakage: a pure-noise high-card group column must not move validation
  loss (the ordered-TS analogue of `test_ordered_ts_resists_leakage`).

## Verdict (2026-09-20): SHIP as opt-in

Gate run `grouped-20260920-113455.json`: 11 sets (6 truth-known synth +
5 real hc:) x seeds 0-2, 5 arms. Statistic is dataset-level W-L over
seed-averaged RMSE (house convention); cell counts beside it. Syn cells
are independent draws (fresh data per seed: effective n=18); real cells
are re-splits (effective n=5 datasets -- a pointer, never a gate alone).

ChimeraRE vs each peer, dataset-level overall (syn / real split):

| peer | all (11 ds) | syn (6) | real (5) | read |
|---|---|---|---|---|
| ChimeraDrop | 10W-1L p=0.012 | 6W-0L | 4W-1L | decisive |
| LightGBMCat | 10W-1L p=0.012 | 6W-0L | 4W-1L | decisive |
| ChimeraCat | 8W-3L p=0.23 | 5W-1L (cells 14W-4L p=0.03) | 3W-2L | leans +, n.s. |
| CatBoostCat | 8W-3L p=0.23 | 6W-0L p=0.03 | 2W-3L | leans +, n.s. |

Median gap positive in all four comparisons (8+ of 11 datasets won).
Dropping any single dataset keeps the Drop/LightGBM verdicts (9W-1L,
p=0.02); no verdict rides one set. Wins concentrate where the
mechanism predicts: many-small 3W-0 and wine-reviews (10k groups)
3W-0 vs ChimeraCat; x-confounded 3W-0. Losses (house -4% and wine
-0.5% vs CatBoost, employee 0W-3L vs ChimeraCat) are the sets where
group effects plausibly interact with X -- intercepts-only cannot
express that, tree splits on the group column can. Unseen slice:
9W-2L vs Drop, 9W-2L vs LightGBM, 6W-5L vs Cat, 8W-3L vs CatBoost --
the F-only fallback holds, no unseen collapse anywhere.

Refinement ablation (`grouped-20260920-114053.json`, fresh seeds 3-5,
RE vs RE-postonly, the only difference the adjusted refit target):
syn sweeps 6W-0L on overall AND seen AND unseen (cells 15W-3L /
17W-1L / 16W-2L, all p<0.01), 1-4% per set; real flat 2W-3L, every
set within 0.5%. Per the pre-registered rule (stays iff it recovers
the synth margins while holding the real sets): STAYS. The unseen
sweep confirms the mechanism -- the refinement genuinely improves F,
it is not just a better b. Cost of the refinement: unmeasurable
(same two fits, different refit target).

Forecast score: strength HIT (large seen wins at high ICC: syn
sweeps + wine 3W-0; unseen flat-or-better as predicted); fit time
HIT, better than forecast (median RE/Cat fit ratio 0.95x -- RE fits
fewer trees with the group column dropped from F, and the solves
are O(N)). Random-split-validation forecast from the 2026-09-20
note: HIT (syn tree counts normalized, syn-vs-Cat recovered from
9W-9L cells to 14W-4L).

Robustness (GATE_ROBUSTNESS read 2026-09-20): effective n printed
above (trap 1/2); no one-set verdicts (trap 2, checked); no ties
(trap 4, continuous metric); same statistic both sides, both levels
labelled (trap 3/5); instrument pre-registered for this decision
(trap 6); refinement/postonly arms post-hoc and labelled so, with
the ablation confirmed on fresh draws (trap 8). Control: flag-off
is bit-identical by test, and low-ICC behaves (RE wins slightly,
no shrinkage pathology).

Mechanism confidence HIGH on synth (truth-known sweeps on all
three slices plus the ablation's unseen sweep); transfer confidence
MEDIUM (real pattern matches the intercepts-vs-interactions theory
but rests on 5 sets). Slice-2 question: random slopes for the
house/employee-shaped gap (group x feature interactions), before
classification.
