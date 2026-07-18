# Strength-axis legibility — make the Grinsztajn chart show winner/loser

Self-sufficient handoff (QUANT_PLAN/GROW_PLAN convention). Opened 2026-07-18
on Nathan's ask: "figure out a way to better represent winner/loser from
Grinsztajn — these all being at 99.x is not useful. Make a plan (don't act)."
This is a PLAN ONLY. No source change yet. Written for the next context window.

## The problem, quantified

`images/pareto.png` y-axis = blended strength = HarmonicMean(RMSE%, ⅔Brier% +
⅓F1%), where every % is "% vs best on that task" (`summarize._pct_vs_best`:
`100 * best / v`). On the latest 5-arm run (20260718-142950) the whole field is:

| model | blended |
|---|--:|
| ChimeraBoostEns8 | 100.0 |
| ChimeraBoost | 96.9 |
| CatBoost | 95.8 |
| sklearn_HGB | 94.8 |
| LightGBM | 94.8 |

Four of five models sit in a 2.1-point band (94.8–96.9). The chart already
band-aids this with a display-only log-gap-to-best transform (`_gap_to_y`),
and it STILL reads as "everyone's at 99.x." Root cause is structural, not
cosmetic: on near-Bayes-optimal tabular data the top models are genuinely
within a couple percent on most datasets, so a ratio-to-best metric saturates
near 100 by construction. No axis transform fixes a saturated metric — the
metric itself has to change to something ordinal.

## What's already available (confirmed by code read, 2026-07-18)

- `summarize._agg_metric(records, key)` → `{dataset: {model: mean_score}}` for
  rmse / brier / f1 / calibration. **The per-dataset per-model scores the rank
  needs already exist** — this is a pure re-analysis of existing result JSONs,
  NO new benchmark runs, offline, seconds to compute.
- `_pct_vs_best` is the saturation culprit; dominance/frontier are computed on
  raw blended, and the y-transform is display-only (`_gap_to_y`) — so the
  strength AXIS can be swapped without touching Pareto-dominance logic.
- Grinsztajn = reg + binary only (0 multiclass per CLAUDE.md), so per-dataset
  "primary metric" is clean: RMSE (reg), Brier (binary). Same machinery extends
  to the hc suite (which has multiclass) later.
- scipy ships with sklearn (available for stats); matplotlib already used. No
  new hard deps needed (critical-difference values can be hand-computed).
- NEAR_SOLVED exclusion (best NRMSE < 2%) already filters degenerate datasets;
  reuse the same scored-dataset set so rank and blended see identical data.

## Candidate representations

### A. Mean rank on the per-dataset primary metric  (RECOMMENDED axis)
Per dataset, rank models on that dataset's primary metric (average-rank for
ties); average ranks across datasets. Lower = better.
- Spread: ranks span 1..k (k≈5). Illustrative (NOT yet computed): Ens8 ~1.3,
  ChimeraBoost ~2.6, CatBoost ~2.9, HGB ~3.9, LightGBM ~4.0 — the full axis,
  not a 2-point smear.
- Winner/loser: this IS the winner/loser ordering, directly.
- Significance: bootstrap over datasets (resample datasets w/ replacement,
  recompute mean ranks, 2.5/97.5 pct) → error bars per point. (Textbook
  alternative: Friedman + Nemenyi critical-difference; CD = q·√(k(k+1)/6N),
  q from a small constant table — hand-computable, no dep. Bootstrap preferred:
  no q-table, gives per-point CIs the chart can draw.)
- Pros: field standard (Demšar 2006); decompresses fully; robust to metric
  scale; ALIGNS the headline chart with how we already gate ships (per-dataset
  sign tests in compare_runs.py are rank/winner-loser in spirit).
- Cons: discards magnitude (win-by-hair == win-by-mile); needs one primary
  metric per classification dataset (see decision D2); rank is pool-relative
  (adding/removing a model shifts ranks — same caveat as Elo).

### B. Internal Elo / Bradley–Terry on Grinsztajn (like our TabArena chart)
Pairwise per-dataset win/loss → fit Bradley–Terry → Elo; bootstrap CIs.
- Spread: hundreds of Elo points — the exact readable feel of
  `images/tabarena_pareto.png`.
- Pros: consistent with the TabArena chart; single interpretable number;
  natural CIs; handles ties via a near-tie margin.
- Cons: heaviest option; needs a per-dataset win + near-tie definition; scale
  is relative; more to explain than rank. Overkill for 5 models.

### C. Head-to-head win-rate matrix  (RECOMMENDED companion figure)
Win rate = fraction of (dataset × opponent) comparisons a model wins; matrix
shows every pairwise rate. Ties by a small relative margin = ½ each.
- Pros: literally "who beats whom," maximally intuitive, great README panel,
  reuses the compare_runs.py sign-test idea; a single "win-rate vs field"
  scalar can annotate the main chart.
- Cons: single win-rate still bunches close models (0.45–0.65); the matrix is
  a companion, not a Pareto axis.

### D. Status quo (blended-% + log-gap transform) — REJECT as primary
Already in place; the ask is precisely that it's not useful. Keep blended-% as
a REPORTED diagnostic (the harmonic-mean "which leg is weak" signal is still
worth having in the text table), not as the headline axis.

## Recommendation — RESOLVED by Nathan 2026-07-18: win percentage

Nathan (2026-07-18): "avg rank is fine But I feel like percent of times won
is a little more intuitive." → **Headline axis = WIN RATE: percent of
head-to-head (dataset × opponent) matchups won**, higher = better.

Key fact (why nothing is lost vs Option A): pairwise win rate vs the field
is mean rank linearly transformed — winrate = (k − mean_rank)/(k − 1) with
ties as ½ — so it carries EXACTLY the ordinal information of mean rank in
friendlier units (0–100%, 50% = mid-pack, "wins 75% of its matchups").
Implementation is Option A's machinery with a final rescale.

1. **Headline chart y-axis → pairwise win rate (%)**, bootstrap CI whiskers
   (resample datasets), good corner up-and-left as today. Frontier/dominance
   on (winrate, slowdown). Exact per-dataset score ties between two models
   count ½ each (rare on real data; note in the figure caption if any occur).
   NOT "% of datasets won outright" (rank-1 share) — that statistic is harsh
   and degenerate with a strong Ens8 arm in the pool (everyone else ≈ 0%);
   record it in the text table if desired, never as the axis.
2. **Add a head-to-head win-rate matrix (Option C)** as a companion figure
   (`images/winrate.png` or a second panel) for the README — the axis scalar
   is this matrix's row mean, so the two figures agree by construction.
3. **Keep blended-% + per-metric % in the text table** as diagnostics (weak-leg
   signal), and keep it computable. Elo (Option B) is NOT built now — record as
   a future option if win rate proves too coarse.

## Decisions for the user (confirm at the START of the build session)

- **D0 — Axis statistic: RESOLVED (win rate, see Recommendation).** Nathan
  2026-07-18. Remaining decisions below are still open.
- **D1 — Does the DECISION metric change too, or just the chart?** The Pareto is
  the north star we ship against. Recommendation: chart moves to rank for
  LEGIBILITY, but ship-gating stays as-is (sign tests on both suites + blended
  as diagnostic) — rank and sign-tests already agree in spirit, so this is
  low-risk and doesn't re-litigate every past verdict. Flag: if the chart shows
  rank while we cite blended, keep the text table showing both so they never
  tell contradictory stories. (If Nathan wants rank to BECOME the steering
  metric, that's a bigger, separate call — note and stop for confirmation.)
- **D2 — Classification primary metric for ranking:** Brier only (proper scoring
  rule, matches the ⅔ Brier weight, calibration-honest) vs a Brier+F1 blend vs
  rank-Brier-and-F1-separately-then-average. Recommendation: **Brier only** —
  cleanest, avoids F1 threshold-noise churning the ranks; F1 still shown in the
  text table.
- **D3 — Replace pareto.png or add rank_pareto.png?** Recommendation: add a
  `--metric rank|blended` flag to make_pareto.py (one source of truth), default
  the committed headline to rank, keep blended reachable. README headline swap
  is then a one-line image path change.

## Phase plan (once D1–D3 are answered)

- **Phase 0 (no chart yet):** add `summarize.per_dataset_ranks(records)` +
  `mean_rank(...)` + `winrate_matrix(...)`; print a text table of mean rank
  (+ bootstrap CI) and the win matrix on the latest 5-arm run. SANITY GATE:
  Ens8 ranks best, rank order broadly agrees with blended order, spread visibly
  wider than 2 points. If rank does NOT separate the pack meaningfully, stop and
  reconsider (fall back to Elo, Option B).
- **Phase 1:** wire the rank axis into make_pareto.py behind `--metric`, CI
  whiskers, relabel axis/corner (down-left = best), frontier on (rank, slowdown).
  Regenerate images/pareto.png (or rank_pareto.png per D3).
- **Phase 2:** win-rate matrix companion figure + README wiring; refresh the
  README headline if rank becomes the default view.
- **Tests:** rank/winrate are deterministic given a results JSON — add a small
  fixture test (known scores → known ranks/winrates, average-tie handling,
  near-solved exclusion honored). No goldens touched (benchmarks-only).

## Constraints / notes

- Pure-Python/benchmarks side only; matplotlib + (optional) scipy.stats, no new
  hard deps. Library source is NOT touched — this is reporting.
- Grinsztajn only here; TabArena stays sealed (its chart already uses Elo, so
  don't conflate). hc suite can get the same treatment in a follow-up.
- Data of record: the fresh 5-arm run 20260718-142950 (today). Re-analysis is
  offline; no benchmark run needed to build or test this.
- Keep make_pareto.py the single source of truth for the metric math
  (make_tabarena_pareto.py is a separate, hardcoded-DATA chart — leave it).

## Checklist

- [ ] D1–D3 answered by Nathan
- [ ] Phase 0: rank + winrate functions + text table; sanity gate passes
- [ ] Phase 1: rank axis in make_pareto.py (--metric), CIs, frontier, image
- [ ] Phase 2: win-rate matrix companion + README
- [ ] Tests: rank/winrate fixture test green
- [ ] Close: headline chart legible (winner/loser visible), record the choice
