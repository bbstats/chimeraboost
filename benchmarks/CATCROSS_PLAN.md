# CATCROSS — group-centered categorical crosses (op="gdiff")

Pre-registered 2026-07-20, BEFORE any suite result. Branch
`worktree-pareto-catcross`. Companion probe: `probe_catcross.py` (run first,
results below). The branch also carries an independent bit-identical speed
lever (audition train-loss skip) — accuracy sign tests are unaffected by it;
suite fit-time reads are the NET of both levers, which is what would ship.

## Mechanism

Oblivious trees share one split across a level, so "is this row's numeric
value above ITS CATEGORY's baseline" needs a per-category staircase — the
worst case for the tree family, and the num×cat analog of the staircase gap
that numeric cross features (diff/prod, 2026-07-13) fixed for num×num. A
gdiff column `x_i − mean_fit(x_i | c_j)` makes the within-category deviation
one split. Target-free (no leakage machinery, same map at fit and predict),
weight-aware, raced by the existing validation selection (top-4 numerics ×
top-3 cats by base-fit importance, ≤12 columns alongside the ≤30 diff/prod).
CatBoost's surviving Brier/RMSE edge lives on real high-card entity data
(hc suite, 86–88% CB Brier winrate) — exactly where per-entity baselines
differ most.

## Probe result (external augmentation, no selection, 3 seeds)

8/14 hc sets better, 4 worse; wins broad but small (+0.1..+0.9% primary);
one single-seed blowup (employee_salaries s2, −46%) of exactly the variance
class the selection race exists to referee. Proceed to tiers.

## Predictions (registered)

- T1 synth: gains concentrated on entity-cat slices (entity_strength high);
  **no-cat sets are structurally bit-identical → exact ties expected there
  (internal control: any no-cat delta = bug/noise)**; canaries flat.
- T2 hc: positive sign test — the target regime. Binary/multiclass Brier and
  reg RMSE on entity sets (kick, black_friday, wine-reviews,
  Traffic_violations) are where wins should sit.
- T2 Grinsztajn: near-inert (low-card cats); cat-variant sets may move a
  little; sign test must be non-negative-ish (no broad regression).
- Gate (--openml, one-shot): non-negative; cat-bearing sets may win.
- Speed: augmented fits pay ≤12 extra columns + a pandas groupby per pair;
  train-loss skip refunds part. Suite fit sums should move single-digit %.

## Kill clauses (registered)

- T1: canaries move, no-cat sets not exact ties, or entity slices show no
  concentration (mechanism absent).
- T2: hc sign test negative, or Grinsztajn broadly negative (the C1/C3
  signature: helps a few, taxes the rest).
- Gate: clearly negative (Nathan's precedent: uniform-direction small losses
  killed B1's final variant).

## Ship bar

hc decisive + Grinsztajn non-negative + gate non-negative. Exact gr-vs-hc
weighting = Nathan's call (per /experiment); recommendation recorded at
verdict time.

## Results log

(appended as tiers complete)
