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

### T1 synth screen — PASS (2026-07-20)

Variant `results/20260720-195938.json` vs base `20260720-100716.json`
(NOT 100831 — that file is the H1/H2 lei=1 EXPERIMENTAL arm; see lesson).
- 9W / 5L / 122T, mean +0.072%.
- Structural controls perfect: cats=none 0-0-79 EXACT ties, n<2000 0-0-48
  exact ties (row gate respected), canaries 0-0-3 flat.
- Concentration as registered: cats=entity 8W-2L +0.235%; entity_strength
  top OLS factor (t=+2.23, positive). card>16 +0.355%.
- Engagement sparse on synth (most cat sets < 2000 rows) → magnitude
  understates; hc is the target regime.

### T2 Grinsztajn — 59/59 EXACT TIES = structurally inert (2026-07-20)

Variant `results/20260720-204829.json` vs base `20260720-101306.json`:
0W/0L/59T. Root cause verified in-process (check_gr_engage): the Grinsztajn
loaders return cat=None even for the `_cat` suite variants (categoricals
arrive pre-encoded as numerics), so cat_features is never passed and gdiff
candidates never exist there. The headline chart is untouched BY
CONSTRUCTION (M1 precedent); the suite cannot express this lever — exactly
the blind spot the hc suite was built for. Grinsztajn non-negative bar:
passed in the strongest possible form.

LESSON (baseline hygiene): the 2026-07-20 morning result pairs are the
H1/H2 lei A/B arms and carry NO config flag marking the experimental arm
(code-level PYTHONPATH arms). 100831 = lei=1 arm (differs on small binary);
100716 = default arm (matches 07-19 certified runs bit-exactly). Same
contamination in the hc pair: 101508 vs 101537 differ on 8 binary cells
(kick, sf-police, kdd_ipums); clean member identified by a main-code
fingerprint run before the hc comparison. gr pair members both clean (lei
fully shadowed there, both match 07-19 exactly).
