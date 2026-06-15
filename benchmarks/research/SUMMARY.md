# CatBoost-gap cascade — verdict ledger

Pre-registered ideas run through the efficient cascade (`cascade.py`). Each was a
default-OFF flag, byte-identical no-op when off, tested, then judged on paired
validation curves (T0) and/or held-out test metric with a sign test (T1). Lower
is better throughout. **Guardrail: TabArena never touched; OpenML T1 is a one-shot
gate, not an iteration target.**

| Idea | Flag | Tier | Verdict | What happened |
|---|---|---|---|---|
| C1 one-hot low-card | `onehot_low_card` | T0 | **KILL** | helps splice/car/adult, regresses bank-marketing (3/8) |
| C3 selective cat-combos (mixed) | `cat_combinations_selective` | T0 | **KILL** | MI-pruning *hurts* all-cat sets (kr-vs-kp +27%); auto-rule already right (1/8) |
| G1 forest joint leaf refit | `forest_leaf_refit` | T1 | **KILL** | overfits (train↓ test↑ +6–14%); binary gain already in `linear_leaves` |
| G4 ordered + leaf machinery | `ordered_leaf_estimation` | T0 | **KILL** | helps some cats, badly hurts kr-vs-kp (+17%); ordered-boosting off by default for a reason (3/8) |
| G3 adaptive leaf-estimation | `adaptive_leaf_estimation` | T0 | **KILL** | flat — size-scheduling the Newton steps buys ~nothing (1/8, mean −0.11%) |
| G2 mass-adaptive leaf shrinkage | `adaptive_leaf_shrinkage` | T0 | **KILL** | helps car (−4.3%), hurts kr-vs-kp (+6.1%) (2/8) |
| C4 cat-aware binning | `cat_aware_binning` | T0 | **KILL** | helps car/splice, hurts kr-vs-kp (+14.7%) (3/8) |
| C2 per-tree TS permutation | `cat_pertree_permutation` | — | **DEFERRED** | needs per-round re-encode — a fundamental break from encode-once-then-bin; not justified for a near-certain kill |

## Headline finding

Seven CatBoost-inspired / scheduling levers, seven kills (C2 deferred by
architecture). Each partial mechanism port **regresses somewhere** (or is flat)
while the wins it chases are **already banked in the current defaults**:
- the all-categorical `cat_combinations` auto-rule (C1/C3 can't beat it),
- the binary `linear_leaves` default (G1's leaf-sharpness target),
- plain boosting with size-adaptive `min_child_weight` (G4's ordered variant loses).

The TabArena CatBoost gap does **not** appear closable by incremental ports of
isolated CatBoost mechanisms into this pure-Python oblivious architecture — the
gap looks like an emergent property of CatBoost's *integrated* ordered-boosting-
on-permutations machinery operating together, not any single transplantable part.

**The canary: `kr-vs-kp`.** Nearly every categorical lever *helps* car/splice but
*regresses* kr-vs-kp (one-hot, combos, mass shrinkage, ordered+leaf, more bins all
make it +6% to +27% worse). Its ordered-TS encoding is already near-perfect (Brier
~0.024), so any added categorical structure just injects variance it overfits.
That single dataset is the clearest evidence the defaults are at a good optimum.

## Engine notes
- One fit = whole `validation_history_` curve; paired same-split deltas; shared
  baseline cache → each verdict reached in ~25–60s (T0) for ~0 marginal cost.
- Post-fit ideas (G1) are invisible to the per-round curve → routed straight to
  the promotion tier (`post_fit=True`).

## Queue complete
Every pre-registered lever has been run (C1, C3, C4, G1, G2, G3, G4) or deferred
with cause (C2). All seven runnable levers killed. The remaining honest directions
are: (a) study CatBoost's machinery *as a whole* — integrated ordered boosting on
permutations, which would be a from-scratch redesign, not a flag; or (b) accept
that ChimeraBoost's identity (pure-Python, oblivious, fast, near-optimal defaults)
is simply a different, valid point on the Pareto front than CatBoost's TabArena
Elo.

**Update (2026-06-15):** the eight default-off flags these levers shipped behind
(`hs_lambda`, `adaptive_leaf_shrinkage`, `adaptive_leaf_estimation`,
`ordered_leaf_estimation`, `forest_leaf_refit`, `onehot_low_card`,
`cat_combinations_selective`, `cat_aware_binning`) were **removed from the library**
to cut API surface (constructor 36 → 24 params). The cascade's verdicts stand as the
record; the narrow car/splice wins are documented here and recoverable from git
history if ever revisited. This ledger is kept as the research log, not live API.
