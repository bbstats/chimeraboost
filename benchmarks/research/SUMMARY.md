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

## Headline finding

Four CatBoost-inspired levers, four kills. Each partial mechanism port **regresses
somewhere** while the wins it chases are **already banked in the current defaults**:
- the all-categorical `cat_combinations` auto-rule (C1/C3 can't beat it),
- the binary `linear_leaves` default (G1's leaf-sharpness target),
- plain boosting with size-adaptive `min_child_weight` (G4's ordered variant loses).

The TabArena CatBoost gap does **not** appear closable by incremental ports of
isolated CatBoost mechanisms into this pure-Python oblivious architecture — the
gap looks like an emergent property of CatBoost's *integrated* ordered-boosting-
on-permutations machinery operating together, not any single transplantable part.

## Engine notes
- One fit = whole `validation_history_` curve; paired same-split deltas; shared
  baseline cache → each verdict reached in ~25–60s (T0) for ~0 marginal cost.
- Post-fit ideas (G1) are invisible to the per-round curve → routed straight to
  the promotion tier (`post_fit=True`).

## Still pending (lower priority)
C2 per-tree TS permutation, C4 cat-aware binning, G2 adaptive leaf shrinkage,
G3 adaptive leaf_estimation. Given 4/4 kills, expectations are low; run them to
confirm, not to hope. The higher-value direction is studying CatBoost's machinery
*as a whole* rather than porting parts.
