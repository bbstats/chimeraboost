# A1 — vector-leaf multiclass with sketched split scoring (flagship)

Self-sufficient handoff (M1_PLAN.md convention). From the 2026-07-24
literature-sweep ranked program (memory: project-tabfm-literature-sweep).
Everything below is registered BEFORE any library change or benchmark run.

## Goal and evidence base

Today `MulticlassBoosting` fits **K trees per round** (one per class): K
histogram passes, K forest walks at predict, and multiclass is the weakest
leg everywhere it is measured — TabArena per-type Elo 1156 (vs binary 1306 /
reg 1196), hc multiclass = CatBoost's crown (M1 narrowed it, did not topple
it). A1 replaces the round with **one tree whose leaves are K-vectors**,
splits scored on a random projection ("sketch") of the K gradient columns:

- SketchBoost (2211.12858): sketched vector-leaf GBDT ≥ parity log loss vs
  one-tree-per-class with large fit/predict speedups; beat CatBoost on
  Otto/Dionis. GBDT-MO (1909.04373) and CatBoost's own MultiClass mode agree
  vector leaves are not a quality sacrifice. Three independent groups.
- Oblivious-natural: the whole tree is already one shared structure; a
  K-vector leaf table is the smallest possible extension of the layout.
- Attacks BOTH Pareto axes on the multiclass slice: fit (1 histogram pass
  per round instead of K) and predict (1 forest walk instead of K).

## Pre-registered design (no new public knobs)

1. **Sketch, s=1, Rademacher.** Per round draw r ∈ {−1,+1}^K from the fit's
   rng stream. Split-scoring gradient g_i = Σ_k r_k·grad_ik; split-scoring
   hessian h_i = Σ_k hess_ik·coupling (coupling = (K−1)/K, unchanged). With
   Rademacher entries r_k² = 1, so h_i is exactly the projected curvature
   rᵀdiag(H_i)r — the (g,h) pair is a principled 1-d Newton sketch whose
   gain estimates the true vector gain in expectation (SketchBoost's Random
   Projections, k=1). **The existing scalar split kernels — fused
   build/split/descend AND the quantized path — are reused verbatim; zero
   new split-search code.**
2. **Vector leaf values.** On the shared leaf partition, per class:
   v_k = −lr·G_k/(H_k + l2) with per-class coupled hessians — today's
   Newton semantics exactly, new kernel `_leaf_values_vec` → (n_leaves, K).
   Train update F += values[leaf]; val update via the tree's leaf assignment.
3. **Predict.** New `pack_forest_vec` + `_predict_forest_vec_rm(_serial)`:
   one walk per tree, K adds from a leaf-major (n_leaves·K) value block.
   Serial twin dispatched at `_SERIAL_PREDICT_N` like the scalar path.
4. **MVS subsample** runs once per round on the sketched (g,h) instead of
   per class. **ordered_boosting**: per-class LOO via the existing
   `_loo_leaf_step` on the shared leaf assignment (K calls).
5. **Structure/compat.** `trees_` becomes a flat list of vector trees
   (`feature_importances_`'s non-list branch already handles it).
   `_predict_raw_impl` keeps an isinstance fallback for models pickled from
   ≤0.24.0 (rounds-of-K lists). Selection races, refit_full, temperature
   scaling, bagging all flow through untouched — they only consume
   `valid_history_` / `predict_raw` / `feature_importances_`.
6. **Depth-0 round** = stop with best-prefix truncation (the scalar
   booster's rule). Watch item: a depth-0 caused by one unlucky projection
   rather than convergence; smoke checks round counts vs BASE.
7. **Goldens:** multiclass goldens re-bless at ship (intended behavior
   change); every reg/binary golden and numerical-identity test must stay
   bit-identical (those paths are untouched).

Registered contingency (not the default path): if tier 1 shows a real
multiclass quality regression, audition sketch dim s ∈ {2,4} (needs a
vector-histogram kernel — new work) before killing. One contingency only.

Out of scope (registered): leaf_estimation_iterations / linear_leaves for
multiclass, min_child_weight retuning (the vector tree sees row-summed
hessian mass ~1−Σp², binary-like, vs per-class ~p_k(1−p_k) today — a
semantics shift the screen must absorb, recorded as a watch item, no knob
changes), suite composition changes, TabArena in any form.

## Treatment surface

- Tier 1 (synth screen, 136 sets): 34 multiclass = treatment (3 canaries
  017/117/317 stay at-ceiling); 102 reg/binary = exact-tie identity surface.
- Grinsztajn: zero multiclass → pure identity canary, 59/59 exact ties.
- hc: 4 multiclass sets (okcupid-stem, Traffic_violations, cjs, eucalyptus)
  = treatment; 10 others = exact ties.
- OpenML gate: 9 multiclass of 29 = treatment; 20 exact ties.
- PMLB: not used (no HP tuning in A1).

## Pre-registered predictions

- Quality: multiclass slice ≈ parity or better (the SketchBoost result);
  Brier read at tier 1 (standing B1 lesson) also ≈ parity. Gains, if any,
  concentrated where K is large (shared structure + all-class updates per
  round act as regularization; more rounds under the same ES budget).
- Speed: multiclass fit time down — roughly the histogram share × (K−1)/K;
  expect ≥1.5× on K≥5 sets, less on K=3. Multiclass predict down toward
  K× fewer walks. Reg/binary timings unchanged (identical code path).
- ES round counts: comparable or moderately higher (each round now spends
  1 tree, not K; capacity per round is lower but denser per class).
- Exact ties: every reg/binary set, every tier, both metrics. Any broken
  tie = implementation bug → run void as evidence; fix and re-screen.

## Kill bars (registered before any run)

A1 is a Pareto change: quality parity at a clear speed win SHIPS; quality
win at speed parity SHIPS; both flat = KILL (churn); quality loss = KILL
(after the one registered contingency).

- **Tier 1:** multiclass slice (31 non-canary sets) primary (F1) mean not
  negative beyond noise AND losses not exceeding wins by a decisive sign
  test (p<0.05 against us = KILL); Brier mean not negative beyond noise;
  canaries at-ceiling; 102/102 reg/binary exact ties. AND a real speed
  win: mean multiclass fit time ratio ≤ 0.8× OR quality decisively
  positive (wins>losses and mean>0).
- **Tier 2:** gr 59/59 exact ties. hc: 10 non-multiclass exact ties; the 4
  multiclass sets not net-negative pooled over 3 seeds (primary + Brier);
  a decisive multiclass quality loss = KILL even with green speed.
- **Gate (one-shot, last):** multiclass sets pooled primary not negative
  beyond noise; every other set exact ties.
- Speed alone cannot kill; a fit-time REGRESSION on multiclass (>1.1×)
  = stop and investigate before tier 2.

## Protocol (per /experiment)

1. Implement on branch `a1-vector-leaf`; full test suite green (new tests:
   vector leaf-value kernel vs per-class oracle on a shared partition,
   packed vector predict vs per-tree loop bit-identity, pickle back-compat,
   reg/binary bit-identity vs main). Smoke on 1-2 hc multiclass sets (round
   counts, fit/predict time, holdout sanity — not decision-grade).
2. Tier-1 synth screen: NEW `--synth --seeds 3 --models ChimeraBoost
   --save` vs newest clean single-arm BASE (bit-comparable since defaults
   are byte-identical post-0.24.0/refit_full-off; else run a fresh BASE
   from a main worktree, PYTHONPATH set, `chimeraboost.__file__` printed).
   Read: compare_runs overall + multiclass/reg-binary slices + Brier +
   synth_report factor attribution (effect must concentrate on multiclass;
   K-large slices strongest).
3. Tier 2, sequential, one benchmark at a time: hc 5-arm canonical vs
   newest clean hc BASE (LightGBM cross-run canary must tie), then gr
   single-arm identity vs newest clean gr BASE (59/59; zero treatment
   surface → competitor arms buy nothing, the M1 precedent).
4. OpenML one-shot gate: fresh BASE from main worktree vs NEW, arms
   ChimeraBoost + LightGBM, seeds 3. Never re-run.
5. Ship: docs (parameters/FAQ if any wording touches multiclass), CHANGELOG
   [Unreleased], verdict here + memory, /pareto refresh (hc multiclass
   speed moves the slowdown axis; gr chart identical by construction).
   TabArena re-read only after release, per the vow.

Aggregate table printed after every run, per standing rule.

## Acceptance checklist

- [ ] Implementation on branch `a1-vector-leaf` + tests green
- [ ] Tier-1 screen vs kill bar
- [ ] Tier-2 hc + gr identity
- [ ] OpenML one-shot gate
- [ ] SHIP or KILL recorded here + memory; pareto/CHANGELOG on ship
