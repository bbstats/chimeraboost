# ChimeraBoost — Project Status & Briefing

*Last updated 2026-06-02. Self-contained handoff doc: what the model is, what we've
shipped, what we've tried and rejected, where we stand, and the benchmarking loop we
develop against.*

---

## 1. What ChimeraBoost is

A from-scratch gradient-boosted decision tree library (NumPy + Numba JIT, no C++),
modeled on **CatBoost's oblivious-tree design** but with our own pipeline. Goal:
**sklearn-class accuracy at near-LightGBM speed**, with first-class categorical
handling and good probability calibration. Public API mirrors CatBoost/sklearn
(`ChimeraBoostRegressor`, `ChimeraBoostClassifier`).

Core design choices (all deliberate, all validated):

- **Oblivious (symmetric) trees** — every node at a given depth splits on the *same*
  (feature, threshold). A tree of depth `d` is just `d` splits; a leaf is a `d`-bit
  number. This is the source of our speed and a big chunk of our regularization. It is
  also the source of our one structural weakness (see §5, Brier). We have *confirmed by
  code audit* there is no leaf-wise code anywhere — everything below lives inside the
  oblivious design.
- **Ordered Target Statistics** for categoricals (`OrderedTargetEncoder`,
  multi-permutation, `n_permutations=4` like CatBoost).
- **Leave-one-out (LOO) leaf correction** as our tractable stand-in for CatBoost's
  ordered boosting (true ordered boosting needs O(log n) model snapshots; not doable as
  post-processing — see §5).
- **Feature-major binned matrix** + **fused Numba forest predictor** (parallel over
  *samples*, not trees) for speed.
- **Temperature scaling** on `predict_proba` for calibration.

### Current shipped defaults (HEAD, main)

| Param | Regressor | Classifier |
|---|---|---|
| `depth` | 6 | 6 |
| `learning_rate` | auto (0.1 w/ early stop) | auto |
| `l2_leaf_reg` | 1.0 | 1.0 |
| `max_bins` | 128 | 128 |
| `ordered_boosting` (LOO) | False | False |
| `leaf_estimation_iterations` | 1 | 3 |
| `min_child_weight` | 1.0 | None → size-adaptive `_auto_min_child_weight(n)` |
| `n_estimators` / patience | 2000 max, patience 50 | same |

`_auto_min_child_weight(n) = clip((2000 − n)/1500, 0, 1)`: full veto (~1) below ~500
training rows, 0 above ~2000. This is one of our most important defaults (see §4).

---

## 2. The north-star metric (how we steer)

One number: **blended model strength vs slowdown**, plotted as a Pareto
(`benchmarks/make_pareto.py` → `images/pareto.png`). All inputs are "% vs best model on
that task type", higher = better.

```
classification = ⅔·BinBrier%  +  ⅓·BinF1%
blended        = HarmonicMean(RegRMSE%, classification)
slowdown       = mean fit-time multiple vs the fastest model (lower = better)
```

The harmonic mean is deliberate — it collapses toward the **weaker leg**, so raising
the blended number *forces* us to fix our worst task rather than pad our best.

**Current standing** (`results/20260601-154554.json`, 3 seeds): ChimeraBoost is
**ON the Pareto frontier** — all four models are non-dominated.

| Model | Blended | Slowdown |
|---|---|---|
| CatBoost | 99.0 | 12.7× |
| sklearn_HGB | 98.1 | 4.1× |
| **ChimeraBoost** | **98.0** | **2.6×** |
| LightGBM | 97.9 | 1.0× |

We are the **2nd-fastest** (beat sklearn *and* CatBoost on speed) and 3rd-strongest,
sitting right on the frontier between LightGBM (fast/slightly weaker) and sklearn
(slower/slightly stronger).

---

## 3. Benchmark suites (the development pipeline)

| Suite | Size | Role today |
|---|---|---|
| **Synthetic** | 7 datasets (`make_regression`, friedman1, breast_cancer, wine, cat_*) | mechanism + size-curve probes; cheap inner loop |
| **Small dev panel** | ~6 real datasets | fast inner-loop sign tests |
| **Grinsztajn** (inria-soda HF mirror) | 59 datasets (36 reg / 23 binary) | the **breadth** reference; the standard for shipping decisions |
| **OpenML suite** | 34 datasets (incl. multiclass) | **independent one-shot gate** — pass/fail, never iterated against |
| **TabArena-Lite** | 51 tasks | **sealed holdout** for a published Elo (see §7) |

Methodology we reuse for every change: **synthetic (mechanism + size-curve) → small
dev panel (inner loop) → full Grinsztajn (breadth, sign test) → OpenML (one-shot
gate)**. A change ships only if a sign test passes over per-dataset deltas (≥~15/27 on
the expanded suite), one change at a time, hypothesis pre-committed before seeing
results.

---

## 4. What we've shipped (accepted wins)

In rough chronological order; each survived the full pipeline + sign test.

**Correctness / algorithm**
- **min_child_weight across all leaves** — a split must satisfy the hessian constraint
  on *all* non-empty leaves, not just one. Fixed sparse-leaf overfitting.
- **Multiclass (K−1)/K hessian coupling** — softmax hessian is rank K−1; matches the
  CatBoost paper formula.
- **Multi-permutation ordered TS** (`n_permutations=4`) — √K variance reduction on
  categorical encoding.
- **MVS (Minimum Variance Sampling)** for `subsample<1` — opt-in, default off.
- **Empty-child exemption** (the big one): in an oblivious tree a leaf that sends all
  samples one way (a *pure* leaf) is normal and must NOT veto the shared split via
  min_child_weight; only a genuinely *sparse* non-empty child is illegal. This removed
  a hidden **depth cap** (effective depth was self-capping at ~4 even when depth=6/8/10
  was requested). Result: **Reg RMSE 95.7→98.0%** (now 2nd-best, beats sklearn &
  LightGBM), F1 98.7→99.2%, Brier 94.2→95.7%, *and faster* (more expressive trees
  early-stop sooner). **This flipped the whole picture: regression went from our main
  gap to a strength.**
- **Size-adaptive `min_child_weight`** for the classifier (the `_auto` formula above) —
  full-depth trees on big data without overfitting small data. Binary Brier +1.6pp,
  blended 97.4→98.0, onto the frontier.
- **l2_leaf_reg 3.0 → 1.0** — Brier 95.7→97.2% (+1.5pp), RMSE/F1 flat. Pulled the
  classification leg even with LightGBM.

**Speed (all bit-identical or ULP-level, all accepted)**
- JIT `_sigmoid` + skip unused `train_history_` — 43% faster on classification.
- Feature-major (transposed) binned matrix — ~15–20% faster tree build, the
  LightGBM/XGBoost memory layout.
- Default learning rate → 0.10 with early stopping — ~44% fewer trees, same accuracy.
- Reuse training-set leaf assignment + fused LOO leaf-step kernel + **fused forest
  predictor** (parallel over samples) — 5.7× faster tree-walk at predict time.
- Net effect: ChimeraBoost went from ~6× faster than CatBoost to ~30–36× faster, and
  from 4.9× → ~2.5× slowdown vs the fastest model.

**Calibration**
- **Temperature scaling** on `predict_proba` (fit T on validation to minimize log
  loss). Monotonic, so F1/accuracy/`predict()` bitwise unchanged. Best MCB (calibration)
  in the field.

**Infra**
- Process-parallel benchmark harness (`run_benchmarks.py --jobs`), `summarize.py`
  (before/after compare), `make_pareto.py`, `make_tables.py`, `/bench` status command,
  near-solved-dataset guard (see §6), `bin_centers_` + `is_numeric_binned_` zero-cost
  infrastructure.

---

## 5. What we've tried and REJECTED (don't re-suggest without a new angle)

This list is the most valuable part for a fresh collaborator — these are dead ends we
*tested rigorously*, not hunches.

**Algorithmic**
- **max_bins 128 → 254/256** — tested on 3 independent suites. Looked good on Grinsztajn
  but did NOT generalize (regression *reversed* to −4.8%; `cpu_act` −30%, `madelon` −10%
  — fine bins overfit noise). ~30–50% slower. The Brier gap is structural, not
  resolution. **Don't retry.**
- **Classifier ordered_boosting=True** — rejected. Confound: OB=True *skips*
  `leaf_estimation_iterations` (mutually exclusive in the booster), trading away ~+1pp
  Brier from `lei=3`. Lost on exactly the high-signal cluster it should help.
- **Adaptive depth (size-based, `depth=8` for n≥20k)** — strong on Grinsztajn but the
  OpenML gate was thin/inconclusive. Diagnosis: the Grinsztajn wins were large *AND
  high-signal* (covertype/road-safety/electricity); a **size-only rule is a proxy for
  signal that doesn't generalize** (adult/bank-marketing are large but low-signal → no
  gain). Reverted. *If revisited, the real lever is signal, not size.* Lives on as a
  user knob (`depth=8`).
- **SAS (sharpness-aware / histogram smoothing)** — `[0.15,0.70,0.15]` kernel on the
  numeric split scan. Decisively rejected (17W/42L, −0.46%). Histogram bins represent
  real signal, not measurement noise, so smoothing biases split selection. *Don't retry
  histogram smoothing without smoothing in gradient space instead of bin space.*
- **CatBoost meta-trained adaptive LR formula** — decisive reject (5/24). Their
  coefficients were meta-trained against *their* full pipeline; category error to port.
- **Cross-fold ordered boosting** (and 2-fold) — rejected. Uses n/2 per Newton step →
  doubles gradient variance; the bias reduction doesn't compensate. **LOO wins because
  it keeps n−1 samples.** All simple ordered-boosting variants are now exhausted; true
  CatBoost-style needs O(log n) snapshots.
- **Feature combinations** (pairwise cat×cat) — opt-in only; helps all-categorical
  (`car` +14% F1) but crowds out numerics on mixed data.
- **min_child_weight 0.5 / 0.0 global, adaptive-LR variants** — global flips fail the
  sign test on small data; the *size-adaptive* version is what shipped.

**Speed (all regressed at our scale — Numba/cache-resident, not the C++/>1M-row regime)**
- uint8 bin indices, float32 histogram buffers (quantization) — Numba indexes int64
  internally; arrays already cache-resident.
- Histogram subtraction trick (both branch and compact-index variants) — breaks Numba
  vectorization / cache prefetch.
- K-tree multiclass parallelism via ThreadPoolExecutor — `parallel=True` over features
  already saturates cores; outer threading just adds queue contention.

**Lesson encoded across these:** quantization/parallelism wins need either a working
set that overflows cache or wider SIMD — Numba-on-CPU at our scale has neither. Future
speed wins must be algorithmic (less compute), not dtype/parallel tricks.

---

## 6. Current state & the one remaining lever

- **Regression is solved** — RMSE 98.0%/97.9%, 2nd-best, beats sklearn & LightGBM, only
  behind CatBoost. (A diagnostic subtlety: two datasets — `visualizing_soil`, `SGEMM` —
  looked catastrophic but are **metric artifacts** where best-RMSE→0; we added a
  `NEAR_SOLVED_NRMSE=0.02` guard that drops them from the RMSE column. R² confirms
  regression parity.) Genuine remaining reg gaps: `pol` (interaction-heavy underfit
  **at the default depth=6 — CURED by `depth=10`, where we become best-in-field +12%;
  see §9.7**), `Brazilian_houses`, `nyc-taxi`, `sulfur` — all isolated, not a size law.
- **The weak leg is classification Brier** (97.2%). It's largely the **oblivious-tree
  sharpness tax**: on high-signal/low-noise sets (electricity, covertype, `pol`) leaf-wise
  LightGBM/sklearn are sharper; CatBoost (also oblivious) trails them there too. We're
  well-calibrated (best MCB) but slightly under-sharp. The only known *real* lever is
  supporting leaf-wise/asymmetric trees — a major architectural change that sacrifices
  the oblivious speed + regularization that define the library. **This is the open
  research question**, and the most likely place a math-strong collaborator adds value:
  *can we recover leaf-wise sharpness on high-signal data without leaving the oblivious
  design?* (e.g., smarter leaf value estimation, gradient-space sharpening, signal-aware
  capacity — not bin resolution, not size-based depth, both already rejected.)

---

## 7. TabArena-Lite — the sealed holdout (in progress)

Goal: a published **Elo** on the official TabArena benchmark as a 2nd independent
robustness check. Pipeline is fully built (AutoGluon wrapper, runner, eval scripts in
`A:\code\tabarena\...`; install recipe + env constraints in the `project-tabarena-elo`
memory). Smoke-tested; full 51-task run + Elo eval pending.

**THE VOW — absolute:** TabArena-Lite is a *sealed* holdout. We never tune on it, and we
**never even look at the per-dataset breakdown.** The only number that leaves it is the
single aggregate Elo we publish. Merely *knowing* "we're weak on dataset X" would
subconsciously steer development → no longer out-of-sample. This is the airtight version
of "benchmarks are for reporting, not tuning."

---

## 8. PROPOSED: the synthetic-first 1-2-3 development loop

This formalizes "benchmarks are for reporting, not tuning" into a clean
**train / test / validate split applied to algorithm development** — the same discipline
ML uses for models, applied to *us tuning the model*. It tightens the existing pipeline
(§3) by making **synthetic the only place we're allowed to fit defaults**.

```
   ┌────────────────────────────────────────────────────────────────┐
   │ 1. TRAIN on SYNTHETIC                                           │
   │    Tune algorithm + hyperparameter defaults here ONLY.          │
   │    Full freedom to investigate, sweep, overfit, inspect.        │
   │    Cheap, fast, infinite data, known ground truth → we can      │
   │    isolate MECHANISM (size-curves, signal/noise ratio, cat      │
   │    cardinality, interaction order) instead of dataset luck.     │
   └───────────────────────────┬────────────────────────────────────┘
                               │  freeze the change
                               ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 2. TEST on GRINSZTAJN                                           │
   │    Is the change REAL on real data? Sign test over 59 datasets. │
   │    Investigation ALLOWED: which datasets better/worse, why.     │
   │    This is where we learn (e.g. "wins are high-signal only").   │
   │    A change that wins on synthetic but washes here is rejected. │
   └───────────────────────────┬────────────────────────────────────┘
                               │  if it passes, ship to main
                               ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 3. GATE on OpenML (34 datasets, multiclass)                    │
   │    Automated one-shot pre-release gate. May DEBUG a failure,    │
   │    but NEVER tune toward a pass (that makes it a 2nd train set).│
   │    Catches hidden flaws before they can spoil the sealed run.   │
   └───────────────────────────┬────────────────────────────────────┘
                               │  if clean
                               ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 4. VALIDATE on TABARENA-LITE                                    │
   │    NO investigation. Aggregate Elo only. Sealed holdout.        │
   │    Confirms we didn't Grinsztajn-overfit. Report-only.          │
   └────────────────────────────────────────────────────────────────┘
```

DECIDED (2026-06-02, after external math review): **keep the 4-stage pipeline above.**
OpenML stays as the Stage-3 automated pre-release gate — retiring it would leave too
large a gap between Grinsztajn (where we look) and TabArena (where we're blind); it is
the safety valve that catches a hidden flaw *before* a run could spoil the sealed Elo.
Stage 3 rule: debugging a regression is allowed; iterating until it passes is not.

**Why this ordering is the right discipline**

- **Synthetic as the training set** is the key move. On synthetic we *know* the data
  generating process, so we can tune to the **mechanism** (does capacity scale with
  data? does this help high-signal vs noisy? does it help high-cardinality cats?) rather
  than to which real datasets happened to be in a suite. That directly attacks our
  recurring failure mode — every rejected change above (max_bins, adaptive depth, SAS)
  *won on some real datasets and lost on others*, and we only understood why **after**
  decomposing the mechanism. Doing the mechanism work first, on synthetic, surfaces the
  "it's signal, not size" insight *before* we waste a Grinsztajn run.
- **Grinsztajn as the test set** keeps its current role but reframes it: it's not where
  we *fit*, it's where we *check generalization* of a synthetic-derived hypothesis. We
  still investigate here (which datasets, why) — that's allowed because Grinsztajn is not
  the published number.
- **TabArena-Lite as the validation set** is already locked down by the Vow. It catches
  Grinsztajn-overfitting the same way a held-out set catches train-set-overfitting.

**What this loop needs that we don't fully have yet** (concrete next infra):

1. A first-class **synthetic generator module** with knobs that map to our known levers:
   `n` (size-curve), `signal_to_noise`, `n_informative` / interaction order, categorical
   cardinality, class imbalance, label noise. Today's 7 synthetic datasets are fixed
   instances; we want *parametric families* so we can sweep a curve, not a point. (The
   size-curve + mechanism probes we already do ad-hoc per experiment — this makes them
   a permanent, reusable Stage 1.)
2. A **Stage-1 harness** that runs a candidate change across a synthetic *grid* and
   reports the mechanism response (e.g. "Brier delta vs signal-to-noise", "RMSE delta vs
   n"), not just an aggregate. The output of Stage 1 should be a *falsifiable prediction*
   about Grinsztajn ("this helps high-signal binary above n≈10k") that Stage 2 then
   tests. If Stage 2 contradicts the Stage-1 prediction, that's a real finding, not just
   a rejection.
3. Keep OpenML as a secondary independent gate if desired, or fold its role into
   TabArena — but **only one** of them can be the sealed published number; the other
   stays a private gate.

**Open questions — RESOLVED (2026-06-02)**
- OpenML: KEPT as the Stage-3 private gate; TabArena-Lite is the public sealed number.
  (See the DECIDED note above.)
- Synthetic families to probe the Brier/sharpness weakness: chosen — see §9.

---

## 9. Stage-1 synthetic spec & backlog (2026-06-02)

### 9.1 The three discriminator families

Built to interrogate the **oblivious sharpness tax** (§6) directly. A and B are a
matched pair (oblivious *should* lose A and tie/win B); C maps the crossover where our
fragmentation flips from a tax to a regularization dividend.

- **Family A — Sparse Local Interaction (asymmetric-friendly).** D=10 features
  `x~U(−1,1)`; `y = 1 iff x1>0.5 ∧ x2>0.5 ∧ x3>0.5`. A leaf-wise tree isolates this in
  ~4 leaves; an oblivious tree splits x1,x2,x3 *globally* → the false-branch region
  fragments into many sparse, over-regularized leaves. Sweep n; metric = Brier vs a
  leaf-wise baseline (LightGBM).
- **Family B — Global Additive (symmetric-friendly).** D=10 `x~N(0,1)`;
  `y~Bernoulli(σ(Σ_{i≤5} βᵢxᵢ))`. No local pockets — a global hyperplane. Oblivious
  trees should match/beat leaf-wise here. This is the control that proves a Family-A
  loss is *structural*, not a bug.
- **Family C — Noise Sweep (the regularization pivot).** Family A with label flips
  `p∈[0.0, 0.45]`. At p=0 leaf-wise wins on sharpness; as p→0.4 oblivious
  fragmentation+shrinkage becomes defensive and should overtake. Locate the crossover.

These are *parametric families* (sweep a curve), not the fixed instances in today's
synthetic set. Stage-1 output is a **falsifiable prediction** about Grinsztajn (e.g.
"helps high-signal binary, n≳10k") that Stage 2 then tests.

### 9.2 Sharpness-recovery backlog (within the oblivious constraint)

From external math review (2026-06-02), vetted against [tree.py:160](../chimeraboost/tree.py#L160)
`values[l] = -lr·G/(H+l2)`:

- **[BACKLOG] Steeper-than-Hessian per-leaf shrinkage.** `λ_j = λ_base·f(n_j)`.
  CAVEAT: flat λ is *already* sample-adaptive — for logloss `H≈0.25·n_leaf`, so a leaf
  of n=4 is shrunk ~50% while n=1000 is shrunk <0.5%. So this is **not** adding
  adaptivity; it's a *steeper* schedule than `1/(H+λ)` already gives. Test framed as
  "is steeper-than-Hessian shrinkage worth it," driving small-leaf λ up / large-leaf λ
  toward 0. Test jointly with the already-shipped size-adaptive `min_child_weight`
  (which gates sparse leaves at the *split* stage, not the *value* stage) for
  interaction. Cheap; pure leaf-value change.
- **[CLOSED] Single-tree joint Ridge over leaf assignments.** Proposed as a re-merging
  mechanism. CLOSED by math: the one-hot leaf matrix `Z` is *disjoint*, so `ZᵀZ=diag(n_j)`
  and the "joint" ridge decouples *exactly* into `w_j=G_j/(n_j+λ)` — i.e. it is already
  [tree.py:160](../chimeraboost/tree.py#L160). Orthogonal columns share no information →
  **no re-merging**. Do not build the single-tree version.
- **[BACKLOG] The salvage of the above — make `Z` non-orthogonal so coupling appears:**
  (a) **forest-level joint refit** — stack leaf-assignment columns from *all T trees*
  into one `Z` (N × Σ2^dₜ); now `ZᵀZ` is dense, jointly refitting all leaf values *does*
  couple/re-merge redundant structure across the ensemble (a global "linear-forest"
  readout). (b) **fused/structured penalty** coupling sibling leaves within a tree
  (`(w_j − w_{j'})²`), non-diagonal by construction. Either could recover sharpness by
  re-merging unnecessary oblivious splits — the actual idea worth pursuing.

### 9.3 Stage-1 harness BUILT + first-run findings (2026-06-02)

`benchmarks/synthetic.py` implements families A/B/C as parametric generators plus a
sweep harness that **reuses `run_benchmarks.RUNNERS`** (so every model is byte-for-byte
the Grinsztajn config — 2000 iters, patience 50, same val split, same Brier). Emits
`images/stage1_family_{a,b,c}_*.png` (absolute Brier curves + a ChimeraBoost−leafwise
delta panel). Run: `python benchmarks/synthetic.py --family all`.

First quick sweeps (2 seeds) — three findings, two of them course-corrections:

1. **Noiseless Family A is DEGENERATE — all models ~0 Brier, no tax visible.** Reason
   (important): a noiseless axis-aligned k-way conjunction is *exactly representable* by
   an oblivious depth-k tree (the positive octant is a single leaf). So there is **no
   asymptotic sharpness tax on clean, representable rules** — the tax is a *finite-sample
   / noise / capacity* phenomenon (over-regularization of fragmented leaves), consistent
   with §6. ⇒ **Family A must be redesigned to bite at large n.** The mechanism that
   actually binds our default is **depth=6**: a single oblivious tree can split on ≤6
   features, so a target with **more than 6 informative features / multiple disjoint
   pockets** (e.g. OR of several 2-way ANDs on disjoint pairs) forces the global-split
   constraint to bind where leaf-wise can carve pockets independently. That's the v2
   Family A.
2. **The noise sweep (C) swamps the tax in Bayes error.** At flip-rate 0.15/0.30 all
   models converge to ~0.26/0.44 Brier — dominated by the *irreducible* noise floor, not
   sharpness. ⇒ **Metric upgrade only possible with synthetic data:** we know the true
   `P(y=1|x)`, so report **excess Brier over the Bayes-optimal Brier** (`Brier_model −
   Brier_Bayes`), which subtracts the noise floor and isolates estimation/sharpness
   error. Generators should return the true prob so the harness can compute it. (At
   noise=0 the small real gap is visible: Chimera 0.006 vs LightGBM 0.000 vs CatBoost
   0.003.)
3. **Family B (control) behaves as predicted:** ChimeraBoost is weak at very small n
   (n=400: 0.349 vs LGB 0.315 / CatBoost 0.305) but converges and even leads by n=2500
   (0.318 vs LGB 0.330). Global-additive structure is *not* where we lose — supports the
   §6 claim that our deficit is structural to *local-interaction* geometry, not generic.

**Side find (unrelated bug):** `run_benchmarks._run_sklearn` passes `n_jobs=` to
`HistGradientBoosting*`, which has no such param (sklearn 1.8.0) → **sklearn_HGB errors
on every run right now**. The Stage-1 harness tolerates it (skips + warns); the real
benchmark would lose its sklearn column. Fix = drop `n_jobs` (HGB parallelises via
OpenMP, controlled by `OMP_NUM_THREADS`/threadpoolctl, not a constructor arg).

**v2 Stage-1 backlog (next loop iteration):** (a) multi-pocket / >6-informative-feature
Family A; (b) excess-Brier-over-Bayes metric using the known true prob; (c) then run the
full sweeps and let the mechanism picture drive the first real hypothesis (likely the
steeper-per-leaf-shrinkage test from §9.2).

### 9.4 Stage-1 v2 results — the tax is TINY and low-noise-only; we WIN under noise (2026-06-02)

Built `family_a_multi_pocket` (4 disjoint 2-way pockets on 8 features > depth-6 budget),
`sklearn_HGB` bug fixed, all generators return `true_prob`, harness reports **excess
Brier over Bayes** (`metrics['brier'] − 2·mean((p*−y)²)`, same convention). 3 seeds.
Two findings, both reframing §6:

**(1) The "sharpness tax" is real but TINY, and only at large-n / low-noise — and it is
NOT the generic oblivious geometry tax.** Family A v2 (noiseless, Bayes=0), excess Brier:

| n | Chimera | LightGBM | CatBoost | sklearn_HGB |
|---|---|---|---|---|
| 500 | 0.048 | 0.042 | 0.049 | 0.083 |
| 2000 | **0.007** | 0.010 | 0.009 | 0.010 |
| 8000 | 0.0086 | 0.0050 | **0.0041** | 0.0062 |

At n=8000 Chimera's excess doesn't shrink toward 0 like the others (~0.004 residual).
**Crucially CatBoost — also oblivious depth-6 — converges to BEST**, so this residual is
**Chimera-specific estimation, NOT the oblivious-geometry tax** (an oblivious learner
*can* nail this structure across boosting rounds). ⇒ the lever is **leaf-value
sharpening** (§9.2 Approach 1 steeper shrinkage / forest-level leaf refit), *not* an
architecture change. Magnitude is small (~0.004 Brier); 3 seeds → suggestive, confirm
with more seeds / larger n before acting.

**CONFIRMED (15 seeds, n→32k, `results/stage1_familyA_confirm.txt`):** the gap is real
and *persistent* — Chimera's excess Brier **plateaus at ~0.0052 from n=16k on** while the
others keep converging; at n=32k Chimera is **worst (0.0052)** and CatBoost **best
(0.0030)**, LightGBM 0.0036, HGB 0.0039. CatBoost (also oblivious depth-6) reaching best
re-confirms a **Chimera-specific leaf-estimation gap**, not geometry — its multi-step
Newton / ordered-boosting leaf machinery extracts more from identical symmetric splits.
The lever is leaf-value estimation quality (§9.2): steeper per-leaf shrinkage, forest-level
leaf refit, OR revisiting `leaf_estimation_iterations`/ordered-boosting *specifically for
the clean-but-complex regime* — to be designed as a Stage-1 hypothesis, not shipped off
this.

**(2) Under label noise, ChimeraBoost is the MOST robust — the regularization dividend
is bigger than the tax.** Family C (multi-pocket + symmetric flip), excess Brier over the
(large) Bayes floor:

| flip p | Bayes floor | Chimera | LightGBM | CatBoost | sklearn_HGB |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.008 | 0.008 | 0.006 | 0.005 |
| 0.1 | 0.177 | **0.015** | 0.023 | 0.024 | 0.022 |
| 0.2 | 0.323 | **0.029** | 0.039 | 0.034 | 0.051 |
| 0.3 | 0.428 | **0.022** | 0.024 | 0.025 | 0.060 |
| 0.4 | 0.481 | 0.016 | 0.015 | 0.015 | 0.037 |

ChimeraBoost has the **lowest excess Brier at p∈{0.1,0.2,0.3}** — our oblivious +
size-adaptive-mcw + l2=1 regularization resists fitting label noise where leaf-wise
(esp. sklearn_HGB, 0.06 at p=0.3) overfits it. This is the "fragmentation becomes
defensive" dividend (§6) made quantitative: **the tax lives only in the near-noiseless
corner; everywhere noisy, regularization makes us the winner.**

**Falsifiable prediction for Stage 2 (Grinsztajn) — to TEST, not tune:** ChimeraBoost's
Brier deficit vs leaf-wise should be **concentrated on the low-noise / high-signal binary
datasets** (electricity, covertype, pol — exactly §6's worst-Brier cluster) and should
**vanish or reverse on noisier datasets**. If true, the right lever is leaf-value
sharpening targeted at high-signal leaves, not architecture and not a global change that
would forfeit the noise-robustness dividend. (Do NOT change source off these synthetic
numbers — §8 Stage 1 fits, Stage 2 tests.) NOTE: this does NOT overwrite §6's real-data
observation that CatBoost trails leaf-wise on those sets — synthetic generates the
hypothesis; Grinsztajn adjudicates.

### 9.5 Stage-2 test of the §9.4 prediction — NOT confirmed by the crude proxy; mechanism refined (2026-06-02)

`benchmarks/analyze_grinsztajn_snr.py` (report-only analysis of the l2=1.0 Grinsztajn
JSON `20260602-001642.json`, 23 binary rows; no fitting, no source change). For each
dataset: ΔBrier = Brier_Chimera − min(LightGBM, sklearn_HGB); signal proxy = 1 −
floor/2. Result:

- **Pearson(signal, Δ) = +0.14, Spearman = −0.01.** Bin means: high-signal half mean Δ
  **+0.0046**, low-signal half **+0.0024** — directionally right (bigger deficit when
  cleaner) but **weak**; the monotone rank relation is absent. ⇒ the simple 1-D proxy
  **does NOT confirm** the Stage-1 prediction.
- **Two counterexamples break the monotone story and are the real lesson:**
  - **pol — the *cleanest* dataset (floor 0.0197) — pays NO tax** (we tie, −0.001). If the
    tax were "low-noise" alone, pol should be worst. It isn't, because pol's clean
    boundary is *oblivious-representable* — this is the **Family A *v1*** case (clean +
    representable → all models ~0, no tax). "Clean" alone ≠ tax.
  - **eye_movements — a *noisy*-looking dataset (floor 0.44) — is one of our BIGGEST
    deficits** (+0.013/+0.010). If we "win under noise," we should win here. We lose.
    "Noisy-looking" alone ≠ dividend.
- **Root cause of the non-confirmation:** the leaf-wise Brier floor is a **bad noise
  proxy** — it conflates *irreducible label noise* (compas: high floor, we WIN, −0.005)
  with *boundary complexity* (eye_movements: high floor because hard/local, we lose). The
  Stage-1 prediction was about the **noise** axis; a 1-D floor proxy can't isolate it.

**Refined mechanism (the actual hypothesis now):** the tax is the *interaction*
**low-noise × high-LOCAL-complexity** (clean boundaries that need leaf-wise locality and
aren't representable by a few global oblivious splits), NOT low-noise alone:
pol = clean+simple → no tax (v1); electricity/covertype/road-safety = clean+locally-complex
→ tax (v2, which did show the small tax); compas = genuinely noisy → we win. The deficit
cluster IS §6's named cluster — the crude proxy just can't *predict* membership because it
collapses the noise and complexity axes into one number.

**Disciplined next test (run AT MOST once, then stop regardless — avoid torturing the
holdout-adjacent data until it confirms):** replace the floor proxy with a **model-free
noise estimate** — kNN label-disagreement rate (fraction of a point's k nearest neighbours
with the opposite label), which estimates Bayes/irreducible noise from data geometry,
decoupled from GBDT capacity. Then re-plot ΔBrier on TWO axes (noise = kNN-disagreement,
complexity = floor−noise). Prediction: deficits sit in the **low-noise, high-complexity**
quadrant; wins in the **high-noise** region. If that holds, the lever is confirmed as
leaf-value sharpening for clean-but-complex leaves. Still report-only.

### 9.6 Stage-1 leaf-estimation experiment — ALL levers inert; the gap is a CatBoost-only artifact, NOT a Chimera weakness. STOP. (2026-06-02)

`benchmarks/stage1_leaf_estimation.py` swept our leaf-value/gradient knobs on Family A v2
(n=4k/8k, 8–10 seeds) against CatBoost (target) and LightGBM (leaf-wise floor):

| lever | result |
|---|---|
| **1A** `l2_leaf_reg` 1.0→0.1→0.05→0.0 | **inert** (0.0056→0.0057 at n=8k; l2=0 slightly *worse*) |
| **1B** `leaf_estimation_iterations` 1→3→5→10 | **inert** (all ~0.0055) |
| **2** `ordered_boosting=True` (our LOO) | **inert** (0.0058) |

**No knob moves us.** Every ChimeraBoost variant clusters at **~0.0055–0.0058** while
CatBoost sits alone at **~0.0033**. The decisive observation: **LightGBM (0.0056) and
sklearn_HGB tie ChimeraBoost** — so this is **NOT a Chimera-vs-field deficit; it is a
CatBoost-only advantage** over the entire rest of the field on this synthetic structure.

**This CORRECTS §9.4 finding #1.** Its "Chimera-specific leaf-estimation gap" was **seed
noise** in Step A's LightGBM number (which happened to dip to 0.0036 at n=32k on seeds
1000–1014); with more seeds LightGBM ties us within the ±0.0027 band, and only CatBoost is
robustly separated. So the residual is CatBoost machinery we don't replicate (true
permutation-snapshot ordered boosting — a known O(log n)-snapshot dead end for us, see
§5), and chasing it would mean reverse-engineering one competitor to win on a **contrived
noiseless toy** for ~0.002 Brier — textbook benchmark-overfit, and it wouldn't even fix a
Chimera-vs-leaf-wise deficit because we already tie LightGBM here.

**DECISION: STOP. Ship nothing.** No leaf-value change (l2/lei/backtracking/adaptive-L2),
no ordered-boosting flip. The synthetic loop did *exactly its job* — it **prevented us
from writing step-halving/adaptive-L2 code to chase a phantom.** The negative result is
the win.

**What this means for the real deficit:** Family A v2 reproduces a *CatBoost*-vs-field
gap, NOT the *Chimera*-vs-leaf-wise gap seen on real electricity/covertype (Step B, §9.5).
⇒ **Family A v2 is not a faithful synthetic proxy for the real-data Brier deficit.** If
that deficit is ever pursued, the prerequisite Stage-1 task is to design a synthetic family
that actually reproduces a *Chimera-worse-than-LightGBM* signal — which none of A/B/C do.
Until such a proxy exists, the real Brier gap stays where §6/the prior decisions left it:
the oblivious-vs-leaf-wise sharpness tax on high-signal data, **not chased** (would
overfit), and we remain on the blended Pareto frontier (§2) at 98.0.

### 9.7 Capacity GATE for "discrete logical-AND splits" (Alt A) — split verdict, A NOT justified (2026-06-02)

`benchmarks/stage1_capacity_gate.py`: before building AND splits (≈ depth-2d interaction
at depth-d leaf count), test the cheap proxy — does raw extra DEPTH close the gap?

**[A] pol regression (n=15k) — depth DECISIVELY helps; we already WIN with the existing
knob.** RMSE: d6 **4.55** (ties LightGBM), d8 **4.13**, **d10 4.05 (BEST IN FIELD)**, d12
4.13 (overfit). Refs: LightGBM 4.55, sklearn_HGB 4.58, **CatBoost 4.77 (worst)**. So
ChimeraBoost at depth-10 beats the best competitor by **+12%** (4.05 vs 4.55). `mcw=0` was
**redundant** (d10==d10-mcw0) — the empty-child fix (§4) already made depth a clean lever,
so the old "depth-10+mcw=0 → 111%" is now depth alone. **⇒ pol's "underfit" (§6, 79% of
best at d6) is a DEFAULT-DEPTH artifact, fully curable with the existing `depth` knob — NOT
a structural gap, and NOT something AND splits are needed for.**

**[B] Family A v2 — extra depth does NOTHING** (d6→d12 all ~0.0072; CatBoost 0.0035). Raw
oblivious depth can't help because isolating 4 disjoint pockets needs splitting all 8
pocket-features *globally* → 2^8 fragmented leaves. This IS the one case AND splits would
*uniquely* address (one AND level = one pocket in a single split) — but Family A v2 is NOT
a real-data proxy (§9.6), so a win here wouldn't promise real-world value.

**VERDICT — do NOT build Alternative A now.** The gate dissolves both motivations:
- A's one *real* target (pol) is already solved by the existing `depth` knob (we become
  best-in-field at d10); A would at most be a more-surgical route to capacity we already
  have, and a size-adaptive-depth *default* for regression was already REJECTED (§ADAPTIVE
  DEPTH: only pol benefits, elevators/Miami regress — benchmark-overfit).
- A's unique-value case (disjoint pockets, where depth can't help) exists only on synthetic
  data we've shown is not a faithful proxy.
Building A is real engineering (pair split-search, 2D histograms, predictor change) with
`cat_combinations`-style overfit risk (§5), and the gate produced **no evidence A closes a
gap existing knobs can't**. Stop — consistent with the session's pattern: cheap tests keep
shrinking the case for added complexity.

**Documentable side win (real):** ChimeraBoost is **best-in-field on pol at depth=10**
(+12% vs LightGBM/CatBoost). §6's "pol = genuine remaining underfit" is depth-default
dependent; the cure is the existing knob, not new machinery. Worth a README/§6 note that
interaction-heavy regression wants `depth≈8–10` (default stays 6 for small-data safety).
