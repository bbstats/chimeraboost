# Step-3 grow kernels — fit-side kernel pass (bit-identical speed program)

Self-sufficient handoff (BAGGING_PLAN/M1_PLAN convention). Approved by Nathan
2026-07-18 as the next program (the recorded runner-up when M1 was chosen;
M1 shipped in 0.17.0). Parent: PARETO_PLAN.md "Step 3 — kernel profiling
(opportunistic, bit-identical)". Written before any measurement or source
change beyond the code survey below.

## Why (state of the world, 2026-07-18)

- Headline point **99.4 blended @ 6.0x** (post selection_rounds); Ens8 sweeps
  at 30.1x; hc single 2.4-2.9x. The strength columns are #1 everywhere —
  the open axis is slowdown.
- PARETO_PLAN step-0 attribution (2026-07-16): **tree growth = 73-92% of
  fit** on Grinsztajn, 49-67% on hc (prep-heavy), and it is per-variant
  (every audition/refit pays it). It is the only lever bigger than
  selection, and selection is done (step 2 shipped; B-prep killed the prep
  redundancy; B1/B2 killed everything strength-adjacent).
- The predict side got its kernel pass 2026-07-13 (row-major walks →
  ~LightGBM parity). Fit kernels never got the equivalent audit.
- Design law (bagging program, 5-for-5): only output-identical engineering
  or more-data-per-member ever shipped. This program is scoped to
  **bit-identical only** by default; FP-drift kernels are Phase 2, opt-in,
  full /experiment (see below).

## Code survey (2026-07-18 — read before assuming; all already TRUE)

The grow path is NOT naive; the obvious wins are taken. `chimeraboost/tree.py`:

- `_build_and_split` (the fit path): FUSED histogram scatter + split scan,
  one parallel launch per level, feature-parallel (disjoint writes, no
  races), active-leaf skipping at small n, per-feature bin-count trimming
  (only [0, nb) zeroed/scanned), transposed leaf-outer scan with per-leaf
  parent term hoisted. Reference kernels `_build_histograms_into` +
  `_best_split` are KEPT as exact-equality oracles
  (tests/test_tree_kernels.py) — keep that discipline for anything new.
- `_descend_leaves` / `_descend_leaves_serial`: in-place level push,
  `_SMALL_N` dispatch (fork/join vs serial measured crossover).
- Binned matrix is already **uint16** (`BIN_DTYPE`), feature-major,
  contiguous rows; hist buffer preallocated once per fit, interleaved
  grad/hess (one cache line per scatter write).
- `build_oblivious_tree` per-tree Python: list appends, np.array
  conversions, `bincount/flatnonzero` at small n, `_leaf_values` launch,
  optional `_linear_leaf_fit` (ridge over split features; binary default
  linear_leaves=True), ObliviousTree construction.

What was NEVER profiled: the split of tree-build time BETWEEN scatter /
scan / descend / `_leaf_values` / `_linear_leaf_fit` / per-tree Python
overhead. Step-0's suggestive anomaly: **binary trees cost 3.6 ms/tree vs
regression 1.2-1.7** on the same kernel — prime suspect is the linear-leaf
ridge (binary-default-on), NOT the scatter. If that holds, "grow kernels"
is substantially a linear-leaf-kernel program, which nobody has looked at.

## Phase 0 — in-tree attribution (no library change; pre-registers everything)

New `benchmarks/profile_grow.py` (clean box, warm JIT, script file):

1. Per panel set (reuse the step-0 panel: cpu_act, diamonds, nyc-taxi,
   MagicTelescope, Higgs, road-safety, kick, wine-reviews, okcupid-stem —
   spans task types, n, width, cats): wall split of one default fit into
   scatter+scan (`_build_and_split`), descend, `_leaf_values`,
   `_linear_leaf_fit`, per-tree Python residue, non-tree residue. Method:
   targeted timers around the call sites (a fork with time.perf_counter
   around each kernel call, summed per fit) — NOT cProfile (numba-opaque).
2. The binary anomaly: same set fit with linear_leaves True vs False,
   ms/tree delta = the ridge's true cost share.
3. Thread-geometry read: feature-parallel saturation — same fit at
   threads 1/2/6/12 on a narrow set (MagicTelescope, 10 features) vs a wide
   one (road-safety 32+); flat scaling on narrow data = the known
   feature-parallel ceiling (fix is Phase-2 class, record only).
4. Multiclass: per-round cost split incl. the K per-class
   `ascontiguousarray` grad/hess copies and K kernel launches (okcupid).
5. Stream-width microbench: scatter with uint8 vs uint16 Xb and int32 vs
   int64 leaf on synthetic shapes — bounds the dtype levers before any edit.

Deliverable: table in this file + a pre-registered lever order with measured
ceilings. **Rule: no Phase-1 edit before its ceiling is measured here.**

### Phase-0 results (2026-07-18; profile_grow.py, results/grow-phase0*.md)

Attribution, 2 seeds, % of estimator fit (all booster fits incl. auditions;
split/descend/leafv/linfit are inside grow; pytree = grow minus its kernels):

| dataset | task | n_train | fit_s | grow% | split% | descend% | leafv% | linfit% | pytree% | nontree% | ms/tree |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cpu_act | reg | 6144 | 0.4 | 81.3 | 45.7 | 2.5 | 1.5 | 16.2 | 15.4 | 18.7 | 0.62 |
| diamonds | reg | 37500 | 1.3 | 85.8 | 41.1 | 3.2 | 2.1 | 24.4 | 15.1 | 14.2 | 2.20 |
| nyc-taxi | reg | 37500 | 3.1 | 91.9 | 57.6 | 2.5 | 1.6 | 18.8 | 11.5 | 8.1 | 3.26 |
| MagicTelescope | bin | 10032 | 0.5 | 64.3 | 36.5 | 1.9 | 1.1 | 14.0 | 10.7 | 35.7 | 0.99 |
| Higgs | bin | 37500 | 1.7 | 69.8 | 46.5 | 1.8 | 1.0 | 12.6 | 8.0 | 30.2 | 3.35 |
| road-safety | bin | 37500 | 2.3 | 74.9 | 51.2 | 1.7 | 1.0 | 13.2 | 7.7 | 25.1 | 3.63 |
| hc:kick | bin | 54737 | 2.1 | 50.2 | 41.0 | 1.1 | 0.8 | 6.3 | 1.1 | 49.8 | 4.29 |
| hc:wine-reviews | reg | 75000 | 1.3 | 70.5 | 45.1 | 4.1 | 4.5 | 12.7 | 4.2 | 29.5 | 1.42 |
| hc:okcupid-stem | multi | 38091 | 4.1 | 45.2 | 37.7 | 1.2 | 0.7 | 0.0 | 5.6 | 54.8 | 3.26 |

Timer caveat: pytree includes ~5-15us/tree of wrapper overhead (reads ~1-2
points high on the smallest sets; bucket is real regardless).

- **Item 2, binary anomaly RESOLVED = the ridge.** linear_leaves False cuts
  binary ms/tree 22-28% (Magic 0.94->0.73, Higgs 3.35->2.40, road-safety
  4.08->3.14, kick 4.01->3.15); in-fit ridge = 13-22% of ll=T grow. At equal
  n, ll=F binary ms/tree ~= regression (Higgs 2.40 vs diamonds 2.20) — the
  anomaly was never the scatter.
- **Item 3, threads:** split saturates x4.8 (narrow, 10 feats) / x5.2 (wide,
  32) at 12 threads — near-IDENTICAL curves, so the limiter is memory/launch
  cost, not feature-parallel geometry. Phase-2 class, record only.
- **Item 4, multiclass:** gradcopy = 0.9% of okcupid fit (1143 copies,
  0.04 s); 3429 split + 3429 descend launches. L-mc has no ceiling.
- **Item 5, micro:** u8 Xb <= x1.13 scatter at 8K, ~x1.00 at >=50K (scatter
  is random-write bound, not stream bound). i32 leaf: descend x1.8-4.4 at
  n>=50K but descend <= 4.1% of fit, and scatter shows regressions (x0.66-
  0.94 in several shapes).

**Lever order (measured ceilings, % of estimator fit):**

1. **L-ridge** — ceiling 6.3-24.4% (gr reg 16-24, gr bin 13-14, hc 6-13,
   multi 0). PROCEED first.
2. **L-pytree** — pytree 7.7-15.4% on gr (1-6 hc) + descend 1.1-4.1% +
   per-level launch overhead. PROCEED second; includes fusing descend (and
   next-level occupancy) into the fused kernel — integer ops, exact by
   construction.
3. **L-mc** — 0.9%: KILLED at ceiling.
4. **L-leaf32** — ~1-2% (descend-only wins; micro scatter regressions):
   KILLED at ceiling.
5. **L-bin8** — <=2-3% at small n only, ~0 at >=50K: KILLED at ceiling.

Phase-2 input recorded: split (fused scatter+scan) = 36.5-57.6% of fit,
>=50% on 2/9 panel sets (nyc-taxi 57.6, road-safety 51.2).

## Phase 1 — bit-identical levers (goldens + oracle tests + timing; no gate)

Candidates, to be ORDERED by Phase-0 ceilings (survey priors in brackets;
every one preserves per-accumulator FP addition order — that is the
bit-identity test):

- **L-ridge: `_linear_leaf_fit` pass** [prime suspect for binary; unknown
  until profiled]. Whatever the profile shows — access pattern, per-leaf
  solve batching, redundant recompute — subject to exact-identity.
- **L-leaf32: `leaf` int64 → int32** [halves the descend+scatter leaf
  stream; indices are exact; touches kernel signatures → warmup() and
  oracle kernels updated together].
- **L-bin8: BIN_DTYPE uint16 → uint8 when max(n_bins) ≤ 255** [default
  max_bins=128 qualifies; halves the Xf stream; dtype-dispatch or global —
  decide at implementation; indices exact].
- **L-pytree: per-tree Python residue** [matters on short-tree/small-n
  fits; hoist per-tree allocs into fit-level buffers like hist already is].
- **L-mc: multiclass copy/launch trims** [K× per round; grad/hess column
  extraction without fresh allocs — order-preserving copies are exact].

### Phase-1 verdicts

- **L-ridge: KILLED 2026-07-18** (implemented, measured, reverted — commits
  332fd39 / 686219e). The restructure (row-major precomputed design table,
  mirrored intercept column, hoisted h*x; bit-identical, oracle test green,
  455 tests passed) was NOT faster: kernel-vs-kernel micro
  (results/grow-lridge-micro.md) x0.92-0.94 at n=8K, **x0.49-0.67 at
  n>=37.5K** — the (n, k) table's extra write+re-read traffic exceeds the
  gather it replaces. The ridge is FMA/accumulator-bound, not gather-bound:
  uint16 Xb rows pack 32 samples/line, centers_std is L1-resident, and
  within-leaf sample order is increasing (prefetch-friendly). Panel
  attribution pre/post was flat (fit-level noise ±10-15% dominates —
  kernel-vs-kernel micro is the decision-grade read for levers this size).
  Recorded follow-up (NOT pursued): per-(j,jj) register accumulation via
  loop interchange is provably bit-identical (per-accumulator sample order
  survives the interchange) but re-reads leaf data k²/2 times — only wins
  if leaves stay L2-resident; ~2-4%-of-fit ceiling, revisit only if the
  program strands below the bar.
- **L-pytree: IMPLEMENTED 2026-07-18, tier-2 pending** (commit 12d0f74).
  One fused launch per level: `_build_split_descend` = `_build_and_split`'s
  search + in-kernel descend + next-level occupancy list at small n; the
  per-level bincount/flatnonzero numpy pair and the descend launch are gone.
  Attribution panel (grow-lpytree-post vs grow-phase0): the pytree residue
  COLLAPSED — cpu_act 15.4→5.3% of fit, diamonds 15.1→2.3, nyc-taxi
  11.5→1.9, Magic 10.7→3.2, Higgs 8.0→1.2, road-safety 7.7→1.1, okcupid
  5.6→0.8 (kick/wine were already ~1-4%) — with ms/tree down 3-13% on the
  gr sets and whole fits equal-or-faster everywhere. Clean-box smoke
  (grow_smoke.py, PYTHONPATH worktree BASE, paths printed): 10/10 md5
  prediction fingerprints EXACT (5 panel sets x single+Ens8, both size
  branches + multiclass). Oracle pipeline test (incl. rejected-level and
  poisoned-buffer checks) + full 455-test suite green; warmup updated (one
  signature covers both branches). Tier-2 identity + fit_time_delta vs gr
  20260717-195429 / hc 20260717-193744 running.
- Microbench gotcha (cost one wrong table): numba dispatchers expose
  `__wrapped__` = the raw py_func, so "unwrapping" one times INTERPRETED
  Python — ~1000x slow and misleadingly flat ratios. Call dispatchers
  directly (fixed in profile_grow.py).

Explicitly OUT of Phase 1 (FP order changes = behavior-changing, NOT here):
histogram subtraction (sibling = parent − child), row-parallel scatter with
chunked reductions, any grad/hess dtype change, any threading that splits a
single accumulator's sample loop.

Ship shape per lever (B4/B-prep precedent for output-identical changes):
goldens + oracle exact-equality tests green → clean-box smoke (PYTHONPATH
worktree BASE, paths printed) → tier-2 identity runs (gr + hc, ALL arms
exact ties — 73/73 datasets — plus `fit_time_delta.py` raw sums with the
LightGBM-drift caveat noted) → OpenML gate VACUOUS (no strength surface).
`chimeraboost.warmup()` must still cover every default-path kernel
signature after changes (the TabArena cold-JIT fix depends on it — verify
via its test).

## Phase 2 — FP-drift kernels (NOT pursued by default; Nathan's opt-in only)

Histogram subtraction is the industry-standard grow win (up to ~2x on the
scatter) but parent−child ≠ direct sum in floating point → occasional
different splits → goldens break, selection races can flip on near-ties.
If Phase 1 lands <10% and Phase 0 shows scatter ≥50% of fit, present the
numbers and let Nathan decide whether a full /experiment (synth screen with
Brier read → both suites → gate) on a drift-class kernel is wanted. Default
answer: no — the design law and the identity-based test suite are worth
more than the residual speed.

## Acceptance / kill (registered)

- **Acceptance:** ≥10% raw summed single-model fit-time reduction on at
  least one decision suite (fit_time_delta.py, net of the LightGBM drift
  read) at EXACT 73/73 identity, with the headline pareto slowdown
  re-measured (expect 6.0x → 5.3-5.5x if grow improves ~15-20%; claim only
  what the canonical 5-arm refresh shows).
- **Kill:** Phase 0 finds no lever with a measured ceiling ≥10% of fit, or
  the implemented levers sum to <5% suite-level — record the profile table
  and close (this is an opportunistic program; a fast honest kill is a
  fine outcome and still pays for itself by settling the "grow is the
  remaining lever" question).
- Per-lever: any golden/oracle mismatch = revert that lever immediately
  (no "small drift" tolerance — that is Phase-2 territory by definition).

## Protocol notes for the implementing session

- Machine quirks apply (script files, one benchmark at a time, worktree
  PYTHONPATH for BASE arms with `print_chimera_path.py`, file reads over
  scrolled stdout).
- Baselines-of-record for identity ties: gr `20260717-153114` + hc
  `20260717-155202` are pre-M1; POST-M1 canonical runs are gr
  `20260717-195429` (single-arm) + hc `20260717-193744` (5-arm) — use the
  post-M1 pair (current main includes M1; multiclass sets now select
  crosses, so pre-M1 baselines would show legitimate non-ties there).
  hc Ens8/CatBoost arms: `193744` is the 5-arm baseline; a gr 5-arm
  baseline must be RUN FRESH at close if the canonical chart refresh needs
  it (the 195429 run is single-arm).
- Branch: `grow-kernels`. Keep reference kernels as oracles; new tests in
  tests/test_tree_kernels.py style (exact equality, not allclose).
- Record every verdict (win or kill) here; memory + PARETO_PLAN checklist
  at close.

## Checklist

- [x] Phase 0: profile_grow.py + attribution table here; levers ordered by
      measured ceiling — 2026-07-18; L-mc/L-leaf32/L-bin8 killed at ceiling,
      L-ridge then L-pytree proceed
- [ ] Phase 1 levers, each: implement → goldens/oracles → smoke → tier-2
      identity + timing; verdicts recorded per lever
- [ ] Phase 2 decision recorded (default: not pursued)
- [ ] Close: pareto refresh if acceptance met; PARETO_PLAN + memory updated
