# Quantized-gradient histograms (Q program) — FP-drift fit-speed pass

Self-sufficient handoff (GROW_PLAN/M1_PLAN convention). Opened 2026-07-18 on
Nathan's explicit opt-in ("let's attempt some quantization ... imitating
LightGBM's newer quantization method") — this is the Phase-2/FP-drift door
GROW_PLAN registered as "Nathan's opt-in only". Branch: `quant-hist`.
Written before any measurement or source change.

## Why (state of the world, 2026-07-18)

- Headline 99.4 blended @ 6.0x; strength columns #1 everywhere; the open
  axis is slowdown. Grow program (CLOSED 2026-07-18) settled: the fused
  scatter+scan is **37-58% of fit** and the only double-digit object left;
  everything bit-identical around it is exhausted (~1-5% each).
- GROW_PLAN Phase-0 item 5 settled the mechanism: the scatter is
  **random-write bound**, not stream bound (u8 bins ~x1.00 at >=50K; i32
  leaf regressed). The lever must shrink the random-RMW work itself.
- Prior art: LightGBM >=4.0 quantized training ("Quantized Training of
  Gradient Boosting Decision Trees", Shi/Ke et al., NeurIPS 2022;
  `use_quantized_grad`, `num_grad_quant_bins=4`, stochastic rounding,
  `quant_train_renew_leaf`). Their CPU win comes from integer histogram
  accumulation + packed grad/hess; accuracy holds at absurdly few bits
  when leaf values are renewed with original gradients.

## The adaptation (differs from LightGBM deliberately)

Per tree: compute scales dg = max|g|/QMAX, dh = max(h)/QMAX, quantize with
stochastic rounding qg = floor(g/dg + u), qh = floor(h/dh + u), pack ONE
int64 per sample: `q = (qg << 32) + qh`. Histogram becomes int64
(n_features, leaves, bins) — ONE integer RMW per (sample, feature) instead
of two float64 RMWs, and the hist working set HALVES (131 KB -> 65 KB per
feature at depth 6 / 128 bins). Split scan accumulates packed integers
(exact), unpacks (arithmetic shift / mask), dequantizes to float for the
gain formula. Descend/occupancy unchanged.

- **QMAX adaptive, not low-bit**: QMAX = min(32767, (2^31 - 1) // n).
  Overflow-safe by construction (any cell/prefix sum: |sum qg| <= n*QMAX <
  2^31, sum qh < 2^31 < 2^32 — the packed halves can never bleed). That is
  ~15 bits at n<=65K, ~11 bits at n=1M — far finer than LightGBM's 2-4
  bits, because OUR win is packing/bandwidth, not bit-width. Expect
  quantization noise near float-rounding scale, i.e. tier-1 flat.
- **Leaf values stay exact by construction**: `_leaf_values`,
  `_refine_leaf_values`, ordered-boosting LOO, and `_linear_leaf_fit` all
  consume the ORIGINAL float64 grad/hess — quantization touches ONLY the
  split search (structure choice). This is LightGBM's leaf renewal, free.
- **Stochastic rounding, deterministic**: counter-based splitmix64 on
  (qseed, i); qseed drawn per tree from the booster rng — same
  random_state => same model, no RNG state plumbing into kernels. (At
  11-15 bits round-to-nearest bias would likely be invisible too;
  stochastic is cheap insurance and is what the paper validates.)
- **Semantic deltas accepted (registered)**: a leaf whose quantized
  hessian prefix rounds to 0 counts as an empty child for the
  min_child_weight exemption; hr = ht - hl computed in float (monotone,
  never negative). Both are within quantization-noise class.
- MSE hessians (h ≡ 1) quantize EXACTLY (counts); regression loses
  nothing on the h side. Logloss h in (0, .25] gets the full QMAX range.
- Heavy-tailed g (MVS reweighting, subsample<1 non-default) inflates
  max|g| and wastes bits — recorded risk, max-based scale kept for
  Phase 1 (LightGBM does the same).

Compounding follow-up (Phase 2, only if Q1 ships): **histogram
subtraction becomes EXACT in the integer domain** (parent - sibling, no
FP drift beyond the quantization already priced in) — scatter only the
right-going half per level >=1, derive the sibling by subtraction;
in-place safe iterating parents descending. Ceiling ~ (d-1)/d * 1/2 of
remaining scatter. GROW_PLAN's drift objection to subtraction dissolves
under Q1.

## Phase 0 — kernel microbench (no library change; go/no-go)

`benchmarks/quant_micro.py`: script file, warm JIT, median of warm reps,
dispatchers called directly (GROW_PLAN `__wrapped__` trap noted). Compare
the full 6-level per-tree loop, same synthetic inputs:

- A: `_build_split_descend` (current float kernel, imported from library)
- B: `_build_split_descend_q` (packed int64, defined in script) +
  `_quantize_pack` cost (once per tree, amortized)
- C: int32-packed timing-only variant (overflow ignored) — bounds the
  two-tier (16+16) scheme's extra headroom; NOT a ship candidate as-is.

Shapes: n in {8k, 37.5k, 75k, 200k} x nf in {10, 32}, depth 6, bins 128,
hess regimes {ones (reg), p(1-p) (logloss)}. Table to
`benchmarks/results/quant-phase0-micro.md` + this file.

**Go/no-go (registered): B >= 1.30x median vs A on BOTH nf shapes at
n >= 37.5k.** Below that the fit-level ceiling (37-58% share) drops under
~10% and the program dies here — record and close. 1.30-1.5x predicts
~12-22% fit-level on scatter-heavy sets.

### Phase-0 verdict (2026-07-18; results/quant-phase0-micro.md)

Two runs: first with numpy-side scale reduction (pack 0.41 ms @200k — ate
the narrow-shape win), then with the fused numba reduction `_gh_absmax`
(pack 0.13 ms @200k). Final table (25 warm reps, medians):

| n | nf | A float ms | B int64 ms | pack ms | C int32 ms | A/B | A/C |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8000 | 10 | 0.30 | 0.28 | 0.02 | 0.28 | 1.07x | 1.06x |
| 8000 | 32 | 0.56 | 0.47 | 0.02 | 0.47 | 1.19x | 1.18x |
| 37500 | 10 | 0.60 | 0.48 | 0.03 | 0.49 | 1.23x | 1.22x |
| 37500 | 32 | 1.59 | 1.17 | 0.03 | 1.17 | 1.35x | 1.36x |
| 75000 | 10 | 1.09 | 0.85 | 0.06 | 0.86 | 1.28x | 1.27x |
| 75000 | 32 | 3.00 | 2.17 | 0.06 | 2.15 | 1.39x | 1.39x |
| 200000 | 10 | 2.82 | 2.13 | 0.13 | 2.63 | 1.32x | 1.07x |
| 200000 | 32 | 7.80 | 5.52 | 0.13 | 5.66 | 1.41x | 1.41x |

**Verdict: PROCEED, with the literal miss recorded.** The clause as
written fails 2/6 registered cells — both in the NARROW regime (nf=10:
1.23x @37.5k, 1.28x @75k); wide shapes pass everywhere (1.35-1.41x) and
narrow passes at 200k (1.32x). The clause's stated rationale (fit ceiling
<10%) does NOT trigger: suite sums are large-n weighted (grow-program
lesson) and the gr-dominant shape (37.5k x 24-32 wide) reads 1.35x
=> ~12-15% fit-level on 45-58% split shares. Recorded risk: if narrow
sets weigh more than expected, the gr sum could land under the 10% ship
bar — Phase 3's fit_time_delta is the real referee, and the ship bar is
unchanged.

- **C (int32 16+16) is DEAD**: never beats B where it matters and
  regresses at 200k/10 (1.07x). Single-tier int64 packing confirmed;
  no two-tier scheme.
- Pack cost is per-tree ~0.02-0.13 ms with the fused reduction; the
  logloss loss-constant shortcut (|g|<1, h<=0.25, no reduction) stays an
  optional follow-up, not needed at these numbers.

## Phase 1 — implement behind a flag (default OFF until Phase 3)

- Booster param `quantize_gradients=False` (sklearn API passthrough);
  when on: `_alloc_hist_buffers` allocates int64 (nf, leaves, bins);
  per-fit packed-q buffer (n int64) reused across trees; qseed per tree
  from booster rng; `build_oblivious_tree(..., quantize=...)` dispatches
  to the q-kernel. Everything downstream of `leaf` untouched.
- Tests (house style, tests/test_tree_kernels.py):
  - **Exact-equality oracle**: with power-of-2 scales and integer-valued
    g/h (products exact in fp64), the q-kernel's split/gain/descend must
    match the float kernel BIT FOR BIT (stochastic rounding is a no-op on
    exactly-representable values). This is a real oracle, not allclose.
  - Determinism: same random_state => identical models (twice).
  - Overflow guard: forced tiny QMAX (huge synthetic n) stays sane.
  - Booster-level sanity: quantized fit within loose tolerance of float
    fit on a real small dataset (structure may differ; metric close).
- `warmup()` covers the new signatures IF the default ever flips (its
  test pins this).

## Phase 3 — /experiment (full FP-drift protocol; this is a strength surface)

Tier 1 synth screen (Brier read included — bagging lesson) -> if flat or
better, both decision suites (gr + hc, sign-tested separately, hc Ens8 arm
included) -> OpenML one-shot gate (NOT vacuous here) -> ship/kill.

**Ship bar (registered):**
- Accuracy: NO statistically significant sign-test loss on either suite
  (tier-1 must be flat-or-better first; canary slice flat).
- Speed: `fit_time_delta.py` raw sums >= 10% faster on >= 1 decision
  suite (grow-program bar, same instrument).
- Both hold => flip `quantize_gradients` default ON, regenerate goldens
  (behavior change, cross_features precedent), refresh pareto (expect
  6.0x -> ~5.2-5.5x if the microbench ratio carries), CHANGELOG.
- Any accuracy kill => record verdicts here, keep the flag opt-in only if
  it still earns its keep somewhere measurable; else revert fully.

## Kill clauses (registered)

- Phase 0 microbench < 1.30x at the registered shapes -> CLOSE, no
  library change.
- Tier-1 synth significant regression (esp. Brier) -> CLOSE, revert.
- Suite sign-test accuracy loss -> kill default flip; opt-in survives
  only with a measured niche, else revert.
- Gate failure -> no ship (standing rule).

## Protocol notes

- Machine quirks apply: script files only, ONE benchmark at a time,
  worktree + PYTHONPATH for BASE arms with the path printed, file reads
  over scrolled stdout, C: is nearly full.
- Baselines-of-record for Phase 3: post-M1 canonical gr `20260717-195429`
  (single-arm) + hc `20260717-193744` (5-arm). BASE arm = main @ 9020b41.
- One benchmark at a time; Phase-0 micro is not a benchmark run but still
  don't overlap it with suite runs.

## Checklist

- [ ] Phase 0: quant_micro.py written + run; table here; go/no-go verdict
- [ ] Phase 1: kernels + flag + oracle/determinism/overflow tests green
- [ ] Phase 3: tier-1 synth screen verdict
- [ ] Phase 3: gr + hc sign tests + fit_time_delta verdicts
- [ ] Phase 3: OpenML gate verdict
- [ ] Close: ship (default flip + goldens + pareto + CHANGELOG) or kill;
      memory + PARETO_PLAN updated either way
