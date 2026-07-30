# GPU backend — feasibility thinking (no measurement, no source change)

Opened 2026-07-27 on Nathan's ask ("begin to think about how GPU might work
for chimeraboost"). **Nothing here was run.** Every number below is either
quoted from an existing recorded result in this repo or is an explicit
back-of-envelope estimate, marked as such. Written before any Phase 0, in
the GROW_PLAN/QUANT_PLAN handoff convention so a later session can pick it
up cold.

## The one number that decides this

GROW_PLAN Phase 0 (2026-07-18) already attributed a default fit. The
GPU-portable kernels are split (fused scatter+scan), descend, and leaf
values; everything else is preprocessing, the linear-leaf ridge, per-tree
Python, and validation predicts. Amdahl caps a GPU fit at 1/(1 - portable):

| dataset | split% | descend% | leafv% | portable now | cap | + ridge ported | cap |
|---|--:|--:|--:|--:|--:|--:|--:|
| cpu_act | 45.7 | 2.5 | 1.5 | 49.7 | 2.0x | 65.9 | 2.9x |
| diamonds | 41.1 | 3.2 | 2.1 | 46.4 | 1.9x | 70.8 | 3.4x |
| nyc-taxi | 57.6 | 2.5 | 1.6 | 61.7 | 2.6x | 80.5 | 5.1x |
| MagicTelescope | 36.5 | 1.9 | 1.1 | 39.5 | 1.7x | 53.5 | 2.2x |
| Higgs | 46.5 | 1.8 | 1.0 | 49.3 | 2.0x | 61.9 | 2.6x |
| road-safety | 51.2 | 1.7 | 1.0 | 53.9 | 2.2x | 67.1 | 3.0x |
| hc:kick | 41.0 | 1.1 | 0.8 | 42.9 | 1.7x | 49.2 | 2.0x |
| hc:wine-reviews | 45.1 | 4.1 | 4.5 | 53.7 | 2.2x | 66.4 | 3.0x |
| hc:okcupid-stem | 37.7 | 1.2 | 0.7 | 39.6 | 1.7x | 39.6 | 1.7x |

Median cap **2.0x** for a histogram-only port, **2.9x** if the ridge goes
too — and that is with an *infinitely fast* GPU. Real GPU kernels at
n = 6k-75k will not be infinitely fast: at those sizes a level is
0.06-2.3 M scatter ops, which is a rounding error of GPU work wrapped in
launch and sync latency.

**Conclusion up front: a straight port cannot pay at decision-suite
scale.** The whole boosting loop has to live on the device (a per-level
H2D of the binned matrix would be ~4.5 MB at n = 37500 x 60 features and
would dwarf the compute), so "port the histogram" is not an available
option — it is all-or-nothing, for a ceiling of ~2x. The only two framings
that survive this table are giant-n, where the CPU numbers are genuinely
bad, and batching the *outer* loops, where the ceiling is not Amdahl-bound
at all.

## What GPU changes about verdicts we already recorded

Several closed questions reopen on a GPU, and one closed question stays
closed. Worth stating explicitly so a later session does not re-derive
them:

- **L-bin8 (uint8 bins) — killed on CPU, likely a win on GPU.** The CPU
  kill was "scatter is random-write bound, not stream bound" (<=1.13x at
  8K, ~1.00x at >=50K). On a GPU the binned matrix read is a *coalesced*
  stream and halving it halves the dominant bandwidth term; it also halves
  any shared-memory tile. Default max_bins = 128 fits in uint8.
- **L-leaf32 (int32 leaf) — same reversal, same reason.**
- **Integer histogram subtraction — killed on CPU 2026-07-18
  (SUBTRACT_PLAN, measured 0.49-0.57x), and the recorded mechanism does
  not transfer.** The CPU kill was: the int64 hist slice is L2-resident,
  so the saved random RMW is cheaper than the added 50/50 `leaf & 1`
  branch mispredict. On a GPU the bottleneck is *atomic throughput*, and
  halving the atomic count halves it directly; and if samples are
  partitioned by leaf (see below) the predicate becomes block-uniform, so
  there is no divergence cost to pay at all. Exactness is unchanged —
  parent minus sibling is exact in the integer domain, which is why
  QUANT_PLAN registered it in the first place.
- **Histogram-subtraction's FP-drift objection stays dissolved**, but only
  on the quantized path. Nothing here revives it for `quantize_gradients=False`.
- **Float64 is a trap on consumer cards.** RTX-class GPUs run FP64 at 1/64
  rate. Our histograms are already integer on the default path (packed
  int64 — full-rate atomics), which is exactly the right form. Design rule:
  integers and float32 for the *split search* (structure choice only, and
  it already tolerates quantization noise by design), float64 reserved for
  the O(n) and O(leaves) reductions where the volume is small.

## The identity story (this is the good news)

The 2026-07-18 quantization ship accidentally made a GPU port testable
under the existing discipline:

- The quantized histogram is **int64 addition, which is exactly
  associative**. Atomic reordering across thousands of threads cannot
  change the result. A GPU histogram can therefore be **bit-identical to
  `_build_split_descend_q`, by construction** — not "close", equal.
- The stochastic rounding is already counter-based (splitmix64 of
  `qseed + i`), so it is index-parallel and order-independent. It ports
  unchanged.
- The gain scan can preserve the CPU's exact float addition order: assign
  one thread per (feature, leaf) for the prefix pass and accumulate the
  per-threshold gain over leaves in ascending order, which is what the CPU
  loop does. Low occupancy, but the scan is the small half.

What **cannot** be made CPU-equal:
- `_linear_leaf_fit` (default-ON for binary, 13-24% of fit). Per-leaf
  normal equations accumulated across threads cannot reproduce the CPU's
  sequential order. A fixed-order tree reduction buys reproducibility, not
  equality.
- The float path (`quantize_gradients=False`): float atomics are not
  order-deterministic.

**Recommended stance:** a GPU backend supports the quantized path only and
is identity-certified against the CPU kernels there; anything else either
falls back to CPU or is documented as reproducible-but-not-CPU-equal, with
its own goldens. That keeps the numerical-identity test suite meaningful
instead of quietly weakening it.

## Sketch of a device-resident design

Layout: `Xb` uint8 feature-major, `q` packed int64 (already the format),
`leaf` int32. All resident; the binned matrix is uploaded once per fit.

1. **Histogram.** First implementation: one thread per (sample, feature),
   `cuda.atomic.add` into the full (n_features, n_leaves, n_bins) int64
   buffer. At 60 features x 64 leaves x 128 bins that buffer is 3.9 MB —
   **it fits in L2 on any modern card**, and int64 atomics resolve at L2.
   No shared memory, no partitioning, trivially correct. Refinements only
   if profiling demands: shared-memory privatised tiles, and partitioning
   samples by leaf (cheap here — oblivious trees share one leaf partition
   across all features, unlike level-wise/leaf-wise competitors, which is
   the structural reason CatBoost's GPU implementation is oblivious-native).
2. **Scan + argmax** on device; the winning (feature, threshold, gain) is
   written to a device slot and never copied to the host mid-tree.
3. **Descend** reads that device slot. The "level rejected" test is
   computed on device and masks the descend, so a tree needs **zero host
   syncs**; the model is read back once per tree (or once per fit, keeping
   the whole forest on device).
4. Early stopping needs one 8-byte D2H per round — ~500 x ~10 us = ~5 ms
   per fit. Not a concern.

Awkward parts, in order of how badly they are usually underestimated:

- **Preprocessing.** On high-cardinality sets `nontree` is 49.8% (kick) and
  54.8% (okcupid) of fit — ordered target statistics, cat crosses, binning.
  Leave it on CPU and Amdahl caps the GPU at ~1.7x on exactly the datasets
  where we are slowest. Ordered TS is a segmented scan over a permutation,
  so it is portable, but it is real work and it is most of a second
  implementation of `preprocessing.py`/`target_encoding.py`.
- **The ridge**, above.
- **Warmup, again.** numba CUDA kernels pay a PTX compile on first call.
  We already have scar tissue here (issue #34, the TabArena cold-JIT read);
  `warmup()` would need a CUDA arm and the same "does it actually cache"
  investigation.
- **Testing without hardware.** `NUMBA_ENABLE_CUDASIM=1` runs CUDA kernels
  in pure Python — far too slow for anything real, but it does let the
  correctness/oracle tests for GPU kernels run in CI on tiny arrays. Worth
  knowing before someone concludes the path is untestable.

## Dependency and honesty constraints

- numba's built-in CUDA target is **deprecated**; the CUDA target now lives
  in NVIDIA's separate `numba-cuda` package (supported through numba 0.62,
  removed no earlier than 0.63; the `numba.cuda` namespace is preserved).
  So this would be `pip install chimeraboost[gpu]` -> `numba-cuda` plus
  CUDA runtime wheels. Kernel source stays Python; no C++ build step for
  us. The hard constraint survives in letter — but it is NVIDIA-only,
  needs a driver, and adds multi-GB of toolkit (and C: has ~4 GB free).
- **The Pareto chart is a CPU-seconds chart.** A GPU arm is not comparable
  to CPU competitors and must not be plotted against them. An honest GPU
  chart needs GPU competitors: LightGBM CUDA, XGBoost `device="cuda"`, and
  CatBoost GPU — and CatBoost GPU is oblivious-native and very fast. Our
  CPU story is "5x slower than LightGBM but strongest"; the GPU story would
  most likely be "slower than CatBoost GPU" with the strength claim
  unverified until re-measured. That is a real risk to the north star's
  meaning, not a presentational detail.
- **Prior art occupies the niche.** Py-Boost (sb-ai-lab) is already a
  pure-Python GPU GBDT on CuPy/Numba, and SketchBoost is its multioutput
  split-search work — the same territory A1's vector leaves drew from.
  "Pure-Python GPU GBDT" is not an empty slot.

## Three framings, with a recommendation

- **Option A — batch the outer loops (recommended, if anything).** Do not
  make one fit fast; make the *batch* dimension the ensemble member or the
  selection audition. The default fits up to 4 boosters for model
  selection; Ens8 fits 8 members. Batching them gives kernels 4-32x more
  work, which fixes GPU under-utilisation at our real data sizes — the
  problem that kills Options B and C at decision-suite scale — and it aims
  at the north star, because Ens8 is the strength leader (98-100% win rate)
  and its 23.6x slowdown is the frontier's weak leg. Members are
  independent, so the batch axis is trivially correct.
- **Option B — giant-n only.** `device="cuda"` engaging above ~200k rows,
  constant leaves, quantized, numeric only. Closes an acknowledged weakness
  (9.5x behind LightGBM at 500k, throughput flat while theirs rises), but
  moves no tracked metric and memory already records that giant-data fit is
  explicitly not a selling point.
- **Option C — full port.** Not recommended. It is a second library, it
  breaks identity testing on the ridge and prep paths, it adds an
  NVIDIA-only dependency, and per the table above its ceiling on the suites
  we actually decide on is ~2-3x.

## Phase 0 (costs nothing, decides everything)

No kernel should be written before these are answered:

1. **Hardware and honesty check (Nathan's call, not measurable):** is there
   a CUDA card on this box, and is a GPU arm something we would ever put on
   a chart, given it must be charted against GPU competitors?
2. **The batching prior, on CPU.** Before any CUDA, confirm the Option-A
   premise by measuring how much of an Ens8 fit is *per-member fixed cost*
   versus scatter work. If members are already saturating 12 CPU cores,
   the batch axis buys less than it looks like.
3. **A single microbench, if a card exists:** the quantized scatter alone,
   n x f = {6k, 37.5k, 200k, 1M} x {10, 60}, numba-cuda global-atomic
   version versus the current CPU kernel, warm. One number decides the
   program: the crossover n at which the GPU histogram beats 12 CPU cores.
   Published guidance for XGBoost puts the GPU crossover near 100k rows
   with <50 features — i.e. **above most of Grinsztajn**. If our crossover
   lands there too, Options B and C are dead on arrival and only Option A
   (which raises the effective n by the batch factor) is live.

**Registered kill clause:** if the measured crossover is above ~50k rows
and the Option-A batching prior shows members already saturating the CPU,
close the program and keep this document as the record. That is a fine
outcome and cheaper than finding it out after a backend exists.

## Open questions for Nathan

- Is GPU a *capability* goal (unlock n >= 1M, cheap ensembling) or a
  *Pareto* goal? The table above says it cannot be the second one at
  current suite sizes.
- Is an optional NVIDIA-only dependency acceptable against the pure-Python
  identity, given the kernel source stays Python?
- If a GPU arm ever shipped, would it get its own chart with GPU
  competitors (the honest option), or stay an unpublished internal
  capability?
