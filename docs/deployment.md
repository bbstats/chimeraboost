# Deployment

What to know when a fitted model leaves the notebook: first-call compile
cost, thread control, prediction latency, big batches, model size, and
fit-time memory.

## The first call pays the numba compile

ChimeraBoost's kernels are JIT-compiled by numba on first use. In a fresh
process with a cold cache, the first `fit` takes several seconds and the
first `predict` over a second — steady-state they are milliseconds
(numbers below). Compiled kernels are cached on disk per user, so later
processes pay only a small cache-load cost.

Measured on a 12-core desktop (0.18-era code):

| First call in a fresh process | stone-cold | warm disk cache |
|---|---|---|
| `fit` (1K rows) | 4.4–9.3 s | ≈ 0.5 s |
| `predict` (1K rows) | 1.3–1.8 s | ≈ 0.2 s |

The serving shape is the painful one: a process that only unpickles a model
pays that first-predict cost on its first request. `warmup()` in a fresh
process took 0.38 s and removed it entirely (first predict ≈ 1 ms after).

For short-lived workers (serverless, per-request processes, benchmark
harnesses that spawn fresh workers), pre-compile at import time:

```python
import chimeraboost
chimeraboost.warmup()          # compile everything now, not on first fit
```

or set the environment variable `CHIMERABOOST_WARMUP=1` (compile at
import) / `CHIMERABOOST_WARMUP=background` (compile on a daemon thread
while your process boots). Timing a fresh worker without warmup measures
numba's compiler, not the model.

## Thread control

`thread_count` applies to both `fit` and `predict`, and the process-global
numba thread setting is restored afterwards — fitting one model with
`thread_count=1` does not cap other numba work in your process.

For serving, prefer controlling the *ambient* thread count — the
`NUMBA_NUM_THREADS` environment variable (before the first numba use) or
`numba.set_num_threads(n)` — and leave `thread_count=None`. A model whose
`thread_count` matches the ambient count applies it for free; a count that
*differs* is switched and restored on every call, which is usually cheap
but has measured up to ~1 ms per call in some process states (numba's omp
layer re-teams on the switch).

## Prediction latency

Small-batch predict is dominated by fixed per-call overhead (input
validation, dtype conversion, binning), not the forest walk:

| Batch (warm, numeric ndarray) | time/call |
|---|---|
| 1 row | ≈ 0.05 ms (regressor) / 0.09 ms (classifier) |
| 1,000 rows | ≈ 0.11 ms / 0.14 ms |

- A one-row pandas DataFrame costs ≈ 0.27 ms (conversion overhead); pass an
  ndarray on the hot path.
- If your serving data is already validated, skip the finiteness scan with
  sklearn's `assume_finite` (see [FAQ](faq.md#how-can-i-make-inference-faster))
  — worth ~10% at this scale.
- Models fit with `cat_features` decode categorical columns through pandas
  on every call: ≈ 1.7 ms per call regardless of batch size. Sub-2 ms is
  fine for most serving, but it is the one predict path with no
  microsecond option.

## Big batches

`predict` processes its input in one pass: a float64 copy of `X` plus a
binned matrix — roughly 10 bytes per cell of transient memory on top of
your input. For very large scoring jobs (tens of millions of rows), chunk
the calls; predictions are row-independent, so the results are identical:

```python
preds = np.concatenate([model.predict(X[i:i + 1_000_000])
                        for i in range(0, len(X), 1_000_000)])
```

## Model size and persistence

A fitted estimator pickles like any scikit-learn object (see
[Recipes](recipes.md#save-and-load-a-model)); a 500-tree model is roughly
0.5 MB on disk (≈ 1.7 MB with linear leaves). The packed predict cache is excluded from pickles
automatically and rebuilds on the first predict after loading. Pickles are
not guaranteed to load across ChimeraBoost versions — store the version
alongside the model (see [FAQ](faq.md#is-the-api-stable)).

## Fit-time memory

The split search allocates a histogram buffer of shape
`(n_features, 2^depth, max_bins)` — negligible at the default `depth=6`,
but exponential in depth: at `depth=14` on 100 features it is ~1.7 GB, and
`depth=16` several times that. Raise depth on wide data with the buffer in
mind. Categorical columns expand before binning (one encoded column per
class for multiclass, plus pairwise combination columns when
`cat_combinations` is on), and each expanded column gets its own histogram
slab.
