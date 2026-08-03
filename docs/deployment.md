# Deployment

What to know when a fitted model leaves the notebook: first-call compile cost, thread
control, concurrency, prediction latency, big batches, model size, and fit-time memory.

## The first call pays the numba compile

ChimeraBoost's kernels are JIT-compiled by numba on first use. With a cold cache the
first `fit` takes seconds; in steady state it is milliseconds. Compiled kernels are
cached on disk per user, so later processes pay only a small cache-load cost.

Run this once after installing and the wait never lands on a real call:

```
pip install -U chimeraboost
chimeraboost-warmup
```

**Re-run it after every upgrade.** Numba stamps each cache entry with its source file's
modification time and size, so installing a new ChimeraBoost version invalidates the
cache and the next run compiles from scratch again. This surprises people who assumed
the cost was once per machine.

Measured on a 12-core desktop:

| First call in a fresh process | cold cache | warm cache | steady state |
|---|---|---|---|
| `import chimeraboost` | 1.35 s | 1.35 s | n/a |
| `warmup()` | 13.2 s | 0.97 s | n/a |
| `fit` (3K rows, 2 categorical) | 10.1 s | 0.83 s | 0.27 s |
| `predict` (1 row, unpickled model) | 0.98 s | 0.27 s | 0.5 ms |

The 0.5 ms in the last row is a repeat call on a classifier with two categorical columns,
so it carries the categorical re-mapping cost. A plain numeric model is faster — see
[Prediction latency](#prediction-latency) for the per-call floor.

Most of that cold compile goes into parallel kernels, which are expensive to compile and
several times faster to run. That trade is why the wait is seconds rather than
milliseconds.

Serving is the shape that hurts: a process that only unpickles a model pays the
first-predict cost on its first request, and `warmup()` removes it.

When you cannot run the command, as with serverless, per-request processes, and
benchmark harnesses that spawn fresh workers, compile from inside the process instead:

```python
import chimeraboost
chimeraboost.warmup()          # compile everything now, not on first fit
```

You can also set `CHIMERABOOST_WARMUP=1` to compile at import, or
`CHIMERABOOST_WARMUP=background` to compile on a daemon thread while your process
boots. Timing a fresh worker without warmup measures numba's compiler rather than the
model.

`warmup()` skips the SHAP kernels by default, since most callers never use them and
they add about 3.7 s. If you serve explanations, use `warmup(shap=True)` or
`chimeraboost-warmup --shap`.

The first fit on a cold cache prints a one-line notice to stderr explaining the wait.
It appears only when the cache really is cold, which is roughly once per installed
version, and `CHIMERABOOST_NO_NOTICE=1` silences it.

There is no way to move this compile into `pip install`. Numba's ahead-of-time compiler
was retired, and a pre-built cache cannot ship in a wheel because its key includes the
exact CPU model and feature set of the machine that built it, plus the installed file's
timestamp. One command after install is the whole of the fix.

## Thread control

`thread_count` applies to both `fit` and `predict`, and the process-global numba thread
setting is restored afterwards, so fitting one model with `thread_count=1` does not cap
other numba work in your process.

For serving, prefer controlling the *ambient* thread count, through the
`NUMBA_NUM_THREADS` environment variable (before the first numba use) or
`numba.set_num_threads(n)`, and leave `thread_count=None`. A model whose
`thread_count` matches the ambient count applies it for free. A count that differs is
switched and restored on every call, which is usually cheap but has measured up to
about 1 ms per call in some process states, since numba's OpenMP layer re-teams on the
switch.

## Serving one model from several threads

One fitted model can serve `predict`, `predict_proba` and `staged_predict` from as many
threads as you like. Fitted state is read-only after `fit`, and everything else a
prediction touches is allocated fresh per call. The one write is a lazily-built internal
cache, and building it twice produces exactly the same bytes, so a race there costs
duplicated work and nothing more. No lock is needed, and no copy per worker — share the
one object. That cache is empty on a model just loaded from a pickle, so call `predict`
once on a single row before the process accepts traffic — concurrent first calls are safe,
but the rebuild does not belong on a real request.

Threads give you safety, though not much extra throughput: the compiled kernels hold the
GIL, so overlapping calls take turns inside the forest walk. The parallelism that matters
is inside each call, where numba spreads the work across cores itself. If you need more
requests per second than one process delivers, run more processes. Each holds its own copy
of the model, which is always safe and costs about 0.5 to 1.7 MB (see
[Model size and persistence](#model-size-and-persistence)).

Two things to watch:

- `shap_values` sets `expected_value_` on the estimator, so concurrent calls can leave you
  reading a baseline that belongs to another call. The returned contribution array is
  always correct; only the attribute races. Serialize those calls if you read it.
- An explicit `thread_count` is applied by changing a process-global numba setting for the
  duration of the call and restoring it afterwards. Overlapping calls can interleave those
  swaps and strand the process on the wrong count. Leave `thread_count=None` and set the
  ambient count instead, as above.

`fit` is the opposite: it writes to the estimator, so concurrent fits each need their own
estimator object. They also share one thread pool, so running several at once splits the
same cores rather than adding any.

## Prediction latency

Small-batch predict is dominated by fixed per-call overhead (input validation, dtype
conversion, binning) rather than by the forest walk:

| Batch (warm, numeric ndarray) | time/call |
|---|---|
| 1 row | ≈ 0.05 ms (regressor) / 0.09 ms (classifier) |
| 1,000 rows | ≈ 0.11 ms / 0.14 ms |

- A one-row pandas DataFrame costs about 0.27 ms in conversion overhead. Pass an
  ndarray on the hot path.
- If your serving data is already validated, skip the finiteness scan with
  scikit-learn's `assume_finite` (see
  [FAQ](faq.md#how-can-i-make-inference-faster)), worth about 10% at this scale.
- Models fit with `cat_features` re-map categorical strings to codes on every call:
  about 0.2 ms for a one-row ndarray, roughly twice the numeric classifier path.

## Big batches

`predict` processes its input in one pass: a float64 copy of `X` plus a binned matrix,
roughly 10 bytes per cell of transient memory on top of your input. For very large
scoring jobs (tens of millions of rows), chunk the calls. Predictions are
row-independent, so the results are identical:

```python
preds = np.concatenate([model.predict(X[i:i + 1_000_000])
                        for i in range(0, len(X), 1_000_000)])
```

## Model size and persistence

A fitted estimator pickles like any scikit-learn object (see
[Recipes](recipes.md#save-and-load-a-model)). A 500-tree model is roughly 0.5 MB on
disk, or about 1.7 MB with linear leaves. Pickles leave out an internal prediction cache,
so the first predict after loading is slower while it rebuilds; later calls are
unaffected. In a fresh process that rebuild is part of the first-predict cost measured
under [the numba compile](#the-first-call-pays-the-numba-compile).

Pickles are not guaranteed to load across ChimeraBoost versions, so store
`chimeraboost.__version__` alongside the model and re-fit after an upgrade — see
[How do I save and load a model?](faq.md#how-do-i-save-and-load-a-model) for the pattern.

## Fit-time memory

The split search allocates a histogram buffer of shape
`(n_features, 2^depth, max_bins)`. That is negligible at the default `depth=6` but
grows exponentially with depth: at `depth=14` on 100 features it is about 1.7 GB, and
`depth=16` several times that. Raise depth on wide data with the buffer in mind.
Categorical columns expand before binning (one encoded column per class for multiclass,
plus pairwise combination columns when `cat_combinations` is on), and each expanded
column gets its own histogram slab.
