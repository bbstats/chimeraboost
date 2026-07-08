# Verification notes — cold-start / 0.13.1 claims (this machine)

Measured on dev hardware (Windows 11, Python 3.13) on branch `release/0.13.1`
(source identical to the merged `_solve_small` solver in commit 832e7a2).

## 1. Compile profile — structural pass

Fresh `NUMBA_CACHE_DIR`, 2000-row binary fit, `n_estimators=60`, single thread.
Monkeypatch on `numba.core.dispatcher.Dispatcher.compile`.

- **`linalg.*` entries: NONE.** The LAPACK bindings are structurally gone — this
  is the real acceptance signal, not the percentage.
- **`tree._solve_small` present at 0.36s** (claim was ~0.3s — matches).
- First-fit wall: **6.54s**.

Top compile costs:
```
  1.63s  tree._best_split
  1.32s  tree._linear_leaf_fit
  1.04s  tree._build_histograms_into
  0.81s  arrayobj.impl
  0.76s  tree._predict_forest_linear
  0.63s  binning._bin_matrix
  0.36s  tree._solve_small
  0.31s  losses._sigmoid
  0.23s  tree._leaf_values
  0.22s  tree._descend_leaves
```

For reference, the pre-patch profile (measured earlier this session on ac90fe8)
showed `linalg.solve_impl` ~2.43s + `linalg.oneD_impl` ~1.74s plus the LAPACK
helper cluster, with `_linear_leaf_fit` inflated to ~3.42s. Post-patch
`_linear_leaf_fit` is down to ~1.32s and the linalg cluster is absent.

## 2. Import cost — DISCREPANCY, correct before posting

Fresh process, `python -c "import time; t0=time.time(); import chimeraboost; ..."`:

- Run 1: **1.39s**
- Run 2: **1.40s**
- Run 3: **1.39s**

**The draft comment says "~1s". This machine measures a consistent ~1.4s.**
Recommend the public comment say "~1.4s on dev hardware" (or "~1–1.5s,
machine-dependent") rather than "~1s". This is import time only (numba +
numpy + chimeraboost module load); it is what a lazy import inside `_fit`
would have billed to every job's reported train time — the motivation for
hoisting the import to module top.

## 3. Cold-start reduction

~25% first-fit JIT reduction on this hardware (measured earlier this session,
8.52s → 6.35s), NOT the 35–40% from the original draft (that band was overfit
to a container with slower LAPACK JIT on older numba/LLVM). Use the ~25% /
"roughly a quarter" figure in any public comment.
