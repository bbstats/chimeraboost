"""Pre-compile ChimeraBoost's numba kernels.

The hot loops are numba kernels compiled on first use. The machine code is
cached on disk (``cache=True``), but a process on a fresh machine or container
pays the full JIT cost (~5-15 s) inside its first ``fit``, and the first
``predict`` in a fresh process pays ~0.2-2 s (kernel compile or cache load).
Long-lived processes never notice; fleets of short-lived workers (benchmark
harnesses, serverless inference, ray/spark tasks) pay it on every task, where
it can dwarf the actual fit/predict work.

``warmup()`` runs three tiny synthetic fits + predictions chosen to touch
every kernel on the default fit and predict paths, so subsequent real calls
run at steady-state speed. Call it at import/startup time, outside anything
you time or bill.
"""

import threading
import time

import numpy as np

from .sklearn_api import ChimeraBoostClassifier, ChimeraBoostRegressor


def warmup(verbose=False, background=False):
    """Compile (or load from the on-disk cache) all default-path kernels.

    Covers binary classification with linear leaves, a categorical feature
    and a validation set; multiclass; regression with ordered boosting and
    non-uniform sample weights (the weighted ordered-TS kernel); and the
    gdiff cross-feature group-sum kernel — together these touch every fit-
    and predict-path numba kernel except the SHAP kernels (compiled on the
    first ``shap_values`` call).

    Instead of calling this yourself, you can set the environment variable
    ``CHIMERABOOST_WARMUP=1`` to run it automatically when ``chimeraboost``
    is imported (``=background`` uses the daemon thread instead).

    Parameters
    ----------
    verbose : bool, default False
        Print per-stage timings.
    background : bool, default False
        Run in a daemon thread and return it immediately, so compilation
        overlaps the caller's own startup (data loading, connections). A fit
        issued before the thread finishes simply blocks on numba's per-kernel
        compile locks, so it is never slower than compiling inline.

    Returns
    -------
    float or threading.Thread
        Wall-clock seconds spent warming up, or the started daemon thread
        when ``background=True`` (``.join()`` it to wait for readiness).
    """
    if background:
        t = threading.Thread(target=warmup, kwargs={"verbose": verbose},
                             name="chimeraboost-warmup", daemon=True)
        t.start()
        return t
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)

    def _log(msg):
        if verbose:
            print(f"chimeraboost.warmup: {msg} ({time.perf_counter() - t0:.2f}s)")

    # Binary, >= LINEAR_LEAVES_MIN_SAMPLES rows so the linear-leaf kernels
    # compile (they are the binary default), one categorical column for the
    # ordered-TS kernel, an eval_set for the per-round validation predict.
    n = 1152
    X = np.column_stack([rng.standard_normal((n, 3)),
                         rng.integers(0, 3, size=n).astype(np.float64)])
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    clf = ChimeraBoostClassifier(n_estimators=2, random_state=0)
    clf.fit(X[128:], y[128:], cat_features=[3], eval_set=(X[:128], y[:128]))
    clf.predict_proba(X[:8])
    clf.predict_proba(X[:1])   # tiny-batch serial predict kernels
    _log("binary + linear leaves + categoricals")

    # Multiclass (vector-leaf tree build + vector forest predictors; the
    # 8-row call warms the parallel kernel, the 1-row call the serial twin).
    ym = np.digitize(X[:320, 0], [-0.5, 0.5])
    mc = ChimeraBoostClassifier(n_estimators=2, random_state=0)
    mc.fit(X[:320, :3], ym)
    mc.predict_proba(X[:8, :3])
    mc.predict_proba(X[:1, :3])   # constant-forest serial twin
    _log("multiclass")

    # Regression, ordered boosting on (the LOO leaf-step kernel), with a
    # categorical column and NON-uniform sample weights: the weighted
    # ordered-TS kernel (`_ordered_ts_weighted`) and the weighted binner
    # borders only compile on a weighted fit (uniform weights collapse to the
    # unweighted path), which the other stages deliberately keep.
    yr = X[:320, 0] + 0.1 * rng.standard_normal(320)
    sw = rng.uniform(0.5, 1.5, size=320)
    reg = ChimeraBoostRegressor(n_estimators=2, random_state=0,
                                ordered_boosting=True)
    reg.fit(X[:320], yr, cat_features=[3], sample_weight=sw)
    reg.predict(X[:8])
    _log("regression + ordered boosting + weighted categoricals")

    # The gdiff (group-centered cross feature) kernel sits on the default
    # path only for fits >= CROSS_MIN_SAMPLES rows -- too big for a warmup
    # fit, so compile it directly with the dtypes the real call uses.
    from .preprocessing import _grouped_kahan_sum
    _grouped_kahan_sum(np.zeros(4, dtype=np.int64), np.ones(4), 2)
    _log("gdiff group-sum kernel")

    # The fused level kernel (`_build_split_descend`) has one signature for
    # both its small-n and large-n branches, so the small fits above compile
    # everything on the tree-build path — no direct kernel calls needed.

    return time.perf_counter() - t0


def _warmup_from_env(value):
    """Dispatch the ``CHIMERABOOST_WARMUP`` env var (called at package import).

    unset/``""``/``"0"`` — do nothing. ``"background"`` — daemon-thread warmup
    so the import returns immediately (useful only when real startup work
    follows the import for the compile to overlap with). Anything else truthy
    (``"1"``) — plain blocking warmup: the import pays the compile once, and
    every later fit/predict runs at steady-state speed.
    """
    if not value or value.strip() == "0":
        return None
    if value.strip().lower() in ("background", "thread", "bg"):
        return warmup(background=True)
    return warmup()
