"""Pre-compile ChimeraBoost's numba kernels.

The hot loops are numba kernels compiled on first use. The machine code is
cached on disk (``cache=True``), but a process on a fresh machine or container
pays the full JIT cost (~5-15 s) inside its first ``fit``, and the first
``predict`` in a fresh process pays ~0.2-2 s (kernel compile or cache load).
Long-lived processes never notice; fleets of short-lived workers (benchmark
harnesses, serverless inference, ray/spark tasks) pay it on every task, where
it can dwarf the actual fit/predict work.

``warmup()`` runs a few tiny synthetic fits + predictions chosen to touch
every kernel on the default fit and predict paths, so subsequent real calls
run at steady-state speed. Call it at import/startup time, outside anything
you time or bill -- or run the ``chimeraboost-warmup`` command once after
installing.

Note that numba stamps each cache entry with its source file's modification
time and size, so **upgrading ChimeraBoost invalidates the cache**: the first
run after ``pip install -U`` pays the compile again.
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

from .sklearn_api import ChimeraBoostClassifier, ChimeraBoostRegressor

# Set once the cold-compile notice has been considered, so it prints at most
# once per process. warmup() arms it up front -- its own fits are the compile.
_NOTICE_DONE = False


def _cache_is_cold():
    """True when numba's on-disk cache holds nothing usable for our kernels.

    Checks one kernel per source file, because numba invalidates per ``.py``
    (the cache entry is stamped with that file's mtime and size). An empty
    index means no cache file, a numba version change, or -- the common case
    after ``pip install -U chimeraboost`` -- a stale source stamp.

    Reaches into numba internals, so any failure is read as "warm": a numba
    reorganisation should silence the notice, never break a fit.
    """
    from .binning import _bin_matrix
    from .tree import _build_split_descend_q
    try:
        for kernel in (_bin_matrix, _build_split_descend_q):
            if kernel.signatures:
                continue          # already compiled or cache-loaded here
            if not kernel._cache._cache_file._load_index():
                return True
    except Exception:
        return False
    return False


def _maybe_notice_cold_compile():
    """Print one line to stderr when a cold compile is about to happen.

    A silent 15-second first call is indistinguishable from a hang; this says
    what it is. Fires at most once per process, and only when the disk cache
    really is cold -- so a given machine sees it about once per release, not
    once per session.
    """
    global _NOTICE_DONE
    if _NOTICE_DONE:
        return
    _NOTICE_DONE = True
    if os.environ.get("CHIMERABOOST_NO_NOTICE"):
        return
    if os.environ.get("CHIMERABOOST_WARMUP"):
        return                    # already opted into paying it at import
    if not _cache_is_cold():
        return
    print("chimeraboost: first run on this machine -- compiling numba kernels "
          "(~15 s, one time, then cached on disk). Run `chimeraboost-warmup` "
          "after installing to pay this up front; set CHIMERABOOST_NO_NOTICE=1 "
          "to silence.", file=sys.stderr, flush=True)


def warmup(verbose=False, background=False, shap=False):
    """Compile (or load from the on-disk cache) all default-path kernels.

    Covers binary classification with linear leaves, a categorical feature
    and a validation set; multiclass; regression with ordered boosting and
    non-uniform sample weights (the weighted ordered-TS kernel); and the
    gdiff cross-feature group-sum kernel — together these touch every fit-
    and predict-path numba kernel except the SHAP kernels (``shap=True``).

    Instead of calling this yourself, run ``chimeraboost-warmup`` once after
    installing, or set the environment variable ``CHIMERABOOST_WARMUP=1`` to
    run it automatically when ``chimeraboost`` is imported (``=background``
    uses a daemon thread instead).

    Parameters
    ----------
    verbose : bool, default False
        Print per-stage timings.
    background : bool, default False
        Run in a daemon thread and return it immediately, so compilation
        overlaps the caller's own startup (data loading, connections). A fit
        issued before the thread finishes simply blocks on numba's per-kernel
        compile locks, so it is never slower than compiling inline.
    shap : bool, default False
        Also compile the SHAP kernels. Off by default: it adds ~3.7 s to a
        cold warmup and most callers never use ``shap_values``.

    Returns
    -------
    float or threading.Thread
        Wall-clock seconds spent warming up, or the started daemon thread
        when ``background=True`` (``.join()`` it to wait for readiness).
    """
    global _NOTICE_DONE
    if background:
        t = threading.Thread(target=warmup,
                             kwargs={"verbose": verbose, "shap": shap},
                             name="chimeraboost-warmup", daemon=True)
        t.start()
        return t
    _NOTICE_DONE = True   # this call *is* the compile -- do not narrate it
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
    mc.predict_proba(X[:1, :3])   # vector-leaf serial twin
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
    # 320 rows is below LINEAR_LEAVES_MIN_SAMPLES, so this model has constant
    # leaves: a <= 4-row call here is the only thing that compiles the plain
    # serial predict twin. Without it a warmed serving process still stalls
    # ~0.3 s on its first single-row request.
    reg.predict(X[:1])
    _log("regression + ordered boosting + weighted categoricals")

    # The gdiff (group-centered cross feature) kernel sits on the default
    # path only for fits >= CROSS_MIN_SAMPLES rows -- too big for a warmup
    # fit, so compile it directly with the dtypes the real call uses.
    from .preprocessing import _grouped_kahan_sum
    _grouped_kahan_sum(np.zeros(4, dtype=np.int64), np.ones(4), 2)
    _log("gdiff group-sum kernel")

    # The fused level kernel (`_build_split_descend_q`) has one signature for
    # both its small-n and large-n branches, so the small fits above compile
    # everything on the tree-build path — no direct kernel calls needed.

    # SHAP is opt-in: `_shap_forest_linear` is an expensive parallel kernel
    # (~3.7 s cold) that most callers never reach. The same call also compiles
    # the column-major `_predict_forest_linear`, which the SHAP path uses.
    if shap:
        clf.shap_values(X[:8])
        _log("shap")

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


def _cache_dir():
    """Where numba put the compiled kernels, for the command to report."""
    try:
        from .binning import _bin_matrix
        return _bin_matrix._cache.cache_path
    except Exception:
        return "(numba default)"


def main(argv=None):
    """``chimeraboost-warmup`` — compile the kernels now, not on first fit.

    Run once after ``pip install`` (and again after every upgrade, which
    invalidates numba's cache). Also reachable as
    ``python -m chimeraboost.warmup`` or ``python -m chimeraboost``.
    """
    from . import __version__
    parser = argparse.ArgumentParser(
        prog="chimeraboost-warmup",
        description="Compile ChimeraBoost's numba kernels and cache them on "
                    "disk, so later runs start at full speed. Re-run after "
                    "upgrading ChimeraBoost -- an upgrade resets the cache.")
    parser.add_argument("--shap", action="store_true",
                        help="also compile the SHAP kernels (~3.7 s more)")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-stage timings")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing on success")
    args = parser.parse_args(argv)
    elapsed = warmup(verbose=args.verbose, shap=args.shap)
    if not args.quiet:
        print(f"chimeraboost {__version__}: kernels ready in {elapsed:.1f}s "
              f"(cache: {_cache_dir()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
