"""Generate tests/golden_metrics.json — the accuracy + timing regression baseline.

Fits a fixed, small, fully-deterministic config (fixed seed, single thread,
early_stopping off) on the 7 offline `run_benchmarks.DATASETS` (no network) and
records, per dataset:
  * metric        primary error metric, lower-is-better (RMSE for regression,
                  log-loss for classification) -- sensitive to accuracy drift.
  * fit_ratio     fit_time / calibration_time
  * predict_ratio predict_time / calibration_time

`calibration_time` is a pure-numpy workload (sorting a fixed array) timed in the
SAME process. Dividing chimera's time by it makes the stored figure machine-
independent (a faster box speeds up both) yet still sensitive to a chimera-only
slowdown (the numpy baseline is untouched by chimera code changes) -- unlike a
calibration that shares chimera's code path, which a uniform slowdown would hide.

`tests/test_no_regression.py` recomputes these on the current code and asserts
they haven't drifted. Regenerate + commit this file ONLY when a change is meant
to move the numbers.

Usage:
    python benchmarks/make_golden.py            # write tests/golden_metrics.json
    python benchmarks/make_golden.py --show     # print current measurements only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "golden_metrics.json")

# Offline panel only (no network). Fixed across runs so metrics are comparable.
PANEL = ["diabetes", "friedman1", "synthetic_reg", "breast_cancer", "wine",
         "cat_binary", "cat_multiclass"]
SEED = 0
SCALE = 0.5          # shrink the synthetic sets so generation stays quick
CONFIG = dict(n_estimators=100, early_stopping=False, thread_count=1,
              random_state=SEED)


def calibration_time(reps=3):
    """Pure-numpy machine-speed probe: median wall time to sort a fixed array.

    Independent of any chimeraboost code, so chimera/calibration time ratios move
    only when chimera itself changes speed, not when the machine does."""
    rng = np.random.default_rng(12345)
    arr = rng.normal(size=4_000_000)
    ts = []
    for _ in range(reps):
        a = arr.copy()
        t = time.perf_counter()
        np.sort(a)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def _metric(task, est, Xte, yte):
    """Lower-is-better primary metric for the task."""
    if task == "regression":
        return float(np.sqrt(mean_squared_error(yte, est.predict(Xte))))
    proba = np.asarray(est.predict_proba(Xte), dtype=float)
    return float(log_loss(yte, proba, labels=list(est.classes_)))


def measure_one(ds, cal):
    """Fit + evaluate one dataset; return dict(metric, fit_ratio, predict_ratio)."""
    rng = np.random.default_rng(SEED)
    X, y, cat, task = rb.DATASETS[ds](SCALE, rng)
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           random_state=SEED, stratify=strat)
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    est = Est(**CONFIG)
    t = time.perf_counter()
    est.fit(Xtr, ytr, cat_features=cat)
    fit_t = time.perf_counter() - t
    t = time.perf_counter()
    est.predict(Xte)
    if task != "regression":
        est.predict_proba(Xte)
    predict_t = time.perf_counter() - t
    return dict(task=task, metric=_metric(task, est, Xte, yte),
                fit_ratio=fit_t / cal, predict_ratio=predict_t / cal)


def measure_all():
    """Measure every panel dataset; returns (records dict, calibration_time)."""
    # Align the ambient numba thread count with CONFIG's thread_count=1: the
    # estimators no longer leak their setting into the process (fit/predict
    # restore it), and a per-call switch to a DIFFERENT count costs ~1 ms in
    # the omp layer, which would swamp the small-panel predict timings.
    # Setting the ambient to match keeps every kernel single-threaded with no
    # per-call switch -- the same regime the golden timings were recorded in.
    import numba
    numba.set_num_threads(1)
    cal = calibration_time()
    return {ds: measure_one(ds, cal) for ds in PANEL}, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="print measurements without writing the golden file")
    args = ap.parse_args()

    records, cal = measure_all()
    print(f"calibration_time = {cal*1e3:.1f} ms (numpy sort of 4e6 floats)\n")
    print(f"{'dataset':16s} {'task':11s} {'metric':>10s} {'fit/cal':>8s} {'pred/cal':>9s}")
    for ds, r in records.items():
        print(f"{ds:16s} {r['task']:11s} {r['metric']:10.5f} "
              f"{r['fit_ratio']:8.2f} {r['predict_ratio']:9.3f}")

    if args.show:
        return 0
    payload = {"config": CONFIG, "scale": SCALE, "seed": SEED, "records": records}
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nwrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
