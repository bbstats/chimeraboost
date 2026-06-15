"""Train-time scaling benchmark on large synthetic data.

Question: how does ChimeraBoost's FIT time scale vs the C++ heavyweights
(LightGBM / XGBoost / sklearn-HGB / CatBoost) as n grows into the millions?

This is a controlled COMPUTE comparison, not an accuracy one:
* Fixed tree count (`--trees`, default 200), NO early stopping -> every library
  builds the same-sized forest, so we measure pure build throughput, not who
  stops earliest.
* Default tree shape per library (ChimeraBoost depth 6 / oblivious, XGB max_depth
  6, LGBM num_leaves 31, HGB max_leaf_nodes 31, CatBoost depth 6) — i.e. what a
  user gets out of the box. Bin counts are each library's default (noted below).
* All cores for everyone. numba is warmed up on a tiny fit BEFORE timing so
  ChimeraBoost's one-off JIT compile is not charged to the first size.

Reports wall-clock fit seconds, throughput (Mrows/s), and s/1K-rows per (size,
model). Synthetic binary classification (make_classification), 30 numeric feats.

Usage:
    python scaling_giant.py --sizes 100000,300000,1000000 --trees 200
    python scaling_giant.py --sizes 1000000,3000000,5000000 --models ChimeraBoost,LightGBM,XGBoost
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np


def _gen(n, n_features=30, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=15, n_redundant=5,
        n_clusters_per_class=2, class_sep=0.8, random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int32)


def _fit_chimera(X, y, trees, threads):
    from chimeraboost import ChimeraBoostClassifier
    m = ChimeraBoostClassifier(
        n_estimators=trees, learning_rate=0.1, depth=6,
        early_stopping=False, thread_count=threads, random_state=0,
    )
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


def _fit_lightgbm(X, y, trees, threads):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=trees, learning_rate=0.1, num_leaves=31,
        n_jobs=threads, random_state=0, verbosity=-1,
    )
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


def _fit_xgboost(X, y, trees, threads):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=trees, learning_rate=0.1, max_depth=6,
        tree_method="hist", n_jobs=threads, random_state=0, verbosity=0,
    )
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


def _fit_sklearn(X, y, trees, threads):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # HGB parallelises via OpenMP (no n_jobs arg); threads set via env in main().
    m = HistGradientBoostingClassifier(
        max_iter=trees, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=False, random_state=0,
    )
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


def _fit_catboost(X, y, trees, threads):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        n_estimators=trees, learning_rate=0.1, depth=6,
        thread_count=threads, verbose=False, random_seed=0,
    )
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


FITTERS = {
    "ChimeraBoost": _fit_chimera,
    "LightGBM": _fit_lightgbm,
    "XGBoost": _fit_xgboost,
    "sklearn_HGB": _fit_sklearn,
    "CatBoost": _fit_catboost,
}


def _available(names):
    ok = []
    for n in names:
        if n == "ChimeraBoost":
            ok.append(n); continue
        mod = {"LightGBM": "lightgbm", "XGBoost": "xgboost",
               "sklearn_HGB": "sklearn", "CatBoost": "catboost"}[n]
        try:
            __import__(mod); ok.append(n)
        except ImportError:
            print(f"  (skip {n}: {mod} not installed)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,300000,1000000",
                    help="comma-separated row counts")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--features", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0,
                    help="thread budget per model (0 = all cores)")
    ap.add_argument("--models", default=",".join(FITTERS))
    ap.add_argument("--out", default=None, help="json output path")
    args = ap.parse_args()

    threads = args.threads or (os.cpu_count() or 1)
    # HGB uses OpenMP threads, not an n_jobs arg.
    os.environ["OMP_NUM_THREADS"] = str(threads)

    sizes = [int(s) for s in args.sizes.split(",")]
    models = _available([m.strip() for m in args.models.split(",")])
    print(f"threads={threads}  trees={args.trees}  features={args.features}")
    print(f"models={models}")
    print(f"sizes={sizes}\n")

    # Warm up every model (esp. ChimeraBoost's numba JIT) on a tiny fit so the
    # one-off compile cost is not charged to the first real size.
    print("warming up (JIT compile, import)...")
    Xw, yw = _gen(2000, args.features, seed=99)
    for name in models:
        try:
            FITTERS[name](Xw, yw, min(args.trees, 20), threads)
        except Exception as e:
            print(f"  warmup {name} failed: {e}")
    print("warm.\n")

    results = {}
    for n in sizes:
        print(f"=== n={n:,} ===")
        X, y = _gen(n, args.features, seed=0)
        for name in models:
            try:
                secs = FITTERS[name](X, y, args.trees, threads)
                thrpt = n / secs / 1e6           # Mrows/s
                s_per_1k = secs / (n / 1000.0)
                results.setdefault(name, {})[n] = secs
                print(f"  {name:14s} {secs:8.2f}s   {thrpt:6.3f} Mrows/s   "
                      f"{s_per_1k:7.4f} s/1K")
            except Exception as e:
                print(f"  {name:14s} FAILED: {e}")
        del X, y
        print()

    # Summary table: rows = sizes, cols = models (fit seconds).
    print("\n===== FIT TIME (seconds) =====")
    hdr = "rows".rjust(12) + "".join(m.rjust(14) for m in models)
    print(hdr)
    for n in sizes:
        row = f"{n:12,}" + "".join(
            f"{results.get(m, {}).get(n, float('nan')):14.2f}" for m in models)
        print(row)

    # Speedup vs ChimeraBoost (×: >1 means competitor is faster).
    if "ChimeraBoost" in models:
        print("\n===== SPEED RATIO vs ChimeraBoost (>1 = competitor faster) =====")
        print(hdr)
        for n in sizes:
            cb = results.get("ChimeraBoost", {}).get(n)
            cells = []
            for m in models:
                v = results.get(m, {}).get(n)
                cells.append(f"{(cb / v):14.2f}" if (cb and v) else "nan".rjust(14))
            print(f"{n:12,}" + "".join(cells))

    out = args.out or os.path.join(
        os.path.dirname(__file__), "results",
        f"scaling-{datetime.now():%Y%m%d-%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": vars(args), "threads": threads,
                   "results": {m: {str(k): v for k, v in d.items()}
                               for m, d in results.items()}}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
