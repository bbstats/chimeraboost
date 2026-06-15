"""Inference (predict) throughput at scale: ChimeraBoost vs the C++ heavyweights.

Companion to scaling_giant.py (which measures FIT). Fits each model once on a
fixed training set (200 trees, default shape), then times predict_proba on a
large held-out batch — the metric that matters for serving. Reports rows/s.

Usage:
    python scaling_predict.py --train 200000 --predict 2000000 --trees 200
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scaling_giant import _gen  # reuse the synthetic generator


def _bench(name, fit_fn, predict_fn, Xtr, ytr, Xpred, trees, threads, reps=3):
    model = fit_fn(Xtr, ytr, trees, threads)
    predict_fn(model, Xpred[:1000])  # warm
    best = min(_time(predict_fn, model, Xpred) for _ in range(reps))
    return best


def _time(predict_fn, model, X):
    t = time.perf_counter()
    predict_fn(model, X)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200000)
    ap.add_argument("--predict", type=int, default=2000000)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--features", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    threads = args.threads or (os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    def cb_fit(X, y, trees, th):
        from chimeraboost import ChimeraBoostClassifier
        m = ChimeraBoostClassifier(n_estimators=trees, depth=6, learning_rate=0.1,
                                   early_stopping=False, thread_count=th, random_state=0)
        return m.fit(X, y)

    def lgb_fit(X, y, trees, th):
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=trees, num_leaves=31, learning_rate=0.1,
                                  n_jobs=th, random_state=0, verbosity=-1).fit(X, y)

    def xgb_fit(X, y, trees, th):
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=trees, max_depth=6, learning_rate=0.1,
                                 tree_method="hist", n_jobs=th, random_state=0,
                                 verbosity=0).fit(X, y)

    def hgb_fit(X, y, trees, th):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=trees, max_leaf_nodes=31,
                                              learning_rate=0.1, early_stopping=False,
                                              random_state=0).fit(X, y)

    def cat_fit(X, y, trees, th):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(n_estimators=trees, depth=6, learning_rate=0.1,
                                  thread_count=th, verbose=False, random_seed=0).fit(X, y)

    models = {
        "ChimeraBoost": (cb_fit, lambda m, X: m.predict_proba(X)),
        "LightGBM": (lgb_fit, lambda m, X: m.predict_proba(X)),
        "XGBoost": (xgb_fit, lambda m, X: m.predict_proba(X)),
        "sklearn_HGB": (hgb_fit, lambda m, X: m.predict_proba(X)),
        "CatBoost": (cat_fit, lambda m, X: m.predict_proba(X)),
    }
    avail = {}
    for n, v in models.items():
        if n == "ChimeraBoost":
            avail[n] = v; continue
        mod = {"LightGBM": "lightgbm", "XGBoost": "xgboost",
               "sklearn_HGB": "sklearn", "CatBoost": "catboost"}[n]
        try:
            __import__(mod); avail[n] = v
        except ImportError:
            print(f"(skip {n})")

    print(f"train={args.train:,}  predict={args.predict:,}  trees={args.trees}  threads={threads}\n")
    Xtr, ytr = _gen(args.train, args.features, seed=0)
    Xpred, _ = _gen(args.predict, args.features, seed=1)

    print(f"{'model':14s}{'predict s':>12s}{'Mrows/s':>12s}")
    rows = {}
    for name, (fit_fn, pred_fn) in avail.items():
        try:
            secs = _bench(name, fit_fn, pred_fn, Xtr, ytr, Xpred, args.trees, threads)
            rows[name] = args.predict / secs / 1e6
            print(f"{name:14s}{secs:12.3f}{rows[name]:12.3f}")
        except Exception as e:
            print(f"{name:14s} FAILED: {e}")

    if "ChimeraBoost" in rows:
        print("\npredict speed ratio (ChimeraBoost / competitor; >1 = ChimeraBoost faster):")
        for name in avail:
            if name != "ChimeraBoost" and name in rows:
                print(f"  vs {name:12s} {rows['ChimeraBoost']/rows[name]:.2f}x")


if __name__ == "__main__":
    main()
