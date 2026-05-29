"""Investigate whether early stopping hurts accuracy, and why.

Disentangles three confounded effects of `early_stopping=True`:
  1. auto LR: ES -> 0.1, no-ES -> clip(20/iters)=0.04 (bigger, finer ensemble)
  2. patience (early_stopping_rounds): default 10 when ES is on
  3. the validation holdout: ES carves off validation_fraction of train data

We hold each lever fixed in turn so we can attribute the gap.
"""
import time
import numpy as np
from sklearn.datasets import (load_diabetes, load_breast_cancer, load_wine,
                              load_digits, fetch_california_housing,
                              make_regression, make_classification)
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, accuracy_score

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


def _datasets():
    out = {}
    X, y = load_diabetes(return_X_y=True);                 out["diabetes"] = ("reg", X, y)
    X, y = fetch_california_housing(return_X_y=True);      out["california"] = ("reg", X, y)
    X, y = make_regression(n_samples=4000, n_features=20, n_informative=10,
                           noise=12.0, random_state=0);    out["synth_reg"] = ("reg", X, y)
    X, y = load_breast_cancer(return_X_y=True);            out["breast_cancer"] = ("clf", X, y)
    X, y = make_classification(n_samples=4000, n_features=20, n_informative=10,
                               random_state=0);            out["synth_clf"] = ("clf", X, y)
    X, y = load_wine(return_X_y=True);                     out["wine"] = ("clf", X, y)
    X, y = load_digits(return_X_y=True);                   out["digits"] = ("clf", X, y)
    return out


# (label, kwargs).  None LR = library auto-default.
CONFIGS = [
    ("no-ES default (auto lr=0.04, 500 trees)", dict(iterations=500, early_stopping=False)),
    ("ES default (auto lr=0.1, patience=10)",   dict(iterations=2000, early_stopping=True, early_stopping_rounds=10)),
    # --- hold LR fixed at 0.1 to remove the auto-LR confound ---
    ("no-ES  lr=0.1, 500 trees",                dict(iterations=500, early_stopping=False, learning_rate=0.1)),
    ("ES     lr=0.1, patience=10",              dict(iterations=2000, early_stopping=True, early_stopping_rounds=10, learning_rate=0.1)),
    ("ES     lr=0.1, patience=50",              dict(iterations=2000, early_stopping=True, early_stopping_rounds=50, learning_rate=0.1)),
    ("ES     lr=0.1, patience=100",             dict(iterations=2000, early_stopping=True, early_stopping_rounds=100, learning_rate=0.1)),
    # --- hold LR fixed at 0.04 (the no-ES default) but turn ES on ---
    ("no-ES  lr=0.04, 500 trees",               dict(iterations=500, early_stopping=False, learning_rate=0.04)),
    ("ES     lr=0.04, patience=50",             dict(iterations=2000, early_stopping=True, early_stopping_rounds=50, learning_rate=0.04)),
]

SEEDS = [0, 1, 2, 3, 4]


def run():
    data = _datasets()
    # warm numba
    Xw, yw = load_wine(return_X_y=True)
    ChimeraBoostClassifier(iterations=20).fit(Xw, yw)

    for name, (kind, X, y) in data.items():
        higher_better = (kind == "clf")
        print(f"\n### {name}  [{kind}]  (n={len(y)})  "
              f"metric={'accuracy' if higher_better else 'RMSE'}")
        rows = {}
        for label, kw in CONFIGS:
            scores, ntrees, times = [], [], []
            for s in SEEDS:
                strat = y if kind == "clf" else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=s, stratify=strat)
                Est = ChimeraBoostClassifier if kind == "clf" else ChimeraBoostRegressor
                m = Est(random_state=s, **kw)
                t0 = time.time()
                m.fit(Xtr, ytr)
                times.append(time.time() - t0)
                pred = m.predict(Xte)
                scores.append(accuracy_score(yte, pred) if higher_better
                              else root_mean_squared_error(yte, pred))
                ntrees.append(m.best_iteration_)
            rows[label] = (float(np.mean(scores)), float(np.mean(ntrees)),
                           float(np.mean(times)))
        # Print sorted by score (best first).
        ordered = sorted(rows.items(), key=lambda kv: kv[1][0],
                         reverse=higher_better)
        best = ordered[0][1][0]
        for label, (sc, nt, tm) in ordered:
            gap = (sc - best) if higher_better else (best - sc)
            tag = "  <-- best" if label == ordered[0][0] else ""
            print(f"  {label:42s} {sc:9.4f}  (gap {gap:+.4f})  "
                  f"trees~{nt:6.0f}  {tm*1000:6.0f}ms{tag}")


if __name__ == "__main__":
    run()
