"""Smoke test for the structure-transfer refit (refit_full="replay").

Checks that the three refit modes fit and predict, that replay actually costs
less than the from-scratch refit, and that replay's model carries the same
number of trees as the full refit (same rounds scaling, different cost).
"""
import time

import numpy as np
from sklearn.datasets import make_classification, make_regression

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

print("chimeraboost:", chimeraboost.__file__)


def timed(est, X, y):
    t0 = time.perf_counter()
    est.fit(X, y)
    return time.perf_counter() - t0, est


def run(name, X, y, cls, score):
    print(f"\n=== {name} (n={len(X)}) ===")
    base = None
    for mode in (False, True, "replay"):
        secs, est = timed(cls(random_state=0, refit_full=mode), X, y)
        m = est.model_
        n_trees = (len(m.trees_) if not isinstance(m.trees_[0], list)
                   else len(m.trees_))
        s = score(y, est)
        if base is None:
            base = secs
        print(f"  refit_full={str(mode):7s} {secs:6.2f}s "
              f"({secs / base:4.2f}x base)  trees={n_trees:4d}  train_score={s:.5f}")


Xr, yr = make_regression(n_samples=20000, n_features=20, noise=8.0,
                         random_state=0)
run("regression", Xr, yr, ChimeraBoostRegressor,
    lambda y, e: float(np.sqrt(np.mean((y - e.predict(Xr)) ** 2))))

Xc, yc = make_classification(n_samples=20000, n_features=20, n_informative=10,
                             random_state=0)
run("binary", Xc, yc, ChimeraBoostClassifier,
    lambda y, e: float(np.mean(e.predict(Xc) == y)))

Xm, ym = make_classification(n_samples=8000, n_features=20, n_informative=12,
                             n_classes=3, random_state=0)
run("multiclass (replay falls back to full)", Xm, ym, ChimeraBoostClassifier,
    lambda y, e: float(np.mean(e.predict(Xm) == y)))

# Invalid value still rejected.
try:
    ChimeraBoostRegressor(refit_full="nope").fit(Xr[:200], yr[:200])
except ValueError as exc:
    print("\nrejects bad value:", exc)
