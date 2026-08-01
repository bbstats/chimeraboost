"""What the 0.30.0 default flip actually moves, checked rather than asserted.

Prints the resolved rate for the single-model path at three sizes, and confirms
the bagged path and the quantile path are untouched.
"""
import numpy as np

import chimeraboost
from chimeraboost import (ChimeraBoostClassifier, ChimeraBoostQuantileRegressor,
                          ChimeraBoostRegressor)

print("chimeraboost from:", chimeraboost.__file__)


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X[:, 0] * 2 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.normal(scale=0.4, size=n)
    return X, y


FIT = dict(n_estimators=150, early_stopping_rounds=20, random_state=0)

print("\nsingle model, resolved learning rate by dataset size")
print(f"{'rows':>8} {'booster rows':>13} {'default':>9} {'flag off':>9}")
for n in (1_200, 6_250, 12_500, 25_000):
    on = ChimeraBoostRegressor(**FIT).fit(*data(n))
    off = ChimeraBoostRegressor(adaptive_learning_rate=False, **FIT).fit(*data(n))
    print(f"{n:8d} {int(n * 0.8):13d} {on.model_.lr_:9.4f} {off.model_.lr_:9.4f}")

print("\nbagged path (members carry an explicit member learning rate)")
X, y = data(1_200)
bag_on = ChimeraBoostRegressor(n_ensembles=3, **FIT).fit(X, y)
bag_off = ChimeraBoostRegressor(n_ensembles=3, adaptive_learning_rate=False,
                                **FIT).fit(X, y)
same = np.array_equal(bag_on.predict(X[:300]), bag_off.predict(X[:300]))
print(f"  3-member bag identical with the flag on vs off: {same}")

print("\nclassifier path")
yc = (y > np.median(y)).astype(int)
c_on = ChimeraBoostClassifier(**FIT).fit(X, yc)
c_off = ChimeraBoostClassifier(adaptive_learning_rate=False, **FIT).fit(X, yc)
print(f"  default rate {c_on.model_.lr_:.4f}, flag-off rate {c_off.model_.lr_:.4f}")

print("\nquantile path (deliberately pinned to the flat rate)")
q = ChimeraBoostQuantileRegressor(quantiles=[0.25, 0.5, 0.75], n_estimators=60,
                                  random_state=0).fit(X, y)
print(f"  adaptive flag {q.model_.adaptive_learning_rate}, "
      f"resolved rate {q.model_.lr_:.4f}")
