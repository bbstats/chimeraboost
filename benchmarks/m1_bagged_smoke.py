"""M1: bagged multiclass members run the new selection in parallel workers."""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from chimeraboost import ChimeraBoostClassifier

rng = np.random.default_rng(0)
n = 4000
X = rng.standard_normal((n, 5))
a = X[:, 0] > X[:, 1]
b = X[:, 2] * X[:, 3] > 0
y = np.where(a & b, 0, np.where(a | b, 1, 2))

m = ChimeraBoostClassifier(n_estimators=300, n_ensembles=2, random_state=0)
m.fit(X, y)
sels = [e.cross_features_selected_ for e in m.estimators_]
print("member selections:", sels)
proba = m.predict_proba(X[:10])
assert proba.shape == (10, 3) and np.isfinite(proba).all()
print("bagged multiclass predict_proba OK")
