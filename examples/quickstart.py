"""Quickstart for ChimeraBoost. Run: python examples/quickstart.py"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from chimeraboost import ChimeraBoostClassifier

# --- numeric-only classification ------------------------------------------
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = ChimeraBoostClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,
    random_state=0,
)
clf.fit(Xtr, ytr, eval_set=(Xte, yte))
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
print(f"breast cancer  AUC={auc:.4f}  trees used={clf.best_iteration_}")

# --- mixed numeric + categorical ------------------------------------------
rng = np.random.default_rng(0)
n = 4000
city = rng.choice(["NYC", "SF", "LA", "CHI", "BOS"], size=n)
plan = rng.choice(["free", "pro", "team", "ent"], size=n)
age = rng.normal(40, 12, n)
usage = rng.gamma(2.0, 2.0, n)

city_w = {"NYC": 0.8, "SF": 1.1, "LA": 0.2, "CHI": -0.4, "BOS": 0.1}
plan_w = {"free": -1.0, "pro": 0.3, "team": 0.9, "ent": 1.6}
logit = (np.array([city_w[c] for c in city])
         + np.array([plan_w[p] for p in plan])
         + 0.02 * (age - 40) + 0.15 * usage + rng.normal(0, 1, n))
churn = (logit > np.median(logit)).astype(int)

X = np.empty((n, 4), dtype=object)
X[:, 0] = city
X[:, 1] = plan
X[:, 2] = age
X[:, 3] = usage
Xtr, Xte, ytr, yte = train_test_split(X, churn, test_size=0.25, random_state=1)

clf = ChimeraBoostClassifier(n_estimators=400, random_state=1)
clf.fit(Xtr, ytr, cat_features=[0, 1])   # columns 0 and 1 are categorical
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
print(f"mixed cat+num  AUC={auc:.4f}")
