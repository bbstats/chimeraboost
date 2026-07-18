"""M1 implementation smoke: multiclass cross selection on real + synth data.

Arms: default (selection on) vs cross_features=False (== pre-M1 behavior,
bit-identical path). Reports the selection verdict, fit times (the plan's
<= ~2.5x envelope), and holdout F1/logloss. One seed; NOT decision-grade.
"""
import sys
import time

sys.path.insert(0, "benchmarks")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier
from run_benchmarks import DATASETS, _add_highcard_datasets
from synthgen import api

print("chimeraboost:", chimeraboost.__file__)
_add_highcard_datasets()
rng = np.random.default_rng(0)


def load(name):
    if name.startswith("syn:"):
        X, y, cat, task, _meta = api.build_dataset(name)
        return X, y, cat
    X, y, cat, task = DATASETS[name](1.0, rng)
    return X, y, cat


for name in ("hc:okcupid-stem", "syn:v2/531", "syn:v2/663"):
    X, y, cat = load(name)
    strat = y
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=strat)
    row = {"set": name}
    for label, kw in (("off", {"cross_features": False}), ("auto", {})):
        m = ChimeraBoostClassifier(n_estimators=2000, early_stopping_rounds=50,
                                   random_state=0, **kw)
        t0 = time.time()
        m.fit(Xtr, ytr, cat_features=cat)
        t = time.time() - t0
        proba = m.predict_proba(Xte)
        row[label] = dict(
            fit_s=t, sel=m.cross_features_selected_,
            n_pairs=len(m.cross_pairs_ or []),
            f1=f1_score(yte, m.predict(Xte), average="macro"),
            ll=log_loss(yte, proba, labels=m.classes_))
    off, auto = row["off"], row["auto"]
    print(f"{name}: sel={auto['sel']} pairs={auto['n_pairs']} | "
          f"fit {off['fit_s']:.2f}s -> {auto['fit_s']:.2f}s "
          f"({auto['fit_s'] / off['fit_s']:.2f}x) | "
          f"F1 {off['f1']:.4f} -> {auto['f1']:.4f} | "
          f"logloss {off['ll']:.4f} -> {auto['ll']:.4f}")
