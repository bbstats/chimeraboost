"""Grow-kernels smoke (GROW_PLAN.md ship shape): single + Ens8 fit time and
byte-exact prediction fingerprints on a panel spanning the fused level
kernel's branches -- small-n (cpu_act, MagicTelescope), large-n (kick,
wine-reviews), multiclass (okcupid-stem). Run once with PYTHONPATH at the
BASE worktree and once from this repo; md5 fingerprints must match EXACTLY
(the change is bit-identical) while small-n fit seconds drop. Run alone for
trustworthy times.

Run: python benchmarks/grow_smoke.py
"""
import hashlib
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["gr:reg_num/cpu_act", "gr:clf_num/MagicTelescope", "hc:kick",
         "hc:wine-reviews", "hc:okcupid-stem"]


def _fp(model, Xte, task):
    p = model.predict(Xte) if task == "regression" \
        else model.predict_proba(Xte)
    return hashlib.md5(np.ascontiguousarray(p).tobytes()).hexdigest()[:16]


def main():
    print(f"chimeraboost: {chimeraboost.__file__}")
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print("Warmup...")
    warmup()
    for key in PANEL:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25,
                                            random_state=0, stratify=strat)
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        t0 = time.perf_counter()
        single = Est(random_state=0).fit(Xtr, ytr, cat_features=cat)
        single_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        bag = Est(random_state=0, n_ensembles=8).fit(Xtr, ytr,
                                                     cat_features=cat)
        bag_s = time.perf_counter() - t0
        print(f"{key}: single {single_s:.2f}s  bag {bag_s:.2f}s  "
              f"fp_single {_fp(single, Xte, task)}  "
              f"fp_bag {_fp(bag, Xte, task)}", flush=True)


if __name__ == "__main__":
    main()
