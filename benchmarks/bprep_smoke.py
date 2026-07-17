"""B-prep smoke: single + Ens8 fit time and a prediction fingerprint on the
Phase-0 attribution panel. Run once with PYTHONPATH at the BASE worktree and
once from this repo; fingerprints must match EXACTLY (the change is
bit-identical) while fit seconds drop. Run alone for trustworthy times.

Run: python benchmarks/bprep_smoke.py
"""
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["gr:reg_num/cpu_act", "gr:clf_num/MagicTelescope", "hc:kick",
         "hc:wine-reviews", "hc:colleges"]


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
        if task == "regression":
            fp_s = float(np.sum(single.predict(Xte)))
            fp_b = float(np.sum(bag.predict(Xte)))
        else:
            fp_s = float(np.sum(single.predict_proba(Xte)[:, 1]))
            fp_b = float(np.sum(bag.predict_proba(Xte)[:, 1]))
        print(f"{key}: single {single_s:.2f}s  bag {bag_s:.2f}s "
              f"(x{bag_s / single_s:.2f})  fp_single {fp_s:.9f}  "
              f"fp_bag {fp_b:.9f}", flush=True)


if __name__ == "__main__":
    main()
