"""B1 smoke check: bagged fit time + pinned selections on the Phase-0 panel.

Fits single + Ens5 (seed-0 split) on the sets where Phase-0 attribution
measured the largest selection redundancy and prints fit seconds, the
bag/single ratio, and each member's pinned selection flags. Phase-0
baselines (clean box): cpu_act bag 2.9s, kick 15.6s, wine-reviews 11.0s,
colleges 6.0s. Run alone for trustworthy times.

Run: python benchmarks/b1_smoke.py
"""
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["gr:reg_num/cpu_act", "hc:kick", "hc:wine-reviews", "hc:colleges"]
BASE_BAG_S = {"gr:reg_num/cpu_act": 2.9, "hc:kick": 15.6,
              "hc:wine-reviews": 11.0, "hc:colleges": 6.0}


def main():
    print(f"chimeraboost: {chimeraboost.__file__}")
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print("Warmup...")
    warmup()
    for key in PANEL:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        strat = y if task != "regression" else None
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=strat)
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        t0 = time.perf_counter()
        Est(random_state=0).fit(Xtr, ytr, cat_features=cat)
        single_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        bag = Est(random_state=0, n_ensembles=5).fit(Xtr, ytr,
                                                     cat_features=cat)
        bag_s = time.perf_counter() - t0
        flags = [(getattr(m, "linear_leaves_selected_", None),
                  m.cross_features_selected_) for m in bag.estimators_]
        print(f"{key}: single {single_s:.1f}s  bag {bag_s:.1f}s "
              f"(x{bag_s / single_s:.1f}; phase-0 bag {BASE_BAG_S[key]}s)  "
              f"member (ll,cf): {flags}")


if __name__ == "__main__":
    main()
