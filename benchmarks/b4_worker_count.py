"""B4 follow-up: worker-count sweep on the imbalance-prone sets.

ensemble_n_jobs in {1,2,3,5} on the sets where 5 workers only bought ~1.2x
(long-member imbalance suspected) plus kick as the good-case control.

Run alone: python benchmarks/b4_worker_count.py
"""
import time

import numpy as np
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["hc:colleges", "gr:reg_cat/nyc-taxi-green-dec-2016", "hc:kick"]
JOBS = (1, 2, 3, 5)


def main():
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print("Warmup...")
    warmup()
    print(f"{'dataset':40s} " + " ".join(f"j={j:>2d}" .rjust(7) for j in JOBS))
    for key in PANEL:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        strat = y if task != "regression" else None
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=strat)
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        row = []
        for jobs in JOBS:
            m = Est(random_state=0, n_ensembles=5, ensemble_n_jobs=jobs)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr, cat_features=cat)
            row.append(time.perf_counter() - t0)
        print(f"{key:40s} " + " ".join(f"{t:7.1f}" for t in row))


if __name__ == "__main__":
    main()
