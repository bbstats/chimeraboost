"""B4 (BAGGING_PLAN.md): parallel bag members vs sequential, same core budget.

Times an Ens5 fit with ensemble_n_jobs=1 (members sequential, each using all
cores) vs ensemble_n_jobs=5 (members concurrent at cores/5 numba threads
each) on real panel sets, and checks the predictions match (members are
independently seeded, so scheduling must not change the model). Numba's
sublinear thread scaling is the bet; per-process JIT/import cost in the
joblib workers counts against it honestly.

Run alone (clean box): python benchmarks/b4_parallel_timing.py
"""
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["gr:reg_num/cpu_act", "gr:reg_num/diamonds", "hc:kick",
         "hc:wine-reviews", "hc:colleges",
         "gr:reg_cat/nyc-taxi-green-dec-2016"]
REPS = 2


def main():
    import numba
    print(f"chimeraboost: {chimeraboost.__file__}")
    print(f"cores={os.cpu_count()}  numba_threads={numba.config.NUMBA_NUM_THREADS}")
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print("Warmup...")
    warmup()
    print(f"{'dataset':40s} {'seq_s':>7s} {'par_s':>7s} {'speedup':>8s}  pred check")
    for key in PANEL:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25,
                                            random_state=0, stratify=strat)
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        times = {}
        preds = {}
        for jobs in (1, 5):
            best = np.inf
            for _ in range(REPS):
                m = Est(random_state=0, n_ensembles=5, ensemble_n_jobs=jobs)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr, cat_features=cat)
                best = min(best, time.perf_counter() - t0)
            times[jobs] = best
            preds[jobs] = (m.predict(Xte[:500]) if task == "regression"
                           else m.predict_proba(Xte[:500]))
        same = np.allclose(preds[1], preds[5], rtol=1e-9, atol=1e-12)
        print(f"{key:40s} {times[1]:7.1f} {times[5]:7.1f} "
              f"{times[1] / times[5]:7.2f}x  {'identical' if same else 'DIFFER'}")


if __name__ == "__main__":
    main()
