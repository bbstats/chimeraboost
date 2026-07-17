"""B7 probe: ISLE-style post-hoc reweighting of the bagged forest (regression).

Zero library change. Per dataset x seed: 75/25 train/test; train splits into
80% member-fit / 20% reweight. Arms:
  bag100  Ens8 on the full train (the shipped mode)
  bag80   Ens8 on the 80% fit portion (fair data-tax baseline)
  rw      bag80 flattened to K x T trees, nonnegative LassoCV weights fit on
          the 20% reweight split (intercept absorbs member inits)
Registered kill bar (BAGGING_PLAN.md): B7 dies unless rw beats bag100 on a
majority of (set, seed) pairs. Identity guard: each member's flattened tree
sum must reproduce member.predict on test rows.

Run: python benchmarks/b7_reweight_probe.py
"""
import json
import time

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

PANEL = ["gr:reg_num/cpu_act", "gr:reg_cat/house_sales",
         "gr:reg_num/wine_quality", "hc:wine-reviews", "hc:colleges"]
SEEDS = [0, 1, 2]
K = 8


def member_tree_matrix(member, X):
    """Per-tree contribution columns for X, plus the member's init.
    Mirrors the booster eval path: transform once, tree.predict per tree."""
    booster = member.model_
    Xb = np.ascontiguousarray(booster.prep_.transform(X).T)
    cols = np.empty((X.shape[0], len(booster.trees_)), dtype=np.float64)
    for t, tree in enumerate(booster.trees_):
        cols[:, t] = tree.predict(Xb)
    return cols, float(booster.init_)


def flatten_bag(bag, X):
    """(n, total_trees) contribution matrix, mean-init, and per-member spans.
    Asserts the flattened sums reproduce each member's predict."""
    blocks, inits, spans = [], [], []
    for m in bag.estimators_:
        cols, init = member_tree_matrix(m, X)
        ref = m.predict(X)
        np.testing.assert_allclose(init + cols.sum(axis=1), ref, rtol=1e-9,
                                   atol=1e-9)
        spans.append(cols.shape[1])
        blocks.append(cols)
        inits.append(init)
    C = np.hstack(blocks)
    return C, float(np.mean(inits)), spans


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - p) ** 2)))


def main():
    print(f"chimeraboost: {chimeraboost.__file__}")
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print("Warmup...")
    warmup()
    rows = []
    for key in PANEL:
        for seed in SEEDS:
            X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
            assert task == "regression"
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed)
            Xfit, Xrw, yfit, yrw = train_test_split(
                Xtr, ytr, test_size=0.20, random_state=seed)

            t0 = time.perf_counter()
            bag100 = ChimeraBoostRegressor(random_state=seed, n_ensembles=K)
            bag100.fit(Xtr, ytr, cat_features=cat)
            bag80 = ChimeraBoostRegressor(random_state=seed, n_ensembles=K)
            bag80.fit(Xfit, yfit, cat_features=cat)
            fit_s = time.perf_counter() - t0

            r100 = rmse(yte, bag100.predict(Xte))
            r80 = rmse(yte, bag80.predict(Xte))

            Crw, init_mean, spans = flatten_bag(bag80, Xrw)
            Cte, _, _ = flatten_bag(bag80, Xte)
            T = Crw.shape[1]
            lasso = LassoCV(positive=True, cv=3, n_alphas=30,
                            max_iter=20000, random_state=0)
            # K-mean-scaled columns: w=1 on every column reproduces the plain
            # bag mean, so weights read directly against uniform averaging.
            lasso.fit(Crw / K, np.asarray(yrw, float) - init_mean)
            w = lasso.coef_
            pred_te = init_mean + lasso.intercept_ + (Cte / K) @ w
            r_rw = rmse(yte, pred_te)
            nz = int(np.sum(w > 1e-12))

            beats100 = r_rw < r100
            rows.append(dict(dataset=key, seed=seed, rmse_bag100=r100,
                             rmse_bag80=r80, rmse_rw=r_rw, trees=T,
                             nonzero=nz, beats_bag100=bool(beats100),
                             fit_s=fit_s, alpha=float(lasso.alpha_)))
            print(f"{key} s{seed}: bag100 {r100:.5f}  bag80 {r80:.5f}  "
                  f"rw {r_rw:.5f}  {'BEATS' if beats100 else 'loses to'} "
                  f"bag100 | trees {T} -> {nz} nonzero "
                  f"({100 * (1 - nz / T):.0f}% pruned)", flush=True)

    wins = sum(r["beats_bag100"] for r in rows)
    print(f"\nrw vs bag100: {wins}/{len(rows)} (set,seed) wins  "
          f"[registered kill bar: majority]")
    d100 = np.mean([(r["rmse_bag100"] - r["rmse_rw"]) / r["rmse_bag100"]
                    for r in rows]) * 100
    d80 = np.mean([(r["rmse_bag80"] - r["rmse_rw"]) / r["rmse_bag80"]
                   for r in rows]) * 100
    prune = np.mean([1 - r["nonzero"] / r["trees"] for r in rows]) * 100
    print(f"mean RMSE delta vs bag100 (+ = rw better): {d100:+.2f}%")
    print(f"mean RMSE delta vs bag80  (+ = rw better): {d80:+.2f}%")
    print(f"mean pruning: {prune:.0f}% of trees zeroed")
    with open("benchmarks/results/b7_probe.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote benchmarks/results/b7_probe.json")


if __name__ == "__main__":
    main()
