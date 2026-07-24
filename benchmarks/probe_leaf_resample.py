"""LRE probe: structure-shared leaf-resampled ensembling (DL-takeaway B).

Zero library change. Question: how much of the bag's accuracy lift can an
ensemble capture when members SHARE the fitted tree structures and differ only
by re-estimating every leaf value on a resampled row subset? If a large share
survives, ensemble-grade strength becomes available at a fraction of the bag's
fit cost (members skip prep, selection, histograms and split search entirely).

Per (dataset, seed): 75/25 train/test, then 80/20 fit/val inside train.
Arms (all see the same fit+val data):
  single  default estimator, explicit eval_set=(val)
  bag8    n_ensembles=8 (blessed member auto-params), explicit eval_set
  lre8    8 members re-estimated on 80% row subsamples of the fit rows with
          the single model's structures; reg averages raw scores, clf averages
          sigmoid(raw / single's temperature)
Identity guard: a full-row re-estimation must reproduce the single model's
test predictions (proves the recurrence mirrors the fit loop exactly).

Capture = (single - lre8) / (single - bag8) on the primary metric
(RMSE reg / Brier clf), per dataset over seed-mean metrics.

Run from benchmarks/:  python probe_leaf_resample.py
"""
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.tree import _leaf_values, _linear_leaf_fit, _linear_predict
from chimeraboost.warmup import warmup
import run_benchmarks as rb

REG = [
    "gr:reg_num/cpu_act",
    "gr:reg_num/pol",
    "gr:reg_num/wine_quality",
    "gr:reg_num/elevators",
    "gr:reg_cat/house_sales",
]
BIN = [
    "gr:clf_num/electricity",
    "gr:clf_num/covertype",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/heloc",
    "gr:clf_cat/road-safety",
]
SEEDS = (0, 1, 2)
K = 8
FRAC = 0.8


def lre_raw_scores(booster, Xb, y, member_rows, Xb_te):
    """Re-run the leaf-value recurrence for each member row set with the
    booster's fixed tree structures. Returns a list of raw test-score arrays.

    Mirrors the plain fit path exactly: first Newton step from the current
    gradients, `leaf_estimation_iterations - 1` refinement steps on constant
    trees, hessian-weighted ridge refit on linear-leaf trees. Ordered boosting,
    leaf-adjusting losses and gradient subsampling are asserted off (defaults).
    """
    assert not booster.ordered_boosting
    assert not getattr(booster.loss_, "adjusts_leaves", False)
    loss, l2, lr = booster.loss_, booster.l2_leaf_reg, booster.lr_
    lei = int(booster.leaf_estimation_iterations or 1)
    n_te = Xb_te.shape[1]
    # Per-member contiguous column subsets (linear-leaf refits index Xb by row).
    Xb_m = [np.ascontiguousarray(Xb[:, rows]) for rows in member_rows]
    y_m = [y[rows] for rows in member_rows]
    F = [np.full(rows.size, booster.init_) for rows in member_rows]
    raw_te = [np.full(n_te, float(booster.init_)) for _ in member_rows]
    for tree in booster.trees_:
        leaf_fit = tree.apply(Xb)
        leaf_te = tree.apply(Xb_te)
        n_lv = tree.values.shape[0]
        for k, rows in enumerate(member_rows):
            lf = np.ascontiguousarray(leaf_fit[rows])
            g, h = loss.grad_hess(y_m[k], F[k])
            if tree.lin_coef is not None:
                coef = _linear_leaf_fit(lf, g, h, n_lv, tree.lin_feats,
                                        tree.centers_std, Xb_m[k], l2,
                                        booster.linear_lambda, lr)
                F[k] += _linear_predict(lf, tree.lin_feats, coef,
                                        tree.centers_std, Xb_m[k])
                raw_te[k] += _linear_predict(leaf_te, tree.lin_feats, coef,
                                             tree.centers_std, Xb_te)
            else:
                vals = _leaf_values(lf, g, h, n_lv, l2, lr)
                for _ in range(lei - 1):
                    g2, h2 = loss.grad_hess(y_m[k], F[k] + vals[lf])
                    vals = vals + _leaf_values(lf, g2, h2, n_lv, l2, lr)
                F[k] += vals[lf]
                raw_te[k] += vals[leaf_te]
    return raw_te


def binned(booster, X):
    return np.ascontiguousarray(booster.prep_.transform(X).T)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - p) ** 2)))


def brier(y01, p):
    return float(np.mean((p - y01) ** 2))


def run_one(key, seed, is_reg):
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.20,
                                              random_state=seed)
    Est = ChimeraBoostRegressor if is_reg else ChimeraBoostClassifier

    t0 = time.perf_counter()
    single = Est(random_state=seed)
    single.fit(Xfit, yfit, cat_features=cat, eval_set=(Xval, yval))
    t_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    bag = Est(random_state=seed, n_ensembles=K)
    bag.fit(Xfit, yfit, cat_features=cat, eval_set=(Xval, yval))
    t_bag = time.perf_counter() - t0

    booster = single.model_
    Xb = binned(booster, Xfit)
    Xb_te = binned(booster, Xte)
    if is_reg:
        y_enc = np.asarray(yfit, dtype=np.float64)
        y_te = np.asarray(yte, dtype=np.float64)
    else:
        y_enc = (np.asarray(yfit) == single.classes_[1]).astype(np.float64)
        y_te = (np.asarray(yte) == single.classes_[1]).astype(np.float64)

    # Identity guard: full-row re-estimation reproduces the fitted model.
    n = Xb.shape[1]
    full = lre_raw_scores(booster, Xb, y_enc, [np.arange(n)], Xb_te)[0]
    ref = booster.predict_raw(Xte)
    np.testing.assert_allclose(full, ref, rtol=1e-7, atol=1e-9)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 1000)
    rows = [np.sort(rng.choice(n, int(FRAC * n), replace=False))
            for _ in range(K)]
    raws = lre_raw_scores(booster, Xb, y_enc, rows, Xb_te)
    t_lre = time.perf_counter() - t0

    if is_reg:
        m_single = rmse(y_te, single.predict(Xte))
        m_bag = rmse(y_te, bag.predict(Xte))
        m_lre = rmse(y_te, np.mean(raws, axis=0))
    else:
        T = single.temperature_
        probs = [1.0 / (1.0 + np.exp(-r / T)) for r in raws]
        m_single = brier(y_te, single.predict_proba(Xte)[:, 1])
        m_bag = brier(y_te, bag.predict_proba(Xte)[:, 1])
        m_lre = brier(y_te, np.mean(probs, axis=0))

    return dict(dataset=key, seed=seed, task="reg" if is_reg else "bin",
                metric_single=m_single, metric_bag=m_bag, metric_lre=m_lre,
                trees=len(booster.trees_), t_single=t_single, t_bag=t_bag,
                t_lre=t_lre)


def main():
    import sys
    print(f"chimeraboost: {chimeraboost.__file__}", flush=True)
    rb._add_grinsztajn_datasets()
    print("Warmup...", flush=True)
    warmup()
    panel = [k for k in REG + BIN
             if not sys.argv[1:] or any(a in k for a in sys.argv[1:])]
    rows = []
    for key in panel:
        is_reg = key in REG
        for seed in SEEDS:
            r = run_one(key, seed, is_reg)
            rows.append(r)
            print(f"{key} s{seed}: single {r['metric_single']:.5f}  "
                  f"bag8 {r['metric_bag']:.5f}  lre8 {r['metric_lre']:.5f}  "
                  f"({r['trees']} trees, lre {r['t_lre']:.1f}s vs "
                  f"fit {r['t_single']:.1f}s / bag {r['t_bag']:.1f}s)",
                  flush=True)
        with open("results/lre_probe.json", "w") as f:
            json.dump(rows, f, indent=1)

    # Aggregate: per-dataset seed means, then capture fraction.
    print("\n=== LRE probe aggregate (per-dataset means over seeds) ===")
    print(f"{'dataset':34s} {'metric':6s} {'single':>9s} {'bag8':>9s} "
          f"{'lre8':>9s} {'bagWin%':>8s} {'lreWin%':>8s} {'capture':>8s}")
    caps = []
    for key in panel:
        sub = [r for r in rows if r["dataset"] == key]
        ms = float(np.mean([r["metric_single"] for r in sub]))
        mb = float(np.mean([r["metric_bag"] for r in sub]))
        ml = float(np.mean([r["metric_lre"] for r in sub]))
        d_bag = (ms - mb) / ms * 100
        d_lre = (ms - ml) / ms * 100
        cap = d_lre / d_bag if d_bag > 1e-12 else float("nan")
        if np.isfinite(cap):
            caps.append(cap)
        met = "rmse" if sub[0]["task"] == "reg" else "brier"
        print(f"{key:34s} {met:6s} {ms:9.5f} {mb:9.5f} {ml:9.5f} "
              f"{d_bag:+8.2f} {d_lre:+8.2f} {cap:8.2f}")
    print(f"\nmean capture fraction (lre share of bag lift): "
          f"{np.mean(caps):.2f}  (n={len(caps)} datasets)")
    print("wrote results/lre_probe.json")


if __name__ == "__main__":
    main()
