"""Full-data refit probe: reclaim the 20% early-stopping data tax.

Zero library change, public API only. The shipped single-model fit carves a
20% validation split for early stopping / calibration and the final model never
trains on those rows. The bagging program's design law ("more effective data
per member beats sampling diversity" — B-samp, B2a) suggests the single-model
analog: once ES has picked the tree count, REFIT on 100% of the training data
at that count and learning rate, transferring the calibration temperature.

Arms per (dataset, seed), config-pinned so both arms are the same model class
(cross_features=False; regression pins linear_leaves=False — both need a val
split to self-select, which arm B doesn't have):
  base    shipped fit on Xtr (auto 80/20 ES split inside)
  refit   early_stopping=False, n_estimators = base's best iteration,
          learning_rate = base's resolved lr, fit on ALL of Xtr;
          clf Brier scored at base's temperature (logit/T transfer)
  refitX  same but rounds scaled by 1/0.8 (T* transfer sensitivity check)

Run from benchmarks/:  python probe_fulldata_refit.py [key-substring ...]
"""
import json
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
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


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - p) ** 2)))


def brier_at_temperature(model, X, y01, T):
    """Brier of `model`'s probabilities rescaled to temperature T. The model
    was fit without a val split, so its own temperature_ is 1.0 and its
    predict_proba logits ARE the raw scores."""
    p1 = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(p1 / (1.0 - p1)) * model.temperature_   # back to raw
    p = 1.0 / (1.0 + np.exp(-logit / T))
    return float(np.mean((p - y01) ** 2))


def run_one(key, seed, is_reg):
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    pins = dict(cross_features=False)
    if is_reg:
        pins["linear_leaves"] = False
    Est = ChimeraBoostRegressor if is_reg else ChimeraBoostClassifier

    t0 = time.perf_counter()
    base = Est(random_state=seed, **pins)
    base.fit(Xtr, ytr, cat_features=cat)
    t_base = time.perf_counter() - t0

    booster = base.model_
    t_star = len(booster.trees_)
    lr = float(booster.lr_)

    def refit(n_rounds):
        m = Est(random_state=seed, early_stopping=False,
                n_estimators=int(n_rounds), learning_rate=lr, **pins)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr, cat_features=cat)
        return m, time.perf_counter() - t0

    re1, t_re1 = refit(t_star)
    rex, t_rex = refit(int(np.ceil(t_star / 0.8)))

    if is_reg:
        y_te = np.asarray(yte, dtype=np.float64)
        m_base = rmse(y_te, base.predict(Xte))
        m_re1 = rmse(y_te, re1.predict(Xte))
        m_rex = rmse(y_te, rex.predict(Xte))
    else:
        y01 = (np.asarray(yte) == base.classes_[1]).astype(np.float64)
        T = float(base.temperature_)
        # base's predict_proba already applies its temperature; score directly.
        p_base = base.predict_proba(Xte)[:, 1]
        m_base = float(np.mean((p_base - y01) ** 2))
        m_re1 = brier_at_temperature(re1, Xte, y01, T)
        m_rex = brier_at_temperature(rex, Xte, y01, T)

    return dict(dataset=key, seed=seed, task="reg" if is_reg else "bin",
                metric_base=m_base, metric_refit=m_re1, metric_refitx=m_rex,
                t_star=t_star, lr=lr, t_base=t_base, t_refit=t_re1,
                t_refitx=t_rex)


def main():
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
            print(f"{key} s{seed}: base {r['metric_base']:.5f}  "
                  f"refit {r['metric_refit']:.5f}  refitX {r['metric_refitx']:.5f}"
                  f"  (T*={r['t_star']}, base {r['t_base']:.1f}s "
                  f"refit {r['t_refit']:.1f}s)", flush=True)
        with open("results/fulldata_refit_probe.json", "w") as f:
            json.dump(rows, f, indent=1)

    print("\n=== full-data refit aggregate (per-dataset means over seeds) ===")
    print(f"{'dataset':34s} {'metric':6s} {'base':>9s} {'refit':>9s} "
          f"{'refitX':>9s} {'refit%':>8s} {'refitX%':>8s}")
    d1s, dxs = [], []
    for key in panel:
        sub = [r for r in rows if r["dataset"] == key]
        mb = float(np.mean([r["metric_base"] for r in sub]))
        m1 = float(np.mean([r["metric_refit"] for r in sub]))
        mx = float(np.mean([r["metric_refitx"] for r in sub]))
        d1 = (mb - m1) / mb * 100
        dx = (mb - mx) / mb * 100
        d1s.append(d1)
        dxs.append(dx)
        met = "rmse" if sub[0]["task"] == "reg" else "brier"
        print(f"{key:34s} {met:6s} {mb:9.5f} {m1:9.5f} {mx:9.5f} "
              f"{d1:+8.2f} {dx:+8.2f}")
    w1 = sum(d > 0 for d in d1s)
    wx = sum(d > 0 for d in dxs)
    print(f"\nrefit : {w1}/{len(d1s)} datasets improved, mean {np.mean(d1s):+.2f}%")
    print(f"refitX: {wx}/{len(dxs)} datasets improved, mean {np.mean(dxs):+.2f}%")
    print("wrote results/fulldata_refit_probe.json")


if __name__ == "__main__":
    main()
