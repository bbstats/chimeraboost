"""Probe: do group-centered numeric columns (x_num minus mean(x_num | cat))
close the high-cardinality categorical gap vs CatBoost?

MECHANISM HYPOTHESIS (pre-registered 2026-07-20, before any results):
Oblivious trees share one split across a whole level, so "is this row's
numeric value above ITS CATEGORY's baseline" needs a per-category staircase
the tree family is structurally worst at. The 2026-07-13 numeric cross
features fixed exactly this pattern for num x num (x_i < x_j as one split);
the num x cat analog is group-centering: a column x_i - mean_train(x_i | c_j)
makes the within-category deviation ONE split. It is target-free (no leakage
machinery needed, unlike TS crosses), unit-consistent (difference of
same-unit quantities), and cheap. CatBoost's remaining Brier/RMSE edge lives
on real high-card entity data (hc suite: 86-88% Brier winrate) where
per-entity baselines differ most -- exactly where group-centering has the
most to say.

PREDICTIONS:
  IMPROVE : hc regression with entity baselines -- employee_salaries (salary
            vs department norm), house_prices_nominal (price vs
            neighborhood), colleges, wine-reviews (price vs winery),
            black_friday, Moneyball; hc binary kick / porto-seguro
            (vehicle-model baselines); gr candidates nyc-taxi /
            delays_zurich / seattlecrime6 (station/route/area baselines).
  SMALL   : hc multiclass (okcupid-stem, Traffic_violations, cjs,
            eucalyptus) -- mechanism applies but numerics are few.
  FLAT    : low-card gr controls electricity / diamonds / house_sales.
  KILL if : controls regress broadly, or the hc gap sets regress (same
            signature that killed C1/C3).

PROTOCOL: paired same-split A/B, 3 seeds, default models, cats passed as
cat_features in BOTH arms. Pair selection uses only the baseline fit's
feature_importances_ (top-4 numeric x top-3 categorical, <= 12 columns);
group means computed on train rows only, unseen categories -> global train
mean. Deltas are % of baseline metric (positive = variant better).
Results: benchmarks/results/probe-catcross.jsonl (resumable); aggregate
table printed at the end.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-catcross.jsonl")
SEEDS = (0, 1, 2)
TOP_NUM = 4
TOP_CAT = 3

HC = [f"hc:{name}" for name in rb.HC_DATASETS]
GR_CANDIDATES = [
    "gr:reg_cat/nyc-taxi-green-dec-2016",
    "gr:reg_cat/delays_zurich_transport",
    "gr:reg_cat/seattlecrime6",
    "gr:reg_cat/Bike_Sharing_Demand",
]
GR_CONTROLS = [
    "gr:clf_cat/electricity",
    "gr:reg_cat/diamonds",
    "gr:reg_cat/house_sales",
]
ALL = HC + GR_CANDIDATES + GR_CONTROLS


def _group_center(col_tr, col_te, cat_tr, cat_te):
    """gdiff columns for one (numeric, categorical) pair: value minus the
    train-side per-category mean; unseen categories fall back to the global
    train mean; NaN numerics propagate."""
    s = pd.Series(col_tr)
    g = pd.Series(cat_tr, dtype=object)
    means = s.groupby(g).mean()
    global_mean = float(s.mean())
    m_tr = g.map(means).astype(float).fillna(global_mean).to_numpy()
    m_te = (pd.Series(cat_te, dtype=object).map(means).astype(float)
            .fillna(global_mean).to_numpy())
    return col_tr - m_tr, col_te - m_te


def _augment(Xtr, Xte, pairs):
    """Append the gdiff column for each (num_idx, cat_idx) pair at the END so
    cat indices stay valid."""
    add_tr, add_te = [], []
    for i, j in pairs:
        tr = Xtr[:, i].astype(np.float64)
        te = Xte[:, i].astype(np.float64)
        ctr = np.asarray(Xtr[:, j], dtype=object)
        cte = np.asarray(Xte[:, j], dtype=object)
        a, b = _group_center(tr, te, ctr, cte)
        add_tr.append(a)
        add_te.append(b)
    def stack(X, cols):
        block = np.column_stack(cols).astype(object)
        return np.concatenate([X, block], axis=1)
    return stack(Xtr, add_tr), stack(Xte, add_te)


def _fit_eval(task, Xtr, ytr, Xte, yte, cat, seed):
    """Fit a default model, return (metrics dict, fit_seconds)."""
    t0 = time.perf_counter()
    if task == "regression":
        m = ChimeraBoostRegressor(random_state=seed)
        m.fit(Xtr, ytr, cat_features=cat)
        secs = time.perf_counter() - t0
        rmse = float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))
        return {"rmse": rmse}, secs, m
    m = ChimeraBoostClassifier(random_state=seed)
    m.fit(Xtr, ytr, cat_features=cat)
    secs = time.perf_counter() - t0
    if task == "binary":
        p = m.predict_proba(Xte)[:, 1]
        return {"brier": float(brier_score_loss(yte, p)),
                "f1": float(f1_score(yte, m.predict(Xte)))}, secs, m
    p = m.predict_proba(Xte)
    return {"logloss": float(log_loss(yte, p, labels=m.classes_)),
            "f1": float(f1_score(yte, m.predict(Xte), average="macro"))}, secs, m


def _pairs_from_importances(imp, cat, n_features):
    """Top-TOP_NUM numeric x top-TOP_CAT categorical index pairs by baseline
    split-gain importance."""
    cat_set = set(cat)
    key = np.zeros(n_features)
    key[:len(imp)] = imp
    nums = sorted((i for i in range(n_features) if i not in cat_set),
                  key=lambda i: -key[i])[:TOP_NUM]
    cats = sorted((i for i in range(n_features) if i in cat_set),
                  key=lambda i: -key[i])[:TOP_CAT]
    return [(i, j) for i in nums for j in cats]


def _load(key):
    if key.startswith("hc:"):
        rb._add_highcard_datasets()
    return rb.DATASETS[key](1.0, np.random.default_rng(0))


def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"]))
    return done


def main():
    import chimeraboost
    print("module:", chimeraboost.__file__, flush=True)
    from chimeraboost import warmup
    warmup()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    done = _done_keys()
    for key in ALL:
        try:
            X, y, cat, task = _load(key)
        except Exception as e:
            print(f"[skip-load] {key}: {e}", flush=True)
            continue
        cat = cat or []
        n_num = X.shape[1] - len(cat)
        if not cat or n_num < 1:
            print(f"[skip-structure] {key}: cats={len(cat)} nums={n_num}",
                  flush=True)
            continue
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=strat)
            base, base_secs, model = _fit_eval(task, Xtr, ytr, Xte, yte,
                                               cat, seed)
            pairs = _pairs_from_importances(model.feature_importances_, cat,
                                            X.shape[1])
            Atr, Ate = _augment(Xtr, Xte, pairs)
            var, var_secs, _ = _fit_eval(task, Atr, ytr, Ate, yte, cat, seed)
            rec = {"dataset": key, "seed": seed, "task": task,
                   "n_pairs": len(pairs), "base": base, "variant": var,
                   "base_secs": round(base_secs, 3),
                   "var_secs": round(var_secs, 3)}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[done] {key} s{seed} base={base} var={var}", flush=True)
    _table()


def _table():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print("\n=== probe-catcross aggregate (positive % = gdiff better) ===")
    print(f"{'dataset':<42}{'task':<12}{'metric':<9}{'delta%':>8}{'fitx':>6}")
    wins = losses = 0
    for ds, rs in by_ds.items():
        task = rs[0]["task"]
        primary = {"regression": "rmse", "binary": "brier",
                   "multiclass": "logloss"}[task]
        deltas = [100.0 * (r["base"][primary] - r["variant"][primary])
                  / max(abs(r["base"][primary]), 1e-9) for r in rs]
        fitx = float(np.mean([r["var_secs"] / max(r["base_secs"], 1e-9)
                              for r in rs]))
        d = float(np.mean(deltas))
        wins += d > 0.05
        losses += d < -0.05
        print(f"{ds:<42}{task:<12}{primary:<9}{d:>+8.3f}{fitx:>6.2f}")
    print(f"\ndatasets better: {wins}  worse: {losses} "
          f"(threshold 0.05%), seeds per set: {len(SEEDS)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--table-only":
        _table()
    else:
        main()
