"""Probe: do numeric cross features (pairwise differences / products) close the
concentrated regression CatBoost gap?

MECHANISM HYPOTHESIS (pre-registered 2026-07-13, before any results):
Oblivious trees approximate numeric interactions with a depth-limited staircase.
CatBoost (also oblivious) grinds 1400-2000 trees on exactly our top gap sets
while we early-stop at 77-578, and the lr=0.03 probe proved the gap is NOT step
size -- consistent with CatBoost slowly building interaction staircases we never
reach. A difference feature makes the x_i < x_j boundary ONE split; a product
captures multiplicative structure. (Modern reference: OpenFE, ICML'23 --
automated pairwise arithmetic features screened on residual gain.)

PREDICTIONS:
  IMPROVE : nyc-taxi x2 (lat/lon pairs -> displacement IS the signal),
            Brazilian_houses x2, cpu_act, sulfur (the CatBoost-gap cluster),
            pol (interaction-heavy).
  FLAT    : controls elevators/houses/wine_quality/abalone. KILL if these
            regress broadly (same signature as the C1/C3/G4 kills).
  BINARY  : electricity/covertype/MagicTelescope Brier improve-or-flat.

PROTOCOL: paired same-split A/B, 3 seeds, harness-default models. Pair selection
uses only the baseline fit's feature_importances_ (top-6 numeric features, all
15 pairs) -- no target peeking beyond what the paired baseline already saw.
Variants: diff only / prod only / both. Deltas are % of baseline metric
(positive = variant better). Results: benchmarks/results/probe-crossfeat.jsonl
(resumable); aggregate table printed at the end.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-crossfeat.jsonl")
SEEDS = (0, 1, 2)
TOP_M = 6          # cross the top-M numeric features by baseline importance
THREADS = None     # all cores (run this probe with nothing else heavy going)

GAP = [
    "gr:reg_num/nyc-taxi-green-dec-2016",
    "gr:reg_cat/nyc-taxi-green-dec-2016",
    "gr:reg_num/Brazilian_houses",
    "gr:reg_cat/Brazilian_houses",
    "gr:reg_num/cpu_act",
    "gr:reg_num/sulfur",
    "gr:reg_num/pol",
]
CONTROL = [
    "gr:reg_num/elevators",
    "gr:reg_num/houses",
    "gr:reg_num/wine_quality",
    "gr:reg_num/abalone",
]
BINARY = [
    "gr:clf_num/electricity",
    "gr:clf_num/covertype",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/heloc",
]
ALL = GAP + CONTROL + BINARY


def _numeric_matrix(X, cols):
    return np.column_stack([X[:, i].astype(np.float64) for i in cols])


def _augment(Xtr, Xte, pairs, kind):
    """Append cross columns for the given (i, j) index pairs. kind in
    {diff, prod, both}. Appended at the END so cat indices stay valid."""
    def blocks(X):
        out = []
        for i, j in pairs:
            a = X[:, i].astype(np.float64)
            b = X[:, j].astype(np.float64)
            if kind in ("diff", "both"):
                out.append(a - b)
            if kind in ("prod", "both"):
                out.append(a * b)
        return np.column_stack(out)

    def cat(X, blk):
        if X.dtype == object:
            return np.concatenate([X, blk.astype(object)], axis=1)
        return np.concatenate([X.astype(np.float64), blk], axis=1)

    return cat(Xtr, blocks(Xtr)), cat(Xte, blocks(Xte))


def _fit_eval(task, Xtr, ytr, Xte, yte, cat):
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    t = time.time()
    m = Est(n_estimators=2000, early_stopping_rounds=50, random_state=0,
            thread_count=THREADS)
    m.fit(Xtr, ytr, cat_features=cat)
    fit_s = time.time() - t
    if task == "regression":
        pred = m.predict(Xte)
        metric = float(np.sqrt(np.mean((yte - pred) ** 2)))
    else:
        metric = float(brier_score_loss(yte, m.predict_proba(Xte)[:, 1]))
    return metric, fit_s, int(m.best_iteration_ or 0), m


def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"]))
    return done


def main():
    done = _done_keys()
    for key in ALL:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=strat)

            base, base_s, base_it, model = _fit_eval(task, Xtr, ytr, Xte, yte, cat)

            imp = np.asarray(model.feature_importances_, dtype=float)
            num_idx = [i for i in range(X.shape[1]) if i not in set(cat)]
            top = sorted(num_idx, key=lambda i: -imp[i])[:TOP_M]
            pairs = [(top[a], top[b]) for a in range(len(top))
                     for b in range(a + 1, len(top))]

            row = {"dataset": key, "seed": seed, "task": task,
                   "group": ("gap" if key in GAP else
                             "control" if key in CONTROL else "binary"),
                   "n_train": int(len(ytr)), "n_pairs": len(pairs),
                   "base": base, "base_s": round(base_s, 2), "base_iter": base_it}
            for kind in ("diff", "prod", "both"):
                Xtr2, Xte2 = _augment(Xtr, Xte, pairs, kind)
                v, v_s, v_it, _ = _fit_eval(task, Xtr2, ytr, Xte2, yte, cat)
                row[kind] = v
                row[f"{kind}_s"] = round(v_s, 2)
                row[f"{kind}_iter"] = v_it
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{key} s{seed}: base={base:.5g} "
                  + " ".join(f"{k}={row[k]:.5g}({100*(base-row[k])/base:+.2f}%)"
                             for k in ("diff", "prod", "both")), flush=True)
    table()


def table():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print(f"\n{'dataset':42} {'grp':7} {'diff%':>8} {'prod%':>8} {'both%':>8} "
          f"{'fitx':>5}")
    agg = {"gap": [], "control": [], "binary": []}
    for ds, rs in by_ds.items():
        d = {k: float(np.mean([100 * (r["base"] - r[k]) / r["base"] for r in rs]))
             for k in ("diff", "prod", "both")}
        fitx = float(np.mean([r["both_s"] / max(r["base_s"], 1e-9) for r in rs]))
        grp = rs[0]["group"]
        agg[grp].append(d)
        print(f"{ds:42} {grp:7} {d['diff']:+8.2f} {d['prod']:+8.2f} "
              f"{d['both']:+8.2f} {fitx:5.2f}")
    print()
    for grp, ds in agg.items():
        if ds:
            for k in ("diff", "prod", "both"):
                m = np.mean([d[k] for d in ds])
                w = sum(d[k] > 0.15 for d in ds)
                l = sum(d[k] < -0.15 for d in ds)
                print(f"  {grp:8} {k:5}: mean {m:+.2f}%  {w}W/{l}L/"
                      f"{len(ds)-w-l}T")
    print("\n(positive = cross features better; metric RMSE for reg, "
          "Brier for binary)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
