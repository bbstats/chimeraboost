"""Probe R4 (CAMPAIGN_PLAN I041): let linear leaves use the target-encoded
columns as linear terms. Zero library change.

`_build_centers_std` zeroes every column whose `is_numeric_binned_` is False,
so `_fit_linear_leaf_tail` can never pick a target-statistic column as a
ridge term: on sf-police (5 of 6 columns categorical) the linear-leaf model
has one usable column. This probe flips the TS (and combo) block to
linear-eligible after the preprocessor fits -- their binned centres are
target-statistic values, ordinal in the target, so a linear term over them
is a well-posed thing to fit -- and asks what the leaves do with them.

Arms, on the DEFAULT estimator, all 14 high-card sets, paired splits, 3
seeds:

  default     the library as is
  ts_linear   `is_numeric_binned_` set True on the TS and combo columns after
              `fit_transform` / `from_base_with_cross` (monkeypatch), so the
              per-leaf ridge may use them. Everything else identical.

Where the change can engage: binary classification (linear leaves are the
default there) and regression (the constant-vs-linear validation race).
Multiclass never fits linear leaves, so the four multiclass sets are the
in-run inert control and must read as exact ties. RMSE for regression,
Brier (K-sum form) for classification. Resumable JSONL; table at the end.

Run:
    python benchmarks/probe_ts_linear_terms.py
    python benchmarks/probe_ts_linear_terms.py --table-only
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.preprocessing import FeaturePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-ts-linear-terms.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50

BINARY = ["hc:sf-police-incidents", "hc:kick", "hc:porto-seguro",
          "hc:kdd_ipums_la_97-small"]
REGRESSION = ["hc:wine-reviews", "hc:colleges", "hc:house_prices_nominal",
              "hc:black_friday", "hc:employee_salaries", "hc:Moneyball"]
MULTICLASS = ["hc:okcupid-stem", "hc:Traffic_violations", "hc:cjs", "hc:eucalyptus"]
ALL = BINARY + REGRESSION + MULTICLASS
ARMS = ["default", "ts_linear"]

_ORIG_FT = FeaturePreprocessor.fit_transform
_ORIG_FB = FeaturePreprocessor.from_base_with_cross.__func__


def _mark_ts_linear(prep):
    nb = getattr(prep, "n_numeric_block_", len(prep.num_features_))
    start = nb + len(prep.cross_pairs)
    prep.is_numeric_binned_[start:] = True


def _ft_patched(self, *a, **k):
    out = _ORIG_FT(self, *a, **k)
    _mark_ts_linear(self)
    return out


def _fb_patched(cls, *a, **k):
    prep, cross_binner, crossb = _ORIG_FB(cls, *a, **k)
    _mark_ts_linear(prep)
    return prep, cross_binner, crossb


def _score(task, yte, m, Xte):
    if task == "regression":
        return float(np.sqrt(np.mean((yte - m.predict(Xte)) ** 2)))
    proba = m.predict_proba(Xte)
    onehot = (np.asarray(yte)[:, None] == np.asarray(m.classes_)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _run(task, Xtr, ytr, Xte, yte, cat, seed, ts_linear):
    if ts_linear:
        FeaturePreprocessor.fit_transform = _ft_patched
        FeaturePreprocessor.from_base_with_cross = classmethod(_fb_patched)
    try:
        Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
        t = time.time()
        m = Est(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE, random_state=seed)
        m.fit(Xtr, ytr, cat_features=cat)
        fit_s = time.time() - t
        return _score(task, yte, m, Xte), fit_s
    finally:
        FeaturePreprocessor.fit_transform = _ORIG_FT
        FeaturePreprocessor.from_base_with_cross = classmethod(_ORIG_FB)


def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["dataset"], r["arm"], r["seed"]))
    return done


def _append(rec):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    done = _done_keys()
    for key in ALL:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        print(f"\n=== {key}  task={task}  n={len(y)}  p={X.shape[1]}  cats={len(cat)}",
              flush=True)
        for seed in SEEDS:
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=strat)
            for arm in ARMS:
                if (key, arm, seed) in done:
                    continue
                s, fit_s = _run(task, Xtr, ytr, Xte, yte, cat, seed, arm == "ts_linear")
                _append({"dataset": key, "arm": arm, "seed": seed, "task": task,
                         "score": s, "fit_time": fit_s})
                print(f"  [{arm:9s} s{seed}] score={s:.6f}  {fit_s:6.1f}s", flush=True)
    table()


def table():
    agg = defaultdict(list)
    ft = defaultdict(list)
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                agg[(r["dataset"], r["arm"])].append(r["score"])
                ft[(r["dataset"], r["arm"])].append(r["fit_time"])
    print("\n" + "=" * 96)
    print("TS COLUMNS AS LINEAR-LEAF TERMS -- RMSE (reg) / Brier (clf), lower better; "
          "d%: change vs the default, + = better; fit: ratio")
    print("=" * 96)
    print(f"{'dataset':26s}{'default':>14s}{'ts_linear':>14s}{'d%':>10s}{'fit x':>8s}")
    summary = defaultdict(list)
    for group, name in ((BINARY, "BINARY (linear leaves on by default)"),
                        (REGRESSION, "REGRESSION (const-vs-linear race)"),
                        (MULTICLASS, "MULTICLASS (no linear leaves: control)")):
        print(f"\n-- {name} " + "-" * max(1, 92 - len(name)))
        for ds in group:
            a = agg.get((ds, "default"))
            b = agg.get((ds, "ts_linear"))
            if not a or not b:
                continue
            ma, mb = float(np.mean(a)), float(np.mean(b))
            d = 100 * (ma - mb) / ma if ma else 0.0
            ratio = float(np.mean(ft[(ds, "ts_linear")]) / np.mean(ft[(ds, "default")]))
            tie = abs(ma - mb) < 1e-12
            summary[name].append((d, tie, ratio))
            print(f"{ds[3:][:26]:26s}{ma:14.6f}{mb:14.6f}{d:+9.3f}%{ratio:8.2f}"
                  + ("   exact tie" if tie else ""))
    print()
    for name, vals in summary.items():
        d = np.array([x for x, _, _ in vals])
        ties = sum(t for _, t, _ in vals)
        wins = int((d > 1e-9).sum())
        losses = int((d < -1e-9).sum())
        print(f"  {name:40s}: {wins}W-{losses}L-{ties}T of {len(d)}, median {np.median(d):+.3f}%, "
              f"fit ratio median {np.median([r for _, _, r in vals]):.2f}")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
