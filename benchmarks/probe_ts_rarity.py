"""Probe R3 Stage D (CAMPAIGN_PLAN I035): the two things CatBoost's CTR still
has that our ordered target statistic does not -- a rarity signal and coarse
quantization -- tried on OUR side, zero library change.

Stages A-C (`probe_catboost_hc_ablation.py`, `probe_ts_prior_target.py`)
established that CatBoost's high-cardinality Brier edge is its encoder, not
its booster, and not the shrinkage target. What its CTR still carries:

  * three prior columns (sum + p) / (n + 1) for p in {0, 0.5, 1}, whose spread
    is exactly 1 / (n + 1): a rarity signal the trees can split on, worth 34%
    of the sf-police edge in Stage A;
  * the CTR value quantized at 15 uniform borders over [0, 1], worth 49-64% of
    the edge on kick and porto-seguro in Stage B.

Arms, on our DEFAULT estimator, same seven sets and paired splits as I033
(its `chimera` and `cb_default` rows are the reference lines):

  rarity    every categorical column gets a companion numeric column holding
            that category's row count in the training rows (test rows look the
            count up; unseen = 0). Trees are invariant to monotone maps, so
            count, log count and 1/(n+1) are the same feature after binning.
  ts_q16    the ordered target statistic quantized to 16 uniform buckets over
            [0, 1] (bucket centres), at fit and at transform -- CatBoost's
            CtrBorderCount=15, Uniform. Classification only: the TS is a
            probability.
  both      rarity + ts_q16.

`ts_q16` is a monkeypatch of `OrderedTargetEncoder.fit_transform` /
`transform` that post-processes their output. Brier in the harness's K-sum
form. Resumable JSONL; table at the end.

Run:
    python benchmarks/probe_ts_rarity.py
    python benchmarks/probe_ts_rarity.py --table-only
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

from chimeraboost import ChimeraBoostClassifier
from chimeraboost.preprocessing import as_model_array
from chimeraboost.target_encoding import OrderedTargetEncoder, factorize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-ts-rarity.jsonl")
REFERENCE = os.path.join(HERE, "results", "probe-cb-hc-ablation.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50
Q = 16

GAP = ["hc:sf-police-incidents", "hc:Traffic_violations", "hc:kick",
       "hc:okcupid-stem", "hc:porto-seguro"]
CONTROL = ["hc:kdd_ipums_la_97-small", "hc:eucalyptus"]
ALL = GAP + CONTROL
ARMS = ["rarity", "ts_q16", "both"]
REF_ARMS = ["chimera", "cb_default"]


def _brier(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# ---- rarity columns -----------------------------------------------------------
def _with_rarity(Xtr, Xte, cat):
    """Append one count column per categorical: the category's frequency in
    the training rows (test rows look it up, unseen -> 0)."""
    A = as_model_array(Xtr, True)
    T = as_model_array(Xte, True)
    cols_a, cols_t = [], []
    for f in cat:
        codes, cats = factorize(A[:, f])
        counts = np.bincount(codes, minlength=len(cats)).astype(np.float64)
        lookup = {v: i for i, v in enumerate(cats)}
        cols_a.append(counts[codes])
        idx = np.array([lookup.get(v, -1) for v in T[:, f]])
        ct = np.where(idx >= 0, counts[np.maximum(idx, 0)], 0.0)
        cols_t.append(ct)
    A2 = np.column_stack([A] + [c[:, None] for c in cols_a]).astype(object)
    T2 = np.column_stack([T] + [c[:, None] for c in cols_t]).astype(object)
    return A2, T2


# ---- TS quantization monkeypatch ---------------------------------------------
_ORIG_FT = OrderedTargetEncoder.fit_transform
_ORIG_T = OrderedTargetEncoder.transform


def _quantize(out):
    q = np.clip(np.floor(out * Q), 0, Q - 1)
    return (q + 0.5) / Q


def _ft_q(self, codes_matrix, y, sample_weight=None):
    return _quantize(_ORIG_FT(self, codes_matrix, y, sample_weight))


def _t_q(self, codes_matrix):
    return _quantize(_ORIG_T(self, codes_matrix))


def _run(Xtr, ytr, Xte, yte, cat, seed, rarity, quantize):
    if rarity:
        Xtr, Xte = _with_rarity(Xtr, Xte, cat)
    if quantize:
        OrderedTargetEncoder.fit_transform = _ft_q
        OrderedTargetEncoder.transform = _t_q
    try:
        t = time.time()
        m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                                   early_stopping_rounds=PATIENCE, random_state=seed)
        m.fit(Xtr, ytr, cat_features=cat)
        fit_s = time.time() - t
        return _brier(yte, m.predict_proba(Xte), m.classes_), fit_s
    finally:
        OrderedTargetEncoder.fit_transform = _ORIG_FT
        OrderedTargetEncoder.transform = _ORIG_T


ARM_FLAGS = {"rarity": (True, False), "ts_q16": (False, True), "both": (True, True)}


# ---- bookkeeping --------------------------------------------------------------
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
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            for arm, (rar, quant) in ARM_FLAGS.items():
                if (key, arm, seed) in done:
                    continue
                b, fit_s = _run(Xtr, ytr, Xte, yte, cat, seed, rar, quant)
                _append({"dataset": key, "arm": arm, "seed": seed, "task": task,
                         "brier": b, "fit_time": fit_s})
                print(f"  [{arm:8s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s", flush=True)
    table()


def table():
    agg = defaultdict(list)
    for path in (REFERENCE, RESULTS):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["arm"] in REF_ARMS + ARMS:
                        agg[(r["dataset"], r["arm"])].append(r["brier"])
    order = REF_ARMS + ARMS
    print("\n" + "=" * 112)
    print("RARITY + TS QUANTIZATION ON OUR SIDE -- Brier (lower better). d%: change vs "
          "our default, + = better. closed: share of CatBoost's edge the arm closes.")
    print("=" * 112)
    print(f"{'dataset':26s}" + "".join(f"{a:>14s}" for a in order)
          + "".join(f"     d%@{a:>6s}" for a in ARMS))
    summary = defaultdict(list)
    for group, name in ((GAP, "GAP SETS"), (CONTROL, "CONTROLS")):
        print(f"\n-- {name} " + "-" * (108 - len(name)))
        for ds in group:
            v = {a: (float(np.mean(agg[(ds, a)])) if agg.get((ds, a)) else None) for a in order}
            if v["chimera"] is None:
                continue
            print(_row_line(ds, v, order, name, summary))
    print()
    _print_summary(summary)


def _row_line(ds, v, order, name, summary):
    ours, cb = v["chimera"], v["cb_default"]
    line = f"{ds[3:][:26]:26s}" + "".join(
        f"{v[a]:14.6f}" if v[a] is not None else f"{'-':>14s}" for a in order)
    for a in ARMS:
        if v[a] is None:
            line += f"{'-':>14s}"
            continue
        d = 100 * (ours - v[a]) / ours
        closed = ((ours - v[a]) / (ours - cb) if cb is not None
                  and abs(ours - cb) > 1e-12 else float("nan"))
        summary[(name, a)].append((d, closed))
        line += f"{d:+13.3f}%"
    return line


def _print_summary(summary):
    for (name, a), vals in sorted(summary.items()):
        d = np.array([x for x, _ in vals])
        closed = np.array([c for _, c in vals])
        wins = int((d > 1e-9).sum())
        losses = int((d < -1e-9).sum())
        print(f"  {name:9s} {a:8s}: {wins}W-{losses}L of {len(d)}, median {np.median(d):+.3f}%  "
              f"(per set {', '.join(f'{x:+.2f}' for x in d)});  edge closed median "
              f"{100 * np.nanmedian(closed):+.0f}%")
    print("\n(reference rows chimera / cb_default from probe-cb-hc-ablation.jsonl, same "
          "splits and seeds)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
