"""Probe R3 (CAMPAIGN_PLAN I033): WHICH CatBoost categorical mechanism carries
its high-cardinality Brier edge?

CatBoost beats the default on the high-card classification sets, and the gap
is monotone in max cardinality (facts ledger 2026-09-21): sf-police −1.17%,
Traffic_violations −1.92%, kick −1.16%, okcupid-stem −0.68%, porto-seguro
−0.14% of our Brier; we win kdd_ipums (+5.7%) and eucalyptus (+0.9%). Seven
partial ports of CatBoost machinery each regressed somewhere (barrier B3), so
before another one is built, ablate the OPPONENT: run CatBoost at the harness
defaults, then with each categorical mechanism switched off, and read what
share of its edge each removal gives back. Same method as
`probe_catboost_ablation.py` (2026-08-01), which found the learning rate.

What CatBoost 1.2.10 actually runs here (read from `get_all_params()` on a
two-categorical toy, binary and multiclass alike): `max_ctr_complexity=1`, so
NO cat×cat CTR combinations; `one_hot_max_size=2`; two simple CTRs per
categorical column -- `Borders` (the target statistic, quantized at 15
borders, with THREE priors 0/0.5/1 so three columns) and `Counter` (the
category's frequency, prior 0); `counter_calc_method=SkipTest`. Plain
boosting (B4). The arms decompose those two statistics:

  cb_default        the harness arm, unchanged
  cb_borders_only   simple_ctr=['Borders'] -- the Counter (frequency) CTR off
  cb_counter_only   simple_ctr=['Counter'] -- the target CTR off
  cb_one_prior      Borders with a single prior 0.5 (one column per cat, the
                    shape of our ordered TS) + Counter

Stage B, pre-registered after Stage A localized the whole edge to the
Borders target CTR (the Counter and the three priors gave back nothing):
  cb_has_time       has_time=True -- the CTR is computed on the data order as
                    its one permutation instead of random permutations
  cb_ctr254         Borders:CtrBorderCount=254 -- the CTR value quantized at
                    254 borders instead of 15 (ours is binned at 255)
  cb_zero_prior     Borders:Prior=0/1 -- a single zero prior, i.e. no
                    smoothing of the target statistic at all

`edge` = (our Brier − CatBoost's) / ours, positive = CatBoost ahead.
`recovered` = 1 − edge(arm) / edge(default): the share of the default edge an
ablation gives back. ChimeraBoost at library defaults runs on the same splits
as the reference line.

PROTOCOL: paired identical splits (test 20%, then the harness's internal
validation carve, stratified), 3 seeds, Brier (the K-sum form the harness
uses, so binary and multiclass share one definition). Resumable JSONL;
aggregate table at the end. Stage A = the four arms above.

Run:
    python benchmarks/probe_catboost_hc_ablation.py
    python benchmarks/probe_catboost_hc_ablation.py --table-only
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

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-cb-hc-ablation.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50

# CatBoost ahead on Brier (campaign-base-20260816.json, 3 seeds).
GAP = ["hc:sf-police-incidents", "hc:Traffic_violations", "hc:kick",
       "hc:okcupid-stem", "hc:porto-seguro"]
# We win: if an ablation moves these as much as the gap sets, it is not
# specific and means nothing.
CONTROL = ["hc:kdd_ipums_la_97-small", "hc:eucalyptus"]
ALL = GAP + CONTROL

_BORDERS = "Borders"
_COUNTER = "Counter"
_ONE_PRIOR = "Borders:Prior=0.5/1"
_BORDERS_254 = "Borders:CtrBorderCount=254"
_ZERO_PRIOR = "Borders:Prior=0/1"
ARMS = {
    "cb_default":      {},
    "cb_borders_only": {"simple_ctr": [_BORDERS], "combinations_ctr": [_BORDERS]},
    "cb_counter_only": {"simple_ctr": [_COUNTER], "combinations_ctr": [_COUNTER]},
    "cb_one_prior":    {"simple_ctr": [_ONE_PRIOR, _COUNTER],
                        "combinations_ctr": [_ONE_PRIOR, _COUNTER]},
    # Stage B (pre-registered after Stage A localized the edge to the target
    # CTR): decompose the CTR's construction.
    "cb_has_time":     {"has_time": True},
    "cb_ctr254":       {"simple_ctr": [_BORDERS_254, _COUNTER],
                        "combinations_ctr": [_BORDERS_254, _COUNTER]},
    "cb_zero_prior":   {"simple_ctr": [_ZERO_PRIOR, _COUNTER],
                        "combinations_ctr": [_ZERO_PRIOR, _COUNTER]},
}
ARM_ORDER = ["chimera"] + list(ARMS)


def _brier(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def _run_catboost(Xtr, ytr, Xte, yte, cat, seed, overrides):
    from catboost import CatBoostClassifier
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=seed, stratify=ytr)
    params = dict(iterations=MAX_ITERS, random_seed=seed, verbose=False,
                  thread_count=-1, allow_writing_files=False,
                  early_stopping_rounds=PATIENCE)
    params.update(overrides)
    t = time.time()
    m = CatBoostClassifier(**params)
    m.fit(Xa, ya, cat_features=list(cat) if cat else None, eval_set=(Xv, yv))
    fit_s = time.time() - t
    proba = m.predict_proba(Xte)
    return _brier(yte, proba, m.classes_), fit_s, int(m.get_best_iteration() or 0)


def _run_chimera(Xtr, ytr, Xte, yte, cat, seed):
    # Library defaults, as the harness measures them: no explicit eval_set, so
    # ChimeraBoost carves its own early-stopping split.
    t = time.time()
    m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                               early_stopping_rounds=PATIENCE, random_state=seed)
    m.fit(Xtr, ytr, cat_features=cat)
    fit_s = time.time() - t
    return _brier(yte, m.predict_proba(Xte), m.classes_), fit_s, 0


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
            if (key, "chimera", seed) not in done:
                b, fit_s, _ = _run_chimera(Xtr, ytr, Xte, yte, cat, seed)
                _append({"dataset": key, "arm": "chimera", "seed": seed, "task": task,
                         "brier": b, "fit_time": fit_s, "best_iter": 0})
                print(f"  [{'chimera':15s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s", flush=True)
            for arm, overrides in ARMS.items():
                if (key, arm, seed) in done:
                    continue
                b, fit_s, bi = _run_catboost(Xtr, ytr, Xte, yte, cat, seed, overrides)
                _append({"dataset": key, "arm": arm, "seed": seed, "task": task,
                         "brier": b, "fit_time": fit_s, "best_iter": bi})
                print(f"  [{arm:15s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s  it={bi}",
                      flush=True)
    table()


def table():
    agg = defaultdict(list)
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                agg[(r["dataset"], r["arm"])].append(r["brier"])

    print("\n" + "=" * 120)
    print("CATBOOST HC ABLATION -- Brier (lower better). edge = (ours - CatBoost) / ours, "
          "positive = CatBoost ahead; recovered = share of the default edge an "
          "ablation gives back.")
    print("=" * 120)
    print(f"{'dataset':28s}" + "".join(f"{a:>16s}" for a in ARM_ORDER)
          + "   edge@default" + "".join(f"  rec@{a[3:]:>11s}" for a in ARMS if a != "cb_default"))
    summary = defaultdict(list)
    for group, name in ((GAP, "GAP SETS"), (CONTROL, "CONTROLS")):
        print(f"\n-- {name} " + "-" * (116 - len(name)))
        for ds in group:
            vals = {a: (float(np.mean(agg[(ds, a)])) if agg.get((ds, a)) else None)
                    for a in ARM_ORDER}
            if vals["chimera"] is None or vals["cb_default"] is None:
                continue
            ours = vals["chimera"]
            e_def = (ours - vals["cb_default"]) / ours
            line = f"{ds[3:][:28]:28s}" + "".join(
                f"{vals[a]:16.6f}" if vals[a] is not None else f"{'-':>16s}" for a in ARM_ORDER)
            line += f"{100 * e_def:14.2f}%"
            for a in ARMS:
                if a == "cb_default":
                    continue
                if vals[a] is None or abs(e_def) < 1e-12:
                    line += f"{'-':>17s}"
                    continue
                rec = 1.0 - (ours - vals[a]) / ours / e_def
                summary[(name, a)].append(rec)
                line += f"{100 * rec:16.0f}%"
            print(line)
    print()
    for (name, a), recs in sorted(summary.items()):
        print(f"  {name:9s} {a:16s} recovered: median {100 * np.median(recs):+.0f}%  "
              f"(per set {', '.join(f'{100 * r:+.0f}%' for r in recs)})")
    print("\n(recovered 100% = the ablated CatBoost falls to our Brier; 0% = the "
          "knob did nothing; negative = removing it made CatBoost better)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
