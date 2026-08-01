"""Probe: WHICH CatBoost mechanism produces its remaining edge over us?

MECHANISM QUESTION (pre-registered 2026-08-01, before any results):
On the 2026-07-31 external field we beat CatBoost on 46 of 57 Grinsztajn sets,
and every remaining loss is noisy low-dimensional BINARY classification
(bank-marketing, credit, heloc, default-of-credit, Diabetes130US, california),
with gaps under 1.1% of Brier.

The calibration explanation is already REFUTED (scratch analysis 2026-08-01):
our miscalibration term (MCB 0.00220) is LOWER than CatBoost's (0.00231),
LightGBM's (0.00233) and HGB's (0.00272), and subtracting each model's own MCB
-- i.e. perfectly recalibrating both sides -- leaves the head-to-head scoreboard
completely unchanged (17W-6L before and after). The deficit is RESOLUTION, not
reliability.

That leaves CatBoost's structural machinery. On these sets the features are
numeric, so its ordered target statistics are irrelevant. Its remaining
distinctive default-ON machinery is two stochastic regularizers we have never
implemented (architecture map degree-of-freedom #1: "no gain dithering"):
  * random_strength=1      -- Gaussian noise added to each split score before
                              the argmax, decaying with the round index.
  * bagging_temperature=1  -- Bayesian bootstrap: per-tree row weights
                              w_i = u_i^(1/T), every row kept, mean weight 1.
Both attack the winner's curse: a split chosen by pure argmax over ~p*128 noisy
gain estimates is selected partly for its noise and underdelivers out of sample.
Our split search is a pure greedy argmax with no dithering and (at the default
subsample=1.0) no row randomness at all.

THE PROBE: ablate the OPPONENT. Run CatBoost at its defaults, then with each
regularizer disabled, on the sets where it beats us. Using CatBoost as an oracle
for its own mechanism costs one short benchmark and answers the question that
decides whether the mechanism is worth building.

PREDICTIONS (stated before running):
  If the hypothesis is RIGHT: disabling both regularizers costs CatBoost most of
  its edge on the loss cluster -- its Brier rises to roughly our level -- while
  the control sets (where we already win big) move little.
  If the hypothesis is WRONG: CatBoost's Brier barely moves, its edge survives
  the ablation, and the mechanism lives somewhere else. Then DO NOT build it.

PROTOCOL: paired identical splits (the harness's own _val_split / train_test_split
convention), 3 seeds, Brier primary (the house decision metric for classification).
ChimeraBoost at library defaults is included on the same splits as the reference
line. Resumable JSONL; aggregate table printed at the end.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost import ChimeraBoostClassifier  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-cb-ablation.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50

# The sets where CatBoost still beats us (from 20260731-142609.json).
LOSS_CLUSTER = [
    "gr:clf_num/bank-marketing",
    "gr:clf_num/credit",
    "gr:clf_num/heloc",
    "gr:clf_num/default-of-credit-card-clients",
    "gr:clf_num/Diabetes130US",
    "gr:clf_num/california",
]
# Controls: binary sets where WE win comfortably. If the ablation moves these
# as much as the loss cluster, the effect is not specific and means nothing.
CONTROL = [
    "gr:clf_num/covertype",
    "gr:clf_num/electricity",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/pol",
]
ALL = LOSS_CLUSTER + CONTROL

# CatBoost arms. "default" is what the benchmark harness runs today.
ARMS = {
    "cb_default":   {},
    "cb_no_rs":     {"random_strength": 0.0},
    "cb_no_bag":    {"bootstrap_type": "No"},
    "cb_no_both":   {"random_strength": 0.0, "bootstrap_type": "No"},
}


def _brier(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _split(X, y, seed):
    """Test split, then the internal validation carve — the harness convention."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                          stratify=y)
    return Xtr, Xte, ytr, yte


def _run_catboost(Xtr, ytr, Xte, yte, cat, seed, overrides):
    from catboost import CatBoostClassifier
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=seed, stratify=ytr)
    params = dict(iterations=MAX_ITERS, random_seed=seed, verbose=False,
                  thread_count=-1, cat_features=list(cat) if cat else None,
                  early_stopping_rounds=PATIENCE)
    params.update(overrides)
    t = time.time()
    m = CatBoostClassifier(**params)
    m.fit(Xa, ya, eval_set=(Xv, yv))
    fit_s = time.time() - t
    proba = m.predict_proba(Xte)
    return _brier(yte, proba, m.classes_), fit_s, int(m.get_best_iteration() or 0)


def _run_chimera(Xtr, ytr, Xte, yte, cat, seed):
    # Library defaults, exactly as the harness measures them: no explicit
    # eval_set, so ChimeraBoost carves its own early-stopping split.
    t = time.time()
    m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                               early_stopping_rounds=PATIENCE,
                               random_state=seed)
    m.fit(Xtr, ytr, cat_features=cat)
    fit_s = time.time() - t
    proba = m.predict_proba(Xte)
    return _brier(yte, proba, m.classes_), fit_s, 0


def _done_keys():
    seen = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        seen.add((r["dataset"], r["arm"], r["seed"]))
                    except Exception:
                        pass
    return seen


def _append(rec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    done = _done_keys()
    for key in ALL:
        X, y, cat, task = rdata.load(key)
        if task != "binary":
            print(f"skip {key}: task={task}")
            continue
        print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}  cats={len(cat) if cat else 0}")
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            for arm, overrides in ARMS.items():
                if (key, arm, seed) in done:
                    continue
                b, fit_s, bi = _run_catboost(Xtr, ytr, Xte, yte, cat, seed, overrides)
                _append({"dataset": key, "arm": arm, "seed": seed,
                         "brier": b, "fit_time": fit_s, "best_iter": bi})
                print(f"  [{arm:11s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s  it={bi}")
            if (key, "chimera", seed) not in done:
                b, fit_s, _ = _run_chimera(Xtr, ytr, Xte, yte, cat, seed)
                _append({"dataset": key, "arm": "chimera", "seed": seed,
                         "brier": b, "fit_time": fit_s, "best_iter": 0})
                print(f"  [{'chimera':11s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s")
    table()


def table():
    rows = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["arm"])].append(r["brier"])

    arms = ["chimera", "cb_default", "cb_no_rs", "cb_no_bag", "cb_no_both"]
    print("\n" + "=" * 118)
    print("CATBOOST ABLATION — Brier (lower better). "
          "'edge' = how much CatBoost beats us, as % of our Brier (positive = CatBoost ahead).")
    print("=" * 118)
    hdr = f"{'dataset':38s}" + "".join(f"{a:>13s}" for a in arms) + f"{'edge@def':>10s}{'edge@none':>11s}"
    print(hdr)

    for group, name in ((LOSS_CLUSTER, "LOSS CLUSTER"), (CONTROL, "CONTROLS")):
        print(f"\n-- {name} " + "-" * (114 - len(name)))
        for ds in group:
            vals = {a: (float(np.mean(agg[(ds, a)])) if agg.get((ds, a)) else None)
                    for a in arms}
            if vals["chimera"] is None or vals["cb_default"] is None:
                continue
            line = f"{ds.replace('gr:', '')[:38]:38s}"
            for a in arms:
                v = vals[a]
                line += f"{v:13.6f}" if v is not None else f"{'-':>13s}"
            ours = vals["chimera"]
            e_def = 100.0 * (ours - vals["cb_default"]) / ours
            line += f"{e_def:9.2f}%"
            if vals["cb_no_both"] is not None:
                e_non = 100.0 * (ours - vals["cb_no_both"]) / ours
                line += f"{e_non:10.2f}%"
            print(line)

    # Summary: does the ablation remove CatBoost's edge where it beats us?
    print("\n" + "=" * 118)
    print("VERDICT — mean CatBoost edge over ChimeraBoost (% of our Brier, positive = they win)")
    print("=" * 118)
    for group, name in ((LOSS_CLUSTER, "loss cluster"), (CONTROL, "controls")):
        for arm in ("cb_default", "cb_no_rs", "cb_no_bag", "cb_no_both"):
            edges = []
            for ds in group:
                o = agg.get((ds, "chimera"))
                c = agg.get((ds, arm))
                if o and c:
                    ours, theirs = float(np.mean(o)), float(np.mean(c))
                    edges.append(100.0 * (ours - theirs) / ours)
            if edges:
                print(f"  {name:14s} {arm:12s} mean edge {np.mean(edges):+7.3f}%   "
                      f"(n={len(edges)}, CatBoost ahead on {sum(1 for e in edges if e > 0)})")
        print()


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
