"""Probe: should each BAGGED MEMBER replay its structure on the full dataset?

MECHANISM HYPOTHESIS (pre-registered 2026-08-01, before any results):

`refit_full="replay"` is default-ON for the single model and was worth +2.1
points of Grinsztajn RMSE% (96.3 -> 98.4): learn the structure on the 80%
early-stopping split, then replay those splits against full-data gradients so
the leaf values see every row.

Bag members never get this. `_fit_bagged` hands each member an explicit
`eval_set` (its OOB rows), and `_refit_on_full` fires only when
`auto_split` is True — which requires `eval_set is None`. So every member's
leaf values are estimated from its 0.8n bag and nothing reclaims the rest.

REFIT_PLAN.md line 100 says "Ens8/bag members have no data tax". That is true
of the ENSEMBLE (every row is in some member's bag) but not of any INDIVIDUAL
member: 20% of the dataset contributes to no leaf value in that member.

Why this should be net positive rather than a diversity loss: the LRE
post-mortem measured that bag lift is STRUCTURAL diversity, and that leaf-only
resampling buys nothing ("capture -0.14, bag lift is structural diversity").
Replay keeps each member's structure exactly as its own bag grew it — the
diverse part — and only re-estimates leaf values on all rows. So it should buy
per-member strength at close to zero diversity cost. If the LRE finding is
right, this is the asymmetry to exploit; if diversity really does live in the
leaves after all, this probe will show it as a wash or a loss.

THE PRIZE IS PARETO, NOT JUST STRENGTH. Ens8 sits at 99.8/99.8 but costs 26.8x.
If refit members let a SMALLER bag reach the same strength, that is a large
frontier move (26.8x -> ~10x at equal strength). So the probe sweeps K.

ARMS (identical member structures within a seed — the refit arm replays the
SAME fitted members, so the comparison is exactly paired):
  plain@K  - average K members as the library builds them today
  refit@K  - same K members, each replayed on all n training rows, averaged

PREDICTIONS:
  If the mechanism is real: refit@K beats plain@K at every K, the gap is
  largest at small K (fewer members = less averaging to hide weak leaves), and
  refit@3 is competitive with plain@8.
  If diversity lives in the leaves: refit@K is flat-to-worse and the whole idea
  dies here, at the cost of one probe and no library change.

Primary metric: RMSE (regression) / Brier (classification).
Resumable JSONL; `--table-only` reprints.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

import chimeraboost  # noqa: E402
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402
from chimeraboost.booster import GradientBoosting  # noqa: E402
from chimeraboost.sklearn_api import (_member_sample_indices,  # noqa: E402
                                      _member_oob_eval_indices)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-bagrefit.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50
K_MAX = 8
KS = (2, 3, 5, 8)
MAX_SAMPLES = 0.8          # the shipped member draw
MEMBER_LR = 0.15           # shipped bagged-member defaults (BAGGING_PLAN B3)
MEMBER_COLSAMPLE = 0.85

DATASETS = [
    # regression, spanning the sizes where bagging is used
    "gr:reg_num/cpu_act",
    "gr:reg_num/sulfur",
    "gr:reg_num/Brazilian_houses",
    "gr:reg_num/elevators",
    "gr:reg_num/wine_quality",
    # binary
    "gr:clf_num/bank-marketing",
    "gr:clf_num/heloc",
    "gr:clf_num/electricity",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/credit",
]


def _score(task, y, pred):
    if task == "regression":
        return float(np.sqrt(np.mean((y - pred) ** 2)))
    # binary Brier in the harness's K=2 sum form
    return float(np.mean(2.0 * (pred - y) ** 2))


def _replay_member(member, X_full, y_full, cat_features, task):
    """Rebuild this member's booster by replaying its splits against full-data
    gradients. Mirrors _refit_on_full's donor path: same rounds, pinned learning
    rate, donor binner adopted, no early stopping."""
    w = member.model_
    donor = (w.trees_, w.prep_)
    kw = dict(
        n_estimators=len(w.trees_),
        learning_rate=float(w.lr_),
        depth=w.depth,
        l2_leaf_reg=w.l2_leaf_reg,
        max_bins=w.max_bins,
        subsample=w.subsample,
        colsample=w.colsample,
        min_child_weight=w.min_child_weight,
        early_stopping_rounds=None,
        thread_count=w.thread_count,
        random_state=w.random_state,
        cat_combinations=w.cat_combinations,
        leaf_estimation_iterations=w.leaf_estimation_iterations,
        linear_leaves=w.linear_leaves,
        linear_lambda=w.linear_lambda,
        cross_pairs=w.cross_pairs,
        quantize_gradients=w.quantize_gradients,
        replay_donor=donor,
    )
    loss = "Logloss" if task != "regression" else w.loss_name
    lkw = dict(getattr(w, "loss_kwargs", {}) or {})
    b = GradientBoosting(loss=loss, loss_kwargs=lkw, **kw)
    b.fit(X_full, y_full, cat_features=cat_features)
    return b


def _member_predict(booster, X, task, temperature=1.0):
    raw = booster.predict_raw(X)
    if task == "regression":
        return np.asarray(raw, dtype=np.float64).ravel()
    p = booster.loss_.transform(np.asarray(raw, dtype=np.float64).ravel()
                               / temperature)
    return p


def _done():
    seen = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        seen.add((r["dataset"], r["seed"]))
                    except Exception:
                        pass
    return seen


def main():
    print(f"chimeraboost from {chimeraboost.__file__}")
    done = _done()
    for key in DATASETS:
        X, y, cat, task = rdata.load(key)
        if task == "multiclass":
            print(f"skip {key}: multiclass replay is not wired")
            continue
        print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}  task={task}")
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed,
                stratify=(y if task != "regression" else None))
            n = len(ytr)
            classes = None
            if task != "regression":
                classes = np.unique(ytr)
                yte01 = (yte == classes[1]).astype(np.float64)
                ytr_fit = ytr
            else:
                yte01 = yte.astype(np.float64)
                ytr_fit = ytr

            seeds_k = np.random.default_rng(seed).integers(
                0, 2**31 - 1, size=K_MAX)
            plain_preds, refit_preds = [], []
            t0 = time.time()
            t_refit = 0.0
            for mseed in seeds_k:
                idx = _member_sample_indices(n, MAX_SAMPLES, int(mseed), None)
                oob = _member_oob_eval_indices(idx, n, None)
                Est = (ChimeraBoostRegressor if task == "regression"
                       else ChimeraBoostClassifier)
                m = Est(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
                        learning_rate=MEMBER_LR, colsample=MEMBER_COLSAMPLE,
                        random_state=int(mseed))
                m._is_bag_member = True
                m.fit(Xtr[idx], ytr_fit[idx], cat_features=cat,
                      eval_set=(Xtr[oob], ytr_fit[oob]))
                temp = getattr(m, "temperature_", 1.0)
                plain_preds.append(
                    _member_predict(m.model_, Xte, task, temp))
                # --- the arm under test: replay this member on ALL train rows
                tr = time.time()
                y_rep = (ytr_fit if task == "regression"
                         else (ytr_fit == classes[1]).astype(np.float64))
                b = _replay_member(m, Xtr, y_rep, cat, task)
                t_refit += time.time() - tr
                refit_preds.append(_member_predict(b, Xte, task, temp))
            total_s = time.time() - t0

            rec = {"dataset": key, "seed": seed, "task": task, "n_train": int(n),
                   "fit_s": total_s, "refit_s": t_refit, "plain": {}, "refit": {}}
            for K in KS:
                rec["plain"][str(K)] = _score(
                    task, yte01, np.mean(plain_preds[:K], axis=0))
                rec["refit"][str(K)] = _score(
                    task, yte01, np.mean(refit_preds[:K], axis=0))
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            g8 = 100.0 * (rec["plain"]["8"] - rec["refit"]["8"]) / rec["plain"]["8"]
            g2 = 100.0 * (rec["plain"]["2"] - rec["refit"]["2"]) / rec["plain"]["2"]
            print(f"  s{seed} {total_s:6.1f}s (replay {t_refit:5.1f}s)  "
                  f"K=2 {g2:+.3f}%   K=8 {g8:+.3f}%")
    table()


def table():
    rows = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    task_of = {}
    for r in rows:
        task_of[r["dataset"]] = r["task"]
        for K in KS:
            agg[r["dataset"]][("plain", K)].append(r["plain"][str(K)])
            agg[r["dataset"]][("refit", K)].append(r["refit"][str(K)])

    print("\n" + "=" * 116)
    print("BAG MEMBER REFIT — % improvement from replaying each member on all train rows")
    print("  positive = refit better. Same member structures in both arms, so this is exactly paired.")
    print("=" * 116)
    print(f"{'dataset':34s}{'task':7s}" + "".join(f"{'K=' + str(K):>12s}" for K in KS))

    gains = defaultdict(list)
    for ds in DATASETS:
        if ds not in agg:
            continue
        line = f"{ds.replace('gr:', '')[:34]:34s}{task_of[ds][:6]:7s}"
        for K in KS:
            p = float(np.mean(agg[ds][("plain", K)]))
            q = float(np.mean(agg[ds][("refit", K)]))
            g = 100.0 * (p - q) / p
            gains[K].append(g)
            line += f"{g:+11.3f}%"
        print(line)

    print("\n" + "=" * 116)
    print("VERDICT — mean improvement by bag size")
    print("=" * 116)
    for K in KS:
        g = gains[K]
        if g:
            print(f"  K={K}: mean {np.mean(g):+7.3f}%   median {np.median(g):+7.3f}%   "
                  f"better on {sum(1 for x in g if x > 0)}/{len(g)} datasets")

    # Pareto question: does a small refit bag match a big plain bag?
    print("\n" + "=" * 116)
    print("PARETO — can a SMALL refit bag match a LARGE plain bag?")
    print("  each cell: refit@K vs plain@8, % better (positive = refit@K wins at a fraction of the cost)")
    print("=" * 116)
    print(f"{'dataset':34s}" + "".join(f"{'refit' + str(K) + ' vs p8':>15s}" for K in KS))
    tot = defaultdict(list)
    for ds in DATASETS:
        if ds not in agg:
            continue
        p8 = float(np.mean(agg[ds][("plain", 8)]))
        line = f"{ds.replace('gr:', '')[:34]:34s}"
        for K in KS:
            q = float(np.mean(agg[ds][("refit", K)]))
            d = 100.0 * (p8 - q) / p8
            tot[K].append(d)
            line += f"{d:+14.3f}%"
        print(line)
    print()
    for K in KS:
        d = tot[K]
        if d:
            print(f"  refit@{K} vs plain@8: mean {np.mean(d):+7.3f}%   "
                  f"wins on {sum(1 for x in d if x > 0)}/{len(d)} datasets")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
