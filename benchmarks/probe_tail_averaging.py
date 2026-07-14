"""Probe: is the early-stopping point noisy enough that (a) smoothing the val
curve or (b) Polyak-style tail-averaging of the boosting trajectory buys test
accuracy for FREE (no extra fit cost, predict-time tree reweighting only)?

MECHANISM (pre-registered 2026-07-13): best_iteration_ = argmin of a NOISY val
curve estimates the optimal stop with variance; averaging F over a window near
the stop (equivalent to triangularly downweighting the last trees) reduces both
the stop-selection variance and the last-trees' noise. Analog of Polyak-Ruppert
iterate averaging / model EMA in DL. Cost: zero fit, tiny predict change.

PREDICTIONS: small broad wins (+0.1-0.4%) concentrated on datasets with small
validation sets / flat val minima; no dataset badly hurt. KILL if wins don't
replicate across seeds or reg/binary disagree wildly.

PROTOCOL: fit once per (dataset, seed) with early_stopping=False at the
production lr, eval_set = a held-out val split sized like production's
validation_fraction. Simulate the production patience-50 stop t* from
validation_history_, then score on test:
  stop      : prediction at t* (production baseline)
  smooth    : prediction at argmin of moving-averaged val curve (w=9)
  tailK     : mean of staged raw predictions over rounds (t*-K, t*], K=5/10/20
  symK      : mean over [t*-K, t*+K] (patience overshoot trees exist in prod)
Binary uses temperature T=1 throughout (relative comparison unaffected).
Results: benchmarks/results/probe-tailavg.jsonl; table printed at the end.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-tailavg.jsonl")
SEEDS = (0, 1, 2)
N_EST = 600
PATIENCE = 50
TEST_CAP = 8000

REG = ["gr:reg_num/cpu_act", "gr:reg_num/elevators", "gr:reg_num/houses",
       "gr:reg_num/wine_quality", "gr:reg_num/sulfur",
       "gr:reg_num/Brazilian_houses", "gr:reg_num/pol"]
BIN = ["gr:clf_num/electricity", "gr:clf_num/MagicTelescope",
       "gr:clf_num/heloc", "gr:clf_num/credit"]


def _patience_stop(val):
    """Round index (0-based) where production patience-50 ES would stop."""
    best, best_i, since = np.inf, 0, 0
    for i, v in enumerate(val):
        if v < best:
            best, best_i, since = v, i, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best_i


def _moving_avg(x, w=9):
    pad = w // 2
    xp = np.concatenate([np.repeat(x[0], pad), x, np.repeat(x[-1], pad)])
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")


def _score(task, yte, raw):
    if task == "regression":
        return float(np.sqrt(np.mean((yte - raw) ** 2)))
    p = np.clip(1.0 / (1.0 + np.exp(-raw)), 1e-12, 1 - 1e-12)
    return float(brier_score_loss(yte, p)), float(log_loss(yte, p))


def main():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"]))

    for key in REG + BIN:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=strat)
            strat2 = ytr if task != "regression" else None
            Xtr, Xval, ytr, yval = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=seed, stratify=strat2)
            if len(yte) > TEST_CAP:
                Xte, yte = Xte[:TEST_CAP], yte[:TEST_CAP]

            Est = (ChimeraBoostRegressor if task == "regression"
                   else ChimeraBoostClassifier)
            m = Est(n_estimators=N_EST, early_stopping=False,
                    learning_rate=0.1, random_state=0)
            m.fit(Xtr, ytr, eval_set=(Xval, yval), cat_features=cat)
            val = np.asarray(m.validation_history_, dtype=float)
            if val.size < 30:
                continue
            t_star = _patience_stop(val)

            # Staged raw predictions on test, rounds 0..min(T, t*+PATIENCE).
            horizon = min(len(val), t_star + PATIENCE + 1)
            if task == "regression":
                gen = m.staged_predict(Xte)
            else:
                gen = m.model_.staged_predict_raw(Xte)
            staged = []
            for i, s in enumerate(gen):
                if i >= horizon:
                    break
                staged.append(np.asarray(s, dtype=float).copy())
            staged = np.asarray(staged)

            t_smooth = int(np.argmin(_moving_avg(val)[:horizon]))
            variants = {"stop": staged[t_star], "smooth": staged[t_smooth]}
            for k in (5, 10, 20):
                lo = max(0, t_star - k + 1)
                variants[f"tail{k}"] = staged[lo:t_star + 1].mean(axis=0)
                hi = min(len(staged), t_star + k + 1)
                variants[f"sym{k}"] = staged[max(0, t_star - k):hi].mean(axis=0)

            # Did patience actually trigger, or is t* just the horizon cap?
            stopped = bool(t_star + PATIENCE < len(val))
            row = {"dataset": key, "seed": seed, "task": task,
                   "t_star": int(t_star), "t_smooth": t_smooth,
                   "rounds": int(len(val)), "stopped": stopped}
            for name, raw in variants.items():
                s = _score(task, yte, raw)
                if task == "regression":
                    row[name] = s
                else:
                    row[name], row[f"{name}_ll"] = s
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            base = row["stop"]
            print(f"{key} s{seed} t*={t_star}: " + " ".join(
                f"{n}={100*(base-row[n])/base:+.2f}%"
                for n in variants if n != "stop"), flush=True)
    table()


def table():
    rows = [json.loads(l) for l in open(RESULTS)]
    names = ["smooth", "tail5", "tail10", "tail20", "sym5", "sym10", "sym20"]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print(f"\n{'dataset':40} " + " ".join(f"{n:>8}" for n in names))
    per_task = {}
    for ds, rs in by_ds.items():
        deltas = {n: float(np.mean([100 * (r["stop"] - r[n]) / r["stop"]
                                    for r in rs])) for n in names}
        per_task.setdefault(rs[0]["task"], []).append(deltas)
        print(f"{ds:40} " + " ".join(f"{deltas[n]:+8.3f}" for n in names))
    for task, ds in per_task.items():
        print(f"\n  {task}: mean " + " ".join(
            f"{n}={np.mean([d[n] for d in ds]):+.3f}%" for n in names))
    # Split by whether patience actually triggered (rows where t* is just the
    # horizon cap are the regime tail-averaging is expected to hurt).
    for regime, flag in (("patience-stopped", True), ("cap-bound", False)):
        sel = [r for r in rows if r.get("stopped") is flag]
        if sel:
            print(f"  {regime} ({len(sel)} rows): mean " + " ".join(
                f"{n}={np.mean([100 * (r['stop'] - r[n]) / r['stop'] for r in sel]):+.3f}%"
                for n in names))
    print("\n(positive = variant better than production patience-stop; "
          "RMSE for reg, Brier for binary, temperature=1)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
