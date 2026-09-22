"""Probe R2 (CAMPAIGN_PLAN I032): the 1-SE early-stopping round.

Production stops at the raw argmin of a held-out validation curve (patience
50) and the refit replays that round count at 1.25x on all rows. The
smoothed-argmin half of R2 was already probed on 2026-07-13
(`probe_tail_averaging.py`, barrier B19): flat, one outlier row. This probe
takes the other half, the one that cannot lose the cost axis: pick the
EARLIEST round whose validation loss is within one noise unit of the
minimum, so the fit (and the replayed refit) carries fewer trees.

Protocol, the July probe's: one fit per (dataset, seed) with
early_stopping=False for N_EST rounds, eval_set = a 20% split of the training
rows sized like production's validation_fraction, production learning rate
(the size-adaptive default). Simulate the patience-50 stop t* from
validation_history_, then score staged test predictions at:

  stop        : round t* (the production rule; the baseline)
  1se-local   : earliest t <= t* with val[t] <= val[t*] + sigma, where sigma is
                the std of (val - 9-round moving average) over [t*-50, t*+50] --
                the curve's own round-to-round noise, the single-curve stand-in
                for the CV standard error the 1-SE rule was defined with
  tol0.1 / tol0.5 : earliest t <= t* with val[t] <= val[t*] * (1 + tol)

Every rule stops at or before t*, so its round ratio t_rule / t* is the cost
read (production replays t* / 0.8 rounds; the ratio carries over unchanged).
Binary uses temperature 1 throughout (relative comparison unaffected).
Results: benchmarks/results/probe-es-one-se.jsonl; table at the end.

Run:
    python benchmarks/probe_es_one_se.py            # fits, appends, prints
    python benchmarks/probe_es_one_se.py --table-only
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-es-one-se.jsonl")
SEEDS = (0, 1, 2)
N_EST = 600
PATIENCE = 50
TEST_CAP = 8000
NOISE_WINDOW = 50
RULES = ["1se-local", "tol0.1", "tol0.5"]

# The July panel plus four more of each task, so a stratum reads on ~10 sets.
REG = ["gr:reg_num/cpu_act", "gr:reg_num/elevators", "gr:reg_num/houses",
       "gr:reg_num/wine_quality", "gr:reg_num/sulfur",
       "gr:reg_num/Brazilian_houses", "gr:reg_num/pol",
       "gr:reg_num/abalone", "gr:reg_num/house_sales",
       "gr:reg_num/MiamiHousing2016", "gr:reg_num/superconduct"]
BIN = ["gr:clf_num/electricity", "gr:clf_num/MagicTelescope",
       "gr:clf_num/heloc", "gr:clf_num/credit",
       "gr:clf_num/Bioresponse", "gr:clf_num/california",
       "gr:clf_num/eye_movements", "gr:clf_num/house_16H",
       "gr:clf_num/jannis", "gr:clf_num/bank-marketing"]


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
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def _local_noise(val, t_star):
    lo, hi = max(0, t_star - NOISE_WINDOW), min(len(val), t_star + NOISE_WINDOW + 1)
    seg = val[lo:hi]
    if seg.size < 5:
        return 0.0
    return float(np.std(seg - _moving_avg(seg)))


def _earliest_within(val, t_star, bound):
    """Earliest round t <= t* with val[t] <= bound."""
    ok = np.nonzero(val[:t_star + 1] <= bound)[0]
    return int(ok[0]) if ok.size else int(t_star)


def rule_rounds(val, t_star):
    """{rule: round} for every rule in RULES, all <= t*."""
    v_star = float(val[t_star])
    return {"1se-local": _earliest_within(val, t_star, v_star + _local_noise(val, t_star)),
            "tol0.1": _earliest_within(val, t_star, v_star * 1.001),
            "tol0.5": _earliest_within(val, t_star, v_star * 1.005)}


def _score(task, yte, raw):
    if task == "regression":
        return float(np.sqrt(np.mean((yte - raw) ** 2)))
    p = np.clip(1.0 / (1.0 + np.exp(-raw)), 1e-12, 1 - 1e-12)
    return float(brier_score_loss(yte, p))


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
            m = Est(n_estimators=N_EST, early_stopping=False, random_state=0)
            m.fit(Xtr, ytr, eval_set=(Xval, yval), cat_features=cat)
            val = np.asarray(m.validation_history_, dtype=float)
            if val.size < 30:
                continue
            t_star = int(_patience_stop(val))
            rounds = rule_rounds(val, t_star)

            gen = (m.staged_predict(Xte) if task == "regression"
                   else m.model_.staged_predict_raw(Xte))
            staged = {}
            want = {t_star} | set(rounds.values())
            for i, s in enumerate(gen):
                if i in want:
                    staged[i] = np.asarray(s, dtype=float).copy()
                if i >= t_star:
                    break

            row = {"dataset": key, "seed": seed, "task": task,
                   "t_star": t_star, "rounds": int(len(val)),
                   "stopped": bool(t_star + PATIENCE < len(val)),
                   "noise": _local_noise(val, t_star),
                   "stop": _score(task, yte, staged[t_star])}
            for name, t in rounds.items():
                row[name] = _score(task, yte, staged[t])
                row[f"{name}_t"] = int(t)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            base = row["stop"]
            print(f"{key} s{seed} t*={t_star}: " + " ".join(
                f"{n}={100 * (base - row[n]) / base:+.2f}%@{row[n + '_t']}"
                for n in RULES), flush=True)
    table()


def table():
    rows = [json.loads(line) for line in open(RESULTS)]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print(f"\n{'dataset':36} " + " ".join(f"{n:>10} {'t/t*':>5}" for n in RULES))
    per_task = {}
    for ds, rs in by_ds.items():
        d = {n: float(np.mean([100 * (r["stop"] - r[n]) / r["stop"] for r in rs]))
             for n in RULES}
        ratio = {n: float(np.mean([r[n + "_t"] / max(r["t_star"], 1) for r in rs]))
                 for n in RULES}
        per_task.setdefault(rs[0]["task"], []).append((ds, d, ratio))
        print(f"{ds:36} " + " ".join(f"{d[n]:+10.3f} {ratio[n]:5.2f}" for n in RULES))
    print()
    for task, items in per_task.items():
        n_ds = len(items)
        for n in RULES:
            deltas = np.array([d[n] for _, d, _ in items])
            wins = int((deltas > 1e-9).sum())
            losses = int((deltas < -1e-9).sum())
            ratio = float(np.mean([r[n] for _, _, r in items]))
            print(f"  {task:10} {n:9}: {wins}W-{losses}L of {n_ds} (bar {n_ds // 2 + 1}+), "
                  f"median {np.median(deltas):+.3f}%, mean {deltas.mean():+.3f}%, "
                  f"rounds ratio {ratio:.2f}")
    print("\n(positive = rule better than the production patience-stop; RMSE for "
          "reg, Brier for binary, temperature 1; t/t* = rounds kept, the cost "
          "read)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
