"""REPLAY probe: does a structure-transfer refit keep the full refit's win?

`refit_full=True` (the default since 2026-07-25) buys accuracy by refitting the
early-stopping winner on all rows -- a second, longer, from-scratch fit that the
attribution profile puts at 37-49% of every default fit. Growing trees is 83-85%
of a fit, and the winner's structures are already known, so `refit_full="replay"`
replays those splits against full-data gradients and refits only the leaf values
(chimeraboost/tree.py:replay_oblivious_tree).

Three arms per (dataset, seed), all on SHIPPED defaults -- nothing pinned, so
the numbers are the ones users would see:

  off     refit_full=False    (the pre-2026-07-25 default)
  full    refit_full=True     (today's default)
  replay  refit_full="replay" (the candidate)

Reported per arm: test metric (RMSE for regression, Brier for binary -- the
primary metrics the Pareto and the ship gate use) and wall-clock fit time.
The decision number is CAPTURE: what fraction of full's gain over off does
replay keep, against what fraction of full's extra cost it pays.

Run from benchmarks/:  python probe_replay_refit.py [key-substring ...]
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

# Same panel REFIT_PLAN's probe used, so the two reads are comparable.
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
ARMS = (("off", False), ("full", True), ("replay", "replay"))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - p) ** 2)))


def run_one(key, seed, is_reg):
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    Est = ChimeraBoostRegressor if is_reg else ChimeraBoostClassifier
    rec = dict(dataset=key, seed=seed, task="reg" if is_reg else "bin")
    for name, mode in ARMS:
        est = Est(random_state=seed, refit_full=mode)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr, cat_features=cat)
        rec[f"t_{name}"] = time.perf_counter() - t0
        if is_reg:
            rec[f"m_{name}"] = rmse(yte, est.predict(Xte))
        else:
            y01 = (np.asarray(yte) == est.classes_[1]).astype(np.float64)
            p1 = est.predict_proba(Xte)[:, 1]
            rec[f"m_{name}"] = float(np.mean((p1 - y01) ** 2))
        rec[f"trees_{name}"] = len(est.model_.trees_)
    return rec


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
            print(f"{key} s{seed}: off {r['m_off']:.5f} full {r['m_full']:.5f} "
                  f"replay {r['m_replay']:.5f}  |  t "
                  f"{r['t_off']:.1f}/{r['t_full']:.1f}/{r['t_replay']:.1f}s",
                  flush=True)
        with open("results/replay_refit_probe.json", "w") as f:
            json.dump(rows, f, indent=1)

    # Lower is better for both metrics, so a gain is (off - arm) / off.
    print("\n=== REPLAY probe: per-dataset means over seeds ===")
    print(f"{'dataset':32s} {'metric':6s} {'full%':>7s} {'replay%':>8s} "
          f"{'capture':>8s} {'full x':>7s} {'replay x':>9s}")
    caps, gf, gr_, cf, cr = [], [], [], [], []
    for key in panel:
        sub = [r for r in rows if r["dataset"] == key]
        if not sub:
            continue
        mo = float(np.mean([r["m_off"] for r in sub]))
        mf = float(np.mean([r["m_full"] for r in sub]))
        mr = float(np.mean([r["m_replay"] for r in sub]))
        to = float(np.mean([r["t_off"] for r in sub]))
        tf = float(np.mean([r["t_full"] for r in sub]))
        tr = float(np.mean([r["t_replay"] for r in sub]))
        g_full = (mo - mf) / mo * 100.0
        g_rep = (mo - mr) / mo * 100.0
        # Capture is only meaningful where the full refit actually gained.
        cap = (g_rep / g_full * 100.0) if abs(g_full) > 1e-9 else float("nan")
        gf.append(g_full)
        gr_.append(g_rep)
        cf.append(tf / to)
        cr.append(tr / to)
        if g_full > 0.05:
            caps.append(cap)
        print(f"{key:32s} {'RMSE' if sub[0]['task'] == 'reg' else 'Brier':6s} "
              f"{g_full:7.3f} {g_rep:8.3f} {cap:7.0f}% {tf / to:7.2f} "
              f"{tr / to:9.2f}")

    print(f"\nmean gain over no-refit:  full {np.mean(gf):+.3f}%   "
          f"replay {np.mean(gr_):+.3f}%")
    if caps:
        print(f"median capture (datasets where full gained >0.05%): "
              f"{np.median(caps):.0f}%  over {len(caps)} sets")
    print(f"mean fit cost vs no-refit: full {np.mean(cf):.2f}x   "
          f"replay {np.mean(cr):.2f}x")
    print(f"=> replay saves {(1 - np.mean(cr) / np.mean(cf)) * 100:.0f}% "
          f"of today's default fit time")
    wins = sum(1 for a, b in zip(gr_, gf) if a >= b)
    print(f"replay >= full on {wins}/{len(gf)} datasets")


if __name__ == "__main__":
    main()
