"""Random-search hyperparameter study for ChimeraBoost on the PMLB tuning suite.

Goal: characterize WHICH knobs help across the suite (not ship a default). The
report suites (Grinsztajn / OpenML-34 / TabArena) stay untouched; PMLB is the
designated tuning ground (see memory project-pmlb-tuning-holdout).

Protocol
--------
* Search on the `tune` fold (13 datasets). For each dataset, evaluate the DEFAULT
  config plus N random configs, each averaged over `--seeds` fresh 75/25 splits
  (the model carves its own early-stopping validation from the 75% train; we score
  macro-F1 for classification / RMSE for regression on the 25% test).
* Per dataset, every config's score is turned into `rel_impr` = signed fractional
  improvement over THAT dataset's default (positive = better than default). This
  removes the dataset scale so configs can be pooled across the suite.
* Knob importance: within-dataset Spearman corr(knob value, rel_impr), averaged
  across datasets (avoids Simpson's paradox). Plus per-value mean rel_impr.
* The single best SHARED config (best mean rel_impr on tune) is then re-evaluated
  on the `holdout` fold — the out-of-sample generalization check.

Only accuracy-relevant knobs are searched; cat_* knobs are skipped because PMLB
loads all-numeric (no categoricals detected). n_estimators is capped and
early_stopping_rounds fixed so every fit is compared in the same budget regime.
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
import run_benchmarks as RB  # noqa: E402  (reuse the PMLB dataset builders)
import chimeraboost as cb    # noqa: E402

# Fixed budget so every config is judged in the same regime.
N_ESTIMATORS = 1500
EARLY_STOP_ROUNDS = 50


def sample_config(rng):
    """One random ChimeraBoost config over the accuracy-relevant knobs."""
    return dict(
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.30)))),
        depth=int(rng.integers(4, 11)),               # 4..10 inclusive
        l2_leaf_reg=float(np.exp(rng.uniform(np.log(0.3), np.log(10.0)))),
        max_bins=int(rng.choice([64, 128, 254])),
        subsample=float(rng.choice([0.6, 0.8, 1.0])),
        colsample=float(rng.choice([0.6, 0.8, 1.0])),
        min_child_weight=float(rng.choice([0.0, 1.0, 5.0, 20.0])),
        leaf_estimation_iterations=int(rng.choice([1, 3, 5, 10])),
        ordered_boosting=bool(rng.choice([False, True])),
        hs_lambda=float(rng.choice([0.0, 0.5, 1.0])),
    )


# Knobs treated as ordinal/continuous for the correlation analysis (ordered_boosting
# is binary and handled separately).
NUMERIC_KNOBS = ["learning_rate", "depth", "l2_leaf_reg", "max_bins", "subsample",
                 "colsample", "min_child_weight", "leaf_estimation_iterations",
                 "hs_lambda"]


def _fit_score(task, X, y, cat, cfg, seed, threads):
    """Fit one config on one split, return the held-out metric (F1 macro / RMSE)."""
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat)
    common = dict(thread_count=threads, random_state=seed,
                  n_estimators=N_ESTIMATORS, early_stopping=True,
                  early_stopping_rounds=EARLY_STOP_ROUNDS, cat_features=cat)
    if cfg:
        common.update(cfg)
    if task == "regression":
        m = cb.ChimeraBoostRegressor(**common)
        m.fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    m = cb.ChimeraBoostClassifier(**common)
    m.fit(Xtr, ytr)
    return float(f1_score(yte, m.predict(Xte), average="macro"))


def _task(ds):
    return RB._task_of(ds)


# ----- worker (top-level + picklable for ProcessPoolExecutor) -----------------
def _eval_task(t):
    ds, cfg_id, cfg, seed, scale, threads = t
    RB._add_pmlb_datasets()
    rng = np.random.default_rng(1000 + seed)
    X, y, cat, task = RB.DATASETS[ds](scale, rng)
    t0 = time.time()
    score = _fit_score(task, X, y, cat, cfg, seed, threads)
    return ds, cfg_id, seed, score, time.time() - t0


def run(fold, n_configs, seeds, jobs, threads_per, out_path, configs=None):
    RB._add_pmlb_datasets()
    datasets = [k for k in RB.DATASETS if k.startswith(f"pm:{fold}/")]
    rng = np.random.default_rng(0)
    if configs is None:
        configs = {"default": None}
        for i in range(n_configs):
            configs[f"cfg{i:03d}"] = sample_config(rng)

    tasks = [(ds, cid, cfg, s, 1.0, threads_per)
             for ds in datasets for cid, cfg in configs.items()
             for s in range(seeds)]
    print(f"[{fold}] {len(datasets)} datasets x {len(configs)} configs x "
          f"{seeds} seeds = {len(tasks)} fits  (jobs={jobs})")

    raw = defaultdict(lambda: defaultdict(list))  # raw[ds][cid] = [scores]
    done = 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_eval_task, t) for t in tasks]
        for fut in as_completed(futs):
            ds, cid, seed, score, secs = fut.result()
            raw[ds][cid].append(score)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)}  elapsed {el:.0f}s  eta {eta:.0f}s",
                      flush=True)

    # mean over seeds
    scores = {ds: {cid: float(np.mean(v)) for cid, v in cfgs.items()}
              for ds, cfgs in raw.items()}
    payload = {"fold": fold, "seeds": seeds, "n_estimators": N_ESTIMATORS,
               "configs": {k: v for k, v in configs.items()},
               "tasks_of": {ds: _task(ds) for ds in datasets},
               "scores": scores}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {out_path}")
    return payload


def rel_impr(task, default, score):
    """Signed fractional improvement over default (positive = better)."""
    if default == 0:
        return 0.0
    if task == "regression":     # RMSE, lower better
        return (default - score) / default
    return (score - default) / default   # F1, higher better


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]):
        return np.nan
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d else np.nan


def analyze(payload):
    tasks_of = payload["tasks_of"]; configs = payload["configs"]
    scores = payload["scores"]
    # rel_impr[ds][cid]
    rel = {}
    for ds, cfgs in scores.items():
        d = cfgs["default"]; t = tasks_of[ds]
        rel[ds] = {cid: rel_impr(t, d, s) for cid, s in cfgs.items() if cid != "default"}

    rng_cids = [c for c in configs if c != "default"]

    # ---- per-dataset headroom: best random config vs default ----
    print("\n=== Per-dataset tuning headroom (best random config vs default) ===")
    print(f"{'dataset':40} {'task':11} {'best rel_impr':>13} {'best cfg':>9}")
    headroom = []
    for ds in sorted(rel):
        best_cid = max(rel[ds], key=rel[ds].get)
        bi = rel[ds][best_cid]
        headroom.append(bi)
        print(f"{ds:40} {tasks_of[ds]:11} {bi*100:+12.2f}% {best_cid:>9}")
    print(f"{'MEAN best-per-dataset headroom':40} {'':11} {np.mean(headroom)*100:+12.2f}%")

    # ---- knob importance: within-dataset Spearman(value, rel_impr), averaged ----
    print("\n=== Knob importance (mean within-dataset Spearman corr with rel_impr) ===")
    print("    positive => larger value tends to beat default; |corr| => strength")
    rows = []
    for knob in NUMERIC_KNOBS:
        corrs = []
        for ds in rel:
            vals = [configs[c][knob] for c in rng_cids]
            ri = [rel[ds][c] for c in rng_cids]
            c = _spearman(vals, ri)
            if not np.isnan(c):
                corrs.append(c)
        rows.append((knob, np.mean(corrs) if corrs else np.nan))
    for knob, mc in sorted(rows, key=lambda r: -abs(r[1] if not np.isnan(r[1]) else 0)):
        print(f"  {knob:28} {mc:+.3f}")
    # ordered_boosting (binary): mean rel_impr on vs off, pooled across datasets
    on = [rel[ds][c] for ds in rel for c in rng_cids if configs[c]["ordered_boosting"]]
    off = [rel[ds][c] for ds in rel for c in rng_cids if not configs[c]["ordered_boosting"]]
    print(f"  {'ordered_boosting=True':28} mean rel_impr {np.mean(on)*100:+.2f}% "
          f"(vs False {np.mean(off)*100:+.2f}%)")

    # ---- per-value marginal: mean rel_impr for each discrete knob value ----
    print("\n=== Marginal: mean rel_impr by knob value (pooled; 0% = ties default) ===")
    discrete = {"max_bins": [64, 128, 254], "subsample": [0.6, 0.8, 1.0],
                "colsample": [0.6, 0.8, 1.0], "min_child_weight": [0.0, 1.0, 5.0, 20.0],
                "leaf_estimation_iterations": [1, 3, 5, 10], "hs_lambda": [0.0, 0.5, 1.0]}
    for knob, vals in discrete.items():
        parts = []
        for v in vals:
            ri = [rel[ds][c] for ds in rel for c in rng_cids if configs[c][knob] == v]
            parts.append(f"{v}:{np.mean(ri)*100:+.2f}%" if ri else f"{v}:NA")
        print(f"  {knob:28} " + "  ".join(parts))

    # ---- best single SHARED config across the tune fold ----
    print("\n=== Best single shared config (mean rel_impr across tune datasets) ===")
    mean_ri = {c: np.mean([rel[ds][c] for ds in rel]) for c in rng_cids}
    best_shared = max(mean_ri, key=mean_ri.get)
    print(f"  {best_shared}: mean rel_impr {mean_ri[best_shared]*100:+.2f}%")
    for k, v in configs[best_shared].items():
        print(f"    {k:28} {v}")
    return best_shared, configs[best_shared]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="tune", choices=["tune", "holdout"])
    ap.add_argument("--n-configs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads-per", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None,
                    help="path to an existing results JSON: skip search, just analyze")
    args = ap.parse_args()

    if args.analyze:
        analyze(json.load(open(args.analyze)))
        return

    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(os.path.dirname(__file__), "results",
                                   f"rsearch-{args.fold}-{stamp}.json")
    payload = run(args.fold, args.n_configs, args.seeds, args.jobs,
                  args.threads_per, out)
    analyze(payload)
    print(f"\n# results JSON: {out}")


if __name__ == "__main__":
    main()
