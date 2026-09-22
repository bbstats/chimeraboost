"""Probe: why does every opponent beat us on gr:reg_num/cpu_act@sus25?

On cpu_act trained on 25% of its rows (1,536 training rows) every opponent
beats our default by 20-28% of RMSE, while on the full cpu_act we beat them
all; our bagged mode is worse still, so it is not variance. This probe asks
which of OUR settings causes it -- constant leaves, the full-data refit, the
adaptive learning rate, or 128 bins already too fine at n=1536 (BARRIERS.md
B20 records cpu_act as the set that overfits when bins get FINER) -- and
where in the test set the extra error sits: a heavy tail of rows, or test
rows whose inputs fall outside the training range, where linear leaves
extrapolate (cpu_act's inputs are heavy-tailed).

PROTOCOL: the harness's own data path (run_benchmarks builders, 75/25 split,
@sus train shrink), all 7 Grinsztajn regression @sus25 keys plus the cpu_act
parent, 3 seeds, threads=2 like the saved run, RMSE. Resumable JSONL plus a
per-row-stats JSON; self-check against the saved run, then two tables.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-small-reg-loss.jsonl")
ROWS = os.path.join(HERE, "results", "probe-small-reg-loss-rows.json")
SAVED_RUN = os.path.join(HERE, "results", "20260922-144219.json")

CPU_SUS = "gr:reg_num/cpu_act@sus25"
CPU_FULL = "gr:reg_num/cpu_act"
SEEDS = (0, 1, 2)
THREADS = 2  # saved run's per-job count (12 cores / 5 jobs)
MAX_BINS_64 = 64  # accepted: library validates 2..65534
CAPTURE_TOL = 1e-12  # max rel RMSE gap between official and capture fits
SELF_CHECK_TOL = 1e-9  # max rel RMSE gap vs the saved run

# The dict main() builds when NO --chimera-* flag is given, reproduced
# literally from the argparse defaults (run_benchmarks.py ~line 2203):
# lr=None, ob None (ordered_boosting defaults True, force_ordered False),
# depth=6, subsample=1.0, colsample=None, mcw=None, cat_combinations=False,
# cat_count_features=False (class default applies), cat_smoothing=None,
# leaf_estimation_iterations=None, linear_leaves=False, linear_lambda=1.0,
# cross_features=False, selection_rounds=None, quantize=False, refit_full=False.
# Matches the saved run's provenance chimera_cfg exactly.
CHIMERA_CFG = dict(lr=None, ordered_boosting=None, depth=6, subsample=1.0,
                   colsample=None, mcw=None, cat_combinations=False,
                   cat_count_features=False, cat_smoothing=None,
                   leaf_estimation_iterations=None, linear_leaves=False,
                   linear_lambda=1.0, cross_features=False,
                   selection_rounds=None, quantize=False, refit_full=False)

ARMS = ("chimera", "chimera_const", "chimera_norefit", "chimera_fixedlr",
        "chimera_bins64", "lgbm", "catboost")
ROW_KEYS = (CPU_SUS, CPU_FULL)
ROW_ARMS = ("chimera", "chimera_const", "chimera_bins64", "lgbm")
ROW_META = {"top_fracs": [0.01, 0.05], "capture_tol": CAPTURE_TOL,
            "oor": "inputs outside the training [min, max] per column"}


def _dataset_keys():
    sus = sorted(k for k in rb.DATASETS
                 if k.startswith("gr:reg") and k.endswith("@sus25"))
    if len(sus) != 7:
        print(f"WARNING: expected 7 gr:reg @sus25 keys, found {len(sus)}: {sus}")
    if CPU_FULL not in rb.DATASETS:
        sys.exit(f"parent key {CPU_FULL} is not registered")
    return sus + [CPU_FULL]


def _load_split(key, seed):
    """The harness's data path (_run_seed_task): build, split 75/25, @sus shrink."""
    rng = np.random.default_rng(1000 + seed)
    X, y, cat, task = rb.DATASETS[key](1.0, rng)
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat)
    variant = key.split(rb.VARIANT_SEP, 1)[1] if rb.VARIANT_SEP in key else ""
    if variant in rb.SUS_FRACTIONS:
        Xtr, ytr = rb._subsample_train(Xtr, ytr, rb.SUS_FRACTIONS[variant], task)
    return Xtr, Xte, ytr, yte, cat, task


def _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat, **kw):
    cfg = dict(CHIMERA_CFG)
    cfg.update(kw)
    out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, THREADS, **cfg)
    metrics, fit_s, _, best = out
    return metrics["rmse"], fit_s, best


def _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat, max_bins=None,
                        linear_off=False, want_preds=False):
    """_run_chimera's estimator construction, written out so max_bins can vary.

    _run_chimera builds an empty kw for the default arm, so this is the same
    estimator; the equivalence check in main() proves it on the first row.
    linear_off=True reproduces the chimera_const arm (kw linear_leaves=False);
    want_preds=True also returns the test-row predictions for the row read.
    """
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    kw = {}
    if max_bins is not None:
        kw["max_bins"] = max_bins
    if linear_off:
        kw["linear_leaves"] = False
    t = time.time()
    m = Est(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
            learning_rate=None, depth=6, subsample=1.0, colsample=None,
            thread_count=THREADS, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    metrics, fit_s, _ = rb._finish(task, yte, m, Xte, t)
    if want_preds:
        return (metrics["rmse"], fit_s, m.best_iteration_,
                np.asarray(m.predict(Xte), dtype=float))
    return metrics["rmse"], fit_s, m.best_iteration_


def _run_lgbm_cap(task, Xtr, ytr, Xte, yte, cat):
    """A copy of rb._run_lightgbm that also returns the test-row predictions.

    The fit path is identical (same val split, encoding, params, _finish), so
    the RMSE matches the official lgbm arm by construction; the row phase
    asserts that per row before using the predictions.
    """
    import lightgbm as lgb
    Xf, Xv, yf, yv = rb._val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=rb.MAX_ITERS, n_jobs=THREADS,
                  random_state=0, verbosity=-1)
    fit_kw = dict(callbacks=[lgb.early_stopping(rb.PATIENCE, verbose=False)])
    if cat is not None:
        Xf_in, Xv_in, Xte_in = rb._lgb_prepare(Xf, Xv, Xte, list(cat))
        fit_kw["categorical_feature"] = list(cat)
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte
    fit_kw["eval_set"] = [(Xv_in, yv)]
    Est = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    m = Est(**common)
    m.fit(Xf_in, yf, **fit_kw)
    metrics, fit_s, _ = rb._finish(task, yte, m, Xte_in, t)
    return (metrics["rmse"], fit_s, m.best_iteration_,
            np.asarray(m.predict(Xte_in), dtype=float))


def _run_opponent(name, task, Xtr, ytr, Xte, yte, cat):
    out = rb.RUNNERS[name](task, Xtr, ytr, Xte, yte, cat, THREADS)
    if out is None:
        sys.exit(f"{name} is not installed; cannot run that arm")
    metrics, fit_s, _, best = out
    return metrics["rmse"], fit_s, best


def _run_arm(arm, task, Xtr, ytr, Xte, yte, cat):
    if arm == "chimera":
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat)
    if arm == "chimera_const":
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat,
                                linear_leaves="off")
    if arm == "chimera_norefit":
        # cfg CAN express "skip the refit": refit_full="off" forces the
        # library flag False (same route as ChimeraBoostNoRefit).
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat,
                                refit_full="off")
    if arm == "chimera_fixedlr":
        # _run_chimera HAS a route: adaptive_lr="off" forces
        # adaptive_learning_rate=False (same as ChimeraBoostFlatLR).
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat,
                                adaptive_lr="off")
    if arm == "chimera_bins64":
        rmse, fit_s, best = _run_chimera_direct(
            task, Xtr, ytr, Xte, yte, cat, max_bins=MAX_BINS_64)
        return rmse, fit_s, best
    if arm == "lgbm":
        return _run_opponent("LightGBM", task, Xtr, ytr, Xte, yte, cat)
    if arm == "catboost":
        return _run_opponent("CatBoost", task, Xtr, ytr, Xte, yte, cat)
    sys.exit(f"unknown arm {arm}")


def _rec(key, arm, seed, rmse, fit_s, best):
    rec = {"dataset": key, "arm": arm, "seed": seed,
           "rmse": rmse, "fit_time": fit_s}
    if best is not None:
        rec["best_iter"] = best
    if arm == "chimera_bins64":
        rec["max_bins"] = MAX_BINS_64
    return rec


def _read_jsonl():
    rows = []
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _done_keys(rows):
    return {(r["dataset"], r["arm"], r["seed"]) for r in rows}


def _append(rec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _read_rows():
    if os.path.exists(ROWS):
        with open(ROWS, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": dict(ROW_META), "data": {}}


def _write_rows(rows):
    rows.setdefault("meta", dict(ROW_META))
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    with open(ROWS, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _row_done(rows, key, seed, arm):
    return (rows.get("data", {}).get(key, {}).get(str(seed), {}).get(arm)
            is not None)


def _store_row(rows, key, seed, arm, stats):
    rows.setdefault("data", {}).setdefault(key, {}).setdefault(
        str(seed), {})[arm] = stats


def _row_stats(yte, preds, Xtr, Xte):
    """Top-tail SSE shares and the in-range / out-of-range SSE split."""
    se = (np.asarray(yte, dtype=float) - np.asarray(preds, dtype=float)) ** 2
    n = len(se)
    sse = float(se.sum())
    k1 = max(1, -(-n // 100))  # ceil(1% of n), at least one row
    k5 = max(1, -(-n // 20))  # ceil(5% of n), at least one row
    order = np.argsort(se, kind="stable")[::-1]
    top1 = float(se[order[:k1]].sum() / sse) if sse else 0.0
    top5 = float(se[order[:k5]].sum() / sse) if sse else 0.0
    Xr = np.asarray(Xtr, dtype=float)
    Xt = np.asarray(Xte, dtype=float)
    lo, hi = Xr.min(axis=0), Xr.max(axis=0)
    oor = ((Xt < lo) | (Xt > hi)).sum(axis=1)
    in_m = oor == 0
    n0, n1 = int(in_m.sum()), int((~in_m).sum())
    sse0, sse1 = float(se[in_m].sum()), float(se[~in_m].sum())
    return {"n_test": n, "n_train": int(np.shape(Xtr)[0]), "k1": k1, "k5": k5,
            "top1_share": top1, "top5_share": top5,
            "in_range": {"n": n0, "sse": sse0,
                         "rmse": float(np.sqrt(sse0 / n0)) if n0 else None},
            "out_of_range": {"n": n1, "sse": sse1,
                             "rmse": float(np.sqrt(sse1 / n1)) if n1 else None}}


def _capture(arm, task, Xtr, ytr, Xte, yte, cat):
    """Refit one arm returning (rmse, fit_s, best, test preds)."""
    if arm == "chimera_bins64":
        return _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat,
                                   max_bins=MAX_BINS_64, want_preds=True)
    if arm == "chimera_const":
        return _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat,
                                   linear_off=True, want_preds=True)
    if arm == "chimera":
        return _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat,
                                   want_preds=True)
    if arm == "lgbm":
        return _run_lgbm_cap(task, Xtr, ytr, Xte, yte, cat)
    sys.exit(f"unknown capture arm {arm}")


def _capture_seed(rows, probe, key, seed):
    arms = [a for a in ROW_ARMS if not _row_done(rows, key, seed, a)]
    if not arms:
        return
    Xtr, Xte, ytr, yte, cat, task = _load_split(key, seed)
    short = key.split("/")[-1][:20]
    for arm in arms:
        rmse, _, _, preds = _capture(arm, task, Xtr, ytr, Xte, yte, cat)
        ref = probe.get((key, arm, seed))
        if ref is None:
            sys.exit(f"capture for {key} {arm} s{seed} has no JSONL row")
        rel = abs(rmse - ref) / abs(ref)
        if rel >= CAPTURE_TOL:
            sys.exit(f"CAPTURE MISMATCH {key} {arm} s{seed}: "
                     f"{rmse:.10f} vs {ref:.10f}")
        stats = _row_stats(yte, preds, Xtr, Xte)
        stats["rmse"] = rmse
        _store_row(rows, key, seed, arm, stats)
        print(f"  [rows {arm:14s} {short:20s} s{seed}] "
              f"top1={stats['top1_share']:.1%} top5={stats['top5_share']:.1%} "
              f"in={stats['in_range']['n']} out={stats['out_of_range']['n']}",
              flush=True)
    _write_rows(rows)


def _collect_row_data(keys, seeds):
    rows = _read_rows()
    probe = {(r["dataset"], r["arm"], r["seed"]): r["rmse"]
             for r in _read_jsonl()}
    cpu = [k for k in keys if k in ROW_KEYS]
    if not any(not _row_done(rows, k, s, a)
               for k in cpu for s in seeds for a in ROW_ARMS):
        print("row data: complete, nothing to capture")
        return rows
    for key in cpu:
        for seed in seeds:
            _capture_seed(rows, probe, key, seed)
    return rows


def _saved_rmses():
    with open(SAVED_RUN, "r", encoding="utf-8") as f:
        saved = json.load(f)
    out = {}
    for r in saved["records"]:
        if r["model"] in ("ChimeraBoost", "LightGBM", "CatBoost"):
            v = r["metrics"].get("rmse")  # classification rows have none
            if v is not None:
                out[(r["dataset"], r["model"], r["seed"])] = v
    return out


def _rel_diff(probe, saved):
    return abs(probe - saved) / (abs(saved) if saved else 1.0)


def _check_cell(probe, saved, key, seed, pa, sa):
    p = probe.get((key, pa, seed))
    s = saved.get((key, sa, seed))
    if p is None or s is None:
        return None
    return (p, s, _rel_diff(p, s))


def _print_check_row(key, seed, cells):
    line = f"{key.replace('gr:', '')[:36]:36s} {seed:2d} "
    for cell in cells:
        if cell is None:
            line += f"{'--missing--':>12s} {'--missing--':>12s} "
        else:
            line += f"{cell[0]:12.6f} {cell[1]:12.6f} "
    print(line)


def _check_all(keys, seeds, pairs, probe, saved):
    recs = []
    for key in keys:
        for seed in seeds:
            cells = [_check_cell(probe, saved, key, seed, pa, sa)
                     for pa, sa in pairs]
            _print_check_row(key, seed, cells)
            recs.append((key, seed, cells))
    return recs


def _worst_of(worst, d, key, seed, pa, p, s):
    if worst is None or d > worst[0]:
        return (d, key, seed, pa, p, s)
    return worst


def _summarize_check(recs, pairs):
    diffs = []
    worst = None
    missing = []
    for key, seed, cells in recs:
        for cell, (pa, _) in zip(cells, pairs):
            if cell is None:
                continue
            diffs.append(cell[2])
            worst = _worst_of(worst, cell[2], key, seed, pa,
                              cell[0], cell[1])
        if any(c is None for c in cells):
            missing.append((key, seed))
    return diffs, worst, missing


def _report_check(diffs, worst, missing):
    mx = max(diffs) if diffs else float("nan")
    print(f"max|rel diff| = {mx:.3e} over {len(diffs)} cells "
          f"(tol {SELF_CHECK_TOL:.0e})")
    if not missing and diffs and mx < SELF_CHECK_TOL:
        print("SELF-CHECK PASS")
        return True
    print("SELF-CHECK FAIL")
    if worst is not None:
        print(f"  worst: |rel|={worst[0]:.3e} {worst[3]} {worst[1]} "
              f"seed {worst[2]} probe={worst[4]:.6f} saved={worst[5]:.6f}")
    for key, seed in missing:
        print(f"  missing row: {key} seed {seed}")
    return False


def self_check(keys, seeds):
    """Probe chimera/lgbm/catboost next to the saved run's RMSE."""
    rows = _read_jsonl()
    probe = {(r["dataset"], r["arm"], r["seed"]): r["rmse"] for r in rows}
    saved = _saved_rmses()
    pairs = (("chimera", "ChimeraBoost"), ("lgbm", "LightGBM"),
             ("catboost", "CatBoost"))
    print("\nSELF-CHECK - probe vs saved run RMSE (exact pairing expected)")
    print(f"{'dataset':36s} {'s':>2s} {'chimera':>12s} {'saved':>12s} "
          f"{'lgbm':>12s} {'saved':>12s} {'catboost':>12s} {'saved':>12s}")
    recs = _check_all(keys, seeds, pairs, probe, saved)
    diffs, worst, missing = _summarize_check(recs, pairs)
    return _report_check(diffs, worst, missing)


def _agree(per_seed_ours, per_seed_arm, mean_change):
    """How many seeds agree in sign with the mean relative change."""
    n = 0
    for s in per_seed_ours:
        if s in per_seed_arm:
            d = per_seed_ours[s] - per_seed_arm[s]
            if (d > 0) == (mean_change > 0) or d == mean_change == 0:
                n += 1
    return n


def _agg_rmse(keys, seeds):
    from collections import defaultdict
    agg = defaultdict(dict)
    for r in _read_jsonl():
        if r["dataset"] in keys and r["seed"] in seeds:
            agg[(r["dataset"], r["arm"])][r["seed"]] = r["rmse"]
    return agg


def _mean_rmse(agg, key, arm, seeds):
    v = [agg[(key, arm)][s] for s in seeds if s in agg[(key, arm)]]
    if len(v) != len(seeds):
        return None  # partial seed coverage reads as incomplete, not a mean
    return float(np.mean(v))


def _print_key_block(agg, key, seeds):
    means = {a: _mean_rmse(agg, key, a, seeds) for a in ARMS}
    if any(v is None for v in means.values()):
        print(f"{key}  -- incomplete --")
        return
    ours = means["chimera"]
    print(f"\n{key}")
    print(f"  {'arm':16s} {'mean_rmse':>12s} {'change':>9s} {'agree':>7s}")
    print(f"  {'chimera':16s} {ours:12.6f} {'--':>9s} {'--':>7s}")
    for arm in ARMS:
        if arm == "chimera":
            continue
        ch = (ours - means[arm]) / ours
        ag = _agree(agg[(key, "chimera")], agg[(key, arm)], ch)
        ns = len([s for s in seeds if s in agg[(key, arm)]])
        print(f"  {arm:16s} {means[arm]:12.6f} {100 * ch:+8.2f}% "
              f"{ag:>3d}/{ns:<3d}")


def table_a(keys, seeds):
    agg = _agg_rmse(keys, seeds)
    print("\n" + "=" * 72)
    print("TABLE A - mean RMSE over seeds (lower better); "
          "change vs chimera (+ = arm better)")
    print("=" * 72)
    for key in keys:
        _print_key_block(agg, key, seeds)


def _fmt_rmse(v):
    return f"{v:9.4f}" if v is not None else "       --"


def _print_row_line(seed, arm, st):
    i, o = st["in_range"], st["out_of_range"]
    print(f"  {seed:2d} {arm:14s} {st['rmse']:9.4f} "
          f"{100 * st['top1_share']:5.1f}% {100 * st['top5_share']:5.1f}% "
          f"{i['n']:6d} {i['sse']:12.1f} {_fmt_rmse(i['rmse'])} "
          f"{o['n']:6d} {o['sse']:12.1f} {_fmt_rmse(o['rmse'])}")


def table_b(keys, seeds):
    rows = _read_rows()
    data = rows.get("data", {})
    print("\n" + "=" * 112)
    print("TABLE B - per-row read: top1/top5 = share of test SSE in the worst "
          "1%/5% of rows;")
    print("in/out = test rows with 0 / >=1 inputs outside the training "
          "[min, max] (RMSE of each group per arm)")
    print("=" * 112)
    for key in [k for k in keys if k in ROW_KEYS]:
        print(f"\n{key}")
        print(f"  {'s':>2s} {'arm':14s} {'rmse':>9s} {'top1%':>6s} "
              f"{'top5%':>6s} {'in_n':>6s} {'in_sse':>12s} "
              f"{'in_rmse':>9s} {'out_n':>6s} {'out_sse':>12s} "
              f"{'out_rmse':>9s}")
        for seed in seeds:
            for arm in ROW_ARMS:
                st = data.get(key, {}).get(str(seed), {}).get(arm)
                if st is None:
                    print(f"  {seed:2d} {arm:14s} -- missing --")
                else:
                    _print_row_line(seed, arm, st)


def _parse_cli(args):
    table_only = "--table-only" in args
    smoke = "--smoke" in args or os.environ.get("PROBE_SMOKE") == "1"
    return table_only, smoke


def _equiv_check(keys, seeds):
    """Prove the direct estimator construction matches the chimera arms.

    Runs on the first row before the run proper starts; returns the
    reference fits so the main loop can keep them instead of refitting.
    """
    key0, seed0 = keys[0], seeds[0]
    Xtr, Xte, ytr, yte, cat, task = _load_split(key0, seed0)
    r_ref = _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat)
    r_dir = _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat)
    print(f"EQUIV-CHECK direct-vs-chimera on {key0} seed {seed0}: "
          f"{r_ref[0]:.10f} vs {r_dir[0]:.10f} "
          f"diff={abs(r_ref[0] - r_dir[0]):.3e}")
    if r_ref[0] != r_dir[0]:
        sys.exit("EQUIV-CHECK FAIL: aborting, the bins64 path is invalid")
    c_ref = _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat,
                             linear_leaves="off")
    c_dir = _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat,
                                linear_off=True)
    print(f"EQUIV-CHECK const-direct-vs-const on {key0} seed {seed0}: "
          f"{c_ref[0]:.10f} vs {c_dir[0]:.10f} "
          f"diff={abs(c_ref[0] - c_dir[0]):.3e}")
    if c_ref[0] != c_dir[0]:
        sys.exit("EQUIV-CHECK FAIL: aborting, the const capture path is invalid")
    print("EQUIV-CHECK PASS")
    return [(key0, seed0, "chimera", r_ref),
            (key0, seed0, "chimera_const", c_ref)]


def _run_missing(keys, seeds, done):
    for key in keys:
        for seed in seeds:
            Xtr, Xte, ytr, yte, cat, task = _load_split(key, seed)
            if seed == seeds[0]:
                ncat = len(cat) if cat else 0
                print(f"\n=== {key}  n={len(ytr) + len(yte)} "
                      f"n_tr={len(ytr)} p={np.shape(Xtr)[1]} cats={ncat} "
                      f"task={task}", flush=True)
            for arm in ARMS:
                if (key, arm, seed) in done:
                    continue
                rmse, fit_s, best = _run_arm(arm, task, Xtr, ytr, Xte, yte, cat)
                _append(_rec(key, arm, seed, rmse, fit_s, best))
                done.add((key, arm, seed))
                print(f"  [{arm:14s} s{seed}] rmse={rmse:.6f}  "
                      f"{fit_s:6.1f}s  it={best}", flush=True)


def main():
    table_only, smoke = _parse_cli(sys.argv[1:])
    rb._add_grinsztajn_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    if smoke:
        keys, seeds = [CPU_SUS], (0,)
    else:
        keys, seeds = _dataset_keys(), SEEDS
    print(f"keys ({len(keys)}): {keys}")
    print(f"seeds: {list(seeds)}  arms: {list(ARMS)}  threads: {THREADS}")
    if table_only:
        self_check(keys, seeds)
        table_a(keys, seeds)
        table_b(keys, seeds)
        return
    refs = _equiv_check(keys, seeds)
    done = _done_keys(_read_jsonl())
    for key, seed, arm, (rmse, fit_s, best) in refs:
        if (key, arm, seed) not in done:
            # The reference fit above already computed this row; keep it.
            _append(_rec(key, arm, seed, rmse, fit_s, best))
            done.add((key, arm, seed))
            print(f"  [{arm:14s} s{seed}] rmse={rmse:.6f}  "
                  f"{fit_s:6.1f}s  it={best}", flush=True)
    _run_missing(keys, seeds, done)
    _collect_row_data(keys, seeds)
    self_check(keys, seeds)
    table_a(keys, seeds)
    table_b(keys, seeds)


if __name__ == "__main__":
    main()

