"""Probe: WHICH split-score machinery explains CatBoost's edge on the gap sets?

On the 20260922-144219 run CatBoost beats our default on a few Grinsztajn
binary sets (california +2.7%, california@sus50 +2.7%, bank-marketing +1.0%,
credit +0.5%, albert@sus25 +0.45%, eye_movements@sus25 +0.30% of Brier) while
we beat it on the controls (MagicTelescope, electricity). This probe ablates
the OPPONENT: CatBoost at its defaults, then with its split machinery switched
to our settings all at once (cb_ours: L2 split score, no leaf backtracking,
quantile border grid, lambda 1.0, 128 bins). Two of our own settings move too
(constant leaves, 254 bins) on the same splits.

Stage A (this task): chimera, chimera_const, chimera_bins254, cb_default,
cb_ours. Stage B (one-factor-at-a-time CatBoost arms) is defined below but
runs ONLY with --stage b.

PROTOCOL: the harness's own data path (run_benchmarks builders, 75/25 split,
@sus train shrink), 3 seeds, threads=2 like the saved run, Brier in the
harness's K-sum form. Resumable JSONL; self-check against the saved run, then
an aggregate table.
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
RESULTS = os.path.join(HERE, "results", "probe-cb-split-score.jsonl")
SAVED_RUN = os.path.join(HERE, "results", "20260922-144219.json")

GAP = ["gr:clf_num/california", "gr:clf_num/california@sus50",
       "gr:clf_num/bank-marketing", "gr:clf_num/credit",
       "gr:clf_cat/albert@sus25", "gr:clf_cat/eye_movements@sus25"]
CONTROL = ["gr:clf_num/MagicTelescope", "gr:clf_num/electricity"]
SEEDS = (0, 1, 2)
THREADS = 2  # saved run's per-job count (12 cores / 5 jobs)
MAX_BINS_254 = 254  # accepted: library validates 2..65534

# The dict main() builds when NO --chimera-* flag is given, reproduced
# literally from the argparse defaults (run_benchmarks.py ~line 2203):
# lr=None, ob None (ordered_boosting defaults True, force_ordered False),
# depth=6, subsample=1.0, colsample=None, mcw=None, cat_combinations=False,
# cat_count_features=False (class default applies), cat_smoothing=None,
# leaf_estimation_iterations=None, linear_leaves=False, linear_lambda=1.0,
# cross_features=False, selection_rounds=None, quantize=False, refit_full=False.
CHIMERA_CFG = dict(lr=None, ordered_boosting=None, depth=6, subsample=1.0,
                   colsample=None, mcw=None, cat_combinations=False,
                   cat_count_features=False, cat_smoothing=None,
                   leaf_estimation_iterations=None, linear_leaves=False,
                   linear_lambda=1.0, cross_features=False,
                   selection_rounds=None, quantize=False, refit_full=False)

CB_OURS_OVERRIDES = {"score_function": "L2",
                     "leaf_estimation_backtracking": "No",
                     "feature_border_type": "Median",
                     "l2_leaf_reg": 1.0,
                     "border_count": 128}

# Stage B: one factor at a time. Defined now, run only with --stage b.
STAGE_B = {
    "cb_l2score": {"score_function": "L2"},
    "cb_nobacktrack": {"leaf_estimation_backtracking": "No"},
    "cb_median": {"feature_border_type": "Median"},
    "cb_lambda1": {"l2_leaf_reg": 1.0},
    "cb_border128": {"border_count": 128},
}

ARMS_A = ("chimera", "chimera_const", "chimera_bins254", "cb_default", "cb_ours")


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
    return metrics["brier"], fit_s, best


def _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat, max_bins=None):
    """_run_chimera's estimator construction, written out so max_bins can vary.

    _run_chimera builds an empty kw for the default arm, so this is the same
    estimator; the equivalence check in main() proves it on the first row.
    """
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    kw = {}
    if max_bins is not None:
        kw["max_bins"] = max_bins
    t = time.time()
    m = Est(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
            learning_rate=None, depth=6, subsample=1.0, colsample=None,
            thread_count=THREADS, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    metrics, fit_s, _ = rb._finish(task, yte, m, Xte, t)
    return metrics["brier"], fit_s, m.best_iteration_


def _run_catboost_ov(task, Xtr, ytr, Xte, yte, cat, overrides):
    """A copy of rb._run_catboost with `overrides` merged into the params last."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    Xf, Xv, yf, yv = rb._val_split(Xtr, ytr, task, 0)
    t = time.time()
    params = dict(allow_writing_files=False,
                  n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
                  thread_count=THREADS, verbose=False, random_seed=0)
    params.update(overrides)
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = Est(**params)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    metrics, fit_s, _ = rb._finish(task, yte, m, Xte, t)
    return metrics["brier"], fit_s, m.best_iteration_


def _run_arm(arm, task, Xtr, ytr, Xte, yte, cat):
    if arm == "chimera":
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat)
    if arm == "chimera_const":
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat,
                                linear_leaves="off")
    if arm == "chimera_bins254":
        b, fit_s, best = _run_chimera_direct(
            task, Xtr, ytr, Xte, yte, cat, max_bins=MAX_BINS_254)
        return b, fit_s, best
    if arm == "cb_default":
        return _run_catboost_ov(task, Xtr, ytr, Xte, yte, cat, {})
    if arm == "cb_ours":
        return _run_catboost_ov(task, Xtr, ytr, Xte, yte, cat,
                                CB_OURS_OVERRIDES)
    return _run_catboost_ov(task, Xtr, ytr, Xte, yte, cat, STAGE_B[arm])


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


def _saved_briers():
    with open(SAVED_RUN, "r", encoding="utf-8") as f:
        saved = json.load(f)
    out = {}
    for r in saved["records"]:
        if r["model"] in ("ChimeraBoost", "CatBoost"):
            b = r["metrics"].get("brier")  # regression rows have none
            if b is not None:
                out[(r["dataset"], r["model"], r["seed"])] = b
    return out


def self_check(keys, seeds):
    """Probe chimera/cb_default next to the saved run's ChimeraBoost/CatBoost."""
    rows = _read_jsonl()
    probe = {(r["dataset"], r["arm"], r["seed"]): r["brier"] for r in rows}
    saved = _saved_briers()
    print("\nSELF-CHECK — probe vs saved run Brier (exact pairing expected)")
    print(f"{'dataset':36s} {'s':>2s} {'chimera':>12s} {'saved':>12s} "
          f"{'cb_default':>12s} {'saved':>12s}")
    diffs = {"chimera": [], "cb_default": []}
    worst = {"chimera": None, "cb_default": None}
    missing = []
    for key in keys:
        for seed in seeds:
            pc = probe.get((key, "chimera", seed))
            sc = saved.get((key, "ChimeraBoost", seed))
            pb = probe.get((key, "cb_default", seed))
            sb = saved.get((key, "CatBoost", seed))
            if None in (pc, sc, pb, sb):
                missing.append((key, seed))
                print(f"{key.replace('gr:', '')[:36]:36s} {seed:2d} "
                      f"{'--missing--':>52s}")
                continue
            dc, db = abs(pc - sc), abs(pb - sb)
            diffs["chimera"].append(dc)
            diffs["cb_default"].append(db)
            if worst["chimera"] is None or dc > worst["chimera"][0]:
                worst["chimera"] = (dc, key, seed, pc, sc)
            if worst["cb_default"] is None or db > worst["cb_default"][0]:
                worst["cb_default"] = (db, key, seed, pb, sb)
            print(f"{key.replace('gr:', '')[:36]:36s} {seed:2d} "
                  f"{pc:12.8f} {sc:12.8f} {pb:12.8f} {sb:12.8f}")
    max_c = max(diffs["chimera"]) if diffs["chimera"] else float("nan")
    max_b = max(diffs["cb_default"]) if diffs["cb_default"] else float("nan")
    print(f"max|diff| chimera={max_c:.3e}  cb_default={max_b:.3e}")
    ok = (not missing and max_c < 1e-9 and max_b < 1e-9)
    if ok:
        print("SELF-CHECK PASS")
    else:
        print("SELF-CHECK FAIL")
        for arm in ("chimera", "cb_default"):
            w = worst[arm]
            if w is not None:
                print(f"  worst {arm}: |diff|={w[0]:.3e} "
                      f"{w[1]} seed {w[2]} probe={w[3]:.8f} saved={w[4]:.8f}")
        for key, seed in missing:
            print(f"  missing row: {key} seed {seed}")
    return ok


def _agree(per_seed_ours, per_seed_arm, mean_change):
    """How many seeds agree in sign with the mean relative change."""
    n = 0
    for s in per_seed_ours:
        if s in per_seed_arm:
            d = per_seed_ours[s] - per_seed_arm[s]
            if (d > 0) == (mean_change > 0) or d == mean_change == 0:
                n += 1
    return n


def _mean_brier(agg, ds, arm, seeds):
    v = [agg[(ds, arm)][s] for s in seeds if s in agg[(ds, arm)]]
    return float(np.mean(v)) if v else None


def _key_stats(agg, ds, seeds):
    """Per-key means, edges and relative changes, or None when incomplete."""
    arms = {"ours": "chimera", "const": "chimera_const",
            "bins": "chimera_bins254", "cbdef": "cb_default",
            "cbours": "cb_ours"}
    st = {k: _mean_brier(agg, ds, a, seeds) for k, a in arms.items()}
    if any(v is None for v in st.values()):
        return None
    ours = st["ours"]
    st["edge_def"] = (ours - st["cbdef"]) / ours
    edge_ours = (ours - st["cbours"]) / ours
    st["recov"] = (1.0 - edge_ours / st["edge_def"]
                   if abs(st["edge_def"]) >= 0.001 else None)
    st["dc"] = (ours - st["const"]) / ours
    st["db"] = (ours - st["bins"]) / ours
    st["ac"] = _agree(agg[(ds, "chimera")], agg[(ds, "chimera_const")],
                      st["dc"])
    st["ab"] = _agree(agg[(ds, "chimera")], agg[(ds, "chimera_bins254")],
                      st["db"])
    st["ns"] = len([s for s in seeds if s in agg[(ds, "chimera")]])
    return st


def _recov_s(recov):
    return f"{100 * recov:7.1f}%" if recov is not None else "     n/a"


def _print_key_row(ds, st):
    print(f"{ds.replace('gr:', '')[:32]:32s}{st['ours']:11.6f}"
          f"{st['const']:11.6f}{st['bins']:11.6f}{st['cbdef']:11.6f}"
          f"{st['cbours']:11.6f}{100 * st['edge_def']:8.2f}%"
          f"{_recov_s(st['recov'])}"
          f"{100 * st['dc']:+7.2f}%({st['ac']}/{st['ns']})"
          f"{100 * st['db']:+7.2f}%({st['ab']}/{st['ns']})")


def _print_group(agg, keys, group, name, seeds):
    from collections import defaultdict
    print(f"\n-- {name} " + "-" * (128 - len(name)))
    cols = defaultdict(list)
    for ds in [k for k in keys if k in group]:
        st = _key_stats(agg, ds, seeds)
        if st is None:
            print(f"{ds.replace('gr:', '')[:32]:32s} -- incomplete --")
            continue
        _print_key_row(ds, st)
        for k in ("ours", "const", "bins", "cbdef", "cbours", "edge_def",
                  "dc", "db", "ac", "ab"):
            cols[k].append(st[k])
        if st["recov"] is not None:
            cols["recov"].append(st["recov"])
    return {k: float(np.median(v)) for k, v in cols.items() if v}


def _print_medians(med, groups, seeds):
    print("\n-- GROUP MEDIANS " + "-" * 113)
    print(f"{'group':32s}{'chimera':>11s}{'const':>11s}{'bins254':>11s}"
          f"{'cb_def':>11s}{'cb_ours':>11s}{'edge_def':>9s}{'recov':>8s}"
          f"{'d_const':>12s}{'d_bins':>12s}")
    for _, name in groups:
        m = med.get(name, {})
        if not m:
            continue
        print(f"{name[:32]:32s}{m['ours']:11.6f}{m['const']:11.6f}"
              f"{m['bins']:11.6f}{m['cbdef']:11.6f}{m['cbours']:11.6f}"
              f"{100 * m['edge_def']:8.2f}%{_recov_s(m.get('recov'))}"
              f"{100 * m['dc']:+7.2f}%({m['ac']:g}/{len(seeds)})"
              f"{100 * m['db']:+7.2f}%({m['ab']:g}/{len(seeds)})")


def _print_stage_b(agg, keys, arms, seeds):
    print("\nStage B arms (one factor at a time):")
    for arm in arms:
        if arm in ARMS_A:
            continue
        print(f"  {arm}: {STAGE_B[arm]}")
        for ds in keys:
            v = _mean_brier(agg, ds, arm, seeds)
            ours = _mean_brier(agg, ds, "chimera", seeds)
            if v is None or ours is None:
                continue
            print(f"    {ds.replace('gr:', '')[:30]:30s} "
                  f"{v:.6f}  edge {(100 * (ours - v) / ours):+6.2f}%")


def table(keys, seeds, arms):
    from collections import defaultdict
    rows = _read_jsonl()
    agg = defaultdict(dict)
    for r in rows:
        if r["dataset"] in keys and r["seed"] in seeds:
            agg[(r["dataset"], r["arm"])][r["seed"]] = r["brier"]
    print("\n" + "=" * 132)
    print("CATBOOST SPLIT-SCORE PROBE — mean Brier over seeds (lower better). "
          "edge = (ours-cb)/ours, positive = CatBoost ahead.")
    print("=" * 132)
    print(f"{'dataset':32s}{'chimera':>11s}{'const':>11s}{'bins254':>11s}"
          f"{'cb_def':>11s}{'cb_ours':>11s}{'edge_def':>9s}{'recov':>8s}"
          f"{'d_const':>12s}{'d_bins':>12s}")
    groups = ((GAP, "GAP (CatBoost ahead in saved run)"),
              (CONTROL, "CONTROL (we ahead in saved run)"))
    med = {name: _print_group(agg, keys, group, name, seeds)
           for group, name in groups}
    _print_medians(med, groups, seeds)
    if any(a not in ARMS_A for a in arms):
        _print_stage_b(agg, keys, arms, seeds)


def _parse_cli(args):
    table_only = "--table-only" in args
    smoke = "--smoke" in args or os.environ.get("PROBE_SMOKE") == "1"
    stage_b = "--stage" in args and args[args.index("--stage") + 1:][:1] == ["b"]
    if "--stage" in args and not stage_b:
        sys.exit("only --stage b is supported")
    arms = list(ARMS_A) + (list(STAGE_B) if stage_b else [])
    keys = GAP + CONTROL
    seeds = SEEDS
    if smoke:
        keys, seeds = ["gr:clf_num/credit"], (0,)
    return table_only, arms, keys, seeds


def _equiv_check(keys, seeds):
    """Prove the direct estimator construction matches the chimera arm.

    Runs on the first row before the run proper starts; returns the
    reference fit so the main loop can keep it instead of refitting.
    """
    key0, seed0 = keys[0], seeds[0]
    Xtr, Xte, ytr, yte, cat, task = _load_split(key0, seed0)
    b_ref, fit_ref, best_ref = _run_chimera_arm(
        task, Xtr, ytr, Xte, yte, cat)
    b_dir, _, _ = _run_chimera_direct(task, Xtr, ytr, Xte, yte, cat)
    print(f"EQUIV-CHECK direct-vs-chimera on {key0} seed {seed0}: "
          f"{b_ref:.10f} vs {b_dir:.10f} diff={abs(b_ref - b_dir):.3e}")
    if b_ref != b_dir:
        sys.exit("EQUIV-CHECK FAIL: aborting, the bins254 path is invalid")
    print("EQUIV-CHECK PASS")
    return key0, seed0, b_ref, fit_ref, best_ref


def _run_missing(keys, seeds, arms, done):
    for key in keys:
        for seed in seeds:
            Xtr, Xte, ytr, yte, cat, task = _load_split(key, seed)
            if seed == seeds[0]:
                ncat = len(cat) if cat else 0
                print(f"\n=== {key}  n={len(ytr) + len(yte)} "
                      f"n_tr={len(ytr)} p={np.shape(Xtr)[1]} cats={ncat} "
                      f"task={task}", flush=True)
            for arm in arms:
                if (key, arm, seed) in done:
                    continue
                b, fit_s, best = _run_arm(arm, task, Xtr, ytr, Xte, yte, cat)
                rec = {"dataset": key, "arm": arm, "seed": seed,
                       "brier": b, "fit_time": fit_s, "best_iter": best}
                if arm == "chimera_bins254":
                    rec["max_bins"] = MAX_BINS_254
                _append(rec)
                done.add((key, arm, seed))
                print(f"  [{arm:14s} s{seed}] brier={b:.6f}  "
                      f"{fit_s:6.1f}s  it={best}", flush=True)


def main():
    table_only, arms, keys, seeds = _parse_cli(sys.argv[1:])
    rb._add_grinsztajn_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    if table_only:
        self_check(keys, seeds)
        table(keys, seeds, arms)
        return
    key0, seed0, b_ref, fit_ref, best_ref = _equiv_check(keys, seeds)
    done = _done_keys(_read_jsonl())
    if (key0, "chimera", seed0) not in done:
        # The reference fit above already computed this row; keep it.
        _append({"dataset": key0, "arm": "chimera", "seed": seed0,
                 "brier": b_ref, "fit_time": fit_ref, "best_iter": best_ref})
        done.add((key0, "chimera", seed0))
        print(f"  [{'chimera':14s} s{seed0}] brier={b_ref:.6f}  "
              f"{fit_ref:6.1f}s  it={best_ref}", flush=True)
    _run_missing(keys, seeds, arms, done)
    self_check(keys, seeds)
    table(keys, seeds, arms)


if __name__ == "__main__":
    main()
