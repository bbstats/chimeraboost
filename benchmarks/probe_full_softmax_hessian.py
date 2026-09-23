"""Probe s8-full-softmax-hessian: exact second-order vector-leaf step.

Multiclass fits grow one tree per round with K-vector leaves. Each leaf's
value is a per-class Newton step with the DIAGONAL softmax Hessian p_k(1-p_k)
scaled by (K-1)/K (booster._apply_vector_update -> tree._leaf_values_vec).
The true Hessian of softmax cross-entropy is diag(p) - p p^T (negative
off-diagonals). This probe tests the exact second-order step

    delta_l = -lr . (sum_{i in l} (diag(p_i) - p_i p_i^T) + lam I)^-1
              . sum_{i in l} g_i

with lam = l2_leaf_reg, against the diagonal step.

Arms: default (harness default ChimeraBoostClassifier) vs full_hessian (the
same fit with MulticlassBoosting._apply_vector_update monkeypatched to the
exact step; the only vector-leaf site -- the replay refit is scalar-only and
multiclass refits from scratch through the same loop, so the class patch
covers the refit too), plus the s8-lr-control arms lr_050 / lr_033 (the
unpatched default estimator at learning_rate 0.05 / 0.033, to separate
"smaller steps" from "better curvature"). --arms selects which arms to run
(default: all four); rows already in the JSONL are never re-run.

PROTOCOL: the harness's own data path (run_benchmarks builders, 75/25 split,
@sus train shrink, @time temporal splits), 3 seeds, threads=2 like the saved
run, Brier in the harness's K-sum form. Resumable JSONL; self-check against
the saved run, then per-panel tables.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb
import synthgen
from chimeraboost.booster import MulticlassBoosting
from chimeraboost.losses import _softmax
from chimeraboost.tree import _add_leaf_values

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-full-softmax-hessian.jsonl")
SAVED_RUN = os.path.join(HERE, "results", "20260922-144219.json")

SEEDS = (0, 1, 2)
THREADS = 2  # saved run's per-job count (12 cores / 5 jobs)
ARMS = ("default", "full_hessian", "lr_050", "lr_033")
# s8-lr-control: explicit learning rates for the slower-step control arms.
# Passed as rb._run_chimera(..., lr=...) -- _run_chimera forwards lr to the
# estimator's learning_rate (run_benchmarks.py line 1024).
LR_RATES = {"lr_050": 0.05, "lr_033": 0.033}
NEAR_SOLVED_BRIER = 0.001

# The dict main() builds when NO --chimera-* flag is given, copied from
# benchmarks/probe_catboost_split_score.py (verified): lr=None, ob None,
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

# ---- full-Hessian monkeypatch -----------------------------------------------
_ORIG_APPLY_VEC = MulticlassBoosting._apply_vector_update
_ORIG_FIT_IMPL = MulticlassBoosting._fit_impl
PATCH_CALLS = {"n": 0}


def _patched_apply_vector_update(self, tree, leaf, F, grad, hess, coupling, K):
    """Exact second-order vector-leaf step with the full softmax Hessian."""
    if self.ordered_boosting:
        raise RuntimeError("full_hessian probe: ordered_boosting is on")
    if float(self.subsample) < 1.0:
        raise RuntimeError("full_hessian probe: subsample<1 (MVS weights "
                           "out of scope)")
    PATCH_CALLS["n"] += 1
    # The same numerically stable softmax the loss uses (losses._softmax:
    # row-max subtracted before exp, numba kernel for K<=7, numpy above).
    P = _softmax(F)
    # tree.values is still the sketch tree's 1-D array here; the original
    # replaces it with the (n_leaves, K) matrix, and so do we.
    n_leaves = tree.values.shape[0]
    n_cls = K
    lr = float(self.lr_)
    lam = float(self.l2_leaf_reg)
    values = np.zeros((n_leaves, n_cls))
    counts = np.bincount(np.asarray(leaf), minlength=n_leaves)
    G = np.zeros((n_leaves, n_cls))
    np.add.at(G, leaf, grad)
    eye = np.eye(n_cls)
    for lv in range(n_leaves):
        if counts[lv] == 0:
            continue
        Pm = P[leaf == lv]
        # sum_i (diag(p_i) - p_i p_i^T) = diag(sum p_i) - P'P.
        H = np.diag(Pm.sum(axis=0)) - Pm.T @ Pm
        values[lv] = -lr * np.linalg.solve(H + lam * eye, G[lv])
    tree.values = values
    # Advance F exactly as the original's plain branch does.
    _add_leaf_values(F, tree.values, leaf)


def _guarded_fit_impl(self, *args, **kwargs):
    """Refuse weighted fits: the patched step has no weight path."""
    sw = kwargs.get("sample_weight", args[4] if len(args) > 4 else None)
    if sw is not None:
        raise RuntimeError("full_hessian probe: sample_weight out of scope")
    return _ORIG_FIT_IMPL(self, *args, **kwargs)


def _install_patch():
    PATCH_CALLS["n"] = 0
    MulticlassBoosting._apply_vector_update = _patched_apply_vector_update
    MulticlassBoosting._fit_impl = _guarded_fit_impl


def _remove_patch():
    MulticlassBoosting._apply_vector_update = _ORIG_APPLY_VEC
    MulticlassBoosting._fit_impl = _ORIG_FIT_IMPL


# ---- data path ---------------------------------------------------------------
def _load_split(key, seed):
    """The harness's data path (_run_seed_task); None when @time is degenerate."""
    rng = np.random.default_rng(1000 + seed)
    X, y, cat, task = rb.DATASETS[key](1.0, rng)
    variant = key.split(rb.VARIANT_SEP, 1)[1] if rb.VARIANT_SEP in key else ""
    if variant == "time":
        split = rb._temporal_split(X, y, seed, task)
        if split is None:
            return None
        Xtr, Xte, ytr, yte = split
        return Xtr, Xte, ytr, yte, cat, task
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat)
    if variant in rb.SUS_FRACTIONS:
        Xtr, ytr = rb._subsample_train(Xtr, ytr, rb.SUS_FRACTIONS[variant], task)
    return Xtr, Xte, ytr, yte, cat, task


def _parse_arms(argv):
    """Arm selection from --arms (default: all four).

    Accepts `--arms a b` (bare tokens up to the next flag) and `--arms=a,b`.
    Returned in canonical order; unknown or empty selections exit loudly
    rather than silently running nothing.
    """
    if "--arms" not in argv and not any(a.startswith("--arms=") for a in argv):
        return list(ARMS)
    sel = []
    for i, a in enumerate(argv):
        if a == "--arms":
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                sel.append(argv[j])
                j += 1
        elif a.startswith("--arms="):
            sel.extend(x for x in a.split("=", 1)[1].split(",") if x)
    unknown = [a for a in sel if a not in ARMS]
    if not sel or unknown:
        raise SystemExit(f"bad --arms selection {sel}; choose from "
                         f"{list(ARMS)}")
    return [a for a in ARMS if a in sel]


def _build_keys(selected):
    hc_keys = sorted(k for k in rb.DATASETS if k.startswith("hc:")
                     and rb._task_of(k) == "multiclass")
    syn_keys = sorted(k for k in rb.DATASETS if k.startswith("syn:")
                      and rb.VARIANT_SEP not in k
                      and synthgen.task_of(k) == "multiclass")
    print(f"config: seeds={list(SEEDS)} threads={THREADS} arms={selected}",
          flush=True)
    print(f"HC multiclass keys+twins ({len(hc_keys)}): {hc_keys}", flush=True)
    print(f"SYN multiclass keys ({len(syn_keys)}): {syn_keys}", flush=True)
    return {"hc": hc_keys, "synth": syn_keys}


# ---- arms --------------------------------------------------------------------
def _run_default_arm(task, Xtr, ytr, Xte, yte, cat):
    out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, THREADS, **CHIMERA_CFG)
    metrics, fit_s, _, best = out
    return metrics["brier"], fit_s, best


def _run_full_arm(task, Xtr, ytr, Xte, yte, cat):
    _install_patch()
    try:
        out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, THREADS,
                              **CHIMERA_CFG)
    finally:
        _remove_patch()
    n_calls = PATCH_CALLS["n"]
    assert n_calls > 0, "full_hessian patch never engaged during fit"
    metrics, fit_s, _, best = out
    return metrics["brier"], fit_s, best, n_calls


def _run_lr_arm(task, Xtr, ytr, Xte, yte, cat, rate):
    """The unpatched default estimator at one explicit learning rate."""
    cfg = dict(CHIMERA_CFG, lr=rate)
    out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, THREADS, **cfg)
    metrics, fit_s, _, best = out
    return metrics["brier"], fit_s, best


# ---- bookkeeping ---------------------------------------------------------------
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
        if r["model"] == "ChimeraBoost":
            b = r["metrics"].get("brier")  # regression rows have none
            if b is not None:
                out[(r["dataset"], r["model"], r["seed"])] = b
    return out


# ---- self-check ----------------------------------------------------------------
def _harness_skip(key, seed, saved):
    """Both sides lack this row because the temporal window is degenerate."""
    return (_load_split(key, seed) is None
            and (key, "ChimeraBoost", seed) not in saved)


def _check_row(key, seed, probe, saved):
    """Print one self-check row; return |diff|, "skip", or "missing"."""
    short = key.replace("hc:", "", 1)[:36]
    pc = probe.get((key, seed))
    sc = saved.get((key, "ChimeraBoost", seed))
    if pc is None and sc is None and _harness_skip(key, seed, saved):
        print(f"{short:36s} {seed:2d}  skip (temporal degenerate here "
              f"and in the saved run)")
        return "skip"
    if pc is None or sc is None:
        print(f"{short:36s} {seed:2d}  --missing--  probe={pc} saved={sc}")
        return "missing"
    print(f"{short:36s} {seed:2d} {pc:12.8f} {sc:12.8f} {abs(pc - sc):12.3e}")
    return abs(pc - sc)


def self_check(hc_keys, seeds):
    """Probe default next to the saved run's ChimeraBoost Brier."""
    rows = _read_jsonl()
    probe = {(r["dataset"], r["seed"]): r["brier"]
             for r in rows if r["arm"] == "default"}
    saved = _saved_briers()
    print("\nSELF-CHECK -- probe default vs saved run ChimeraBoost Brier")
    print("Synth keys are not in that run; self-check covers hc: keys only.")
    print(f"{'dataset':36s} {'s':>2s} {'probe':>12s} {'saved':>12s} "
          f"{'|diff|':>12s}")
    diffs, missing = [], []
    worst = None
    for key in hc_keys:
        for seed in seeds:
            res = _check_row(key, seed, probe, saved)
            if res == "skip":
                continue
            if res == "missing":
                missing.append((key, seed))
                continue
            diffs.append(res)
            if worst is None or res > worst[0]:
                worst = (res, key, seed)
    max_d = max(diffs) if diffs else float("nan")
    print(f"max|diff| = {max_d:.3e} over {len(diffs)} paired rows")
    if not missing and diffs and max_d < 1e-9:
        print("SELF-CHECK PASS")
        return True
    print("SELF-CHECK FAIL")
    if worst is not None:
        print(f"  worst: |diff|={worst[0]:.3e} {worst[1]} seed {worst[2]}")
    for key, seed in missing:
        print(f"  missing row: {key} seed {seed}")
    return False


# ---- tables --------------------------------------------------------------------
def _agree(base, arm, mean_change):
    """How many paired seeds agree in sign with the mean relative change."""
    n = 0
    for s in base:
        if s in arm:
            d = base[s] - arm[s]
            if (d > 0) == (mean_change > 0) or d == mean_change == 0:
                n += 1
    return n


def _paired_means(agg, ds, seeds, base="default", cmp="full_hessian"):
    """Mean brier/fit/trees per arm over seeds paired in both arms."""
    out = {base: {}, cmp: {}}
    paired = [s for s in seeds
              if s in agg[(ds, base)] and s in agg[(ds, cmp)]]
    for arm in (base, cmp):
        for field in ("brier", "fit", "trees"):
            vals = [agg[(ds, arm)][s][field] for s in paired]
            out[arm][field] = float(np.mean(vals)) if vals else None
    return out, paired


def _summarize_vs(keys, agg, seeds, base, cmp):
    """One head-to-head summary of cmp against base over complete keys.

    Returns (wins, losses, ties, complete, med_change, med_tree_x,
    med_fit_x, n_near): W/L/T count complete keys (at least one seed
    paired in both arms); the medians run over non-near-solved keys only,
    where near-solved means the better of the two mean Briers is below
    NEAR_SOLVED_BRIER. change = (base-cmp)/base, + = cmp better; the
    ratios are cmp / base mean tree counts and fit times.
    """
    changes, tree_xs, fit_xs = [], [], []
    n_win = n_loss = n_tie = n_near = 0
    for ds in keys:
        means, _ = _paired_means(agg, ds, seeds, base, cmp)
        mb, mc = means[base]["brier"], means[cmp]["brier"]
        if mb is None or mc is None:
            continue
        ch = (mb - mc) / mb if mb > 1e-12 else 0.0
        if min(mb, mc) < NEAR_SOLVED_BRIER:
            n_near += 1
        else:
            changes.append(ch)
            tree_xs.append(means[cmp]["trees"] / means[base]["trees"])
            fit_xs.append(means[cmp]["fit"] / means[base]["fit"])
        if mc == mb:
            n_tie += 1
        elif mc < mb:
            n_win += 1
        else:
            n_loss += 1
    med_c = float(np.median(changes)) if changes else None
    med_t = float(np.median(tree_xs)) if tree_xs else None
    med_f = float(np.median(fit_xs)) if fit_xs else None
    return (n_win, n_loss, n_tie, n_win + n_loss + n_tie,
            med_c, med_t, med_f, n_near)


def _fmt_summary_row(label, summ, show_fit=True):
    """One fixed-width summary line, or a no-rows-yet placeholder."""
    n_win, n_loss, n_tie, complete, mc, mt, mf, n_near = summ
    if complete == 0:
        return f"{label:12s} -- no paired rows yet --"
    line = f"{label:12s}{n_win:>4d}{n_loss:>4d}{n_tie:>4d}"
    if mc is None:
        return line + "   no non-solved keys for the medians"
    line += f"{100 * mc:+12.3f}%{mt:12.2f}"
    if show_fit:
        line += f"{mf:10.2f}"
    return line + f"  (n={complete - n_near}, {n_near} solved*)"


def _table_vs_default(panels, agg, seeds, only):
    print("\n" + "=" * 84)
    print("LR CONTROL vs DEFAULT -- mean Brier over paired seeds "
          "(lower better).")
    print("change = (default-arm)/default, + = arm better; tree_x/fit_x = "
          "arm /")
    print("default mean tree-count / fit-time ratios; * = near-solved "
          "(out of medians).")
    print("=" * 84)
    for name in ("hc", "synth"):
        keys = [k for k in panels[name] if k in only]
        print(f"\n-- {name} ({len(keys)} keys) " + "-" * 40)
        print(f"{'arm':12s}{'W':>4s}{'L':>4s}{'T':>4s}"
              f"{'med_change':>12s}{'med_tree_x':>12s}{'med_fit_x':>10s}")
        for arm in ("full_hessian", "lr_050", "lr_033"):
            summ = _summarize_vs(keys, agg, seeds, "default", arm)
            print(_fmt_summary_row(arm, summ))


def _table_full_vs_lr(panels, agg, seeds, only):
    print("\n" + "=" * 84)
    print("FULL_HESSIAN vs LR ARMS -- head to head (lower Brier better).")
    print("W/L/T = full_hessian wins/losses/ties; change = (lr-full)/lr,")
    print("+ = full_hessian better; tree_x = full_hessian / lr-arm mean")
    print("tree-count ratio.")
    print("=" * 84)
    for name in ("hc", "synth"):
        keys = [k for k in panels[name] if k in only]
        print(f"\n-- {name} ({len(keys)} keys) " + "-" * 40)
        print(f"{'vs':12s}{'W':>4s}{'L':>4s}{'T':>4s}"
              f"{'med_change':>12s}{'med_tree_x':>12s}")
        for arm in ("lr_050", "lr_033"):
            summ = _summarize_vs(keys, agg, seeds, arm, "full_hessian")
            print(_fmt_summary_row(arm, summ, show_fit=False))


def _panel_table(name, keys, agg, seeds):
    print(f"\n-- {name} ({len(keys)} keys) " + "-" * 40)
    print(f"{'dataset':28s}{'default':>12s}{'full_hess':>12s}"
          f"{'change':>10s}{'agree':>8s}{'fit_x':>8s}{'tree_x':>8s}")
    changes, fit_ratios = [], []
    n_win = n_loss = n_tie = n_ns = 0
    for ds in keys:
        short = ds[:27] + ("~" if len(ds) > 28 else "")
        means, paired = _paired_means(agg, ds, seeds)
        md, mf = means["default"]["brier"], means["full_hessian"]["brier"]
        if md is None or mf is None:
            print(f"{short:28s} -- incomplete --")
            continue
        # Near-zero denominators make the ratio meaningless; the key is
        # near-solved and excluded from the medians either way.
        ch = (md - mf) / md if md > 1e-12 else 0.0
        ch_s = f"{100 * ch:+9.3f}%" if md > 1e-12 else "      n/a"
        base = {s: agg[(ds, "default")][s]["brier"] for s in paired}
        arm = {s: agg[(ds, "full_hessian")][s]["brier"] for s in paired}
        ag = _agree(base, arm, ch)
        fit_x = means["full_hessian"]["fit"] / means["default"]["fit"]
        tree_x = means["full_hessian"]["trees"] / means["default"]["trees"]
        near = min(md, mf) < NEAR_SOLVED_BRIER
        mark = "*" if near else ""
        if near:
            n_ns += 1
        else:
            changes.append(ch)
            fit_ratios.append(fit_x)
        if mf == md:
            n_tie += 1
        elif mf < md:
            n_win += 1
        else:
            n_loss += 1
        print(f"{short:28s}{md:12.6f}{mf:12.6f}{ch_s}"
              f"{ag:>5d}/{len(paired)}{fit_x:8.2f}{tree_x:8.2f}{mark:>2s}")
    med_c = float(np.median(changes)) if changes else None
    med_f = float(np.median(fit_ratios)) if fit_ratios else None
    med_s = (f"median {med_c * 100:+.3f}% (n={len(changes)}), "
             f"median fit x{med_f:.2f}" if med_c is not None
             else "no non-solved keys for the medians")
    print(f"  {name}: {n_win}W-{n_loss}L-{n_tie}T of "
          f"{n_win + n_loss + n_tie} complete keys, {med_s}")
    if n_ns:
        print(f"  ({n_ns} near-solved keys marked *, left out of the medians)")


def table(panels, seeds, only):
    rows = _read_jsonl()
    agg = defaultdict(dict)
    for r in rows:
        if r["dataset"] in only and r["seed"] in seeds:
            agg[(r["dataset"], r["arm"])][r["seed"]] = {
                "brier": r["brier"], "fit": r["fit_time"],
                "trees": r["best_iter"]}
    print("\n" + "=" * 84)
    print("FULL SOFTMAX HESSIAN -- mean Brier over paired seeds (lower better).")
    print("change = (default-full)/default, + = full_hessian better; agree =")
    print("seeds agreeing in sign with the mean change; fit_x/tree_x = full /")
    print("default mean fit-time / tree-count ratios; * = near-solved.")
    print("=" * 84)
    for name in ("hc", "synth"):
        _panel_table(name, [k for k in panels[name] if k in only], agg, seeds)
    _table_vs_default(panels, agg, seeds, only)
    _table_full_vs_lr(panels, agg, seeds, only)


# ---- run -------------------------------------------------------------------------
def _run_missing(panels, key_order, seeds, done, arms):
    panel_of = {k: name for name, ks in panels.items() for k in ks}
    for key in key_order:
        for seed in seeds:
            split = _load_split(key, seed)
            if split is None:
                print(f"  [skip] {key} (seed {seed}): temporal window "
                      f"degenerate, as the harness does", flush=True)
                continue
            Xtr, Xte, ytr, yte, cat, task = split
            if seed == seeds[0]:
                ncat = len(cat) if cat else 0
                print(f"\n=== {key}  [{panel_of[key]}]  "
                      f"n={len(ytr) + len(yte)} n_tr={len(ytr)} "
                      f"p={np.shape(Xtr)[1]} cats={ncat} task={task}",
                      flush=True)
            for arm in arms:
                if (key, arm, seed) in done:
                    continue
                if arm == "full_hessian":
                    b, fit_s, best, n_calls = _run_full_arm(
                        task, Xtr, ytr, Xte, yte, cat)
                elif arm in LR_RATES:
                    b, fit_s, best = _run_lr_arm(
                        task, Xtr, ytr, Xte, yte, cat, LR_RATES[arm])
                    n_calls = 0
                else:
                    b, fit_s, best = _run_default_arm(
                        task, Xtr, ytr, Xte, yte, cat)
                    n_calls = 0
                _append({"dataset": key, "panel": panel_of[key], "arm": arm,
                         "seed": seed, "brier": b, "fit_time": fit_s,
                         "best_iter": best, "calls": n_calls})
                done.add((key, arm, seed))
                print(f"  [{arm:12s} s{seed}] brier={b:.6f}  "
                      f"{fit_s:6.1f}s  it={best}  calls={n_calls}", flush=True)


def main():
    argv = sys.argv[1:]
    table_only = "--table-only" in argv
    smoke = "--smoke" in argv
    arms = _parse_arms(argv)
    rb._add_highcard_datasets()
    rb._add_synth_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    panels = _build_keys(arms)
    key_order = panels["hc"] + panels["synth"]
    seeds = SEEDS
    if smoke:
        cand = ["hc:eucalyptus"] + panels["synth"][:1]
        key_order = [k for k in cand if k in key_order]
        seeds = (0,)
        print(f"SMOKE: keys={key_order} seeds={list(seeds)} arms={arms}",
              flush=True)
    if table_only:
        hc_keys = [k for k in key_order if k.startswith("hc:")]
        self_check(hc_keys, seeds)
        table(panels, seeds, set(key_order))
        return
    done = _done_keys(_read_jsonl())
    _run_missing(panels, key_order, seeds, done, arms)
    hc_keys = [k for k in key_order if k.startswith("hc:")]
    self_check(hc_keys, seeds)
    table(panels, seeds, set(key_order))


if __name__ == "__main__":
    main()
