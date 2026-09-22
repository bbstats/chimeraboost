"""Probe s6-ts-quant-small: size-gated TS quantization on fresh small data.

On 2026-09-22 (I035) quantizing the ordered target statistic (TS) to 16
uniform buckets hurt the big high-cardinality sets but helped the two small
ones, kdd_ipums (+1.65% Brier) and eucalyptus (+3.26%). Those two sets
suggested the idea, so they cannot confirm it. This probe tests the
size-gated form -- quantize only when the fit has fewer than N_GATE training
rows -- on small categorical classification data that did NOT suggest it.

Panels (classification only; regression keys skipped everywhere):
  SELECT        the two suggesting sets plus every registered twin. Report
                only, never in a bar.
  CONFIRM_REAL  every other hc: key or twin with < N_GATE train rows on seed 0.
  CONFIRM_SYNTH every syn: classification key with >= 1 categorical column
                and < N_GATE train rows on seed 0.
  CONTROL       every hc: key or twin with >= N_GATE train rows on seed 0.
                The gated arm never patches there, so every row must tie.

Arms: chimera (harness default) vs ts_q16_gated (same fit, TS quantized to
Q buckets when n_train < N_GATE, untouched otherwise).

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
from chimeraboost.target_encoding import OrderedTargetEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-ts-quant-small.jsonl")
SAVED_RUN = os.path.join(HERE, "results", "20260922-144219.json")

N_GATE = 10_000
Q = 16
SEEDS = (0, 1, 2)
THREADS = 2  # saved run's per-job count (12 cores / 5 jobs)

SELECT_BASES = ("hc:kdd_ipums_la_97-small", "hc:eucalyptus")
PANEL_ORDER = ("SELECT", "CONFIRM_REAL", "CONFIRM_SYNTH", "CONTROL")
ARMS = ("chimera", "ts_q16_gated")

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

# ---- TS quantization monkeypatch (exactly probe_ts_rarity.py) ----------------
_ORIG_FT = OrderedTargetEncoder.fit_transform
_ORIG_T = OrderedTargetEncoder.transform


def _quantize(out):
    q = np.clip(np.floor(out * Q), 0, Q - 1)
    return (q + 0.5) / Q


def _ft_q(self, codes_matrix, y, sample_weight=None):
    return _quantize(_ORIG_FT(self, codes_matrix, y, sample_weight))


def _t_q(self, codes_matrix):
    return _quantize(_ORIG_T(self, codes_matrix))


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


# ---- panels ------------------------------------------------------------------
def _select_keys():
    out = [b for b in SELECT_BASES if b in rb.DATASETS]
    for key in sorted(rb.DATASETS):
        if rb.VARIANT_SEP in key:
            parent = key.split(rb.VARIANT_SEP, 1)[0]
            if parent in SELECT_BASES:
                out.append(key)
    return out


def _hc_panels(select):
    """Split non-SELECT hc: classification keys by seed-0 train rows."""
    confirm, control, skipped, regr = [], [], [], []
    for key in sorted(k for k in rb.DATASETS if k.startswith("hc:")):
        if key in select:
            continue
        split = _load_split(key, 0)
        if split is None:
            skipped.append(key)
            continue
        _, _, ytr, _, _, task = split
        if task == "regression":
            regr.append(key)
            continue
        if len(ytr) < N_GATE:
            confirm.append(key)
        else:
            control.append(key)
    return confirm, control, skipped, regr


def _synth_panel():
    """Qualifying syn: classification keys with >= 1 cat and < N_GATE rows."""
    qual = []
    n_cls = n_cat = n_regr = 0
    for key in sorted(k for k in rb.DATASETS if k.startswith("syn:")):
        if rb.VARIANT_SEP in key:
            continue
        if synthgen.task_of(key) == "regression":
            n_regr += 1
            continue
        n_cls += 1
        _, _, ytr, _, cat, _ = _load_split(key, 0)
        if not cat:
            continue
        n_cat += 1
        if len(ytr) < N_GATE:
            qual.append(key)
    return qual, n_cls, n_cat, n_regr


def _build_panels():
    print(f"config: N_GATE={N_GATE} Q={Q} seeds={list(SEEDS)} "
          f"threads={THREADS}", flush=True)
    select = _select_keys()
    confirm_real, control, skipped, regr = _hc_panels(set(select))
    confirm_synth, n_cls, n_cat, n_regr = _synth_panel()
    panels = {"SELECT": select, "CONFIRM_REAL": confirm_real,
              "CONFIRM_SYNTH": confirm_synth, "CONTROL": control}
    print(f"SELECT ({len(select)}): {select}", flush=True)
    print(f"CONFIRM_REAL ({len(confirm_real)}): {confirm_real}", flush=True)
    print(f"CONFIRM_SYNTH: {len(confirm_synth)} qualify of {n_cls} syn "
          f"classification keys ({n_cat} with >=1 cat, {n_regr} regression "
          f"skipped)", flush=True)
    print(f"CONTROL ({len(control)}): {control}", flush=True)
    print(f"hc regression keys skipped ({len(regr)}): {regr}", flush=True)
    print(f"hc keys skipped, temporal degenerate on seed 0 "
          f"({len(skipped)}): {skipped}", flush=True)
    return panels


# ---- arms --------------------------------------------------------------------
def _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat):
    out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, THREADS, **CHIMERA_CFG)
    metrics, fit_s, _, best = out
    return metrics["brier"], fit_s, best


def _run_gated_arm(task, Xtr, ytr, Xte, yte, cat):
    if len(ytr) >= N_GATE:
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat)
    OrderedTargetEncoder.fit_transform = _ft_q
    OrderedTargetEncoder.transform = _t_q
    try:
        return _run_chimera_arm(task, Xtr, ytr, Xte, yte, cat)
    finally:
        OrderedTargetEncoder.fit_transform = _ORIG_FT
        OrderedTargetEncoder.transform = _ORIG_T


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
    """Probe chimera next to the saved run's ChimeraBoost Brier."""
    rows = _read_jsonl()
    probe = {(r["dataset"], r["seed"]): r["brier"]
             for r in rows if r["arm"] == "chimera"}
    saved = _saved_briers()
    print("\nSELF-CHECK -- probe chimera vs saved run ChimeraBoost Brier")
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
def _agree(per_seed_base, per_seed_arm, mean_change):
    """How many seeds agree in sign with the mean relative change."""
    n = 0
    for s in per_seed_base:
        if s in per_seed_arm:
            d = per_seed_base[s] - per_seed_arm[s]
            if (d > 0) == (mean_change > 0) or d == mean_change == 0:
                n += 1
    return n


def _means(agg, ds, arm, seeds):
    have = [agg[(ds, arm)][s] for s in seeds if s in agg[(ds, arm)]]
    return float(np.mean(have)) if len(have) == len(seeds) else None


def _row_ties(agg, keys, seeds):
    rt = rn = 0
    for ds in keys:
        for s in seeds:
            a = agg[(ds, "chimera")].get(s)
            b = agg[(ds, "ts_q16_gated")].get(s)
            if a is None or b is None:
                continue
            rn += 1
            if a == b:
                rt += 1
    return rt, rn


def _panel_table(name, keys, agg, seeds):
    note = " (report only -- suggested the idea, never in a bar)" \
        if name == "SELECT" else ""
    print(f"\n-- {name} ({len(keys)} keys){note} " + "-" * 40)
    print(f"{'dataset':32s}{'chimera':>12s}{'gated':>12s}"
          f"{'change':>10s}{'agree':>8s}")
    changes = []
    n_win = n_loss = n_tie = 0
    for ds in keys:
        short = ds[:32]
        mc = _means(agg, ds, "chimera", seeds)
        mg = _means(agg, ds, "ts_q16_gated", seeds)
        if mc is None or mg is None:
            print(f"{short:32s} -- incomplete --")
            continue
        ch = (mc - mg) / mc
        ag = _agree(agg[(ds, "chimera")], agg[(ds, "ts_q16_gated")], ch)
        changes.append(ch)
        if mg == mc:
            n_tie += 1
        elif mg < mc:
            n_win += 1
        else:
            n_loss += 1
        print(f"{short:32s}{mc:12.6f}{mg:12.6f}{100 * ch:+9.3f}%"
              f"{ag:>5d}/{len(seeds)}")
    if changes:
        med = float(np.median(changes))
        print(f"  {name}: {n_win}W-{n_loss}L-{n_tie}T of {len(changes)} "
              f"complete keys, median {med * 100:+.3f}%")
    else:
        print(f"  {name}: no complete keys")
    if name == "CONTROL":
        rt, rn = _row_ties(agg, keys, seeds)
        print(f"  CONTROL exact ties: {n_tie}/{len(changes)} keys, "
              f"{rt}/{rn} rows (must be all)")


def table(panels, seeds, only):
    rows = _read_jsonl()
    agg = defaultdict(dict)
    for r in rows:
        if r["dataset"] in only and r["seed"] in seeds:
            agg[(r["dataset"], r["arm"])][r["seed"]] = r["brier"]
    print("\n" + "=" * 76)
    print("SIZE-GATED TS QUANTIZATION -- mean Brier over seeds (lower better).")
    print("change = (chimera-gated)/chimera, + = gated better; agree = seeds")
    print("agreeing in sign with the mean change.")
    print("=" * 76)
    for name in PANEL_ORDER:
        _panel_table(name, [k for k in panels[name] if k in only], agg, seeds)


# ---- run -------------------------------------------------------------------------
def _run_missing(panels, key_order, seeds, done):
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
            for arm in ARMS:
                if (key, arm, seed) in done:
                    continue
                run = _run_gated_arm if arm == "ts_q16_gated" \
                    else _run_chimera_arm
                b, fit_s, _ = run(task, Xtr, ytr, Xte, yte, cat)
                _append({"dataset": key, "panel": panel_of[key], "arm": arm,
                         "seed": seed, "brier": b, "n_train": len(ytr),
                         "fit_time": fit_s})
                done.add((key, arm, seed))
                print(f"  [{arm:12s} s{seed}] brier={b:.6f}  "
                      f"n_tr={len(ytr)}  {fit_s:6.1f}s", flush=True)


def main():
    table_only = "--table-only" in sys.argv[1:]
    smoke = "--smoke" in sys.argv[1:]
    rb._add_highcard_datasets()
    rb._add_synth_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    panels = _build_panels()
    key_order = [k for name in PANEL_ORDER for k in panels[name]]
    seeds = SEEDS
    if smoke:
        cand = ["hc:eucalyptus"] + panels["CONFIRM_SYNTH"][:1]
        key_order = [k for k in cand if k in key_order]
        seeds = (0,)
        print(f"SMOKE: keys={key_order} seeds={list(seeds)}", flush=True)
    if table_only:
        hc_keys = [k for k in key_order if k.startswith("hc:")]
        self_check(hc_keys, seeds)
        table(panels, seeds, set(key_order))
        return
    done = _done_keys(_read_jsonl())
    _run_missing(panels, key_order, seeds, done)
    hc_keys = [k for k in key_order if k.startswith("hc:")]
    self_check(hc_keys, seeds)
    table(panels, seeds, set(key_order))


if __name__ == "__main__":
    main()
