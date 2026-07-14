"""Aggregate + pretty-print benchmark result JSONs.

Importable helpers shared by the status reporter (`bench_status.py`) and ad-hoc
analysis. Reads the sidecar `.json` produced by `run_benchmarks.py --save` and
collapses it to the five headline columns we track:

    Reg RMSE%   Bin F1%   Bin Brier%   Bin Calib   Speed

All "%" columns are "% vs best on that task" (100 = best model on that dataset,
averaged across datasets). Calib is mean miscalibration (MCB) in units of
10^-3 (lower better). Speed is the mean fit-time multiple vs the fastest model
on each dataset (1.0 = fastest).

CLI:
    python benchmarks/summarize.py <results.json>              # one table
    python benchmarks/summarize.py <base.json> <new.json>     # before/after + delta
    python benchmarks/summarize.py --latest                   # newest json in results/
"""
import json
import os
import glob
from collections import defaultdict

import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_ORDER = ["ChimeraBoost", "ChimeraBoostEns2", "ChimeraBoostEns5",
               "ChimeraBoostEns10", "CatBoost", "LightGBM", "sklearn_HGB", "XGBoost"]
COLS = ["Reg RMSE%", "Bin F1%", "Bin Brier%", "Bin Calib", "Speed"]

# Regression datasets where the BEST model's NRMSE (best_RMSE / y_std) is below
# this are "near-solved": every model nails them (R^2 ~ 1), so the "% vs best"
# RMSE ratio turns a practically-zero absolute gap into a huge fake deficit. We
# drop them from the RMSE aggregate -- the regression analog of the Brier
# skip_best_below guard. The threshold is in a flat valley: anything in
# [~1.6%, ~7%] excludes the same 2 datasets on the Grinsztajn suite (clean cliff
# between artifacts <1.5% and the next real dataset at 7.5%), so it isn't tuned.
NEAR_SOLVED_NRMSE = 0.02


def near_solved_datasets(rmse_per_ds, ds_list, y_std, thresh=NEAR_SOLVED_NRMSE):
    """Subset of ds_list that is near-perfectly solved (best NRMSE < thresh).

    rmse_per_ds: {dataset: {model: mean RMSE}}. y_std: {dataset: target std}.
    Datasets with no recorded y_std can't be judged and are never skipped.
    """
    out = []
    for ds in ds_list:
        scores = [v for v in rmse_per_ds.get(ds, {}).values() if v is not None]
        scale = y_std.get(ds)
        if scores and scale and min(scores) / scale < thresh:
            out.append(ds)
    return out


def load(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def latest_json(results_dir=RESULTS_DIR):
    """Path to the most recently modified results .json that has a 'records' key, or None."""
    files = glob.glob(os.path.join(results_dir, "*.json"))
    valid = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                if "records" in json.load(fh):
                    valid.append(f)
        except Exception:
            pass
    return max(valid, key=os.path.getmtime) if valid else None


def _agg_metric(records, key):
    b = defaultdict(lambda: defaultdict(list))
    for r in records:
        v = r["metrics"].get(key)
        if v is not None:
            b[r["dataset"]][r["model"]].append(v)
    return {ds: {m: float(np.mean(vs)) for m, vs in ms.items()}
            for ds, ms in b.items()}


def _agg_speed(records):
    b = defaultdict(lambda: defaultdict(list))
    for r in records:
        b[r["dataset"]][r["model"]].append(r["fit_time"])
    return {ds: {m: float(np.mean(vs)) for m, vs in ms.items()}
            for ds, ms in b.items()}


def _pct_vs_best(per_ds, ds_list, lower, skip_below=None):
    sums = defaultdict(list)
    for ds in ds_list:
        scores = per_ds.get(ds, {})
        vals = [v for v in scores.values() if v is not None]
        if not vals:
            continue
        best = min(vals) if lower else max(vals)
        if best == 0 or (skip_below and best < skip_below):
            continue
        for m, v in scores.items():
            if v is None or (lower and v <= 0):
                continue
            sums[m].append(100.0 * best / v if lower else 100.0 * v / best)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def _mean_over(per_ds, ds_list):
    sums = defaultdict(list)
    for ds in ds_list:
        for m, v in per_ds.get(ds, {}).items():
            if v is not None:
                sums[m].append(v)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def _mult_vs_best(per_ds, ds_list):
    sums = defaultdict(list)
    for ds in ds_list:
        scores = per_ds.get(ds, {})
        vals = [v for v in scores.values() if v and v > 0]
        if not vals:
            continue
        best = min(vals)
        for m, v in scores.items():
            if v and v > 0:
                sums[m].append(v / best)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def aggregate(data):
    """Return (cols, meta) where cols maps column name -> {model: value} and
    meta carries dataset counts for the caption."""
    records = data["records"]
    datasets = data["datasets"]
    f1 = _agg_metric(records, "f1_macro")
    brier = _agg_metric(records, "brier")
    cal = _agg_metric(records, "calibration_mcb")
    rmse = _agg_metric(records, "rmse")
    speed = _agg_speed(records)

    all_ds = list(datasets)
    reg_ds = [d for d in all_ds if datasets[d]["task"] == "regression"]
    bin_ds = [d for d in all_ds if datasets[d]["task"] == "binary"]
    mul_ds = [d for d in all_ds if datasets[d]["task"] == "multiclass"]

    # Drop near-solved regression datasets from the RMSE column (see the guard
    # comment above). Needs per-dataset target std, stored in dataset meta by
    # run_benchmarks; older JSONs without it simply skip nothing.
    y_std = {d: datasets[d].get("y_std") for d in reg_ds}
    near = near_solved_datasets(rmse, reg_ds, y_std)
    reg_scored = [d for d in reg_ds if d not in near]

    cols = {
        "Reg RMSE%": _pct_vs_best(rmse, reg_scored, lower=True),
        "Bin F1%": _pct_vs_best(f1, bin_ds, lower=False),
        "Bin Brier%": _pct_vs_best(brier, bin_ds, lower=True, skip_below=1e-3),
        "Bin Calib": _mean_over(cal, bin_ds),
        "Speed": _mult_vs_best(speed, all_ds),
    }
    cfg = data.get("config", {})
    meta = {"n_reg": len(reg_ds), "n_bin": len(bin_ds), "n_mul": len(mul_ds),
            "n_reg_excl": len(near), "n_total": len(all_ds),
            "seeds": cfg.get("seeds"),
            "max_iters": cfg.get("max_iters", 2000),
            "patience": cfg.get("patience", 50),
            "threads_per_model": cfg.get("threads_per_model"),
            "suite": _suite_label(all_ds)}
    return cols, meta


def _suite_label(ds_names):
    """Human label for the caption, inferred from dataset key prefixes."""
    tags = {"gr:": "Grinsztajn et al. (2022)", "pm:": "PMLB tuning suite",
            "oml:": "OpenML suite", "syn:": "SynthGen suite"}
    found = {label for pre, label in tags.items()
             if any(d.startswith(pre) for d in ds_names)}
    if not found:
        return "Built-in panel"
    if len(found) == 1:
        return found.pop()
    return "Mixed suites"


def _fmt(v, col):
    if v is None:
        return f"{'--':>11}"
    if col == "Bin Calib":
        return f"{v * 1000:>10.2f}m"
    if col == "Speed":
        return f"{v:>10.1f}x"
    return f"{v:>10.1f}%"


def _models_present(cols):
    seen = set()
    for d in cols.values():
        seen |= set(d)
    return [m for m in MODEL_ORDER if m in seen] + \
           [m for m in seen if m not in MODEL_ORDER]


def format_table(data, label=None):
    """Return a printable string for one results JSON."""
    cols, meta = aggregate(data)
    models = _models_present(cols)
    lines = []
    if label:
        lines.append(label)
    hdr = f"{'Model':<22}" + "".join(f"{c:>11}" for c in COLS)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in models:
        row = f"{m:<22}" + "".join(_fmt(cols[c].get(m), c) for c in COLS)
        lines.append(row)
    seeds = f" | {meta['seeds']} seeds" if meta.get("seeds") else ""
    cap = (f"{meta['suite']} — {meta['n_total']} datasets "
           f"({meta['n_reg']} reg, {meta['n_bin']} binary, "
           f"{meta['n_mul']} multiclass){seeds} | "
           f"100% = best | Calib MCB x10^-3 lower=better | Speed vs fastest")
    lines.append(cap)
    if meta.get("n_reg_excl"):
        n = meta["n_reg_excl"]
        lines.append(
            f"* Reg RMSE% excludes {n} dataset{'s' if n != 1 else ''} every model "
            "solves near-perfectly (best NRMSE < 2%), where the ratio is meaningless.")
    return "\n".join(lines)


def format_compare(base_data, new_data, base_label="BEFORE", new_label="AFTER",
                   focus="ChimeraBoost"):
    """Return before/after tables plus a per-column delta for `focus` model."""
    base_cols, _ = aggregate(base_data)
    new_cols, _ = aggregate(new_data)
    out = [format_table(base_data, f"=== {base_label} ==="), "",
           format_table(new_data, f"=== {new_label} ==="), "",
           f"=== {focus} delta ({new_label} vs {base_label}) ==="]
    for c in COLS:
        bv = base_cols[c].get(focus)
        nv = new_cols[c].get(focus)
        if bv is None or nv is None:
            continue
        if c == "Bin Calib":
            d = (bv - nv) * 1000
            out.append(f"  {c:<12} {bv*1000:.2f}m -> {nv*1000:.2f}m  "
                       f"({d:+.2f}m {'better' if d > 0 else 'worse'})")
        elif c == "Speed":
            d = bv - nv
            out.append(f"  {c:<12} {bv:.1f}x -> {nv:.1f}x  "
                       f"({d:+.1f}x {'faster' if d > 0 else 'slower'})")
        else:
            d = nv - bv
            out.append(f"  {c:<12} {bv:.1f}% -> {nv:.1f}%  ({d:+.1f}pp)")
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="*",
                    help="one json (single table) or two (before/after compare)")
    ap.add_argument("--latest", action="store_true",
                    help="use the most recent results json")
    args = ap.parse_args()

    paths = list(args.json_paths)
    if args.latest:
        lj = latest_json()
        if lj:
            paths = [lj]
    if not paths:
        lj = latest_json()
        if not lj:
            print("No results json found.")
            return
        paths = [lj]

    if len(paths) == 1:
        print(format_table(load(paths[0]), f"# {os.path.basename(paths[0])}"))
    else:
        print(format_compare(load(paths[0]), load(paths[1]),
                             base_label=os.path.basename(paths[0]),
                             new_label=os.path.basename(paths[1])))


if __name__ == "__main__":
    main()
