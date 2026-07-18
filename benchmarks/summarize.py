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
# Multiclass columns, rendered ONLY when the results contain multiclass datasets
# (the HC suite adds them; Grinsztajn has none). They are report-only: the
# blended north star (make_pareto) is unchanged and ignores these. Inserted
# before Speed by _display_cols so Speed stays the last column.
MULTI_COLS = ["Multi F1%", "Multi Brier%"]
COL_W = 13

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


# ---------------------------------------------------------------------------
# Head-to-head machinery (benchmarks/STRENGTH_VIZ_PLAN.md). One primary metric
# per dataset — RMSE (regression), Brier (binary/multiclass; proper scoring
# rule, F1 stays a table diagnostic) — both lower = better. Powers the win-rate
# axis in make_pareto; the blended-% columns above stay as diagnostics.
# ---------------------------------------------------------------------------

def primary_scores(data):
    """{dataset: {model: score}} on each dataset's primary metric (lower=better).

    Reuses the exact exclusions the "% vs best" columns apply, so win rate /
    rank and blended see identical data: near-solved regression datasets
    (best NRMSE < NEAR_SOLVED_NRMSE) and near-solved-Brier classification
    datasets (best < 1e-3) are dropped.
    """
    records = data["records"]
    datasets = data["datasets"]
    rmse = _agg_metric(records, "rmse")
    brier = _agg_metric(records, "brier")

    reg_ds = [d for d in datasets if datasets[d]["task"] == "regression"]
    clf_ds = [d for d in datasets
              if datasets[d]["task"] in ("binary", "multiclass")]
    y_std = {d: datasets[d].get("y_std") for d in reg_ds}
    near = set(near_solved_datasets(rmse, reg_ds, y_std))

    out = {}
    for ds in reg_ds:
        scores = {m: v for m, v in rmse.get(ds, {}).items() if v is not None}
        if ds not in near and scores and min(scores.values()) > 0:
            out[ds] = scores
    for ds in clf_ds:
        scores = {m: v for m, v in brier.get(ds, {}).items() if v is not None}
        if scores and min(scores.values()) >= 1e-3:
            out[ds] = scores
    return out


def per_dataset_ranks(primary):
    """{dataset: {model: rank}}, 1 = best; exact score ties share the average rank."""
    out = {}
    for ds, scores in primary.items():
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
        ranks = {}
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            for k in range(i, j + 1):
                ranks[ordered[k][0]] = (i + j) / 2.0 + 1.0
            i = j + 1
        out[ds] = ranks
    return out


def mean_rank(ranks):
    """{model: mean rank across the datasets it competed on} (lower = better)."""
    sums = defaultdict(list)
    for ds_ranks in ranks.values():
        for m, r in ds_ranks.items():
            sums[m].append(r)
    return {m: float(np.mean(v)) for m, v in sums.items()}


def _win_counts(primary):
    """Per-dataset matchup tallies: ({ds: {model: wins}}, {ds: {model: n_opp}}).

    A win against an opponent = strictly lower primary score on that dataset;
    an exact score tie counts 1/2 for each side. Datasets with fewer than two
    scored models have no matchups and are dropped.
    """
    wins, counts = {}, {}
    for ds, scores in primary.items():
        ms = list(scores)
        if len(ms) < 2:
            continue
        wins[ds] = {
            m: sum(1.0 if scores[m] < scores[o]
                   else 0.5 if scores[m] == scores[o] else 0.0
                   for o in ms if o != m)
            for m in ms}
        counts[ds] = {m: len(ms) - 1 for m in ms}
    return wins, counts


def n_tied_matchups(primary):
    """Number of (dataset x model-pair) matchups with an exactly tied score."""
    n = 0
    for scores in primary.values():
        ms = list(scores)
        n += sum(1 for i in range(len(ms)) for j in range(i + 1, len(ms))
                 if scores[ms[i]] == scores[ms[j]])
    return n


def winrate_vs_field(primary):
    """{model: percent of its head-to-head (dataset x opponent) matchups won}.

    0-100, higher = better, 50 = mid-pack. With every model scored on every
    dataset this is exactly (k - mean_rank) / (k - 1) * 100, i.e. mean rank in
    friendlier units — and the row mean of winrate_matrix.
    """
    wins, counts = _win_counts(primary)
    w, c = defaultdict(float), defaultdict(int)
    for ds in wins:
        for m in wins[ds]:
            w[m] += wins[ds][m]
            c[m] += counts[ds][m]
    return {m: 100.0 * w[m] / c[m] for m in w}


def winrate_matrix(primary):
    """(models, matrix): matrix[i][j] = % of shared datasets where models[i]
    beats models[j] on the primary metric (ties 1/2); None where a pair never
    met. Models sorted by win rate vs field, best first."""
    field = winrate_vs_field(primary)
    models = sorted(field, key=lambda m: -field[m])
    mat = [[None] * len(models) for _ in models]
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                continue
            w = t = 0.0
            for scores in primary.values():
                if a in scores and b in scores:
                    t += 1
                    w += (1.0 if scores[a] < scores[b]
                          else 0.5 if scores[a] == scores[b] else 0.0)
            if t:
                mat[i][j] = 100.0 * w / t
    return models, mat


def bootstrap_winrate_ci(primary, n_boot=10000, seed=0):
    """{model: (lo, hi)}: 95% percentile bootstrap CI for winrate_vs_field,
    resampling datasets with replacement (deterministic for a given seed)."""
    wins, counts = _win_counts(primary)
    ds_list = sorted(wins)
    models = sorted({m for ds in ds_list for m in wins[ds]})
    W = np.zeros((len(ds_list), len(models)))
    C = np.zeros_like(W)
    for i, ds in enumerate(ds_list):
        for j, m in enumerate(models):
            if m in wins[ds]:
                W[i, j] = wins[ds][m]
                C[i, j] = counts[ds][m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds_list), size=(n_boot, len(ds_list)))
    tw = W[idx].sum(axis=1)
    tc = C[idx].sum(axis=1)
    with np.errstate(invalid="ignore"):
        rates = np.where(tc > 0, 100.0 * tw / np.maximum(tc, 1e-9), np.nan)
    lo = np.nanpercentile(rates, 2.5, axis=0)
    hi = np.nanpercentile(rates, 97.5, axis=0)
    return {m: (float(lo[j]), float(hi[j])) for j, m in enumerate(models)}


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
        # Multiclass columns use the SAME per-class-sum Brier and macro-F1 as the
        # binary ones (run_benchmarks._compute_metrics computes both for K>2). The
        # same near-solved Brier guard (skip_below) applies per column.
        "Multi F1%": _pct_vs_best(f1, mul_ds, lower=False),
        "Multi Brier%": _pct_vs_best(brier, mul_ds, lower=True, skip_below=1e-3),
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
            "oml:": "OpenML suite", "syn:": "SynthGen suite",
            "hc:": "HC high-cardinality suite"}
    found = {label for pre, label in tags.items()
             if any(d.startswith(pre) for d in ds_names)}
    if not found:
        return "Built-in panel"
    if len(found) == 1:
        return found.pop()
    return "Mixed suites"


def _display_cols(meta):
    """Column order for the table: the base five, with the two multiclass columns
    inserted before Speed only when the run actually has multiclass datasets."""
    cols = list(COLS)
    if meta.get("n_mul", 0) > 0:
        cols[cols.index("Speed"):cols.index("Speed")] = MULTI_COLS
    return cols


def _fmt(v, col):
    if v is None:
        return f"{'--':>{COL_W}}"
    if col == "Bin Calib":
        return f"{v * 1000:>{COL_W - 1}.2f}m"
    if col == "Speed":
        return f"{v:>{COL_W - 1}.1f}x"
    return f"{v:>{COL_W - 1}.1f}%"


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
    show = _display_cols(meta)
    lines = []
    if label:
        lines.append(label)
    hdr = f"{'Model':<22}" + "".join(f"{c:>{COL_W}}" for c in show)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in models:
        row = f"{m:<22}" + "".join(_fmt(cols[c].get(m), c) for c in show)
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
    base_cols, base_meta = aggregate(base_data)
    new_cols, new_meta = aggregate(new_data)
    out = [format_table(base_data, f"=== {base_label} ==="), "",
           format_table(new_data, f"=== {new_label} ==="), "",
           f"=== {focus} delta ({new_label} vs {base_label}) ==="]
    # Show whichever column set is richer (multiclass columns appear if either
    # run has multiclass datasets).
    show = _display_cols(base_meta if base_meta.get("n_mul") else new_meta)
    for c in show:
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
