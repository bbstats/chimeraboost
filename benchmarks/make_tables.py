"""Generate the benchmark summary table as a PNG image.

Reads a JSON file produced by `run_benchmarks.py --save` and writes a single
`summary.png` into `images/`. The table has one row per model and six metric
columns: RMSE %, Binary F1 %, Binary log loss %, Multiclass F1 %, Multiclass
log loss %, and fit-time × slowdown vs fastest.

For "% vs best" cells: 100% = best model, lower = worse. For the fit-time
cell: 1× = fastest, higher = slower (log-color scale).

Run:
    python benchmarks/make_tables.py benchmarks/results/<stamp>.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import near_solved_datasets  # noqa: E402  (shared RMSE guard)


MODEL_ORDER = ["ChimeraBoost", "ChimeraBoostEns10", "CatBoost", "sklearn_HGB",
               "XGBoost", "LightGBM"]


def aggregate_metric(records, metric_key):
    """{dataset: {model: mean over seeds}} for a given metric key."""
    bucket = defaultdict(lambda: defaultdict(list))
    for r in records:
        v = r["metrics"].get(metric_key)
        if v is not None:
            bucket[r["dataset"]][r["model"]].append(v)
    return {ds: {m: float(np.mean(vs)) for m, vs in models.items()}
            for ds, models in bucket.items()}


def aggregate_speed(records):
    bucket = defaultdict(lambda: defaultdict(list))
    for r in records:
        bucket[r["dataset"]][r["model"]].append(r["fit_time"])
    return {ds: {m: float(np.mean(vs)) for m, vs in models.items()}
            for ds, models in bucket.items()}


def pct_vs_best(per_dataset, datasets_in_bin, lower_is_better, skip_best_below=None):
    """Average per-model % vs best across `datasets_in_bin`.

    For each dataset, compute every model's % relative to that dataset's best
    score (100% = best, less = worse). Then average per model. Returns
    {model: avg_pct or None}.

    skip_best_below: for loss columns, drop datasets where the best model's loss
    is below this threshold. On a near-perfectly-solved dataset (e.g. mushroom,
    Brier ~1e-6 for everyone) the best/value ratio explodes a meaningless 1e-4
    difference into a 0%-vs-100% gap, distorting the whole-column average. Such
    datasets carry no probability-quality signal, so we exclude them.
    """
    sums = defaultdict(list)
    for ds in datasets_in_bin:
        if ds not in per_dataset:
            continue
        scores = per_dataset[ds]
        if not scores:
            continue
        vals = [v for v in scores.values() if v is not None]
        if not vals:
            continue
        best = min(vals) if lower_is_better else max(vals)
        if best == 0:
            continue
        if skip_best_below is not None and best < skip_best_below:
            continue
        for m, v in scores.items():
            if v is None:
                continue
            if lower_is_better:
                if v <= 0:
                    continue
                pct = 100.0 * best / v
            else:
                pct = 100.0 * v / best if best > 0 else None
            if pct is not None:
                sums[m].append(pct)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def multiple_vs_best(per_dataset, datasets_in_bin):
    """Average per-model fit-time multiple vs fastest across datasets in bin.

    For each dataset: multiple = model_time / fastest_time. 1.0 means tied
    for fastest; 2.0 means twice as slow. Averaged across datasets.
    """
    sums = defaultdict(list)
    for ds in datasets_in_bin:
        if ds not in per_dataset:
            continue
        scores = per_dataset[ds]
        if not scores:
            continue
        vals = [v for v in scores.values() if v is not None and v > 0]
        if not vals:
            continue
        best = min(vals)
        for m, v in scores.items():
            if v is None or v <= 0:
                continue
            sums[m].append(v / best)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def mean_over_datasets(per_dataset, datasets_in_bin):
    """Plain per-model mean of a metric across `datasets_in_bin`. Used for the
    calibration (MCB) column, which is a raw near-zero loss reported as-is rather
    than as a best/value ratio (that ratio explodes near zero)."""
    sums = defaultdict(list)
    for ds in datasets_in_bin:
        for m, v in per_dataset.get(ds, {}).items():
            if v is not None:
                sums[m].append(v)
    return {m: float(np.mean(v)) if v else None for m, v in sums.items()}


def render_table(data, row_labels, col_labels, col_groups, title, out_path,
                 highlight_row="ChimeraBoost", kind="pct", col_kinds=None,
                 subtitle=None, footnote=None):
    """Render one table as a PNG.

    data: 2-D list/array shape (n_rows, n_cols); cells are floats or None.
    row_labels: list of n_rows strings (model names).
    col_labels: list of n_cols strings (the column header text under groups).
    col_groups: list of (group_label, span) tuples that sum to n_cols. Used
                for the top header row that spans multiple value columns.
                Pass None or [] to skip the group header row.
    title: figure title.
    subtitle: optional 2nd line of title (smaller font, e.g. dataset count).
    out_path: where to save the PNG.
    kind: 'pct' or 'speed' default applied to ALL columns when col_kinds is None.
    col_kinds: optional list of per-column kinds, length == n_cols. Mixes
        'pct' and 'speed' cells in the same table (e.g. quality columns +
        a speed column).
    """
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    # Per-column kind (defaults to the table-wide `kind`).
    if col_kinds is None:
        col_kinds = [kind] * n_cols
    assert len(col_kinds) == n_cols

    # Layout: 1 label column + n_cols data columns.
    # Row heights: optional group header row + 1 column header row + n_rows data rows.
    has_groups = bool(col_groups)
    header_rows = (1 if has_groups else 0) + 1

    # Cell width depends on whether group headers are wide (per-group span).
    # Widen cells when a group header text is long but the group spans only
    # 1 column (would otherwise crash into neighbors).
    cell_w = 1.5
    if col_groups:
        max_label_chars = max(
            max(len(line) for line in str(label).split("\n"))
            for label, _ in col_groups
        )
        min_required_w = max_label_chars * 0.10 + 0.4
        min_span = min(span for _, span in col_groups)
        if min_span * cell_w < min_required_w:
            cell_w = min_required_w / min_span
    cell_h = 0.55
    label_w = 1.9
    fig_w = label_w + n_cols * cell_w + 0.6
    fig_h = (n_rows + header_rows) * cell_h + 1.2 + (0.45 if footnote else 0.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    # Title (optional subtitle on 2nd line)
    if subtitle:
        ax.text(fig_w / 2, 0.25, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#222")
        ax.text(fig_w / 2, 0.55, subtitle, ha="center", va="center",
                fontsize=10.5, color="#555")
    else:
        ax.text(fig_w / 2, 0.35, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#222")

    # Color scheme:
    #   pct cells   : fixed scale, 100% = green, 75% = red, below 75 clamps red.
    #   speed cells : log-scale per-column, fastest = green, slowest = red.
    #                 No hardcoded ceiling; the column's own range sets the scale,
    #                 so e.g. a 39x cell looks much worse than 5x in the same column.
    import math

    def _grad_color(norm):
        # Smooth red -> amber -> green; norm in [0, 1] (0 red, 1 green).
        if norm < 0.5:
            t = norm / 0.5
            r, g, b = 1.0, 0.65 * t + 0.35, 0.35 + 0.1 * t
        else:
            t = (norm - 0.5) / 0.5
            r = 1.0 - 0.55 * t
            g = 0.95 - 0.25 * t
            b = 0.45 - 0.05 * t
        # Soften (mix with white) so text reads well
        mix = 0.45
        return (r * (1 - mix) + mix, g * (1 - mix) + mix, b * (1 - mix) + mix)

    # Precompute per-column min/max for speed columns (used as the log-scale
    # endpoints). For pct columns the scale is fixed (75-100), values unused.
    col_min = [0.0] * n_cols
    col_max = [0.0] * n_cols
    for c in range(n_cols):
        if col_kinds[c] in ("speed", "calib"):
            col_vals = [data[r][c] for r in range(n_rows) if data[r][c] is not None]
            if col_vals:
                col_min[c] = min(col_vals)
                col_max[c] = max(col_vals)

    def cell_color(val, c):
        if val is None:
            return "#e8e8e8"
        if col_kinds[c] == "speed":
            # Log-spaced: lowest val (fastest) -> green, highest -> red.
            if col_max[c] - col_min[c] < 1e-6:
                return _grad_color(0.5)
            log_min = math.log2(max(col_min[c], 1.0))
            log_max = math.log2(max(col_max[c], 1.0))
            log_val = math.log2(max(val, 1.0))
            denom = max(log_max - log_min, 1e-6)
            norm = 1.0 - (log_val - log_min) / denom   # flip: 0 = red, 1 = green
        elif col_kinds[c] == "calib":
            # Miscalibration (lower = better). Linear per-column scale; the values
            # are near-zero losses so a log scale (like speed) wouldn't apply.
            denom = max(col_max[c] - col_min[c], 1e-12)
            norm = 1.0 - (val - col_min[c]) / denom    # lowest -> green
        else:
            # pct: clamp scale to [75, 100]; 75% = red, 100% = green.
            norm = (val - 75.0) / 25.0
        return _grad_color(max(0.0, min(1.0, norm)))

    # Per-column best is still tracked, but only to bold the winning cell;
    # color shading no longer depends on it. For speed, "best" = lowest.
    col_best = []
    for c in range(n_cols):
        col_vals = [data[r][c] for r in range(n_rows) if data[r][c] is not None]
        if not col_vals:
            col_best.append(None)
        else:
            col_best.append(
                min(col_vals) if col_kinds[c] in ("speed", "calib")
                else max(col_vals)
            )

    y = 0.7
    # Optional group-header row.
    if has_groups:
        x = label_w
        # Blank left cell
        rect = mpatches.FancyBboxPatch((0.1, y), label_w - 0.1, cell_h,
                                       boxstyle="round,pad=0.0", linewidth=0,
                                       facecolor="#ffffff")
        ax.add_patch(rect)
        for label, span in col_groups:
            rect = mpatches.FancyBboxPatch((x, y), span * cell_w, cell_h,
                                           boxstyle="round,pad=0.0",
                                           linewidth=0, facecolor="#36454F")
            ax.add_patch(rect)
            ax.text(x + span * cell_w / 2, y + cell_h / 2, label,
                    ha="center", va="center", color="white",
                    fontsize=10.5, fontweight="bold")
            x += span * cell_w
        y += cell_h

    # Column-label row
    rect = mpatches.Rectangle((0.1, y), label_w - 0.1, cell_h,
                              linewidth=0, facecolor="#4a5560")
    ax.add_patch(rect)
    ax.text((label_w - 0.1) / 2 + 0.1, y + cell_h / 2, "Model",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    x = label_w
    for c, lab in enumerate(col_labels):
        rect = mpatches.Rectangle((x, y), cell_w, cell_h,
                                  linewidth=0, facecolor="#4a5560")
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y + cell_h / 2, lab,
                ha="center", va="center", color="white", fontsize=9.5)
        x += cell_w
    y += cell_h

    # Data rows
    for r in range(n_rows):
        is_us = (row_labels[r] == highlight_row)
        # Row label cell
        bg = "#dfe9f5" if is_us else ("#f7f7f7" if r % 2 == 0 else "#ffffff")
        rect = mpatches.Rectangle((0.1, y), label_w - 0.1, cell_h,
                                  linewidth=0, facecolor=bg)
        ax.add_patch(rect)
        ax.text(0.25, y + cell_h / 2, row_labels[r],
                ha="left", va="center", color="#222",
                fontsize=10.5,
                fontweight="bold" if is_us else "normal")
        # Data cells
        x = label_w
        for c in range(n_cols):
            val = data[r][c]
            color = cell_color(val, c)
            rect = mpatches.Rectangle((x, y), cell_w, cell_h,
                                      linewidth=0, facecolor=color)
            ax.add_patch(rect)
            if val is None:
                txt = "—"
                weight = "normal"
            else:
                if col_kinds[c] == "speed":
                    txt = f"{val:.1f}×"
                elif col_kinds[c] == "calib":
                    txt = f"{val * 1000:.1f}"   # MCB in units of 10^-3
                else:
                    txt = f"{val:.1f}%"
                weight = "bold" if val == col_best[c] else "normal"
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", color="#1a1a1a",
                    fontsize=10, fontweight=weight)
            x += cell_w
        y += cell_h

    # Optional footnote sentence below the table (e.g. the RMSE exclusion note).
    if footnote:
        ax.text(0.25, y + 0.28, footnote, ha="left", va="top",
                fontsize=8.5, color="#666", style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to run_benchmarks.py --save .json output")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for PNGs (default: ../images/ alongside repo)")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    datasets = data["datasets"]

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "images"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-aggregate per-(dataset, model) for each metric. Brier (not log loss)
    # is the probability-quality column: it's bounded and proper, so a single
    # near-separable dataset can't distort the cross-dataset aggregate the way
    # unbounded log loss does (e.g. mushroom, where best/value explodes).
    f1 = aggregate_metric(records, "f1_macro")
    brier = aggregate_metric(records, "brier")
    cal = aggregate_metric(records, "calibration_mcb")
    rmse = aggregate_metric(records, "rmse")
    speed = aggregate_speed(records)

    # Keep canonical order, but drop models that weren't run in this benchmark.
    present = {r["model"] for r in records}
    models = [m for m in MODEL_ORDER if m in present]

    # ============================================================
    # Summary table: one row per model, columns = RMSE, Binary F1,
    # Binary LL, Multiclass F1, Multiclass LL, fit-time multiplier.
    # ============================================================
    all_ds = list(datasets.keys())
    reg_ds_all = [d for d in all_ds if datasets[d]["task"] == "regression"]
    bin_ds_all = [d for d in all_ds if datasets[d]["task"] == "binary"]
    mul_ds_all = [d for d in all_ds if datasets[d]["task"] == "multiclass"]

    sum_cols = []
    sum_col_labels = []
    sum_col_kinds = []

    # Exclude near-solved regression datasets (best NRMSE below threshold) from
    # the RMSE column -- their "% vs best" ratio is a near-zero/near-zero artifact.
    # Shared with summarize.py so the table and the Pareto agree.
    reg_y_std = {d: datasets[d].get("y_std") for d in reg_ds_all}
    reg_near = near_solved_datasets(rmse, reg_ds_all, reg_y_std)
    reg_ds_scored = [d for d in reg_ds_all if d not in reg_near]

    if reg_ds_all:
        sum_cols.append(pct_vs_best(rmse, reg_ds_scored, lower_is_better=True))
        sum_col_labels.append("RMSE*" if reg_near else "RMSE")
        sum_col_kinds.append("pct")
    if bin_ds_all:
        sum_cols.append(pct_vs_best(f1, bin_ds_all, lower_is_better=False))
        sum_col_labels.append("F1 macro")
        sum_col_kinds.append("pct")
        sum_cols.append(pct_vs_best(brier, bin_ds_all, lower_is_better=True,
                                    skip_best_below=1e-3))
        sum_col_labels.append("Brier")
        sum_col_kinds.append("pct")
        sum_cols.append(mean_over_datasets(cal, bin_ds_all))
        sum_col_labels.append("Calib")
        sum_col_kinds.append("calib")
    if mul_ds_all:
        sum_cols.append(pct_vs_best(f1, mul_ds_all, lower_is_better=False))
        sum_col_labels.append("F1 macro")
        sum_col_kinds.append("pct")
        sum_cols.append(pct_vs_best(brier, mul_ds_all, lower_is_better=True,
                                    skip_best_below=1e-3))
        sum_col_labels.append("Brier")
        sum_col_kinds.append("pct")
        sum_cols.append(mean_over_datasets(cal, mul_ds_all))
        sum_col_labels.append("Calib")
        sum_col_kinds.append("calib")
    # Single speed column across all datasets.
    sum_cols.append(multiple_vs_best(speed, all_ds))
    sum_col_labels.append("fit time")
    sum_col_kinds.append("speed")

    sum_groups = [
        ("Regression", 1 if reg_ds_all else 0),
        ("Binary", 3 if bin_ds_all else 0),      # F1, Brier, Calib
        ("Multiclass", 3 if mul_ds_all else 0),  # F1, Brier, Calib
        ("Slowdown", 1),
    ]
    sum_groups = [(lab, span) for lab, span in sum_groups if span > 0]

    sum_table = [[c.get(m) for c in sum_cols] for m in models]
    render_table(
        sum_table, models, sum_col_labels, sum_groups,
        title="ChimeraBoost vs other GBMs",
        subtitle=("avg % vs best  ·  Calib = miscalibration ×10⁻³ (lower better)  "
                  f"·  fit time as × slowdown  ·  Grinsztajn et al. (2022)"),
        out_path=os.path.join(out_dir, "summary.png"),
        col_kinds=sum_col_kinds,
        footnote=(
            f"* The RMSE column excludes {len(reg_near)} dataset"
            f"{'s' if len(reg_near) != 1 else ''} that every model solves "
            "near-perfectly (best NRMSE < 2%), where the percent-of-best ratio "
            "is meaningless." if reg_near else None),
    )

    print(f"Wrote summary.png to {out_dir}")


if __name__ == "__main__":
    main()
