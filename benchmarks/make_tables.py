"""Generate the 6 benchmark summary tables as PNG images.

Reads a JSON file produced by `run_benchmarks.py --save` (the sidecar to the
.txt log) and writes 6 PNGs into `benchmarks/results/figures/`:

  by_task_quality.png         by_task_speed.png
  by_categorical_quality.png  by_categorical_speed.png
  by_size_quality.png         by_size_speed.png

Cells are "% relative to best" averaged across datasets in the bin. For each
metric, the convention is: 100% = best model, lower = worse. For "higher is
better" metrics (F1 macro), pct = ours / best * 100. For "lower is better"
metrics (RMSE, log loss, fit time), pct = best / ours * 100. Cells are
color-graded (green = good, red = bad), best per column shown in bold.

Run:
    python benchmarks/make_tables.py benchmarks/results/<stamp>.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


MODEL_ORDER = ["ChimeraBoost", "CatBoost", "sklearn_HGB", "XGBoost", "LightGBM"]
SMALL_THRESHOLD = 3000     # < SMALL is "small"
LARGE_THRESHOLD = 20000    # >= LARGE is "large"; in-between is "medium"


def size_bucket(n_train):
    if n_train < SMALL_THRESHOLD:
        return "small"
    if n_train < LARGE_THRESHOLD:
        return "medium"
    return "large"


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


def pct_vs_best(per_dataset, datasets_in_bin, lower_is_better):
    """Average per-model % vs best across `datasets_in_bin`.

    For each dataset, compute every model's % relative to that dataset's best
    score (100% = best, less = worse). Then average per model. Returns
    {model: avg_pct or None}.
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


def render_table(data, row_labels, col_labels, col_groups, title, out_path,
                 highlight_row="ChimeraBoost", kind="pct", col_kinds=None,
                 subtitle=None):
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
    fig_h = (n_rows + header_rows) * cell_h + 1.2

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

    # Absolute color scale (NOT per-column): every cell is graded on the same
    # axis so different metrics are comparable.
    #
    # kind='pct'   : 60-100% linear, 100% green, 60% red. (quality)
    # kind='speed' : 1x-10x log2, 1x green, 10x+ red. (fit-time multiple)
    def cell_color(val, col_kind):
        if val is None:
            return "#e8e8e8"
        if col_kind == "speed":
            # log2: 1x -> 0, 2x -> 1, 4x -> 2, 8x -> 3, 10x -> 3.32 ~ ceiling.
            import math
            norm = math.log2(max(val, 1.0)) / math.log2(10.0)
            norm = max(0.0, min(1.0, norm))
            # FLIP: 0 (1x) = green, 1 (10x+) = red.
            norm = 1.0 - norm
        else:
            SCALE_FLOOR = 60.0
            SCALE_CEIL = 100.0
            norm = (val - SCALE_FLOOR) / (SCALE_CEIL - SCALE_FLOOR)
            norm = max(0.0, min(1.0, norm))
        # Smooth red->amber->green (norm=0 red, 1 green).
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
        r = r * (1 - mix) + 1.0 * mix
        g = g * (1 - mix) + 1.0 * mix
        b = b * (1 - mix) + 1.0 * mix
        return (r, g, b)

    # Per-column best is still tracked, but only to bold the winning cell;
    # color shading no longer depends on it. For speed, "best" = lowest.
    col_best = []
    for c in range(n_cols):
        col_vals = [data[r][c] for r in range(n_rows) if data[r][c] is not None]
        if not col_vals:
            col_best.append(None)
        else:
            col_best.append(
                min(col_vals) if col_kinds[c] == "speed" else max(col_vals)
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
            color = cell_color(val, col_kinds[c])
            rect = mpatches.Rectangle((x, y), cell_w, cell_h,
                                      linewidth=0, facecolor=color)
            ax.add_patch(rect)
            if val is None:
                txt = "—"
                weight = "normal"
            else:
                if col_kinds[c] == "speed":
                    txt = f"{val:.1f}×"
                else:
                    txt = f"{val:.1f}%"
                weight = "bold" if val == col_best[c] else "normal"
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", color="#1a1a1a",
                    fontsize=10, fontweight=weight)
            x += cell_w
        y += cell_h

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

    # Bucket datasets by each axis.
    by_task = defaultdict(list)
    by_cats = defaultdict(list)
    by_size = defaultdict(list)
    for ds, meta in datasets.items():
        by_task[meta["task"]].append(ds)
        by_cats["with categoricals" if meta["has_cats"] else "no categoricals"].append(ds)
        by_size[size_bucket(meta["n_train"])].append(ds)

    # Pre-aggregate per-(dataset, model) for each metric.
    f1 = aggregate_metric(records, "f1_macro")
    ll = aggregate_metric(records, "log_loss")
    rmse = aggregate_metric(records, "rmse")
    speed = aggregate_speed(records)

    models = [m for m in MODEL_ORDER]
    n_models = len(models)

    def fill_row(d):
        """Map a {model: pct} dict to a row in MODEL_ORDER, missing -> None."""
        return [d.get(m) for m in models]

    # ---- helper: build classification + regression quality block for a bin
    def quality_block_for(bin_datasets):
        """Returns three column dicts (RMSE%, F1%, LL%) for this bin."""
        reg_ds = [d for d in bin_datasets if datasets[d]["task"] == "regression"]
        cls_ds = [d for d in bin_datasets if datasets[d]["task"] != "regression"]
        rmse_pct = pct_vs_best(rmse, reg_ds, lower_is_better=True) if reg_ds else {}
        f1_pct = pct_vs_best(f1, cls_ds, lower_is_better=False) if cls_ds else {}
        ll_pct = pct_vs_best(ll, cls_ds, lower_is_better=True) if cls_ds else {}
        return rmse_pct, f1_pct, ll_pct

    def speed_for(bin_datasets):
        return multiple_vs_best(speed, bin_datasets)

    # ============================================================
    # Table 1: Quality by Task Type
    # ============================================================
    reg_ds = by_task.get("regression", [])
    bin_ds = by_task.get("binary", [])
    mul_ds = by_task.get("multiclass", [])
    cols = []
    col_labels = []
    col_groups = []
    if reg_ds:
        cols.append(pct_vs_best(rmse, reg_ds, lower_is_better=True))
        col_labels.append("RMSE")
        col_groups.append((f"Regression\n({len(reg_ds)} datasets)", 1))
    if bin_ds:
        cols.append(pct_vs_best(f1, bin_ds, lower_is_better=False))
        cols.append(pct_vs_best(ll, bin_ds, lower_is_better=True))
        col_labels.extend(["F1 macro", "Log loss"])
        col_groups.append((f"Binary\n({len(bin_ds)} datasets)", 2))
    if mul_ds:
        cols.append(pct_vs_best(f1, mul_ds, lower_is_better=False))
        cols.append(pct_vs_best(ll, mul_ds, lower_is_better=True))
        col_labels.extend(["F1 macro", "Log loss"])
        col_groups.append((f"Multiclass\n({len(mul_ds)} datasets)", 2))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Quality by task type  (avg % relative to best model)",
        os.path.join(out_dir, "by_task_quality.png"),
    )

    # ============================================================
    # Table 2: Speed by Task Type
    # ============================================================
    cols = []
    col_labels = []
    col_groups = []
    for label, ds_list in [("Regression", reg_ds), ("Binary", bin_ds),
                            ("Multiclass", mul_ds)]:
        if not ds_list:
            continue
        cols.append(speed_for(ds_list))
        col_labels.append("fit speed")
        col_groups.append((f"{label}\n({len(ds_list)} datasets)", 1))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Fit time by task type  (× slower than fastest, 1× = best)",
        os.path.join(out_dir, "by_task_speed.png"),
        kind="speed",
    )

    # ============================================================
    # Table 3: Quality by Categorical
    # ============================================================
    cols, col_labels, col_groups = [], [], []
    for label in ["with categoricals", "no categoricals"]:
        ds_list = by_cats.get(label, [])
        if not ds_list:
            continue
        rmse_pct, f1_pct, ll_pct = quality_block_for(ds_list)
        sub_cols = []
        sub_labels = []
        if any(rmse_pct.values()):
            sub_cols.append(rmse_pct); sub_labels.append("RMSE")
        if any(f1_pct.values()):
            sub_cols.append(f1_pct); sub_labels.append("F1 macro")
        if any(ll_pct.values()):
            sub_cols.append(ll_pct); sub_labels.append("Log loss")
        if not sub_cols:
            continue
        cols.extend(sub_cols)
        col_labels.extend(sub_labels)
        col_groups.append((f"{label.title()}\n({len(ds_list)} datasets)", len(sub_cols)))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Quality by categorical presence  (avg % relative to best model)",
        os.path.join(out_dir, "by_categorical_quality.png"),
    )

    # ============================================================
    # Table 4: Speed by Categorical
    # ============================================================
    cols, col_labels, col_groups = [], [], []
    for label in ["with categoricals", "no categoricals"]:
        ds_list = by_cats.get(label, [])
        if not ds_list:
            continue
        cols.append(speed_for(ds_list))
        col_labels.append("fit speed")
        col_groups.append((f"{label.title()}\n({len(ds_list)} datasets)", 1))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Fit time by categorical presence  (× slower than fastest, 1× = best)",
        os.path.join(out_dir, "by_categorical_speed.png"),
        kind="speed",
    )

    # ============================================================
    # Table 5: Quality by Size
    # ============================================================
    cols, col_labels, col_groups = [], [], []
    for label in ["small", "medium", "large"]:
        ds_list = by_size.get(label, [])
        if not ds_list:
            continue
        rmse_pct, f1_pct, ll_pct = quality_block_for(ds_list)
        sub_cols, sub_labels = [], []
        if any(rmse_pct.values()):
            sub_cols.append(rmse_pct); sub_labels.append("RMSE")
        if any(f1_pct.values()):
            sub_cols.append(f1_pct); sub_labels.append("F1 macro")
        if any(ll_pct.values()):
            sub_cols.append(ll_pct); sub_labels.append("Log loss")
        if not sub_cols:
            continue
        cols.extend(sub_cols)
        col_labels.extend(sub_labels)
        size_hdr = {"small": f"Small (<{SMALL_THRESHOLD // 1000}K)",
                    "medium": f"Medium ({SMALL_THRESHOLD // 1000}K–{LARGE_THRESHOLD // 1000}K)",
                    "large": f"Large (≥{LARGE_THRESHOLD // 1000}K)"}[label]
        col_groups.append((f"{size_hdr}\n({len(ds_list)} datasets)", len(sub_cols)))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Quality by dataset size  (avg % relative to best model)",
        os.path.join(out_dir, "by_size_quality.png"),
    )

    # ============================================================
    # Table 6: Speed by Size
    # ============================================================
    cols, col_labels, col_groups = [], [], []
    for label in ["small", "medium", "large"]:
        ds_list = by_size.get(label, [])
        if not ds_list:
            continue
        cols.append(speed_for(ds_list))
        col_labels.append("fit speed")
        size_hdr = {"small": f"Small (<{SMALL_THRESHOLD // 1000}K)",
                    "medium": f"Medium ({SMALL_THRESHOLD // 1000}K–{LARGE_THRESHOLD // 1000}K)",
                    "large": f"Large (≥{LARGE_THRESHOLD // 1000}K)"}[label]
        col_groups.append((f"{size_hdr}\n({len(ds_list)} datasets)", 1))
    table = [[c.get(m) for c in cols] for m in models]
    render_table(
        table, models, col_labels, col_groups,
        "Fit time by dataset size  (× slower than fastest, 1× = best)",
        os.path.join(out_dir, "by_size_speed.png"),
        kind="speed",
    )

    # ============================================================
    # SUMMARY TABLE: one row per model, six metric columns total.
    # This is the table embedded in the README; the other six are
    # available for deeper inspection.
    # ============================================================
    all_ds = list(datasets.keys())
    reg_ds_all = [d for d in all_ds if datasets[d]["task"] == "regression"]
    bin_ds_all = [d for d in all_ds if datasets[d]["task"] == "binary"]
    mul_ds_all = [d for d in all_ds if datasets[d]["task"] == "multiclass"]

    sum_cols = []
    sum_col_labels = []
    sum_col_kinds = []

    if reg_ds_all:
        sum_cols.append(pct_vs_best(rmse, reg_ds_all, lower_is_better=True))
        sum_col_labels.append("RMSE")
        sum_col_kinds.append("pct")
    if bin_ds_all:
        sum_cols.append(pct_vs_best(f1, bin_ds_all, lower_is_better=False))
        sum_col_labels.append("F1 macro")
        sum_col_kinds.append("pct")
        sum_cols.append(pct_vs_best(ll, bin_ds_all, lower_is_better=True))
        sum_col_labels.append("Log loss")
        sum_col_kinds.append("pct")
    if mul_ds_all:
        sum_cols.append(pct_vs_best(f1, mul_ds_all, lower_is_better=False))
        sum_col_labels.append("F1 macro")
        sum_col_kinds.append("pct")
        sum_cols.append(pct_vs_best(ll, mul_ds_all, lower_is_better=True))
        sum_col_labels.append("Log loss")
        sum_col_kinds.append("pct")
    # Single speed column across all datasets.
    sum_cols.append(multiple_vs_best(speed, all_ds))
    sum_col_labels.append("fit time")
    sum_col_kinds.append("speed")

    sum_groups = [
        ("Regression", 1 if reg_ds_all else 0),
        ("Binary", 2 if bin_ds_all else 0),
        ("Multiclass", 2 if mul_ds_all else 0),
        ("Speed", 1),
    ]
    sum_groups = [(lab, span) for lab, span in sum_groups if span > 0]

    sum_table = [[c.get(m) for c in sum_cols] for m in models]
    render_table(
        sum_table, models, sum_col_labels, sum_groups,
        title="ChimeraBoost vs other GBMs",
        subtitle=f"avg % vs best  ·  fit time as × slowdown  ·  {len(all_ds)} OpenML datasets",
        out_path=os.path.join(out_dir, "summary.png"),
        col_kinds=sum_col_kinds,
    )

    print(f"Wrote 7 PNG tables to {out_dir}")


if __name__ == "__main__":
    main()
