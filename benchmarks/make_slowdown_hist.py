"""Plot the distribution of fit-time slowdowns vs the fastest model per dataset.

Reads a JSON file produced by `run_benchmarks.py --save`. For every dataset the
fastest model's mean fit time is the baseline (1.0×); every model's slowdown on
that dataset is `model_time / fastest_time`. We then draw one histogram per
model showing how those per-dataset slowdowns are distributed -- so you can see
not just the average multiple (as in summary.png) but the whole spread: where a
model is tied for fastest, and where it blows up.

Writes `slowdown_hist.png` into `images/`.

Run:
    python benchmarks/make_slowdown_hist.py benchmarks/results/<stamp>.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


MODEL_ORDER = ["ChimeraBoost", "ChimeraBoostEns10", "CatBoost", "sklearn_HGB",
               "XGBoost", "LightGBM"]

# Distinct, muted colors; ChimeraBoost gets the same blue it has in summary.png.
MODEL_COLOR = {
    "ChimeraBoost": "#3b6fb0",
    "ChimeraBoostEns10": "#2b4a73",   # darker blue, kin to ChimeraBoost
    "CatBoost": "#d1495b",
    "sklearn_HGB": "#e0a32e",
    "XGBoost": "#8d6cab",
    "LightGBM": "#5a9e6f",
}


def per_dataset_slowdowns(records):
    """{model: [slowdown on each dataset]}.

    Mean fit time per (dataset, model) over seeds; per dataset the fastest model
    is the 1.0× baseline; slowdown = model_time / fastest_time.
    """
    times = defaultdict(lambda: defaultdict(list))
    for r in records:
        times[r["dataset"]][r["model"]].append(r["fit_time"])
    mean_time = {ds: {m: float(np.mean(v)) for m, v in models.items()}
                 for ds, models in times.items()}

    slow = defaultdict(list)
    for ds, models in mean_time.items():
        vals = [t for t in models.values() if t > 0]
        if not vals:
            continue
        best = min(vals)
        for m, t in models.items():
            if t > 0:
                slow[m].append(t / best)
    return slow


def render(slowdowns, n_datasets, out_path):
    models = [m for m in MODEL_ORDER if m in slowdowns]

    # Shared log-spaced bins across all models so the panels are comparable.
    all_vals = np.concatenate([np.asarray(slowdowns[m]) for m in models])
    lo = max(all_vals.min(), 1.0)
    hi = all_vals.max()
    bins = np.logspace(np.log10(lo), np.log10(hi * 1.05), 22)

    # sharey so bar heights are comparable across panels: a model that is
    # fastest on most datasets towers at 1x, one that is rarely fastest does not.
    # Taller per-panel allotment + hspace so each panel can carry its own x-axis
    # tick labels without crowding the panel below.
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ticks = [t for t in ticks if lo <= t <= hi * 1.05]
    fmt = FuncFormatter(lambda v, _: f"{v:g}×")

    fig, axes = plt.subplots(len(models), 1, figsize=(8, 1.85 * len(models) + 1),
                             sharex=True, sharey=True, dpi=150)
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        vals = np.asarray(slowdowns[m])
        color = MODEL_COLOR.get(m, "#777777")
        ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="white",
                linewidth=0.5)
        med = float(np.median(vals))
        ax.axvline(med, color="#222", linestyle="--", linewidth=1.2, zorder=5)
        ax.text(0.99, 0.86, f"{m}", transform=ax.transAxes, ha="right",
                va="top", fontsize=11, fontweight="bold", color=color)
        ax.text(0.99, 0.62,
                f"median {med:.1f}×   ·   max {vals.max():.0f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#555")
        ax.set_xscale("log")
        ax.set_ylabel("datasets", fontsize=8.5, color="#666")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Plain "N×" ticks on every panel (not just the bottom one).
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="both", labelsize=9, labelbottom=True)

    axes[-1].set_xlabel("fit-time slowdown vs fastest model on each dataset (log scale)",
                        fontsize=10)

    fig.suptitle("Fit-time slowdown distribution — Grinsztajn et al. (2022)",
                 fontsize=13, fontweight="bold",
                 y=0.98)
    fig.text(0.5, 0.93, "dashed line = median   ·   1× = fastest on that dataset",
             ha="center", fontsize=9.5, color="#666")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to run_benchmarks.py --save .json output")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for the PNG (default: ../images/)")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "images")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    slowdowns = per_dataset_slowdowns(data["records"])
    out_path = os.path.join(out_dir, "slowdown_hist.png")
    render(slowdowns, len(data["datasets"]), out_path)
    print(f"Wrote slowdown_hist.png to {out_dir}")


if __name__ == "__main__":
    main()
