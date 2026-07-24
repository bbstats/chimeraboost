"""TabArena-Lite Pareto: Elo vs (train + predict) time, default configs.

Plots one aggregate point per method from the official TabArena-Lite leaderboard
(default config, 8-fold bagged): TabArena-Lite Elo against median train + predict
time per 1K rows, with the Pareto frontier highlighted. Among the default-config
gradient-boosting / tree baselines, ChimeraBoost sits on the frontier — ahead of
XGBoost and LightGBM on both Elo and speed, behind only CatBoost (at ~10x its time).

Numbers are the aggregate leaderboard figures (51 tasks); update DATA below if the
leaderboard is regenerated.

Run:
    python benchmarks/make_tabarena_pareto.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Same palette as make_pareto.py so ChimeraBoost stays the consistent blue.
MODEL_COLOR = {
    "ChimeraBoost": "#3b6fb0",
    "CatBoost": "#d1495b",
    "XGBoost": "#8d6cab",
    "LightGBM": "#5a9e6f",
    "RandomForest": "#e0a32e",
    "Linear": "#6c757d",
}

# TabArena-Lite, default config, bagged. (elo, elo_plus, elo_minus, train_s/1K, predict_s/1K)
# Refreshed 2026-07-23 (51-task run, 0/51 fail, fresh fits) on 0.23.0, after the
# predict-side engineering releases 0.21.0-0.23.0 (per-member cat-transform sharing,
# cross-block cast reuse, serial kernel twins for tiny batches -- all bit-identical
# on default paths). Elo is relative, so the whole pool was re-read from the
# regenerated leaderboard.
# HONEST READ: every Elo in the pool is byte-identical to the 2026-07-20 read
# (ChimeraBoost 1278, rank 31/68) -- exactly what bit-identical engineering should
# do to a sealed holdout. The payoff is the timing column: predict 0.162 -> 0.125
# s/1K (-23%), recovering most of the 0.109 -> 0.162 regression the 0.20.0
# cat-cross columns introduced. Train 0.83 -> 0.82 = flat (Lite's median task is
# small and overhead-dominated -- don't read TabArena fit speed from it).
DATA = {
    "CatBoost":     (1348, 42, 43, 6.70, 0.088),
    "ChimeraBoost": (1278, 55, 57, 0.82, 0.125),
    "XGBoost":      (1187, 54, 53, 2.06, 0.122),
    "LightGBM":     (1155, 50, 46, 2.20, 0.171),
    "RandomForest": (1000, 58, 58, 0.43, 0.053),
    "Linear":       (813, 81, 108, 1.23, 0.115),
}


def total_time(m):
    _, _, _, tr, pr = DATA[m]
    return tr + pr


def elo(m):
    return DATA[m][0]


def pareto_frontier(models):
    """Non-dominated set: maximize Elo, minimize total time."""
    front = set()
    for m in models:
        dominated = any(
            o != m
            and elo(o) >= elo(m) and total_time(o) <= total_time(m)
            and (elo(o) > elo(m) or total_time(o) < total_time(m))
            for o in models
        )
        if not dominated:
            front.add(m)
    return front


def render_image(out_path):
    models = list(DATA)
    front = pareto_frontier(models)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=220)

    # Frontier step line (under the points), sorted by time ascending.
    fr = sorted(front, key=total_time)
    if len(fr) >= 2:
        ax.plot([total_time(m) for m in fr], [elo(m) for m in fr],
                color="#888", linestyle="--", linewidth=1.4, zorder=1,
                label="Pareto frontier")

    for m in models:
        e, ep, em, _, _ = DATA[m]
        t = total_time(m)
        color = MODEL_COLOR.get(m, "#777777")
        on_front = m in front
        is_us = m == "ChimeraBoost"
        # Faint asymmetric Elo 95% CI bar.
        ax.errorbar(t, e, yerr=[[em], [ep]], fmt="none", ecolor=color,
                    elinewidth=1.0, capsize=3, alpha=0.45, zorder=2)
        ax.scatter(t, e, s=280 if is_us else 170, color=color,
                   edgecolor="#222" if on_front else "white",
                   linewidth=1.8 if on_front else 1.0,
                   zorder=4 if is_us else 3, alpha=0.95)
        # Label nudges to avoid collisions.
        _offsets = {
            "ChimeraBoost": (10, -14, "left"),
            "XGBoost": (10, 6, "left"),
            "LightGBM": (10, 6, "left"),
            "CatBoost": (-10, 6, "right"),
            "RandomForest": (10, 6, "left"),
        }
        ox, oy, ha = _offsets.get(m, (9, 5, "left"))
        ax.annotate(m, (t, e), textcoords="offset points", xytext=(ox, oy),
                    ha=ha, fontsize=10,
                    fontweight="bold" if is_us else "normal", color="#1a1a1a")

    xs = [total_time(m) for m in models]
    ax.set_xlim(0, max(xs) * 1.1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}s"))

    # Fast (low time) on the LEFT, strong (high Elo) UP -> best corner up-left.
    ax.set_xlabel("Time", fontsize=10.5)
    ax.set_ylabel("Elo", fontsize=10.5)

    ax.text(0.02, 0.98, "stronger + faster", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color="#2b8a3e", fontstyle="italic")

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#ccc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Train + Predict Time vs TabArena-Lite Elo",
                 fontsize=13, fontweight="bold", y=0.98)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return front


def main():
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "images"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tabarena_pareto.png")
    front = render_image(out_path)

    # Phone-readable text table.
    print("TabArena-Lite — default configs (Elo vs train+predict time/1K rows)")
    print(f"{'Model':<14}{'Elo':>6}{'Time(s/1K)':>12}{'Pareto':>8}")
    print("-" * 40)
    for m in sorted(DATA, key=lambda x: -elo(x)):
        star = "  <-- ours" if m == "ChimeraBoost" else ""
        on = "yes" if m in front else "-"
        print(f"{m:<14}{elo(m):>6}{total_time(m):>12.2f}{on:>8}{star}")
    print(f"\nWrote tabarena_pareto.png to {out_dir}")


if __name__ == "__main__":
    main()
