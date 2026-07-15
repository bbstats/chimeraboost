"""Blended-strength vs slowdown Pareto plot (+ phone-readable text table).

One scalar "blended model strength" per model, plotted against fit-time slowdown,
with the Pareto frontier highlighted. This is the single number we steer by: a
model is only worth shipping if it pushes the strength/speed frontier.

Blended strength is built from the SAME "% vs best on that task" columns the
headline summary uses (100 = best model on a dataset, averaged across datasets,
higher = better), so it stays consistent with summary.png / summarize.py:

    classification = (2/3) * Bin Brier%  +  (1/3) * Bin F1%     (weighted avg)
    blended        = HarmonicMean(Reg RMSE%, classification)

The harmonic mean (not arithmetic) is deliberate: it collapses toward the WEAKER
side, so a model that is excellent at classification but mediocre at regression
(ChimeraBoost today) is scored by its weak leg, not flattered by its strong one.
That makes "raise the blended number" mean "fix the worst task", which is the aim.

The other axis is Slowdown: mean fit-time multiple vs the fastest model on each
dataset (1.0x = fastest), straight from summarize's Speed column. Lower = better,
so the frontier we want is up-and-to-the-left (strong AND fast).

Run:
    python benchmarks/make_pareto.py                      # newest results json
    python benchmarks/make_pareto.py benchmarks/results/<stamp>.json
    python benchmarks/make_pareto.py --no-image           # text table only
"""
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402  (canonical aggregation: % vs best, speed multiple)


# Same palette as make_slowdown_hist so ChimeraBoost is the consistent blue.
MODEL_COLOR = {
    "ChimeraBoost": "#3b6fb0",
    "ChimeraBoostEns2": "#5b8fc8",
    "ChimeraBoostEns5": "#4070a8",
    "ChimeraBoostEns10": "#2b4a73",
    "CatBoost": "#d1495b",
    "sklearn_HGB": "#e0a32e",
    "XGBoost": "#8d6cab",
    "LightGBM": "#5a9e6f",
}

# Weights for the classification half of the blend.
W_BRIER = 2.0 / 3.0
W_F1 = 1.0 / 3.0


def _harmonic_mean(a, b):
    """Harmonic mean of two positive scores; None if either is missing/<=0."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return 2.0 * a * b / (a + b)


def blended_strength(cols):
    """{model: dict} with the blended strength and its parts, from summarize cols.

    cols is summarize.aggregate(data)[0]: column-name -> {model: value}, where
    Reg RMSE% / Bin F1% / Bin Brier% are all "% vs best" (higher better) and
    Speed is the slowdown multiple (lower better).
    """
    rmse = cols["Reg RMSE%"]
    f1 = cols["Bin F1%"]
    brier = cols["Bin Brier%"]
    speed = cols["Speed"]
    models = set(rmse) | set(f1) | set(brier) | set(speed)

    out = {}
    for m in models:
        r = rmse.get(m)
        f = f1.get(m)
        b = brier.get(m)
        clf = (W_BRIER * b + W_F1 * f) if (b is not None and f is not None) else None
        out[m] = {
            "rmse": r, "f1": f, "brier": b, "clf": clf,
            "blended": _harmonic_mean(r, clf),
            "slowdown": speed.get(m),
        }
    return out


def pareto_frontier(scored):
    """Set of model names on the strength/speed Pareto frontier.

    A model is dominated if some other model is at least as strong AND at least
    as fast, with at least one strictly better. Frontier = the non-dominated set
    (maximize blended, minimize slowdown).
    """
    usable = {m: s for m, s in scored.items()
              if s["blended"] is not None and s["slowdown"] is not None}
    front = set()
    for m, s in usable.items():
        dominated = any(
            o != m
            and t["blended"] >= s["blended"] and t["slowdown"] <= s["slowdown"]
            and (t["blended"] > s["blended"] or t["slowdown"] < s["slowdown"])
            for o, t in usable.items()
        )
        if not dominated:
            front.add(m)
    return front


def _ordered_models(scored):
    order = summarize.MODEL_ORDER
    present = [m for m in order if m in scored]
    return present + [m for m in scored if m not in order]


def format_text(data, label=None):
    """Phone-readable text table: blended strength, its parts, slowdown, frontier."""
    cols, meta = summarize.aggregate(data)
    scored = blended_strength(cols)
    front = pareto_frontier(scored)
    # Strongest first so the "general to beat" is on top; ties broken by speed.
    models = sorted(
        _ordered_models(scored),
        key=lambda m: (-(scored[m]["blended"] or -1),
                       scored[m]["slowdown"] or 1e9),
    )

    def f(v, suf="", w=8):
        return f"{'--':>{w}}" if v is None else f"{v:>{w}.1f}{suf}"

    lines = []
    if label:
        lines.append(label)
    hdr = (f"{'Model':<14}{'Blended':>9}{'Slowdown':>10}{'Pareto':>8}   "
           f"{'RMSE%':>7}{'Clf':>7}{'(Brier':>8}{'F1)':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in models:
        s = scored[m]
        star = "  <-- ours" if m == "ChimeraBoost" else ""
        on = "yes" if m in front else "-"
        lines.append(
            f"{m:<14}{f(s['blended'],'',9)}{f(s['slowdown'],'x',9)}{on:>8}   "
            f"{f(s['rmse'],'%',6)}{f(s['clf'],'',7)}{f(s['brier'],'%',8)}"
            f"{f(s['f1'],'%',7)}{star}"
        )
    seeds = f" | {meta['seeds']} seeds" if meta.get("seeds") else ""
    lines.append(
        f"Grinsztajn et al. (2022) — {meta['n_total']} datasets "
        f"({meta['n_reg']} reg, {meta['n_bin']} binary){seeds}")
    lines.append(
        "Blended = HarmonicMean(RMSE%, 2/3*Brier% + 1/3*F1%) | "
        "all % vs best (higher=better) | Slowdown vs fastest (lower=better)")
    if meta.get("n_reg_excl"):
        n = meta["n_reg_excl"]
        lines.append(
            f"* RMSE excludes {n} dataset{'s' if n != 1 else ''} that every model "
            "solves near-perfectly (best NRMSE < 2%), where the percent-of-best "
            "ratio is meaningless.")
    return "\n".join(lines)


# Gap-to-best ticks (percentage points below 100), log-spaced; filtered per-plot
# to whatever range the data actually spans.
GAP_TICKS = [16, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.05]


def _gap_to_y(blended):
    """Log gap-to-best: y grows as the shortfall from 100 shrinks.

    All models here are near Bayes-optimal on Grinsztajn (ratios-to-best cluster
    in a ~97-100 band), so a linear axis crams every model's real, hard-won
    ranking gap into a couple of pixels. This is a monotonic display-only
    transform of the same blended number (floor gap at 0.05pp so it never hits
    -inf) - frontier/dominance are computed on raw blended, never on this.
    """
    return -math.log10(max(100.0 - blended, 0.05))


def _y_to_blended(y):
    return 100.0 - 10 ** (-y)


def render_image(data, out_path):
    cols, meta = summarize.aggregate(data)
    scored = blended_strength(cols)
    front = pareto_frontier(scored)
    pts = {m: s for m, s in scored.items()
           if s["blended"] is not None and s["slowdown"] is not None}

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)

    # Frontier step line (drawn under the points): sort by slowdown ascending.
    fr = sorted(front, key=lambda m: pts[m]["slowdown"])
    if len(fr) >= 2:
        fx = [pts[m]["slowdown"] for m in fr]
        fy = [_gap_to_y(pts[m]["blended"]) for m in fr]
        ax.plot(fx, fy, color="#888", linestyle="--", linewidth=1.4,
                zorder=1, label="Pareto frontier")

    for m, s in pts.items():
        color = MODEL_COLOR.get(m, "#777777")
        on_front = m in front
        is_us = m == "ChimeraBoost"
        y = _gap_to_y(s["blended"])
        ax.scatter(s["slowdown"], y,
                   s=260 if is_us else 170,
                   color=color, edgecolor="#222" if on_front else "white",
                   linewidth=1.8 if on_front else 1.0,
                   zorder=4 if is_us else 3, alpha=0.95)
        label = m
        # Per-model nudges to avoid label collisions.
        _offsets = {
            "ChimeraBoostEns2": (-9, 5, "right"),
        }
        ox, oy, ha = _offsets.get(m, (9, 5, "left"))
        ax.annotate(label, (s["slowdown"], y),
                    textcoords="offset points", xytext=(ox, oy), ha=ha,
                    fontsize=9.5,
                    fontweight="bold" if m.startswith("ChimeraBoost") else "normal",
                    color="#1a1a1a")

    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ticks = [1, 2, 3, 5, 8, 12, 20, 40]
    xs = [s["slowdown"] for s in pts.values()]
    lo, hi = min(xs), max(xs)
    ticks = [t for t in ticks if lo / 1.3 <= t <= hi * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    # Default log x-axis already puts the fastest (1×) on the LEFT, so the
    # best corner (strong + fast) is up-and-to-the-left. No inversion needed.
    ax.set_xlabel("← Slowdown — mean fit-time multiple vs fastest model "
                  "(log scale, lower = better)", fontsize=10.5)

    gaps = [max(100.0 - s["blended"], 0.05) for s in pts.values()]
    glo, ghi = min(gaps), max(gaps)
    gap_ticks = [g for g in GAP_TICKS if glo / 1.6 <= g <= ghi * 1.6]
    if len(gap_ticks) < 2:
        gap_ticks = GAP_TICKS
    ax.yaxis.set_major_locator(FixedLocator([_gap_to_y(100.0 - g) for g in gap_ticks]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{_y_to_blended(v):g}"))
    ax.set_ylabel("Blended model strength  (log gap-to-best, higher = better) →",
                  fontsize=10.5)

    # Up-and-to-the-left is best; annotate that corner.
    ax.text(0.02, 0.98, "stronger + faster", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color="#2b8a3e",
            fontstyle="italic")

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#ccc",
            zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    seeds_s = f"{meta['seeds']} seeds  ·  " if meta.get("seeds") else ""
    max_iter_s = f"{meta.get('max_iters', 2000):,}"
    patience_s = str(meta.get('patience', 50))
    tpm = meta.get("threads_per_model")
    threads_s = f"  ·  {tpm} cores/model" if tpm is not None else ""
    fig.suptitle("Blended strength vs slowdown — Grinsztajn et al. (2022)",
                 fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(
        f"Blended = HarmonicMean(RMSE%, ⅔·Brier% + ⅓·F1%)  ·  "
        f"{meta['n_total']} datasets ({meta['n_reg']} reg, {meta['n_bin']} bin)  ·  "
        f"{seeds_s}max {max_iter_s} trees  ·  patience {patience_s}  ·  20% val split"
        f"{threads_s}",
        fontsize=9.5, color="#555", pad=8)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    if meta.get("n_reg_excl"):
        n = meta["n_reg_excl"]
        fig.text(0.5, 0.012,
                 f"* RMSE% excludes {n} dataset{'s' if n != 1 else ''} every model "
                 "solves near-perfectly (best NRMSE < 2%), where the percent-of-best "
                 "ratio is meaningless.",
                 ha="center", fontsize=8, color="#777", style="italic")
    fig.tight_layout(rect=[0, 0.03 if meta.get("n_reg_excl") else 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=None,
                    help="results json (default: newest in benchmarks/results/)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for pareto.png (default: ../images/)")
    ap.add_argument("--no-image", action="store_true",
                    help="print the text table only, skip the PNG")
    args = ap.parse_args()

    path = args.json_path or summarize.latest_json()
    if not path:
        print("No results json found.")
        return
    data = summarize.load(path)

    print(format_text(data, f"# {os.path.basename(path)}"))

    if not args.no_image:
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "images")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pareto.png")
        render_image(data, out_path)
        print(f"\nWrote pareto.png to {out_dir}")


if __name__ == "__main__":
    main()
