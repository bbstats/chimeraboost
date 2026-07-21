"""Strength vs slowdown Pareto plot (+ phone-readable text tables).

One scalar strength per model, plotted against fit-time slowdown, with the
Pareto frontier highlighted. This is the chart we steer by: a model is only
worth shipping if it pushes the strength/speed frontier.

Headline strength axis (STRENGTH_VIZ_PLAN, resolved by Nathan 2026-07-18):
**head-to-head win rate** — the percent of (dataset x opponent) matchups a
model wins on that dataset's primary metric (RMSE for regression, Brier for
classification; exact ties count 1/2 each). 50% = mid-pack; with complete
data it equals (k - mean_rank)/(k - 1), i.e. mean rank in friendlier units.
The old blended-% axis saturated: on near-Bayes-optimal tabular data every
strong model lands within ~2% of best, so ratios-to-best cluster at 99.x and
no axis transform can decompress them. Win rate is ordinal, so it spreads the
field by who actually beats whom. Whiskers are 95% bootstrap CIs (resampling
datasets). A companion figure (winrate_matrix.png) shows the full pairwise
matrix; the axis scalar is that matrix's row mean, so the two figures agree.

Blended strength stays as the DIAGNOSTIC (text table + --metric blended):

    classification = (2/3) * Bin Brier%  +  (1/3) * Bin F1%     (weighted avg)
    blended        = HarmonicMean(Reg RMSE%, classification)

with every % being "% vs best on that task" from summarize.py. The harmonic
mean collapses toward the WEAKER side, so it still answers "which leg is
weak" — it just no longer carries the headline axis. Ship-gating is unchanged
(sign tests on the decision suites; see /experiment).

The other axis is Slowdown: mean fit-time multiple vs the fastest model on
each dataset (1.0x = fastest), straight from summarize's Speed column. Lower =
better, so the frontier we want is up-and-to-the-left (strong AND fast).

Run:
    python benchmarks/make_pareto.py                      # newest results json
    python benchmarks/make_pareto.py benchmarks/results/<stamp>.json
    python benchmarks/make_pareto.py --no-image           # text tables only
    python benchmarks/make_pareto.py --metric blended     # legacy blended axis
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402  (canonical aggregation + head-to-head machinery)


def _plt():
    """Lazy matplotlib import: the scoring/table half of this module (and its
    tests) must work without matplotlib installed — CI installs only the
    library deps. Only the two render_* functions need it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Same palette as make_tabarena_pareto / make_slowdown_hist so ChimeraBoost is
# the consistent blue.
MODEL_COLOR = {
    "ChimeraBoost": "#3b6fb0",
    "ChimeraBoostEns2": "#5b8fc8",
    "ChimeraBoostEns5": "#4070a8",
    "ChimeraBoostEns8": "#2b4a73",
    "ChimeraBoostEns10": "#2b4a73",
    "CatBoost": "#d1495b",
    "sklearn_HGB": "#e0a32e",
    "XGBoost": "#8d6cab",
    "LightGBM": "#5a9e6f",
}

# Compact names for matrix column headers / tight tables.
SHORT_NAME = {
    "ChimeraBoost": "ChimB",
    "ChimeraBoostEns2": "Ens2",
    "ChimeraBoostEns5": "Ens5",
    "ChimeraBoostEns8": "Ens8",
    "ChimeraBoostEns10": "Ens10",
    "CatBoost": "CatB",
    "LightGBM": "LGBM",
    "sklearn_HGB": "HGB",
    "XGBoost": "XGB",
}

N_BOOT = 10000

# Weights for the classification half of the blend.
W_BRIER = 2.0 / 3.0
W_F1 = 1.0 / 3.0


def _short(m):
    return SHORT_NAME.get(m, m[:5])


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


def score_models(data, n_boot=N_BOOT):
    """(scored, meta, primary): everything both renderers + tables need.

    scored[model] carries the blended diagnostics plus winrate / wr_lo / wr_hi
    (95% bootstrap CI) / mean_rank; primary is summarize.primary_scores(data).
    """
    cols, meta = summarize.aggregate(data)
    scored = blended_strength(cols)
    primary = summarize.primary_scores(data)
    field = summarize.winrate_vs_field(primary)
    ci = summarize.bootstrap_winrate_ci(primary, n_boot=n_boot)
    ranks = summarize.mean_rank(summarize.per_dataset_ranks(primary))
    for m, s in scored.items():
        s["winrate"] = field.get(m)
        s["wr_lo"], s["wr_hi"] = ci.get(m, (None, None))
        s["mean_rank"] = ranks.get(m)
    meta["n_h2h"] = len(primary)
    meta["n_ties"] = summarize.n_tied_matchups(primary)
    return scored, meta, primary


def pareto_frontier(scored, key="winrate"):
    """Set of model names on the strength/speed Pareto frontier.

    A model is dominated if some other model is at least as strong (higher
    `key`) AND at least as fast, with at least one strictly better. Frontier =
    the non-dominated set (maximize key, minimize slowdown).
    """
    usable = {m: s for m, s in scored.items()
              if s.get(key) is not None and s["slowdown"] is not None}
    front = set()
    for m, s in usable.items():
        dominated = any(
            o != m
            and t[key] >= s[key] and t["slowdown"] <= s["slowdown"]
            and (t[key] > s[key] or t["slowdown"] < s["slowdown"])
            for o, t in usable.items()
        )
        if not dominated:
            front.add(m)
    return front


def _f(v, suf="", w=8, dec=1):
    return f"{'--':>{w}}" if v is None else f"{v:>{w}.{dec}f}{suf}"


def matrix_text(primary):
    """The head-to-head matrix as a phone-readable text block."""
    models, mat = summarize.winrate_matrix(primary)
    field = summarize.winrate_vs_field(primary)
    name_w = max(len(m) for m in models) + 2
    lines = ["Head-to-head matrix (% of datasets where ROW beats COLUMN, ties ½):"]
    hdr = " " * name_w + "".join(f"{_short(m):>7}" for m in models) + f"{'vs field':>11}"
    lines.append(hdr)
    for i, m in enumerate(models):
        cells = "".join(
            f"{'--':>7}" if i == j else
            (f"{mat[i][j]:>7.0f}" if mat[i][j] is not None else f"{'?':>7}")
            for j in range(len(models)))
        lines.append(f"{m:<{name_w}}{cells}{field[m]:>10.1f}%")
    return "\n".join(lines)


def _caption_lines(meta):
    lines = []
    seeds = f" | {meta['seeds']} seeds" if meta.get("seeds") else ""
    excl = meta["n_total"] - meta["n_h2h"]
    excl_s = f" ({excl} near-solved excluded)" if excl else ""
    lines.append(
        f"{meta['suite']} — head-to-head on {meta['n_h2h']} of "
        f"{meta['n_total']} datasets{excl_s} "
        f"({meta['n_reg']} reg, {meta['n_bin']} binary){seeds}")
    lines.append(
        "Win rate = % of (dataset × opponent) matchups won on the primary "
        "metric (RMSE reg / Brier clf) | 50% = mid-pack | CI = 95% bootstrap "
        "over datasets")
    if meta.get("n_ties"):
        n = meta["n_ties"]
        lines.append(f"* {n} matchup{'s' if n != 1 else ''} tied exactly "
                     "(counted ½ each side).")
    return lines


def format_text(scored, meta, primary, label=None, metric="winrate"):
    """Phone-readable tables: headline axis, head-to-head matrix, diagnostics."""
    lines = []
    if label:
        lines.append(label)

    if metric == "winrate":
        front = pareto_frontier(scored, key="winrate")
        models = sorted(
            [m for m in scored],
            key=lambda m: (-(scored[m]["winrate"] if scored[m]["winrate"]
                             is not None else -1),
                           scored[m]["slowdown"] or 1e9))
        hdr = (f"{'Model':<18}{'WinRate':>8}{'95% CI':>15}{'MeanRank':>10}"
               f"{'Slowdown':>10}{'Pareto':>8}")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for m in models:
            s = scored[m]
            ci = ("--" if s["wr_lo"] is None
                  else f"[{s['wr_lo']:.1f}, {s['wr_hi']:.1f}]")
            star = "  <-- ours" if m == "ChimeraBoost" else ""
            on = "yes" if m in front else "-"
            lines.append(
                f"{m:<18}{_f(s['winrate'], '%', 7)}{ci:>15}"
                f"{_f(s['mean_rank'], '', 10, 2)}{_f(s['slowdown'], 'x', 9)}"
                f"{on:>8}{star}")
        lines.append("")
        lines.append(matrix_text(primary))
        lines.append("")
        lines.append("Diagnostics — blended % vs best (higher=better; "
                     "harmonic mean tracks the weak leg):")
    else:
        front = pareto_frontier(scored, key="blended")
        lines.append("Blended strength (--metric blended; % vs best, "
                     "higher=better):")

    dmodels = sorted(
        [m for m in scored],
        key=lambda m: (-(scored[m]["blended"] or -1), scored[m]["slowdown"] or 1e9))
    hdr = (f"{'Model':<18}{'Blended':>9}{'Slowdown':>10}"
           + (f"{'Pareto':>8}" if metric == "blended" else "")
           + f"   {'RMSE%':>7}{'Clf':>7}{'(Brier':>8}{'F1)':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in dmodels:
        s = scored[m]
        mid = f"{'yes' if m in front else '-':>8}" if metric == "blended" else ""
        lines.append(
            f"{m:<18}{_f(s['blended'], '', 9)}{_f(s['slowdown'], 'x', 9)}{mid}   "
            f"{_f(s['rmse'], '%', 6)}{_f(s['clf'], '', 7)}{_f(s['brier'], '%', 8)}"
            f"{_f(s['f1'], '%', 7)}")
    lines.append("")
    lines.extend(_caption_lines(meta))
    if meta.get("n_reg_excl"):
        n = meta["n_reg_excl"]
        lines.append(
            f"* RMSE% (and the reg ranking) excludes {n} "
            f"dataset{'s' if n != 1 else ''} that every model solves "
            "near-perfectly (best NRMSE < 2%).")
    return "\n".join(lines)


# Gap-to-best ticks (percentage points below 100), log-spaced; filtered per-plot
# to whatever range the data actually spans. (--metric blended only.)
GAP_TICKS = [16, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.05]


def _gap_to_y(blended):
    """Log gap-to-best: y grows as the shortfall from 100 shrinks.

    Display-only transform for the legacy blended axis (kept reachable via
    --metric blended); frontier/dominance are computed on raw values, never on
    this. The transform can't fix blended's structural saturation — that's why
    the headline moved to win rate.
    """
    return -math.log10(max(100.0 - blended, 0.05))


def _y_to_blended(y):
    return 100.0 - 10 ** (-y)


def _subtitle(meta):
    seeds_s = f"{meta['seeds']} seeds  ·  " if meta.get("seeds") else ""
    tpm = meta.get("threads_per_model")
    threads_s = f"  ·  {tpm} cores/model" if tpm is not None else ""
    return (f"{seeds_s}max {meta.get('max_iters', 2000):,} trees  ·  "
            f"patience {meta.get('patience', 50)}  ·  20% val split{threads_s}")


def render_image(scored, meta, out_path, metric="winrate"):
    front = pareto_frontier(scored, key=metric)
    ykey = "winrate" if metric == "winrate" else "blended"
    pts = {m: s for m, s in scored.items()
           if s.get(ykey) is not None and s["slowdown"] is not None}

    def to_y(v):
        return v if metric == "winrate" else _gap_to_y(v)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)

    if metric == "winrate":
        # Recessive mid-pack reference: 50% = wins as many matchups as it loses.
        ax.axhline(50.0, color="#bbb", linestyle=":", linewidth=1.0, zorder=0)

    # Frontier step line (drawn under the points): sort by slowdown ascending.
    fr = sorted(front, key=lambda m: pts[m]["slowdown"])
    if len(fr) >= 2:
        fx = [pts[m]["slowdown"] for m in fr]
        fy = [to_y(pts[m][ykey]) for m in fr]
        ax.plot(fx, fy, color="#888", linestyle="--", linewidth=1.4,
                zorder=1, label="Pareto frontier")

    for m, s in pts.items():
        color = MODEL_COLOR.get(m, "#777777")
        on_front = m in front
        is_us = m == "ChimeraBoost"
        y = to_y(s[ykey])
        if metric == "winrate" and s["wr_lo"] is not None:
            ax.errorbar(s["slowdown"], y,
                        yerr=[[y - s["wr_lo"]], [s["wr_hi"] - y]],
                        fmt="none", ecolor=color, elinewidth=1.0, capsize=3,
                        alpha=0.45, zorder=2)
        ax.scatter(s["slowdown"], y,
                   s=260 if is_us else 170,
                   color=color, edgecolor="#222" if on_front else "white",
                   linewidth=1.8 if on_front else 1.0,
                   zorder=4 if is_us else 3, alpha=0.95)
        # Per-model nudges to avoid label collisions.
        _offsets = {
            "ChimeraBoostEns2": (-9, 5, "right"),
            "ChimeraBoostEns8": (-13, -4, "right"),
        }
        ox, oy, ha = _offsets.get(m, (9, 5, "left"))
        ax.annotate(m, (s["slowdown"], y),
                    textcoords="offset points", xytext=(ox, oy), ha=ha,
                    fontsize=9.5,
                    fontweight="bold" if m.startswith("ChimeraBoost") else "normal",
                    color="#1a1a1a")

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    xs = [s["slowdown"] for s in pts.values()]
    ax.set_xlim(0, max(xs) * 1.09)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    # Fastest (1×) on the LEFT, so the best corner (strong + fast) is
    # up-and-to-the-left. No inversion needed.
    ax.set_xlabel("← Slowdown — mean fit-time multiple vs fastest model "
                  "(lower = better)", fontsize=10.5)

    if metric == "winrate":
        ylo = min(s["wr_lo"] if s["wr_lo"] is not None else s["winrate"]
                  for s in pts.values())
        yhi = max(s["wr_hi"] if s["wr_hi"] is not None else s["winrate"]
                  for s in pts.values())
        ax.set_ylim(max(0.0, ylo - 5.0), min(100.0, yhi + 5.0))
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_ylabel("Head-to-head win rate — % of matchups won "
                      "(higher = better) →", fontsize=10.5)
        excl = meta["n_total"] - meta["n_h2h"]
        excl_s = f" ({excl} near-solved excl.)" if excl else ""
        ties_s = f"  ·  {meta['n_ties']} exact ties = ½" if meta.get("n_ties") else ""
        sub = (f"Win rate = % of (dataset × opponent) matchups won on the "
               f"primary metric (RMSE reg / Brier clf)  ·  whiskers = 95% "
               f"bootstrap CI\n{meta['n_h2h']} of {meta['n_total']} datasets"
               f"{excl_s}  ·  {_subtitle(meta)}{ties_s}")
        title = f"Head-to-head win rate vs slowdown — {meta['suite']}"
    else:
        gaps = [max(100.0 - s["blended"], 0.05) for s in pts.values()]
        glo, ghi = min(gaps), max(gaps)
        gap_ticks = [g for g in GAP_TICKS if glo / 1.6 <= g <= ghi * 1.6]
        if len(gap_ticks) < 2:
            gap_ticks = GAP_TICKS
        ax.yaxis.set_major_locator(
            FixedLocator([_gap_to_y(100.0 - g) for g in gap_ticks]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{_y_to_blended(v):g}"))
        ax.set_ylabel("Blended model strength  (log gap-to-best, "
                      "higher = better) →", fontsize=10.5)
        sub = (f"Blended = HarmonicMean(RMSE%, ⅔·Brier% + ⅓·F1%)  ·  "
               f"{meta['n_total']} datasets ({meta['n_reg']} reg, "
               f"{meta['n_bin']} bin)  ·  {_subtitle(meta)}")
        title = f"Blended strength vs slowdown — {meta['suite']}"

    # Up-and-to-the-left is best; annotate that corner.
    ax.text(0.02, 0.98, "stronger + faster", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color="#2b8a3e",
            fontstyle="italic")

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#ccc",
            zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(sub, fontsize=9, color="#555", pad=8)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    foot = None
    if metric == "blended" and meta.get("n_reg_excl"):
        n = meta["n_reg_excl"]
        foot = (f"* RMSE% excludes {n} dataset{'s' if n != 1 else ''} every "
                "model solves near-perfectly (best NRMSE < 2%), where the "
                "percent-of-best ratio is meaningless.")
        fig.text(0.5, 0.012, foot, ha="center", fontsize=8, color="#777",
                 style="italic")
    fig.tight_layout(rect=[0, 0.03 if foot else 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_matrix(primary, meta, out_path):
    """Companion figure: who beats whom, every pairwise win rate as a matrix.

    Cell = % of datasets where the ROW model beats the COLUMN model (ties ½).
    Diverging fill around the 50% midpoint (blue = row wins, red = row loses);
    every cell is direct-labeled so color is reinforcement, never the only
    encoding. The bold right-hand column is the row mean = the win-rate axis
    of pareto.png, so the two figures agree by construction.
    """
    import numpy as np

    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    models, mat = summarize.winrate_matrix(primary)
    field = summarize.winrate_vs_field(primary)
    k = len(models)
    arr = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if mat[i][j] is not None:
                arr[i, j] = mat[i][j]

    cmap = LinearSegmentedColormap.from_list(
        "winlose", ["#d1495b", "#f6f5f2", "#3b6fb0"])
    cmap.set_bad("#eceae6")

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=220)
    ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=0, vmax=100,
              aspect="auto")

    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10,
                        color="#999")
                continue
            v = arr[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(v / 100.0)
            ink = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#1a1a1a"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=10.5, color=ink)

    # Row mean = the pareto.png y-axis, bold, in its own right-hand gutter.
    for i, m in enumerate(models):
        ax.text(k - 0.28, i, f"{field[m]:.1f}%", ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(k - 0.28, -0.62, "vs field", ha="left", va="center", fontsize=9.5,
            color="#555", fontstyle="italic")

    ax.set_xlim(-0.5, k + 0.45)
    ax.set_xticks(range(k))
    ax.set_xticklabels([_short(m) for m in models], fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(k))
    ax.set_yticklabels(models, fontsize=10)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("ChimeraBoost"):
            lbl.set_fontweight("bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    excl = meta["n_total"] - meta["n_h2h"]
    excl_s = f" ({excl} near-solved excl.)" if excl else ""
    fig.suptitle("Who beats whom — head-to-head win rate",
                 fontsize=13, fontweight="bold", y=0.99)
    ax.set_title(
        f"cell = % of datasets where ROW beats COLUMN on the primary metric "
        f"(RMSE reg / Brier clf; ties ½)\nblue = row wins, red = row loses  ·  "
        f"{meta['suite']}, {meta['n_h2h']} of {meta['n_total']} datasets"
        f"{excl_s}  ·  bold = row mean (the pareto.png axis)",
        fontsize=8.5, color="#555", pad=30)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=None,
                    help="results json (default: newest in benchmarks/results/)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for the PNGs (default: ../images/)")
    ap.add_argument("--no-image", action="store_true",
                    help="print the text tables only, skip the PNGs")
    ap.add_argument("--metric", choices=["winrate", "blended"],
                    default="winrate",
                    help="headline axis: winrate (default) or the legacy "
                         "blended %% (writes pareto_blended.png)")
    args = ap.parse_args()

    path = args.json_path or summarize.latest_json()
    if not path:
        print("No results json found.")
        return
    data = summarize.load(path)

    scored, meta, primary = score_models(data)
    print(format_text(scored, meta, primary, f"# {os.path.basename(path)}",
                      metric=args.metric))

    if not args.no_image:
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "images")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        wrote = []
        if args.metric == "winrate":
            render_image(scored, meta, os.path.join(out_dir, "pareto.png"),
                         metric="winrate")
            wrote.append("pareto.png")
            render_matrix(primary, meta,
                          os.path.join(out_dir, "winrate_matrix.png"))
            wrote.append("winrate_matrix.png")
        else:
            render_image(scored, meta,
                         os.path.join(out_dir, "pareto_blended.png"),
                         metric="blended")
            wrote.append("pareto_blended.png")
        print(f"\nWrote {', '.join(wrote)} to {out_dir}")


if __name__ == "__main__":
    main()
