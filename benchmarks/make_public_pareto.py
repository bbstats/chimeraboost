"""The PUBLIC strength-vs-speed chart (issue #37).

Differences from make_pareto.py, which stays the internal north star:

1. **Competitor-relative win rate.** Every ChimeraBoost operating point is
   scored against CatBoost and LightGBM only, never against its sibling rungs.
   The internal chart's field-relative rate moves whenever an arm is added, and
   with several of our rungs against two competitors most of any rung's
   opponents would be our own arms -- "wins N% of matchups" would largely be us
   beating ourselves. See summarize.winrate_vs_opponents.
2. **Two competitors, by design.** LightGBM is the speed reference and CatBoost
   the quality reference; XGBoost tracks LightGBM closely and RandomForest is
   not in the harness at all.
3. **Sealed source.** It is meant to run on the `pub:` suite, which is never
   read to justify a source change. Pointing it at a decision-suite run is
   supported for smoke-testing the renderer and is labelled as such on the
   figure, because a chart drawn on the suites we tune against is in-sample.

Usage:
    python benchmarks/make_public_pareto.py [results.json] [--no-image]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402
import make_pareto  # noqa: E402

COMPETITORS = ("CatBoost", "LightGBM")
IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "images")
OUT_PNG = os.path.join(IMAGES, "public_pareto.png")

# Arms shown on the public chart. The fast rung is deliberately absent for now
# (Nathan, issue #37); it was also left uncharted on the TabArena figure.
PUBLIC_ARMS = ("ChimeraBoost", "ChimeraBoostEns8", "CatBoost", "LightGBM")


def score(data, arms=PUBLIC_ARMS, n_boot=make_pareto.N_BOOT):
    """{model: {winrate, wr_lo, wr_hi, slowdown}} on the competitor-relative axis."""
    cols, meta = summarize.aggregate(data)
    primary = summarize.primary_scores(data)
    keep = set(arms)
    primary = {ds: {m: v for m, v in s.items() if m in keep}
               for ds, s in primary.items()}
    primary = {ds: s for ds, s in primary.items() if len(s) >= 2}

    opponents = [c for c in COMPETITORS if c in keep]
    rates = summarize.winrate_vs_opponents(primary, opponents)
    ci = summarize.bootstrap_winrate_vs_opponents_ci(
        primary, opponents, n_boot=n_boot)
    speed = cols["Speed"]

    out = {}
    for m in arms:
        if m not in rates:
            continue
        lo, hi = ci.get(m, (None, None))
        out[m] = {"winrate": rates[m], "wr_lo": lo, "wr_hi": hi,
                  "slowdown": speed.get(m)}
    meta["n_h2h"] = len(primary)
    meta["sealed"] = all(d.startswith("pub:") for d in data["datasets"])
    return out, meta


def text_table(scored, meta):
    front = make_pareto.pareto_frontier(scored)
    lines = []
    if not meta.get("sealed"):
        lines.append(
            "NOT A PUBLISHABLE READ: this run is not the sealed pub: suite, so "
            "these\nnumbers are in-sample for anything we tune. Renderer smoke "
            "test only.\n")
    lines.append(f"{'Model':<22}{'win% vs competitors':>21}{'95% CI':>16}"
                 f"{'slowdown':>11}  frontier")
    lines.append("-" * 76)
    for m, s in sorted(scored.items(), key=lambda kv: -(kv[1]["winrate"] or 0)):
        ci = (f"[{s['wr_lo']:.0f}-{s['wr_hi']:.0f}]"
              if s["wr_lo"] is not None else "--")
        sl = f"{s['slowdown']:.1f}x" if s["slowdown"] is not None else "--"
        lines.append(f"{m:<22}{s['winrate']:>20.1f}%{ci:>16}{sl:>11}"
                     f"  {'yes' if m in front else ''}")
    lines.append("")
    lines.append(f"{meta['n_h2h']} datasets scored | win rate = % of "
                 f"(dataset x competitor) matchups won,")
    lines.append("primary metric RMSE (regression) / Brier (classification), "
                 "ties count 1/2.")
    lines.append("Opponents are CatBoost and LightGBM only -- sibling rungs are "
                 "never opponents,")
    lines.append("so adding a rung does not move any other row.")
    return "\n".join(lines)


def render(scored, meta, path=OUT_PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = make_pareto.pareto_frontier(scored)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    for m, s in scored.items():
        if s["slowdown"] is None:
            continue
        color = make_pareto.MODEL_COLOR.get(m, "#777777")
        yerr = None
        if s["wr_lo"] is not None:
            yerr = [[s["winrate"] - s["wr_lo"]], [s["wr_hi"] - s["winrate"]]]
        ax.errorbar(s["slowdown"], s["winrate"], yerr=yerr, fmt="o",
                    ms=11 if m in front else 8, color=color,
                    ecolor=color, elinewidth=1.2, capsize=3, alpha=0.95,
                    zorder=3)
        ax.annotate(m, (s["slowdown"], s["winrate"]), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, color=color)

    fp = sorted((scored[m]["slowdown"], scored[m]["winrate"]) for m in front)
    if len(fp) > 1:
        ax.plot([p[0] for p in fp], [p[1] for p in fp], "--", lw=1,
                color="#999999", zorder=1)

    ax.axhline(50, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("fit-time slowdown vs fastest (log scale)")
    ax.set_ylabel("% of matchups won vs CatBoost + LightGBM")
    title = "ChimeraBoost - strength vs speed"
    if not meta.get("sealed"):
        title += "  [SMOKE TEST - not the sealed suite]"
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=None)
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()

    path = args.results or summarize.latest_json()
    if not path:
        ap.error("no results JSON found")
    data = summarize.load(path)
    scored, meta = score(data)
    if not scored:
        ap.error("no chartable arms in that run "
                 f"(need some of {list(PUBLIC_ARMS)})")
    print(f"# {os.path.basename(path)}\n")
    print(text_table(scored, meta))
    if not args.no_image:
        print(f"\nwrote {render(scored, meta)}")


if __name__ == "__main__":
    main()
