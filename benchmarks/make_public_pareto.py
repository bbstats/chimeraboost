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
3. **Public-suite source.** It is meant to run on the `pub:` suite. Pointing it
   at a decision-suite run is supported for smoke-testing the renderer and is
   labelled as such on the figure, because a chart drawn on the suites we tune
   against is in-sample and says nothing about generalisation.

The public suite is NOT sealed (that changed 2026-07-27). It is a validation
suite: read it freely, and it never blocks a ship. The one sealed holdout is
TabArena's full run, executed by its authors.

Usage:
    python benchmarks/make_public_pareto.py [results.json] [--no-image]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402
import make_pareto  # noqa: E402
import suite_weights  # noqa: E402

COMPETITORS = ("CatBoost", "LightGBM")
IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "images")
OUT_PNG = os.path.join(IMAGES, "public_pareto.png")

# The whole `quality` ladder, not just the default and the top rung. These are
# the five named operating points a user actually selects with quality=N
# (chimeraboost.sklearn_api.QUALITY_NAMES), mapped to the harness arms that
# reproduce each one exactly:
#
#   quality=1 fast      one booster fit, linear leaves pinned, no refit
#   quality=2 balanced  the full search, without the refit
#   quality=3 accurate  refit the early-stopping winner on all rows  <- default
#   quality=4 ensemble  5 bagged members
#   quality=5 max       8 bagged members
#
# Charting all five is the point: the reader picks a rung off the frontier,
# and a ladder with two rungs shown is not a ladder. Sibling rungs are never
# each other's opponents (see score()), so adding rungs cannot move any row.
QUALITY_ARMS = {
    "ChimeraBoostOneLin":  (1, "fast"),
    "ChimeraBoostNoRefit": (2, "balanced"),
    "ChimeraBoost":        (3, "accurate"),
    "ChimeraBoostEns5":    (4, "ensemble"),
    "ChimeraBoostEns8":    (5, "max"),
}
DEFAULT_RUNG = "ChimeraBoost"
# Only two rungs are charted (Nathan, 2026-07-27, for efficiency): the default
# and the ensemble one above it. Rungs 1, 2 and 5 come off. quality=5 was the
# most expensive arm on the board and scored IDENTICALLY to quality=4 across 22
# datasets -- 64.3% each, same interval, 45% more compute -- so charting it was
# paying for a duplicate point. quality=1 was beaten by LightGBM on strength
# while being 2.3x slower. Both stay runnable via --models for a one-off.
CHARTED_RUNGS = ("ChimeraBoost", "ChimeraBoostEns5")
PUBLIC_ARMS = CHARTED_RUNGS + COMPETITORS

# One blue family, light (fast) to dark (max), so the ladder reads as a ladder
# rather than five unrelated models.
RUNG_COLOR = {
    "ChimeraBoostOneLin":  "#a8cbe8",
    "ChimeraBoostNoRefit": "#6fa3d2",
    "ChimeraBoost":        "#3b6fb0",
    "ChimeraBoostEns5":    "#2b5183",
    "ChimeraBoostEns8":    "#17324f",
}


def display_name(model):
    """Chart/table label: the knob a user would set, not our internal arm name."""
    if model in QUALITY_ARMS:
        level, name = QUALITY_ARMS[model]
        suffix = ", default" if model == DEFAULT_RUNG else ""
        return f"quality={level} ({name}{suffix})"
    return model


def arm_color(model):
    return RUNG_COLOR.get(model) or make_pareto.MODEL_COLOR.get(model, "#777777")


def slowdown_stats(data, arms, weights=None):
    """{model: (mean, median)} fit-time multiple vs the fastest arm per dataset.

    The mean is what summarize reports, and on this suite it is not a
    representative number: CatBoost needs 2,883s against LightGBM's 3s on
    pub:fars, and that single 970x ratio drags its mean to 121x while its
    median dataset costs 53x. Ratios are right-skewed by construction -- a
    model can be 900x slower but never 900x faster -- so the median is the
    typical dataset and the mean is the whole-suite bill. Report both.
    """
    import statistics
    ft = {}
    for r in data["records"]:
        ft.setdefault(r["dataset"], {}).setdefault(r["model"], []).append(
            r["fit_time"])
    ratios = {m: [] for m in arms}
    for ds, per_model in ft.items():
        times = {m: sum(v) / len(v) for m, v in per_model.items() if m in arms}
        if len(times) < 2:
            continue
        best = min(times.values())
        if best <= 0:
            continue
        for m, t in times.items():
            ratios[m].append((t / best, ds))
    out = {}
    for m, v in ratios.items():
        if not v:
            continue
        vals = [r for r, _ in v]
        # `weights` is keyed by the full dataset key (pub:<name>), same as `ds`.
        wts = [1.0 if weights is None else weights.get(ds, 0.0) for _, ds in v]
        if weights is None:
            out[m] = (sum(vals) / len(vals), statistics.median(vals))
        else:
            out[m] = (suite_weights.weighted_mean(vals, wts),
                      suite_weights.weighted_median(vals, wts))
    return out


def _strip(ds_key):
    """`pub:foo` -> `foo`, so a dataset key matches a PUBLIC_DATASETS name."""
    return ds_key.split(":", 1)[1] if ":" in ds_key else ds_key


def public_weights():
    """{dataset key: weight} balancing task/size/cardinality, or None if the
    suite is not frozen. Keys are full `pub:<name>` dataset keys."""
    import run_benchmarks as rb
    if not rb.PUBLIC_DATASETS:
        return None, None
    facets = suite_weights.dataset_facets(rb.PUBLIC_DATASETS, rb.PUBLIC_FACETS)
    w = suite_weights.raked_weights(facets)
    return {f"pub:{k}": v for k, v in w.items()}, facets


def score(data, arms=PUBLIC_ARMS, n_boot=make_pareto.N_BOOT, weighted=True):
    """{model: {rank, rank_lo, rank_hi, winrate, slowdown, slowdown_mean}}.

    Y is weighted average rank (1 = best), computed COMPETITOR-RELATIVE: each of
    our rungs is ranked against CatBoost and LightGBM alone, never against a
    sibling rung. Ranking the whole field together would hand the stronger rung
    a free point every time it beat the weaker one, which is us beating
    ourselves. See suite_weights.competitor_relative_rank.
    """
    cols, meta = summarize.aggregate(data)
    primary = summarize.primary_scores(data)
    keep = set(arms)
    primary = {ds: {m: v for m, v in s.items() if m in keep}
               for ds, s in primary.items()}
    primary = {ds: s for ds, s in primary.items() if len(s) >= 2}

    weights, facets = (public_weights() if weighted else (None, None))
    if weights is not None:
        weights = {d: w for d, w in weights.items() if d in primary}

    opponents = [c for c in COMPETITORS if c in keep]
    rates = summarize.winrate_vs_opponents(primary, opponents)
    ours = [m for m in arms if m not in COMPETITORS
            and any(m in s for s in primary.values())]
    ranks = suite_weights.competitor_relative_rank(
        primary, ours, opponents, weights)
    rank_ci = suite_weights.bootstrap_competitor_relative_rank_ci(
        primary, ours, opponents, weights, n_boot=min(n_boot, 2000))
    stats = slowdown_stats(data, arms, weights)

    out = {}
    for m in arms:
        if m not in ranks:
            continue
        lo, hi = rank_ci.get(m, (None, None))
        mean_x, med_x = stats.get(m, (None, None))
        # The chart plots the MEDIAN slowdown: it is the typical dataset, and
        # the mean is a different claim (the whole-suite bill) that one 970x
        # outlier owns.
        out[m] = {"rank": ranks[m], "rank_lo": lo, "rank_hi": hi,
                  "winrate": rates.get(m), "slowdown": med_x,
                  "slowdown_mean": mean_x}
    meta["n_h2h"] = len(primary)
    meta["all_public"] = all(d.startswith("pub:") for d in data["datasets"])
    meta["weights"] = weights
    meta["facets"] = facets
    if weights:
        meta["ess"] = suite_weights.effective_sample_size(weights)
        meta["balance"] = suite_weights.facet_balance(
            {k: v for k, v in facets.items() if f"pub:{k}" in weights},
            {_strip(d): w for d, w in weights.items()})
    return out, meta


def frontier_of(scored):
    """Pareto frontier on (low rank, low slowdown). make_pareto's helper wants a
    higher-is-better y, so feed it the negated rank."""
    shim = {m: {"winrate": -s["rank"], "slowdown": s["slowdown"]}
            for m, s in scored.items() if s.get("slowdown") is not None}
    return make_pareto.pareto_frontier(shim)


def weights_table(meta):
    """The weight listing, so the aggregate is auditable rather than a black box."""
    w, facets = meta.get("weights"), meta.get("facets")
    if not w:
        return "UNWEIGHTED: the public suite is not frozen, every dataset counts once."
    n = len(w)
    lines = [f"Dataset weights -- task/size/cardinality raked to equal mass, "
             f"capped at [{suite_weights.WEIGHT_CAP[0]}x, {suite_weights.WEIGHT_CAP[1]}x].",
             f"Effective sample size {meta['ess']:.1f} of {n} datasets "
             f"(equal weighting would be {n}.0; balancing costs the difference, "
             f"widening error bars ~{100 * ((n / meta['ess']) ** 0.5 - 1):.0f}%).",
             ""]
    for facet, acc in meta["balance"].items():
        tgt = 1.0 / len(acc)
        got = ", ".join(f"{k} {v:.3f}" for k, v in sorted(acc.items()))
        lines.append(f"  {facet:5} target {tgt:.3f} each -> {got}")
    lines.append("")
    lines.append(f"  {'dataset':46}{'task':11}{'size':6}{'card':6}{'rel wt':>7}")
    for ds, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        f = facets[_strip(ds)]
        lines.append(f"  {_strip(ds)[:44]:46}{f['task']:11}{f['size']:6}"
                     f"{f['card']:6}{wt * n:>7.2f}")
    return "\n".join(lines)


def text_table(scored, meta):
    front = frontier_of(scored)
    lines = []
    if not meta.get("all_public"):
        lines.append(
            "NOT THE PUBLIC SUITE: this run contains non-pub: datasets, so "
            "these numbers\nare in-sample for anything we tune. Renderer smoke "
            "test only.\n")
    lines.append(f"{'Model':<30}{'avg rank':>10}{'95% CI':>14}{'win%':>8}"
                 f"{'median x':>10}{'mean x':>9}  frontier")
    lines.append("-" * 88)
    for m, s in sorted(scored.items(), key=lambda kv: kv[1]["rank"]):
        ci = (f"[{s['rank_lo']:.2f}-{s['rank_hi']:.2f}]"
              if s["rank_lo"] is not None else "--")
        wr = f"{s['winrate']:.1f}%" if s.get("winrate") is not None else "--"
        sl = f"{s['slowdown']:.1f}x" if s["slowdown"] is not None else "--"
        mn = (f"{s['slowdown_mean']:.1f}x"
              if s.get("slowdown_mean") is not None else "--")
        lines.append(f"{display_name(m):<30}{s['rank']:>10.2f}{ci:>14}{wr:>8}"
                     f"{sl:>10}{mn:>9}  {'yes' if m in front else ''}")
    lines.append("")
    lines.append(f"{meta['n_h2h']} datasets scored, weighted. Average rank is vs "
                 "CatBoost + LightGBM only")
    lines.append("(1 = best of three, ties share the midrank); win% is the same "
                 "matchups as a rate.")
    lines.append("Both are competitor-relative: our rungs are never each "
                 "other's opponents, so")
    lines.append("adding or dropping a rung cannot move any other row.")
    lines.append("Slowdown is the fit-time multiple vs the fastest arm on each "
                 "dataset. The chart")
    lines.append("plots the median; the mean is shown because one dataset "
                 "(fars: 2883s vs 3s)")
    lines.append("alone doubles CatBoost's.")
    return "\n".join(lines)


def render(scored, meta, path=OUT_PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = frontier_of(scored)
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)

    pts = {m: s for m, s in scored.items() if s["slowdown"] is not None}
    xmax = max(s["slowdown"] for s in pts.values()) if pts else 1.0
    # Linear, not log: a log axis labelled 10^0 / 10^1 hid how far right
    # CatBoost actually sits. Headroom so the last label is not clipped.
    ax.set_xlim(-0.03 * xmax, 1.16 * xmax)
    # Rank 1 is best, so the axis runs downward and "up" still means "better".
    ax.invert_yaxis()

    # Points can land on the same rank, which overprints their labels into mush.
    # Place labels in x order and drop one below its point on a collision.
    order = sorted(pts, key=lambda m: pts[m]["slowdown"])
    yspan = (max(s["rank"] for s in pts.values())
             - min(s["rank"] for s in pts.values())) or 1.0
    placed = []
    for m in order:
        s = pts[m]
        color = arm_color(m)
        yerr = None
        if s["rank_lo"] is not None:
            yerr = [[s["rank"] - s["rank_lo"]], [s["rank_hi"] - s["rank"]]]
        ax.errorbar(s["slowdown"], s["rank"], yerr=yerr, fmt="o",
                    ms=12 if m in front else 9, color=color,
                    ecolor=color, elinewidth=1.4, capsize=4, alpha=0.95,
                    zorder=3)
        x, y = s["slowdown"], s["rank"]
        collides = any(abs(y - py) < 0.12 * yspan and abs(x - px) < 0.28 * xmax
                       for px, py in placed)
        right = x > 0.66 * xmax
        ax.annotate(display_name(m), (x, y), textcoords="offset points",
                    xytext=(-12 if right else 12, 16 if collides else -8),
                    ha="right" if right else "left",
                    fontsize=11.5, color=color)
        if not collides:
            placed.append((x, y))

    fp = sorted((scored[m]["slowdown"], scored[m]["rank"]) for m in front)
    if len(fp) > 1:
        ax.plot([p[0] for p in fp], [p[1] for p in fp], "--", lw=1.2,
                color="#999999", zorder=1)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x" if v > 0 else ""))
    ax.set_xlabel("fit-time slowdown vs fastest (median dataset)", fontsize=13)
    ax.set_ylabel("average rank  (1 = best)", fontsize=13)
    ax.tick_params(labelsize=12)
    title = "ChimeraBoost - strength vs speed"
    if not meta.get("all_public"):
        title += "  [SMOKE TEST - not the public suite]"
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=None)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--no-weights", action="store_true",
                    help="aggregate with every dataset counting once, ignoring "
                         "the task/size/cardinality balance (for auditing what "
                         "the weighting changed).")
    ap.add_argument("--hide-weight-table", action="store_true",
                    help="suppress the per-dataset weight listing.")
    args = ap.parse_args()

    path = args.results or summarize.latest_json()
    if not path:
        ap.error("no results JSON found")
    data = summarize.load(path)
    scored, meta = score(data, weighted=not args.no_weights)
    if not scored:
        ap.error("no chartable arms in that run "
                 f"(need some of {list(PUBLIC_ARMS)})")
    print(f"# {os.path.basename(path)}\n")
    print(text_table(scored, meta))
    if not args.hide_weight_table:
        print()
        print(weights_table(meta))
    if not args.no_image:
        print(f"\nwrote {render(scored, meta)}")


if __name__ == "__main__":
    main()
