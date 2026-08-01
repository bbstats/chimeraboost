"""Where do we stand against the external field, per decision-suite stratum?

The decide tier runs 7 strata: Grinsztajn and the high-card suite, each with
their @sus25 / @sus50 small-data twins and @time temporal splits. Until
2026-08-01 no run had ever combined external competitors with the variant
strata, so our standing in the small-data and shift regimes was unmeasured.

Reports, per stratum and per opponent: head-to-head win rate on the primary
metric (RMSE regression / Brier classification), the median relative gap, and
the count of near-tie matchups (|gap| < 0.25%) -- the near-ties are where the
leverage is, since a broad shift converts them wholesale.

Usage:  python benchmarks/strata_standing.py RESULTS.json
        python benchmarks/strata_standing.py --latest
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import primary_scores  # noqa: E402

OURS = "ChimeraBoost"
OPPS = ["CatBoost", "LightGBM", "sklearn_HGB", "XGBoost"]
NEAR_TIE = 0.25          # % of opponent metric


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stratum(ds, meta):
    suite = "hc" if ds.startswith("hc:") else "gr"
    var = meta.get("variant") or "base"
    return f"{suite}:{var}"


def analyse(path):
    blob = _load(path)
    recs, meta = blob["records"], blob["datasets"]
    cfg = blob.get("config", {})
    print(f"# {os.path.basename(path)}   seeds={cfg.get('seeds')}  "
          f"timing={cfg.get('timing')}")

    present = sorted({r["model"] for r in recs})
    opps = [o for o in OPPS if o in present]
    print(f"models present: {', '.join(present)}\n")

    # primary_scores applies the house near-solved exclusions (regression
    # best-NRMSE < 2%, classification best-Brier < 1e-3), so this sees exactly
    # the data the headline table and the win-rate axis see.
    primary = primary_scores(blob)
    n_excluded = len(meta) - len(primary)
    if n_excluded:
        print(f"near-solved datasets excluded: {n_excluded}\n")

    # stratum -> opponent -> list of (dataset, relative gap %)
    book = defaultdict(lambda: defaultdict(list))

    for ds, scores in primary.items():
        ours = scores.get(OURS)
        if ours is None:
            continue
        st = _stratum(ds, meta[ds])
        for opp in opps:
            theirs = scores.get(opp)
            if theirs is None or theirs <= 0:
                continue
            gap = 100.0 * (ours - theirs) / theirs     # positive = we lose
            book[st][opp].append((ds, gap))

    order = ["gr:base", "gr:sus25", "gr:sus50", "gr:time",
             "hc:base", "hc:sus25", "hc:sus50", "hc:time"]
    strata = [s for s in order if s in book] + \
             [s for s in sorted(book) if s not in order]

    print("=" * 104)
    print("HEAD-TO-HEAD BY STRATUM — win rate on the primary metric (higher = better for us)")
    print("=" * 104)
    hdr = f"{'stratum':12s}{'n':>4s}  " + "".join(f"{o[:13]:>22s}" for o in opps)
    print(hdr)
    print("-" * 104)

    for st in strata:
        n_ds = max(len(v) for v in book[st].values()) if book[st] else 0
        line = f"{st:12s}{n_ds:4d}  "
        for opp in opps:
            rows = book[st].get(opp, [])
            if not rows:
                line += f"{'-':>22s}"
                continue
            w = sum(1 for _, g in rows if g < 0)
            tot = len(rows)
            med = float(np.median([g for _, g in rows]))
            line += f"{f'{w}/{tot} = {100*w/tot:4.0f}%  m{med:+5.2f}':>22s}"
        print(line)

    # Leverage: how many matchups sit inside the near-tie band, per stratum
    print("\n" + "=" * 104)
    print(f"LEVERAGE — matchups inside a {NEAR_TIE}% band (a broad shift converts these wholesale)")
    print("=" * 104)
    print(f"{'stratum':12s}{'matchups':>10s}{'near-ties':>11s}{'  of which losses':>19s}"
          f"{'   win rate':>12s}{'  if all flip':>14s}")
    grand = [0, 0, 0, 0]
    for st in strata:
        allrows = [(ds, g) for opp in opps for ds, g in book[st].get(opp, [])]
        if not allrows:
            continue
        tot = len(allrows)
        wins = sum(1 for _, g in allrows if g < 0)
        near = [g for _, g in allrows if abs(g) < NEAR_TIE]
        near_loss = sum(1 for g in near if g >= 0)
        grand[0] += tot
        grand[1] += wins
        grand[2] += len(near)
        grand[3] += near_loss
        print(f"{st:12s}{tot:10d}{len(near):11d}{near_loss:19d}"
              f"{100*wins/tot:11.1f}%{100*(wins+near_loss)/tot:13.1f}%")
    if grand[0]:
        print("-" * 104)
        print(f"{'ALL':12s}{grand[0]:10d}{grand[2]:11d}{grand[3]:19d}"
              f"{100*grand[1]/grand[0]:11.1f}%{100*(grand[1]+grand[3])/grand[0]:13.1f}%")
        print(f"\none matchup flip = {100/grand[0]:.2f} win-rate points; "
              f"the +2-point bar needs {int(np.ceil(0.02*grand[0]))} flips")

    # Worst losses, so the next mechanism has a target list
    print("\n" + "=" * 104)
    print("WORST LOSSES ACROSS ALL STRATA (positive = we lose, % of opponent metric)")
    print("=" * 104)
    flat = [(g, st, opp, ds) for st in strata for opp in opps
            for ds, g in book[st].get(opp, []) if g > 0]
    flat.sort(reverse=True)
    for g, st, opp, ds in flat[:22]:
        print(f"  {g:+7.2f}%  {st:10s} vs {opp:12s} {ds}")
    if not flat:
        print("  (none — we win every matchup)")

    # Per-stratum mean gap: is a whole regime weak?
    print("\n" + "=" * 104)
    print("MEAN / MEDIAN RELATIVE GAP BY STRATUM (negative = we are ahead)")
    print("=" * 104)
    for st in strata:
        allg = [g for opp in opps for _, g in book[st].get(opp, [])]
        if allg:
            print(f"  {st:12s} mean {np.mean(allg):+7.3f}%   median "
                  f"{np.median(allg):+7.3f}%   n={len(allg)}")


def main():
    # --ours NAME reads the standing for a variant arm (e.g. ChimeraBoostALR)
    # instead of the shipped default, so an A/B run can be read from either
    # side without editing the module.
    global OURS
    argv = list(sys.argv[1:])
    if "--ours" in argv:
        i = argv.index("--ours")
        OURS = argv[i + 1]
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    if "--latest" in argv or not args:
        here = os.path.dirname(os.path.abspath(__file__))
        cands = sorted(glob.glob(os.path.join(here, "results", "2026*.json")))
        if not cands:
            print("no results json found")
            return
        path = cands[-1]
    else:
        path = args[0]
    analyse(path)


if __name__ == "__main__":
    main()
