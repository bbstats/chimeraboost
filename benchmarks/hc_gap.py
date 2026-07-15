"""HC first-read: the real CatBoost-vs-ChimeraBoost Brier gap on the HC suite,
sliced by categorical cardinality, vs the SynthGen v2 entity-cat prediction.

The Brier-gap program (synthgen/PAYOFF.md) found CatBoost's edge lives entirely
in the entity-cat / high-card regime: synth entity_strength Q4 = 91% CatBoost
Brier winrate at 5.2x gap concentration, while cats=none is dead flat. Grinsztajn
has no such datasets, so that prediction had never been checked on real data.
This reads the HC baseline and reports whether it holds -- the fidelity test of
the v2 entity-cat prior (feed the answer into synthgen/PAYOFF.md v3 watch items).

Usage:
    python benchmarks/hc_gap.py benchmarks/results/hc-baseline.json
        [--ours ChimeraBoost] [--theirs CatBoost]
"""
import argparse
from collections import defaultdict

import numpy as np

import summarize

EPS = 1e-9

# Measured max categorical cardinality per hc dataset (on the 100k subsample;
# from the Step-1 property audit -- benchmarks/HIGHCARD_PLAN.md). Used only to
# SLICE the read, never to select datasets.
HC_MAX_CARD = {
    "hc:kick": 1063, "hc:porto-seguro": 104, "hc:sf-police-incidents": 15165,
    "hc:kdd_ipums_la_97-small": 191, "hc:okcupid-stem": 7019,
    "hc:Traffic_violations": 3830, "hc:cjs": 57, "hc:eucalyptus": 27,
    "hc:wine-reviews": 31959, "hc:colleges": 6039, "hc:house_prices_nominal": 25,
    "hc:black_friday": 7, "hc:employee_salaries": 2264, "hc:Moneyball": 39,
}


def _per_ds_metric(records, key):
    b = defaultdict(lambda: defaultdict(list))
    for r in records:
        v = r["metrics"].get(key)
        if v is not None:
            b[r["dataset"]][r["model"]].append(v)
    return {ds: {m: float(np.mean(vs)) for m, vs in ms.items()}
            for ds, ms in b.items()}


def _winrate_and_gap(per_ds, ds_list, ours, theirs, lower_better=True):
    """Head-to-head over ds_list: (theirs winrate, mean gap ours-theirs, n)."""
    wins_them, gaps = 0, []
    n = 0
    for ds in ds_list:
        o = per_ds.get(ds, {}).get(ours)
        t = per_ds.get(ds, {}).get(theirs)
        if o is None or t is None:
            continue
        n += 1
        gap = o - t if lower_better else t - o   # + = theirs better
        gaps.append(gap)
        if (t < o) if lower_better else (t > o):
            wins_them += 1
    wr = wins_them / n if n else float("nan")
    return wr, (float(np.mean(gaps)) if gaps else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ours", default="ChimeraBoost")
    ap.add_argument("--theirs", default="CatBoost")
    args = ap.parse_args()

    data = summarize.load(args.run)
    records, datasets = data["records"], data["datasets"]
    brier = _per_ds_metric(records, "brier")
    f1 = _per_ds_metric(records, "f1_macro")
    rmse = _per_ds_metric(records, "rmse")

    def task(ds):
        return datasets[ds]["task"]

    clf = [d for d in datasets if task(d) in ("binary", "multiclass")]
    reg = [d for d in datasets if task(d) == "regression"]
    hicard = [d for d in clf if HC_MAX_CARD.get(d, 0) >= 50]
    xcard = [d for d in clf if HC_MAX_CARD.get(d, 0) >= 1000]
    locard = [d for d in clf if HC_MAX_CARD.get(d, 0) < 50]

    print(f"HC first-read: {args.ours} vs {args.theirs}   [{args.run}]")
    print(f"{len(clf)} classification sets ({len(hicard)} max-card>=50, "
          f"{len(xcard)} max-card>=1000), {len(reg)} regression\n")

    print(f"{'slice':22s} {'n':>3s} {'CatB Brier win%':>16s} "
          f"{'mean gap/set':>13s} {'CatB F1 win%':>13s}")
    print("-" * 72)
    for label, ds_list in [("all classification", clf),
                           ("  binary", [d for d in clf if task(d) == "binary"]),
                           ("  multiclass", [d for d in clf if task(d) == "multiclass"]),
                           ("max-card >= 50", hicard),
                           ("max-card >= 1000", xcard),
                           ("max-card < 50", locard)]:
        if not ds_list:
            continue
        bwr, bgap, n = _winrate_and_gap(brier, ds_list, args.ours, args.theirs, True)
        fwr, _, _ = _winrate_and_gap(f1, ds_list, args.ours, args.theirs, False)
        print(f"{label:22s} {n:3d} {bwr:15.0%} {bgap:+13.5f} {fwr:12.0%}")

    # %-of-best Brier via the canonical summarize aggregation
    cols, meta = summarize.aggregate(data)
    print("\n% of best (100 = best model on that set, averaged):")
    for c in ("Bin Brier%", "Multi Brier%", "Bin F1%", "Multi F1%", "Reg RMSE%"):
        row = cols.get(c, {})
        o, t = row.get(args.ours), row.get(args.theirs)
        if o is None and t is None:
            continue
        os_ = f"{o:.1f}" if o is not None else "  -"
        ts_ = f"{t:.1f}" if t is not None else "  -"
        print(f"  {c:14s} {args.ours} {os_:>6s}   {args.theirs} {ts_:>6s}")

    # regression head-to-head (RMSE), for completeness
    rwr, rgap, rn = _winrate_and_gap(rmse, reg, args.ours, args.theirs, True)
    print(f"\nregression RMSE: {args.theirs} win {rwr:.0%} over {rn} sets "
          f"(mean gap/set {rgap:+.4f})")

    # the fidelity verdict vs the synth entity prediction
    bwr_hi, _, n_hi = _winrate_and_gap(brier, hicard, args.ours, args.theirs, True)
    bwr_lo, _, n_lo = _winrate_and_gap(brier, locard, args.ours, args.theirs, True)
    print("\n--- fidelity vs SynthGen v2 entity prediction ---")
    print(f"synth predicted: entity_str Q4 = 91% CatBoost Brier winrate, "
          f"cats=none flat (~50%)")
    print(f"real HC:  max-card>=50 = {bwr_hi:.0%} CatBoost Brier winrate (n={n_hi});  "
          f"max-card<50 = {bwr_lo:.0%} (n={n_lo})")


if __name__ == "__main__":
    main()
