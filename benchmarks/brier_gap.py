"""Where does CatBoost's Brier edge over ChimeraBoost concentrate? (PAYOFF step 1)

Reads ONE run_benchmarks JSON containing records for both models (the synv2
full baseline) and slices the classification sets by the synth recipe meta.
Per slice: head-to-head Brier winrate, mean excess Brier over the Bayes floor
for both models, the mean paired gap, and mean MCB (CORP miscalibration) for
both. The "conc" column is (share of the total summed gap carried by the
slice) / (share of sets in the slice): >1 = the edge concentrates there.

Usage:
    python benchmarks/brier_gap.py benchmarks/results/synv2-full-baseline.json
        [--ours ChimeraBoost] [--theirs CatBoost] [--min-n 12]
"""
import argparse

import numpy as np

import synth_report
from synthgen.suites import CANARIES, VERSION

EPS = 1e-9


def _per_ds(per_model, model, key):
    out = {}
    for ds, mlist in per_model.get(model, {}).items():
        vals = [m[key] for m in mlist if m.get(key) is not None]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def _quartile_specs(values_by_ds, prefix, universe):
    """Quartile slices of a meta field over the datasets in `universe`."""
    vals = np.array([values_by_ds[ds] for ds in universe])
    if len(vals) < 8:
        return []
    qs = np.quantile(vals, [0.25, 0.5, 0.75])
    edges = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
    return [(f"{prefix} Q{i}",
             lambda ds, lo=lo, hi=hi: ds in universe
             and lo <= values_by_ds[ds] < hi)
            for i, (lo, hi) in enumerate(edges, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ours", default="ChimeraBoost")
    ap.add_argument("--theirs", default="CatBoost")
    ap.add_argument("--min-n", type=int, default=12)
    args = ap.parse_args()

    metas, per_model = synth_report.load_run(args.run)
    b_ours = _per_ds(per_model, args.ours, "brier")
    b_them = _per_ds(per_model, args.theirs, "brier")
    mcb_ours = _per_ds(per_model, args.ours, "calibration_mcb")
    mcb_them = _per_ds(per_model, args.theirs, "calibration_mcb")
    shared = sorted(set(b_ours) & set(b_them))
    if not shared:
        raise SystemExit(f"no classification sets shared by {args.ours} and "
                         f"{args.theirs} in {args.run}")

    def synth(ds, f):
        return metas[ds]["synth"].get(f)

    def is_canary(ds):
        return (synth(ds, "gen_version") == VERSION
                and synth(ds, "recipe_id") in CANARIES)

    entity_sets = {ds for ds in shared if (synth(ds, "n_cat_entity") or 0) > 0}
    es = {ds: float(synth(ds, "entity_strength") or 0.0) for ds in shared}
    noise = {ds: float(synth(ds, "noise_level") or 0.0) for ds in shared}
    imb = {ds: float(synth(ds, "imbalance") or 0.0) for ds in shared}

    specs = [
        ("all", lambda ds: True),
        ("task=binary", lambda ds: metas[ds]["task"] == "binary"),
        ("task=multiclass", lambda ds: metas[ds]["task"] == "multiclass"),
        ("canary", is_canary),
        ("cats=none", lambda ds: synth(ds, "n_cat") == 0),
        ("cats=mixed", lambda ds: 0 < synth(ds, "cat_fraction") < 1.0),
        ("cats=all", lambda ds: synth(ds, "cat_fraction") >= 1.0),
        ("cats=entity", lambda ds: ds in entity_sets),
        ("card>8", lambda ds: synth(ds, "max_cardinality") > 8),
        ("card>16", lambda ds: synth(ds, "max_cardinality") > 16),
        ("n<2000", lambda ds: synth(ds, "n") < 2000),
        ("n>=2000", lambda ds: synth(ds, "n") >= 2000),
        ("depth<=2", lambda ds: synth(ds, "interaction_depth") <= 2),
        ("depth>=3", lambda ds: synth(ds, "interaction_depth") >= 3),
        ("saturated", lambda ds: synth(ds, "saturated")),
        ("imbalance>0.7", lambda ds: imb[ds] > 0.7),
        ("missing>0", lambda ds: synth(ds, "missing_fraction") > 0),
        ("irrelevant>0.3", lambda ds: synth(ds, "irrelevant_fraction") > 0.3),
    ]
    specs += _quartile_specs(es, "entity_str", entity_sets)
    specs += _quartile_specs(noise, "noise", set(shared))
    for kind in ("linear", "neural", "tree", "product", "plateau", "cellrule"):
        specs.append((f"func={kind}",
                      lambda ds, k=kind: synth(ds, "func_dominant") == k))

    gap = {ds: b_ours[ds] - b_them[ds] for ds in shared}   # + = theirs better
    total_gap = sum(gap.values())
    n_all = len(shared)

    print(f"Brier gap location: {args.ours} vs {args.theirs} over {n_all} "
          f"classification sets [{args.run}]")
    print(f"total summed Brier gap {total_gap:+.4f} "
          f"(mean {total_gap / n_all:+.5f}/set; + = {args.theirs} better)\n")
    hdr = (f"{'slice':16s} {'n':>4s} {'theyW%':>7s} {'xs ours':>8s} "
           f"{'xs them':>8s} {'gap/set':>9s} {'conc':>6s} "
           f"{'MCB ours':>9s} {'MCB them':>9s}")
    print(hdr)

    ranked = []
    for label, pred in specs:
        ds_in = [ds for ds in shared if pred(ds)]
        if not ds_in:
            continue
        they_w = np.mean([gap[ds] > EPS for ds in ds_in])
        xs_o = [b_ours[ds] - synth(ds, "bayes_brier") for ds in ds_in
                if synth(ds, "bayes_brier") is not None]
        xs_t = [b_them[ds] - synth(ds, "bayes_brier") for ds in ds_in
                if synth(ds, "bayes_brier") is not None]
        g = float(np.mean([gap[ds] for ds in ds_in]))
        share = sum(gap[ds] for ds in ds_in) / total_gap if total_gap else 0.0
        conc = share / (len(ds_in) / n_all)
        mos = [mcb_ours[ds] for ds in ds_in if ds in mcb_ours]
        mts = [mcb_them[ds] for ds in ds_in if ds in mcb_them]
        mo = float(np.mean(mos)) if mos else float("nan")
        mt = float(np.mean(mts)) if mts else float("nan")
        xs_o_s = f"{np.mean(xs_o):8.4f}" if xs_o else "       -"
        xs_t_s = f"{np.mean(xs_t):8.4f}" if xs_t else "       -"
        print(f"{label:16s} {len(ds_in):4d} {they_w:7.0%} {xs_o_s} {xs_t_s} "
              f"{g:+9.5f} {conc:6.2f} {mo:9.4f} {mt:9.4f}")
        if label not in ("all", "canary") and len(ds_in) >= args.min_n:
            ranked.append((label, len(ds_in), they_w, g, conc, mo - mt))

    print(f"\nconcentration ranking (n>={args.min_n}, by conc = gap share / "
          "set share):")
    for label, n, w, g, conc, dmcb in sorted(ranked, key=lambda r: -r[4])[:6]:
        print(f"  {label:16s} n={n:3d}  theyW {w:4.0%}  gap/set {g:+.5f}  "
              f"conc {conc:5.2f}  dMCB(ours-them) {dmcb:+.4f}")
    print("\nMCB-heavy slices (ours-them > +0.005 -> L4 calibration lever "
          "unlocks there):")
    hits = [r for r in ranked if r[5] > 0.005]
    for label, n, w, g, conc, dmcb in sorted(hits, key=lambda r: -r[5]):
        print(f"  {label:16s} n={n:3d}  dMCB {dmcb:+.4f}")
    if not hits:
        print("  none — refinement, not calibration, is the story everywhere")


if __name__ == "__main__":
    main()
