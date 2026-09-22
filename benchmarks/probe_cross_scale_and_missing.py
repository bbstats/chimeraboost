"""Two zero-fit reads from the 2026-09-22 refill (CAMPAIGN_PLAN I045).

S4 -- are the `diff` cross columns duplicates of their larger-scale parent?
`_cross_block` builds `diff` as the raw difference `x_i - x_j`. When the two
parents live on very different scales the difference is rank-nearly-identical
to the larger one -- a column the trees already have, occupying a cross slot.
For every Grinsztajn regression set at seed 0 (the harness split), fit the
DEFAULT regressor, read `cross_pairs_`, and for each selected `diff` pair
record the training-scale ratio sigma_max / sigma_min and Spearman's rho
between the difference and each parent. Kill (pre-registered): median |rho|
against the larger parent < 0.95 OR median scale ratio < 3 -- then the raw
difference is not degenerate and there is no slot being wasted.

S9 -- does any decision-suite dataset carry numeric missing values at all?
The binner routes NaN to the top bin, so missing rows always go right at
every split; XGBoost / LightGBM learn the direction. Before anyone writes a
kernel: count, for every Grinsztajn and high-card set, the numeric columns
holding NaN or inf and the rows affected. Kill: fewer than 3 gr sets carry
any -- then the decision suites cannot measure a learned direction.

Run:
    python benchmarks/probe_cross_scale_and_missing.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb
from research import datasets as rdata

from chimeraboost import ChimeraBoostRegressor
from chimeraboost.preprocessing import as_model_array

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "probe-cross-scale-and-missing-20260922.json")


def _gr_keys():
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    keys = [k for k in rb.DATASETS if "@" not in k and (k.startswith("gr:") or k.startswith("hc:"))]
    return keys


# ---- S4 ------------------------------------------------------------------------
def cross_scale_read(keys):
    rows = []
    for key in keys:
        if not key.startswith("gr:reg"):
            continue
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
        t = time.time()
        m = ChimeraBoostRegressor(random_state=0)
        m.fit(Xtr, ytr, cat_features=cat)
        pairs = m.cross_pairs_ or []
        A = as_model_array(Xtr, bool(cat))
        rec = {"dataset": key, "fit_s": round(time.time() - t, 2),
               "cross_selected": bool(m.cross_features_selected_),
               "n_pairs": len(pairs),
               "ops": {op: sum(1 for _, _, o in pairs if o == op) for op in ("diff", "prod", "gdiff")},
               "diff_pairs": []}
        for i, j, op in pairs:
            if op != "diff":
                continue
            a = np.asarray(A[:, i], dtype=np.float64)
            b = np.asarray(A[:, j], dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            sa, sb = float(np.std(a)), float(np.std(b))
            big, small = (a, b) if sa >= sb else (b, a)
            d = a - b
            rho_big = float(abs(spearmanr(d, big).correlation)) if d.std() > 0 else float("nan")
            rho_small = float(abs(spearmanr(d, small).correlation)) if d.std() > 0 else float("nan")
            rec["diff_pairs"].append({"i": int(i), "j": int(j),
                                      "scale_ratio": float(max(sa, sb) / max(min(sa, sb), 1e-300)),
                                      "rho_big": rho_big, "rho_small": rho_small})
        rows.append(rec)
        dp = rec["diff_pairs"]
        summary = (f"{len(dp)} diff pairs, median ratio "
                   f"{np.median([p['scale_ratio'] for p in dp]):.2f}, median |rho| vs larger "
                   f"{np.median([p['rho_big'] for p in dp]):.3f}" if dp else "no diff pairs")
        print(f"  {key:44s} cross={rec['cross_selected']!s:5s} pairs={rec['n_pairs']:2d} "
              f"{rec['ops']}  {summary}", flush=True)
    return rows


# ---- S9 ------------------------------------------------------------------------
def missing_read(keys):
    rows = []
    for key in keys:
        X, y, cat, task = rdata.load(key)
        cat = set(cat or [])
        A = as_model_array(X, bool(cat))
        num_cols = [c for c in range(A.shape[1]) if c not in cat]
        bad_cols, bad_rows = [], np.zeros(A.shape[0], dtype=bool)
        for c in num_cols:
            col = np.asarray(A[:, c], dtype=np.float64)
            miss = ~np.isfinite(col)
            if miss.any():
                bad_cols.append((c, int(miss.sum())))
                bad_rows |= miss
        rows.append({"dataset": key, "task": task, "n": int(A.shape[0]),
                     "n_numeric": len(num_cols), "numeric_cols_with_missing": len(bad_cols),
                     "rows_affected": int(bad_rows.sum()),
                     "worst_col_missing": max((m for _, m in bad_cols), default=0)})
        if bad_cols:
            print(f"  {key:44s} {len(bad_cols)}/{len(num_cols)} numeric cols with missing, "
                  f"{int(bad_rows.sum())}/{A.shape[0]} rows", flush=True)
    return rows


def main():
    keys = _gr_keys()
    print("=== S4: diff cross pairs vs their larger parent (gr regression, seed 0) ===")
    s4 = cross_scale_read(keys)
    print("\n=== S9: numeric missing values on every gr + hc set ===")
    s9 = missing_read(keys)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"s4": s4, "s9": s9}, f, indent=1)

    dp = [p for r in s4 for p in r["diff_pairs"]]
    engaged = [r for r in s4 if r["cross_selected"]]
    print("\n--- S4 summary ---")
    print(f"regression sets {len(s4)}, cross selected on {len(engaged)}, diff pairs {len(dp)}, "
          f"prod pairs {sum(r['ops']['prod'] for r in s4)}, gdiff pairs {sum(r['ops']['gdiff'] for r in s4)}")
    if dp:
        ratios = np.array([p["scale_ratio"] for p in dp])
        rb_ = np.array([p["rho_big"] for p in dp])
        rs_ = np.array([p["rho_small"] for p in dp])
        print(f"scale ratio: median {np.median(ratios):.2f}, share >= 3: {np.mean(ratios >= 3):.0%}, "
              f"share >= 10: {np.mean(ratios >= 10):.0%}")
        print(f"|rho|(diff, larger parent): median {np.nanmedian(rb_):.3f}, share >= 0.95: "
              f"{np.nanmean(rb_ >= 0.95):.0%}; |rho|(diff, smaller parent): median {np.nanmedian(rs_):.3f}")
        print(f"pairs that are near-duplicates (ratio >= 3 AND |rho| >= 0.95): "
              f"{int(np.sum((ratios >= 3) & (rb_ >= 0.95)))} of {len(dp)}")
    gr = [r for r in s9 if r["dataset"].startswith("gr:")]
    hc = [r for r in s9 if r["dataset"].startswith("hc:")]
    print("\n--- S9 summary ---")
    print(f"gr sets with any numeric missing: {sum(r['numeric_cols_with_missing'] > 0 for r in gr)} of {len(gr)}; "
          f"hc: {sum(r['numeric_cols_with_missing'] > 0 for r in hc)} of {len(hc)}")
    for r in sorted(s9, key=lambda r: -r["rows_affected"])[:8]:
        if r["rows_affected"]:
            print(f"  {r['dataset']:44s} {r['numeric_cols_with_missing']} cols, "
                  f"{100 * r['rows_affected'] / r['n']:.1f}% rows")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
