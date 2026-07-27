"""Print the per-booster-fit sequence of a default ChimeraBoost fit.

profile_fit.py --attribution aggregates booster fits by config label, which
merges the full-data refit into whichever audition shares its config. This
reader keeps the fits in call order, so the audition / winner / refit legs are
separable and the audition share of total fit time is readable directly.

    python benchmarks/attr_legs.py benchmarks/results/attr-current.json
"""
import json
import sys


def main(path):
    recs = json.load(open(path))
    if isinstance(recs, dict):
        recs = recs.get("records", recs.get("results", []))
    rows = []
    for r in recs:
        fits = r.get("attr_fits") or r.get("fits") or []
        if not fits:
            continue
        total = sum(f["secs"] for f in fits)
        seq = " | ".join(f"{f['label']}:{f['rounds']}r/{f['secs']:.2f}s"
                         for f in fits)
        # Call order is: [const, linear]? , cross?, refit-of-winner?, refit_full.
        # The last fit is the full-data refit whenever refit_full engaged; the
        # auditions are every fit that stopped at the selection_rounds cap.
        cap = r.get("selection_rounds", 100)
        aud = sum(f["secs"] for f in fits[:-1] if f["rounds"] >= cap)
        rows.append((r["dataset"], r.get("seed", 0), r.get("fit_secs", total),
                     total, len(fits), aud, aud / total * 100.0, seq))

    print(f"{'dataset':38s} {'seed':>4s} {'total_s':>8s} {'nfit':>4s} "
          f"{'audit_s':>8s} {'audit%':>7s}")
    for d, s, fs, total, n, aud, pct, _ in rows:
        print(f"{d:38s} {s:4d} {total:8.2f} {n:4d} {aud:8.2f} {pct:7.1f}")
    if rows:
        print(f"\nmean audition share: "
              f"{sum(r[6] for r in rows) / len(rows):.1f}%")
    print("\nper-fit sequence (call order):")
    for d, s, fs, total, n, aud, pct, seq in rows:
        print(f"  {d} [seed {s}]\n    {seq}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "benchmarks/results/attr-current.json")
