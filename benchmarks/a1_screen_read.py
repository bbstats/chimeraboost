"""A1 tier-1 screen read: multiclass treatment slice vs reg/binary identity.

Usage: python benchmarks/a1_screen_read.py BASE.json NEW.json

Slices per A1_PLAN.md: 34 multiclass sets (3 canaries 017/117/317 reported
separately), 102 reg/binary sets that must be EXACT ties. Reports primary
(F1) and Brier W/L/T + means on the treatment, per-set deltas, and the
multiclass fit-time ratio.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "benchmarks")
from synthgen import api, suites  # noqa: E402

CANARIES = {"syn:v2/017", "syn:v2/117", "syn:v2/317"}
MC = {"syn:v2/%03d" % i for i in suites.SUITES["screen"]
      if str(api.task_of("syn:v2/%03d" % i)) == "multiclass"}


def load(path, model="ChimeraBoost"):
    d = json.load(open(path))
    recs = d.get("results", d.get("records"))
    out = {}
    for r in recs:
        if r["model"] != model:
            continue
        out.setdefault(r["dataset"], []).append(r)
    return out


def agg(rows, key):
    vals = [r["metrics"].get(key) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def sign_slice(base, new, sets, key, higher_better):
    w = l = t = 0
    deltas = []
    rows = []
    for ds in sorted(sets):
        if ds not in base or ds not in new:
            continue
        b, n = agg(base[ds], key), agg(new[ds], key)
        if b is None or n is None:
            continue
        d = (n - b) if higher_better else (b - n)     # >0 = NEW wins
        rel = d / max(abs(b), 1e-12) * 100.0
        deltas.append(rel)
        rows.append((ds, b, n, rel))
        if abs(n - b) < 1e-9:
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    return w, l, t, (float(np.mean(deltas)) if deltas else 0.0), rows


def main():
    base_p, new_p = sys.argv[1], sys.argv[2]
    base, new = load(base_p), load(new_p)
    treat = (MC - CANARIES) & set(base) & set(new)
    ident = (set(base) & set(new)) - MC

    print(f"BASE {base_p}  NEW {new_p}")
    print(f"sets: {len(treat)} multiclass treatment, {len(CANARIES)} "
          f"canaries, {len(ident)} reg/binary identity\n")

    # Identity surface: exact ties on primary.
    broken = []
    for ds in sorted(ident):
        b, n = agg(base[ds], "primary"), agg(new[ds], "primary")
        if b is None or n is None or abs(b - n) > 1e-9:
            broken.append((ds, b, n))
    print(f"IDENTITY (reg/binary): {len(ident) - len(broken)}/{len(ident)} "
          f"exact ties" + ("" if not broken else f"  BROKEN: {broken[:10]}"))

    for ds in sorted(CANARIES & set(base) & set(new)):
        b, n = agg(base[ds], "primary"), agg(new[ds], "primary")
        print(f"canary {ds}: base {b:.6f} new {n:.6f} "
              f"delta {(n - b):+.6f}")

    for key, hb, label in (("primary", True, "PRIMARY (F1)"),
                           ("brier", False, "BRIER (lower better)")):
        w, l, t, mean, rows = sign_slice(base, new, treat, key, hb)
        print(f"\n{label} treatment slice: {w}W-{l}L-{t}T  mean {mean:+.3f}%")
        for ds, b, n, rel in sorted(rows, key=lambda r: r[3]):
            print(f"  {ds}: {b:.5f} -> {n:.5f}  {rel:+.3f}%")

    # Fit-time ratio on the treatment slice.
    ratios = []
    for ds in sorted(treat):
        bt = np.mean([r["fit_time"] for r in base[ds]])
        nt = np.mean([r["fit_time"] for r in new[ds]])
        ratios.append(nt / bt)
    if ratios:
        print(f"\nFIT TIME multiclass: geomean ratio "
              f"{float(np.exp(np.mean(np.log(ratios)))):.3f}x "
              f"(min {min(ratios):.2f} max {max(ratios):.2f})")
    rounds_b = np.mean([np.mean([r["best_iter"] for r in base[ds]])
                        for ds in sorted(treat)])
    rounds_n = np.mean([np.mean([r["best_iter"] for r in new[ds]])
                        for ds in sorted(treat)])
    print(f"mean rounds multiclass: {rounds_b:.0f} -> {rounds_n:.0f}")


if __name__ == "__main__":
    main()
