"""
Stage-2 TEST (report-only): does ChimeraBoost's binary Brier deficit concentrate
on LOW-NOISE / HIGH-SIGNAL datasets, as the Stage-1 synthetic sweep predicted
(docs/PROJECT_STATUS.md §9.4)?

This is pure analysis of an EXISTING Grinsztajn results JSON — no fitting, no
source change. Per §8, Stage 2 ADJUDICATES the synthetic hypothesis; we read the
correlation, we do not tune to it.

Mechanism under test: Stage 1 found the "sharpness tax" appears only in the
near-noiseless corner (and we WIN under noise). Prediction on real data:
  ΔBrier = Brier_Chimera − Brier_leafwise   should be POSITIVE (a deficit) where
  the dataset is high-signal (leaf-wise Brier low) and vanish/reverse where it is
  noisy (leaf-wise Brier high). I.e. ΔBrier should NEGATIVELY correlate with the
  leaf-wise Brier floor (= POSITIVELY correlate with the signal proxy 1−floor).

Brier here is run_benchmarks' multiclass-sum convention (binary -> 2·standard),
identical across models, so the delta is on equal footing.

Usage:  python benchmarks/analyze_grinsztajn_snr.py [results.json]
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEAFWISE = ["LightGBM", "sklearn_HGB"]   # strongest of these = the leaf-wise floor
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "images")


def load_binary_brier(path):
    """Return {dataset: {model: mean_brier_over_seeds}} for binary datasets only."""
    d = json.load(open(path, encoding="utf-8"))
    binsets = {k for k, v in d["datasets"].items() if v.get("task") == "binary"}
    acc = defaultdict(lambda: defaultdict(list))
    for r in d["records"]:
        if r["dataset"] in binsets and "brier" in r["metrics"]:
            acc[r["dataset"]][r["model"]].append(r["metrics"]["brier"])
    out = {}
    for ds, perm in acc.items():
        out[ds] = {m: float(np.mean(v)) for m, v in perm.items()}
    return out, d["datasets"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/20260602-001642.json"
    brier, meta = load_binary_brier(path)

    rows = []
    for ds, mb in sorted(brier.items()):
        if "ChimeraBoost" not in mb:
            continue
        floors = [mb[m] for m in LEAFWISE if m in mb]
        if not floors:
            continue
        floor = min(floors)                  # strongest leaf-wise = the signal proxy
        which = LEAFWISE[int(np.argmin([mb.get(m, np.inf) for m in LEAFWISE]))]
        delta = mb["ChimeraBoost"] - floor   # >0 => Chimera deficit
        signal = 1.0 - floor / 2.0           # 0..1, higher = cleaner signal
        rows.append((ds.split("/")[-1], floor, which, mb["ChimeraBoost"],
                     delta, signal, meta[ds].get("n_train")))

    rows.sort(key=lambda r: r[5], reverse=True)   # highest-signal first

    floor = np.array([r[1] for r in rows])
    delta = np.array([r[4] for r in rows])
    signal = np.array([r[5] for r in rows])

    # Correlations (signal proxy vs deficit). Positive => deficit grows with signal.
    pear = float(np.corrcoef(signal, delta)[0, 1])
    sr = _spearman(signal, delta)

    print(f"\nStage-2 SNR test on {os.path.basename(path)} — {len(rows)} binary datasets")
    print("(Brier = sum-convention; ΔBrier>0 means ChimeraBoost trails the best "
          "leaf-wise model)\n")
    print(f"{'dataset':28s} {'n_train':>7s} {'lw_floor':>8s} {'lw':>11s} "
          f"{'Chimera':>8s} {'Δ':>8s} {'signal':>7s}")
    for ds, fl, which, ch, dl, sg, ntr in rows:
        flag = "  <== deficit" if dl > 0.002 else ("  (win)" if dl < -0.002 else "")
        print(f"{ds:28s} {str(ntr):>7s} {fl:8.4f} {which:>11s} {ch:8.4f} "
              f"{dl:+8.4f} {sg:7.3f}{flag}")

    # High- vs low-signal split at the median floor.
    med = float(np.median(floor))
    hi = delta[floor <= med]   # low floor = high signal
    lo = delta[floor > med]
    print(f"\nMedian leaf-wise Brier floor = {med:.4f}")
    print(f"HIGH-signal half (floor<=med, n={len(hi)}): mean ΔBrier = {hi.mean():+.4f} "
          f"({int((hi>0).sum())}/{len(hi)} are deficits)")
    print(f"LOW-signal  half (floor> med, n={len(lo)}): mean ΔBrier = {lo.mean():+.4f} "
          f"({int((lo>0).sum())}/{len(lo)} are deficits)")
    print(f"\nPearson(signal, Δ) = {pear:+.3f}   Spearman = {sr:+.3f}")
    verdict = ("SUPPORTS" if (pear > 0.2 and hi.mean() > lo.mean()) else
               "does NOT support")
    print(f"Stage-1 prediction (deficit concentrates at high signal): {verdict}.")

    # Plot ΔBrier vs signal proxy.
    os.makedirs(IMAGES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(signal, delta, s=40, c=np.where(delta > 0, "crimson", "green"),
               alpha=0.8, zorder=3)
    if len(rows) > 2:
        b, a = np.polyfit(signal, delta, 1)
        xs = np.linspace(signal.min(), signal.max(), 50)
        ax.plot(xs, a + b * xs, "--", color="gray",
                label=f"fit slope={b:+.3f}, Pearson={pear:+.2f}")
    for ds, fl, which, ch, dl, sg, ntr in rows:
        if abs(dl) > 0.004:
            ax.annotate(ds, (sg, dl), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("dataset signal proxy  (1 − best-leaf-wise-Brier/2;  right = cleaner)")
    ax.set_ylabel("ΔBrier = ChimeraBoost − best leaf-wise   (>0 = deficit)")
    ax.set_title("Stage-2 test: is ChimeraBoost's Brier deficit a high-signal "
                 "'sharpness tax'?")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    out = os.path.join(IMAGES_DIR, "stage2_brier_vs_signal.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n-> saved {out}")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    main()
