"""
Stage-1 leaf-estimation experiment (docs/PROJECT_STATUS.md §9.4/§9.6).

Steps A+B isolated the Brier deficit to a CHIMERA-SPECIFIC leaf-estimation gap
in the noiseless/low-noise, high-local-complexity regime (Family A v2): at large
n our excess Brier plateaus ~0.0052 while CatBoost (also oblivious depth-6)
reaches ~0.0030. Target: close that ~0.0022 gap with leaf-VALUE levers, NOT
geometry — and without touching the noise-robustness dividend.

This script sweeps our EXISTING knobs first (cheapest hypotheses) on Family A v2:
  1A  l2_leaf_reg in {1.0, 0.1, 0.05, 0.0}            (lei=3 fixed)
  1B  leaf_estimation_iterations in {1, 3, 5, 10}     (l2=1.0 fixed)
CatBoost + LightGBM are shown as references (the bar to reach / the leaf-wise
floor). Metric = excess Brier over Bayes (Bayes=0 here, noiseless) in the same
run_benchmarks sum-convention.

Report-only Stage 1: this INFORMS which lever to pursue; nothing ships off these
numbers without the full synthetic->Grinsztajn->OpenML pipeline (§8).

Usage:  python benchmarks/stage1_leaf_estimation.py [--runs N] [--n 4000 8000]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402
from chimeraboost import ChimeraBoostClassifier  # noqa: E402
from synthetic import family_a_multi_pocket, _bayes_brier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def _fit_chimera_brier(Xtr, ytr, Xte, yte, threads=None, **kw):
    """Mirror run_benchmarks._run_chimera exactly (val split + early stopping),
    but with overridable leaf-estimation kwargs. Returns the sum-convention
    Brier via the shared _compute_metrics, so it's comparable to the RUNNERS."""
    Xf, Xv, yf, yv = rb._val_split(Xtr, ytr, "binary", 0)
    m = ChimeraBoostClassifier(n_estimators=rb.MAX_ITERS,
                               early_stopping_rounds=rb.PATIENCE,
                               depth=6, thread_count=threads, random_state=0, **kw)
    m.fit(Xf, yf, eval_set=(Xv, yv))
    return rb._compute_metrics("binary", yte, m, Xte)["brier"]


# Chimera variants to test. Name -> kwargs. "baseline" == current shipped default.
CONFIGS = {
    "baseline (l2=1.0,lei=3)": dict(l2_leaf_reg=1.0, leaf_estimation_iterations=3),
    # 1A — lower l2 in the noiseless regime
    "1A l2=0.1": dict(l2_leaf_reg=0.1, leaf_estimation_iterations=3),
    "1A l2=0.05": dict(l2_leaf_reg=0.05, leaf_estimation_iterations=3),
    "1A l2=0.0": dict(l2_leaf_reg=0.0, leaf_estimation_iterations=3),
    # 1B — more Newton leaf iterations
    "1B lei=1": dict(l2_leaf_reg=1.0, leaf_estimation_iterations=1),
    "1B lei=5": dict(l2_leaf_reg=1.0, leaf_estimation_iterations=5),
    "1B lei=10": dict(l2_leaf_reg=1.0, leaf_estimation_iterations=10),
    # 2  — ordered boosting (CatBoost's signature unbiased-gradient mechanism;
    #      our LOO approximation, off by default). lei is skipped when OB=True.
    "2 ordered_boosting=True": dict(l2_leaf_reg=1.0, ordered_boosting=True),
}
REFS = ["CatBoost", "LightGBM"]  # bar-to-reach / leaf-wise floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 8000])
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    names = list(CONFIGS) + REFS
    print(f"Stage-1 leaf-estimation sweep | Family A v2 | runs={args.runs} | n={args.n}")
    print("excess Brier over Bayes (Bayes=0, noiseless); lower=better. "
          "Target = match CatBoost.\n")

    for n in args.n:
        excess = {nm: [] for nm in names}
        for run_idx in range(args.runs):
            seed = 2000 + run_idx
            X, y, true_prob = family_a_multi_pocket(n, seed=seed)
            Xtr, Xte, ytr, yte, _ptr, pte = train_test_split(
                X, y, true_prob, test_size=0.2, random_state=seed, stratify=y)
            bayes = _bayes_brier(yte, pte)
            for nm, kw in CONFIGS.items():
                try:
                    b = _fit_chimera_brier(Xtr, ytr, Xte, yte, threads=args.threads, **kw)
                    excess[nm].append(max(0.0, b - bayes))
                except Exception as e:
                    print(f"  ! {nm} failed n={n} seed={seed}: {e}")
                    excess[nm].append(np.nan)
            for nm in REFS:
                try:
                    metrics, _t, _it = rb.RUNNERS[nm](
                        "binary", Xtr, ytr, Xte, yte, None, args.threads)
                    excess[nm].append(max(0.0, metrics["brier"] - bayes))
                except Exception as e:
                    print(f"  ! {nm} failed n={n} seed={seed}: {e}")
                    excess[nm].append(np.nan)

        cat = float(np.nanmean(excess["CatBoost"]))
        print(f"n={n}:")
        ordered = sorted(names, key=lambda nm: np.nanmean(excess[nm]))
        for nm in ordered:
            mu = float(np.nanmean(excess[nm]))
            sd = float(np.nanstd(excess[nm]))
            gap = mu - cat
            tag = "  <-- CatBoost target" if nm == "CatBoost" else \
                  ("  *matches/beats CatBoost*" if gap <= 0.0003 and nm in CONFIGS else "")
            print(f"  {nm:28s} {mu:.4f} +/- {sd:.4f}   (vs CatBoost {gap:+.4f}){tag}")
        print()


if __name__ == "__main__":
    main()
