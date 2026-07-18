"""Integration smoke for QUANT_PLAN.md Phase 1: does quantize_gradients=True
actually speed up REAL wrapper-level fits (selection auditions, linear-leaf
selection, cross-features refits all engaged), and does accuracy stay sane?

NOT a decision instrument — Phase 3's suite runs are. This only verifies the
flag engages end to end and the fit-level direction matches the Phase-0 micro
before suite time is spent.

Usage: python benchmarks/quant_smoke.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chimeraboost  # noqa: E402
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor  # noqa: E402


def make_reg(n, nf, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, nf))
    y = (2.0 * X[:, 0] + np.sin(3.0 * X[:, 1]) + X[:, 2] * X[:, 3]
         + 0.5 * rng.standard_normal(n))
    return X, y


def make_bin(n, nf, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, nf))
    z = X[:, 0] + 0.8 * np.sin(2.0 * X[:, 1]) + 0.6 * X[:, 2] * X[:, 3]
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    return X, y


def run(task, n, nf, seed=0):
    if task == "reg":
        X, y = make_reg(n, nf, seed)
        Model = ChimeraBoostRegressor
    else:
        X, y = make_bin(n, nf, seed)
        Model = ChimeraBoostClassifier
    ho = int(0.25 * n)
    Xtr, ytr, Xte, yte = X[:-ho], y[:-ho], X[-ho:], y[-ho:]
    out = {}
    for label, q in (("float", False), ("quant", True)):
        t0 = time.perf_counter()
        m = Model(random_state=seed, quantize_gradients=q).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        if task == "reg":
            metric = float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))
        else:
            p = m.predict_proba(Xte)[:, 1]
            eps = 1e-12
            metric = float(-np.mean(yte * np.log(p + eps)
                                    + (1 - yte) * np.log(1 - p + eps)))
        out[label] = (fit_s, metric)
    return out


def main():
    print(f"chimeraboost: {chimeraboost.__file__}")
    # JIT warm both paths on a tiny fit so timings below are honest.
    run("reg", 3000, 8, seed=9)
    run("bin", 3000, 8, seed=9)

    rows = []
    for task, n, nf in [("reg", 37500, 24), ("bin", 37500, 24),
                        ("reg", 150000, 16), ("bin", 150000, 16)]:
        r = run(task, n, nf)
        fs, fm = r["float"]
        qs, qm = r["quant"]
        rows.append((task, n, nf, fs, qs, fs / qs, fm, qm))
        print(f"{task} n={n:>6} nf={nf:>2}  fit float {fs:6.2f}s  "
              f"quant {qs:6.2f}s  speedup {fs / qs:5.2f}x  "
              f"metric float {fm:.5f}  quant {qm:.5f}")

    print("\n| task | n | nf | float fit s | quant fit s | speedup | "
          "float metric | quant metric |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for (task, n, nf, fs, qs, sp, fm, qm) in rows:
        print(f"| {task} | {n} | {nf} | {fs:.2f} | {qs:.2f} | {sp:.2f}x | "
              f"{fm:.5f} | {qm:.5f} |")


if __name__ == "__main__":
    main()
