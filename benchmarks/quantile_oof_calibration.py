"""Out-of-fold calibration study of the multi-quantile head on real data.

Real data has no analytic conditional truth, so "is the 5% quantile right for
this row" is not directly measurable. What IS measurable out of fold, and what
this script reports per dataset over pooled 5-fold OOF predictions:

  * Did it learn anything?  CRPS skill vs the fold-marginal grid (0 = learned
    nothing, 1 = perfect). Needed because the marginal forecast is perfectly
    CALIBRATED while knowing nothing -- calibration alone proves little.
  * Is the whole grid calibrated?  The 19 predicted levels cut the line into
    20 cells the model claims hold 5% each; we count where held-out outcomes
    actually land. Reported as the empirical KL(observed || claimed) in nats
    and the worst single cell's deviation, plus the two tail cells (share of
    outcomes below the 5% level / above the 95% level -- want 0.05 each).
  * Do intervals cover?  Empirical coverage of the central 50% and 90%.
  * Are predict_thresh probabilities honest?  Exceedance probabilities at the
    fold-train marginal 10/30/50/70/90% points, scored two ways: ECE (binned
    predicted probability vs observed frequency, 10 bins) and Brier skill vs
    the fold-marginal exceedance rate.
  * Point accuracy of the derived heads: kind="mean" as R^2 vs the fold-mean
    baseline, kind="median" as pinball@0.5 skill vs the fold-median.

Known-by-design limits measured rather than hidden:
  * conformalize=False (the default) -- the raw grid is what most users get.
  * predict_thresh clamps to [0.05, 0.95]; rows whose true exceedance is more
    extreme carry an irreducible error, which lands in ECE/Brier here.
  * abalone's target is a small integer (ring counts): predicted quantiles tie
    exactly with outcomes, so its cell occupancies are lumpy by nature -- kept
    in as the discrete-target stress case, read it with that in mind.

One-off validation study for issue #106 -- not a gate, not part of --decide.
Run:  python benchmarks/quantile_oof_calibration.py [--conformalize] [names...]
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import ChimeraBoostQuantileRegressor          # noqa: E402
from chimeraboost.quantile_api import DEFAULT_QUANTILES         # noqa: E402
from chimeraboost.quantile_metrics import (crps, crossing_rate,  # noqa: E402
                                           interval_coverage, pinball_loss)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data_cache", "grinsztajn")
DATASETS = ["abalone", "Bike_Sharing_Demand", "elevators", "houses",
            "diamonds", "MiamiHousing2016", "medical_charges"]
MAX_ROWS = 20_000
N_FOLDS = 5
SEED = 0
TAUS = DEFAULT_QUANTILES
THRESH_LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def load(name):
    df = pd.read_csv(os.path.join(CACHE, f"reg_num__{name}.csv"))
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    return X, y


def cell_hits(y, Q):
    """Which of the 20 nominal-5% cells each outcome landed in: the count of
    predicted levels strictly below it (ties go down, which is what makes a
    discrete target visibly lumpy rather than silently fine)."""
    return (Q < y[:, None]).sum(axis=1)


def ece(p, outcome, bins=10):
    """Expected calibration error: bin by predicted probability, weight each
    bin's |mean predicted - observed frequency| by its share of rows."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total, err = p.size, 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.sum() / total * abs(p[m].mean() - outcome[m].mean())
    return err


def run_dataset(name, conformalize=False):
    X, y = load(name)
    n = len(y)
    K = TAUS.shape[0]

    Q = np.empty((n, K))
    baseQ = np.empty((n, K))
    pred_mean = np.empty(n)
    p_hat = np.empty((n, THRESH_LEVELS.size))
    p_base = np.empty((n, THRESH_LEVELS.size))
    exceeded = np.empty((n, THRESH_LEVELS.size))
    base_mean = np.empty(n)

    t0 = time.time()
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        est = ChimeraBoostQuantileRegressor(random_state=SEED,
                                            conformalize=conformalize)
        est.fit(X[tr], y[tr])

        Q[te] = est.predict(X[te])
        pred_mean[te] = est.predict(X[te], kind="mean")
        baseQ[te] = np.quantile(y[tr], TAUS)[None, :]
        base_mean[te] = y[tr].mean()

        thr = np.quantile(y[tr], THRESH_LEVELS)
        p_hat[te] = est.predict_thresh(X[te], thr)
        p_base[te] = (y[tr][:, None] > thr[None, :]).mean(axis=0)
        exceeded[te] = y[te][:, None] > thr[None, :]
    fit_s = time.time() - t0

    r = {"name": name, "n": n, "fit_s": fit_s}
    r["skill"] = 1.0 - crps(y, Q, TAUS) / crps(y, baseQ, TAUS)
    r["crossing"] = crossing_rate(Q)

    # 20-cell occupancy: claimed 5% each.
    freq = np.bincount(cell_hits(y, Q), minlength=K + 1) / n
    claimed = np.concatenate([[TAUS[0]], np.diff(TAUS), [1.0 - TAUS[-1]]])
    nz = freq > 0
    r["cell_kl"] = float((freq[nz] * np.log(freq[nz] / claimed[nz])).sum())
    r["cell_max"] = float(np.abs(freq - claimed).max())
    r["tail_lo"], r["tail_hi"] = float(freq[0]), float(freq[-1])
    r["cells"] = freq

    cov = {round(iv["nominal"], 2): iv["coverage"]
           for iv in interval_coverage(y, Q, TAUS)}
    r["cov50"], r["cov90"] = cov[0.5], cov[0.9]

    r["ece"] = ece(p_hat.ravel(), exceeded.ravel())
    bs_model = float(((p_hat - exceeded) ** 2).mean())
    bs_base = float(((p_base - exceeded) ** 2).mean())
    r["brier_skill"] = 1.0 - bs_model / bs_base

    mi = int(np.argmin(np.abs(TAUS - 0.5)))
    pb_model = pinball_loss(y, Q[:, [mi]], TAUS[[mi]])[0]
    pb_base = pinball_loss(y, baseQ[:, [mi]], TAUS[[mi]])[0]
    r["med_skill"] = 1.0 - pb_model / pb_base
    r["mean_r2"] = 1.0 - (((pred_mean - y) ** 2).mean()
                          / ((base_mean - y) ** 2).mean())
    return r


def main():
    args = sys.argv[1:]
    conformalize = "--conformalize" in args
    names = [a for a in args if not a.startswith("--")] or DATASETS
    if conformalize:
        print("conformalize=True (CQR-calibrated grid)")
    rows = []
    for name in names:
        r = run_dataset(name, conformalize)
        rows.append(r)
        print(f"done {r['name']:<20} ({r['n']} rows, {r['fit_s']:.0f}s, "
              f"crossing rate {r['crossing']:.4f})", flush=True)

    print("\nSkill vs the fold-marginal no-skill forecast "
          "(1 perfect, 0 learned nothing)")
    print(f"{'dataset':<20}{'n':>7}{'CRPSskill':>10}{'medSkill':>9}"
          f"{'meanR2':>8}")
    for r in rows:
        print(f"{r['name']:<20}{r['n']:>7}{r['skill']:>10.3f}"
              f"{r['med_skill']:>9.3f}{r['mean_r2']:>8.3f}")

    print("\nCalibration of the OOF distribution "
          "(want: cov50 .50, cov90 .90, tails .050 each, cellKL ~0)")
    print(f"{'dataset':<20}{'cov50':>7}{'cov90':>7}{'tailLo':>7}{'tailHi':>7}"
          f"{'cellKL':>8}{'cellMax':>8}{'ECE':>7}{'BrierSk':>8}")
    for r in rows:
        print(f"{r['name']:<20}{r['cov50']:>7.3f}{r['cov90']:>7.3f}"
              f"{r['tail_lo']:>7.3f}{r['tail_hi']:>7.3f}{r['cell_kl']:>8.4f}"
              f"{r['cell_max']:>8.3f}{r['ece']:>7.3f}{r['brier_skill']:>8.3f}")

    print("\n20-cell occupancy (each should be ~0.050; left to right = "
          "below q05 ... above q95)")
    for r in rows:
        cells = " ".join(f"{v:.3f}" for v in r["cells"])
        print(f"{r['name']:<20} {cells}")

    print("\nECE and Brier skill score predict_thresh at the fold-marginal "
          "10/30/50/70/90% thresholds; the [0.05, 0.95] clamp is included "
          "in both, as delivered to a user. abalone's target is a small "
          "integer, so its cells are lumpy by nature (ties).")


if __name__ == "__main__":
    main()
