"""Probe: can a richer calibration map recover Brier that our scalar temperature leaves on the table?

BACKGROUND (2026-08-01). Two facts that look contradictory but are not:

  (a) We are ALREADY the best-calibrated model in the field. Mean CORP
      miscalibration (MCB) across the 23 Grinsztajn classification sets:
      ChimeraBoost 0.00220 < CatBoost 0.00231 < LightGBM 0.00233 < HGB 0.00272.
      Perfectly recalibrating BOTH sides leaves the head-to-head scoreboard
      completely unchanged (17W-6L vs CatBoost before and after).

  (b) That is the wrong question for a shipping decision. (a) asks "are we
      relatively miscalibrated?" — no. The shipping question is "can we recover
      more of OUR OWN remaining 0.00220 while the opponents stay put?" Those
      have different answers, and 5 of our 9 losing binary matchups have gaps
      SMALLER than a fifth of our MCB.

Today `_fit_temperature` (sklearn_api.py) learns a single scalar T minimizing
validation log loss of sigmoid(raw/T). That family has no intercept, so it
cannot shift the operating point of a skewed problem at all — on an 11%-positive
set like bank-marketing the only reachable maps are ones that fix p=0.5 at
raw=0. A 2-parameter Platt map sigmoid(a*raw + b) can; beta calibration
(Kull et al., AISTATS 2017) adds a third shape parameter.

THE PROBE: fit ChimeraBoost with an explicit calibration fold, then compare
calibration maps fitted on that fold and scored OUT OF SAMPLE on the test set.
Nothing is fitted on test. No library change.

Arms (all fitted on the calibration fold, all scored on test):
  raw        - no calibration at all (the floor)
  temp       - scalar T, log-loss objective          == what ships today
  temp_brier - scalar T, Brier objective             (does the objective matter?)
  platt      - sigmoid(a*raw + b), log-loss objective (adds the intercept)
  beta       - beta calibration, 3 parameters
  isotonic   - out-of-sample isotonic (the nonparametric ceiling, high variance)

PREDICTIONS (stated before running):
  If the intercept matters, `platt` beats `temp` on the skewed sets
  (bank-marketing ~11% positives, Diabetes130US, default-of-credit) and is
  roughly flat on the balanced ones (california, credit, heloc, electricity).
  If `platt` is flat or negative everywhere, the calibration family is
  exhausted and the whole reliability lane closes for good.

Primary metric is Brier (the house decision metric for classification).
Resumable JSONL; `--table-only` reprints.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost import ChimeraBoostClassifier  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-calib-map.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50

# The binary sets where we currently lose a matchup, plus balanced controls
# where we win comfortably (a calibration lever must not hurt these).
LOSING = [
    "gr:clf_num/bank-marketing",
    "gr:clf_num/credit",
    "gr:clf_num/heloc",
    "gr:clf_num/default-of-credit-card-clients",
    "gr:clf_num/Diabetes130US",
    "gr:clf_num/california",
    "gr:clf_num/electricity",
]
CONTROL = [
    "gr:clf_num/covertype",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/pol",
    "gr:clf_num/Bioresponse",
]
ALL = LOSING + CONTROL


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _brier(y, p):
    """Binary Brier in the harness's K=2 sum form, so numbers are comparable
    to the benchmark's `brier` field: sum_k (p_k - onehot_k)^2 = 2*(p-y)^2."""
    return float(np.mean(2.0 * (p - y) ** 2))


def _nll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- calibration maps: each returns a function raw -> probability -----------

def fit_temp_nll(raw, y):
    def loss(T):
        z = raw / T
        return float(np.mean(np.log1p(np.exp(-np.abs(z)))
                             + np.maximum(z, 0.0) - y * z))
    r = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded",
                        options={"xatol": 1e-4})
    T = float(r.x) if r.success else 1.0
    return lambda v: _sig(v / T), {"T": T}


def fit_temp_brier(raw, y):
    def loss(T):
        return _brier(y, _sig(raw / T))
    r = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded",
                        options={"xatol": 1e-4})
    T = float(r.x) if r.success else 1.0
    return lambda v: _sig(v / T), {"T": T}


def fit_platt(raw, y):
    """sigmoid(a*raw + b), log-loss objective. Starts at the shipped scalar
    solution (a=1/T, b=0) so it can only improve the training objective."""
    _, info = fit_temp_nll(raw, y)
    x0 = np.array([1.0 / info["T"], 0.0])

    def loss(th):
        z = th[0] * raw + th[1]
        return float(np.mean(np.log1p(np.exp(-np.abs(z)))
                             + np.maximum(z, 0.0) - y * z))
    r = minimize(loss, x0, method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
    a, b = (r.x if r.success else x0)
    return lambda v: _sig(a * v + b), {"a": float(a), "b": float(b)}


def fit_beta(raw, y):
    """Beta calibration on the model's own probabilities:
    p' = sigmoid(c + a*ln(p) - b*ln(1-p)). Three parameters."""
    p = np.clip(_sig(raw), 1e-6, 1 - 1e-6)
    lp, lq = np.log(p), np.log(1 - p)

    def loss(th):
        z = th[0] * lp - th[1] * lq + th[2]
        return float(np.mean(np.log1p(np.exp(-np.abs(z)))
                             + np.maximum(z, 0.0) - y * z))
    r = minimize(loss, np.array([1.0, 1.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    a, b, c = (r.x if r.success else np.array([1.0, 1.0, 0.0]))

    def apply(v):
        pv = np.clip(_sig(v), 1e-6, 1 - 1e-6)
        return _sig(a * np.log(pv) - b * np.log(1 - pv) + c)
    return apply, {"a": float(a), "b": float(b), "c": float(c)}


def fit_isotonic(raw, y):
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(_sig(raw), y)
    return lambda v: iso.predict(_sig(v)), {}


MAPS = {
    "temp": fit_temp_nll,            # == shipped
    "temp_brier": fit_temp_brier,
    "platt": fit_platt,
    "beta": fit_beta,
    "isotonic": fit_isotonic,
}


def _done():
    seen = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        seen.add((r["dataset"], r["seed"]))
                    except Exception:
                        pass
    return seen


def main():
    done = _done()
    for key in ALL:
        X, y, cat, task = rdata.load(key)
        if task != "binary":
            print(f"skip {key}: {task}")
            continue
        classes = np.unique(y)
        pos_rate = float(np.mean(y == classes[1]))
        print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}  positives={pos_rate:.1%}")
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y)
            # Explicit calibration fold, carved from train only. The model
            # early-stops on it, exactly as the auto-split default would.
            Xa, Xc, ya, yc = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
            t = time.time()
            m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                                       early_stopping_rounds=PATIENCE,
                                       random_state=seed)
            m.fit(Xa, ya, eval_set=(Xc, yc), cat_features=cat)
            fit_s = time.time() - t

            raw_c = np.asarray(m.model_.predict_raw(Xc), dtype=np.float64).ravel()
            raw_t = np.asarray(m.model_.predict_raw(Xte), dtype=np.float64).ravel()
            yc01 = (yc == classes[1]).astype(np.float64)
            yt01 = (yte == classes[1]).astype(np.float64)

            rec = {"dataset": key, "seed": seed, "pos_rate": pos_rate,
                   "fit_time": fit_s,
                   "brier": {"raw": _brier(yt01, _sig(raw_t))},
                   "nll": {"raw": _nll(yt01, _sig(raw_t))}, "params": {}}
            for name, fitter in MAPS.items():
                apply, info = fitter(raw_c, yc01)
                pt = np.clip(apply(raw_t), 0.0, 1.0)
                rec["brier"][name] = _brier(yt01, pt)
                rec["nll"][name] = _nll(yt01, pt)
                rec["params"][name] = info
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            b = rec["brier"]
            print(f"  s{seed} {fit_s:5.1f}s  temp={b['temp']:.6f}  "
                  f"platt={b['platt']:.6f} ({100*(b['temp']-b['platt'])/b['temp']:+.3f}%)  "
                  f"beta={b['beta']:.6f}  iso={b['isotonic']:.6f}")
    table()


def table():
    rows = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    pos = {}
    for r in rows:
        pos[r["dataset"]] = r["pos_rate"]
        for k, v in r["brier"].items():
            agg[r["dataset"]][k].append(v)

    arms = ["raw", "temp", "temp_brier", "platt", "beta", "isotonic"]
    print("\n" + "=" * 120)
    print("CALIBRATION MAP PROBE — test Brier, maps fitted out-of-sample on the calibration fold")
    print("  gain columns = improvement over `temp` (what ships today), % of temp's Brier. Positive = better.")
    print("=" * 120)
    print(f"{'dataset':38s}{'pos%':>6s}" + "".join(f"{a:>11s}" for a in arms)
          + f"{'platt':>9s}{'beta':>8s}{'iso':>8s}")

    deltas = defaultdict(list)
    for group, name in ((LOSING, "LOSING MATCHUPS"), (CONTROL, "CONTROLS (we win these)")):
        print(f"\n-- {name} " + "-" * (110 - len(name)))
        for ds in group:
            if ds not in agg:
                continue
            means = {a: float(np.mean(agg[ds][a])) for a in arms if agg[ds][a]}
            if "temp" not in means:
                continue
            line = f"{ds.replace('gr:', '')[:38]:38s}{100*pos[ds]:5.1f}%"
            for a in arms:
                line += f"{means.get(a, float('nan')):11.6f}"
            t = means["temp"]
            for a in ("platt", "beta", "isotonic"):
                d = 100.0 * (t - means[a]) / t
                deltas[(name, a)].append(d)
                line += f"{d:+8.3f}%"
            print(line)

    print("\n" + "=" * 120)
    print("VERDICT — mean improvement over the shipped scalar temperature (% of Brier)")
    print("=" * 120)
    for name in ("LOSING MATCHUPS", "CONTROLS (we win these)"):
        for a in ("platt", "beta", "isotonic"):
            d = deltas[(name, a)]
            if d:
                wins = sum(1 for x in d if x > 0)
                print(f"  {name:26s} {a:9s} mean {np.mean(d):+7.3f}%   "
                      f"better on {wins}/{len(d)} datasets   median {np.median(d):+7.3f}%")
        print()


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
