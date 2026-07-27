"""Does the replay refit's saving hold as n grows?

Every decision suite caps dataset size (Grinsztajn subsamples to 50K, the public
suite subsamples too), so the measured -35% is really "-35% at up to ~50K rows".
This walks the same comparison up the row count.

Note that `scaling_giant.py` cannot answer this: it fits a fixed tree count with
early stopping OFF, and that is precisely the path on which no refit happens at
all. The refit only exists on the automatic-split path, so this script fits with
ordinary defaults and varies only `refit_full`.

    python benchmarks/replay_scaling.py --sizes 50000,200000,500000
"""
import argparse
import time

import numpy as np

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup


def make_data(n, rng, n_num=12, n_cat=3, cardinality=200):
    """Mixed numeric + categorical, with real categorical signal so the ordered
    target encoder is exercised (that is where the refit does its extra work)."""
    num = rng.standard_normal((n, n_num))
    cat = rng.integers(0, cardinality, size=(n, n_cat))
    cat_effect = rng.standard_normal(cardinality)[cat].sum(axis=1)
    signal = (num[:, 0] + 0.7 * num[:, 1] * num[:, 2] - 0.5 * num[:, 3]
              + 0.8 * cat_effect)
    X = np.hstack([num, cat.astype(float)])
    y = signal + 0.4 * rng.standard_normal(n)
    return X, y, list(range(n_num, n_num + n_cat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="50000,200000,500000")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    print("chimeraboost:", chimeraboost.__file__, flush=True)
    warmup()
    print(f"\n{'task':6s} {'n':>9s} {'scratch_s':>10s} {'replay_s':>9s} "
          f"{'ratio':>7s} {'saving':>8s} {'metric_d%':>10s}")
    for n in sizes:
        rng = np.random.default_rng(a.seed)
        X, y, cat = make_data(n, rng)
        cut = int(n * 0.75)
        Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
        for task in ("reg", "bin"):
            if task == "reg":
                Est, ytr_, yte_ = ChimeraBoostRegressor, ytr, yte
            else:
                thr = np.median(ytr)
                Est = ChimeraBoostClassifier
                ytr_, yte_ = (ytr > thr).astype(int), (yte > thr).astype(int)
            secs, score = {}, {}
            for name, mode in (("scratch", True), ("replay", "replay")):
                est = Est(random_state=a.seed, refit_full=mode)
                t0 = time.perf_counter()
                est.fit(Xtr, ytr_, cat_features=cat)
                secs[name] = time.perf_counter() - t0
                if task == "reg":
                    score[name] = float(np.sqrt(np.mean(
                        (yte_ - est.predict(Xte)) ** 2)))
                else:
                    p = est.predict_proba(Xte)[:, 1]
                    score[name] = float(np.mean((p - yte_) ** 2))
            r = secs["replay"] / secs["scratch"]
            # Lower is better for both metrics, so a positive delta is a win.
            d = (score["scratch"] - score["replay"]) / score["scratch"] * 100
            print(f"{task:6s} {n:9d} {secs['scratch']:10.1f} "
                  f"{secs['replay']:9.1f} {r:7.3f} {(1 - r) * 100:7.1f}% "
                  f"{d:+10.3f}", flush=True)


if __name__ == "__main__":
    main()
