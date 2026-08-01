"""Which CatBoost defaults are SIZE-DEPENDENT?

Finding 3 says our win rate vs CatBoost collapses as training rows shrink
(81% -> 50% -> 33%), while against LightGBM/HGB we stay dominant everywhere.
A collapse that tracks dataset size is the signature of a mechanism that
*switches on* at small n. CatBoost resolves several defaults from the data,
so before hypothesising anything we just ask it what it actually ran.

Zero library change, seconds to run: fit on synthetic data at a range of row
counts and print `get_all_params()`, then diff against the largest arm.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor, CatBoostClassifier  # noqa: E402

SIZES = [200, 500, 1000, 2000, 5000, 10_000, 20_000, 60_000]
# The harness's shared budget (benchmarks/run_benchmarks.py).
MAX_ITERS, PATIENCE = 2000, 50


def resolved(task, n, n_feat=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    signal = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] - 0.3 * X[:, 3] ** 2
    Xv, sv = X[: max(2, n // 5)], signal[: max(2, n // 5)]
    if task == "regression":
        y = signal + rng.normal(scale=0.5, size=n)
        yv = sv + rng.normal(scale=0.5, size=len(sv))
        Est = CatBoostRegressor
    else:
        y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
        yv = (sv + rng.normal(scale=0.5, size=len(sv)) > 0).astype(int)
        Est = CatBoostClassifier
    m = Est(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
            verbose=False, random_seed=0)
    m.fit(X, y, eval_set=(Xv, yv))
    return m.get_all_params()


def main():
    for task in ("regression", "binary"):
        print(f"\n=== {task} ===")
        params = {n: resolved(task, n) for n in SIZES}
        ref = params[SIZES[-1]]
        # Any key whose value is not constant across the size sweep.
        varying = sorted(
            k for k in ref
            if len({repr(params[n].get(k)) for n in SIZES}) > 1
        )
        if not varying:
            print("  no size-dependent defaults found")
            continue
        head = "  " + "rows".ljust(8) + "".join(k[:22].rjust(24) for k in varying)
        print(head)
        for n in SIZES:
            row = "  " + str(n).ljust(8)
            for k in varying:
                row += repr(params[n].get(k))[:22].rjust(24)
            print(row)
    print("\n(reference: full resolved param set at n=60000 has "
          f"{len(resolved('regression', 60_000))} keys)")


if __name__ == "__main__":
    main()
