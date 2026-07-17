"""M1 plan prep: which hc multiclass sets clear the cross-feature gates
(post-split train n >= CROSS_MIN_SAMPLES, >= 2 numeric columns)?"""
import sys

sys.path.insert(0, "benchmarks")
sys.stdout.reconfigure(encoding="utf-8")

from run_benchmarks import DATASETS, _add_highcard_datasets  # noqa: E402
import numpy as np  # noqa: E402
from chimeraboost.sklearn_api import CROSS_MIN_SAMPLES  # noqa: E402

_add_highcard_datasets()
rng = np.random.default_rng(0)
for name in ("hc:okcupid-stem", "hc:Traffic_violations", "hc:cjs",
             "hc:eucalyptus"):
    X, y, cat, task = DATASETS[name](1.0, rng)
    n, d = X.shape
    n_cat = len(cat or [])
    n_num = d - n_cat
    K = len(np.unique(y))
    n_fit = int(n * 0.75 * 0.8)
    ok = n_fit >= CROSS_MIN_SAMPLES and n_num >= 2
    print(f"{name:<24} n={n:>6} d={d:>3} num={n_num:>3} K={K} "
          f"n_fit~{n_fit:>6} eligible={ok}")
