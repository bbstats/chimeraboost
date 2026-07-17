"""M1 plan prep: enumerate the synth screen suite's multiclass slice and how
many of those sets would actually exercise multiclass cross features
(CROSS_MIN_SAMPLES rows after the ES split, >= 2 numeric columns).

Also: per-class counts (K) and the harness-visible sizes, so the plan can
state the treatment surface honestly. Read-only over frozen synth data.
"""
import sys

sys.path.insert(0, "benchmarks")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from synthgen import api, suites
from chimeraboost.sklearn_api import CROSS_MIN_SAMPLES

print("CROSS_MIN_SAMPLES =", CROSS_MIN_SAMPLES)
rows = []
for key in suites.frozen_keys("screen"):
    if api.task_of(key) != "multiclass":
        continue
    X, y, cat, task, meta = api.build_dataset(key)
    n, d = X.shape
    n_cat = len(cat or [])
    n_num = d - n_cat
    K = len(np.unique(y))
    # Harness: 75/25 train/test, then the classifier's internal ES split
    # (validation_fraction default) -- selection needs len(X_train_post_split)
    # >= CROSS_MIN_SAMPLES. The gate is checked on the post-ES-split X in
    # fit(), i.e. ~ 0.75 * 0.8 * n. Use the classifier's own constants.
    n_fit = int(n * 0.75 * 0.8)
    eligible = (n_fit >= CROSS_MIN_SAMPLES) and (n_num >= 2)
    rows.append((key, n, d, n_num, K, n_fit, eligible))

rows.sort()
print(f"{'key':<14}{'n':>7}{'d':>5}{'num':>5}{'K':>4}{'n_fit':>7}  eligible")
for r in rows:
    print(f"{r[0]:<14}{r[1]:>7}{r[2]:>5}{r[3]:>5}{r[4]:>4}{r[5]:>7}  {r[6]}")
n_el = sum(1 for r in rows if r[6])
print(f"\nmulticlass sets in screen: {len(rows)} of "
      f"{len(suites.frozen_keys('screen'))}; cross-eligible: {n_el}")
canary_keys = {api.key_for(i) for i in suites.CANARIES}
print("multiclass canaries:", sorted(k for k, *_ in rows if k in canary_keys))
