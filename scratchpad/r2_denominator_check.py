"""Does the R2 leg care whether y_std is full-data or test-split?

make_pareto's regression skill score divides the test-set RMSE by `y_std`,
which run_benchmarks records over the FULL target. The textbook R2 uses the
evaluation set's own variance. The classification leg was switched to a
test-split reference (class_prior) when the axis shipped, so this asks whether
the regression leg needs the same treatment or whether the difference is noise.

Replicates the harness split exactly -- train_test_split(test_size=0.25,
random_state=seed), no stratification for regression -- and recomputes R2 both
ways. No model fitting.
"""
import json
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, "benchmarks")
import run_benchmarks as rb  # noqa: E402

PATH = sys.argv[1]
SEEDS = (0, 1, 2)

blob = json.load(open(PATH, "r", encoding="utf-8"))
meta, records = blob["datasets"], blob["records"]
rb._add_grinsztajn_datasets()

# mean test RMSE per (dataset, model)
rmse = {}
for r in records:
    if meta.get(r["dataset"], {}).get("task") != "regression":
        continue
    v = r["metrics"].get("rmse")
    if v is not None:
        rmse.setdefault((r["dataset"], r["model"]), []).append(v)
rmse = {k: float(np.mean(v)) for k, v in rmse.items()}

reg_ds = sorted({ds for ds, _ in rmse})
print(f"{len(reg_ds)} regression datasets\n")

std_test = {}
worst = []
for ds in reg_ds:
    builder = rb.DATASETS.get(ds)
    if builder is None:
        continue
    X, y = builder(1.0, np.random.default_rng(0))[:2]
    stds = []
    for seed in SEEDS:
        _, _, _, yte = train_test_split(X, y, test_size=0.25,
                                        random_state=seed, stratify=None)
        stds.append(float(np.std(np.asarray(yte, dtype=float))))
    std_test[ds] = float(np.mean(stds))
    full = meta[ds].get("y_std")
    if full:
        worst.append((abs(std_test[ds] / full - 1.0), ds, full, std_test[ds]))

worst.sort(reverse=True)
print("largest full-vs-test target-std disagreements:")
for rel, ds, full, test in worst[:8]:
    print(f"  {rel:7.2%}  {ds:42s} full {full:12.4f}  test {test:12.4f}")

models = sorted({m for _, m in rmse})
print(f"\n{'model':24s}{'R2 (full y_std)':>17s}{'R2 (test y_std)':>17s}{'delta':>10s}")
print("-" * 70)
rows = []
for m in models:
    a, b = [], []
    for ds in reg_ds:
        v = rmse.get((ds, m))
        full = meta[ds].get("y_std")
        if v is None or not full or ds not in std_test:
            continue
        a.append(1.0 - (v / full) ** 2)
        b.append(1.0 - (v / std_test[ds]) ** 2)
    if a:
        rows.append((m, float(np.mean(a)), float(np.mean(b))))
for m, a, b in sorted(rows, key=lambda r: -r[1]):
    print(f"{m:24s}{a:17.4f}{b:17.4f}{b - a:+10.4f}")

order_full = [m for m, _, _ in sorted(rows, key=lambda r: -r[1])]
order_test = [m for m, _, _ in sorted(rows, key=lambda r: -r[2])]
print(f"\nranking identical: {order_full == order_test}")
if order_full != order_test:
    print(f"  full: {order_full}")
    print(f"  test: {order_test}")
spread_a = max(r[1] for r in rows) - min(r[1] for r in rows)
spread_b = max(r[2] for r in rows) - min(r[2] for r in rows)
print(f"axis span: full {spread_a:.4f}   test {spread_b:.4f}")
