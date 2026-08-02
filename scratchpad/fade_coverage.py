"""How much of full-size Grinsztajn does the size fade actually touch?

The internal Pareto chart runs on Grinsztajn base, where most datasets are far
above the fade's upper threshold. This counts how many are below it, so the
chart's (non-)movement can be read honestly rather than guessed at.
"""
import json
import sys

from chimeraboost.booster import _AUTO_LR_HI, _AUTO_LR_LO

PATH = sys.argv[1]
blob = json.load(open(PATH, "r", encoding="utf-8"))
meta = blob["datasets"]

# The booster trains on (1 - validation_fraction) of the rows the harness hands
# it, and the harness holds out a test split first. Both are 20%.
below_lo = above_hi = middle = 0
rows = []
for ds, m in sorted(meta.items()):
    n = m.get("n_train")
    if not n:
        continue
    # n_train is what the harness hands the estimator; the estimator then takes
    # validation_fraction=0.2 out before the booster sees the rest.
    n_boost = int(n * 0.8)
    rows.append((n_boost, ds))
    if n_boost <= _AUTO_LR_LO:
        below_lo += 1
    elif n_boost >= _AUTO_LR_HI:
        above_hi += 1
    else:
        middle += 1

total = below_lo + middle + above_hi
print(f"{PATH}: {total} datasets with a row count\n")
print(f"  fully faded (<= {_AUTO_LR_LO:,} booster rows, rate 0.07): {below_lo}")
print(f"  partially faded ({_AUTO_LR_LO:,}-{_AUTO_LR_HI:,}):          {middle}")
print(f"  UNTOUCHED (>= {_AUTO_LR_HI:,}, rate stays 0.100):        {above_hi}")
if total:
    print(f"\n  => the flip can move {below_lo + middle} of {total} datasets "
          f"({100 * (below_lo + middle) / total:.0f}%); the rest are "
          f"byte-identical to 0.29.0")

print("\nsmallest ten by booster rows:")
for n_boost, ds in sorted(rows)[:10]:
    print(f"  {n_boost:8,}  {ds}")
