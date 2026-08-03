"""Check the high-card loader against the route it replaced.

The hc: suite used to load through ``sklearn.datasets.fetch_openml``, which asks
OpenML's metadata API for the target column and silently drops any column the
uploader flagged a row identifier. It now reads the dataset's parquet directly,
with the target pinned in HC_DATASETS and those dropped columns named in
``drop_cols``. That is a change to a decision-tier suite, so it was audited
rather than assumed, and this script is the audit.

For each dataset it builds (X, y) both ways, through the same ``_prepare_frame``
the harness uses, and compares column by column: NaN-aware, and insensitive to
dtype spelling (the parquet stores an int where the ARFF gave a float, which the
harness casts away anyway). Anything left is a real difference.

Result when this shipped: thirteen of fourteen datasets are value-identical.
The fourteenth, colleges, agrees to a relative 1.4e-14 on ten feature columns
and its target -- the ARFF stored decimal text and the parquet stores the binary
double, so the two round-trip to neighbouring float64s. Moneyball shows the same
thing at 3.8e-16 on five columns. Nothing else moved.

Needs `pip install -e ".[bench]"` plus scikit-learn's OpenML cache to still hold
the ARFF files, so it only runs where the old route ran. It is a record of a
one-time migration, not a gate -- there is nothing to re-verify unless the
frozen list changes.

    python benchmarks/verify_highcard_load.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

# Anything below this is decimal round-trip noise between the two file formats,
# not a difference in the data.
ULP_TOL = 1e-12


def _compare(sa, sb):
    """(verdict, n_disagreeing, worst_relative_gap) for one column."""
    na, nb = sa.isna().to_numpy(), sb.isna().to_numpy()
    if not np.array_equal(na, nb):
        return "DIFFERENT (missingness)", int((na != nb).sum()), float("nan")
    present = ~na
    an, bn = pd.to_numeric(sa, errors="coerce"), pd.to_numeric(sb, errors="coerce")
    numeric = ((an.notna().to_numpy() | na).all()
               and (bn.notna().to_numpy() | nb).all())
    if numeric:
        av, bv = an.to_numpy(float)[present], bn.to_numpy(float)[present]
        if av.size == 0 or np.array_equal(av, bv):
            return "identical", 0, 0.0
        rel = np.abs(av - bv) / np.maximum(np.abs(av), 1e-300)
        n_real = int((rel > ULP_TOL).sum())
        verdict = "float-noise" if n_real == 0 else "DIFFERENT (value)"
        return verdict, int((av != bv).sum()), float(rel.max())
    # Categorical / free text: compare the strings, missing already aligned.
    a_s, b_s = sa.astype(str).to_numpy()[present], sb.astype(str).to_numpy()[present]
    n_bad = int((a_s != b_s).sum())
    return ("identical" if n_bad == 0 else "DIFFERENT (string)"), n_bad, float("nan")


def main():
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:                                   # pragma: no cover
        print("scikit-learn is required to reproduce the old route")
        return 2

    failures = []
    for name, spec in rb.HC_DATASETS.items():
        ds = fetch_openml(data_id=spec["data_id"], as_frame=True)
        old_X, old_y = rb._prepare_frame(ds.frame, ds.target.name, {}, None,
                                         rb._HIGHCARD_MAX_ROWS)
        frame = pd.read_parquet(rb._public_parquet_path(spec["data_id"]))
        new_X, new_y = rb._prepare_frame(frame, spec["target"], spec, None,
                                         rb._HIGHCARD_MAX_ROWS)

        if list(old_X.columns) != list(new_X.columns):
            failures.append(name)
            only_old = [c for c in old_X.columns if c not in new_X.columns]
            only_new = [c for c in new_X.columns if c not in old_X.columns]
            print(f"FAIL   {name:24s} columns differ: "
                  f"old-only={only_old} new-only={only_new}")
            continue

        noisy, broken, worst = 0, [], 0.0
        for col in old_X.columns:
            verdict, _, gap = _compare(old_X[col].reset_index(drop=True),
                                       new_X[col].reset_index(drop=True))
            if verdict == "float-noise":
                noisy += 1
                worst = max(worst, gap)
            elif verdict.startswith("DIFFERENT"):
                broken.append((col, verdict))
        y_verdict, _, y_gap = _compare(pd.Series(np.asarray(old_y)),
                                       pd.Series(np.asarray(new_y)))
        if y_verdict.startswith("DIFFERENT"):
            broken.append(("<target>", y_verdict))
        worst = max(worst, 0.0 if y_gap != y_gap else y_gap)

        if broken:
            failures.append(name)
        status = "FAIL  " if broken else ("noise " if noisy else "same  ")
        print(f"{status} {name:24s} {len(old_X.columns):3d} cols, "
              f"{noisy} with round-trip noise, worst relative gap {worst:.1e}")
        for col, verdict in broken[:5]:
            print(f"         ! {col}: {verdict}")

    print()
    if failures:
        print(f"{len(failures)} dataset(s) changed: {', '.join(failures)}")
        return 1
    print(f"all {len(rb.HC_DATASETS)} datasets load the same both ways")
    return 0


if __name__ == "__main__":
    sys.exit(main())
