"""F4 C2 microbench: three ways to factorize a real string column.

The first attempt at a fast path (cast to a fixed-width `U` array, `np.unique`,
re-rank) came out a WASH or worse in a same-process fit A/B, so the candidate
gets measured at the column level before any more fit time is spent on it.
`np.unique` sorts, and sorting 38K wide strings is not obviously cheaper than
hashing them once -- which is exactly what the dict loop already does.

Third contender: keep the dict (so every ==/hash class is preserved for free,
including the cross-type ones the U-cast cannot express) but drive it from C.
`map(mapping.setdefault, values, count())` runs the whole loop inside CPython's
map/dict machinery with no per-row bytecode, and the code it assigns to a
category is the ROW INDEX of that category's first appearance -- so sorting the
raw codes recovers first-appearance order, and `np.unique(..., return_inverse)`
turns them into 0..K-1 in one C pass.

Columns come from the real datasets, not synthetic ones: category count and
string length are exactly what decide this.

Run: python benchmarks/f4_c2_micro.py [--datasets hc:okcupid-stem hc:kick]
"""
import argparse
import os
import statistics
import sys
import time
from itertools import count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

from chimeraboost.target_encoding import _factorize_hashed  # noqa: E402


def uniq_path(col):
    """The KILLED design, kept so the comparison stays reproducible: cast to
    fixed-width unicode, `np.unique`, re-rank into first-appearance order. It
    sorts, and sorting wide strings is what loses. Handles only str + missing,
    which is the other half of why it was the wrong idea."""
    lst = col.tolist()
    if not lst or not all(type(v) is str or v is None or v != v for v in lst):
        return None
    miss = np.not_equal(col, col) | np.equal(col, None)
    if miss.any():
        col = col.copy()
        col[miss] = "__nan__"
    arr = col.astype(np.str_)
    su, first, inv = np.unique(arr, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    cats = np.empty(order.size, dtype=object)
    cats[:] = su[order].tolist()
    return rank[np.ravel(inv)], cats


def loop_path(col):
    """The current general case, verbatim from `factorize`."""
    codes = np.empty(col.shape[0], dtype=np.int64)
    mapping = {}
    cats = []
    for i, v in enumerate(col.tolist()):
        if v is None:
            v = "__nan__"
        else:
            try:
                if v != v:
                    v = "__nan__"
            except (TypeError, ValueError):
                v = "__nan__"
        code = mapping.get(v)
        if code is None:
            code = mapping[v] = len(cats)
            cats.append(v)
        codes[i] = code
    return codes, np.asarray(cats, dtype=object)


def setdefault_sort_path(col):
    """The shipped idea's first form, kept for the record: it recovers the 0..K-1
    codes with `np.unique` over the raw first-appearance indices. Correct, but it
    pays a sort that the scatter in `_factorize_hashed` does not."""
    miss = np.not_equal(col, col) | np.equal(col, None)
    if miss.any():
        col = col.copy()
        col[miss] = "__nan__"
    mapping = {}
    raw = np.fromiter(map(mapping.setdefault, col.tolist(), count()),
                      dtype=np.int64, count=col.shape[0])
    _, codes = np.unique(raw, return_inverse=True)
    return np.ravel(codes), np.asarray(list(mapping), dtype=object)


def time_it(fn, col, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(col)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["hc:okcupid-stem", "hc:kick"])
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    variants = (("U-unique", uniq_path), ("setdefault+sort", setdefault_sort_path),
                ("hashed (shipped)", _factorize_hashed))
    print("| dataset | col | n | K | maxlen | loop ms | "
          + " | ".join(f"{n} ms" for n, _ in variants) + " | best |")
    print("|---|---|---|---|---|---|" + "---|" * (len(variants) + 1))
    totals = {"loop": 0.0}
    for key in args.datasets:
        X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
        for j in (cat_idx or []):
            col = np.asarray(X[:, j], dtype=object)
            ref_codes, ref_cats = loop_path(col)
            times = {}
            for name, fn in variants:
                got = fn(col)
                if got is not None:
                    assert np.array_equal(got[0], ref_codes), (key, j, name)
                    assert list(got[1]) == list(ref_cats), (key, j, name)
                times[name] = time_it(fn, col)
                totals[name] = totals.get(name, 0.0) + times[name]
            t_loop = time_it(loop_path, col)
            totals["loop"] += t_loop
            best = min([(t_loop, "loop")] + [(t, n) for n, t in times.items()])[1]
            lens = [len(v) for v in col.tolist() if isinstance(v, str)]
            cells = " | ".join(f"{times[n] * 1e3:.1f}" for n, _ in variants)
            print(f"| {key} | {j} | {col.size} | {len(ref_cats)} | "
                  f"{max(lens) if lens else 0} | {t_loop * 1e3:.1f} | "
                  f"{cells} | {best} |")
    print(f"\nTOTAL over all columns: loop {totals['loop'] * 1e3:.0f} ms")
    for name, _ in variants:
        print(f"  {name:18s} {totals[name] * 1e3:6.0f} ms "
              f"({totals[name] / totals['loop']:.2f}x the loop)")


if __name__ == "__main__":
    main()
