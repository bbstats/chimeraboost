"""Profile a ChimeraBoost fit on a representative dataset.

Picks adult (n=32K, mixed numeric/categorical) by default since it is the
slowest dataset in our benchmark and exercises every code path. Reports:

  * End-to-end wall-clock fit time
  * Per-phase breakdown (tree build vs everything else)
  * Top cProfile hotspots at the Python level

Note: numba @njit functions are opaque to cProfile (they show up as a single
call into the dispatcher). Use the per-phase breakdown for tree-internal time
and cProfile to spot unexpected pure-Python overhead.

Run:
    python benchmarks/profile_fit.py
    python benchmarks/profile_fit.py --dataset car        # multiclass path
"""
import argparse
import cProfile
import io
import pstats
import time

import numpy as np

# Patch BEFORE constructing any booster so the timing wrapper is picked up.
import chimeraboost.booster as bm

_phase_times = {"build_tree": 0.0}
_orig_build = bm.build_oblivious_tree


def _timed_build(*args, **kw):
    t0 = time.perf_counter()
    r = _orig_build(*args, **kw)
    _phase_times["build_tree"] += time.perf_counter() - t0
    return r


bm.build_oblivious_tree = _timed_build

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# Same OpenML loader the main benchmark uses, distilled.
DATASETS = {
    "adult":       dict(data_id=1590, task="binary"),
    "bank":        dict(data_id=1461, task="binary"),
    "car":         dict(data_id=40975, task="multiclass"),
    "phoneme":     dict(data_id=1489, task="binary"),
    "electricity": dict(data_id=151,  task="binary"),
    "cpu_act":     dict(data_id=197,  task="regression"),
}


def load(name):
    spec = DATASETS[name]
    ds = fetch_openml(data_id=spec["data_id"], as_frame=True)
    df = ds.frame
    y = ds.target
    X_df = df.drop(columns=[ds.target.name])

    def _is_cat(d):
        s = str(d).lower()
        return s in ("category", "object") or s.startswith("string")
    cat_idx = [i for i, c in enumerate(X_df.columns) if _is_cat(X_df[c].dtype)]

    task = spec["task"]
    if task == "regression":
        y = y.astype(float).to_numpy()
    else:
        y = y.astype("category").cat.codes.to_numpy()

    if cat_idx:
        import pandas as pd
        cols = []
        for i, c in enumerate(X_df.columns):
            s = X_df[c]
            if i in cat_idx:
                cols.append(s.astype(object).where(s.notna(), "__nan__"))
            else:
                cols.append(s.astype(float))
        X = pd.concat(cols, axis=1).to_numpy(dtype=object)
    else:
        X = X_df.to_numpy(dtype=float)
    return X, y, (cat_idx or None), task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS))
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--top", type=int, default=25,
                    help="how many cProfile rows to print")
    args = ap.parse_args()

    print(f"Loading {args.dataset}...")
    X, y, cat_idx, task = load(args.dataset)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None,
    )
    print(f"  n_train={len(Xtr)}, n_features={Xtr.shape[1]}, "
          f"cat_features={len(cat_idx) if cat_idx else 0}, task={task}")

    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier

    # JIT warmup so first-iteration compile cost doesn't pollute the profile.
    print("Warmup (compiling numba kernels)...")
    warm_n = min(500, len(Xtr))
    Est(n_estimators=5, random_state=0).fit(
        Xtr[:warm_n], ytr[:warm_n], cat_features=cat_idx
    )
    _phase_times["build_tree"] = 0.0

    print("Profiling fit...")
    kw = dict(n_estimators=args.n_estimators, random_state=0)
    if not args.no_early_stopping:
        kw.update(early_stopping=True, early_stopping_rounds=50,
                  validation_fraction=0.15)

    t0 = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    m = Est(**kw).fit(Xtr, ytr, cat_features=cat_idx)
    profiler.disable()
    total = time.perf_counter() - t0

    n_trees = (len(m.model_.trees_)
               if not isinstance(m.model_.trees_[0], list)
               else sum(len(t) for t in m.model_.trees_))
    print(f"\nTotal fit: {total:.2f}s  trees={n_trees}  "
          f"(per-tree: {1000*total/max(n_trees,1):.2f} ms)")
    tb = _phase_times["build_tree"]
    print(f"  build_oblivious_tree: {tb:.2f}s  ({100*tb/total:.1f}%)")
    print(f"  everything else:      {total-tb:.2f}s  ({100*(total-tb)/total:.1f}%)")

    print(f"\nTop {args.top} cProfile rows by cumulative time:")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(args.top)
    print(s.getvalue())

    print(f"Top {args.top} cProfile rows by self (tottime):")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    ps.print_stats(args.top)
    print(s.getvalue())


if __name__ == "__main__":
    main()
