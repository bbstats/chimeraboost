"""Phase-0 grow-kernel attribution (GROW_PLAN.md).

Splits tree-build time BETWEEN its kernels -- the fused scatter+scan
(`_build_and_split`), leaf descend, constant `_leaf_values`, the linear-leaf
ridge (`_linear_leaf_fit`), and per-tree Python residue -- via timing wrappers
monkeypatched around the chimeraboost.tree call sites. numba is opaque to
cProfile, so this is targeted perf_counter accounting, summed per estimator
fit (auditions and refits included: every booster fit pays the grow cost).
No library change; the wrappers add ~1us per kernel call (~15 calls/tree)
against kernel times in the 100us+ range.

Modes (run one at a time, plan items 1-5):

    python benchmarks/profile_grow.py --attribution --seeds 2
        Items 1+4: wall split of default fits on the frozen 9-set panel,
        incl. multiclass copy/launch accounting (okcupid-stem).

    python benchmarks/profile_grow.py --ll-delta --seeds 2
        Item 2: binary sets fit with linear_leaves forced True vs False;
        ms/tree delta = the ridge's true cost share.

    python benchmarks/profile_grow.py --threads
        Item 3: feature-parallel saturation, narrow (MagicTelescope) vs
        wide (road-safety) at thread_count 1/2/6/12.

    python benchmarks/profile_grow.py --micro
        Item 5: scatter + descend stream-width microbench, uint8 vs uint16
        Xb and int32 vs int64 leaf, synthetic shapes.

Writes benchmarks/results/<out>.json and .md, and prints the tables.
"""
import argparse
import collections
import json
import os
import time

import numpy as np

import chimeraboost.tree as tmod
import chimeraboost.booster as bmod
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


# --------------------------------------------------------------------------
# Timing wrappers (installed once, gated by _REC so warmup/loads stay out)
# --------------------------------------------------------------------------
_REC = {"on": False}
_T = collections.defaultdict(float)
_N = collections.defaultdict(int)


def _wrap(name, func):
    def wrapped(*a, **kw):
        if not _REC["on"]:
            return func(*a, **kw)
        t0 = time.perf_counter()
        r = func(*a, **kw)
        _T[name] += time.perf_counter() - t0
        _N[name] += 1
        return r
    return wrapped


class _TimedNp:
    """numpy proxy for the booster module namespace only: times the K
    per-class ascontiguousarray grad/hess column copies of the multiclass
    round loop (1-D float64 is unique to them during fit -- the prep/SHAP
    copies are 2-D binned matrices). Everything else delegates untouched."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def ascontiguousarray(self, a, *args, **kw):
        real = self._real
        if not _REC["on"]:
            return real.ascontiguousarray(a, *args, **kw)
        t0 = time.perf_counter()
        r = real.ascontiguousarray(a, *args, **kw)
        dt = time.perf_counter() - t0
        if getattr(a, "ndim", None) == 1 and \
                getattr(a, "dtype", None) == real.float64:
            _T["gradcopy"] += dt
            _N["gradcopy"] += 1
        return r


def _install_patches():
    """Kernels are module globals resolved at call time by the pure-Python
    build_oblivious_tree, so patching tmod reaches the fit path; the booster
    holds its own reference to build_oblivious_tree, so that one is patched
    in bmod."""
    tmod._build_and_split = _wrap("split", tmod._build_and_split)
    tmod._descend_leaves = _wrap("descend", tmod._descend_leaves)
    tmod._descend_leaves_serial = _wrap("descend", tmod._descend_leaves_serial)
    tmod._leaf_values = _wrap("leaf_values", tmod._leaf_values)
    tmod._linear_leaf_fit = _wrap("linear_fit", tmod._linear_leaf_fit)
    bmod.build_oblivious_tree = _wrap("grow", bmod.build_oblivious_tree)
    bmod.np = _TimedNp(np)


def _timed_fit(est, Xtr, ytr, cat):
    """Fit with recording on; returns (total_s, times, counts, est)."""
    _T.clear()
    _N.clear()
    _REC["on"] = True
    t0 = time.perf_counter()
    est.fit(Xtr, ytr, cat_features=cat)
    total = time.perf_counter() - t0
    _REC["on"] = False
    return total, dict(_T), dict(_N), est


def _warmup():
    print("Warmup (compiling numba kernels)...")
    from chimeraboost.warmup import warmup
    warmup()


# --------------------------------------------------------------------------
# Panels (same keys / loaders / row caps as the decision suites)
# --------------------------------------------------------------------------
# The frozen step-0 panel: spans task types, n, width, cat regime.
ATTR_DATASETS = [
    "gr:reg_num/cpu_act",                  # regression, 8K, numeric only
    "gr:reg_num/diamonds",                 # regression, 50K cap, numeric only
    "gr:reg_cat/nyc-taxi-green-dec-2016",  # regression, 50K cap, with cats
    "gr:clf_num/MagicTelescope",           # binary, 13K, numeric only
    "gr:clf_num/Higgs",                    # binary, 50K cap, numeric only
    "gr:clf_cat/road-safety",              # binary, 50K cap, with cats
    "hc:kick",                             # binary, 73K, high-card cats
    "hc:wine-reviews",                     # regression, 100K cap, high-card
    "hc:okcupid-stem",                     # multiclass, high-card cats
]

# Item 2: the binary anomaly (3.6 ms/tree vs regression 1.2-1.7).
LL_DATASETS = [
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/Higgs",
    "gr:clf_cat/road-safety",
    "hc:kick",
]

# Item 3: narrow (10 features) vs wide (32+) for the feature-parallel read.
THREAD_DATASETS = ["gr:clf_num/MagicTelescope", "gr:clf_cat/road-safety"]
THREAD_COUNTS = [1, 2, 6, 12]


def _load_panel(keys):
    import run_benchmarks as rb
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    out = []
    for key in keys:
        print(f"Loading {key}...")
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        out.append((key, X, y, cat, task))
    return out


def _split_xy(X, y, task, seed):
    from sklearn.model_selection import train_test_split
    strat = y if task != "regression" else None
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat)
    return Xtr, ytr


def _est_for(task, **kw):
    Est = ChimeraBoostRegressor if task == "regression" \
        else ChimeraBoostClassifier
    return Est(random_state=0, **kw)


def _pct(x, tot):
    return f"{100.0 * x / tot:5.1f}" if tot > 0 else "  0.0"


def _save(out, results, report):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", out)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".json", "w") as f:
        json.dump(results, f)
    with open(base + ".md", "w", newline="\n") as f:
        f.write(report)
    print(report)
    print(f"Saved {base}.json and {base}.md")


# --------------------------------------------------------------------------
# Mode 1+4: attribution panel
# --------------------------------------------------------------------------
def run_attribution(args):
    panel = _load_panel(args.datasets or ATTR_DATASETS)
    _warmup()
    _install_patches()

    results = []
    for key, X, y, cat, task in panel:
        for seed in range(args.seeds):
            Xtr, ytr = _split_xy(X, y, task, seed)
            total, t, n, est = _timed_fit(_est_for(task), Xtr, ytr, cat)
            rec = {"dataset": key, "task": task, "seed": seed,
                   "n_train": int(Xtr.shape[0]),
                   "n_features": int(Xtr.shape[1]),
                   "n_cats": len(cat) if cat else 0,
                   "total_s": total, "t": t, "n": n}
            if task == "multiclass":
                rec["rounds"] = len(est.model_.trees_)
                rec["K"] = int(est.model_.n_classes_)
            results.append(rec)
            print(f"  {key} seed {seed}: {total:.1f}s, "
                  f"grow {_pct(t.get('grow', 0), total)}% "
                  f"({n.get('grow', 0)} trees)")
    _save(args.out or "grow-phase0", results, attr_report(results))


def attr_report(results):
    by_ds = collections.defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)

    lines = ["# Grow-kernel attribution (GROW_PLAN.md Phase 0, items 1+4)",
             "",
             "## Tree-build wall split (% of estimator fit; all booster fits"
             " incl. auditions)",
             "",
             "| dataset | task | n_train | fit_s | grow% | split% | descend% "
             "| leafv% | linfit% | pytree% | nontree% | ms/tree |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for ds, recs in by_ds.items():
        r0 = recs[0]
        tot = sum(r["total_s"] for r in recs)
        t = collections.defaultdict(float)
        trees = 0
        for r in recs:
            for k, v in r["t"].items():
                t[k] += v
            trees += r["n"].get("grow", 0)
        pytree = t["grow"] - t["split"] - t["descend"] \
            - t["leaf_values"] - t["linear_fit"]
        lines.append(
            f"| {ds} | {r0['task']} | {r0['n_train']} "
            f"| {tot / len(recs):.1f} | {_pct(t['grow'], tot)} "
            f"| {_pct(t['split'], tot)} | {_pct(t['descend'], tot)} "
            f"| {_pct(t['leaf_values'], tot)} | {_pct(t['linear_fit'], tot)} "
            f"| {_pct(pytree, tot)} | {_pct(tot - t['grow'], tot)} "
            f"| {1000 * t['grow'] / max(trees, 1):.2f} |")
    lines += ["",
              "split/descend/leafv/linfit are inside grow; pytree = grow "
              "minus its kernels (per-tree Python: list/np.array builds, "
              "bincount/flatnonzero at small n, ObliviousTree ctor). "
              "nontree = prep + loss + F updates + val predict (split in "
              "pareto-step0).", ""]

    mc = [r for r in results if r["task"] == "multiclass"]
    if mc:
        lines += ["## Multiclass detail (item 4: per-class copies and "
                  "kernel launches)",
                  "",
                  "| dataset | K | rounds | trees | split calls | descend "
                  "calls | copies | gradcopy_s | gradcopy% |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for ds in {r["dataset"] for r in mc}:
            recs = [r for r in mc if r["dataset"] == ds]
            tot = sum(r["total_s"] for r in recs)
            gc = sum(r["t"].get("gradcopy", 0) for r in recs)
            m = len(recs)
            lines.append(
                f"| {ds} | {recs[0]['K']} "
                f"| {sum(r['rounds'] for r in recs) / m:.0f} "
                f"| {sum(r['n'].get('grow', 0) for r in recs) / m:.0f} "
                f"| {sum(r['n'].get('split', 0) for r in recs) / m:.0f} "
                f"| {sum(r['n'].get('descend', 0) for r in recs) / m:.0f} "
                f"| {sum(r['n'].get('gradcopy', 0) for r in recs) / m:.0f} "
                f"| {gc / m:.2f} | {_pct(gc, tot)} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mode 2: linear_leaves True vs False (the binary ms/tree anomaly)
# --------------------------------------------------------------------------
def run_ll_delta(args):
    panel = _load_panel(args.datasets or LL_DATASETS)
    _warmup()
    _install_patches()

    results = []
    for key, X, y, cat, task in panel:
        for seed in range(args.seeds):
            Xtr, ytr = _split_xy(X, y, task, seed)
            rec = {"dataset": key, "seed": seed, "n_train": int(Xtr.shape[0])}
            for ll in (True, False):
                total, t, n, _ = _timed_fit(
                    _est_for(task, linear_leaves=ll), Xtr, ytr, cat)
                tag = "T" if ll else "F"
                rec[tag] = {"total_s": total, "t": t, "n": n}
                print(f"  {key} seed {seed} ll={ll}: {total:.1f}s, "
                      f"{n.get('grow', 0)} trees, "
                      f"{1000 * t.get('grow', 0) / max(n.get('grow', 0), 1):.2f}"
                      f" ms/tree")
            results.append(rec)
    _save(args.out or "grow-phase0-ll", results, ll_report(results))


def ll_report(results):
    by_ds = collections.defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    lines = ["# linear_leaves True vs False (GROW_PLAN.md Phase 0, item 2)",
             "",
             "ms/tree = grow wall / trees built, whole estimator fit "
             "(normalizes for different early-stop rounds).",
             "",
             "| dataset | ll=T ms/tree | ll=F ms/tree | delta | ridge share "
             "of ll=T grow | trees T/F | fit_s T/F |",
             "|---|--:|--:|--:|--:|---|---|"]
    for ds, recs in by_ds.items():
        mt = mf = rs = tt = tf = st = sf = 0.0
        for r in recs:
            m = len(recs)
            gT, nT = r["T"]["t"].get("grow", 0), r["T"]["n"].get("grow", 1)
            gF, nF = r["F"]["t"].get("grow", 0), r["F"]["n"].get("grow", 1)
            mt += 1000 * gT / nT / m
            mf += 1000 * gF / nF / m
            rs += r["T"]["t"].get("linear_fit", 0) / max(gT, 1e-9) / m
            tt += nT / m
            tf += nF / m
            st += r["T"]["total_s"] / m
            sf += r["F"]["total_s"] / m
        lines.append(
            f"| {ds} | {mt:.2f} | {mf:.2f} | {100 * (mt - mf) / mt:+.0f}% "
            f"| {100 * rs:.0f}% | {tt:.0f}/{tf:.0f} | {st:.1f}/{sf:.1f} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mode 3: thread geometry (feature-parallel saturation)
# --------------------------------------------------------------------------
def run_threads(args):
    import numba
    max_t = numba.config.NUMBA_NUM_THREADS
    counts = sorted({min(c, max_t) for c in THREAD_COUNTS})
    print(f"NUMBA_NUM_THREADS={max_t}; measuring at {counts}")
    panel = _load_panel(args.datasets or THREAD_DATASETS)
    _warmup()
    _install_patches()

    results = []
    for key, X, y, cat, task in panel:
        Xtr, ytr = _split_xy(X, y, task, 0)
        for tc in counts:
            total, t, n, _ = _timed_fit(
                _est_for(task, thread_count=tc), Xtr, ytr, cat)
            results.append({"dataset": key, "n_features": int(Xtr.shape[1]),
                            "threads": tc, "total_s": total, "t": t, "n": n})
            print(f"  {key} threads={tc}: fit {total:.1f}s, "
                  f"split {t.get('split', 0):.2f}s")
    _save(args.out or "grow-phase0-threads", results, threads_report(results))


def threads_report(results):
    by_ds = collections.defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    lines = ["# Feature-parallel thread scaling (GROW_PLAN.md Phase 0, "
             "item 3)",
             "",
             "Flat split_s scaling on the narrow set = the known "
             "feature-parallel ceiling (Phase-2 class; record only).",
             "",
             "| dataset | feats | threads | split_s | split speedup | fit_s "
             "| fit speedup |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for ds, recs in by_ds.items():
        recs = sorted(recs, key=lambda r: r["threads"])
        s1 = recs[0]["t"].get("split", 0)
        f1 = recs[0]["total_s"]
        for r in recs:
            s = r["t"].get("split", 0)
            lines.append(
                f"| {ds} | {r['n_features']} | {r['threads']} | {s:.2f} "
                f"| x{s1 / max(s, 1e-9):.2f} | {r['total_s']:.1f} "
                f"| x{f1 / r['total_s']:.2f} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# L-ridge kernel-vs-kernel microbench (isolates the restructure from fit
# noise: same inputs, median of warm reps, ref vs new)
# --------------------------------------------------------------------------
def run_ridge_micro(args):
    if not hasattr(tmod, "_linear_leaf_fit_ref"):
        print("Needs the (reverted) L-ridge variant, which shipped a "
              "_linear_leaf_fit_ref oracle; see GROW_PLAN.md L-ridge "
              "verdict for the recorded numbers.")
        return
    REPS = 21
    rng = np.random.default_rng(0)
    results = []
    for n in (8192, 37500, 75000):
        for k in (2, 4, 6):
            n_features, max_bins, n_leaves = 8, 128, 64
            Xb = rng.integers(0, max_bins, size=(n_features, n)) \
                .astype(np.uint16)
            grad = rng.standard_normal(n)
            hess = rng.random(n) + 0.1
            leaf = rng.integers(0, n_leaves, n).astype(np.int64)
            lin_feats = np.arange(k, dtype=np.int64)
            centers_std = rng.standard_normal((n_features, max_bins))
            centers_std[rng.random((n_features, max_bins)) < 0.05] = np.nan
            a = (leaf, grad, hess, n_leaves, lin_feats, centers_std, Xb,
                 1.0, 1.0, 0.15)
            row = {"n": n, "k": k}
            # NOTE: call the dispatchers directly — numba sets __wrapped__ to
            # the raw py_func, so "unwrapping" here times interpreted Python.
            for name, fn in (("ref", tmod._linear_leaf_fit_ref),
                             ("new", tmod._linear_leaf_fit)):
                fn(*a)                                    # compile + warm
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    fn(*a)
                    ts.append(time.perf_counter() - t0)
                row[name + "_ms"] = 1000 * sorted(ts)[REPS // 2]
            results.append(row)
            print(f"  n={n} k={k}: ref {row['ref_ms']:.3f}ms "
                  f"new {row['new_ms']:.3f}ms "
                  f"(x{row['ref_ms'] / row['new_ms']:.2f})")
    lines = ["# L-ridge kernel-vs-kernel microbench (median of 21 warm reps,"
             " 64 leaves)",
             "",
             "| n | k | ref_ms | new_ms | speedup |",
             "|--:|--:|--:|--:|--:|"]
    for r in results:
        lines.append(f"| {r['n']} | {r['k']} | {r['ref_ms']:.3f} "
                     f"| {r['new_ms']:.3f} "
                     f"| x{r['ref_ms'] / r['new_ms']:.2f} |")
    lines.append("")
    _save(args.out or "grow-lridge-micro", results, "\n".join(lines))


# --------------------------------------------------------------------------
# Mode 5: stream-width microbench (dtype levers, no library change)
# --------------------------------------------------------------------------
def run_micro(args):
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def scatter(Xb, grad, hess, leaf, hist, nb):
        # The fused kernel's zero+scatter portion, verbatim; numba
        # specializes per (Xb, leaf) dtype pair so one source serves all
        # four variants.
        n_features, n_samples = Xb.shape
        n_leaves = hist.shape[1]
        for f in prange(n_features):
            for l in range(n_leaves):
                for b in range(nb):
                    hist[f, l, b, 0] = 0.0
                    hist[f, l, b, 1] = 0.0
            Xf = Xb[f]
            for i in range(n_samples):
                l = leaf[i]
                b = Xf[i]
                hist[f, l, b, 0] += grad[i]
                hist[f, l, b, 1] += hess[i]

    @njit(cache=True, parallel=True)
    def descend(leaf, Xf, t):
        for i in prange(leaf.shape[0]):
            leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > t else 0)

    NB, N_LEAVES = 128, 32
    SHAPES = [(10, 8192), (32, 8192), (10, 50000), (32, 50000),
              (10, 200000), (32, 200000)]
    REPS = 15
    rng = np.random.default_rng(0)
    results = []
    for n_features, n in SHAPES:
        Xb16 = rng.integers(0, NB, size=(n_features, n)).astype(np.uint16)
        Xb8 = Xb16.astype(np.uint8)
        grad = rng.standard_normal(n)
        hess = rng.random(n) + 0.5
        leaf0_64 = rng.integers(0, N_LEAVES, n).astype(np.int64)
        leaf0_32 = leaf0_64.astype(np.int32)
        hist = np.zeros((n_features, N_LEAVES, NB, 2))
        for xb_name, Xb in (("u16", Xb16), ("u8", Xb8)):
            for lf_name, leaf0 in (("i64", leaf0_64), ("i32", leaf0_32)):
                leaf = leaf0.copy()
                scatter(Xb, grad, hess, leaf, hist, NB)   # compile + warm
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    scatter(Xb, grad, hess, leaf, hist, NB)
                    ts.append(time.perf_counter() - t0)
                s_ms = 1000 * sorted(ts)[REPS // 2]
                descend(leaf, Xb[0], NB // 2)             # compile + warm
                ts = []
                for _ in range(REPS):
                    leaf[:] = leaf0                       # un-shift, untimed
                    t0 = time.perf_counter()
                    descend(leaf, Xb[0], NB // 2)
                    ts.append(time.perf_counter() - t0)
                d_ms = 1000 * sorted(ts)[REPS // 2]
                results.append({"n_features": n_features, "n": n,
                                "Xb": xb_name, "leaf": lf_name,
                                "scatter_ms": s_ms, "descend_ms": d_ms})
                print(f"  F={n_features} n={n} Xb={xb_name} leaf={lf_name}: "
                      f"scatter {s_ms:.3f}ms descend {d_ms:.3f}ms")
    _save(args.out or "grow-phase0-micro", results, micro_report(results))


def micro_report(results):
    lines = ["# Scatter/descend stream-width microbench (GROW_PLAN.md "
             "Phase 0, item 5)",
             "",
             "Median of 15 warm reps; n_leaves=32, 128 bins. Ratios are vs "
             "the current u16/i64 layout (bounds for L-bin8 / L-leaf32).",
             "",
             "| feats | n | Xb/leaf | scatter_ms | vs u16/i64 | descend_ms "
             "| vs u16/i64 |",
             "|--:|--:|---|--:|--:|--:|--:|"]
    by_shape = collections.defaultdict(dict)
    for r in results:
        by_shape[(r["n_features"], r["n"])][(r["Xb"], r["leaf"])] = r
    for (nf, n), d in by_shape.items():
        base = d[("u16", "i64")]
        for (xb, lf), r in d.items():
            lines.append(
                f"| {nf} | {n} | {xb}/{lf} | {r['scatter_ms']:.3f} "
                f"| x{base['scatter_ms'] / max(r['scatter_ms'], 1e-9):.2f} "
                f"| {r['descend_ms']:.3f} "
                f"| x{base['descend_ms'] / max(r['descend_ms'], 1e-9):.2f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attribution", action="store_true")
    mode.add_argument("--ll-delta", action="store_true")
    mode.add_argument("--threads", action="store_true")
    mode.add_argument("--micro", action="store_true")
    mode.add_argument("--ridge-micro", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="run_benchmarks DATASETS keys (default: the mode's "
                         "registered panel)")
    ap.add_argument("--out", default=None,
                    help="results/<out>.json|.md (default per mode)")
    args = ap.parse_args()

    if args.attribution:
        run_attribution(args)
    elif args.ll_delta:
        run_ll_delta(args)
    elif args.threads:
        run_threads(args)
    elif args.micro:
        run_micro(args)
    elif args.ridge_micro:
        run_ridge_micro(args)


if __name__ == "__main__":
    main()
