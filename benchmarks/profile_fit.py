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

Attribution mode (PARETO_PLAN.md Track 1 step 0): time the DEFAULT estimator
fit across representative Grinsztajn + hc datasets, split by selection fit
(const / linear / cross-augmented) and by phase (tree growth, ordered-TS
encoding, binning, validation predict, leaf refinement), record the
linear_leaves / cross_features selection outcomes, and keep each variant's
validation curve so a raced selector can be simulated offline.

    python benchmarks/profile_fit.py --attribution --seeds 3 --out pareto-step0

Writes benchmarks/results/<out>.json (full records incl. val curves) and
benchmarks/results/<out>.md (the report tables), and prints the tables.
"""
import argparse
import collections
import cProfile
import io
import json
import os
import pstats
import time


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


# --------------------------------------------------------------------------
# Attribution mode (PARETO_PLAN.md Track 1 step 0)
# --------------------------------------------------------------------------
# Default panel: 6 Grinsztajn + 3 hc sets spanning task type (regression /
# binary / multiclass), size (8K -> 100K rows), and cat regime (none -> high
# cardinality). Keys are run_benchmarks.DATASETS keys, so loaders and row caps
# are identical to the decision suites.
ATTR_DATASETS = [
    "gr:reg_num/cpu_act",                  # regression, 8K, numeric only
    "gr:reg_num/diamonds",                 # regression, 50K cap, numeric only
    "gr:reg_cat/nyc-taxi-green-dec-2016",  # regression, 50K cap, with cats
    "gr:clf_num/MagicTelescope",           # binary, 13K, numeric only
    "gr:clf_num/Higgs",                    # binary, 50K cap, numeric only
    "gr:clf_cat/road-safety",              # binary, 50K cap, with cats
    "hc:kick",                             # binary, 73K, high-card cats
    "hc:wine-reviews",                     # regression, 100K cap, high-card cats
    "hc:okcupid-stem",                     # multiclass, high-card cats
]

# Shared recorder. `stack` nests per-booster-fit phase accumulators (depth is
# 1 in practice; the stack keeps post-fit calibration/eval calls, which run
# with an empty stack, out of the numbers). `fits` is the per-estimator-fit
# list of booster-fit records, or None while recording is off (warmup, loads).
_ATTR = {"stack": [], "fits": None}


def _phase_wrap(name, func):
    def wrapped(*a, **kw):
        if not _ATTR["stack"]:
            return func(*a, **kw)
        t0 = time.perf_counter()
        r = func(*a, **kw)
        _ATTR["stack"][-1][name] += time.perf_counter() - t0
        return r
    return wrapped


def _fit_wrap(orig_fit, multiclass):
    def wrapped(self, *a, **kw):
        phases = collections.defaultdict(float)
        _ATTR["stack"].append(phases)
        t0 = time.perf_counter()
        try:
            return orig_fit(self, *a, **kw)
        finally:
            secs = time.perf_counter() - t0
            _ATTR["stack"].pop()
            if _ATTR["fits"] is not None:
                if multiclass:
                    label = "multiclass"
                    n_trees = sum(len(r) if isinstance(r, list) else 1
                                  for r in self.trees_)
                else:
                    # Labels mirror _fit_booster's config: the cross-augmented
                    # refit carries cross_pairs; the binary base fit has
                    # linear_leaves=True (the auto default), so roles resolve
                    # from label + position during reporting.
                    label = ("cross" if self.cross_pairs else
                             "linear" if self.linear_leaves else "const")
                    n_trees = len(self.trees_)
                _ATTR["fits"].append({
                    "label": label, "secs": secs, "n_trees": n_trees,
                    "rounds": len(self.valid_history_),
                    "valid_history": [float(v) for v in self.valid_history_],
                    "phases": dict(phases),
                })
    return wrapped


def _install_attribution_patches():
    """Wrap booster fits and phase functions with wall-clock recorders.
    Class-attribute patches reach every call site; module-global patches
    (build_oblivious_tree, _linear_predict) cover the booster's lookups."""
    import chimeraboost.booster as bmod
    import chimeraboost.tree as tmod
    import chimeraboost.preprocessing as pmod
    import chimeraboost.target_encoding as temod
    import chimeraboost.binning as bnmod

    bmod.GradientBoosting.fit = _fit_wrap(bmod.GradientBoosting.fit, False)
    bmod.MulticlassBoosting.fit = _fit_wrap(bmod.MulticlassBoosting.fit, True)

    bmod.build_oblivious_tree = _phase_wrap("grow", bmod.build_oblivious_tree)
    bmod._BaseBooster._refine_leaf_values = _phase_wrap(
        "leaf_refine", bmod._BaseBooster._refine_leaf_values)
    bmod._linear_predict = _phase_wrap("linear_update", bmod._linear_predict)
    tmod.ObliviousTree.predict = _phase_wrap(
        "val_predict", tmod.ObliviousTree.predict)
    pmod.FeaturePreprocessor.fit_transform = _phase_wrap(
        "prep_fit", pmod.FeaturePreprocessor.fit_transform)
    pmod.FeaturePreprocessor.transform = _phase_wrap(
        "val_transform", pmod.FeaturePreprocessor.transform)
    # Nested INSIDE prep_fit -- report as sub-items, never added to the total.
    temod.OrderedTargetEncoder.fit_transform = _phase_wrap(
        "ts_encode", temod.OrderedTargetEncoder.fit_transform)
    bnmod.Binner.fit_transform = _phase_wrap(
        "bin_fit", bnmod.Binner.fit_transform)


def _attr_roles(task, fits):
    """Map the recorded booster fits of one estimator fit to semantic roles.
    Regression order: const [, linear][, cross]; binary: base[, cross];
    multiclass: multiclass. Returns {role: record}."""
    roles = {}
    if task == "multiclass" or (fits and fits[0]["label"] == "multiclass"):
        roles["multiclass"] = fits[0]
        return roles
    rest = list(fits)
    if task == "regression":
        roles["const"] = rest.pop(0)
        if rest and rest[0]["label"] == "linear":
            roles["linear"] = rest.pop(0)
    else:
        roles["base"] = rest.pop(0)
    if rest and rest[0]["label"] == "cross":
        roles["cross"] = rest.pop(0)
    return roles


def _race_events(task, roles, n_estimators=2000):
    """Reconstruct the selection decisions of one estimator fit as
    (name, incumbent_history, challenger_history) tuples, in decision order.
    Winner rule mirrors sklearn_api: challenger wins on strictly lower best
    validation loss."""
    events = []
    if "linear" in roles:
        events.append(("ll", roles["const"]["valid_history"],
                       roles["linear"]["valid_history"]))
    if "cross" in roles:
        if task == "regression":
            inc = (roles["linear"] if "linear" in roles
                   and min(roles["linear"]["valid_history"])
                   < min(roles["const"]["valid_history"])
                   else roles.get("const") or roles["base"])
        else:
            inc = roles["base"]
        events.append(("cross", inc["valid_history"],
                       roles["cross"]["valid_history"]))
    return events


def _race_agree(inc, cha, k):
    """Would racing both variants to k rounds pick the same winner as the
    full fits did? Histories are ES-truncated already, so a variant that
    stopped before k contributes its final best. Returns (agree, regret_pct):
    regret is the full-run best-val gap conceded when the race mispicks."""
    full_cha_wins = min(cha) < min(inc)
    race_cha_wins = min(cha[:k]) < min(inc[:k])
    agree = full_cha_wins == race_cha_wins
    if agree:
        return True, 0.0
    best_full = min(min(inc), min(cha))
    best_pick = min(cha) if race_cha_wins else min(inc)
    denom = abs(best_full) if best_full != 0 else 1.0
    return False, 100.0 * (best_pick - best_full) / denom


def _pct(x, tot):
    return f"{100.0 * x / tot:5.1f}" if tot > 0 else "  0.0"


def run_attribution(args):
    import numpy as np
    import run_benchmarks as rb
    from sklearn.model_selection import train_test_split

    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    keys = args.datasets or ATTR_DATASETS

    print("Warmup (compiling numba kernels)...")
    from chimeraboost.warmup import warmup
    warmup()
    _install_attribution_patches()

    results = []
    for key in keys:
        print(f"Loading {key}...")
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        for seed in range(args.seeds):
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=strat)
            Est = (ChimeraBoostRegressor if task == "regression"
                   else ChimeraBoostClassifier)
            _ATTR["fits"] = []
            t0 = time.perf_counter()
            m = Est(random_state=0).fit(Xtr, ytr, cat_features=cat)
            total = time.perf_counter() - t0
            fits, _ATTR["fits"] = _ATTR["fits"], None
            rec = {
                "dataset": key, "task": task, "seed": seed,
                "n_train": int(Xtr.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "n_cats": len(cat) if cat else 0,
                "total_s": total,
                # bool() strips numpy scalars (cf_selected is an np.bool_).
                "ll_selected": (None if m.linear_leaves_selected_ is None
                                else bool(m.linear_leaves_selected_))
                               if hasattr(m, "linear_leaves_selected_") else None,
                "cf_selected": (None if m.cross_features_selected_ is None
                                else bool(m.cross_features_selected_)),
                "fits": fits,
            }
            results.append(rec)
            print(f"  seed {seed}: {total:.1f}s, {len(fits)} booster fits, "
                  f"ll_selected={rec['ll_selected']} "
                  f"cf_selected={rec['cf_selected']}")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results"), exist_ok=True)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", args.out)
    with open(base + ".json", "w") as f:
        json.dump(results, f)
    report = attr_report(results)
    with open(base + ".md", "w", newline="\n") as f:
        f.write(report)
    print(report)
    print(f"Saved {base}.json and {base}.md")


def attr_report(results):
    """Render the three step-0 tables from the recorded results."""
    by_ds = collections.defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)

    lines = ["# Fit attribution (PARETO_PLAN.md Track 1 step 0)", ""]

    # ---- Table 1: variant split ------------------------------------------
    lines += ["## Where the fit time goes: selection fits",
              "",
              "| dataset | task | n_train | feats(cats) | fit_s | const/base_s"
              " | linear_s | cross_s | ms/tree (base) |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for ds, recs in by_ds.items():
        r0 = recs[0]
        tot = sum(r["total_s"] for r in recs) / len(recs)
        role_s = collections.defaultdict(float)
        base_ms = []
        for r in recs:
            roles = _attr_roles(r["task"], r["fits"])
            for role, rec in roles.items():
                role_s[role] += rec["secs"] / len(recs)
            base = roles.get("const") or roles.get("base") \
                or roles.get("multiclass")
            base_ms.append(1000 * base["secs"] / max(base["n_trees"], 1))
        b = role_s.get("const", 0) + role_s.get("base", 0) \
            + role_s.get("multiclass", 0)
        lines.append(
            f"| {ds} | {r0['task']} | {r0['n_train']} "
            f"| {r0['n_features']}({r0['n_cats']}) | {tot:.1f} "
            f"| {b:.1f} | {role_s.get('linear', 0):.1f} "
            f"| {role_s.get('cross', 0):.1f} "
            f"| {sum(base_ms)/len(base_ms):.2f} |")

    # ---- Table 2: phase split --------------------------------------------
    lines += ["", "## Where the fit time goes: phases (% of estimator fit)",
              "",
              "| dataset | grow | ts_enc | bin | prep_other | val_tf "
              "| val_pred | leaf_ref | lin_upd | other | outside |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for ds, recs in by_ds.items():
        tot = sum(r["total_s"] for r in recs)
        ph = collections.defaultdict(float)
        boost_s = 0.0
        for r in recs:
            boost_s += sum(f["secs"] for f in r["fits"])
            for f in r["fits"]:
                for k, v in f["phases"].items():
                    ph[k] += v
        prep_other = ph["prep_fit"] - ph["ts_encode"] - ph["bin_fit"]
        accounted = (ph["grow"] + ph["prep_fit"] + ph["val_transform"]
                     + ph["val_predict"] + ph["leaf_refine"]
                     + ph["linear_update"])
        lines.append(
            f"| {ds} | {_pct(ph['grow'], tot)} | {_pct(ph['ts_encode'], tot)} "
            f"| {_pct(ph['bin_fit'], tot)} | {_pct(prep_other, tot)} "
            f"| {_pct(ph['val_transform'], tot)} "
            f"| {_pct(ph['val_predict'], tot)} "
            f"| {_pct(ph['leaf_refine'], tot)} "
            f"| {_pct(ph['linear_update'], tot)} "
            f"| {_pct(boost_s - accounted, tot)} "
            f"| {_pct(tot - boost_s, tot)} |")

    # ---- Table 3: selection outcomes + race preview ----------------------
    lines += ["", "## Selection outcomes (per seed) and flip rates", "",
              "| dataset | ll_selected | cf_selected |", "|---|---|---|"]
    ll_n = ll_y = cf_n = cf_y = 0
    for ds, recs in by_ds.items():
        lls = [r["ll_selected"] for r in recs]
        cfs = [r["cf_selected"] for r in recs]
        for v in lls:
            if v is not None:
                ll_n += 1
                ll_y += bool(v)
        for v in cfs:
            if v is not None:
                cf_n += 1
                cf_y += bool(v)
        fmt = lambda vs: " ".join("-" if v is None else "YN"[not v] for v in vs)
        lines.append(f"| {ds} | {fmt(lls)} | {fmt(cfs)} |")
    lines += ["",
              f"linear_leaves selected {ll_y}/{ll_n}; "
              f"cross_features selected {cf_y}/{cf_n} "
              "(each Y = the extra fit changed the shipped model).", ""]

    lines += ["## Race preview: truncated selection vs full selection", "",
              "Agreement if variants were raced to k rounds and only the",
              "leader continued (regret = full-run best-val loss conceded",
              "on mispicks, % of the better variant's best val loss).", "",
              "| selection | k=50 | k=100 | k=200 | k=500 |",
              "|---|---|---|---|---|"]
    events = collections.defaultdict(list)
    for r in results:
        roles = _attr_roles(r["task"], r["fits"])
        for name, inc, cha in _race_events(r["task"], roles):
            events[name].append((inc, cha))
    for name, evs in events.items():
        cells = []
        for k in (50, 100, 200, 500):
            outs = [_race_agree(inc, cha, k) for inc, cha in evs]
            n_agree = sum(1 for a, _ in outs if a)
            regs = [g for a, g in outs if not a]
            cell = f"{n_agree}/{len(outs)}"
            if regs:
                cell += f" (regret {max(regs):.2f}%)"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult", choices=list(DATASETS))
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--top", type=int, default=25,
                    help="how many cProfile rows to print")
    ap.add_argument("--attribution", action="store_true",
                    help="run the PARETO_PLAN step-0 attribution suite "
                         "instead of the single-dataset cProfile mode")
    ap.add_argument("--seeds", type=int, default=3,
                    help="attribution mode: splits per dataset")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="attribution mode: run_benchmarks DATASETS keys "
                         "(default: the frozen 9-set panel)")
    ap.add_argument("--out", default="pareto-step0",
                    help="attribution mode: results/<out>.json|.md")
    args = ap.parse_args()

    if args.attribution:
        run_attribution(args)
        return

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
