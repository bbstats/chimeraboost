"""Entity-ID auto-route probe for issue #113 (campaign log I068).

Does routing one high-cardinality entity-ID column to a shrunk random
intercept beat the default categorical path? Three hc: regression sets,
one routed column each (the qualifying column with the most levels),
arms A (default) / B (route) / C (route + train level count), test RMSE.

    python benchmarks/probe_entity_route.py

Saves one timestamped JSON under benchmarks/results/ and prints the table
plus the pre-registered verdict line.
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb
from grouped_suite import GROUPED_HC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import ChimeraBoostRegressor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

DATASETS = ["hc:wine-reviews", "hc:colleges", "hc:employee_salaries"]
SEEDS = (0, 1, 2)
TRAIN_FRAC = 0.75

# Qualifying rule (on the training rows): a categorical column with at least
# this many distinct levels and at most this median rows per level.
MIN_LEVELS = 1000
MAX_MEDIAN_ROWS = 5


def _load(key):
    """Load one hc: set exactly the way grouped_suite does (same registry,
    same parquet, same _prepare_frame shaping, same row cap)."""
    import pandas as pd

    name = key[len("hc:"):]
    cap = GROUPED_HC[key][1]
    spec = rb.HC_DATASETS[name]
    frame = pd.read_parquet(rb._public_parquet_path(spec["data_id"]))
    return rb._prepare_frame(frame, spec["target"], spec, None, cap)


def _train_test_split(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(TRAIN_FRAC * n))
    return idx[:cut], idx[cut:]


def _qualifying(X_df, train_idx):
    """Every categorical column meeting the rule on the training rows.

    Sorted by level count, descending, so [0] is the routed column.
    """
    out = []
    for col in X_df.columns:
        if not rb._is_categorical_dtype(X_df[col].dtype):
            continue
        vc = X_df[col].iloc[train_idx].value_counts(dropna=False)
        # Categorical-dtype columns list unobserved categories with count 0;
        # only levels present in the training rows count toward the rule.
        vc = vc[vc > 0]
        n_levels = int(len(vc))
        med = float(np.median(vc.to_numpy())) if n_levels else 0.0
        if n_levels >= MIN_LEVELS and med <= MAX_MEDIAN_ROWS:
            out.append({"column": col, "n_levels": n_levels,
                        "median_rows": med})
    out.sort(key=lambda d: d["n_levels"], reverse=True)
    return out


def _alias_drop(X_df, ref):
    """Drop columns in one-to-one correspondence with `ref`.

    The grouped_suite loader's alias fix, applied against the routed column
    so arms B/C cannot see the routed levels through an alias.
    """
    nunique_g = X_df[ref].nunique(dropna=False)
    dropped = [c for c in X_df.columns
               if c != ref
               and X_df[c].nunique(dropna=False) == nunique_g
               and len(X_df[[ref, c]].drop_duplicates()) == nunique_g]
    if dropped:
        X_df = X_df.drop(columns=dropped)
    return X_df, dropped


def _level_key(v):
    """Hashable level key; missing values share one sentinel level."""
    try:
        if v is None or v != v:
            return "__nan__"
    except Exception:
        return "__nan__"
    try:
        hash(v)
    except TypeError:
        return repr(v)
    return v


def _fit_predict(make, Xtr, ytr, Xte, cat, gtr=None, gte=None):
    m = make()
    t = time.time()
    if gtr is None:
        m.fit(Xtr, ytr, cat_features=cat)
    else:
        m.fit(Xtr, ytr, cat_features=cat, groups=gtr)
    fit_s = time.time() - t
    if gte is None:
        p = m.predict(Xte)
    else:
        p = m.predict(Xte, groups=gte)
    return p, fit_s, m.best_iteration_


def _rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def run_one(key, seed):
    """Run all three arms for one (dataset, seed). Returns (routing, rows).

    `routing` describes the qualifying/routed columns (or the skip reason);
    `rows` is one record per arm (empty when skipped).
    """
    X_df, y = _load(key)
    n = len(y)
    train_idx, test_idx = _train_test_split(n, seed)
    qual = _qualifying(X_df, train_idx)
    routing = {"dataset": key, "seed": seed, "n_train": int(len(train_idx)),
               "n_test": int(len(test_idx)), "qualifying": qual,
               "routed": qual[0]["column"] if qual else None,
               "alias_dropped": [], "skipped": None}
    if not qual:
        routing["skipped"] = (
            f"no categorical column with >={MIN_LEVELS} levels and median "
            f"rows/level <={MAX_MEDIAN_ROWS} on the {len(train_idx)} "
            "training rows")
        return routing, []

    routed = routing["routed"]
    X_df, dropped = _alias_drop(X_df, routed)
    routing["alias_dropped"] = dropped
    groups = X_df[routed].to_numpy()
    gtr, gte = groups[train_idx], groups[test_idx]

    # Arm A: X as loaded, the routed column a categorical.
    Xa, ya, cata, _ = rb._frame_to_dataset(X_df, y, "auto", "regression")
    pa, fa, ba = _fit_predict(
        lambda: ChimeraBoostRegressor(random_state=0),
        Xa[train_idx], ya[train_idx], Xa[test_idx], cata)

    # Arm B: routed column dropped, random intercept on its labels.
    Xb_df = X_df.drop(columns=[routed])
    Xb, yb, catb, _ = rb._frame_to_dataset(Xb_df, y, "auto", "regression")
    pb, fb, bb = _fit_predict(
        lambda: ChimeraBoostRegressor(random_state=0, random_effects=True),
        Xb[train_idx], yb[train_idx], Xb[test_idx], catb, gtr, gte)

    # Arm C: B plus the train level count (0 for unseen levels).
    counts = Counter(_level_key(v) for v in groups[train_idx])
    count_col = np.array([counts.get(_level_key(v), 0) for v in groups],
                         dtype=np.float64)
    cname = f"{routed}__train_count"
    while cname in Xb_df.columns:
        cname += "_"
    Xc_df = Xb_df.copy()
    Xc_df[cname] = count_col
    Xc, yc, catc, _ = rb._frame_to_dataset(Xc_df, y, "auto", "regression")
    pc, fc, bc = _fit_predict(
        lambda: ChimeraBoostRegressor(random_state=0, random_effects=True),
        Xc[train_idx], yc[train_idx], Xc[test_idx], catc, gtr, gte)

    yte = ya[test_idx]
    rows = [
        {"dataset": key, "seed": seed, "arm": "A-default",
         "rmse": _rmse(yte, pa), "fit_s": fa, "best_iter": int(ba)},
        {"dataset": key, "seed": seed, "arm": "B-route",
         "rmse": _rmse(yte, pb), "fit_s": fb, "best_iter": int(bb)},
        {"dataset": key, "seed": seed, "arm": "C-route-count",
         "rmse": _rmse(yte, pc), "fit_s": fc, "best_iter": int(bc)},
    ]
    return routing, rows


def _verdict(rows):
    """Pre-registered verdict: per route arm, sets beaten vs A (seed-mean
    RMSE) and the median % gap (positive = route better)."""
    sets = sorted({r["dataset"] for r in rows})
    mean = {}
    for ds in sets:
        for arm in ("A-default", "B-route", "C-route-count"):
            vals = [r["rmse"] for r in rows
                    if r["dataset"] == ds and r["arm"] == arm]
            mean[(ds, arm)] = float(np.mean(vals))
    out = {}
    for arm in ("B-route", "C-route-count"):
        gaps = []
        wins = 0
        for ds in sets:
            a, v = mean[(ds, "A-default")], mean[(ds, arm)]
            gaps.append((a - v) / a * 100.0 if a else 0.0)
            if v < a:
                wins += 1
        out[arm] = {"wins": wins, "n_sets": len(sets),
                    "median_gap_pct": float(np.median(gaps)), "gaps": gaps,
                    "sets": sets}
    return mean, out


def _print_table(rows, mean):
    print()
    head = f"{'dataset':22s}{'seed':>5s}{'A rmse':>10s}{'B rmse':>10s}" \
        f"{'C rmse':>10s}{'A fit':>8s}{'B fit':>8s}{'C fit':>8s}"
    print(head)
    print("-" * len(head))
    for ds in sorted({r["dataset"] for r in rows}):
        seeds = sorted({r["seed"] for r in rows if r["dataset"] == ds})
        for seed in seeds:
            got = {r["arm"]: r for r in rows
                   if r["dataset"] == ds and r["seed"] == seed}
            print(f"{ds:22s}{seed:5d}"
                  f"{got['A-default']['rmse']:10.4f}"
                  f"{got['B-route']['rmse']:10.4f}"
                  f"{got['C-route-count']['rmse']:10.4f}"
                  f"{got['A-default']['fit_s']:8.1f}"
                  f"{got['B-route']['fit_s']:8.1f}"
                  f"{got['C-route-count']['fit_s']:8.1f}")
        print(f"{ds + ' mean':22s}{'':>5s}"
              f"{mean[(ds, 'A-default')]:10.4f}"
              f"{mean[(ds, 'B-route')]:10.4f}"
              f"{mean[(ds, 'C-route-count')]:10.4f}")


def _print_verdict(verdict):
    print()
    keep = False
    for arm in ("B-route", "C-route-count"):
        v = verdict[arm]
        print(f"{arm}: beats A on {v['wins']}/{v['n_sets']} sets, "
              f"median gap {v['median_gap_pct']:+.2f}% "
              f"(positive = {arm} better)")
        if v["wins"] >= 2 and v["median_gap_pct"] > 0:
            keep = True
    print("VERDICT: "
          + ("KEEP the auto-route for a fuller gate"
             if keep else
             "KILL the auto-route (no route arm won >=2 sets with "
             "median gap > 0)"))
    return keep


def main():
    print(f"entity-route probe: {len(DATASETS)} datasets, "
          f"seeds {list(SEEDS)}", flush=True)
    routings, rows = [], []
    for key in DATASETS:
        for seed in SEEDS:
            t0 = time.time()
            routing, recs = run_one(key, seed)
            routings.append(routing)
            rows.extend(recs)
            if routing["skipped"] is not None:
                print(f"  {key} seed {seed}: SKIPPED "
                      f"({routing['skipped']})", flush=True)
            else:
                q = ", ".join(f"{d['column']} ({d['n_levels']} levels, "
                              f"med {d['median_rows']:.1f})"
                              for d in routing["qualifying"])
                print(f"  {key} seed {seed}: routed "
                      f"{routing['routed']}; qualifying: {q}", flush=True)
                if routing["alias_dropped"]:
                    print(f"    alias fix dropped "
                          f"{routing['alias_dropped']}", flush=True)
                for r in recs:
                    print(f"    {r['arm']:13s} rmse={r['rmse']:.4f} "
                          f"fit={r['fit_s']:.1f}s iter={r['best_iter']}",
                          flush=True)
            print(f"  [{key} seed {seed}: {time.time() - t0:.1f}s total]",
                  flush=True)

    if not rows:
        print("\nno dataset produced a routed column; nothing to judge.")
        return 1

    mean, verdict = _verdict(rows)
    _print_table(rows, mean)
    _print_verdict(verdict)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS_DIR, f"probe-entity-route-{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "config": {"datasets": DATASETS, "seeds": list(SEEDS),
                       "train_frac": TRAIN_FRAC, "min_levels": MIN_LEVELS,
                       "max_median_rows": MAX_MEDIAN_ROWS,
                       "timing": "fit_only", "suite": "probe-entity-route"},
            "provenance": rb._provenance(sys.argv, {}),
            "routing": routings,
            "records": rows,
            "verdict": {k: {kk: vv for kk, vv in v.items() if kk != "sets"}
                        for k, v in verdict.items()},
        }, fh, indent=1)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
