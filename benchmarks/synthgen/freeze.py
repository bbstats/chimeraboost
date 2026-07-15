"""Scan candidate ids, filter, and select the frozen suites (dev script).

Prints (never writes): scan statistics, the SUITES literal to paste into
suites.py, and golden hashes to paste into tests/golden_synthgen.json.

Usage:
  python benchmarks/synthgen/freeze.py --count 400 --scan-only     # probe
  python benchmarks/synthgen/freeze.py --count 400                 # full freeze
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synthgen                       # noqa: E402
from synthgen import filters          # noqa: E402

TASK_MIX = {"regression": 0.35, "binary": 0.40, "multiclass": 0.25}
SAT_SHARE = 0.12
# workstream A (v2): the screen's n-mix follows the corpus shape within
# [600, 8000] instead of greedy small-n packing -- v1's small-n skew is where
# the depth4 disagree lived. Cap the n<2000 share.
SMALL_N = 2000
SMALL_SHARE_CAP = 0.35
MIN_CAT_CANARIES = 3


def scan(start, count):
    records = []
    t0 = time.time()
    for did in range(start, start + count):
        key = synthgen.key_for(did)
        try:
            t1 = time.time()
            X, y, cat, task, meta = synthgen.build_dataset(key)
            gen_s = time.time() - t1
        except Exception as exc:  # noqa: BLE001 - report, don't die mid-scan
            records.append({"id": did, "error": repr(exc)})
            continue
        rec = {"id": did, "task": task, "n": meta["n"], "d": meta["d"],
               "n_cat": meta["n_cat"], "max_card": meta["max_cardinality"],
               "depth": meta["interaction_depth"], "saturated": meta["saturated"],
               "func": meta["func_dominant"], "gen_s": round(gen_s, 3),
               "bayes_brier": meta["bayes_brier"], "degenerate": meta["degenerate"]}
        ok, why = filters.degeneracy_ok(X, y, task)
        if ok and meta["degenerate"]:
            ok, why = False, "emit degenerate flag"
        if ok:
            ok, why = filters.tractable(meta)
        if ok and not meta["saturated"]:
            ok, detail = filters.learnable(X, y, cat, task)
            why = "" if ok else f"unlearnable {detail}"
            rec["learn"] = detail
        if ok and meta["saturated"]:
            # workstream C: canary status is EARNED (baseline verified at the
            # ceiling), never assumed from the construction
            t1 = time.time()
            can, detail = filters.at_ceiling(X, y, cat, task, meta)
            rec["canary"] = bool(can)
            rec.update(detail)
            rec["fit_s"] = round(time.time() - t1, 1)
        rec["accept"] = bool(ok)
        rec["why"] = why
        records.append(rec)
        if (did - start + 1) % 25 == 0:
            done = did - start + 1
            print(f"  scanned {done}/{count} ({time.time() - t0:.0f}s)", flush=True)
        synthgen.build_dataset.cache_clear()   # keep memory flat during scans
    return records


def report(records):
    errs = [r for r in records if "error" in r]
    ok = [r for r in records if r.get("accept")]
    rej = [r for r in records if "accept" in r and not r["accept"]]
    print(f"\nscan: {len(records)} ids -> {len(ok)} accepted, {len(rej)} rejected, "
          f"{len(errs)} errors", flush=True)
    if errs:
        for r in errs[:5]:
            print(f"  ERROR id {r['id']}: {r['error']}", flush=True)
    why = Counter(r["why"].split(" ")[0] for r in rej)
    print(f"  reject reasons: {dict(why)}", flush=True)
    for field in ("task", "depth", "func"):
        print(f"  {field}: {dict(Counter(str(r[field]) for r in ok))}", flush=True)
    n_arr = np.array([r["n"] for r in ok]) if ok else np.array([0])
    print(f"  n: median {np.median(n_arr):.0f} p90 {np.percentile(n_arr, 90):.0f}  "
          f"cats>0: {np.mean([r['n_cat'] > 0 for r in ok]):.2f}  "
          f"saturated: {np.mean([r['saturated'] for r in ok]):.2f}", flush=True)
    gen_s = np.array([r["gen_s"] for r in ok]) if ok else np.array([0])
    print(f"  gen time: median {np.median(gen_s):.2f}s max {gen_s.max():.2f}s", flush=True)
    sat_ok = [r for r in ok if r["saturated"]]
    n_can = sum(r.get("canary", False) for r in sat_ok)
    n_can_cat = sum(r.get("canary", False) and r["n_cat"] > 0 for r in sat_ok)
    print(f"  canaries: {n_can}/{len(sat_ok)} saturated verified at ceiling "
          f"({n_can_cat} cat-bearing)", flush=True)


def _pop(pool, need_large):
    for i in range(len(pool) - 1, -1, -1):
        if not need_large or pool[i]["n"] >= SMALL_N:
            return pool.pop(i)
    return None


def _fill(pool_by_task, sat_pool, row_budget, n_cap, already, rng, small_cap=None):
    """Stratified fill honoring task mix, saturated share, row budget and
    (screen tier only) the small-n share cap -- a hard cap: fill stops early
    rather than pack more n<2000 sets once it binds."""
    chosen = list(already)
    rows = sum(r["n"] for r in chosen)
    pools = {t: [r for r in v if r["n"] <= n_cap and r not in chosen]
             for t, v in pool_by_task.items()}
    sats = [r for r in sat_pool if r["n"] <= n_cap and r not in chosen]
    for p in pools.values():
        rng.shuffle(p)
    rng.shuffle(sats)
    while rows < row_budget:
        counts = Counter(r["task"] for r in chosen)
        total = max(1, len(chosen))
        n_sat = sum(r["saturated"] for r in chosen)
        n_small = sum(r["n"] < SMALL_N for r in chosen)
        need_large = small_cap is not None and n_small / total >= small_cap
        sources = []
        if sats and n_sat / total < SAT_SHARE:
            sources.append(sats)
        deficit = {t: TASK_MIX[t] - counts.get(t, 0) / total for t in TASK_MIX}
        sources += [pools[t] for t in sorted(deficit, key=deficit.get, reverse=True)]
        take = None
        for src in sources:
            take = _pop(src, need_large)
            if take is not None:
                break
        if take is None:
            break
        chosen.append(take)
        rows += take["n"]
        for t in pools:
            pools[t] = [r for r in pools[t] if r["id"] != take["id"]]
        sats = [r for r in sats if r["id"] != take["id"]]
    return chosen, rows


def select(records, budget_screen, budget_full):
    ok = [r for r in records if r.get("accept")]
    rng = np.random.default_rng(0)
    by_task = {t: sorted([r for r in ok if r["task"] == t and not r["saturated"]],
                         key=lambda r: r["id"]) for t in TASK_MIX}
    sat_pool = sorted([r for r in ok if r["saturated"]], key=lambda r: r["id"])

    # workstream C: the screen is seeded with cat-bearing VERIFIED canaries so
    # the catcombo canary slice can never be vacuously empty again
    cat_canaries = [r for r in sat_pool
                    if r.get("canary") and r["n_cat"] > 0 and r["n"] <= 8000]
    if len(cat_canaries) < MIN_CAT_CANARIES:
        print(f"\nFREEZE WARNING: only {len(cat_canaries)} cat-bearing verified "
              f"canaries in the pool (need >= {MIN_CAT_CANARIES}) -- raise "
              "--count or iterate workstream C before freezing", flush=True)
    seed = cat_canaries[:MIN_CAT_CANARIES]

    screen, screen_rows = _fill(by_task, sat_pool, budget_screen, 8000, seed,
                                rng, small_cap=SMALL_SHARE_CAP)
    full, full_rows = _fill(by_task, sat_pool, budget_full, 32000, screen, rng)

    smoke, want = [], {"regression": 2, "binary": 3, "multiclass": 1}
    for r in sorted(screen, key=lambda r: r["n"]):
        if want.get(r["task"], 0) > 0 and r["n"] <= 2500:
            smoke.append(r)
            want[r["task"]] -= 1
    if not any(r["n_cat"] > 0 for r in smoke):
        cats = [r for r in sorted(screen, key=lambda r: r["n"]) if r["n_cat"] > 0]
        if cats:
            smoke[-1] = cats[0]

    def ids(rs):
        return sorted(r["id"] for r in rs)

    print(f"\nscreen: {len(screen)} datasets, {screen_rows} rows "
          f"(tasks {dict(Counter(r['task'] for r in screen))}, "
          f"sat {sum(r['saturated'] for r in screen)}, "
          f"cats {sum(r['n_cat'] > 0 for r in screen)}, "
          f"n<{SMALL_N} share {np.mean([r['n'] < SMALL_N for r in screen]):.2f}, "
          f"cat-canaries {sum(r.get('canary', False) and r['n_cat'] > 0 for r in screen)})",
          flush=True)
    print(f"full:   {len(full)} datasets, {full_rows} rows "
          f"(tasks {dict(Counter(r['task'] for r in full))})", flush=True)
    est = 0.9 * 2.9 * 3 / 1000  # s/row: chimera s-per-1K x all-model factor x seeds
    print(f"rough all-model 3-seed serial estimate: screen "
          f"{screen_rows * est / 60:.0f} min, full {full_rows * est / 60:.0f} min "
          f"(/jobs for wall time)", flush=True)

    print("\n# ---- paste into suites.py ----", flush=True)
    print("SUITES = {", flush=True)
    for name, rs in (("smoke", smoke), ("screen", screen), ("full", full)):
        print(f"    {name!r}: {ids(rs)},", flush=True)
    print("}", flush=True)
    canary_ids = sorted(r["id"] for r in full if r.get("canary"))
    canary_lit = ("{" + ", ".join(map(str, canary_ids)) + "}") if canary_ids else "set()"
    print(f"CANARIES = {canary_lit}", flush=True)

    golden_ids = ids(smoke) + [r["id"] for r in screen
                               if r["n_cat"] > 0 and r not in smoke][:2]
    goldens = {synthgen.key_for(i): synthgen.hash_dataset(synthgen.key_for(i))
               for i in golden_ids}
    print("\n# ---- paste into tests/golden_synthgen.json ----", flush=True)
    print(json.dumps(goldens, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--row-budget-screen", type=int, default=400_000)
    ap.add_argument("--row-budget-full", type=int, default=1_600_000)
    args = ap.parse_args()

    print(f"scanning ids {args.start}..{args.start + args.count - 1} "
          f"(generator {synthgen.VERSION})", flush=True)
    records = scan(args.start, args.count)
    report(records)
    if not args.scan_only:
        select(records, args.row_budget_screen, args.row_budget_full)


if __name__ == "__main__":
    main()
