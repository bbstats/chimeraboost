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


def _fill(pool_by_task, sat_pool, row_budget, n_cap, already, rng):
    """Greedy stratified fill honoring task mix, saturated share, row budget."""
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
        take = None
        if sats and n_sat / total < SAT_SHARE:
            take = sats.pop()
        else:
            deficit = {t: TASK_MIX[t] - counts.get(t, 0) / total for t in TASK_MIX}
            for t in sorted(deficit, key=deficit.get, reverse=True):
                if pools[t]:
                    take = pools[t].pop()
                    break
        if take is None:
            break
        if take in chosen:
            continue
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

    screen, screen_rows = _fill(by_task, sat_pool, budget_screen, 8000, [], rng)
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
          f"cats {sum(r['n_cat'] > 0 for r in screen)})", flush=True)
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
