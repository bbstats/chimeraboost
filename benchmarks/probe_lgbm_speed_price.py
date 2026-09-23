"""Probe: price the LightGBM-vs-default fit-time gap in three parts.

On the Grinsztajn suite LightGBM fits about 5x faster than our default
(median over datasets). This probe prices that gap as:
  (1) round count (trees grown per fit),
  (2) single-thread cost per round,
  (3) thread scaling (speedup from more threads).

Part A (no fits): from the saved run's records, per base `gr:` key, our
ChimeraBoost vs LightGBM mean fit time and mean best_iter over seeds, the
fit ratio, the round ratio, and the per-round ratio, sorted by absolute
excess seconds.

Part B (fits): for the top 15 keys by excess seconds, seed 0, the
harness's own split, fit each model at thread counts {1, 2, 4, 8}, 3
repeats each, median kept. One at a time, nothing in parallel.

Output: benchmarks/results/probe-lgbm-speed-price.jsonl (one line per
timed fit, resumable) and tables in the log. `--table-only` prints the
tables without fitting. `--smoke` runs Part A fully and Part B for one
key at t in {1, 2} with one repeat.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb
from probe_catboost_split_score import _load_split

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-lgbm-speed-price.jsonl")
SAVED_RUN = os.path.join(HERE, "results", "20260922-144219.json")

SEED = 0
THREADS = (1, 2, 4, 8)
REPEATS = 3
TOP_N = 15
OURS = "ChimeraBoost"
LGBM = "LightGBM"

# Copy of CHIMERA_CFG from probe_catboost_split_score.py: the dict main()
# builds when no --chimera-* flag is given (argparse defaults).
CHIMERA_CFG = dict(lr=None, ordered_boosting=None, depth=6, subsample=1.0,
                   colsample=None, mcw=None, cat_combinations=False,
                   cat_count_features=False, cat_smoothing=None,
                   leaf_estimation_iterations=None, linear_leaves=False,
                   linear_lambda=1.0, cross_features=False,
                   selection_rounds=None, quantize=False, refit_full=False)


def _read_jsonl():
    rows = []
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _append(rec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def part_a():
    """Per-key ours-vs-LightGBM means from the saved run, sorted by excess."""
    with open(SAVED_RUN, "r", encoding="utf-8") as f:
        saved = json.load(f)
    meta = saved["datasets"]
    agg = {}
    for r in saved["records"]:
        key = r["dataset"]
        if not key.startswith("gr:") or rb.VARIANT_SEP in key:
            continue
        if r["model"] not in (OURS, LGBM):
            continue
        agg.setdefault((key, r["model"]), []).append(r)
    rows = []
    skipped = []
    for key in sorted({k for k, _ in agg}):
        a = agg.get((key, OURS), [])
        b = agg.get((key, LGBM), [])
        if not a or not b:
            skipped.append(key)
            continue
        fit_o = float(np.mean([r["fit_time"] for r in a]))
        fit_l = float(np.mean([r["fit_time"] for r in b]))
        it_o = float(np.mean([r["best_iter"] for r in a]))
        it_l = float(np.mean([r["best_iter"] for r in b]))
        rows.append({
            "dataset": key,
            "n_train": meta.get(key, {}).get("n_train"),
            "n_features": meta.get(key, {}).get("n_features"),
            "fit_ours": fit_o, "fit_lgbm": fit_l,
            "excess": fit_o - fit_l,
            "fit_ratio": fit_o / fit_l if fit_l else float("nan"),
            "it_ours": it_o, "it_lgbm": it_l,
            "round_ratio": it_o / it_l if it_l else float("nan"),
            "perround_ratio": ((fit_o / it_o) / (fit_l / it_l)
                               if it_o and it_l and fit_l else float("nan")),
        })
    rows.sort(key=lambda r: r["excess"], reverse=True)
    return rows, skipped


def print_part_a(rows, skipped):
    print("=" * 124)
    print("PART A — saved run, base gr: keys: mean fit / mean rounds over seeds.")
    print("per-round = (fit/rounds ours) / (fit/rounds LightGBM); our fit also")
    print("contains the validation races and the full-data refit, so per-round")
    print("is per round of the WHOLE fit, not per boosting round.")
    print("=" * 124)
    print(f"{'dataset':32s}{'n_tr':>8s}{'p':>5s}{'ours_s':>9s}{'lgbm_s':>9s}"
          f"{'excess':>9s}{'fit_x':>7s}{'ours_it':>8s}{'lgbm_it':>8s}"
          f"{'rnd_x':>7s}{'prnd_x':>7s}")
    for r in rows:
        ds = r["dataset"].replace("gr:", "")[:32]
        print(f"{ds:32s}{r['n_train']:8d}{r['n_features']:5d}"
              f"{r['fit_ours']:9.1f}{r['fit_lgbm']:9.1f}{r['excess']:9.1f}"
              f"{r['fit_ratio']:7.2f}{r['it_ours']:8.0f}{r['it_lgbm']:8.0f}"
              f"{r['round_ratio']:7.2f}{r['perround_ratio']:7.2f}")
    total = sum(r["excess"] for r in rows)
    top = sum(r["excess"] for r in rows[:TOP_N])
    med = float(np.median([r["fit_ratio"] for r in rows])) if rows else float("nan")
    print(f"keys={len(rows)} skipped={len(skipped)} "
          f"total_excess={total:.1f}s top{TOP_N}_excess={top:.1f}s "
          f"top{TOP_N}_share={100 * top / total:.1f}% median_fit_ratio={med:.2f}x")
    if skipped:
        print(f"skipped (missing model in saved run): {skipped}")


def _med(rows, model, threads):
    v = [r["fit_time"] for r in rows
         if r["model"] == model and r["threads"] == threads]
    return float(np.median(v)) if v else None


def _med_rounds(rows, model, threads):
    v = [r["best_iter"] for r in rows
         if r["model"] == model and r["threads"] == threads]
    return float(np.median(v)) if v else None


def _key_stats(key, rows, threads):
    """Median fits, speedups, ratios and scaling gaps for one key."""
    st = {"dataset": key, "fit": {}, "rounds": {}, "speedup": {}}
    for model in (OURS, LGBM):
        for t in threads:
            st["fit"][(model, t)] = _med(rows, model, t)
            st["rounds"][(model, t)] = _med_rounds(rows, model, t)
        f1 = st["fit"][(model, threads[0])]
        for t in threads:
            ft = st["fit"][(model, t)]
            st["speedup"][(model, t)] = (f1 / ft if f1 and ft else None)
    st["ratio"] = {}
    for t in threads:
        fo = st["fit"][(OURS, t)]
        fl = st["fit"][(LGBM, t)]
        st["ratio"][t] = (fo / fl if fo and fl else None)
    st["gap"] = {}
    for t in threads[1:]:
        so = st["speedup"][(OURS, t)]
        sl = st["speedup"][(LGBM, t)]
        st["gap"][t] = (sl / so if so and sl else None)
    ro = st["rounds"][(OURS, threads[0])]
    rl = st["rounds"][(LGBM, threads[0])]
    fo1 = st["fit"][(OURS, threads[0])]
    fl1 = st["fit"][(LGBM, threads[0])]
    st["perround_t1"] = ((fo1 / ro) / (fl1 / rl)
                         if fo1 and fl1 and ro and rl else None)
    return st


def _f(v, width=9, prec=1):
    return f"{v:{width}.{prec}f}" if v is not None else " " * width


def _x(v, width=7):
    return f"{v:{width}.2f}" if v is not None else " " * width


def _n(v, width=7):
    return f"{v:{width}.0f}" if v is not None else " " * width


def print_part_b(keys, threads, repeats=REPEATS):
    rows = [r for r in _read_jsonl() if r["dataset"] in keys]
    have = {(r["dataset"], r["model"], r["threads"]) for r in rows}
    print("=" * 124)
    print(f"PART B — seed {SEED}, median of {repeats} repeats per (key, model, t); "
          f"speedup(t) = fit(1)/fit(t); gap(t) = lgbm_su(t)/ours_su(t).")
    print("=" * 124)
    stats = []
    for key in keys:
        kr = [r for r in rows if r["dataset"] == key]
        st = _key_stats(key, kr, threads)
        stats.append(st)
        n_rep = {(m, t): sum(1 for r in kr if r["model"] == m and r["threads"] == t)
                 for m in (OURS, LGBM) for t in threads}
        tag = "" if all(n == repeats for n in n_rep.values()) else f"  INCOMPLETE {n_rep}"
        print(f"--- {key}{tag}")
        hdr = f"{'model':12s}" + "".join(f"{'t=' + str(t):>9s}" for t in threads)
        hdr += " |" + "".join(f"{'su(' + str(t) + ')':>7s}" for t in threads[1:])
        hdr += " |" + "".join(f"{'it' + str(t):>7s}" for t in threads)
        print(hdr)
        for m, lab in ((OURS, "ours"), (LGBM, "lgbm")):
            line = (f"{lab:12s}"
                    + "".join(_f(st["fit"][(m, t)]) for t in threads)
                    + " |"
                    + "".join(_x(st["speedup"][(m, t)]) for t in threads[1:])
                    + " |"
                    + "".join(_n(st["rounds"][(m, t)]) for t in threads))
            print(line)
        ratio_line = "ratio_o/l    " + "".join(_x(st["ratio"][t], 9) for t in threads)
        if st["perround_t1"] is not None:
            ratio_line += f"   perround@t1={st['perround_t1']:.2f}"
        print(ratio_line)
        print("scaling_gap  " + "".join(_x(st["gap"][t], 9) for t in threads[1:]))
    _print_part_b_medians(stats, threads, keys, have)
    return stats


def _print_part_b_medians(stats, threads, keys, have):
    n_full = sum(1 for k in keys
                 if all((k, m, t) in have for m in (OURS, LGBM) for t in threads))
    print(f"--- MEDIANS over {n_full}/{len(keys)} keys with all (model, t) present ---")
    for t in threads:
        v = [s["ratio"][t] for s in stats if s["ratio"][t] is not None]
        print(f"  median ratio ours/lgbm @t={t}: "
              f"{float(np.median(v)):.2f}x (n={len(v)})" if v
              else f"  median ratio ours/lgbm @t={t}: n/a")
    for t in threads[1:]:
        v = [s["gap"][t] for s in stats if s["gap"][t] is not None]
        print(f"  median scaling gap @t={t}: "
              f"{float(np.median(v)):.2f}x (n={len(v)})" if v
              else f"  median scaling gap @t={t}: n/a")
    for m, lab in ((OURS, "ours"), (LGBM, "lgbm")):
        for t in threads[1:]:
            v = [s["speedup"][(m, t)] for s in stats
                 if s["speedup"][(m, t)] is not None]
            print(f"  median speedup {lab} @t={t}: "
                  f"{float(np.median(v)):.2f}x (n={len(v)})" if v
                  else f"  median speedup {lab} @t={t}: n/a")
    v = [s["perround_t1"] for s in stats if s["perround_t1"] is not None]
    print(f"  median single-thread per-round ratio: "
          f"{float(np.median(v)):.2f}x (n={len(v)})" if v
          else "  median single-thread per-round ratio: n/a")


def _run_one(model, task, Xtr, ytr, Xte, yte, cat, t):
    if model == OURS:
        out = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, t, **CHIMERA_CFG)
    else:
        out = rb.RUNNERS[LGBM](task, Xtr, ytr, Xte, yte, cat, t)
    _, fit_s, _, best = out
    return fit_s, best


def _warmup(key):
    """One untimed fit of ours so numba compile never lands in a timed fit."""
    print(f"warm-up (untimed, ours, t=1): {key}", flush=True)
    Xtr, Xte, ytr, yte, cat, task = _load_split(key, SEED)
    _run_one(OURS, task, Xtr, ytr, Xte, yte, cat, 1)
    print("warm-up done", flush=True)


def run_part_b(keys, threads, repeats):
    done = {(r["dataset"], r["model"], r["threads"], r["rep"])
            for r in _read_jsonl()}
    todo = [(k, m, t, rep) for k in keys for m in (OURS, LGBM)
            for t in threads for rep in range(repeats)
            if (k, m, t, rep) not in done]
    if not todo:
        print("Part B: all rows already present, nothing to fit.")
        return
    _warmup(keys[-1])  # smallest excess first: compiles kernels on the cheap fit
    for i, (key, model, t, rep) in enumerate(todo):
        Xtr, Xte, ytr, yte, cat, task = _load_split(key, SEED)
        fit_s, best = _run_one(model, task, Xtr, ytr, Xte, yte, cat, t)
        _append({"dataset": key, "model": model, "seed": SEED, "threads": t,
                 "rep": rep, "fit_time": fit_s, "best_iter": int(best)})
        print(f"  [{i + 1}/{len(todo)}] {key} {model} t={t} rep={rep}: "
              f"{fit_s:.1f}s it={best}", flush=True)


def main():
    args = sys.argv[1:]
    table_only = "--table-only" in args
    smoke = "--smoke" in args or os.environ.get("PROBE_SMOKE") == "1"
    rb._add_grinsztajn_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    rows_a, skipped = part_a()
    top_keys = [r["dataset"] for r in rows_a[:TOP_N]]
    print_part_a(rows_a, skipped)
    print(f"top{TOP_N} keys for Part B:")
    for k in top_keys:
        print(f"  {k}")
    if smoke:
        threads, repeats, keys_b = (1, 2), 1, top_keys[:1]
    else:
        threads, repeats, keys_b = THREADS, REPEATS, top_keys
    if table_only:
        print_part_b(keys_b, threads, repeats)
        return
    try:
        run_part_b(keys_b, threads, repeats)
    except KeyboardInterrupt:
        print("\ninterrupted; partial table follows (JSONL is resumable)")
    print_part_b(keys_b, threads, repeats)


if __name__ == "__main__":
    main()
