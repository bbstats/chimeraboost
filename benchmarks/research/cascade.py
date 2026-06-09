"""Cascade orchestration: idea -> T0 (paired curves) -> gate -> T1 -> gate ->
T2 (+ OpenML one-shot gate inside) -> PMLB holdout, with a pre-registered
hypothesis and a consolidated per-idea report.

Efficiency: paired same-split deltas (variance crusher), shared-baseline cache
(only the variant refits), and a sequential sign-test early-stop per tier. Fits
parallelize over datasets (ProcessPoolExecutor).

GUARDRAIL: never loads TabArena; OpenML is a one-shot gate (T1), not iterated.

CLI:
  python -m benchmarks.research.cascade --idea C1_onehot_low_card --tier T0 T1
  python benchmarks/research/cascade.py --idea linear_leaves --tier T0   # self-test
  python benchmarks/research/cascade.py --selftest
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import curves, datasets, ideas, runner, report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Sign test (paired). Counts datasets where the variant IMPROVES the primary
# (lower-is-better) metric, ties within eps excluded. Two-sided binomial p.
# ---------------------------------------------------------------------------
def sign_test(deltas, eps=0.0):
    """``deltas`` are (variant - baseline) on a lower-is-better metric, so a
    negative delta is an improvement. Returns wins/losses/ties + binomial p."""
    wins = sum(1 for d in deltas if d < -eps)     # variant better
    losses = sum(1 for d in deltas if d > eps)    # variant worse
    ties = len(deltas) - wins - losses
    n = wins + losses
    if n == 0:
        return dict(wins=wins, losses=losses, ties=ties, n=n, p=1.0)
    try:
        from scipy.stats import binomtest
        p = binomtest(min(wins, losses), n, 0.5,
                      alternative="two-sided").pvalue
    except Exception:                              # normal approx fallback
        z = abs(wins - losses) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2))
    return dict(wins=wins, losses=losses, ties=ties, n=n, p=float(p))


# ---------------------------------------------------------------------------
# Per-dataset workers (top-level so ProcessPoolExecutor can pickle them).
# ---------------------------------------------------------------------------
def eval_fast(dataset, params, seed, threads, max_rows):
    """Fast tier: baseline (cached) vs variant paired validation curves."""
    X, y, cat, task = datasets.load(dataset, max_rows=max_rows)
    sp = runner.three_way_split(X, y, task, seed)
    X_tr, y_tr, X_val, y_val, _Xte, _yte = sp
    base = runner.cached_baseline_curve(dataset, task, X_tr, y_tr, X_val, y_val,
                                        cat, seed, threads)
    var = runner.fast_curve(task, params, X_tr, y_tr, X_val, y_val, cat, seed,
                            threads)
    stat = curves.compare(base, var)
    stat.update(dataset=dataset, task=task, seed=seed)
    return stat


def eval_promo(dataset, params, seed, threads, max_rows):
    """Promotion tier: baseline (cached) vs variant true held-out test metric."""
    X, y, cat, task = datasets.load(dataset, max_rows=max_rows)
    sp = runner.three_way_split(X, y, task, seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = sp
    (bm, bt) = runner.cached_baseline_promotion(
        dataset, task, X_tr, y_tr, X_val, y_val, X_te, y_te, cat, seed, threads)
    vm, vt = runner.promotion_metrics(task, params, X_tr, y_tr, X_val, y_val,
                                      X_te, y_te, cat, seed, threads)
    delta = vm["primary"] - bm["primary"]
    rel = delta / (abs(bm["primary"]) + 1e-12)
    return dict(dataset=dataset, task=task, seed=seed,
                base=bm, variant=vm, base_trees=bt, variant_trees=vt,
                delta=delta, rel=rel)


def _map(fn, keys, params, seed, threads, max_rows, jobs):
    """Run ``fn`` over dataset keys, parallel when jobs != 1. Failures are logged
    and skipped (a flaky download shouldn't sink the whole tier)."""
    results = []
    if jobs == 1:
        for k in keys:
            try:
                results.append(fn(k, params, seed, threads, max_rows))
            except Exception as e:
                print(f"  ! {k}: {e}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=abs(jobs)) as ex:
        futs = {ex.submit(fn, k, params, seed, threads, max_rows): k
                for k in keys}
        for fut in futs:
            k = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  ! {k}: {e}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Tier runners + gates.
# ---------------------------------------------------------------------------
def run_fast_tier(idea_params, seed, threads, jobs, max_rows=datasets.T0_MAX_ROWS,
                  keys=None):
    keys = keys if keys is not None else datasets.tier_keys("T0")
    t0 = time.time()
    rows = _map(eval_fast, keys, idea_params, seed, threads, max_rows, jobs)
    favorable = [r for r in rows if r["best_val_delta"] < -1e-9]
    dominated = [r for r in rows if r["dominance"] < 0.1]
    verdict = dict(
        n=len(rows), favorable=len(favorable), dominated=len(dominated),
        mean_best_delta_pct=float(np.mean([r["best_val_delta_pct"]
                                           for r in rows])) if rows else 0.0,
        mean_dominance=float(np.mean([r["dominance"] for r in rows]))
        if rows else 0.0,
        seconds=time.time() - t0, rows=rows)
    # Gate T0->T1: favorable on >= ceil(0.6*n) AND not strongly dominated.
    need = math.ceil(0.6 * len(rows)) if rows else 1
    verdict["passed"] = (len(favorable) >= need
                         and verdict["mean_dominance"] > 0.1)
    verdict["need"] = need
    return verdict


def run_promo_tier(idea_params, seed, threads, jobs, keys, max_rows=None,
                   speed_tol=1.25):
    t0 = time.time()
    rows = _map(eval_promo, keys, idea_params, seed, threads, max_rows, jobs)
    deltas = [r["rel"] for r in rows]
    st = sign_test(deltas)
    # Speed regression guard: mean variant/baseline tree-count ratio.
    ratios = [r["variant_trees"] / max(1, r["base_trees"]) for r in rows]
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0
    verdict = dict(n=len(rows), sign_test=st,
                   mean_rel=float(np.mean(deltas)) if deltas else 0.0,
                   mean_tree_ratio=mean_ratio, seconds=time.time() - t0,
                   rows=rows)
    # Gate: more wins than losses, significant (p < 0.1), no large speed regress.
    verdict["passed"] = (st["wins"] > st["losses"] and st["p"] < 0.1
                         and mean_ratio < speed_tol)
    return verdict


# ---------------------------------------------------------------------------
# Full cascade for one idea.
# ---------------------------------------------------------------------------
def cascade(idea_name, tiers, seed, threads, jobs):
    spec = ideas.get(idea_name)
    if not spec["implemented"]:
        raise SystemExit(
            f"idea {idea_name!r} is not implemented yet (no library flag). "
            f"Implement its default-off flag first.")
    params = spec["params"]
    out = dict(idea=idea_name, hypothesis=spec["hypothesis"],
               category=spec["category"], params=params, tiers={})
    print(report.preregister(idea_name, spec), flush=True)

    # Post-fit ideas (e.g. G1 forest leaf refit) rewrite the model AFTER boosting,
    # so the per-round validation_history_ curve cannot see them -- the fast
    # (curve) tier would read a false flat. Skip T0 for them and judge at the
    # promotion tier, which fits the full model (refit included) and scores the
    # held-out test set.
    if spec.get("post_fit") and "T0" in tiers:
        print("\n[T0] SKIPPED -- post-fit idea is invisible to the validation "
              "curve; evaluating at the promotion tier instead.", flush=True)
        tiers = [t for t in tiers if t != "T0"]

    if "T0" in tiers:
        print("\n[T0] fast tier -- paired validation curves ...", flush=True)
        v = run_fast_tier(params, seed, threads, jobs)
        out["tiers"]["T0"] = v
        print(report.fast_tier(v), flush=True)
        if not v["passed"]:
            out["verdict"] = "KILL @ T0"
            print(report.final(out), flush=True)
            return out

    if "T1" in tiers:
        print("\n[T1] promotion -- OpenML categorical gate (paired sign test) ...",
              flush=True)
        keys = datasets.tier_keys("T1")
        v = run_promo_tier(params, seed, threads, jobs, keys)
        out["tiers"]["T1"] = v
        print(report.promo_tier("T1", v), flush=True)
        if not v["passed"]:
            out["verdict"] = "KILL @ T1"
            print(report.final(out), flush=True)
            return out

    if "T2" in tiers:
        print("\n[T2] large -- full sign test (Grinsztajn + OpenML) ...",
              flush=True)
        keys = datasets.tier_keys("T2")
        v = run_promo_tier(params, seed, threads, jobs, keys)
        out["tiers"]["T2"] = v
        print(report.promo_tier("T2", v), flush=True)
        if not v["passed"]:
            out["verdict"] = "KEEP AS OPT-IN @ T2"
            print(report.final(out), flush=True)
            return out

    if "HOLDOUT" in tiers:
        print("\n[HOLDOUT] PMLB holdout -- out-of-sample generalization ...",
              flush=True)
        keys = datasets.tier_keys("HOLDOUT")
        v = run_promo_tier(params, seed, threads, jobs, keys)
        out["tiers"]["HOLDOUT"] = v
        print(report.promo_tier("HOLDOUT", v), flush=True)

    out["verdict"] = "SHIP (review Pareto)" if out["tiers"] else "no tiers run"
    print(report.final(out), flush=True)
    return out


def selftest(seed, threads, jobs):
    """Reproduce two known truths on T0, proving the harness discriminates a real
    win from a no-op:

      * linear_leaves -- a clear POSITIVE where it is OFF by default. (NOTE: it is
        already the auto-default for *binary* classification, so it is correctly
        flat there; the off-by-default win shows on regression -- e.g. pol.) The
        anchor: at least one dataset improves strongly (min best-val delta well
        below 0) with high pointwise dominance.
      * patience300 -- a NO-OP on the fast tier (early stopping is disabled, so
        patience cannot move the curve): the trajectory is IDENTICAL to baseline.
    """
    print("=== ENGINE SELF-TEST (T0) ===", flush=True)
    pos = run_fast_tier(ideas.get("linear_leaves")["params"], seed, threads, jobs)
    print(report.fast_tier(pos, label="linear_leaves (expect a strong "
                           "off-by-default win, e.g. regression pol)"),
          flush=True)
    flat = run_fast_tier(ideas.get("patience300")["params"], seed, threads, jobs)
    print(report.fast_tier(flat, label="patience300 (expect FLAT / no-op)"),
          flush=True)
    min_delta = min((r["best_val_delta_pct"] for r in pos["rows"]), default=0.0)
    ok_pos = (pos["favorable"] >= 1 and min_delta < -0.01
              and pos["mean_dominance"] > 0.5)
    ok_flat = abs(flat["mean_best_delta_pct"]) < 1e-6
    print(f"\nSELF-TEST: linear_leaves discriminates (min bestD "
          f"{100*min_delta:+.2f}%, favorable={pos['favorable']})={ok_pos} | "
          f"patience300 flat={ok_flat} | "
          f"{'PASS' if ok_pos and ok_flat else 'FAIL'}", flush=True)
    return ok_pos and ok_flat


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idea", help=f"idea name; one of {sorted(ideas.IDEAS)}")
    ap.add_argument("--tier", nargs="+", default=["T0", "T1"],
                    choices=["T0", "T1", "T2", "HOLDOUT"])
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes over datasets (1 = serial)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the engine self-test (known-truth reproduction)")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest(args.seed, args.threads, args.jobs)
        raise SystemExit(0 if ok else 1)
    if not args.idea:
        ap.error("pass --idea NAME or --selftest")
    cascade(args.idea, args.tier, args.seed, args.threads, args.jobs)


if __name__ == "__main__":
    main()
