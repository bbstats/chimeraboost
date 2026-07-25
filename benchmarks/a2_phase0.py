"""A2 Phase 0 — config-portfolio headroom + offline race simulation.

Pre-registered design and kill bars: benchmarks/A2_PLAN.md.

Fits every portfolio configuration to full early stopping on the PMLB TUNING
fold (the one suite this project tunes hyperparameters against — Grinsztajn,
the high-card suite, the OpenML gate and TabArena are all untouched here), and
records each fit's whole validation curve, its test metrics and its wall time.
The race is then simulated offline from those curves, so no library source
changes and no extra fits are needed to answer "what budget would we need".

Protocol mirrors the decision suites exactly: same loaders, same
default_rng(1000 + seed), same 25% test split, same shared budget, and
fit(X, y) with NO explicit eval_set so every fit uses the estimator's own
internal early-stopping split (out-of-box default behavior).

    python benchmarks/a2_phase0.py --seeds 3 --out a2-phase0
    python benchmarks/a2_phase0.py --report-only --out a2-phase0
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import train_test_split

import run_benchmarks as rb
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# Everything not named here stays at the shipped default.
PORTFOLIO = {
    "default": dict(depth=6),
    "d4":      dict(depth=4),
    "d8":      dict(depth=8),
    "sub08":   dict(depth=6, subsample=0.8),
}
RACE_BUDGETS = (50, 100, 200, 500)


def _loss(task, metrics):
    """Lower-is-better primary loss, on the north-star convention:
    RMSE for regression, Brier for classification."""
    return metrics["rmse"] if task == "regression" else metrics["brier"]


def _pct_better(task, base, cand):
    """Percent by which `cand` improves on `base` (positive = cand better)."""
    b, c = _loss(task, base), _loss(task, cand)
    return 100.0 * (b - c) / b if b else 0.0


def run(args):
    rb._add_pmlb_datasets()
    keys = args.datasets or [k for k in rb.DATASETS if k.startswith("pm:tune/")]
    keys.sort()

    print("Warmup (compiling numba kernels)...")
    from chimeraboost.warmup import warmup
    warmup()
    print(f"chimeraboost from: {__import__('chimeraboost').__file__}")

    threads = args.threads or os.cpu_count() or 1
    records = []
    for key in keys:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            X, y, cat, task = rb.DATASETS[key](args.scale, rng)
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=strat)
            for cfg_id, cfg in PORTFOLIO.items():
                Est = (ChimeraBoostRegressor if task == "regression"
                       else ChimeraBoostClassifier)
                m = Est(n_estimators=rb.MAX_ITERS,
                        early_stopping_rounds=rb.PATIENCE,
                        thread_count=threads, random_state=0, **cfg)
                t0 = time.time()
                m.fit(Xtr, ytr, cat_features=cat)
                secs = time.time() - t0
                metrics = rb._compute_metrics(task, yte, m, Xte)
                hist = [float(v) for v in (m.validation_history_ or [])]
                records.append({
                    "dataset": key, "seed": seed, "task": task,
                    "config": cfg_id, "secs": secs,
                    "n_train": int(Xtr.shape[0]),
                    "rmse": metrics.get("rmse"),
                    "brier": metrics.get("brier"),
                    "f1_macro": metrics.get("f1_macro"),
                    "valid_history": hist,
                    "rounds": len(hist),
                })
                print(f"  {key} seed{seed} {cfg_id:8s} "
                      f"loss={_loss(task, metrics):.5f} "
                      f"rounds={len(hist):4d} {secs:6.1f}s")

    out = {"portfolio": {k: v for k, v in PORTFOLIO.items()},
           "seeds": args.seeds, "records": records}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", f"{args.out}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {path}")
    return out


# --------------------------------------------------------------------------
# Offline analysis
# --------------------------------------------------------------------------
def _cells(records):
    """Group records into (dataset, seed) cells -> {config: record}."""
    cells = {}
    for r in records:
        cells.setdefault((r["dataset"], r["seed"]), {})[r["config"]] = r
    return cells


def _pick_by_val(cell, k=None):
    """Lowest best-validation-loss config; ties go to `default` (the incumbent
    rule everywhere else in this codebase)."""
    best_id, best_val = "default", np.inf
    for cfg_id, r in cell.items():
        h = r["valid_history"]
        if not h:
            continue
        v = min(h[:k]) if k else min(h)
        if v < best_val:
            best_id, best_val = cfg_id, v
        elif v == best_val and cfg_id == "default":
            best_id = cfg_id
    return best_id


def report(data):
    records = data["records"]
    cells = _cells(records)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# A2 Phase 0 - config-portfolio headroom (PMLB tune fold)")
    emit()
    emit(f"{len(cells)} cells (dataset x seed), "
         f"{len(PORTFOLIO)} configs, seeds={data['seeds']}")
    emit()

    # ---- per-config summary vs default -----------------------------------
    emit("## Each configuration vs the shipped default")
    emit()
    emit("| config | mean % vs default | wins | losses | ties | mean fit s |")
    emit("|---|--:|--:|--:|--:|--:|")
    for cfg_id in PORTFOLIO:
        deltas, w, l, t, secs = [], 0, 0, 0, []
        for cell in cells.values():
            if cfg_id not in cell or "default" not in cell:
                continue
            d = _pct_better(cell[cfg_id]["task"], cell["default"], cell[cfg_id])
            deltas.append(d)
            secs.append(cell[cfg_id]["secs"])
            if d > 0:
                w += 1
            elif d < 0:
                l += 1
            else:
                t += 1
        emit(f"| {cfg_id} | {np.mean(deltas):+.3f}% | {w} | {l} | {t} | "
             f"{np.mean(secs):.1f} |")
    emit()

    # ---- Bars A and B ----------------------------------------------------
    oracle_d, valsel_d, valsel_w, valsel_l, valsel_t = [], [], 0, 0, 0
    pick_counts, oracle_counts = {}, {}
    for cell in cells.values():
        if "default" not in cell:
            continue
        task = cell["default"]["task"]
        best = min(cell.values(), key=lambda r: _loss(r["task"], r))
        oracle_d.append(_pct_better(task, cell["default"], best))
        oracle_counts[best["config"]] = oracle_counts.get(best["config"], 0) + 1
        pick = _pick_by_val(cell)
        pick_counts[pick] = pick_counts.get(pick, 0) + 1
        d = _pct_better(task, cell["default"], cell[pick])
        valsel_d.append(d)
        if d > 0:
            valsel_w += 1
        elif d < 0:
            valsel_l += 1
        else:
            valsel_t += 1

    non_tied = valsel_w + valsel_l
    win_rate = 100.0 * valsel_w / non_tied if non_tied else 0.0
    bar_a = np.mean(valsel_d) >= 0.30 and win_rate >= 55.0
    bar_b = np.mean(oracle_d) >= 0.50

    emit("## Bars A and B - is there headroom, and can validation see it?")
    emit()
    emit("| quantity | value | bar | verdict |")
    emit("|---|--:|---|---|")
    emit(f"| Test-oracle headroom (ceiling) | {np.mean(oracle_d):+.3f}% "
         f"| >= +0.50% | {'PASS' if bar_b else 'FAIL'} |")
    emit(f"| Validation-selected headroom | {np.mean(valsel_d):+.3f}% "
         f"| >= +0.30% | {'PASS' if np.mean(valsel_d) >= 0.30 else 'FAIL'} |")
    emit(f"| Validation-selected win rate | {win_rate:.1f}% "
         f"({valsel_w}W-{valsel_l}L-{valsel_t}T) | >= 55% "
         f"| {'PASS' if win_rate >= 55.0 else 'FAIL'} |")
    emit()
    emit(f"Oracle picks: {oracle_counts}")
    emit(f"Validation picks: {pick_counts}")
    emit()
    recovered = (100.0 * np.mean(valsel_d) / np.mean(oracle_d)
                 if np.mean(oracle_d) else 0.0)
    emit(f"Validation selection recovers {recovered:.0f}% of the oracle.")
    emit()

    # ---- Bar C — race fidelity ------------------------------------------
    emit("## Bar C - would a capped race pick the same configuration?")
    emit()
    emit("| k rounds | agreement with full-fit pick | mean test regret |")
    emit("|--:|--:|--:|")
    bar_c = False
    for k in RACE_BUDGETS:
        agree, regrets = 0, []
        for cell in cells.values():
            if "default" not in cell:
                continue
            full_pick = _pick_by_val(cell)
            race_pick = _pick_by_val(cell, k=k)
            if race_pick == full_pick:
                agree += 1
            else:
                task = cell["default"]["task"]
                regrets.append(max(0.0, _pct_better(task, cell[race_pick],
                                                    cell[full_pick])))
        n = len([c for c in cells.values() if "default" in c])
        ag = 100.0 * agree / n if n else 0.0
        mean_regret = float(np.sum(regrets) / n) if n else 0.0
        if k == 100:
            bar_c = ag >= 75.0 and mean_regret <= 0.15
        emit(f"| {k} | {ag:.1f}% | {mean_regret:.3f}% |")
    emit()

    # ---- Bar D — cost ----------------------------------------------------
    emit("## Bar D - projected cost")
    emit()
    per_cfg_secs, per_cfg_rounds = {}, {}
    for cfg_id in PORTFOLIO:
        rs = [r for r in records if r["config"] == cfg_id]
        per_cfg_secs[cfg_id] = float(np.mean([r["secs"] for r in rs]))
        per_cfg_rounds[cfg_id] = float(np.mean([r["rounds"] for r in rs]))
    base_secs = per_cfg_secs["default"]
    # An audition costs k rounds of its config's per-round cost; the winner
    # still runs its own full fit (the fallback design already in the library).
    extra = 0.0
    for cfg_id in PORTFOLIO:
        if cfg_id == "default":
            continue
        per_round = per_cfg_secs[cfg_id] / max(per_cfg_rounds[cfg_id], 1)
        extra += per_round * min(100, per_cfg_rounds[cfg_id])
    projected = (base_secs + extra) / base_secs
    bar_d = projected <= 1.35
    emit("| config | mean fit s | mean rounds |")
    emit("|---|--:|--:|")
    for cfg_id in PORTFOLIO:
        emit(f"| {cfg_id} | {per_cfg_secs[cfg_id]:.1f} | "
             f"{per_cfg_rounds[cfg_id]:.0f} |")
    emit()
    emit(f"Projected fit-time multiple with 3 extra k=100 auditions: "
         f"{projected:.2f}x (bar <= 1.35x) "
         f"{'PASS' if bar_d else 'FAIL'}")
    emit()
    # Diagnostic only -- the bar above stays as pre-registered. This says what a
    # cheaper portfolio shape would cost, to inform a Phase 1 redesign if the
    # headroom bars pass. An audition is only cheap when the full fit runs many
    # more than k rounds, which is exactly what short multiclass fits do not do.
    emit("Cost of cheaper portfolio shapes (diagnostic, not the bar):")
    emit()
    emit("| extra configs | k=50 | k=100 |")
    emit("|---|--:|--:|")
    others = [c for c in PORTFOLIO if c != "default"]
    for n_extra in (1, 2, 3):
        row = []
        for k in (50, 100):
            cost = 0.0
            for cfg_id in others[:n_extra]:
                per_round = per_cfg_secs[cfg_id] / max(per_cfg_rounds[cfg_id], 1)
                cost += per_round * min(k, per_cfg_rounds[cfg_id])
            row.append(f"{(base_secs + cost) / base_secs:.2f}x")
        emit(f"| {n_extra} | {row[0]} | {row[1]} |")
    emit()

    emit("## Verdict")
    emit()
    for name, ok in (("A (capturable headroom)", bar_a),
                     ("B (oracle sanity)", bar_b),
                     ("C (race fidelity @ k=100)", bar_c),
                     ("D (cost)", bar_d)):
        emit(f"- Bar {name}: {'PASS' if ok else 'FAIL'}")
    emit()
    emit("**A2 PROCEEDS**" if all((bar_a, bar_b, bar_c, bar_d))
         else "**A2 KILLED at Phase 0**")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--out", default="a2-phase0")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", f"{args.out}.json")
    data = json.load(open(path)) if args.report_only else run(args)
    text = report(data)
    with io.open(path.replace(".json", ".md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
