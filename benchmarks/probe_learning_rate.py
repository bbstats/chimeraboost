"""Probe C4: is our size-blind learning rate the small-data gap vs CatBoost?

MECHANISM HYPOTHESIS (pre-registered 2026-08-01, before any results —
benchmarks/SMALLDATA_PLAN.md "FINDING 5" / "C4"):

Of CatBoost's 43 resolved parameters, exactly ONE varies with dataset size,
and it is the learning rate (scratchpad/catboost_size_defaults.py). It follows
a clean power law in n — regression 0.0259*(n/200)^0.157, binary
0.0158*(n/200)^0.247 — sitting below our flat 0.1 at every size in our suites,
but widening from ~1.6x at 60k rows to 4x (regression) / 6.4x (binary) at 200
rows. That is the shape of the collapse curve in Finding 3. Notably
`boosting_type` does NOT vary, so ordered boosting is not what switches on at
small n.

`_auto_learning_rate` returns a flat 0.1 whenever early stopping is on, on the
grounds that it "converges in ~half the trees of a smaller rate with no
measured accuracy cost". That was measured at full size; nobody tested it at a
quarter size.

This is a different axis from the three capacity levers the predecessor
killed. Depth / min_child_weight / the leaf guard all restrict what a tree can
express; the learning rate restricts nothing, since early stopping is free to
buy back capacity with more rounds. What it changes is the step size along the
boosting path, and therefore how finely early stopping can pick where to stop
on a validation curve that is noisy and shallow when rows are few.

DESIGN. 12 decision-suite datasets: the 6 binary sets where CatBoost beats us,
2 binary controls where we win comfortably, 4 regression sets. Split 25% test
at random_state=seed (harness convention), then shrink TRAINING rows to
{100%, 50%, 25%} with the harness's own `_subsample_train` (random_state=0,
test set unchanged) — each dataset becomes its own three-point learning curve
scored on identical test rows.

ARMS (exactly paired: same split, same shrink, only the rate moves):
    ours     lr in {0.1 (shipped), 0.05, 0.03, cb}
    opponent CatBoost @ auto (its default), CatBoost @ 0.1 (ablate the opponent)
`cb` is CatBoost's power law evaluated at the cell's own row count. PRIMARY ARM
is `cb` at frac 0.25, named before the run — it is the size-adaptive shape that
would actually ship. Sign tests carry a Holm correction across the 4 arms.

The CatBoost@0.1 arm is the "ablate the opponent" method that earned its keep
twice in the predecessor: if denying CatBoost its rate schedule erases its
small-data edge, the mechanism is confirmed from its side for one benchmark.

PREDICTIONS:
  C4 right: `cb` beats 0.1 at frac 0.25 on a Holm-corrected sign test, the
  advantage grows monotonically as rows shrink, and CatBoost's edge shrinks
  when forced to 0.1. Ship shape: a size-adaptive `_auto_learning_rate`, then
  the standard tier-1 synth / tier-2 decide gates.
  C4 wrong: flat-to-negative at every size => thread closes, the flat 0.1 is
  vindicated at small size too. Read the COST block first: if round counts are
  ALSO unmoved the arms never bound, and the honest finding is "these rates did
  nothing", not a refutation.
  Pareto kill: a strength win still dies if the fit-time ratio where it wins
  costs more on the frontier than the strength buys. The chart is the bar.

CAVEATS THE OUTPUT STATES RATHER THAN HIDES:
  * rows are capped at 20,000 for this pilot, so frac=1.00 here is NOT the
    decide regime on the largest sets. A survivor is re-run uncapped.
  * `rounds` is `best_iteration_`, which under the default refit_full="replay"
    is the full-data refit budget, not the raw early-stopping optimum.
  * fit seconds are probe-internal (thread_count unpinned), never comparable to
    a harness slowdown figure.
  * the `cb` schedule is evaluated at PRE-split rows; the model's own split
    keeps 80%, which moves the rate by under 4% at these exponents.

Primary metric: RMSE (regression) / Brier (binary). Resumable JSONL;
`--table-only` reprints.
"""

import json
import math
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb                     # noqa: E402
from research import datasets as rdata          # noqa: E402
from summarize import NEAR_SOLVED_NRMSE         # noqa: E402

import chimeraboost                             # noqa: E402
from chimeraboost import (ChimeraBoostRegressor,  # noqa: E402
                          ChimeraBoostClassifier)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-learning-rate.jsonl")
SEEDS = (0, 1, 2)
FRACS = (1.0, 0.5, 0.25)
FIXED_ARMS = ("0.1", "0.05", "0.03")     # "0.1" == the shipped default
BASE_ARM = "0.1"
OUR_ARMS = FIXED_ARMS + ("cb",)
CAT_ARMS = ("cat", "cat@0.1")
PRIMARY_ARM = "cb"                       # named before the run
PRIMARY_FRAC = 0.25
TIE_BAND = 1e-9                          # compare_runs' dead band
PILOT_MAX_ROWS = 20_000

# --knee: the follow-up run. The pilot measured 0.1 / 0.05 / 0.03 and found the
# STRENGTH saturates by 0.05 while the COST keeps climbing (rounds 2.2x -> 3.2x),
# so the best point on the curve is somewhere the pilot never sampled. This mode
# fills in 0.07 and drops every arm not needed to locate the knee.
#
# It also PINS thread_count. The pilot left it at the class default, which is
# why its fit seconds carried a "probe-internal, never compare to a harness
# slowdown" caveat. With fit time now the axis that decides this, the least
# trustworthy number in the table was the one being decided on. Pinning makes
# the ratio between arms decision-grade; the absolute seconds still are not.
KNEE_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "probe-learning-rate-knee.jsonl")
KNEE_ARMS = ("0.1", "0.07", "0.05")
KNEE_THREADS = 4
THREAD_COUNT = None                      # set to KNEE_THREADS in --knee mode

# CatBoost's own auto rate, fitted to the 8-point sweep in
# scratchpad/catboost_size_defaults.py (reproduces every point to under 1%).
_CB_LR = {"regression": (0.025914, 0.15697), "binary": (0.015751, 0.24700)}


def cb_learning_rate(n, task):
    a, b = _CB_LR["regression" if task == "regression" else "binary"]
    return float(a * (max(n, 2) / 200.0) ** b)


DATASETS = (
    # the 6 binary sets where CatBoost beats us (BREAKTHROUGH_PLAN baseline)
    "gr:clf_num/bank-marketing",
    "gr:clf_num/credit",
    "gr:clf_num/heloc",
    "gr:clf_num/california",
    "gr:clf_num/default-of-credit-card-clients",
    "gr:clf_num/Diabetes130US",
    # binary controls where we win comfortably
    "gr:clf_num/pol",
    "gr:clf_num/MagicTelescope",
    # regression, incl. the set that has confessed a capacity preference twice
    "gr:reg_num/cpu_act",
    "gr:reg_num/sulfur",
    "gr:reg_num/houses",
    "gr:reg_num/superconduct",
)


def _done():
    seen = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        seen.add((r["dataset"], r["seed"], r["frac"]))
                    except Exception:
                        pass
    return seen


def _score(task, yte, pred):
    """House primary: RMSE for regression, Brier for binary."""
    if task == "regression":
        return float(np.sqrt(np.mean((yte - pred) ** 2)))
    return float(np.mean((pred - yte) ** 2))


def main():
    print(f"chimeraboost from {chimeraboost.__file__}")
    done = _done()
    try:
        for key in DATASETS:
            try:
                X, y, cat, task = rdata.load(key, max_rows=PILOT_MAX_ROWS)
                print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}  {task}",
                      flush=True)
                for seed in SEEDS:
                    strat = y if task != "regression" else None
                    Xtr0, Xte, ytr0, yte = train_test_split(
                        X, y, test_size=0.25, random_state=seed,
                        stratify=strat)
                    y_std = float(np.std(y)) if task == "regression" else 1.0
                    for frac in FRACS:
                        if (key, seed, frac) in done:
                            continue
                        _run_cell(key, task, seed, frac, Xtr0, ytr0, Xte, yte,
                                  cat, y_std)
            except Exception as exc:            # one bad dataset must not
                print(f"  [skip] {key}: {type(exc).__name__}: {exc}",
                      flush=True)               # cost the other 11
    finally:
        table()


def _fit_ours(task, lr, Xtr, ytr, Xte, cat):
    """Out-of-box default path: NO explicit eval_set, so the internal
    early-stopping split and refit_full run exactly as a user gets them."""
    Est = (ChimeraBoostRegressor if task == "regression"
           else ChimeraBoostClassifier)
    kw = {} if THREAD_COUNT is None else {"thread_count": THREAD_COUNT}
    m = Est(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
            learning_rate=lr, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    pred = (m.predict(Xte) if task == "regression"
            else m.predict_proba(Xte)[:, 1])
    return pred, int(m.best_iteration_)


def _fit_cat(task, lr, Xtr, ytr, Xte, cat):
    """Mirrors the harness's `_run_catboost`: an explicit 80/20 eval_set."""
    from catboost import CatBoostRegressor, CatBoostClassifier
    Xf, Xv, yf, yv = rb._val_split(Xtr, ytr, task, 0)
    common = dict(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
                  verbose=False, random_seed=0)
    if lr is not None:
        common["learning_rate"] = lr
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = Est(**common)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    pred = (m.predict(Xte) if task == "regression"
            else m.predict_proba(Xte)[:, 1])
    return pred, int(m.best_iteration_)


def _run_cell(key, task, seed, frac, Xtr0, ytr0, Xte, yte, cat, y_std):
    Xtr, ytr = rb._subsample_train(Xtr0, ytr0, frac, task)
    n = int(len(ytr))
    rec = {"dataset": key, "task": task, "seed": seed, "frac": frac,
           "n_train": n, "y_std": y_std, "cb_lr": cb_learning_rate(n, task),
           "score": {}, "rounds": {}, "fit_s": {}}
    line = f"  s{seed} f{frac:4.2f} n={n:>6d} "
    for arm in OUR_ARMS + CAT_ARMS:
        t0 = time.time()
        if arm in CAT_ARMS:
            lr = None if arm == "cat" else 0.1
            pred, rounds = _fit_cat(task, lr, Xtr, ytr, Xte, cat)
        else:
            lr = rec["cb_lr"] if arm == "cb" else float(arm)
            pred, rounds = _fit_ours(task, lr, Xtr, ytr, Xte, cat)
        rec["fit_s"][arm] = time.time() - t0
        rec["score"][arm] = _score(task, yte, pred)
        rec["rounds"][arm] = rounds
        line += f" | {arm}: {rec['score'][arm]:.5g}"
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Analysis (conventions copied from probe_reg_mcw.py)
# ---------------------------------------------------------------------------

def _sign_p(w, l):
    """Two-sided exact binomial p at q=0.5 (ties already excluded)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def _holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, pvals[i] * (len(pvals) - rank))
        running = max(running, adj)
        out[i] = running
    return out


def _median_ci(vals, n_boot=2000, seed=0):
    if not vals:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    meds = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _wlt(gains):
    w = sum(1 for g in gains if g > TIE_BAND)
    l = sum(1 for g in gains if g < -TIE_BAND)
    return w, l, len(gains) - w - l


def _cells(rows):
    """{(dataset, frac): {arm: mean score over seeds}}; seeds averaged on the
    METRIC before any ratio (compare_runs convention)."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["frac"])].append(r)
    out = {}
    for cellkey, rs in grouped.items():
        keys = [k for k in rs[0]["score"]
                if all(k in r["score"] for r in rs)]
        out[cellkey] = {
            "score": {k: float(np.mean([r["score"][k] for r in rs]))
                      for k in keys},
            "rounds": {k: float(np.mean([r["rounds"][k] for r in rs]))
                       for k in keys},
            "fit_s": {k: float(np.mean([r["fit_s"][k] for r in rs]))
                      for k in keys},
            "task": rs[0]["task"],
            "n_train": int(np.mean([r["n_train"] for r in rs])),
            "y_std": rs[0]["y_std"],
            "cb_lr": float(np.mean([r["cb_lr"] for r in rs])),
            "n_seeds": len(rs),
        }
    return out


def table():
    if not os.path.exists(RESULTS):
        print("no results yet")
        return
    rows = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no results yet")
        return

    cells = _cells(rows)
    arm_keys = [a for a in OUR_ARMS if a != BASE_ARM]

    # Near-solved exclusion on the BEST arm in the cell (compare_runs rule).
    kept, dropped = {}, []
    for ck, c in cells.items():
        base = c["score"].get(BASE_ARM)
        if not base or base <= 0:
            dropped.append((ck, "degenerate"))
        elif (c["task"] == "regression"
              and min(c["score"].values()) / c["y_std"] < NEAR_SOLVED_NRMSE):
            dropped.append((ck, "near-solved"))
        else:
            kept[ck] = c

    def gain(c, k):
        """% improvement over the shipped 0.1 arm (positive = better)."""
        return 100.0 * (c["score"][BASE_ARM] - c["score"][k]) / c["score"][BASE_ARM]

    datasets = sorted({ds for ds, _ in kept})
    print("\n" + "=" * 112)
    print("C4 PROBE - learning-rate sweep vs the shipped flat 0.1")
    print("  % improvement over lr=0.1 in the house primary (RMSE regression / "
          "Brier binary).")
    print("  Cells are 3-seed means of the metric, then one ratio. Positive = "
          "the lower rate helps.")
    print(f"  PRIMARY ARM (named before the run): '{PRIMARY_ARM}' at frac "
          f"{PRIMARY_FRAC:.2f}. Sign tests Holm-corrected across the 4 arms.")
    print(f"  PILOT CAVEAT: rows capped at {PILOT_MAX_ROWS:,}, so frac=1.00 is "
          "NOT the decide regime on the largest sets.")
    print("=" * 112)

    # ---- Block A: per-dataset rows, BEFORE any verdict ---------------------
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        print(f"\n-- per dataset, train fraction {frac:.2f} " + "-" * 68)
        print(f"{'dataset':42s}{'rows':>7s}{'cb_lr':>8s}"
              + "".join(f"{a:>10s}" for a in arm_keys) + "    best")
        for ds in present:
            c = kept[(ds, frac)]
            best = min(c["score"], key=lambda k: c["score"][k])
            line = f"{ds[:42]:42s}{c['n_train']:>7d}{c['cb_lr']:>8.4f}"
            for a in arm_keys:
                line += f"{gain(c, a):+9.3f}%"
            print(line + f"    {best:>7s}")

    # ---- Block B: verdict per fraction -------------------------------------
    print("\n" + "=" * 112)
    print("VERDICT by train fraction (W-L-T on compare_runs' 1e-9 dead band; "
          "p is Holm-adjusted across arms)")
    print("=" * 112)
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        gains = {a: [gain(kept[(ds, frac)], a) for ds in present]
                 for a in arm_keys}
        adj_p = _holm([_sign_p(*_wlt(gains[a])[:2]) for a in arm_keys])
        print(f"\n  frac {frac:.2f}   ({len(present)} datasets)")
        for a, p in zip(arm_keys, adj_p):
            w, l, t = _wlt(gains[a])
            lo, hi = _median_ci(gains[a])
            star = ("  <== PRIMARY" if a == PRIMARY_ARM and frac == PRIMARY_FRAC
                    else "")
            print(f"    lr={a:>5s}  median {np.median(gains[a]):+7.3f}% "
                  f"[{lo:+.3f}, {hi:+.3f}]   {w:>2d}W-{l:>2d}L-{t:>2d}T   "
                  f"p={p:.3f}{star}")

    # ---- Block C: cost (pre-registered) ------------------------------------
    print("\n" + "=" * 112)
    print("COST - median ratio to the lr=0.1 arm. Arms that move NEITHER column "
          "never bound, which")
    print("  reads as 'the rate did nothing' rather than a refutation. (fit "
          "seconds are probe-internal:")
    print("  thread_count is unpinned, so never compare these to a harness "
          "slowdown figure.)")
    print("=" * 112)
    print(f"{'frac':>6s}" + "".join(f"{'lr=' + a:>22s}" for a in arm_keys))
    print(f"{'':>6s}" + "".join(f"{'rounds    fit':>22s}" for _ in arm_keys))
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        line = f"{frac:>6.2f}"
        for a in arm_keys:
            rr = np.median([kept[(ds, frac)]["rounds"][a]
                            / max(kept[(ds, frac)]["rounds"][BASE_ARM], 1)
                            for ds in present])
            fr = np.median([kept[(ds, frac)]["fit_s"][a]
                            / max(kept[(ds, frac)]["fit_s"][BASE_ARM], 1e-9)
                            for ds in present])
            line += f"{rr:>13.2f}x{fr:>7.2f}x"
        print(line)

    # ---- Block D: ablate the opponent --------------------------------------
    if not CAT_ARMS:
        _mechanism_block(datasets, kept)      # knee mode carries no CatBoost arms
        return
    print("\n" + "=" * 112)
    print("ABLATE THE OPPONENT - CatBoost's edge over our SHIPPED arm, at its "
          "own auto rate vs forced")
    print("  to our 0.1. Positive = CatBoost ahead. If its edge shrinks when "
          "denied the schedule, the")
    print("  mechanism is confirmed from its side.")
    print("=" * 112)
    print(f"{'frac':>6s}{'n':>6s}{'cat@auto edge':>16s}{'leads':>8s}"
          f"{'cat@0.1 edge':>16s}{'leads':>8s}{'  edge explained by rate':>26s}")
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        auto, forced = [], []
        for ds in present:
            c = kept[(ds, frac)]
            b = c["score"][BASE_ARM]
            auto.append(100.0 * (b - c["score"]["cat"]) / b)
            forced.append(100.0 * (b - c["score"]["cat@0.1"]) / b)
        ma, mf = float(np.mean(auto)), float(np.mean(forced))
        # "Share of CatBoost's edge explained by its rate" is only defined when
        # CatBoost HAS an edge. Where we are ahead on average (ma <= 0) the
        # ratio is not a share of anything, so print n/a rather than a number
        # that reads like one.
        share = (f"{100.0 * (ma - mf) / ma:>24.0f}%" if ma > 1e-9
                 else f"{'n/a (we lead)':>25s}")
        print(f"{frac:>6.2f}{len(present):>6d}{ma:>15.3f}%"
              f"{sum(1 for v in auto if v > 0):>5d}/{len(auto):<2d}"
              f"{mf:>15.3f}%{sum(1 for v in forced if v > 0):>5d}/"
              f"{len(forced):<2d}{share}")

    _mechanism_block(datasets, kept)
    if dropped:
        print(f"\nexcluded cells ({len(dropped)}): "
              + ", ".join(f"{d}@{f}[{why}]" for (d, f), why in dropped))


def _mechanism_block(datasets, kept):
    print("\n" + "=" * 112)
    print("MECHANISM - does each dataset's OWN best rate FALL as its rows "
          "shrink? Counted within")
    print("  datasets, so bucket membership cannot stand in for dataset "
          "identity. NOTE the panel is")
    print("  FIXED across fractions (same 12 datasets at every size), so the "
          "population trend in the")
    print("  VERDICT block cannot be a composition effect; only this "
          "per-dataset argmin is noisy.")
    print("=" * 112)
    order = {a: i for i, a in enumerate(FIXED_ARMS)}
    down = up = flat = 0
    print(f"{'dataset':42s}{'f=1.00':>10s}{'f=0.50':>10s}{'f=0.25':>10s}"
          f"{'  direction':>12s}")
    for ds in datasets:
        best = {}
        for frac in FRACS:
            if (ds, frac) in kept:
                c = kept[(ds, frac)]
                fixed = {k: v for k, v in c["score"].items() if k in order}
                if fixed:
                    best[frac] = min(fixed, key=lambda k: fixed[k])
        if len(best) < 2:
            continue
        hi_f, lo_f = max(best), min(best)
        d = order[best[lo_f]] - order[best[hi_f]]   # smaller data minus larger
        direction = "falls" if d > 0 else ("rises" if d < 0 else "flat")
        down += d > 0
        up += d < 0
        flat += d == 0
        print(f"{ds[:42]:42s}"
              + "".join(f"{best.get(f, '-'):>10}" for f in FRACS)
              + f"{direction:>12s}")
    print(f"\n  best fixed rate FALLS as data shrinks on {down} datasets, "
          f"RISES on {up}, unchanged on {flat}.")
    print(f"  sign test on the datasets that moved: p={_sign_p(down, up):.3f}")


def _enter_knee_mode():
    """Point every module-level knob at the knee run (see KNEE_RESULTS)."""
    global RESULTS, OUR_ARMS, CAT_ARMS, FIXED_ARMS, PRIMARY_ARM, THREAD_COUNT
    RESULTS = KNEE_RESULTS
    OUR_ARMS, CAT_ARMS, FIXED_ARMS = KNEE_ARMS, (), KNEE_ARMS
    PRIMARY_ARM = "0.07"
    THREAD_COUNT = KNEE_THREADS


if __name__ == "__main__":
    if "--knee" in sys.argv:
        _enter_knee_mode()
    if "--table-only" in sys.argv:
        table()
    else:
        main()
