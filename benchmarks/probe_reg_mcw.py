"""Probe C3: does small-data REGRESSION want a real min_child_weight floor?

MECHANISM HYPOTHESIS (pre-registered 2026-08-01, before any results —
BREAKTHROUGH_PLAN.md "C3"):

For squared error the Hessian is exactly 1 per row, so the regressor's
min_child_weight=1.0 means "each child holds >= 1 sample" — no constraint at
all, at every dataset size. The classifier fades a veto in below ~2k rows
(`_auto_min_child_weight`: full veto under 500 rows, gone above 2000); the
regressor pins 1.0 forever. The PMLB random-search study found
min_child_weight is the ONE knob whose tuning transfers. And Finding 3 says
our win rate vs CatBoost collapses exactly on small data, where sparse leaves
are the obvious suspect the regressor is structurally blind to.

For squared error, min_child_weight IS min-rows-per-child, so the arms below
read directly as a per-leaf sample floor.

DESIGN. The decide strata carry too few regression sets at @sus* (6-7 in
gr:sus25, 3 in gr:sus50) to answer this, so the probe applies the sus
mechanism to EVERY decision-suite regression set: split 25% test at
random_state=seed (the harness convention), then shrink the TRAINING rows to
{100%, 50%, 25%} with the harness's own `_subsample_train` (random_state=0,
test set unchanged) — each dataset becomes its own three-point learning
curve, scored on identical test rows. NO extra row cap: frac=1.0 is the
harness's own size (the 50k gr: / 100k hc: builder caps), so the top of the
curve is the regime the decide gate actually runs.

ARMS (exactly paired: same split, same shrink, only min_child_weight moves):
    mcw in {1 (shipped), 4, 8, 16, 32}
Everything else is the out-of-box default the harness measures
(n_estimators=2000, early_stopping_rounds=50, random_state=0). PRIMARY ARM
is mcw=8, named before the run; the other three are supporting evidence and
the sign tests carry a Holm correction across the four, because four arms
times three sizes is twelve chances to find a majority in noise.

PREDICTIONS:
  C3 right: the primary arm beats mcw=1 at frac 0.25 on a Holm-corrected
  sign test, each dataset's own best mcw rises as its rows shrink, and at
  full size large mcw is neutral-to-harmful (the oblivious-veto underfit that
  made the classifier auto fade to zero). Ship shape: a size-adaptive
  regressor auto, then the standard tier-1/tier-2 gates.
  C3 wrong: flat-to-negative at every size => thread closes; the small-data
  deficit is not a min-leaf story. Read the COST block before concluding
  that: if rounds and fit seconds are also unmoved, the veto never bound and
  the honest finding is "the arm did nothing at these values", not "the
  mechanism is refuted".

READING CONVENTIONS, all matched to the house tools:
  * seeds are averaged on the METRIC, then one ratio is formed per cell
    (compare_runs / probe_bag_member_refit convention) — not a mean of
    per-seed percentages, which would weight the easiest split most.
  * wins/losses/ties use compare_runs' +/-1e-9 dead band, so "the veto never
    bound" shows as a tie rather than being booked as a loss.
  * near-solved cells are excluded on the BEST arm in the cell, matching
    compare_runs.is_near_solved (best across models), not on the baseline.
  * per-dataset rows are printed before any verdict, so the size claim is
    checkable rather than eyeballed off a population median.

CAVEATS THE OUTPUT STATES RATHER THAN HIDES:
  * 8 of the 42 keys are the same source data entered twice (abalone,
    Bike_Sharing_Demand, Brazilian_houses, delays_zurich_transport, diamonds,
    house_sales, medical_charges, nyc-taxi-green-dec-2016 appear in both
    reg_num and reg_cat), so the win denominators rest on ~34 independent
    sources.
  * `rounds` is `best_iteration_`, which under the default refit_full="replay"
    is the full-data refit budget (~t_star/0.8, clipped at n_estimators), not
    the raw early-stopping optimum. Direction survives, magnitude is rescaled.
  * fit seconds are probe-internal only (thread_count is left at the class
    default, while the harness pins it), never comparable to a harness
    slowdown figure.
  * n_train is PRE-split; the model's own early-stopping split takes 20%, so
    the classifier's 500/2000-row fade region lands at 625/2500 on this axis.
    The tables print post-split rows to keep the comparison honest.

Primary metric: test RMSE. Resumable JSONL; `--table-only` reprints.
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
from chimeraboost import ChimeraBoostRegressor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-reg-mcw.jsonl")
SEEDS = (0, 1, 2)
FRACS = (1.0, 0.5, 0.25)
ARMS = (1.0, 4.0, 8.0, 16.0, 32.0)      # 1.0 == the shipped regressor default
PRIMARY_ARM = "8"                       # named before the run
TIE_BAND = 1e-9                         # compare_runs' dead band
ES_FRACTION = 0.8                       # model's own early-stopping split keeps 80%

# hc: first — those six need a live OpenML fetch (nothing is cached locally and
# two are 100k rows), so a network problem surfaces in the first minute rather
# than after an hour of Grinsztajn work.
DATASETS = (
    [f"hc:{n}" for n, spec in rb.HC_DATASETS.items()
     if spec["task"] == "regression"]
    + [f"gr:reg_num/{n}" for n in rb.GRINSZTAJN_DATASETS["reg_num"]]
    + [f"gr:reg_cat/{n}" for n in rb.GRINSZTAJN_DATASETS["reg_cat"]]
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


def main():
    print(f"chimeraboost from {chimeraboost.__file__}")
    rb._add_highcard_datasets()   # rdata registers oml/gr/pm but not hc
    done = _done()
    try:
        for key in DATASETS:
            try:
                X, y, cat, task = rdata.load(key)
                assert task == "regression", f"{key} is not regression"
                print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}", flush=True)
                for seed in SEEDS:
                    Xtr0, Xte, ytr0, yte = train_test_split(
                        X, y, test_size=0.25, random_state=seed)
                    y_std = float(np.std(y))
                    for frac in FRACS:
                        if (key, seed, frac) in done:
                            continue
                        _run_cell(key, seed, frac, Xtr0, ytr0, Xte, yte,
                                  cat, y_std)
            except Exception as exc:            # one bad dataset must not
                print(f"  [skip] {key}: {type(exc).__name__}: {exc}",
                      flush=True)               # cost the other 41
    finally:
        table()


def _run_cell(key, seed, frac, Xtr0, ytr0, Xte, yte, cat, y_std):
    Xtr, ytr = rb._subsample_train(Xtr0, ytr0, frac, "regression")
    rec = {"dataset": key, "seed": seed, "frac": frac,
           "n_train": int(len(ytr)), "y_std": y_std,
           "rmse": {}, "rounds": {}, "fit_s": {}}
    line = f"  s{seed} f{frac:4.2f} n={len(ytr):>6d} "
    for mcw in ARMS:
        t0 = time.time()
        m = ChimeraBoostRegressor(
            n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
            min_child_weight=mcw, random_state=0)
        m.fit(Xtr, ytr, cat_features=cat)
        fit_s = time.time() - t0
        k = str(int(mcw))
        rec["rmse"][k] = float(np.sqrt(np.mean((yte - m.predict(Xte)) ** 2)))
        rec["rounds"][k] = int(m.best_iteration_)
        rec["fit_s"][k] = fit_s
        line += f" | {k}: {rec['rmse'][k]:.5g}"
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Analysis
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
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, pvals[i] * (len(pvals) - rank))
        running = max(running, adj)      # enforce monotonicity
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
    """{(dataset, frac): {arm: mean rmse over seeds, ...}} plus metadata.

    Seeds are averaged on the METRIC before any ratio is formed — the house
    convention (compare_runs, probe_bag_member_refit)."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["frac"])].append(r)
    out = {}
    for cellkey, rs in grouped.items():
        arms = {k: float(np.mean([r["rmse"][k] for r in rs]))
                for k in rs[0]["rmse"]}
        out[cellkey] = {
            "rmse": arms,
            "rounds": {k: float(np.mean([r["rounds"][k] for r in rs]))
                       for k in rs[0]["rounds"]},
            "fit_s": {k: float(np.mean([r["fit_s"][k] for r in rs]))
                      for k in rs[0]["fit_s"]},
            "n_train": int(np.mean([r["n_train"] for r in rs])),
            "y_std": rs[0]["y_std"],
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

    arm_keys = [str(int(a)) for a in ARMS if a != 1.0]
    all_keys = [str(int(a)) for a in ARMS]
    cells = _cells(rows)

    # Near-solved exclusion on the BEST arm in the cell (compare_runs rule),
    # plus explicit drops for degenerate targets that the ratio cannot express.
    kept, dropped = {}, []
    for ck, c in cells.items():
        base = c["rmse"].get("1")
        best = min(c["rmse"].values()) if c["rmse"] else None
        if not base or base <= 0 or c["y_std"] <= 0:
            dropped.append((ck, "degenerate"))
        elif best / c["y_std"] < NEAR_SOLVED_NRMSE:
            dropped.append((ck, "near-solved"))
        else:
            kept[ck] = c

    def gain(c, k):
        return 100.0 * (c["rmse"]["1"] - c["rmse"][k]) / c["rmse"]["1"]

    datasets = sorted({ds for ds, _ in kept})
    print("\n" + "=" * 104)
    print("C3 PROBE - regressor min_child_weight sweep")
    print("  % RMSE improvement over the shipped mcw=1. Cells are 3-seed means "
          "of RMSE, then one ratio.")
    print("  Positive = the min-leaf floor helps. Near-solved cells excluded on "
          "the best arm.")
    print(f"  PRIMARY ARM (named before the run): mcw={PRIMARY_ARM}. Sign tests "
          "are Holm-corrected across the 4 arms.")
    print("  NOTE: 8 of 42 keys are the same source data in both reg_num and "
          "reg_cat, so the")
    print("  denominators rest on ~34 independent sources.")
    print("=" * 104)

    # ---- Block A: per-dataset rows, printed BEFORE any verdict -------------
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        print(f"\n-- per dataset, train fraction {frac:.2f} " + "-" * 60)
        print(f"{'dataset':44s}{'rows':>7s}{'post':>7s}"
              + "".join(f"{'mcw=' + k:>11s}" for k in arm_keys) + "   best")
        for ds in present:
            c = kept[(ds, frac)]
            best = min(all_keys, key=lambda k: c["rmse"][k])
            line = (f"{ds[:44]:44s}{c['n_train']:>7d}"
                    f"{int(c['n_train'] * ES_FRACTION):>7d}")
            for k in arm_keys:
                line += f"{gain(c, k):+10.3f}%"
            print(line + f"   {best:>4s}")

    # ---- Block B: verdict per fraction, W-L-T + Holm sign test ------------
    print("\n" + "=" * 104)
    print("VERDICT by train fraction (W-L-T uses compare_runs' 1e-9 dead band; "
          "p is Holm-adjusted)")
    print("=" * 104)
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        gains = {k: [gain(kept[(ds, frac)], k) for ds in present]
                 for k in arm_keys}
        raw_p = [_sign_p(*_wlt(gains[k])[:2]) for k in arm_keys]
        adj_p = _holm(raw_p)
        print(f"\n  frac {frac:.2f}   ({len(present)} datasets)")
        for k, p in zip(arm_keys, adj_p):
            w, l, t = _wlt(gains[k])
            lo, hi = _median_ci(gains[k])
            star = "  <== PRIMARY" if k == PRIMARY_ARM else ""
            print(f"    mcw={k:>2s}  median {np.median(gains[k]):+7.3f}% "
                  f"[{lo:+.3f}, {hi:+.3f}]   {w:>2d}W-{l:>2d}L-{t:>2d}T   "
                  f"p={p:.3f}{star}")

    # ---- Block C: the cost read (pre-registered) --------------------------
    print("\n" + "=" * 104)
    print("COST - median ratio to the mcw=1 arm. A veto that never BINDS moves "
          "neither column;")
    print("  that reads as 'the arm did nothing', not as a refutation. "
          "(fit seconds are probe-internal:")
    print("  thread_count is unpinned, so never compare these to a harness "
          "slowdown.)")
    print("=" * 104)
    print(f"{'frac':>6s}" + "".join(f"{'mcw=' + k:>22s}" for k in arm_keys))
    print(f"{'':>6s}" + "".join(f"{'rounds    fit':>22s}" for k in arm_keys))
    for frac in FRACS:
        present = [ds for ds in datasets if (ds, frac) in kept]
        if not present:
            continue
        line = f"{frac:>6.2f}"
        for k in arm_keys:
            rr = np.median([kept[(ds, frac)]["rounds"][k]
                            / max(kept[(ds, frac)]["rounds"]["1"], 1)
                            for ds in present])
            fr = np.median([kept[(ds, frac)]["fit_s"][k]
                            / max(kept[(ds, frac)]["fit_s"]["1"], 1e-9)
                            for ds in present])
            line += f"{rr:>13.2f}x{fr:>7.2f}x"
        print(line)

    # ---- Block D: the mechanism read, per dataset -------------------------
    # "the best mcw rises as rows shrink" is a WITHIN-dataset claim, so count
    # datasets, not pooled cells.
    print("\n" + "=" * 104)
    print("MECHANISM - does each dataset's OWN best mcw rise as its rows "
          "shrink?")
    print("  Per dataset: the argmin-RMSE arm at each fraction. This is the "
          "pre-registered claim,")
    print("  counted within datasets so bucket membership cannot stand in for "
          "dataset identity.")
    print("=" * 104)
    up = down = flat = 0
    print(f"{'dataset':44s}{'f=1.00':>9s}{'f=0.50':>9s}{'f=0.25':>9s}"
          f"{'  direction':>12s}")
    for ds in datasets:
        best = {}
        for frac in FRACS:
            if (ds, frac) in kept:
                c = kept[(ds, frac)]
                best[frac] = int(min(all_keys, key=lambda k: c["rmse"][k]))
        if len(best) < 2:
            continue
        hi_f, lo_f = max(best), min(best)
        d = best[lo_f] - best[hi_f]      # smaller data minus larger data
        direction = "rises" if d > 0 else ("falls" if d < 0 else "flat")
        up += d > 0
        down += d < 0
        flat += d == 0
        print(f"{ds[:44]:44s}"
              + "".join(f"{best.get(f, '-'):>9}" for f in FRACS)
              + f"{direction:>12s}")
    print(f"\n  best arm RISES as data shrinks on {up} datasets, FALLS on "
          f"{down}, unchanged on {flat}.")
    print(f"  sign test on the datasets that moved: p={_sign_p(up, down):.3f}")

    # ---- Block E: extremes, with the denominator visible ------------------
    flat_list = []
    for (ds, frac), c in kept.items():
        for k in arm_keys:
            flat_list.append((gain(c, k), ds, frac, k,
                              c["rmse"]["1"] / c["y_std"],
                              c["rmse"]["1"] - c["rmse"][k]))
    flat_list.sort()
    print("\n" + "=" * 104)
    print("EXTREMES - baseline NRMSE and the ABSOLUTE delta are printed so a "
          "big % on a tiny")
    print("  denominator cannot pass for a big effect.")
    print("=" * 104)
    for g, ds, frac, k, nrmse, delta in flat_list[:6]:
        print(f"  {g:+9.3f}%  {ds[:40]:40s} f={frac:.2f} mcw={k:>2s}  "
              f"NRMSE={nrmse:6.3f}  dRMSE={delta:+.5g}")
    print("  ...")
    for g, ds, frac, k, nrmse, delta in flat_list[-6:]:
        print(f"  {g:+9.3f}%  {ds[:40]:40s} f={frac:.2f} mcw={k:>2s}  "
              f"NRMSE={nrmse:6.3f}  dRMSE={delta:+.5g}")

    if dropped:
        print(f"\nexcluded cells ({len(dropped)}): "
              + ", ".join(f"{d}@{f}[{why}]" for (d, f), why in dropped))


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
