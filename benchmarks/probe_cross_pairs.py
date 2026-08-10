"""E2 step 1 probe (SELECT_PLAN.md, E2): can rung 1 take cross features
without the race?

Three arms per (dataset, seed), all the inner GradientBoosting at a
rung-1-equivalent config (RMSE, linear_leaves=True, n_estimators=2000,
ES 50 on a 0.2 validation split, auto lr, depth 6 -- quality=1 minus the
sklearn wrapper, which cannot force cross pairs):

  plain  : cross_pairs=[]                                       (rung 1 today)
  probe  : cross_pairs from a 25-round probe fit's importances  (the E2 design)
  oracle : cross_pairs from the plain FULL fit's importances    (the ceiling)

Pair choice is the production `_cross_candidate_pairs` in both cross arms, so
diff/prod/gdiff and the top-M rules are exactly what would ship.

Kill bars (registered in SELECT_PLAN.md E2 BEFORE this ran), in order:
  1 headroom : oracle-over-plain test-RMSE gain >= +0.3% median on >= 30 fits
  2 fidelity : probe holds the oracle's headroom within noise (paired deltas)
  3 cost     : median augmented/plain fit-time ratio <= 1.5

Also reported: the 25-round probe's share of the arm's time (B12) and the
probe/oracle pair-set overlap. Results:
benchmarks/results/probe-cross-pairs-e2.jsonl (resumable, delete to rerun);
aggregate table printed at the end.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import ShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost.booster import GradientBoosting  # noqa: E402
from chimeraboost.sklearn_api import _cross_candidate_pairs  # noqa: E402

TOP_M = None                      # --top-m N: the pre-authorized cost retry
if "--top-m" in sys.argv:         # (SELECT_PLAN.md E2 step 1) narrows the
    TOP_M = int(sys.argv[sys.argv.index("--top-m") + 1])   # numeric block
    import chimeraboost.sklearn_api as _ska
    _ska.CROSS_TOP_M = TOP_M

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results",
    "probe-cross-pairs-e2" + (f"-m{TOP_M}" if TOP_M else "") + ".jsonl")
SEEDS = (0, 1, 2)
PROBE_ROUNDS = 25

# 7 sets where the racing A/B improved (cross should help), 4 controls it left
# flat, plus E1's swing dataset. All >= 2000 train rows, so every fit engages.
PANEL = [
    ("gr:reg_num/nyc-taxi-green-dec-2016", "gap"),
    ("gr:reg_cat/nyc-taxi-green-dec-2016", "gap"),
    ("gr:reg_num/Brazilian_houses", "gap"),
    ("gr:reg_cat/Brazilian_houses", "gap"),
    ("gr:reg_num/cpu_act", "gap"),
    ("gr:reg_num/sulfur", "gap"),
    ("gr:reg_num/pol", "gap"),
    ("gr:reg_num/elevators", "control"),
    ("gr:reg_num/houses", "control"),
    ("gr:reg_num/wine_quality", "control"),
    ("gr:reg_num/abalone", "control"),
    ("gr:reg_cat/analcatdata_supreme", "swing"),
]

CFG = dict(loss="RMSE", n_estimators=2000, early_stopping_rounds=50,
           depth=6, linear_leaves=True, random_state=0)


def _fit(Xf, yf, ev, cat, pairs, n_estimators=None, es=True):
    kw = dict(CFG, cross_pairs=pairs or None)
    if n_estimators is not None:
        kw["n_estimators"] = n_estimators
    if not es:
        kw["early_stopping_rounds"] = None
    m = GradientBoosting(**kw)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat, eval_set=ev)
    return m, time.time() - t


def _rmse(m, Xte, yte):
    return float(np.sqrt(np.mean((yte - m.predict_raw(Xte)) ** 2)))


def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"]))
    return done


def main():
    done = _done_keys()
    for key, group in PANEL:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        assert task == "regression", key
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed)
            # The estimator's auto ES split: 0.2 held out of the train rows.
            tr_idx, va_idx = next(ShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed).split(Xtr))
            Xf, yf = Xtr[tr_idx], ytr[tr_idx]
            ev = (Xtr[va_idx], ytr[va_idx])

            plain, plain_s = _fit(Xf, yf, ev, cat, None)
            base = _rmse(plain, Xte, yte)

            pfit, probe_s = _fit(Xf, yf, ev, cat, None,
                                 n_estimators=PROBE_ROUNDS, es=False)
            p_pairs = _cross_candidate_pairs(
                np.asarray(pfit.feature_importances_, dtype=float),
                cat, X.shape[1])
            o_pairs = _cross_candidate_pairs(
                np.asarray(plain.feature_importances_, dtype=float),
                cat, X.shape[1])

            probe_m, probe_fit_s = _fit(Xf, yf, ev, cat, p_pairs)
            oracle_m, oracle_fit_s = _fit(Xf, yf, ev, cat, o_pairs)

            inter = len(set(p_pairs) & set(o_pairs))
            union = len(set(p_pairs) | set(o_pairs))
            row = {"dataset": key, "seed": seed, "group": group,
                   "n_train": int(len(yf)), "n_pairs": len(o_pairs),
                   "pair_jaccard": round(inter / union, 3) if union else 1.0,
                   "base": base,
                   "probe": _rmse(probe_m, Xte, yte),
                   "oracle": _rmse(oracle_m, Xte, yte),
                   "plain_s": round(plain_s, 2),
                   "probe_fit_s": round(probe_fit_s, 2),
                   "oracle_fit_s": round(oracle_fit_s, 2),
                   "probe_s": round(probe_s, 2)}
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{key} s{seed}: base={base:.5g} "
                  f"probe={100 * (base - row['probe']) / base:+.2f}% "
                  f"oracle={100 * (base - row['oracle']) / base:+.2f}% "
                  f"jac={row['pair_jaccard']:.2f} "
                  f"fitx={(row['probe_fit_s'] + row['probe_s']) / max(plain_s, 1e-9):.2f}",
                  flush=True)
    table()


def table():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    print(f"\n{'dataset':40} {'grp':8} {'probe%':>8} {'oracle%':>8} "
          f"{'jac':>5} {'fitx':>5} {'probeshare':>10}")
    for ds, rs in sorted(by_ds.items()):
        p = np.mean([100 * (r["base"] - r["probe"]) / r["base"] for r in rs])
        o = np.mean([100 * (r["base"] - r["oracle"]) / r["base"] for r in rs])
        j = np.mean([r["pair_jaccard"] for r in rs])
        fx = np.mean([(r["probe_fit_s"] + r["probe_s"]) / max(r["plain_s"], 1e-9)
                      for r in rs])
        sh = np.mean([r["probe_s"] / max(r["probe_fit_s"] + r["probe_s"], 1e-9)
                      for r in rs])
        print(f"{ds:40} {rs[0]['group']:8} {p:+8.2f} {o:+8.2f} {j:5.2f} "
              f"{fx:5.2f} {sh:10.2f}")

    o_all = [100 * (r["base"] - r["oracle"]) / r["base"] for r in rows]
    p_all = [100 * (r["base"] - r["probe"]) / r["base"] for r in rows]
    fid = [p - o for p, o in zip(p_all, o_all)]
    fx_all = [(r["probe_fit_s"] + r["probe_s"]) / max(r["plain_s"], 1e-9)
              for r in rows]
    print(f"\nn fits              : {len(rows)} (bar needs >= 30)")
    print(f"BAR 1 headroom      : oracle median {np.median(o_all):+.3f}% "
          f"(mean {np.mean(o_all):+.3f}%)  [bar: >= +0.3%]")
    print(f"BAR 2 fidelity      : probe median {np.median(p_all):+.3f}% "
          f"(mean {np.mean(p_all):+.3f}%); paired probe-minus-oracle median "
          f"{np.median(fid):+.3f}%")
    print(f"BAR 3 cost          : median aug/plain fit ratio "
          f"{np.median(fx_all):.2f} (mean {np.mean(fx_all):.2f})  [bar: <= 1.5]")
    print("(positive = cross arm better than plain rung 1; test RMSE; fitx "
          "includes the 25-round probe)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
