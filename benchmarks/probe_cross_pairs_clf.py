"""F3 S1 probe (CAMPAIGN_PLAN.md I019): can the rung-1 CLASSIFIER take cross
features without the race, the way the regressor does since E2?

The regressor's `cross_features="always"` (SELECT_PLAN.md E2) replaced the
validation race with a 25-round importance probe and a forced top-4 block.
The classifier still pins cross features OFF at rung 1 because no forced
counterpart was ever measured. This is that measurement, zero library change,
the E2 step-1 design transplanted to binary log loss.

Three arms per (dataset, seed), all the inner GradientBoosting at the
rung-1 classifier config (Logloss, linear_leaves=True -- the classifier's
auto rule for binary -- n_estimators=2000, ES 50 on a 0.2 validation split,
auto lr, depth 6):

  plain  : cross_pairs=[]                                        (rung 1 today)
  probe  : cross_pairs from a 25-round probe fit's importances    (the E2 design)
  oracle : cross_pairs from the plain FULL fit's importances      (the ceiling)

Both cross arms use the production `_cross_candidate_pairs` at the SHIPPED
forced width (`FORCED_CROSS_TOP_M` = 4), so the block is exactly what a
classifier "always" mode would carry. Metric is test Brier (the mandatory
classification read), log loss reported beside it.

Kill bars (registered in CAMPAIGN_PLAN.md I019 BEFORE this ran), in order:
  1 headroom : oracle-over-plain test-Brier gain >= +0.3% median on >= 30 fits
  2 fidelity : paired probe-minus-oracle median >= -0.1% (probe holds the
               oracle's headroom within noise)
  3 cost     : median augmented/plain fit-time ratio <= 1.5 (probe included)

Panel: every Grinsztajn clf_num set (16; clf_cat are the same sets with
numerically encoded extras and would double-count). No gap/control labels
exist for the classifier -- the oracle column IS the per-set headroom read.
Results: benchmarks/results/probe-cross-pairs-f3-clf.jsonl (resumable, delete
to rerun); aggregate table printed at the end.

Run: python benchmarks/probe_cross_pairs_clf.py [--table-only]
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
from chimeraboost.sklearn_api import (FORCED_CROSS_TOP_M,  # noqa: E402
                                      _cross_candidate_pairs)

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results",
    "probe-cross-pairs-f3-clf.jsonl")
SEEDS = (0, 1, 2)
PROBE_ROUNDS = 25

PANEL = [
    "gr:clf_num/Bioresponse",
    "gr:clf_num/Diabetes130US",
    "gr:clf_num/Higgs",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/MiniBooNE",
    "gr:clf_num/bank-marketing",
    "gr:clf_num/california",
    "gr:clf_num/covertype",
    "gr:clf_num/credit",
    "gr:clf_num/default-of-credit-card-clients",
    "gr:clf_num/electricity",
    "gr:clf_num/eye_movements",
    "gr:clf_num/heloc",
    "gr:clf_num/house_16H",
    "gr:clf_num/jannis",
    "gr:clf_num/pol",
]

CFG = dict(loss="Logloss", n_estimators=2000, early_stopping_rounds=50,
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


def _scores(m, Xte, yte):
    p = 1.0 / (1.0 + np.exp(-m.predict_raw(Xte)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - yte) ** 2))
    ll = float(-np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p)))
    return brier, ll


def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["seed"]))
    return done


def _pairs(model, cat, n_features):
    return _cross_candidate_pairs(
        np.asarray(model.feature_importances_, dtype=float), cat, n_features,
        top_m=FORCED_CROSS_TOP_M)


def main():
    done = _done_keys()
    for key in PANEL:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        classes = np.unique(y)
        assert task != "regression" and len(classes) == 2, (key, task)
        y = (y == classes[1]).astype(np.float64)
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y)
            tr_idx, va_idx = next(ShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed).split(Xtr))
            Xf, yf = Xtr[tr_idx], ytr[tr_idx]
            ev = (Xtr[va_idx], ytr[va_idx])

            plain, plain_s = _fit(Xf, yf, ev, cat, None)
            base, base_ll = _scores(plain, Xte, yte)

            pfit, probe_s = _fit(Xf, yf, ev, cat, None,
                                 n_estimators=PROBE_ROUNDS, es=False)
            p_pairs = _pairs(pfit, cat, X.shape[1])
            o_pairs = _pairs(plain, cat, X.shape[1])

            probe_m, probe_fit_s = _fit(Xf, yf, ev, cat, p_pairs)
            oracle_m, oracle_fit_s = _fit(Xf, yf, ev, cat, o_pairs)
            probe_b, probe_ll = _scores(probe_m, Xte, yte)
            oracle_b, oracle_ll = _scores(oracle_m, Xte, yte)

            inter = len(set(p_pairs) & set(o_pairs))
            union = len(set(p_pairs) | set(o_pairs))
            row = {"dataset": key, "seed": seed,
                   "n_train": int(len(yf)), "n_pairs": len(o_pairs),
                   "pair_jaccard": round(inter / union, 3) if union else 1.0,
                   "base": base, "probe": probe_b, "oracle": oracle_b,
                   "base_ll": base_ll, "probe_ll": probe_ll,
                   "oracle_ll": oracle_ll,
                   "plain_s": round(plain_s, 2),
                   "probe_fit_s": round(probe_fit_s, 2),
                   "oracle_fit_s": round(oracle_fit_s, 2),
                   "probe_s": round(probe_s, 2)}
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{key} s{seed}: brier={base:.5g} "
                  f"probe={100 * (base - probe_b) / base:+.2f}% "
                  f"oracle={100 * (base - oracle_b) / base:+.2f}% "
                  f"jac={row['pair_jaccard']:.2f} "
                  f"fitx={(probe_fit_s + probe_s) / max(plain_s, 1e-9):.2f}",
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

    print(f"\n{'dataset':44} {'probe%':>8} {'oracle%':>8} {'ll_or%':>7} "
          f"{'jac':>5} {'fitx':>5} {'probeshare':>10}")
    for ds, rs in sorted(by_ds.items()):
        p = np.mean([100 * (r["base"] - r["probe"]) / r["base"] for r in rs])
        o = np.mean([100 * (r["base"] - r["oracle"]) / r["base"] for r in rs])
        ol = np.mean([100 * (r["base_ll"] - r["oracle_ll"]) / r["base_ll"]
                      for r in rs])
        j = np.mean([r["pair_jaccard"] for r in rs])
        fx = np.mean([(r["probe_fit_s"] + r["probe_s"]) / max(r["plain_s"], 1e-9)
                      for r in rs])
        sh = np.mean([r["probe_s"] / max(r["probe_fit_s"] + r["probe_s"], 1e-9)
                      for r in rs])
        print(f"{ds:44} {p:+8.2f} {o:+8.2f} {ol:+7.2f} {j:5.2f} "
              f"{fx:5.2f} {sh:10.2f}")

    o_all = [100 * (r["base"] - r["oracle"]) / r["base"] for r in rows]
    p_all = [100 * (r["base"] - r["probe"]) / r["base"] for r in rows]
    fid = [p - o for p, o in zip(p_all, o_all)]
    fx_all = [(r["probe_fit_s"] + r["probe_s"]) / max(r["plain_s"], 1e-9)
              for r in rows]
    o_w = sum(1 for v in o_all if v > 0)
    o_l = sum(1 for v in o_all if v < 0)
    print(f"\nn fits              : {len(rows)} (bar needs >= 30)")
    print(f"BAR 1 headroom      : oracle median {np.median(o_all):+.3f}% "
          f"(mean {np.mean(o_all):+.3f}%; {o_w}W-{o_l}L)  [bar: >= +0.3%]")
    print(f"BAR 2 fidelity      : probe median {np.median(p_all):+.3f}% "
          f"(mean {np.mean(p_all):+.3f}%); paired probe-minus-oracle median "
          f"{np.median(fid):+.3f}%  [bar: >= -0.1%]")
    print(f"BAR 3 cost          : median aug/plain fit ratio "
          f"{np.median(fx_all):.2f} (mean {np.mean(fx_all):.2f})  [bar: <= 1.5]")
    print("(positive = cross arm better than plain rung 1; test Brier; fitx "
          "includes the 25-round probe; ll_or% = oracle log-loss gain)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
