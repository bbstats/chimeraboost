"""F2 S1 probe (CAMPAIGN_PLAN.md I002/I016): can a CV-averaged race referee
cross features below the CROSS_MIN_SAMPLES gate?

HYPOTHESIS (I002, re-scoped at S0): below 2000 rows the single validation
split is too small for a trustworthy cross-feature race (B1's mechanism), but
sub-gate fits are cheap, so averaging the race over CV folds may repair the
signal. B14 closed budget/decision rules, not signal quality, so this axis is
open. eucalyptus (441 rows, 5 classes) is the first target: the biggest hc
CatBoost gap, out of M1's reach.

PROTOCOL: paired same-split A/B, 3 seeds, inner MulticlassBoosting at the
production multiclass config (depth 6, ES 50 on a 0.2 inner split, auto LR
fade, size-adaptive mcw resolved per fit). Pairs are the production
_cross_candidate_pairs from the full base fit's importances -- the race judges
a FIXED pair set, isolating the SIGNAL question from pair fidelity.
  plain  : no cross columns (what ships today below the gate)
  cvrace : 3-fold CV-averaged val-logloss race over the fixed pair set; the
           winner refit on the full train rows
  single : shipped single-split race over the same pair set (comparison: does
           CV change the decision, and which agrees with test?)
  oracle : better of plain/augmented on TEST (the ceiling)
Metric: multiclass Brier on the held-out test split (the decision metric);
race criterion is val logloss (production). Probabilities are uncalibrated
(T=1); temperature would apply equally to all arms.

PANEL: eucalyptus (gap), cjs (canary: saturated -- CV must not invent
signal), okcupid-stem (control: above gate, CV and single-split should
agree, validating the instrument).

KILL BARS (pre-registered in CAMPAIGN_PLAN.md I016 BEFORE this runs):
  0 headroom: oracle-over-plain test-Brier gain >= +0.2% mean on gap fits.
     Below that there is no prize and F2 dies as "nothing to win".
  1 signal  : cvrace beats plain on >= 2/3 gap fits with positive mean gain.
  2 canary  : no CV-picked augmented loss beyond -0.1% on any cjs fit.
  3 control : CV pick == single pick on >= 2/3 okcupid fits.
Bar 0 or 1 fails -> KILL F2 with the mechanism recorded.

Results: benchmarks/results/probe-subgate-race.jsonl (resumable, delete to
rerun); aggregate table printed at the end.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import (ShuffleSplit, StratifiedKFold,
                                     train_test_split)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata  # noqa: E402

from chimeraboost.booster import MulticlassBoosting  # noqa: E402
from chimeraboost.sklearn_api import (  # noqa: E402
    _auto_min_child_weight,
    _best_val,
    _cross_candidate_pairs,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-subgate-race.jsonl")
SEEDS = (0, 1, 2)
N_FOLDS = 3
THREADS = None  # all cores (run this probe with nothing else heavy going)

PANEL = [
    ("hc:eucalyptus", "gap"),
    ("hc:cjs", "canary"),
    ("hc:okcupid-stem", "control"),
]

CFG = dict(n_estimators=2000, early_stopping_rounds=50, depth=6,
           random_state=0)


def _fit(Xf, yf, ev, cat, pairs):
    kw = dict(CFG, cross_pairs=pairs or None,
              min_child_weight=_auto_min_child_weight(len(yf)))
    m = MulticlassBoosting(**kw)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat, eval_set=ev)
    return m, time.time() - t


def _brier(m, Xte, yte, classes):
    """Multiclass Brier vs one-hot, aligned to the global class set."""
    P = np.asarray(m.loss_.transform(m.predict_raw(Xte)), dtype=float)
    order = np.searchsorted(np.asarray(m.classes_), np.asarray(classes))
    Y = np.zeros_like(P)
    Y[np.arange(len(yte)), np.searchsorted(np.asarray(classes),
                                           np.asarray(yte))] = 1.0
    return float(np.mean(np.sum((P[:, order] - Y) ** 2, axis=1)))


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
        classes = np.unique(np.asarray(y))
        assert classes.size > 2, (key, "not multiclass")
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y)
            # Production inner ES split: 0.2 held out of the train rows.
            tr_idx, va_idx = next(ShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed).split(Xtr))
            Xf, yf = Xtr[tr_idx], ytr[tr_idx]
            ev = (Xtr[va_idx], ytr[va_idx])

            base, base_s = _fit(Xf, yf, ev, cat, None)
            imp = np.asarray(base.feature_importances_, dtype=float)
            pairs = _cross_candidate_pairs(imp, cat, X.shape[1])
            plain_b = _brier(base, Xte, yte, classes)

            row = {"dataset": key, "seed": seed, "group": group,
                   "n_train": int(len(yf)), "n_pairs": len(pairs),
                   "plain": plain_b, "plain_s": round(base_s, 2)}
            if not pairs:
                row.update(cv_pick="nopairs", single_pick="nopairs",
                           oracle_pick="plain", cvrace=plain_b,
                           single=plain_b, oracle=plain_b)
                _emit(row)
                continue

            # CV-averaged race over the fixed pair set.
            fold_deltas = []
            cv_s = 0.0
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)
            for ftr, fva in skf.split(Xtr, ytr):
                fev = (Xtr[fva], ytr[fva])
                mp, sp = _fit(Xtr[ftr], ytr[ftr], fev, cat, None)
                ma, sa = _fit(Xtr[ftr], ytr[ftr], fev, cat, pairs)
                fold_deltas.append(_best_val(mp) - _best_val(ma))
                cv_s += sp + sa
            cv_delta = float(np.mean(fold_deltas))
            cv_pick = "aug" if cv_delta > 0 else "plain"

            # Shipped single-split race over the same pair set.
            aug, aug_s = _fit(Xf, yf, ev, cat, pairs)
            single_delta = _best_val(base) - _best_val(aug)
            single_pick = "aug" if single_delta > 0 else "plain"
            aug_b = _brier(aug, Xte, yte, classes)

            row.update(
                cv_delta=round(cv_delta, 6),
                single_delta=round(float(single_delta), 6),
                cv_pick=cv_pick, single_pick=single_pick,
                oracle_pick="aug" if aug_b < plain_b else "plain",
                # Metrics at FULL precision: the canary's Brier runs ~1e-8,
                # where rounding to 8 decimals destroys the comparison.
                aug=aug_b,
                cvrace=aug_b if cv_pick == "aug" else plain_b,
                single=aug_b if single_pick == "aug" else plain_b,
                oracle=min(aug_b, plain_b),
                aug_s=round(aug_s, 2), cv_s=round(cv_s, 2))
            _emit(row)
    table()


def _emit(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    base, aug = row["plain"], row.get("aug", row["plain"])
    if base < 1e-6:
        # Saturated set: percents are noise over ~0, read absolute deltas.
        d = lambda v: base - v  # tiny local formatter, ungated dir
        print(f"{row['dataset']} s{row['seed']}: base={base:.3g} "
              f"aug={d(aug):+.2g} cv[{row['cv_pick']}]={d(row['cvrace']):+.2g} "
              f"1split[{row['single_pick']}]={d(row['single']):+.2g} (abs) "
              f"pairs={row['n_pairs']}", flush=True)
        return
    g = lambda v: 100 * (base - v) / base  # tiny local formatter, ungated dir
    print(f"{row['dataset']} s{row['seed']}: base={base:.5g} "
          f"aug={g(aug):+.2f}% cv[{row['cv_pick']}]={g(row['cvrace']):+.2f}% "
          f"1split[{row['single_pick']}]={g(row['single']):+.2f}% "
          f"pairs={row['n_pairs']}", flush=True)


def _pct(base, v):
    return 100 * (base - v) / base if base > 1e-12 else 0.0


def table():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print(f"\n{'dataset':28} {'grp':7} {'cv%':>8} {'1split%':>8} "
          f"{'oracle%':>8} {'cv=or':>5} {'1s=or':>5}")
    agg = {}
    for ds, rs in by_ds.items():
        sat = np.mean([r["plain"] for r in rs]) < 1e-6
        cv = np.mean([_pct(r["plain"], r["cvrace"]) for r in rs])
        sg = np.mean([_pct(r["plain"], r["single"]) for r in rs])
        oc = np.mean([_pct(r["plain"], r["oracle"]) for r in rs])
        cvo = sum(r["cv_pick"] == r["oracle_pick"] for r in rs)
        sgo = sum(r["single_pick"] == r["oracle_pick"] for r in rs)
        grp = rs[0]["group"]
        agg.setdefault(grp, []).append((cv, sg, oc, sat))
        if sat:
            abd = np.mean([r["plain"] - r["cvrace"] for r in rs])
            print(f"{ds:28} {grp:7} {'sat':>8} {'sat':>8} {'sat':>8} "
                  f"{cvo}/{len(rs)} {sgo}/{len(rs)} (abs {abd:+.2g})")
        else:
            print(f"{ds:28} {grp:7} {cv:+8.2f} {sg:+8.2f} {oc:+8.2f} "
                  f"{cvo}/{len(rs)} {sgo}/{len(rs)}")
    print()
    for grp, vals in agg.items():
        live = [v for v in vals if not v[3]]
        if not live:
            print(f"  {grp:8} saturated — read the absolute row above")
            continue
        cv = np.mean([v[0] for v in live])
        sg = np.mean([v[1] for v in live])
        oc = np.mean([v[2] for v in live])
        print(f"  {grp:8} cv {cv:+.2f}%  1split {sg:+.2f}%  oracle {oc:+.2f}%")
    print("\n(positive = better test Brier than plain; cv=or / 1s=or count "
          "race-oracle pick agreement)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
