"""Probe C5: is the lower rate's temporal regression real, and is it fixable?

PRE-REGISTERED 2026-08-01, before any results (benchmarks/SMALLDATA_PLAN.md
"C5"). Tier-2 withheld the default flip for `adaptive_learning_rate` on the
strength of FOUR datasets in hc:time. This probe tests that regression.

WHY NOT JUST RUN MORE SEEDS. `_temporal_split` takes
TEMPORAL_CUTS[seed % 3] with TEMPORAL_CUTS = (0.65, 0.70, 0.75) and no other
seed-dependent randomness, so seed 3 reproduces seed 0 EXACTLY. The hc:time
universe is 7 datasets x 3 cuts and nothing else; extra seeds would return
identical numbers wearing the costume of more evidence.

HYPOTHESIS. A lower rate buys its gain with more rounds; under drift those
extra rounds fit a stale past more tightly. And our early-stopping holdout is a
RANDOM slice of the training rows -- drawn from the past as well -- so early
stopping watches it improve and keeps going, blind to the fact that fitting the
past harder has stopped paying. If that is the mechanism, validating on the
most RECENT slice should see the drift and stop earlier.

DESIGN (2 rates x 3 holdouts, exactly paired per window):
  rate    in {0.1 (shipped), 0.07 (the knee)}
  holdout in {auto (shipped path: internal random split, refit_full ON),
              rand (explicit random 20%, refit OFF),
              tail (explicit LAST 20% chronologically, refit OFF)}
`tail` vs `rand` isolates the holdout's COMPOSITION -- both have the refit off,
so the refit cannot confound the comparison. Fixed rates, not the fade: the
mechanism question is about the rate, and the fade is only a schedule over it.

6 rolling-origin cuts instead of the suite's 3, so "does it reproduce" gets
twice the windows from the same sources.

PREDICTIONS:
  right      -> 0.07 loses under auto/rand, shrinks or reverses under tail,
                and runs FEWER rounds under tail than under rand.
  noise      -> auto@0.07 vs auto@0.1 is a wash over the 42 windows.
  unfixable  -> 0.07 loses under every holdout type.

CAVEATS THE OUTPUT STATES RATHER THAN HIDES:
  * rows capped at 30,000, which pulls the three largest sets (kick,
    sf-police-incidents, Traffic_violations) into the size range where the rate
    matters; in tier-2 they sat above the fade threshold and tied.
  * 7 datasets is the whole universe, so 42 windows rest on 7 independent
    sources. Windows from one dataset overlap heavily and are NOT independent;
    the per-dataset block is printed for that reason.
  * `rounds` for the auto arms is the post-refit budget, not the raw ES
    optimum; the rand/tail arms have the refit off so theirs are raw.

Primary metric: RMSE (regression) / Brier (binary and multiclass).
Resumable JSONL; `--table-only` reprints.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb                     # noqa: E402

import chimeraboost                             # noqa: E402
from chimeraboost import (ChimeraBoostRegressor,  # noqa: E402
                          ChimeraBoostClassifier)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "probe-temporal-lr.jsonl")
CUTS = (0.45, 0.55, 0.65, 0.70, 0.75, 0.82)
RATES = (0.1, 0.07)
HOLDOUTS = ("auto", "rand", "tail")
BASE_RATE = "0.1"
MAX_ROWS = 30_000
VAL_FRAC = 0.2
TIE_BAND = 1e-9


def _arm(rate, holdout):
    return f"{holdout}@{rate:g}"


ARMS = [_arm(r, h) for h in HOLDOUTS for r in RATES]


def _done():
    seen = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        seen.add((r["dataset"], r["cut"]))
                    except Exception:
                        pass
    return seen


def _score(task, yte, model, Xte):
    if task == "regression":
        return float(np.sqrt(np.mean((yte - model.predict(Xte)) ** 2)))
    P = model.predict_proba(Xte)
    classes = list(model.classes_)
    Y = np.zeros_like(P)
    for i, v in enumerate(yte):
        Y[i, classes.index(v)] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def _window(X, y, cut, task):
    """One rolling origin, mirroring rb._temporal_split's semantics."""
    n = len(y)
    i = int(n * cut)
    j = min(n, i + int(n * rb.TEMPORAL_TEST_FRAC))
    if i < 20 or j - i < 5:
        return None
    Xtr, ytr, Xte, yte = X[:i], y[:i], X[i:j], y[i:j]
    if task != "regression":
        seen = np.unique(ytr)
        if len(seen) < 2:
            return None
        keep = np.isin(yte, seen)
        if keep.sum() < 5:
            return None
        Xte, yte = Xte[keep], yte[keep]
    return Xtr, ytr, Xte, yte


def _fit(task, rate, holdout, Xtr, ytr, Xte, cat):
    Est = (ChimeraBoostRegressor if task == "regression"
           else ChimeraBoostClassifier)
    common = dict(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
                  learning_rate=rate, random_state=0)
    if holdout == "auto":
        m = Est(**common)
        m.fit(Xtr, ytr, cat_features=cat)
    else:
        n = len(ytr)
        k = max(5, int(n * VAL_FRAC))
        if holdout == "tail":
            # chronological: the most RECENT rows are the validation set
            idx_tr, idx_va = np.arange(n - k), np.arange(n - k, n)
        else:
            rng = np.random.default_rng(0)
            perm = rng.permutation(n)
            idx_va, idx_tr = perm[:k], perm[k:]
        if task != "regression":
            # an explicit holdout must not hide a class from training
            if len(np.unique(ytr[idx_tr])) < len(np.unique(ytr)):
                return None, None
        m = Est(**common)
        m.fit(Xtr[idx_tr], ytr[idx_tr], cat_features=cat,
              eval_set=(Xtr[idx_va], ytr[idx_va]))
    return m, int(m.best_iteration_)


def main():
    print(f"chimeraboost from {chimeraboost.__file__}")
    rb._add_highcard_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    keys = [f"{k}@time" for k in rb.TEMPORAL_COLUMNS if k.startswith("hc:")]
    done = _done()
    try:
        for key in keys:
            if key not in rb.DATASETS:
                print(f"  [skip] {key}: not registered", flush=True)
                continue
            try:
                X, y, cat, task = rb.DATASETS[key](1.0, np.random.default_rng(0))
                if len(y) > MAX_ROWS:            # keep the LAST rows: the cap
                    X, y = X[-MAX_ROWS:], y[-MAX_ROWS:]   # must not break order
                print(f"\n=== {key}  n={len(y)}  p={X.shape[1]}  {task}",
                      flush=True)
                for cut in CUTS:
                    if (key, cut) in done:
                        continue
                    _run_cell(key, task, cut, X, y, cat)
            except Exception as exc:
                print(f"  [skip] {key}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        table()


def _run_cell(key, task, cut, X, y, cat):
    w = _window(X, y, cut, task)
    if w is None:
        print(f"  cut {cut:4.2f}: degenerate window, skipped", flush=True)
        return
    Xtr, ytr, Xte, yte = w
    rec = {"dataset": key, "task": task, "cut": cut,
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "score": {}, "rounds": {}, "fit_s": {}}
    line = f"  cut {cut:4.2f} n={len(ytr):>6d} "
    for h in HOLDOUTS:
        for r in RATES:
            a = _arm(r, h)
            t0 = time.time()
            m, rounds = _fit(task, r, h, Xtr, ytr, Xte, cat)
            if m is None:
                continue
            rec["fit_s"][a] = time.time() - t0
            rec["score"][a] = _score(task, yte, m, Xte)
            rec["rounds"][a] = rounds
            line += f" | {a}: {rec['score'][a]:.5g}"
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def _rows():
    if not os.path.exists(RESULTS):
        return []
    out = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _gain(rec, h):
    """% improvement of 0.07 over 0.1 within holdout h (positive = 0.07 wins)."""
    lo, hi = _arm(0.07, h), _arm(0.1, h)
    if lo not in rec["score"] or hi not in rec["score"]:
        return None
    base = rec["score"][hi]
    if base <= 0:
        return None
    return 100.0 * (base - rec["score"][lo]) / base


def table():
    rows = _rows()
    if not rows:
        print("no results yet")
        return

    print("\n" + "=" * 108)
    print("C5 PROBE - does the lower rate really lose under temporal drift, and "
          "does a tail ES split fix it?")
    print("  Every number is the % gain of lr=0.07 over lr=0.1 WITHIN one "
          "holdout type (positive = 0.07 wins).")
    print(f"  Rows capped at {MAX_ROWS:,}. {len(CUTS)} rolling origins vs the "
          "suite's 3.")
    print("  NOTE: windows from one dataset overlap heavily and are NOT "
          "independent -- read the per-dataset")
    print("  block below the aggregate; 7 sources is the entire hc:time "
          "universe.")
    print("=" * 108)

    # ---- Block A: per-window --------------------------------------------
    print(f"\n{'dataset':34s}{'cut':>6s}"
          + "".join(f"{h:>12s}" for h in HOLDOUTS))
    for r in sorted(rows, key=lambda r: (r["dataset"], r["cut"])):
        line = f"{r['dataset'][:34]:34s}{r['cut']:>6.2f}"
        for h in HOLDOUTS:
            g = _gain(r, h)
            line += f"{g:+11.3f}%" if g is not None else f"{'--':>12s}"
        print(line)

    # ---- Block B: aggregate per holdout ----------------------------------
    print("\n" + "=" * 108)
    print("VERDICT - lr=0.07 vs lr=0.1, within each holdout type")
    print("=" * 108)
    agg = {}
    for h in HOLDOUTS:
        gains = [g for g in (_gain(r, h) for r in rows) if g is not None]
        if not gains:
            continue
        w = sum(1 for g in gains if g > TIE_BAND)
        l = sum(1 for g in gains if g < -TIE_BAND)
        agg[h] = (np.median(gains), np.mean(gains), w, l, len(gains) - w - l)
        med, mean, w, l, t = agg[h]
        print(f"  {h:>5s} holdout   median {med:+7.3f}%   mean {mean:+7.3f}%   "
              f"{w:>2d}W-{l:>2d}L-{t:>2d}T   p={_sign_p(w, l):.3f}")

    if "rand" in agg and "tail" in agg:
        print(f"\n  Does the tail holdout rescue the lower rate?  "
              f"rand median {agg['rand'][0]:+.3f}%  ->  "
              f"tail median {agg['tail'][0]:+.3f}%")

    # ---- Block C: per dataset (the honest denominator) -------------------
    print("\n" + "=" * 108)
    print("PER DATASET - median gain of 0.07 over 0.1 across that dataset's "
          "windows. 7 sources, so this")
    print("  is the denominator that matters; the window count is not an "
          "independent n.")
    print("=" * 108)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    print(f"{'dataset':34s}{'windows':>9s}"
          + "".join(f"{h:>12s}" for h in HOLDOUTS))
    for ds in sorted(by_ds):
        line = f"{ds[:34]:34s}{len(by_ds[ds]):>9d}"
        for h in HOLDOUTS:
            gs = [g for g in (_gain(r, h) for r in by_ds[ds]) if g is not None]
            line += f"{np.median(gs):+11.3f}%" if gs else f"{'--':>12s}"
        print(line)

    # ---- Block D: the mechanism, tested rather than asserted -------------
    print("\n" + "=" * 108)
    print("MECHANISM - the claim is 'the lower rate loses because it runs MORE "
          "rounds against a stale past'.")
    print("  (a) does the tail holdout make 0.07 stop EARLIER than the random "
          "one? (b) across windows, does")
    print("  a bigger round increase go with a bigger loss? Both are "
          "predictions, so both are checkable.")
    print("=" * 108)
    for h in HOLDOUTS:
        rr = [r["rounds"][_arm(0.07, h)] / max(r["rounds"][_arm(0.1, h)], 1)
              for r in rows
              if _arm(0.07, h) in r["rounds"] and _arm(0.1, h) in r["rounds"]]
        if rr:
            print(f"  {h:>5s}: rounds(0.07)/rounds(0.1) median {np.median(rr):.2f}x")
    for h in HOLDOUTS:
        pairs = [(r["rounds"][_arm(0.07, h)] / max(r["rounds"][_arm(0.1, h)], 1),
                  _gain(r, h)) for r in rows
                 if _arm(0.07, h) in r["rounds"] and _arm(0.1, h) in r["rounds"]
                 and _gain(r, h) is not None]
        if len(pairs) > 3:
            x = np.array([p[0] for p in pairs])
            g = np.array([p[1] for p in pairs])
            if x.std() > 1e-9 and g.std() > 1e-9:
                print(f"  {h:>5s}: corr(round ratio, gain) = "
                      f"{np.corrcoef(x, g)[0, 1]:+.3f}  (mechanism predicts "
                      f"NEGATIVE: more extra rounds -> bigger loss)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
