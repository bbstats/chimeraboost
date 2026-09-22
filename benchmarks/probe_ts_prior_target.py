"""Probe R3 Stage C (CAMPAIGN_PLAN I034): is the hc Brier gap in the ENCODER or
in the BOOSTER, and does CatBoost's shrinkage target transfer to ours?

Stages A and B (`probe_catboost_hc_ablation.py`, I033) localized CatBoost's
high-cardinality edge to its Borders target CTR and, inside it, to the
shrinkage target: `(sum + 0.5) / (count + 1)` against our
`(sum + mean * a) / (count + a)` with a = 1. Two questions remain, and each
has a zero-library-change arm on the same seven sets and the same paired
splits as I033 (so the `chimera` and `cb_default` rows of that JSONL are the
reference lines here):

  chimera_prior05   our default estimator with the ordered-TS prior held at
                    the constant 0.5 instead of the target mean (CatBoost's
                    target), same weight a = 1
  chimera_prior0    the same with the constant 0 -- CatBoost's zero-prior arm
                    lost 21-74% of its edge to this; if ours loses too, the
                    two encoders respond alike and the target is not the gap
  cb_our_ts         CatBoost at the harness defaults with NO cat_features: the
                    categorical columns are replaced by OUR ordered target
                    statistics (fit on its training carve, full-total
                    transform for its validation carve and the test rows, one
                    column per class target, unseen categories at the prior),
                    so it boosts on exactly the encoding we boost on. If it
                    still beats us, the edge is in the booster, not the
                    encoder; if it falls to our Brier, the edge is the encoder.

The prior arms monkeypatch `OrderedTargetEncoder.fit_transform` with a copy
of its body whose prior is the constant; `transform` reads `prior_` and
follows. Brier in the harness's K-sum form. Resumable JSONL; table at the end.

Run:
    python benchmarks/probe_ts_prior_target.py
    python benchmarks/probe_ts_prior_target.py --table-only
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research import datasets as rdata

import chimeraboost.target_encoding as te
from chimeraboost import ChimeraBoostClassifier
from chimeraboost.preprocessing import as_model_array
from chimeraboost.target_encoding import OrderedTargetEncoder, factorize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-ts-prior-target.jsonl")
REFERENCE = os.path.join(HERE, "results", "probe-cb-hc-ablation.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50

GAP = ["hc:sf-police-incidents", "hc:Traffic_violations", "hc:kick",
       "hc:okcupid-stem", "hc:porto-seguro"]
CONTROL = ["hc:kdd_ipums_la_97-small", "hc:eucalyptus"]
ALL = GAP + CONTROL
ARMS = ["chimera_prior05", "chimera_prior0", "cb_our_ts"]
REF_ARMS = ["chimera", "cb_default"]


def _brier(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# ---- the prior monkeypatch ---------------------------------------------------
_ORIG_FIT_TRANSFORM = OrderedTargetEncoder.fit_transform


def _fit_transform_with_prior(prior_value):
    """`OrderedTargetEncoder.fit_transform` with `prior_` pinned to a constant.
    The body is the library's, with one line changed."""
    def fit_transform(self, codes_matrix, y, sample_weight=None):
        codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        w = None if sample_weight is None else np.ascontiguousarray(
            sample_weight, dtype=np.float64)
        n_samples, n_cols = codes_matrix.shape
        rng = np.random.default_rng(self.random_state)
        self.prior_ = float(prior_value)            # <-- the one change
        self.sums_, self.counts_, self.n_cat_ = [], [], []
        out = np.zeros((n_samples, n_cols), dtype=np.float64)
        for j in range(n_cols):
            codes = np.ascontiguousarray(codes_matrix[:, j])
            n_cat = int(codes.max()) + 1 if codes.size else 1
            acc = np.zeros(n_samples, dtype=np.float64)
            sums = counts = None
            for _ in range(self.n_permutations):
                perm = rng.permutation(n_samples)
                if w is None:
                    enc, sums, counts = te._ordered_ts(
                        codes, y, perm, n_cat, self.prior_, self.smoothing)
                else:
                    enc, sums, counts = te._ordered_ts_weighted(
                        codes, y, w, perm, n_cat, self.prior_, self.smoothing)
                acc += enc
            out[:, j] = acc / self.n_permutations
            self.sums_.append(sums)
            self.counts_.append(counts)
            self.n_cat_.append(n_cat)
        return out
    return fit_transform


def _run_chimera(Xtr, ytr, Xte, yte, cat, seed, prior_value):
    OrderedTargetEncoder.fit_transform = _fit_transform_with_prior(prior_value)
    try:
        t = time.time()
        m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                                   early_stopping_rounds=PATIENCE, random_state=seed)
        m.fit(Xtr, ytr, cat_features=cat)
        fit_s = time.time() - t
        return _brier(yte, m.predict_proba(Xte), m.classes_), fit_s
    finally:
        OrderedTargetEncoder.fit_transform = _ORIG_FIT_TRANSFORM


# ---- CatBoost on our encoding ------------------------------------------------
def _our_ts_matrices(Xa, ya, Xv, Xte, cat, seed):
    """Replace the categorical columns by our ordered target statistics.

    Fit on the training carve ``Xa`` (ordered TS, 4 permutations averaged, the
    library defaults), full-total transform for ``Xv`` and ``Xte``; one column
    per class target for K > 2, as the library does; unseen categories map to
    -1 and land at the prior. Numeric columns pass through as float64."""
    classes = np.unique(ya)
    targets = ([(ya == classes[1]).astype(np.float64)] if len(classes) == 2
               else [(ya == c).astype(np.float64) for c in classes])
    cat = list(cat)
    num = [f for f in range(Xa.shape[1]) if f not in cat]
    A = as_model_array(Xa, True)
    V = as_model_array(Xv, True)
    T = as_model_array(Xte, True)

    def _num(M):
        return (np.asarray(M[:, num], dtype=np.float64) if num
                else np.empty((M.shape[0], 0)))

    codes_a = np.empty((A.shape[0], len(cat)), dtype=np.int64)
    codes_v = np.empty((V.shape[0], len(cat)), dtype=np.int64)
    codes_t = np.empty((T.shape[0], len(cat)), dtype=np.int64)
    for j, f in enumerate(cat):
        c, cats = factorize(A[:, f])
        lookup = {v: i for i, v in enumerate(cats)}
        codes_a[:, j] = c
        codes_v[:, j] = [lookup.get(v, -1) for v in V[:, f]]
        codes_t[:, j] = [lookup.get(v, -1) for v in T[:, f]]

    blocks_a, blocks_v, blocks_t = [_num(A)], [_num(V)], [_num(T)]
    for t_idx, target in enumerate(targets):
        enc = OrderedTargetEncoder(smoothing=1.0, random_state=seed + t_idx,
                                   n_permutations=4)
        blocks_a.append(enc.fit_transform(codes_a, target))
        blocks_v.append(enc.transform(codes_v))
        blocks_t.append(enc.transform(codes_t))
    return (np.hstack(blocks_a), np.hstack(blocks_v), np.hstack(blocks_t))


def _run_cb_our_ts(Xtr, ytr, Xte, yte, cat, seed):
    from catboost import CatBoostClassifier
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=seed, stratify=ytr)
    t = time.time()
    Fa, Fv, Ft = _our_ts_matrices(Xa, ya, Xv, Xte, cat, seed)
    m = CatBoostClassifier(iterations=MAX_ITERS, random_seed=seed, verbose=False,
                           thread_count=-1, allow_writing_files=False,
                           early_stopping_rounds=PATIENCE)
    m.fit(Fa, ya, eval_set=(Fv, yv))
    fit_s = time.time() - t
    return _brier(yte, m.predict_proba(Ft), m.classes_), fit_s, int(m.get_best_iteration() or 0)


# ---- bookkeeping --------------------------------------------------------------
def _done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["dataset"], r["arm"], r["seed"]))
    return done


def _append(rec):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    done = _done_keys()
    for key in ALL:
        X, y, cat, task = rdata.load(key)
        cat = list(cat or [])
        print(f"\n=== {key}  task={task}  n={len(y)}  p={X.shape[1]}  cats={len(cat)}",
              flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            for arm, prior in (("chimera_prior05", 0.5), ("chimera_prior0", 0.0)):
                if (key, arm, seed) in done:
                    continue
                b, fit_s = _run_chimera(Xtr, ytr, Xte, yte, cat, seed, prior)
                _append({"dataset": key, "arm": arm, "seed": seed, "task": task,
                         "brier": b, "fit_time": fit_s})
                print(f"  [{arm:15s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s", flush=True)
            if (key, "cb_our_ts", seed) not in done:
                b, fit_s, bi = _run_cb_our_ts(Xtr, ytr, Xte, yte, cat, seed)
                _append({"dataset": key, "arm": "cb_our_ts", "seed": seed, "task": task,
                         "brier": b, "fit_time": fit_s, "best_iter": bi})
                print(f"  [{'cb_our_ts':15s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s  it={bi}",
                      flush=True)
    table()


def table():
    agg = defaultdict(list)
    for path in (REFERENCE, RESULTS):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["arm"] in REF_ARMS + ARMS:
                        agg[(r["dataset"], r["arm"])].append(r["brier"])
    order = REF_ARMS + ARMS
    print("\n" + "=" * 118)
    print("OUR ENCODER, CATBOOST'S TARGET -- Brier (lower better). d%: change vs "
          "our default, + = better. closed: share of CatBoost's edge over our "
          "default that the arm closes (100% = reaches cb_default).")
    print("=" * 118)
    print(f"{'dataset':26s}" + "".join(f"{a:>16s}" for a in order)
          + "".join(f"  d%@{a[8:] if a.startswith('chimera_') else a:>8s}" for a in ARMS))
    summary = defaultdict(list)
    for group, name in ((GAP, "GAP SETS"), (CONTROL, "CONTROLS")):
        print(f"\n-- {name} " + "-" * (114 - len(name)))
        for ds in group:
            v = {a: (float(np.mean(agg[(ds, a)])) if agg.get((ds, a)) else None) for a in order}
            if v["chimera"] is None:
                continue
            print(_row_line(ds, v, order, name, summary))
    print()
    _print_summary(summary)


def _row_line(ds, v, order, name, summary):
    """One dataset's table row; appends (delta%, edge-closed) per arm to
    ``summary`` as a side effect."""
    ours, cb = v["chimera"], v["cb_default"]
    line = f"{ds[3:][:26]:26s}" + "".join(
        f"{v[a]:16.6f}" if v[a] is not None else f"{'-':>16s}" for a in order)
    for a in ARMS:
        if v[a] is None:
            line += f"{'-':>14s}"
            continue
        d = 100 * (ours - v[a]) / ours
        closed = ((ours - v[a]) / (ours - cb) if cb is not None
                  and abs(ours - cb) > 1e-12 else float("nan"))
        summary[(name, a)].append((d, closed))
        line += f"{d:+13.3f}%"
    return line


def _print_summary(summary):
    for (name, a), vals in sorted(summary.items()):
        d = np.array([x for x, _ in vals])
        closed = np.array([c for _, c in vals])
        wins = int((d > 1e-9).sum())
        losses = int((d < -1e-9).sum())
        print(f"  {name:9s} {a:16s}: {wins}W-{losses}L of {len(d)}, median {np.median(d):+.3f}%  "
              f"(per set {', '.join(f'{x:+.2f}' for x in d)});  edge closed median "
              f"{100 * np.nanmedian(closed):+.0f}%")
    print("\n(the reference rows chimera / cb_default come from probe-cb-hc-ablation.jsonl, "
          "same splits and seeds)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
