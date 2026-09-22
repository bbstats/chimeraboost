"""Probe F5 (CAMPAIGN_PLAN I040): per-column calibrated shrinkage for the
ordered target statistic -- the random-effects form of what the count column
does by proxy. Zero library change.

The ordered target statistic (sum + prior * a) / (n + a) is the random-
intercept posterior mean with a FIXED variance ratio a = cat_smoothing = 1.
B18 measured its consequence on high-card columns: rare categories are
over-trusted (reliability 0.63-0.70), i.e. a = 1 shrinks them too little.
The count column (I035-I039) lets the trees recover the reliability of the
statistic after the fact. The principled version estimates the variance
ratio per column from the data -- the one-way random-effects ANOVA
(method-of-moments) estimator, the closed form REML reduces to for balanced
groups -- and shrinks each column's rare categories as hard as its own data
says, at no extra columns and no extra fit time.

  lambda_j = sigma2_within / sigma2_between      (per categorical column j,
                                                  per class target)
  sigma2_within  = pooled within-category variance of y
  sigma2_between = (MS_between - sigma2_within) / n0, floored at
                   1e-3 * sigma2_within (a column with no between-category
                   signal gets lambda = 1000: almost fully shrunk)
  a_j = clip(lambda_j, 0.1, 1e4)

Arms, on the DEFAULT estimator, same seven sets and paired splits as I033
(its `chimera` rows are the reference line; `cb_default` the CatBoost line):

  reml         a_j per column (a monkeypatch of OrderedTargetEncoder.fit_transform
               / transform that computes a_j on the encoder's training rows)
  count        cat_count_features=True (the library flag, PR #138) -- the
               same read on these splits, for a paired comparison
  reml+count   both
  reml_cap10   reml with lambda clipped to [1, 10] (added after the first read:
               lambda saturated at its floor on every set)

Brier in the harness's K-sum form. Resumable JSONL; table at the end.

Run:
    python benchmarks/probe_ts_reml_smoothing.py
    python benchmarks/probe_ts_reml_smoothing.py --table-only
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
from chimeraboost.target_encoding import OrderedTargetEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "probe-ts-reml-smoothing.jsonl")
REFERENCE = os.path.join(HERE, "results", "probe-cb-hc-ablation.jsonl")
SEEDS = (0, 1, 2)
MAX_ITERS = 2000
PATIENCE = 50
LAMBDA_LO, LAMBDA_HI = 0.1, 1e4
_CLIP = [LAMBDA_LO, LAMBDA_HI]     # the active clip; reml_cap10 narrows it

GAP = ["hc:sf-police-incidents", "hc:Traffic_violations", "hc:kick",
       "hc:okcupid-stem", "hc:porto-seguro"]
CONTROL = ["hc:kdd_ipums_la_97-small", "hc:eucalyptus"]
ALL = GAP + CONTROL
ARMS = ["reml", "count", "reml+count", "reml_cap10"]
REF_ARMS = ["chimera", "cb_default"]
LAMBDA_LOG = defaultdict(list)     # dataset -> per-column lambdas seen


def _brier(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# ---- the variance-ratio estimator -----------------------------------------------
def variance_ratio(codes, y):
    """One-way random-effects ANOVA estimate of sigma2_within / sigma2_between
    for the grouping ``codes`` (0..G-1) of target ``y``; clipped to
    [LAMBDA_LO, LAMBDA_HI]."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    G = int(codes.max()) + 1 if codes.size else 1
    cnt = np.bincount(codes, minlength=G).astype(np.float64)
    s1 = np.bincount(codes, weights=y, minlength=G)
    s2 = np.bincount(codes, weights=y * y, minlength=G)
    nz = cnt > 0
    G_eff = int(nz.sum())
    if G_eff < 2 or n <= G_eff:
        return 1.0
    means = np.where(nz, s1 / np.maximum(cnt, 1.0), 0.0)
    ss_within = float((s2 - cnt * means * means)[nz].sum())
    sigma2_w = ss_within / (n - G_eff)
    grand = float(y.mean())
    ms_between = float((cnt * (means - grand) ** 2)[nz].sum()) / (G_eff - 1)
    n0 = (n - float((cnt * cnt).sum()) / n) / (G_eff - 1)
    sigma2_b = (ms_between - sigma2_w) / max(n0, 1e-12)
    sigma2_b = max(sigma2_b, 1e-3 * max(sigma2_w, 1e-12))
    if sigma2_w <= 0.0:
        return LAMBDA_LO
    return float(np.clip(sigma2_w / sigma2_b, _CLIP[0], _CLIP[1]))


# ---- the monkeypatch ---------------------------------------------------------------
_ORIG_FT = OrderedTargetEncoder.fit_transform
_ORIG_T = OrderedTargetEncoder.transform
_CURRENT_DS = [None]


def _ft_reml(self, codes_matrix, y, sample_weight=None):
    """The library body with one change: a per-column ``a_j`` from
    ``variance_ratio`` replaces the scalar ``self.smoothing``."""
    codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    w = None if sample_weight is None else np.ascontiguousarray(
        sample_weight, dtype=np.float64)
    n_samples, n_cols = codes_matrix.shape
    rng = np.random.default_rng(self.random_state)
    self.prior_ = (float(np.mean(y)) if w is None
                   else float(np.average(y, weights=w)))
    self.sums_, self.counts_, self.n_cat_ = [], [], []
    self.smoothing_per_col_ = []
    out = np.zeros((n_samples, n_cols), dtype=np.float64)
    for j in range(n_cols):
        codes = np.ascontiguousarray(codes_matrix[:, j])
        n_cat = int(codes.max()) + 1 if codes.size else 1
        a_j = variance_ratio(codes, y)
        self.smoothing_per_col_.append(a_j)
        LAMBDA_LOG[_CURRENT_DS[0]].append(a_j)
        acc = np.zeros(n_samples, dtype=np.float64)
        sums = counts = None
        for _ in range(self.n_permutations):
            perm = rng.permutation(n_samples)
            if w is None:
                enc, sums, counts = te._ordered_ts(
                    codes, y, perm, n_cat, self.prior_, a_j)
            else:
                enc, sums, counts = te._ordered_ts_weighted(
                    codes, y, w, perm, n_cat, self.prior_, a_j)
            acc += enc
        out[:, j] = acc / self.n_permutations
        self.sums_.append(sums)
        self.counts_.append(counts)
        self.n_cat_.append(n_cat)
    return out


def _t_reml(self, codes_matrix):
    codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
    n_samples, n_cols = codes_matrix.shape
    out = np.empty((n_samples, n_cols), dtype=np.float64)
    for j in range(n_cols):
        a = self.smoothing_per_col_[j]
        codes = codes_matrix[:, j]
        sums, counts, n_cat = self.sums_[j], self.counts_[j], self.n_cat_[j]
        enc = np.full(n_samples, self.prior_, dtype=np.float64)
        valid = (codes >= 0) & (codes < n_cat)
        c = codes[valid]
        enc[valid] = (sums[c] + self.prior_ * a) / (counts[c] + a)
        out[:, j] = enc
    return out


def _run(Xtr, ytr, Xte, yte, cat, seed, reml, count, clip=(LAMBDA_LO, LAMBDA_HI)):
    _CLIP[0], _CLIP[1] = clip
    if reml:
        OrderedTargetEncoder.fit_transform = _ft_reml
        OrderedTargetEncoder.transform = _t_reml
    try:
        t = time.time()
        m = ChimeraBoostClassifier(n_estimators=MAX_ITERS,
                                   early_stopping_rounds=PATIENCE, random_state=seed,
                                   cat_count_features=bool(count))
        m.fit(Xtr, ytr, cat_features=cat)
        fit_s = time.time() - t
        return _brier(yte, m.predict_proba(Xte), m.classes_), fit_s
    finally:
        OrderedTargetEncoder.fit_transform = _ORIG_FT
        OrderedTargetEncoder.transform = _ORIG_T


ARM_FLAGS = {"reml": (True, False), "count": (False, True), "reml+count": (True, True),
             "reml_cap10": (True, False)}
# reml_cap10, pre-registered after the first three read: lambda clipped to
# [1, 10] -- never less shrinkage than today, at most ten times more -- to
# separate estimator saturation (lambda at the 1000 floor on every set) from
# the direction of the change.
ARM_CLIP = {"reml_cap10": (1.0, 10.0)}


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
        _CURRENT_DS[0] = key
        print(f"\n=== {key}  task={task}  n={len(y)}  p={X.shape[1]}  cats={len(cat)}",
              flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            for arm, (reml, count) in ARM_FLAGS.items():
                if (key, arm, seed) in done:
                    continue
                b, fit_s = _run(Xtr, ytr, Xte, yte, cat, seed, reml, count,
                                ARM_CLIP.get(arm, (LAMBDA_LO, LAMBDA_HI)))
                rec = {"dataset": key, "arm": arm, "seed": seed, "task": task,
                       "brier": b, "fit_time": fit_s}
                if reml and LAMBDA_LOG[key]:
                    lam = np.array(LAMBDA_LOG[key])
                    rec["lambda_median"] = float(np.median(lam))
                    rec["lambda_max"] = float(lam.max())
                    LAMBDA_LOG[key] = []
                _append(rec)
                extra = (f"  lambda med {rec['lambda_median']:.2f} max {rec['lambda_max']:.0f}"
                         if "lambda_median" in rec else "")
                print(f"  [{arm:10s} s{seed}] brier={b:.6f}  {fit_s:6.1f}s{extra}", flush=True)
    table()


def table():
    agg = defaultdict(list)
    lam = defaultdict(list)
    for path in (REFERENCE, RESULTS):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["arm"] in REF_ARMS + ARMS:
                        agg[(r["dataset"], r["arm"])].append(r["brier"])
                    if "lambda_median" in r and r["arm"] == "reml":
                        lam[r["dataset"]].append((r["lambda_median"], r["lambda_max"]))
    order = REF_ARMS + ARMS
    print("\n" + "=" * 112)
    print("CALIBRATED SHRINKAGE vs THE COUNT COLUMN -- Brier (lower better). d%: change "
          "vs our default, + = better.")
    print("=" * 112)
    print(f"{'dataset':26s}" + "".join(f"{a:>13s}" for a in order)
          + "".join(f"  d%@{a:>10s}" for a in ARMS) + "   lambda med/max")
    summary = defaultdict(list)
    for group, name in ((GAP, "GAP SETS"), (CONTROL, "CONTROLS")):
        print(f"\n-- {name} " + "-" * (108 - len(name)))
        for ds in group:
            v = {a: (float(np.mean(agg[(ds, a)])) if agg.get((ds, a)) else None) for a in order}
            if v["chimera"] is None:
                continue
            print(_row_line(ds, v, order, name, summary, lam.get(ds)))
    print()
    _print_summary(summary)


def _row_line(ds, v, order, name, summary, lam):
    ours, cb = v["chimera"], v["cb_default"]
    line = f"{ds[3:][:26]:26s}" + "".join(
        f"{v[a]:13.6f}" if v[a] is not None else f"{'-':>13s}" for a in order)
    for a in ARMS:
        if v[a] is None:
            line += f"{'-':>15s}"
            continue
        d = 100 * (ours - v[a]) / ours
        closed = ((ours - v[a]) / (ours - cb) if cb is not None
                  and abs(ours - cb) > 1e-12 else float("nan"))
        summary[(name, a)].append((d, closed))
        line += f"{d:+14.3f}%"
    if lam:
        line += f"   {np.median([m for m, _ in lam]):.1f} / {max(x for _, x in lam):.0f}"
    return line


def _print_summary(summary):
    for (name, a), vals in sorted(summary.items()):
        d = np.array([x for x, _ in vals])
        closed = np.array([c for _, c in vals])
        wins = int((d > 1e-9).sum())
        losses = int((d < -1e-9).sum())
        print(f"  {name:9s} {a:10s}: {wins}W-{losses}L of {len(d)}, median {np.median(d):+.3f}%  "
              f"(per set {', '.join(f'{x:+.2f}' for x in d)});  edge closed median "
              f"{100 * np.nanmedian(closed):+.0f}%")
    print("\n(reference rows chimera / cb_default from probe-cb-hc-ablation.jsonl, same "
          "splits and seeds; lambda = the per-column variance ratio the reml arm used)")


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        table()
    else:
        main()
