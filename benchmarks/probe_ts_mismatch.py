"""F6 probe: are trees grown on one target-statistic distribution and scored on
another?

`OrderedTargetEncoder.fit_transform` encodes a training row from the rows of
its category that come BEFORE it in a random permutation (averaged over 4
permutations). For a category of m rows the prefix length k is uniform on
0..m-1, so the expected weight on the category's own mean is

    h(m) = mean over k < m of k / (k + a)        (m=5, a=1: 0.54; m=1: 0)

`transform` encodes every other row -- the early-stopping split, calibration,
and every prediction -- from the full totals, weight m / (m + a) (m=5: 0.83;
m=1: 0.5). Nobody has measured what that asymmetry does. This probe does, with
zero library change, in two parts.

Part 1 (no model): fit the default preprocessor on a 75% split and compare the
raw, pre-binning encodings of the training rows against those of the held-out
rows, per TS column and per stratum of category train count. The stratified
read is the point: a count-dependent shrink can only repair a mismatch that
VARIES with the count (BARRIERS B5), so a mismatch that does not concentrate in
rare categories kills the fix before any model is fit. The reliability slope --
OLS of the target on the encoding -- is honest on training rows because ordered
TS never shows a row its own label.

Part 2 (transform-only A/B, by monkeypatch): the default estimator, paired on
split seeds, under three transform weights --

    A0  shipped          m / (m + a)
    A1  count-matched    g(m) = mean over k <= m of k / (k + a): what the row
                         would have read had it been a training row
    A2  half-matched     the mean of the two weights (registered before any
                         data: permutation averaging makes train encodings
                         less noisy than one prefix, so A1 should over-correct)

Unseen categories read the prior in every arm and `fit_transform` is untouched,
so the trees are grown identically; only what they are stopped, calibrated and
scored on moves. A0 calls the library's own `transform`, so it is the shipped
arithmetic exactly.

Run: python benchmarks/probe_ts_mismatch.py [--datasets KEY ...] [--seeds 3]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import digamma
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb

import chimeraboost.target_encoding as temod
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.preprocessing import FeaturePreprocessor, as_model_array
from chimeraboost.sklearn_api import _auto_cat_combinations

GAP = ["hc:sf-police-incidents", "hc:okcupid-stem", "hc:Traffic_violations",
       "hc:kick"]
SECONDARY = ["hc:wine-reviews", "hc:colleges"]
CONTROL = ["hc:kdd_ipums_la_97-small", "hc:porto-seguro"]
PANEL = GAP + SECONDARY + CONTROL

HIGH_CARD = 1000          # a "high-card" column, as the shortlist defined it
LOW_CARD = 200            # the in-dataset control columns
STRATA = (("m<=5", 1, 5), ("m6-50", 6, 50), ("m>50", 51, np.inf))
ARMS = ("A0", "A1", "A2")

_ORIG_TRANSFORM = temod.OrderedTargetEncoder.transform
_MODE = ["A0"]


def matched_weight(m, a):
    """g(m): expected train-time weight on the category mean for a row
    inserted at a uniform position among m others. Closed form of
    mean_{k=0..m} k/(k+a), valid for real m (weighted counts)."""
    return 1.0 - a * (digamma(m + 1.0 + a) - digamma(a)) / (m + 1.0)


def _patched_transform(self, codes_matrix):
    if _MODE[0] == "A0":
        return _ORIG_TRANSFORM(self, codes_matrix)
    codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
    n_samples, n_cols = codes_matrix.shape
    out = np.empty((n_samples, n_cols), dtype=np.float64)
    a = self.smoothing
    for j in range(n_cols):
        codes = codes_matrix[:, j]
        sums, counts, n_cat = self.sums_[j], self.counts_[j], self.n_cat_[j]
        enc = np.full(n_samples, self.prior_, dtype=np.float64)
        valid = (codes >= 0) & (codes < n_cat)
        c = codes[valid]
        m = counts[c]
        seen = m > 0
        w = matched_weight(m, a)
        if _MODE[0] == "A2":
            w = 0.5 * (w + m / (m + a))
        mean_c = np.where(seen, sums[c] / np.where(seen, m, 1.0), self.prior_)
        enc[valid] = self.prior_ + (mean_c - self.prior_) * w
        out[:, j] = enc
    return out


# --------------------------------------------------------------------------
# Part 1: the moment read
# --------------------------------------------------------------------------
def _targets_for(task, ytr, yte):
    """The encoder's targets, built as the boosters build them."""
    if task == "regression":
        return [np.asarray(ytr, float)], [np.asarray(yte, float)], ["y"]
    classes = np.unique(ytr)
    if task == "binary":
        pos = classes[1]
        return ([(ytr == pos).astype(float)], [(yte == pos).astype(float)],
                [f"y=={pos}"])
    return ([(ytr == k).astype(float) for k in classes],
            [(yte == k).astype(float) for k in classes],
            [f"y=={k}" for k in classes])


def _slope(t, e):
    v = e.var()
    return float(np.cov(t, e, bias=True)[0, 1] / v) if v > 0 else float("nan")


def moment_read(key, X, y, cat, task):
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=strat)
    Xtr = as_model_array(Xtr, bool(cat))
    Xte = as_model_array(Xte, bool(cat))
    t_tr, t_te, t_names = _targets_for(task, np.asarray(ytr), np.asarray(yte))

    prep = FeaturePreprocessor(
        max_bins=128, cat_smoothing=1.0, random_state=0, cat_n_permutations=4,
        cat_combinations=_auto_cat_combinations(cat, Xtr.shape[1], len(Xtr)))
    captured = []
    orig_ft = temod.OrderedTargetEncoder.fit_transform

    def cap_ft(self, codes_matrix, yy, sample_weight=None):
        out = orig_ft(self, codes_matrix, yy, sample_weight)
        captured.append((np.asarray(codes_matrix, dtype=np.int64), out))
        return out

    temod.OrderedTargetEncoder.fit_transform = cap_ft
    try:
        prep.fit_transform(Xtr, t_tr, cat)
    finally:
        temod.OrderedTargetEncoder.fit_transform = orig_ft

    codes_te = prep._codes_for_transform(Xte)
    if prep.combo_pairs_:
        codes_te = np.hstack([codes_te, prep._combo_codes_for_transform(Xte)])
    names = ([f"cat{f}" for f in prep.cat_features_]
             + [f"cat{a}xcat{b}" for a, b in prep.combo_pairs_])

    cols = []
    for ti, enc in enumerate(prep.encoders_):
        codes_tr, e_tr_all = captured[ti]
        e_te_all = _ORIG_TRANSFORM(enc, codes_te)
        for j, name in enumerate(names):
            counts = enc.counts_[j]
            e_tr, e_te = e_tr_all[:, j], e_te_all[:, j]
            m_tr = counts[codes_tr[:, j]]
            ok = (codes_te[:, j] >= 0) & (codes_te[:, j] < enc.n_cat_[j])
            m_te = np.where(ok, counts[np.where(ok, codes_te[:, j], 0)], 0.0)
            sd_tr = e_tr.std()
            rec = {
                "column": name, "target": t_names[ti],
                "card": int(enc.n_cat_[j]),
                "unseen_frac": float(1.0 - ok.mean()),
                "heldout_m_le2": float(((m_te >= 1) & (m_te <= 2)).mean()),
                "heldout_m_le5": float(((m_te >= 1) & (m_te <= 5)).mean()),
                "dmean_sd": float((e_te.mean() - e_tr.mean()) / sd_tr)
                if sd_tr > 0 else 0.0,
                "sd_ratio": float(e_te.std() / sd_tr) if sd_tr > 0
                else float("nan"),
                "ks": float(ks_2samp(e_tr, e_te).statistic),
                "strata": {},
            }
            for sname, lo, hi in STRATA:
                a_tr = (m_tr >= lo) & (m_tr <= hi)
                a_te = (m_te >= lo) & (m_te <= hi)
                if a_tr.sum() < 200 or a_te.sum() < 200:
                    continue
                sp_tr = float(np.abs(e_tr[a_tr] - enc.prior_).mean())
                sp_te = float(np.abs(e_te[a_te] - enc.prior_).mean())
                sl_tr = _slope(t_tr[ti][a_tr], e_tr[a_tr])
                sl_te = _slope(t_te[ti][a_te], e_te[a_te])
                rec["strata"][sname] = {
                    "n_train": int(a_tr.sum()), "n_heldout": int(a_te.sum()),
                    "spread_ratio": sp_te / sp_tr if sp_tr > 0
                    else float("nan"),
                    "slope_train": sl_tr, "slope_heldout": sl_te,
                    "reliability_ratio": sl_te / sl_tr
                    if sl_tr and np.isfinite(sl_tr) else float("nan"),
                }
            cols.append(rec)
    return cols


def _med(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def moment_summary(key, cols):
    """Per dataset: medians over its high-card columns, and over its low-card
    columns as the in-dataset control."""
    out = {}
    for label, pick in (("high", lambda c: c["card"] >= HIGH_CARD),
                        ("low", lambda c: c["card"] < LOW_CARD)):
        sel = [c for c in cols if pick(c)]
        row = {"n_cols": len(sel),
               "max_card": max((c["card"] for c in sel), default=0),
               "dmean_sd": _med([abs(c["dmean_sd"]) for c in sel]),
               "sd_ratio": _med([c["sd_ratio"] for c in sel]),
               "ks": _med([c["ks"] for c in sel]),
               "heldout_m_le2": _med([c["heldout_m_le2"] for c in sel]),
               "unseen": _med([c["unseen_frac"] for c in sel])}
        for sname, _, _ in STRATA:
            row[f"spread[{sname}]"] = _med(
                [c["strata"].get(sname, {}).get("spread_ratio") for c in sel])
            row[f"reliab[{sname}]"] = _med(
                [c["strata"].get(sname, {}).get("reliability_ratio")
                 for c in sel])
        out[label] = row
    return out


# --------------------------------------------------------------------------
# Part 2: the transform-only A/B
# --------------------------------------------------------------------------
def _metrics(task, model, Xte, yte):
    if task == "regression":
        err = np.asarray(model.predict(Xte), float) - np.asarray(yte, float)
        return {"primary": float(np.sqrt(np.mean(err ** 2)))}
    P = np.clip(model.predict_proba(Xte), 1e-15, 1.0)
    Y = (np.asarray(yte)[:, None] == model.classes_[None, :]).astype(float)
    if task == "binary":
        brier = float(np.mean((P[:, 1] - Y[:, 1]) ** 2))
    else:
        brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    return {"primary": brier,
            "logloss": float(-np.mean(np.log(P[Y.astype(bool)])))}


def ab_read(key, X, y, cat, task, seeds):
    Est = ChimeraBoostRegressor if task == "regression" \
        else ChimeraBoostClassifier
    rows = []
    for seed in range(seeds):
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=strat)
        rec = {"dataset": key, "seed": seed}
        for arm in ARMS:
            _MODE[0] = arm
            t0 = time.perf_counter()
            model = Est(random_state=0).fit(Xtr, ytr, cat_features=cat)
            rec[arm] = _metrics(task, model, Xte, yte)
            rec[arm]["fit_s"] = time.perf_counter() - t0
            rec[arm]["trees"] = int(model.best_iteration_)
        _MODE[0] = "A0"
        rows.append(rec)
        base = rec["A0"]["primary"]
        print(f"  seed {seed}: A0 {base:.6f}  " + "  ".join(
            f"{arm} {100.0 * (base - rec[arm]['primary']) / base:+.3f}%"
            for arm in ARMS[1:]), flush=True)
    return rows


def _gain(rec, arm, metric="primary"):
    """Percent improvement over A0; positive = the arm is better (all three
    metrics are lower-is-better)."""
    base = rec["A0"][metric]
    return 100.0 * (base - rec[arm][metric]) / base


def report(moments, ab):
    out = ["# F6 probe: ordered-TS train vs held-out mismatch", "",
           "## Part 1. Moments, medians over each dataset's columns "
           "(held-out / train)", "",
           "| dataset | cols | n | max card | |dmean|/SD | SD ratio | KS "
           "| held-out rows m<=2 | unseen | spread m<=5 | spread m6-50 "
           "| spread m>50 | reliab m<=5 | reliab m6-50 | reliab m>50 |",
           "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for key, summ in moments.items():
        for label in ("high", "low"):
            r = summ[label]
            if not r["n_cols"]:
                continue
            out.append(
                f"| {key} | {label} | {r['n_cols']} | {r['max_card']} "
                f"| {r['dmean_sd']:.3f} | {r['sd_ratio']:.3f} "
                f"| {r['ks']:.3f} | {100 * r['heldout_m_le2']:.1f}% "
                f"| {100 * r['unseen']:.1f}% "
                + " ".join(f"| {r[f'spread[{s}]']:.2f}" for s, _, _ in STRATA)
                + " "
                + " ".join(f"| {r[f'reliab[{s}]']:.2f}" for s, _, _ in STRATA)
                + " |")
    out += ["", "cols: high = card >= 1000, low = card < 200 (the in-dataset "
            "control). spread = E|e - prior|, reliab = OLS slope of the "
            "target on the encoding; both are held-out over train, so 1.00 "
            "is no mismatch and reliab < 1 is over-trust.", "",
            "## Part 2. Transform-only A/B, % improvement in the primary "
            "metric over A0 (Brier / RMSE; positive = better)", "",
            "| dataset | role | A0 primary | A1 per seed | A1 mean "
            "| A2 per seed | A2 mean | trees A0/A1/A2 (seed 0) |",
            "|---|---|--:|---|--:|---|--:|---|"]
    for key, rows in ab.items():
        role = ("gap" if key in GAP else
                "secondary" if key in SECONDARY else "control")
        cells = []
        for arm in ARMS[1:]:
            g = [_gain(r, arm) for r in rows]
            cells.append(" ".join(f"{v:+.3f}" for v in g))
            cells.append(f"{np.mean(g):+.3f}")
        t = rows[0]
        out.append(f"| {key} | {role} "
                   f"| {np.mean([r['A0']['primary'] for r in rows]):.5f} | "
                   + " | ".join(cells)
                   + f" | {t['A0']['trees']}/{t['A1']['trees']}/"
                     f"{t['A2']['trees']} |")
    out += ["", "## Part 2 sign read", ""]
    for group, keys in (("gap sets", GAP), ("secondary", SECONDARY),
                        ("controls", CONTROL)):
        for arm in ARMS[1:]:
            g = [_gain(r, arm) for k in keys if k in ab for r in ab[k]]
            if not g:
                continue
            w = sum(v > 0 for v in g)
            l = sum(v < 0 for v in g)
            out.append(f"- {group}, {arm}: {w}W-{l}L-{len(g) - w - l}T over "
                       f"{len(g)} (set, seed) pairs, median "
                       f"{np.median(g):+.3f}%, mean {np.mean(g):+.3f}%")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="probe-ts-mismatch")
    ap.add_argument("--skip-ab", action="store_true")
    args = ap.parse_args()

    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(temod.__file__)}", flush=True)
    temod.OrderedTargetEncoder.transform = _patched_transform

    moments, columns, ab = {}, {}, {}
    for key in args.datasets:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        print(f"\n=== {key} ({task}, n={len(X)}, cats="
              f"{len(cat) if cat else 0}) ===", flush=True)
        columns[key] = moment_read(key, X, y, cat, task)
        moments[key] = moment_summary(key, columns[key])
        hi = moments[key]["high"]
        print(f"  high-card cols {hi['n_cols']}: SD ratio {hi['sd_ratio']:.3f}"
              f"  spread m<=5 {hi['spread[m<=5]']:.2f}"
              f"  reliab m<=5 {hi['reliab[m<=5]']:.2f}", flush=True)
        if not args.skip_ab:
            ab[key] = ab_read(key, X, y, cat, task, args.seeds)

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", args.out)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".json", "w") as f:
        json.dump({"moments": moments, "columns": columns, "ab": ab}, f)
    text = report(moments, ab)
    with open(base + ".md", "w", newline="\n") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved {base}.json and {base}.md")


if __name__ == "__main__":
    main()
