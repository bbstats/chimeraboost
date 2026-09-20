"""Grouped-data benchmark: random intercepts against the field (#109).

Why this exists. Slice 1 (random intercepts, regression-only) engages only
when the user passes group labels, and neither decision suite has group
columns -- ``--decide`` cannot see it. So this suite owns grouped data the
way ``quantile_suite.py`` owns quantiles, and writes the harness's JSON shape
so ``compare_runs.py`` can sign-test it:

    python benchmarks/grouped_suite.py --seeds 3 --save
    python benchmarks/compare_runs.py grouped-BASE.json grouped-NEW.json \\
        --model ChimeraDrop --model-new ChimeraRE

Deliberately NOT wired into ``run_benchmarks.py --decide``: the decision
suites have no group structure, and a group-aware task kind would ripple
through the variant families and per-stratum gates for no gain. This borrows
the harness's registry and budget constants, and nothing else.

Protocol. Every dataset is split by GROUP (seeded): ~20% of groups are held
out entirely (their test rows are the UNSEEN slice -- no intercept can help
there), and ~20% of the remaining rows form the SEEN slice. The forecast
(RANDEFF_PLAN.md) is a large win on seen rows at high intra-class
correlation and roughly flat on unseen rows; a win anywhere else fails the
mechanism story. Train and test rows are shared by every arm; every arm
validates on a RANDOM carve (the Chimera arms self-split, LightGBM/CatBoost
share one carved set) -- a whole-group holdout blinds early stopping, the
RANDEFF_PLAN.md finding that levelled this footing.

Arms
----
ChimeraRE     random_effects=True on groups; the group column is DROPPED
              from X (the issue's "no group ID features" rule)
ChimeraRE-postonly
              the identical plain fit plus one EB solve after -- the
              refinement ablation, kept as the inner baseline
ChimeraCat    plain ChimeraBoost with the group column as a categorical --
              the in-house baseline the new machinery must justify itself
              against (ordered target statistics on the IDs)
ChimeraDrop   plain ChimeraBoost with the group column dropped -- what a
              careful user does today
LightGBMCat   LightGBM with the group column as a categorical
CatBoostCat   CatBoost with the group column as a categorical

Real sets are hc: regression data with natural cluster columns
(winery / state / neighborhood / department / team). Rossmann (pub:) is
deliberately NOT here: pub: data must not answer a gate. Synthetic sets
(``gsyn:``) carry known truth and known intra-class correlation.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import ChimeraBoostRegressor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

# Real grouped sets: hc key -> (group column, row cap). The cap is this
# suite's own (deterministic seed-0 subsample inside _prepare_frame); the
# group column must survive its near-unique filter, which --list-datasets
# verifies by loading every set.
GROUPED_HC = {
    "hc:wine-reviews": ("winery", 30000),
    "hc:colleges": ("state", 20000),
    "hc:house_prices_nominal": ("Neighborhood", 20000),
    "hc:employee_salaries": ("department", 20000),
    "hc:Moneyball": ("Team", 20000),
    # No hc:black_friday: its IDs (User_ID, Product_ID) are >90% unique and
    # the loader drops them; the surviving cats (2-21 levels) are attributes,
    # not clusters. No hc:college-zip either: 6039 zips at median 1 row carry
    # no pooling signal; state (59 groups) is the honest grouping.
}

UNSEEN_GROUP_FRAC = 0.2
SEEN_TEST_FRAC = 0.2
VAL_FRAC = 0.2


# --- splits ---------------------------------------------------------------

def _grouped_test_split(groups, seed):
    """Whole-group unseen slice + within-group seen slice + train rows.

    Returns ``(train_idx, seen_idx, unseen_idx)``. Deterministic in seed.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    # Whole groups until ~20% of rows are held out unseen.
    sizes = {g: int((groups == g).sum()) for g in uniq}
    target = UNSEEN_GROUP_FRAC * len(groups)
    acc, cut = 0, 0
    for cut, g in enumerate(uniq):
        acc += sizes[g]
        if acc >= target:
            break
    unseen_groups = set(uniq[:cut + 1].tolist())
    unseen_idx = np.flatnonzero(
        np.array([g in unseen_groups for g in groups]))
    rest = np.flatnonzero(
        np.array([g not in unseen_groups for g in groups]))
    rng.shuffle(rest)
    n_seen = int(round(SEEN_TEST_FRAC * len(rest)))
    return rest[n_seen:], rest[:n_seen], unseen_idx


def _random_val_split(n, seed):
    """Random validation carve (~20% of rows) shared by every arm.

    Deliberately NOT whole-group: a group holdout blinds early stopping
    (the RANDEFF_PLAN.md finding -- the field arms stopped at a dozen-odd
    trees on synthetic sets under the old carve), and every Chimera arm
    self-splits randomly, so this keeps all arms on the same footing.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(VAL_FRAC * n))
    return idx[n_val:], idx[:n_val]


# --- synthetic grouped data (known truth) ---------------------------------

def _synth_base(rng, n_groups, sizes, group_sd, noise, x_shift=0.0):
    codes = np.repeat(np.arange(n_groups), sizes)
    n = len(codes)
    X = rng.normal(size=(n, 5))
    if x_shift:
        X[:, 0] += x_shift * (codes - codes.mean()) / codes.std()
    f = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
         + np.sin(2.0 * X[:, 4]))
    b = rng.normal(0.0, group_sd, size=n_groups)
    y = f + b[codes] + rng.normal(0.0, noise, size=n)
    return X, y, codes


def synth_dataset(config, seed):
    """One truth-known grouped set. Configs vary ICC, skew and X/group ties."""
    rng = np.random.default_rng(1000 + seed)
    if config == "base":
        return _synth_base(rng, 40, 50, 5.0, 1.0)
    if config == "low-icc":
        return _synth_base(rng, 40, 50, 1.0, 2.0)
    if config == "skewed":
        sizes = (rng.pareto(1.5, size=60) * 8).astype(int) + 5
        return _synth_base(rng, 60, sizes, 5.0, 1.0)
    if config == "many-small":
        return _synth_base(rng, 200, 10, 4.0, 1.0)
    if config == "few-big":
        return _synth_base(rng, 10, 200, 5.0, 1.0)
    if config == "x-confounded":
        return _synth_base(rng, 40, 50, 5.0, 1.0, x_shift=1.5)
    raise ValueError(f"unknown synth config {config!r}")


SYNTH_CONFIGS = ["base", "low-icc", "skewed", "many-small", "few-big",
                 "x-confounded"]


# --- real grouped data (hc: registry) -------------------------------------

def _load_hc_grouped(key):
    """X with the group column last, y, cat list, group labels, group pos."""
    import pandas as pd

    name = key[len("hc:"):]
    group_col, cap = GROUPED_HC[key]
    spec = rb.HC_DATASETS[name]
    frame = pd.read_parquet(rb._public_parquet_path(spec["data_id"]))
    X_df, y = rb._prepare_frame(frame, spec["target"], spec, None, cap)
    if group_col not in X_df.columns:
        raise ValueError(f"{key}: group column {group_col!r} did not "
                         f"survive the load (dropped as near-unique?)")
    groups = X_df[group_col].to_numpy()
    # Group column last, so arms slice it off positionally.
    cols = [c for c in X_df.columns if c != group_col] + [group_col]
    X_df = X_df[cols]
    X, y, cat, _ = rb._frame_to_dataset(X_df, y, "auto", "regression")
    return X, y, cat, groups, X.shape[1] - 1


# --- arms -----------------------------------------------------------------

def _chimera(thread_count, **kw):
    params = dict(n_estimators=rb.MAX_ITERS,
                  early_stopping_rounds=rb.PATIENCE,
                  thread_count=thread_count, random_state=0)
    params.update(kw)
    return ChimeraBoostRegressor(**params)


def _fit_chimera_re(tr, val, te, gpos, cat, threads):
    # NOTE: no explicit eval_set -- the RE arm self-splits randomly
    # internally (a group holdout would blind the plain winner's early
    # stopping) and the full-data refit engages, refining on the adjusted
    # target. Train/test rows are shared by all arms.
    Xtr, ytr, gtr = tr
    Xte, _, gte = te
    keep = [i for i in range(Xtr.shape[1]) if i != gpos]
    cats = [c - (1 if c > gpos else 0) for c in (cat or []) if c != gpos]
    m = _chimera(threads, random_effects=True)
    t = time.time()
    m.fit(Xtr[:, keep], ytr, groups=gtr, cat_features=cats or None)
    fit_s = time.time() - t
    t = time.time()
    p = m.predict(Xte[:, keep], groups=gte)
    return p, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_re_postonly(tr, val, te, gpos, cat, threads):
    # Post-only ablation: the identical plain fit (same rows, same
    # self-split -- random in both arms), then one EB solve after. The
    # ONLY difference from ChimeraRE is the refinement: ChimeraRE's
    # full-data refit trains on the group-adjusted target, this arm's on
    # raw y. Kept as the permanent inner baseline for RE work.
    from chimeraboost.random_effects import (codes_for_labels,
                                             estimate_ratio_reml,
                                             solve_intercepts)
    from chimeraboost.target_encoding import factorize
    Xtr, ytr, gtr = tr
    Xte, _, gte = te
    keep = [i for i in range(Xtr.shape[1]) if i != gpos]
    cats = [c - (1 if c > gpos else 0) for c in (cat or []) if c != gpos]
    m = _chimera(threads)
    t = time.time()
    m.fit(Xtr[:, keep], ytr, cat_features=cats or None)
    codes_tr, labels = factorize(gtr)
    resid = (np.asarray(ytr, dtype=np.float64)
             - m.predict_raw(Xtr[:, keep]))
    ratio = estimate_ratio_reml(resid, codes_tr, len(labels))
    b = solve_intercepts(resid, codes_tr, len(labels), ratio)
    fit_s = time.time() - t
    t = time.time()
    off = np.zeros(len(gte), dtype=np.float64)
    c = codes_for_labels(gte, labels)
    seen = c >= 0
    off[seen] = b[c[seen]]
    p = m.predict(Xte[:, keep]) + off
    return p, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_cat(tr, val, te, gpos, cat, threads):
    # Self-splits like the RE arm (default UX); the shared val is only for
    # the field arms below.
    Xtr, ytr, _g = tr
    Xte, _, _gte = te
    cats = sorted(set(cat or []) | {gpos})
    m = _chimera(threads)
    t = time.time()
    m.fit(Xtr, ytr, cat_features=cats)
    fit_s = time.time() - t
    t = time.time()
    p = m.predict(Xte)
    return p, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_drop(tr, val, te, gpos, cat, threads):
    # Self-splits like the RE arm (default UX).
    Xtr, ytr, _g = tr
    Xte, _, _gte = te
    keep = [i for i in range(Xtr.shape[1]) if i != gpos]
    cats = [c - (1 if c > gpos else 0) for c in (cat or []) if c != gpos]
    m = _chimera(threads)
    t = time.time()
    m.fit(Xtr[:, keep], ytr, cat_features=cats or None)
    fit_s = time.time() - t
    t = time.time()
    p = m.predict(Xte[:, keep])
    return p, fit_s, time.time() - t, m.best_iteration_


def _fit_lightgbm_cat(tr, val, te, gpos, cat, threads):
    import lightgbm as lgb
    Xtr, ytr, _g = tr
    Xv, yv, _gv = val
    Xte, _, _gte = te
    cats = sorted(set(cat or []) | {gpos})
    Xtr_in, Xv_in, Xte_in = rb._lgb_prepare(Xtr, Xv, Xte, cats)
    m = lgb.LGBMRegressor(n_estimators=rb.MAX_ITERS, n_jobs=threads or -1,
                          random_state=0, verbosity=-1)
    t = time.time()
    m.fit(Xtr_in, ytr, categorical_feature=cats,
          eval_set=[(Xv_in, yv)],
          callbacks=[lgb.early_stopping(rb.PATIENCE, verbose=False)])
    fit_s = time.time() - t
    t = time.time()
    p = np.asarray(m.predict(Xte_in), dtype=np.float64).ravel()
    return p, fit_s, time.time() - t, m.best_iteration_


def _fit_catboost_cat(tr, val, te, gpos, cat, threads):
    from catboost import CatBoostRegressor
    Xtr, ytr, _g = tr
    Xv, yv, _gv = val
    Xte, _, _gte = te
    cats = sorted(set(cat or []) | {gpos})
    m = CatBoostRegressor(allow_writing_files=False, iterations=rb.MAX_ITERS,
                          early_stopping_rounds=rb.PATIENCE,
                          thread_count=threads or -1, random_seed=0,
                          verbose=False)
    t = time.time()
    m.fit(Xtr, ytr, cat_features=cats, eval_set=(Xv, yv))
    fit_s = time.time() - t
    t = time.time()
    p = np.asarray(m.predict(Xte), dtype=np.float64).ravel()
    return p, fit_s, time.time() - t, int(m.tree_count_)


ARMS = {
    "ChimeraRE": _fit_chimera_re,
    "ChimeraRE-postonly": _fit_chimera_re_postonly,
    "ChimeraCat": _fit_chimera_cat,
    "ChimeraDrop": _fit_chimera_drop,
    "LightGBMCat": _fit_lightgbm_cat,
    "CatBoostCat": _fit_catboost_cat,
}


# --- scoring ----------------------------------------------------------------

def score(yte, pred, seen_mask):
    """Overall / seen / unseen RMSE; primary = -rmse (higher is better)."""
    yte = np.asarray(yte, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64).ravel()
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    rs = float(np.sqrt(np.mean((pred[seen_mask] - yte[seen_mask]) ** 2)))
    ru = float(np.sqrt(np.mean((pred[~seen_mask] - yte[~seen_mask]) ** 2)))
    var = float(np.var(yte))
    return {"primary": -rmse, "rmse": rmse, "rmse_seen": rs,
            "rmse_unseen": ru, "r2": 1.0 - (rmse ** 2) / var if var else 0.0,
            "n_seen": int(seen_mask.sum()),
            "n_unseen": int((~seen_mask).sum())}


def run_one(key, seed, threads, models):
    if key.startswith("gsyn:"):
        X, y, groups = synth_dataset(key[len("gsyn:"):], seed)
        cat, gpos = None, X.shape[1]  # appended below
        # Object dtype with REAL ints in the group column: CatBoost refuses
        # cat_features on a pure-float matrix, and column_stack would cast
        # the codes to float before astype(object) -- ints must be placed,
        # not cast.
        Xo = np.empty((len(y), X.shape[1] + 1), dtype=object)
        Xo[:, :X.shape[1]] = X
        Xo[:, X.shape[1]] = groups
        X = Xo
    else:
        X, y, cat, groups, gpos = _load_hc_grouped(key)
    train_idx, seen_idx, unseen_idx = _grouped_test_split(groups, seed)
    te_idx = np.concatenate([seen_idx, unseen_idx])
    seen_mask = np.arange(len(te_idx)) < len(seen_idx)
    tr_groups = groups[train_idx]
    f_idx, v_idx = _random_val_split(len(train_idx), seed)

    tr = (X[train_idx][f_idx], y[train_idx][f_idx], tr_groups[f_idx])
    val = (X[train_idx][v_idx], y[train_idx][v_idx], tr_groups[v_idx])
    te = (X[te_idx], y[te_idx], groups[te_idx])

    meta = {"task": "regression-grouped", "n_train": int(len(f_idx)),
            "n_seen": int(len(seen_idx)), "n_unseen": int(len(unseen_idx)),
            "n_total": int(len(y)), "n_features": int(X.shape[1] - 1),
            "n_groups": int(len(np.unique(groups))),
            "y_std_test": float(np.std(y[te_idx])), "variant": None}

    out = {}
    for name in models:
        try:
            p, fit_s, pred_s, best = ARMS[name](tr, val, te, gpos, cat,
                                               threads)
            out[name] = (score(y[te_idx], p, seen_mask), fit_s, pred_s,
                         best)
        except Exception as e:
            print(f"  [skip] {name} on {key} (seed {seed}): "
                  f"{type(e).__name__}: {e}")
            out[name] = None
    return meta, out


# --- aggregate --------------------------------------------------------------

def _sign_test(wins, losses):
    from scipy.stats import binomtest
    n = wins + losses
    if not n:
        return float("nan")
    return float(binomtest(wins, n, 0.5).pvalue)


def aggregate(records, models, ref="ChimeraRE"):
    rows = {}
    for m in models:
        vals = [r["metrics"] for r in records if r["model"] == m]
        if not vals:
            continue
        rows[m] = {
            k: float(np.mean([v[k] for v in vals]))
            for k in ("rmse", "rmse_seen", "rmse_unseen", "r2")}
        rows[m]["fit_s"] = float(np.mean(
            [r["fit_time"] for r in records if r["model"] == m]))
        rows[m]["n"] = len(vals)
    # Head-to-head: the RE arm vs each baseline, overall and seen-only.
    wins = {}
    for m in models:
        if m == ref:
            continue
        w = w_s = l = l_s = 0
        for r in records:
            if r["model"] != ref:
                continue
            peer = next((q for q in records
                         if q["model"] == m and q["dataset"] == r["dataset"]
                         and q["seed"] == r["seed"]), None)
            if peer is None:
                continue
            if r["metrics"]["rmse"] < peer["metrics"]["rmse"]:
                w += 1
            elif r["metrics"]["rmse"] > peer["metrics"]["rmse"]:
                l += 1
            if r["metrics"]["rmse_seen"] < peer["metrics"]["rmse_seen"]:
                w_s += 1
            elif r["metrics"]["rmse_seen"] > peer["metrics"]["rmse_seen"]:
                l_s += 1
        wins[m] = {"w": w, "l": l, "p": _sign_test(w, l),
                   "w_s": w_s, "l_s": l_s, "p_s": _sign_test(w_s, l_s)}
    return rows, wins


def format_table(rows, wins, ref="ChimeraRE"):
    head = (f"{'model':14s}{'rmse':>10s}{'seen':>10s}{'unseen':>10s}"
            f"{'r2':>8s}{'fit s':>9s}{'vs ' + ref:>12s}")
    lines = [head, "-" * len(head)]
    fit0 = rows.get(ref, {}).get("fit_s")
    for m, r in sorted(rows.items(), key=lambda kv: kv[1]["rmse"]):
        tag = "   (ref)" if m == ref else ""
        rel = (f"{r['fit_s'] / fit0:8.2f}x" if fit0 else "        -")
        lines.append(
            f"{m:14s}{r['rmse']:10.4f}{r['rmse_seen']:10.4f}"
            f"{r['rmse_unseen']:10.4f}{r['r2']:8.3f}{r['fit_s']:9.2f}"
            f"{rel if m != ref else tag:>12s}")
    lines.append("")
    lines.append(f"{ref} head-to-head (overall rmse | seen rmse):")
    for m, s in wins.items():
        lines.append(f"  vs {m:14s} {s['w']:2d}W-{s['l']:2d}L p={s['p']:.4f} "
                     f"| {s['w_s']:2d}W-{s['l_s']:2d}L p={s['p_s']:.4f}")
    return "\n".join(lines)


# --- main -------------------------------------------------------------------

def all_datasets():
    rb._add_highcard_datasets()
    return sorted(GROUPED_HC) + [f"gsyn:{c}" for c in SYNTH_CONFIGS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first seed value; seeds run offset..offset+seeds-1 "
                    "(fresh draws past the gate's seeds for confirmatory "
                    "ablations)")
    ap.add_argument("--threads", type=int, default=None,
                    help="thread budget per model (None = all cores).")
    ap.add_argument("--models", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys; default = every grouped set")
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    names = args.datasets or all_datasets()

    if args.list_datasets:
        rb._add_highcard_datasets()
        for n in names:
            if n.startswith("gsyn:"):
                X, y, g = synth_dataset(n[len("gsyn:"):], 0)
                print(f"{n}: synth n={len(y)} groups={len(np.unique(g))}")
            else:
                try:
                    X, y, cat, g, _ = _load_hc_grouped(n)
                    ug, cs = np.unique(g, return_counts=True)
                    print(f"{n}: n={len(y)} groups={len(ug)} "
                          f"rows/group min={cs.min()} med={int(np.median(cs))} "
                          f"max={cs.max()} feats={X.shape[1] - 1}")
                except Exception as e:
                    print(f"{n}: LOAD FAILED: {type(e).__name__}: {e}")
        print(f"\n{len(names)} datasets x {args.seeds} seeds x "
              f"{len(args.models)} models")
        return 0

    print(f"{len(names)} datasets, {args.seeds} seed(s), "
          f"models: {', '.join(args.models)}", flush=True)

    records, ds_meta = [], {}
    for ds in names:
        for seed in range(args.seed_offset,
                          args.seed_offset + args.seeds):
            t0 = time.time()
            meta, out = run_one(ds, seed, args.threads, args.models)
            ds_meta[ds] = meta
            for name, got in out.items():
                if got is None:
                    continue
                m, fit_s, pred_s, best = got
                records.append({"dataset": ds, "model": name, "seed": seed,
                                "metrics": m, "fit_time": fit_s,
                                "predict_time": pred_s, "best_iter": best})
            print(f"  {ds} seed {seed}: {time.time() - t0:.1f}s",
                  flush=True)

    rows, wins = aggregate(records, args.models)
    print()
    print(format_table(rows, wins))

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RESULTS_DIR, f"grouped-{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "config": {"seeds": args.seeds, "seed_offset": args.seed_offset,
                           "models": args.models, "timing": "fit_only",
                           "suite": "grouped"},
                "provenance": rb._provenance(sys.argv, {}),
                "datasets": ds_meta,
                "records": records,
            }, fh, indent=1)
        print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
