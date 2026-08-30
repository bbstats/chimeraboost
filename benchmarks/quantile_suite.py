"""Real-data quantile benchmark: the shared-tree head against the field.

Why this exists. The multi-quantile head shipped in 0.26.0 and had never been
scored on real data. `quantile_head.py` runs on synthetic draws against
LightGBM only; `probe_quantile_band.py` runs three datasets. Neither has ever
run CatBoost's `MultiQuantile`, which `docs/quantiles.md` names as the design
precedent, and neither emits a `run_benchmarks`-shaped JSON, so
`compare_runs.py` cannot sign-test either of them.

This runs the Grinsztajn regression suite -- the same datasets the decision
tier uses -- and writes the harness's own JSON shape, so the existing analysis
stack works on the output unchanged:

    python benchmarks/quantile_suite.py --seeds 3 --save
    python benchmarks/compare_runs.py BASE.json NEW.json --metric crps

Deliberately NOT wired into `run_benchmarks.py --decide`. That tier is
protocol-gated, and a third task kind would ripple through the variant
families, the per-stratum sign tests and the Pareto panels for no gain. This
borrows the harness's dataset registry and its split, and nothing else.

Arms
----
ChimeraBoostQuantile   the head: one booster, K-vector leaves
ChimeraBoostPerLevel   K independent `ChimeraBoostRegressor(loss="Quantile")`
                       -- the in-house baseline, which is what the shared
                       structure has to justify itself against
LightGBMPerLevel       K independent LightGBM quantile boosters
CatBoostMultiQuantile  CatBoost `MultiQuantile`, the same idea as ours

Scoring is `chimeraboost.quantile_metrics`, so the numbers here and the ones a
user reads from `model.report()` are the same numbers.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402  (the repo's one dataset loader)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import (ChimeraBoostQuantileRegressor,  # noqa: E402
                          ChimeraBoostRegressor)
from chimeraboost import quantile_metrics as qm  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

# The head's own default. Nineteen symmetric levels, so every central interval
# from 90% down to 10% is an adjacent column pair.
TAUS = np.round(np.arange(0.05, 0.9501, 0.05), 10)

# Reported alphas: the ones a user actually asks for.
ALPHAS = (0.1, 0.2, 0.5)


def _fit_chimera_head(split, Xte, cat, threads, taus):
    Xf, Xv, yf, yv = split
    m = ChimeraBoostQuantileRegressor(
        quantiles=taus, n_estimators=rb.MAX_ITERS,
        early_stopping_rounds=rb.PATIENCE, thread_count=threads,
        random_state=0)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
    fit_s = time.time() - t
    t = time.time()
    Q = m.predict(Xte)
    return Q, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_per_level(split, Xte, cat, threads, taus):
    """K independent `loss="Quantile"` boosters, sharing nothing.

    The comparison the head has to justify itself against: same budget per
    level, K times the work, and no structural reason the levels come out
    ordered.
    """
    Xf, Xv, yf, yv = split
    cols, fit_s, pred_s, iters = [], 0.0, 0.0, []
    for tau in taus:
        m = ChimeraBoostRegressor(
            loss="Quantile", alpha=float(tau), n_estimators=rb.MAX_ITERS,
            early_stopping_rounds=rb.PATIENCE, thread_count=threads,
            random_state=0)
        t = time.time()
        m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
        fit_s += time.time() - t
        t = time.time()
        cols.append(np.asarray(m.predict(Xte), dtype=np.float64).ravel())
        pred_s += time.time() - t
        iters.append(m.best_iteration_)
    return (np.column_stack(cols), fit_s, pred_s,
            int(np.mean([i for i in iters if i is not None] or [0])))


def _fit_lightgbm_per_level(split, Xte, cat, threads, taus):
    import lightgbm as lgb
    Xf, Xv, yf, yv = split
    if cat:
        Xf_in, Xv_in, Xte_in = rb._lgb_prepare(Xf, Xv, Xte, list(cat))
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte

    cols, fit_s, pred_s, iters = [], 0.0, 0.0, []
    for tau in taus:
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau),
                              n_estimators=rb.MAX_ITERS, n_jobs=threads or -1,
                              random_state=0, verbosity=-1)
        fit_kw = dict(eval_set=[(Xv_in, yv)],
                      callbacks=[lgb.early_stopping(rb.PATIENCE,
                                                    verbose=False)])
        if cat:
            fit_kw["categorical_feature"] = list(cat)
        t = time.time()
        m.fit(Xf_in, yf, **fit_kw)
        fit_s += time.time() - t
        t = time.time()
        cols.append(np.asarray(m.predict(Xte_in), dtype=np.float64).ravel())
        pred_s += time.time() - t
        iters.append(m.best_iteration_)
    return (np.column_stack(cols), fit_s, pred_s,
            int(np.mean([i for i in iters if i is not None] or [0])))


def _fit_catboost_mq(split, Xte, cat, threads, taus):
    """CatBoost's own shared-head design, the one `docs/quantiles.md` cites.

    `MultiQuantile` early-stops on itself, so this arm gets the same budget
    and the same validation rows as every other -- fit times here are
    like-for-like.
    """
    from catboost import CatBoostRegressor
    Xf, Xv, yf, yv = split
    levels = ",".join(f"{t:g}" for t in taus)
    m = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={levels}",
        allow_writing_files=False, iterations=rb.MAX_ITERS,
        early_stopping_rounds=rb.PATIENCE,
        thread_count=threads or -1, random_seed=0, verbose=False)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
    fit_s = time.time() - t
    t = time.time()
    Q = np.asarray(m.predict(Xte), dtype=np.float64)
    return Q, fit_s, time.time() - t, int(m.tree_count_)


ARMS = {
    "ChimeraBoostQuantile": _fit_chimera_head,
    "ChimeraBoostPerLevel": _fit_chimera_per_level,
    "LightGBMPerLevel": _fit_lightgbm_per_level,
    "CatBoostMultiQuantile": _fit_catboost_mq,
}


def score(y, Q, taus, y_train):
    """Every number this benchmark judges on, from the shipped scorer.

    `primary` is negated CRPS so that higher is better, which is the
    convention `compare_runs.py` and `summarize.py` already assume for
    regression (`-rmse`).
    """
    Q = np.asarray(Q, dtype=np.float64)
    rep = qm.quantile_report(y, Q, taus, baseline=y_train)
    out = {
        "primary": -rep["crps"],
        "crps": rep["crps"],
        "crps_skill": rep["skill"],
        "crossing_rate": rep["crossing_rate"],
        "pinball_median": float(rep["pinball"][len(taus) // 2]),
    }
    by_nominal = {round(iv["nominal"], 4): iv for iv in rep["intervals"]}
    for a in ALPHAS:
        iv = by_nominal.get(round(1.0 - a, 4))
        if iv is None:
            continue
        tag = f"{int(round((1 - a) * 100))}"
        out[f"coverage_{tag}"] = iv["coverage"]
        out[f"width_{tag}"] = iv["width"]
        out[f"interval_score_{tag}"] = iv["score"]
    return out


def run_one(ds_name, seed, taus, threads, models):
    X, y, cat, _ = rb.DATASETS[ds_name](1, np.random.default_rng(seed))
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    # One early-stopping split, shared by every arm, so no model is judged on
    # more data than another. Same carve the harness uses.
    split = rb._val_split(Xtr, ytr, "regression", 0)

    meta = {"task": "quantile", "n_train": int(Xtr.shape[0]),
            "n_total": int(X.shape[0]), "n_features": int(X.shape[1]),
            "has_cats": bool(cat), "variant": None,
            "y_std": float(np.std(y)), "y_std_test": float(np.std(yte)),
            # The no-skill CRPS: what the unconditional grid scores. This is
            # the quantile twin of y_std / class_prior, so a reader can form a
            # skill score without rebuilding the dataset.
            "crps_marginal": float(qm.crps(
                yte, qm.marginal_grid(split[2], taus, len(yte)), taus))}

    out = {}
    for name in models:
        try:
            Q, fit_s, pred_s, best = ARMS[name](split, Xte, cat, threads,
                                                taus)
            out[name] = (score(yte, Q, taus, split[2]), fit_s, pred_s, best)
        except Exception as e:
            # Same convention as run_benchmarks: a model that structurally
            # cannot handle a dataset is recorded as skipped, not allowed to
            # abort the run.
            print(f"  [skip] {name} on {ds_name} (seed {seed}): "
                  f"{type(e).__name__}: {e}")
            out[name] = None
    return meta, out


def aggregate(records, models):
    """Mean of each metric per model, over every (dataset, seed) it ran."""
    keys = ["crps", "crps_skill", "coverage_90", "width_90",
            "interval_score_90", "coverage_80", "crossing_rate"]
    rows = {}
    for m in models:
        vals = [r["metrics"] for r in records if r["model"] == m]
        if not vals:
            continue
        rows[m] = {k: float(np.mean([v[k] for v in vals if k in v]))
                   for k in keys if any(k in v for v in vals)}
        rows[m]["fit_s"] = float(np.mean(
            [r["fit_time"] for r in records if r["model"] == m]))
        rows[m]["n"] = len(vals)
    return rows


def format_table(rows, base="ChimeraBoostQuantile"):
    head = (f"{'model':24s}{'CRPS':>10s}{'skill':>9s}{'cov90':>8s}"
            f"{'width90':>10s}{'IS90':>10s}{'cross':>8s}{'fit s':>9s}"
            f"{'vs head':>9s}")
    lines = [head, "-" * len(head)]
    ref = rows.get(base, {}).get("fit_s")
    for m, r in sorted(rows.items(), key=lambda kv: kv[1].get("crps", 9e9)):
        rel = (f"{r['fit_s'] / ref:8.2f}x" if ref else "        -")
        lines.append(
            f"{m:24s}{r.get('crps', float('nan')):10.4f}"
            f"{r.get('crps_skill', float('nan')):9.3f}"
            f"{r.get('coverage_90', float('nan')):8.3f}"
            f"{r.get('width_90', float('nan')):10.4f}"
            f"{r.get('interval_score_90', float('nan')):10.4f}"
            f"{r.get('crossing_rate', float('nan')):8.4f}"
            f"{r['fit_s']:9.2f}{rel:>9s}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                    help="thread budget per model (None = all cores).")
    ap.add_argument("--models", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys; default = every Grinsztajn regression "
                         "set")
    ap.add_argument("--quantiles", type=float, nargs="+", default=None,
                    help="tau grid; default = the head's own 19 levels")
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    rb._add_grinsztajn_datasets()
    names = args.datasets or sorted(
        k for k in rb.DATASETS
        if k.startswith("gr:") and rb._task_of(k) == "regression")
    taus = (np.asarray(args.quantiles, dtype=np.float64) if args.quantiles
            else TAUS)

    if args.list_datasets:
        for n in names:
            print(n)
        print(f"\n{len(names)} datasets x {args.seeds} seeds x "
              f"{len(args.models)} models, K={len(taus)}")
        return 0

    print(f"{len(names)} datasets, {args.seeds} seed(s), K={len(taus)} levels, "
          f"models: {', '.join(args.models)}", flush=True)

    records, ds_meta = [], {}
    for ds in names:
        for seed in range(args.seeds):
            t0 = time.time()
            meta, out = run_one(ds, seed, taus, args.threads, args.models)
            ds_meta[ds] = meta
            for name, got in out.items():
                if got is None:
                    continue
                m, fit_s, pred_s, best = got
                records.append({"dataset": ds, "model": name, "seed": seed,
                                "metrics": m, "fit_time": fit_s,
                                "predict_time": pred_s, "best_iter": best})
            # Flushed: a full run is long enough that a buffered log is
            # indistinguishable from a hung process.
            print(f"  {ds} seed {seed}: {time.time() - t0:.1f}s",
                  flush=True)

    rows = aggregate(records, args.models)
    print()
    print(format_table(rows))

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RESULTS_DIR, f"quantile-{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "config": {"seeds": args.seeds, "models": args.models,
                           "quantiles": [float(t) for t in taus],
                           "timing": "fit_only", "suite": "quantile"},
                "provenance": rb._provenance(sys.argv, {}),
                "datasets": ds_meta,
                "records": records,
            }, fh, indent=1)
        print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
