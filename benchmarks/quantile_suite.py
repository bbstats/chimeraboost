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

With `--decide` it runs the full decision tier for quantiles: Grinsztajn
regression plus high-cardinality regression, their `@sus25` / `@sus50`
small-data twins, and the `@time` twins of the hc regressions.

Row pairing. From 2026-09-23 on, each dataset builder is called with
``np.random.default_rng(1000 + seed)`` and split exactly the way
``run_benchmarks._run_seed_task`` splits it -- the 75/25 random split, or the
temporal window for `@time` twins, then the `@sus` train shrink -- so these
runs are row-paired with the point-model `--decide` runs. The 2026-08-30 JSON
called the builders with ``default_rng(seed)``, which changes only builders
that draw from the stream; the 36 Grinsztajn base keys draw nothing, and
re-run on 2026-09-23 they came back bit-identical for every arm but LightGBM
(`QUANTILE_PLAN.md`, Q-B4).

Deliberately NOT wired into `run_benchmarks.py --decide`. That tier is
protocol-gated, and a third task kind would ripple through the variant
families, the per-stratum sign tests and the Pareto panels for no gain. This
mirrors the harness's suite registration and per-(dataset, seed) data path
instead, and stays a separate script.

Arms
----
ChimeraBoostQuantile   the head: one booster, K-vector leaves
ChimeraBoostPerLevel   K independent `ChimeraBoostRegressor(loss="Quantile")`
                       -- the in-house baseline, which is what the shared
                       structure has to justify itself against
LightGBMPerLevel       K independent LightGBM quantile boosters
CatBoostMultiQuantile  CatBoost `MultiQuantile`, the same idea as ours
RigidShift             the trivial conditional-location baseline from LEAFTUNE
                       P10-P14: one default squared-error fit, plus the
                       empirical quantiles of its validation residuals
                       (`np.quantile` with numpy's default method) added to
                       every row -- one width for every row
ChimeraBoostQuantileCQR  the head with `conformalize=True`
NGBoost                `NGBRegressor(Dist=Normal)` on the shared split,
                       categoricals ordinal-encoded, quantiles from the
                       fitted Normal

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
import summarize  # noqa: E402  (stratum labels for --list-datasets)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import (ChimeraBoostQuantileRegressor,  # noqa: E402
                          ChimeraBoostRegressor)
from chimeraboost import quantile_metrics as qm  # noqa: E402
from chimeraboost.quantile_api import (
    _centre, _cqr_scales, _median_index)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

# The head's own default. Nineteen symmetric levels, so every central interval
# from 90% down to 10% is an adjacent column pair.
TAUS = np.round(np.arange(0.05, 0.9501, 0.05), 10)

# Reported alphas: the ones a user actually asks for.
ALPHAS = (0.1, 0.2, 0.5)

# The uncapped probe's budget: four times the shared cap, so a head
# that stopped for lack of rounds has room to show it.
UNCAPPED_ITERS = 8000


def _fit_head_model(split, cat, threads, taus, n_estimators, depth=None):
    """Construct and fit the suite's head.

    Shared by the head arm and the ValScaled probe so the two cannot
    drift apart: same constructor, same rows, same early-stopping
    split. ``depth=None`` is the head's own default, as before.
    """
    Xf, Xv, yf, yv = split
    m = ChimeraBoostQuantileRegressor(
        quantiles=taus, n_estimators=n_estimators,
        early_stopping_rounds=rb.PATIENCE, thread_count=threads,
        random_state=0, depth=depth)
    m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
    return m


def _fit_chimera_head(split, Xte, cat, threads, taus):
    t = time.time()
    m = _fit_head_model(split, cat, threads, taus, rb.MAX_ITERS)
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


def _fit_rigid_shift(split, Xte, cat, threads, taus):
    """One squared-error fit plus the empirical quantiles of its residuals.

    The trivial conditional-location baseline from LEAFTUNE P10-P14: the
    point model predicts the location and every row gets the SAME width --
    the quantiles of the validation residuals, added as a constant offset
    vector. One width for every row. Any head that conditions width on x has
    to beat this to justify its structure. Offsets use `np.quantile` with
    numpy's default method (linear interpolation).
    """
    Xf, Xv, yf, yv = split
    m = ChimeraBoostRegressor(n_estimators=rb.MAX_ITERS,
                              early_stopping_rounds=rb.PATIENCE,
                              thread_count=threads, random_state=0)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
    fit_s = time.time() - t
    t = time.time()
    pred_val = np.asarray(m.predict(Xv), dtype=np.float64).ravel()
    offsets = np.quantile(np.asarray(yv, dtype=np.float64) - pred_val, taus)
    pred_test = np.asarray(m.predict(Xte), dtype=np.float64).ravel()
    Q = pred_test[:, None] + offsets[None, :]
    return Q, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_cqr(split, Xte, cat, threads, taus):
    """The head exactly as `_fit_chimera_head`, with `conformalize=True`."""
    Xf, Xv, yf, yv = split
    m = ChimeraBoostQuantileRegressor(
        quantiles=taus, n_estimators=rb.MAX_ITERS,
        early_stopping_rounds=rb.PATIENCE, thread_count=threads,
        random_state=0, conformalize=True)
    t = time.time()
    m.fit(Xf, yf, cat_features=cat or None, eval_set=(Xv, yv))
    fit_s = time.time() - t
    t = time.time()
    Q = m.predict(Xte)
    return Q, fit_s, time.time() - t, m.best_iteration_


def _fit_ngboost(split, Xte, cat, threads, taus):
    """NGBoost with a Normal predictive distribution, on the shared split.

    The import lives inside the arm: NGBoost is a benchmark opponent, never
    a library dependency, so a missing install must skip only this arm.
    """
    import ngboost
    from ngboost.learners import default_tree_learner
    from sklearn.base import clone
    from sklearn.preprocessing import OrdinalEncoder
    Xf, Xv, yf, yv = split
    cat_idx = sorted(set(cat)) if cat else []
    num_idx = [i for i in range(Xf.shape[1]) if i not in set(cat_idx)]

    def _num(X):
        if not num_idx:
            return np.empty((X.shape[0], 0), dtype=np.float64)
        # NaN left as NaN: scikit-learn 1.8 trees accept it.
        return np.asarray(X[:, num_idx], dtype=np.float64)

    if cat_idx:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1, encoded_missing_value=-1)
        enc.fit(np.asarray(Xf[:, cat_idx], dtype=object))

        def _enc(X):
            return enc.transform(np.asarray(X[:, cat_idx], dtype=object))

        Xf_in = np.hstack([_num(Xf), _enc(Xf)])
        Xv_in = np.hstack([_num(Xv), _enc(Xv)])
        Xte_in = np.hstack([_num(Xte), _enc(Xte)])
    else:
        Xf_in, Xv_in, Xte_in = _num(Xf), _num(Xv), _num(Xte)

    # NGBRegressor's random_state never reaches its base learner (the
    # default tree has random_state=None), so ties break at random and reruns
    # drift. Clone the default learner with a fixed seed, keeping every other
    # parameter.
    base = clone(default_tree_learner).set_params(random_state=0)
    m = ngboost.NGBRegressor(Dist=ngboost.distns.Normal,
                             n_estimators=rb.MAX_ITERS,
                             early_stopping_rounds=rb.PATIENCE,
                             Base=base, random_state=0, verbose=False)
    t = time.time()
    m.fit(Xf_in, yf, X_val=Xv_in, Y_val=yv)
    fit_s = time.time() - t
    t = time.time()
    best = getattr(m, "best_val_loss_itr", None)
    # NGBoost stops `early_stopping_rounds` AFTER its best validation round
    # and keeps those trees, while every other arm predicts at its best
    # round -- so score it at its best round too. `best` is a 0-based index
    # and `pred_param` breaks before applying tree `max_iter`, hence + 1
    # (which is also why `best=0` still works: `max_iter` is tested for
    # truthiness, so a bare 0 would silently use all trees).
    if best is not None:
        dist = m.pred_dist(Xte_in, max_iter=int(best) + 1)
    else:
        dist = m.pred_dist(Xte_in)
    # ppf takes one level at a time (a vector argument does not broadcast
    # against the per-row parameters), so the (n, K) grid is stacked.
    Q = np.column_stack([np.asarray(dist.ppf(float(tau)), dtype=np.float64)
                         for tau in taus])
    pred_s = time.time() - t
    return Q, fit_s, pred_s, int(best) if best is not None else None


ARMS = {
    "ChimeraBoostQuantile": _fit_chimera_head,
    "ChimeraBoostPerLevel": _fit_chimera_per_level,
    "LightGBMPerLevel": _fit_lightgbm_per_level,
    "CatBoostMultiQuantile": _fit_catboost_mq,
    "RigidShift": _fit_rigid_shift,
    "ChimeraBoostQuantileCQR": _fit_chimera_cqr,
    "NGBoost": _fit_ngboost,
}


def _fit_chimera_uncapped(split, Xte, cat, threads, taus):
    """The head with the round cap lifted to `UNCAPPED_ITERS`.

    In Q-B4 the head reached the 2000-round cap on 11 of 36 Grinsztajn
    sets and lost them; this asks whether that loss is truncation.
    """
    t = time.time()
    m = _fit_head_model(split, cat, threads, taus, UNCAPPED_ITERS)
    fit_s = time.time() - t
    t = time.time()
    Q = m.predict(Xte)
    return Q, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_depth6(split, Xte, cat, threads, taus):
    """The head at depth 6.

    CatBoost MultiQuantile's default depth; the head's 4 was never
    measured against CRPS on real data.
    """
    t = time.time()
    m = _fit_head_model(split, cat, threads, taus, rb.MAX_ITERS,
                        depth=6)
    fit_s = time.time() - t
    t = time.time()
    Q = m.predict(Xte)
    return Q, fit_s, time.time() - t, m.best_iteration_


def _fit_chimera_recentred(split, Xte, cat, threads, taus):
    """The head's shape on a squared-error centre.

    The head's grid shifted row by row onto RigidShift's predicted
    median. The shift is constant along each row, so rows stay ordered.
    """
    Q_head, fit_h, pred_h, best = _fit_chimera_head(
        split, Xte, cat, threads, taus)
    Q_rigid, fit_r, pred_r, _ = _fit_rigid_shift(
        split, Xte, cat, threads, taus)
    mi, mw = _median_index(np.asarray(taus))
    shift = (_centre(Q_rigid, mi, mw) - _centre(Q_head, mi, mw))[:, None]
    return Q_head + shift, fit_h + fit_r, pred_h + pred_r, best


def _fit_chimera_valscaled(split, Xte, cat, threads, taus):
    """The head with the library's own CQR factors, fit on the
    validation rows.

    The factors come from the shared early-stopping validation rows
    instead of a carved 20% fold, so no training rows are given up;
    that fold also chose the stopping round, a mild optimism the
    RigidShift offsets share.
    """
    Xf, Xv, yf, yv = split
    t = time.time()
    m = _fit_head_model(split, cat, threads, taus, rb.MAX_ITERS)
    m.conformal_scale_ = _cqr_scales(
        m.predict(Xv), np.asarray(yv, dtype=np.float64), m.quantiles_,
        *m._median_idx_)
    fit_s = time.time() - t
    t = time.time()
    Q = m.predict(Xte)
    return Q, fit_s, time.time() - t, m.best_iteration_


PROBES = {
    "ChimeraBoostQuantileUncapped": _fit_chimera_uncapped,
    "ChimeraBoostQuantileDepth6": _fit_chimera_depth6,
    "ChimeraBoostQuantileRecentred": _fit_chimera_recentred,
    "ChimeraBoostQuantileValScaled": _fit_chimera_valscaled,
}


def _register_for_keys(names, decide):
    """Register every suite the requested keys need. Idempotent.

    Without ``decide`` the registration follows the keys: ``gr:`` keys need
    Grinsztajn, ``hc:`` keys need high-card, and any ``@`` twin needs the
    variant pass after its base suite -- so an explicit ``--datasets``
    selection runs the suites it names instead of only Grinsztajn. With
    ``decide`` this is the full decision tier. The parent calls this before
    validating the keys, and every worker calls it for its own key, since
    workers spawn fresh on Windows and register the suites themselves.
    """
    if decide:
        rb._add_grinsztajn_datasets()
        rb._add_highcard_datasets()
        rb._add_variant_datasets(list(rb.DATASETS))
        return
    if any(k.startswith("gr:") for k in names):
        rb._add_grinsztajn_datasets()
    if any(k.startswith("hc:") for k in names):
        rb._add_highcard_datasets()
    if any(rb.VARIANT_SEP in k for k in names):
        rb._add_variant_datasets(list(rb.DATASETS))


def select_datasets(decide):
    """Dataset keys for the run.

    Without ``decide`` this is today's default: every Grinsztajn regression
    base key. With it, every REGRESSION key across the gr:/hc: suites and
    their registered `@sus25`/`@sus50` twins, plus the `@time` twins of the
    hc: regressions -- registered the way the harness registers them, and
    judged by `rb._task_of`, never by hard-coded names.
    """
    if not decide:
        rb._add_grinsztajn_datasets()
        return sorted(
            k for k in rb.DATASETS
            if k.startswith("gr:") and rb._task_of(k) == "regression")
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    rb._add_variant_datasets(list(rb.DATASETS))
    out = []
    for k in rb.DATASETS:
        if not (k.startswith("gr:") or k.startswith("hc:")):
            continue
        if rb._task_of(k) != "regression":
            continue
        variant = (k.split(rb.VARIANT_SEP, 1)[1]
                   if rb.VARIANT_SEP in k else "")
        if variant in ("", "sus25", "sus50"):
            out.append(k)
        elif variant == "time" and k.startswith("hc:"):
            out.append(k)
    return sorted(out)


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
    """One (dataset, seed) draw, on the harness's data path, exactly.

    Returns (meta, out). A degenerate `@time` window returns (None, {}) with
    a printed note, as the harness does.
    """
    X, y, cat, _ = rb.DATASETS[ds_name](1, np.random.default_rng(1000 + seed))
    variant = (ds_name.split(rb.VARIANT_SEP, 1)[1]
               if rb.VARIANT_SEP in ds_name else "")
    if variant == "time":
        win = rb._temporal_split(X, y, seed, "regression")
        if win is None:
            print(f"  [skip] {ds_name} (seed {seed}): temporal window "
                  "degenerate (fewer than 2 training or test rows)")
            return None, {}
        Xtr, Xte, ytr, yte = win
    else:
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=seed)
        if variant in rb.SUS_FRACTIONS:
            # Shrink TRAINING rows only, so the twin keeps the parent's test
            # rows -- the harness's convention.
            Xtr, ytr = rb._subsample_train(Xtr, ytr,
                                           rb.SUS_FRACTIONS[variant],
                                           "regression")
    # One early-stopping split, shared by every arm, so no model is judged on
    # more data than another. Same carve the harness uses.
    split = rb._val_split(Xtr, ytr, "regression", 0)

    meta = {"task": "quantile", "n_train": int(Xtr.shape[0]),
            "n_total": int(X.shape[0]), "n_features": int(X.shape[1]),
            "has_cats": bool(cat), "variant": variant or None,
            "y_std": float(np.std(y)), "y_std_test": float(np.std(yte)),
            # The no-skill CRPS: what the unconditional grid scores. This is
            # the quantile twin of y_std / class_prior, so a reader can form a
            # skill score without rebuilding the dataset.
            "crps_marginal": float(qm.crps(
                yte, qm.marginal_grid(split[2], taus, len(yte)), taus))}

    out = {}
    for name in models:
        try:
            Q, fit_s, pred_s, best = {**ARMS, **PROBES}[name](
                split, Xte, cat, threads, taus)
            out[name] = (score(yte, Q, taus, split[2]), fit_s, pred_s, best)
        except Exception as e:
            # Same convention as run_benchmarks: a model that structurally
            # cannot handle a dataset is recorded as skipped, not allowed to
            # abort the run.
            print(f"  [skip] {name} on {ds_name} (seed {seed}): "
                  f"{type(e).__name__}: {e}")
            out[name] = None
    return meta, out


def _run_seed_task_quantile(task):
    """Fit every requested arm on one (dataset, seed) draw. Top-level and
    picklable so it can run in a worker process -- spawn on Windows, so each
    worker registers the suites itself. Returns (ds, seed, meta, out); a
    failing task is reported as a skip, never allowed to kill the run."""
    ds_name, seed, taus, threads, models, decide = task
    _register_for_keys([ds_name], decide)
    try:
        meta, out = run_one(ds_name, seed, taus, threads, models)
    except Exception as e:
        print(f"  [skip] {ds_name} (seed {seed}): {type(e).__name__}: {e}",
              flush=True)
        return ds_name, seed, None, {}
    return ds_name, seed, meta, out


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


def _print_dataset_list(names, args, taus):
    """The --list-datasets page: the selection grouped by stratum (suite x
    variant) with counts, the way the harness prints it."""
    strata = summarize.split_strata(names)
    for stratum, ds_names in strata.items():
        print(f"\n{summarize.stratum_label(stratum)}  ({len(ds_names)})")
        for ds in ds_names:
            print(f"  {ds}")
    n_str = len(strata)
    print(f"\ntotal: {len(names)} datasets in {n_str} "
          f"{'stratum' if n_str == 1 else 'strata'} x {args.seeds} "
          f"seed(s) x {len(args.models)} models, K={len(taus)}")


def _run_serial(tasks):
    """Today's serial behaviour: every (dataset, seed) task inline."""
    collected = {}  # (ds, seed) -> (meta, out)
    for t in tasks:
        ds, s = t[0], t[1]
        t0 = time.time()
        _, seed, meta, out = _run_seed_task_quantile(t)
        collected[(ds, seed)] = (meta, out)
        # Flushed: a full run is long enough that a buffered log is
        # indistinguishable from a hung process.
        print(f"  {ds} seed {s}: {time.time() - t0:.1f}s",
              flush=True)
    return collected


def _run_parallel(tasks, jobs):
    """The same tasks across a process pool, like run_benchmarks."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    collected = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_seed_task_quantile, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                ds, seed, meta, out = fut.result()
            except Exception as e:
                ds, seed = futs[fut][0], futs[fut][1]
                print(f"  [skip] {ds} (seed {seed}): "
                      f"{type(e).__name__}: {e}", flush=True)
                collected[(ds, seed)] = (None, {})
                continue
            collected[(ds, seed)] = (meta, out)
            print(f"  {ds} seed {seed} done", flush=True)
    return collected


def _assemble_records(names, seeds, models, collected):
    """Records in a deterministic order -- dataset, then seed, then model --
    whatever order the workers finished in."""
    records, ds_meta = [], {}
    for ds in names:
        for seed in range(seeds):
            got_all = collected.get((ds, seed))
            if got_all is None:
                continue
            meta, out = got_all
            if meta is None:
                continue  # skipped task (degenerate window or failure)
            ds_meta[ds] = meta
            for name in models:
                got = out.get(name)
                if got is None:
                    continue
                m, fit_s, pred_s, best = got
                records.append({"dataset": ds, "model": name, "seed": seed,
                                "metrics": m, "fit_time": fit_s,
                                "predict_time": pred_s, "best_iter": best})
    return records, ds_meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                    help="total thread budget across all parallel jobs "
                         "(None = all cores).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="(dataset, seed) tasks to run in parallel processes; "
                         "each gets threads/jobs threads (default: 1, serial).")
    ap.add_argument("--decide", action="store_true",
                    help="the decision tier: Grinsztajn + high-card regression "
                         "suites with their @sus25/@sus50 twins and the @time "
                         "twins of the hc: regressions.")
    ap.add_argument("--models", nargs="+", default=list(ARMS),
                    choices=list({**ARMS, **PROBES}))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys; default = every Grinsztajn regression "
                         "set (--decide: the full decision tier)")
    ap.add_argument("--quantiles", type=float, nargs="+", default=None,
                    help="tau grid; default = the head's own 19 levels")
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    names = args.datasets or select_datasets(args.decide)
    _register_for_keys(names, args.decide)
    unknown = [k for k in names if k not in rb.DATASETS]
    if unknown:
        ap.error(f"Unknown datasets: {sorted(unknown)} "
                 f"({len(rb.DATASETS)} registered).")
    taus = (np.asarray(args.quantiles, dtype=np.float64) if args.quantiles
            else TAUS)

    if args.list_datasets:
        _print_dataset_list(names, args, taus)
        return 0

    # Split the thread budget across parallel jobs: GBDT thread scaling is
    # sublinear, so running J tasks at threads/J each beats one fit at all
    # cores. Same convention as run_benchmarks.
    jobs = max(1, args.jobs)
    total_threads = args.threads or os.cpu_count() or 1
    threads_per = max(1, total_threads // jobs)

    print(f"{len(names)} datasets, {args.seeds} seed(s), K={len(taus)} levels, "
          f"jobs={jobs} threads/job={threads_per}, "
          f"models: {', '.join(args.models)}", flush=True)

    tasks = [(ds, s, taus, threads_per, args.models, args.decide)
             for ds in names for s in range(args.seeds)]
    collected = (_run_serial(tasks) if jobs == 1
                 else _run_parallel(tasks, jobs))
    records, ds_meta = _assemble_records(names, args.seeds, args.models,
                                         collected)

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
                           "timing": "fit_only", "suite": "quantile",
                           "decide": args.decide, "jobs": jobs,
                           "threads_per_model": threads_per},
                "provenance": rb._provenance(sys.argv, {}),
                "datasets": ds_meta,
                "records": records,
            }, fh, indent=1)
        print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
