"""Synthetic quantile screen with known true quantiles (Q-B2).

Why this exists. `quantile_suite.py` scores the head on real data, where the
truth is unknown and every comparison carries real-data noise. Here the TRUE
conditional quantiles are known -- in closed form, or to 1e-10 by root-finding
the mixture CDF for the bimodal regime -- so a change scores as excess CRPS
over the oracle, CRPS(model) - CRPS(oracle) on the same test rows. A mechanism
shows up without waiting on 59 datasets.

Why not `quantile_head.py`. That screen fits a FIXED round count, and the
2026-08-30 lesson is that fixed-round synthetic fits flatter the head. Every
arm here early-stops exactly as `quantile_suite.py` fits it -- the same
`ARMS[name](split, Xte, cat, threads, taus)` call on the shared
`run_benchmarks._val_split` -- and the noise adds skew, heavy tails and
bimodality on top of the Gaussian location/scale cases.

    python benchmarks/quantile_synth.py --seeds 1 --sizes 1000 --jobs 3
    python benchmarks/quantile_synth.py --seeds 3 --save   # the full screen

Regimes (mu(x) = 2*x0 + sin(2*x2) + 0.5*x3*x4 in all of them; X is n x 8
standard normal, plus one string categorical column for catscale):

  location  y = mu + N(0, 1)
  hetero    y = mu + exp(0.6*x1) * N(0, 1)
  skewed    y = mu + s(x) * (Gamma(k=2, th=1) - 2), s = 0.5 + 1.5*1[x1 > 0]
  heavy     y = mu + s(x) * t(df=3), with the same s(x) as skewed
  bimodal   y = mu +/- 2 (P(+) = sigmoid(2*x1)) + N(0, 0.5^2)
  catscale  y = mu + sigma_c * N(0, 1), sigma_c from 0.2 to 3 across 20
            uniformly-drawn string levels -- scale carried by a categorical,
            the high-card case in small

The random stream. Each (regime, n, seed) draw uses
``np.random.default_rng(10_000 + seed)`` and draws X first, so X is shared
across regimes at the same (n, seed) and nested across sizes (the n=1000
rows are the n=10000 prefix). Keys are ``qsyn:<regime>/n<N>``.

Output. With --save, the harness JSON shape (records/datasets/config/
provenance) at ``benchmarks/results/quantile-synth-<stamp>.json``, readable
by ``compare_runs.py --metric crps``. Records carry ``excess_crps``; dataset
meta carries ``oracle_crps`` (last seed wins, the suite's own convention for
seed-dependent meta -- any seed's floor is recoverable from its records as
crps - excess_crps).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import expit
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm
from scipy.stats import t as t_dist
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb
import quantile_suite as qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chimeraboost import quantile_metrics as qm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

# The grid is the suite's grid, not a copy of it: models and oracles are only
# comparable on the same levels.
TAUS = qs.TAUS

SIZES = (1000, 10000)
DEFAULT_SEEDS = 3

# catscale: 20 string levels, sigma from 0.2 to 3 in level order. The mapping
# is fixed -- level i always means sigma i, on every seed and size.
CAT_LEVELS = tuple(f"L{i:02d}" for i in range(20))
CAT_SIGMAS = tuple(np.linspace(0.2, 3.0, len(CAT_LEVELS)))


def _mu(X):
    """The shared conditional centre, on the 8 numeric columns."""
    X = np.asarray(X, dtype=np.float64)
    return (2.0 * X[:, 0] + np.sin(2.0 * X[:, 2])
            + 0.5 * X[:, 3] * X[:, 4])


def _spread(x1):
    """The two-level scale both non-Gaussian one-sided regimes share."""
    return 0.5 + 1.5 * (np.asarray(x1) > 0)


def _base_stream(n, seed):
    """The documented stable stream: seed 10_000 + seed, X drawn first."""
    rng = np.random.default_rng(10_000 + seed)
    return rng, rng.standard_normal((n, 8))


def regime_location(n, seed, taus=TAUS):
    """Homoscedastic Gaussian noise; only the centre moves with x."""
    rng, X = _base_stream(n, seed)
    mu = _mu(X)
    y = mu + rng.standard_normal(n)
    Q = mu[:, None] + norm.ppf(np.asarray(taus))[None, :]
    return X, y, Q


def regime_hetero(n, seed, taus=TAUS):
    """Smooth heteroscedastic Gaussian noise: sd = exp(0.6*x1)."""
    rng, X = _base_stream(n, seed)
    mu = _mu(X)
    sd = np.exp(0.6 * X[:, 1])
    y = mu + sd * rng.standard_normal(n)
    Q = mu[:, None] + sd[:, None] * norm.ppf(np.asarray(taus))[None, :]
    return X, y, Q


def regime_skewed(n, seed, taus=TAUS):
    """Centred Gamma(2, 1) noise, scaled up where x1 > 0."""
    rng, X = _base_stream(n, seed)
    mu = _mu(X)
    s = _spread(X[:, 1])
    y = mu + s * (rng.standard_gamma(2.0, size=n) - 2.0)
    centred = gamma_dist.ppf(np.asarray(taus), a=2.0) - 2.0
    Q = mu[:, None] + s[:, None] * centred[None, :]
    return X, y, Q


def regime_heavy(n, seed, taus=TAUS):
    """Student-t(3) noise under the same two-level scale as skewed."""
    rng, X = _base_stream(n, seed)
    mu = _mu(X)
    s = _spread(X[:, 1])
    y = mu + s * rng.standard_t(3.0, size=n)
    Q = mu[:, None] + s[:, None] * t_dist.ppf(np.asarray(taus), df=3)[None, :]
    return X, y, Q


def bimodal_cdf(q0, p):
    """CDF of the de-meaned mixture: p*N(+2, 0.5^2) + (1-p)*N(-2, 0.5^2)."""
    q0 = np.asarray(q0, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (p * norm.cdf((q0 - 2.0) / 0.5)
            + (1.0 - p) * norm.cdf((q0 + 2.0) / 0.5))


def _bimodal_oracle(p, taus):
    """Per-row quantiles of the de-meaned mixture, by vectorized bisection.

    The bracket [-7, +7] holds every tau in (0, 1): F(-7) < 1e-23 and
    F(7) > 1 - 1e-23 at any mixing weight. Fifty halvings shrink the width-14
    bracket to 14/2^50 ~ 1e-14, past the 1e-10 asked.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    taus = np.asarray(taus, dtype=np.float64).ravel()
    Q0 = np.empty((p.shape[0], taus.shape[0]))
    for j, tau in enumerate(taus):
        lo = np.full(p.shape, -7.0)
        hi = np.full(p.shape, 7.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            below = bimodal_cdf(mid, p) < tau
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        Q0[:, j] = 0.5 * (lo + hi)
    return Q0


def regime_bimodal(n, seed, taus=TAUS):
    """Two modes at mu +/- 2, P(+) = sigmoid(2*x1), plus N(0, 0.5^2)."""
    rng, X = _base_stream(n, seed)
    mu = _mu(X)
    p = expit(2.0 * X[:, 1])
    sign = np.where(rng.random(n) < p, 1.0, -1.0)
    y = mu + sign * 2.0 + 0.5 * rng.standard_normal(n)
    Q = mu[:, None] + _bimodal_oracle(p, taus)
    return X, y, Q


def regime_catscale(n, seed, taus=TAUS):
    """Gaussian noise whose scale rides a 20-level string categorical."""
    rng, Xnum = _base_stream(n, seed)
    mu = _mu(Xnum)
    lvl = rng.integers(0, len(CAT_LEVELS), size=n)
    sig = np.asarray(CAT_SIGMAS)[lvl]
    y = mu + sig * rng.standard_normal(n)
    Q = mu[:, None] + sig[:, None] * norm.ppf(np.asarray(taus))[None, :]
    X = np.empty((n, Xnum.shape[1] + 1), dtype=object)
    X[:, :-1] = Xnum
    X[:, -1] = [CAT_LEVELS[i] for i in lvl]
    return X, y, Q


REGIMES = {
    "location": regime_location,
    "hetero": regime_hetero,
    "skewed": regime_skewed,
    "heavy": regime_heavy,
    "bimodal": regime_bimodal,
    "catscale": regime_catscale,
}


def dataset_key(regime, n):
    """Harness key for one synthetic dataset: ``qsyn:<regime>/n<N>``."""
    return f"qsyn:{regime}/n{n}"


def run_one(regime, n, seed, taus, threads, models):
    """One (regime, size, seed) draw, every requested arm on the suite's path.

    The 75/25 split is the suite's (same test_size and random_state, applied
    to row indices so the oracle rows stay aligned), and so is the shared
    early-stopping split. Each arm is fitted by the suite's own call.
    """
    X, y, Q_or = REGIMES[regime](n, seed, taus)
    key = dataset_key(regime, n)
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.25,
                                      random_state=seed)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    Qo_te = Q_or[idx_te]
    split = qs._SplitWithFull(rb._val_split(Xtr, ytr, "regression", 0),
                              (Xtr, ytr))
    cat = [X.shape[1] - 1] if regime == "catscale" else None
    oracle_crps = float(qm.crps(yte, Qo_te, taus))

    meta = {"task": "quantile", "n_train": int(Xtr.shape[0]),
            "n_total": int(n), "n_features": int(X.shape[1]),
            "has_cats": bool(cat), "variant": None,
            "y_std": float(np.std(y)), "y_std_test": float(np.std(yte)),
            "crps_marginal": float(qm.crps(
                yte, qm.marginal_grid(split[2], taus, len(yte)), taus)),
            "oracle_crps": oracle_crps}

    out = {}
    for name in models:
        try:
            res = {**qs.ARMS, **qs.PROBES}[name](
                split, Xte, cat, threads, taus)
            if len(res) == 5:
                Q, fit_s, pred_s, best, extra = res
            else:
                Q, fit_s, pred_s, best = res
                extra = None
            m = qs.score(yte, Q, taus, split[2])
            if extra:
                m.update(extra)
            m["excess_crps"] = float(m["crps"] - oracle_crps)
            out[name] = (m, fit_s, pred_s, best)
        except Exception as e:
            # Same convention as the suite: a model that cannot handle a
            # dataset is recorded as skipped, not allowed to abort the run.
            print(f"  [skip] {name} on {key} (seed {seed}): "
                  f"{type(e).__name__}: {e}")
            out[name] = None
    return meta, out


def _run_seed_task_synth(task):
    """Fit every requested arm on one (regime, size, seed) draw. Top-level and
    picklable so it can run in a worker process -- spawn on Windows, so each
    worker re-imports this module (and the suite with it) itself."""
    regime, n, seed, taus, threads, models = task
    key = dataset_key(regime, n)
    try:
        meta, out = run_one(regime, n, seed, taus, threads, models)
    except Exception as e:
        print(f"  [skip] {key} (seed {seed}): {type(e).__name__}: {e}",
              flush=True)
        return key, seed, None, {}
    return key, seed, meta, out


def _run_serial(tasks):
    """Today's serial behaviour: every (regime, size, seed) task inline."""
    collected = {}
    for t in tasks:
        key = dataset_key(t[0], t[1])
        t0 = time.time()
        _, seed, meta, out = _run_seed_task_synth(t)
        collected[(key, seed)] = (meta, out)
        print(f"  {key} seed {t[2]}: {time.time() - t0:.1f}s", flush=True)
    return collected


def _run_parallel(tasks, jobs):
    """The same tasks across a process pool, like run_benchmarks."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    collected = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_seed_task_synth, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            key = dataset_key(t[0], t[1])
            try:
                _, seed, meta, out = fut.result()
            except Exception as e:
                print(f"  [skip] {key} (seed {t[2]}): "
                      f"{type(e).__name__}: {e}", flush=True)
                collected[(key, t[2])] = (None, {})
                continue
            collected[(key, seed)] = (meta, out)
            print(f"  {key} seed {seed} done", flush=True)
    return collected


def _assemble_records(keys, seeds, models, collected):
    """Records in a deterministic order -- key, then seed, then model --
    whatever order the workers finished in."""
    records, ds_meta = [], {}
    for key in keys:
        for seed in range(seeds):
            got_all = collected.get((key, seed))
            if got_all is None:
                continue
            meta, out = got_all
            if meta is None:
                continue
            ds_meta[key] = meta
            for name in models:
                got = out.get(name)
                if got is None:
                    continue
                m, fit_s, pred_s, best = got
                records.append({"dataset": key, "model": name, "seed": seed,
                                "metrics": m, "fit_time": fit_s,
                                "predict_time": pred_s, "best_iter": best})
    return records, ds_meta


def _oracle_floor(records, key):
    """Mean oracle CRPS for a key over seeds, recovered from its records.

    Every arm on one (key, seed) shares the same test rows, so any record's
    crps - excess_crps is that seed's floor; the mean over seeds is what the
    excess column below sits above.
    """
    by_seed = {}
    for r in records:
        if r["dataset"] == key and r["seed"] not in by_seed:
            by_seed[r["seed"]] = r["metrics"]["crps"] - r["metrics"][
                "excess_crps"]
    return float(np.mean(list(by_seed.values()))) if by_seed else float("nan")


def aggregate(records, keys, models):
    """Per (key, model) means over seeds: excess CRPS, 90% coverage error in
    points, fit seconds -- plus the oracle floor per key."""
    per = {}
    for key in keys:
        per[key] = {"oracle": _oracle_floor(records, key)}
        for m in models:
            vals = [r for r in records
                    if r["dataset"] == key and r["model"] == m]
            if not vals:
                continue
            exc = [v["metrics"]["excess_crps"] for v in vals]
            cov = [abs(v["metrics"]["coverage_90"] - 0.90) * 100.0
                   for v in vals if "coverage_90" in v["metrics"]]
            per[key][m] = {
                "excess": float(np.mean(exc)),
                "coverr": float(np.mean(cov)) if cov else float("nan"),
                "fit_s": float(np.mean([v["fit_time"] for v in vals])),
                "n": len(vals),
            }
    return per


def format_block(per, keys, models, field, scale, title, show_oracle):
    """One per-key x per-arm block, then the median over keys per arm."""
    wkey = max([len(k) for k in keys] + [len("MEDIAN")])
    wmod = max([len(m) for m in models] + [10])
    head_cells = [f"{'key':{wkey}s}"]
    if show_oracle:
        head_cells.append(f"{'oracle':>10s}")
    head_cells += [f"{m:>{wmod}s}" for m in models]
    head = " ".join(head_cells)
    lines = [title, head, "-" * len(head)]
    for key in keys:
        cells = [f"{key:{wkey}s}"]
        if show_oracle:
            cells.append(f"{per[key]['oracle']:10.4f}")
        for m in models:
            v = per[key].get(m, {}).get(field, float("nan"))
            cells.append(f"{v * scale:>{wmod}.2f}")
        lines.append(" ".join(cells))
    med_cells = [f"{'MEDIAN':{wkey}s}"]
    if show_oracle:
        med_cells.append(
            f"{np.median([per[k]['oracle'] for k in keys]):10.4f}")
    for m in models:
        vals = [per[k][m][field] for k in keys if m in per[k]]
        med_cells.append(f"{float(np.median(vals)) * scale:>{wmod}.2f}"
                         if vals else f"{'n/a':>{wmod}s}")
    lines += ["-" * len(head), " ".join(med_cells)]
    return "\n".join(lines)


def format_fit(records, models):
    """Mean fit seconds per arm, over every (key, seed) run."""
    lines = ["mean fit seconds per arm (over every run):"]
    for m in models:
        secs = [r["fit_time"] for r in records if r["model"] == m]
        if secs:
            lines.append(f"  {m:24s} {float(np.mean(secs)):8.2f}s  "
                         f"(n={len(secs)})")
    return "\n".join(lines)


def _print_dataset_list(keys, args, taus):
    """The --list-datasets page: every key, then the run shape."""
    for key in keys:
        print(f"  {key}")
    print(f"\ntotal: {len(keys)} datasets x {args.seeds} "
          f"seed(s) x {len(args.models)} models, K={len(taus)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help="seed count; seeds are 0..seeds-1 "
                         f"(default: {DEFAULT_SEEDS}, the full screen).")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                    help="dataset sizes (default: 1000 10000).")
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES),
                    choices=list(REGIMES))
    ap.add_argument("--threads", type=int, default=None,
                    help="total thread budget across all parallel jobs "
                         "(None = all cores).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="(regime, size, seed) tasks to run in parallel "
                         "processes; each gets threads/jobs threads "
                         "(default: 1, serial).")
    ap.add_argument("--models", nargs="+", default=list(qs.ARMS),
                    choices=list({**qs.ARMS, **qs.PROBES}))
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    taus = TAUS
    keys = [dataset_key(r, n) for r in args.regimes for n in args.sizes]

    if args.list_datasets:
        _print_dataset_list(keys, args, taus)
        return 0

    jobs = max(1, args.jobs)
    total_threads = args.threads or os.cpu_count() or 1
    threads_per = max(1, total_threads // jobs)

    print(f"{len(keys)} datasets, {args.seeds} seed(s), K={len(taus)} levels, "
          f"jobs={jobs} threads/job={threads_per}, "
          f"models: {', '.join(args.models)}", flush=True)

    tasks = [(r, n, s, taus, threads_per, args.models)
             for r in args.regimes for n in args.sizes
             for s in range(args.seeds)]
    collected = (_run_serial(tasks) if jobs == 1
                 else _run_parallel(tasks, jobs))
    records, ds_meta = _assemble_records(keys, args.seeds, args.models,
                                         collected)

    per = aggregate(records, keys, args.models)
    print()
    print(format_block(per, keys, args.models, "excess", 1000.0,
                       "excess CRPS x1000 over the oracle "
                       "(mean over seeds; lower better)", True))
    print()
    print(format_block(per, keys, args.models, "coverr", 1.0,
                       "90% coverage error in points "
                       "(mean |coverage - 0.90| over seeds; lower better)",
                       False))
    print()
    print(format_fit(records, args.models))

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RESULTS_DIR, f"quantile-synth-{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "config": {"seeds": args.seeds, "sizes": args.sizes,
                           "regimes": args.regimes, "models": args.models,
                           "quantiles": [float(t) for t in taus],
                           "timing": "fit_only", "suite": "quantile-synth",
                           "jobs": jobs, "threads_per_model": threads_per},
                "provenance": rb._provenance(sys.argv, {}),
                "datasets": ds_meta,
                "records": records,
            }, fh, indent=1)
        print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
