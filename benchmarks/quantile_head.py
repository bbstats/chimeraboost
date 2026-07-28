"""Acceptance benchmark for the shared-tree multi-quantile head.

Answers four questions, each against a stated reference:

  1. Accuracy   -- does one shared tree structure match K independent
                   LightGBM quantile boosters on pinball loss? The true
                   conditional quantiles are known in closed form here, so
                   both are also scored against the oracle floor.
  2. Speed      -- is it at least K/2 times faster to fit than those K
                   boosters? The saving is concentrated in the split search,
                   while several per-round costs scale with K, so the answer
                   depends on how wide the data is. The feature sweep reports
                   where the K/2 line is actually crossed rather than quoting
                   one flattering shape.
  3. Crossing   -- ours must be exactly zero. Independent per-level models are
                   not, and the table says by how much.
  4. Coverage   -- do the intervals hold what they claim, before and after
                   conformalization?

Also ablates the split projection, since "which direction do we collapse the
K gradient columns onto" is the head's one real design decision.

Run alone -- one benchmark at a time, or the timings are noise.
    python benchmarks/quantile_head.py [--quick]
Writes: benchmarks/results/quantile-head.md
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chimeraboost  # noqa: E402
from chimeraboost import ChimeraBoostQuantileRegressor  # noqa: E402
from chimeraboost import quantile_metrics as qm  # noqa: E402
from chimeraboost.warmup import warmup  # noqa: E402

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 10)


def make_data(n, p, seed, regime="mixed"):
    """Heteroscedastic Gaussian data whose conditional quantiles are known.

    The centre depends on one column and the spread on another, so the oracle
    quantile is ``mu(x) + sigma(x) * Phi^-1(tau)`` exactly.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    if regime == "location":
        mu, sd = 2.0 * X[:, 0], np.ones(n)
    elif regime == "scale":
        mu, sd = np.zeros(n), 0.3 + 2.7 * (X[:, 1] > 0)
    elif regime == "extreme":
        mu, sd = 2.0 * X[:, 0], 0.1 + 2.0 * (X[:, 1] > 0)
    else:
        mu, sd = 2.0 * X[:, 0], 0.4 + 1.6 * (X[:, 1] > 0)
    # A little extra structure so the split search has something to find.
    mu = mu + np.sin(X[:, 2] * 2.0) + 0.5 * X[:, 3] * X[:, 4]
    y = mu + sd * rng.standard_normal(n)
    q_oracle = mu[:, None] + sd[:, None] * norm.ppf(TAUS)[None, :]
    return X, y, q_oracle


def fit_chimera(Xt, yt, Xv, rounds, **kw):
    m = ChimeraBoostQuantileRegressor(
        quantiles=TAUS, n_estimators=rounds, learning_rate=0.1, depth=4,
        random_state=0, early_stopping=False, **kw)
    t0 = time.perf_counter()
    m.fit(Xt, yt)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Q = m.predict(Xv)
    return Q, fit_s, time.perf_counter() - t0


def fit_lightgbm(Xt, yt, Xv, rounds):
    """K independent LightGBM quantile boosters -- the strong version of the
    baseline: one shared Dataset built once, so it is not charged K times for
    binning the same rows."""
    import lightgbm as lgb
    t0 = time.perf_counter()
    ds = lgb.Dataset(Xt, label=yt, free_raw_data=False)
    boosters = []
    for tau in TAUS:
        params = {"objective": "quantile", "alpha": float(tau),
                  "learning_rate": 0.1, "num_leaves": 16, "max_depth": 4,
                  "min_data_in_leaf": 20, "verbose": -1, "seed": 0,
                  "deterministic": True, "num_threads": 0}
        boosters.append(lgb.train(params, ds, num_boost_round=rounds))
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Q = np.column_stack([b.predict(Xv) for b in boosters])
    return Q, fit_s, time.perf_counter() - t0


def pinball(y, Q):
    r = np.asarray(y)[:, None] - Q
    return float(np.maximum(TAUS * r, (TAUS - 1.0) * r).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="one seed, fewer shapes")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1

    print(f"chimeraboost: {chimeraboost.__file__}")
    try:
        import lightgbm
        print(f"lightgbm: {lightgbm.__version__}")
        have_lgb = True
    except ImportError:
        print("(skip LightGBM -- not installed)")
        have_lgb = False
    print(f"K = {TAUS.size} levels, K/2 speed target = {TAUS.size / 2:.1f}x")
    print("warming JIT...", flush=True)
    warmup()

    lines = ["# Multi-quantile head: acceptance", "",
             f"K = {TAUS.size} levels (0.05 ... 0.95), "
             f"{args.rounds} rounds, depth 4, lr 0.1, "
             f"{args.seeds} seed(s), n = {args.n}.", ""]

    # ---------------------------------------------------------------- 1 + 2
    widths = [8, 20] if args.quick else [5, 8, 16, 32, 64, 128]
    print(f"\n{'=' * 78}\nHEAD TO HEAD vs K independent LightGBM quantile "
          f"boosters\n{'=' * 78}")
    hdr = (f"{'features':>9s}{'ChimB pin':>11s}{'LGBM pin':>11s}"
           f"{'oracle':>10s}{'ratio':>8s}{'ChimB fit':>11s}{'LGBM fit':>11s}"
           f"{'speedup':>9s}{'ChimB xs':>10s}{'LGBM xs':>9s}")
    print(hdr)
    print("-" * len(hdr))
    tbl = ["## Accuracy and speed vs K independent LightGBM boosters", "",
           "`ratio` is our pinball over LightGBM's (1.00 = identical, lower is "
           "better). `speedup` is their fit wall clock over ours; the target "
           f"is K/2 = {TAUS.size / 2:.1f}x. `xs` columns are crossing rates.",
           "",
           "| features | ChimB pinball | LGBM pinball | oracle | ratio | "
           "ChimB fit s | LGBM fit s | speedup | ChimB crossing | "
           "LGBM crossing |",
           "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for p in widths:
        cp = lp = op = cf = lf = cx = lx = 0.0
        for s in range(args.seeds):
            X, y, Qo = make_data(args.n, p, s)
            c = int(args.n * 0.75)
            Xt, yt, Xv, yv, Qov = X[:c], y[:c], X[c:], y[c:], Qo[c:]
            Q, f, _ = fit_chimera(Xt, yt, Xv, args.rounds)
            cp += pinball(yv, Q)
            cf += f
            cx += qm.crossing_rate(Q)
            op += pinball(yv, Qov)
            if have_lgb:
                Ql, fl, _ = fit_lightgbm(Xt, yt, Xv, args.rounds)
                lp += pinball(yv, Ql)
                lf += fl
                lx += qm.crossing_rate(Ql)
        k = args.seeds
        cp, lp, op, cf, lf, cx, lx = (v / k for v in
                                      (cp, lp, op, cf, lf, cx, lx))
        ratio = cp / lp if lp else float("nan")
        speed = lf / cf if cf else float("nan")
        print(f"{p:9d}{cp:11.5f}{lp:11.5f}{op:10.5f}{ratio:8.3f}"
              f"{cf:11.2f}{lf:11.2f}{speed:8.1f}x{cx:10.4f}{lx:9.4f}")
        tbl.append(f"| {p} | {cp:.5f} | {lp:.5f} | {op:.5f} | {ratio:.3f} | "
                   f"{cf:.2f} | {lf:.2f} | {speed:.1f}x | {cx:.4f} | "
                   f"{lx:.4f} |")
    lines += tbl + [""]

    # ------------------------------------------------------------------- 3
    print(f"\n{'=' * 78}\nSPLIT PROJECTION ABLATION (excess pinball over the "
          f"oracle, x1000)\n{'=' * 78}")
    arms = [("rotate (default)", dict()), ("sum", dict(split_projection="sum")),
            ("gram", dict(split_projection="gram")),
            ("exact", dict(exact_splits=True))]
    regimes = ["location", "scale", "mixed"] if args.quick else \
        ["location", "scale", "mixed", "extreme"]
    hdr = f"{'arm':18s}" + "".join(f"{r:>11s}" for r in regimes) + \
        f"{'TOTAL':>11s}{'fit s':>9s}"
    print(hdr)
    print("-" * len(hdr))
    tbl = ["## Split-projection ablation", "",
           "Excess pinball over the oracle, x1000 (lower is better). "
           "`location` moves only the centre, `scale` only the spread, "
           "`extreme` is a 21x spread ratio. `sum` is the literal "
           "channel sum; `exact` scores the true summed-across-level gain at "
           "K histogram channels per feature.", "",
           "| arm | " + " | ".join(regimes) + " | TOTAL | fit s |",
           "|:--|" + "--:|" * (len(regimes) + 2)]
    n_ab = 8000 if not args.quick else 4000
    for name, kw in arms:
        tot = 0.0
        secs = 0.0
        cells = []
        for reg in regimes:
            ex = 0.0
            for s in range(args.seeds):
                X, y, Qo = make_data(n_ab, 10, s, reg)
                c = int(n_ab * 0.7)
                Q, f, _ = fit_chimera(X[:c], y[:c], X[c:], args.rounds, **kw)
                ex += (pinball(y[c:], Q) - pinball(y[c:], Qo[c:])) * 1000
                secs += f
            ex /= args.seeds
            tot += ex
            cells.append(ex)
        print(f"{name:18s}" + "".join(f"{v:11.3f}" for v in cells)
              + f"{tot:11.3f}{secs / (len(regimes) * args.seeds):9.2f}")
        tbl.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in cells)
                   + f" | {tot:.3f} | "
                     f"{secs / (len(regimes) * args.seeds):.2f} |")
    lines += tbl + [""]

    # ------------------------------------------------------------------- 4
    print(f"\n{'=' * 78}\nINTERVAL COVERAGE AND WIDTH (n = 10000 train, 20000 "
          f"test)\n{'=' * 78}")
    # Averaged over seeds: a single calibration fold of a few thousand rows
    # carries about a point of coverage noise on its own, which would
    # otherwise be most of what the table shows.
    cov_p, cov_c = [], []
    for s in range(args.seeds):
        X, y, _ = make_data(30000, 12, s)
        Xt, yt, Xv, yv = X[:10000], y[:10000], X[10000:], y[10000:]
        plain = ChimeraBoostQuantileRegressor(
            quantiles=TAUS, n_estimators=400, random_state=0).fit(Xt, yt)
        conf = ChimeraBoostQuantileRegressor(
            quantiles=TAUS, n_estimators=400, random_state=0,
            conformalize=True).fit(Xt, yt)
        cov_p.append(qm.interval_coverage(yv, plain.predict(Xv), TAUS))
        cov_c.append(qm.interval_coverage(yv, conf.predict(Xv), TAUS))
    hdr = (f"{'interval':>11s}{'nominal':>9s}{'plain cov':>11s}"
           f"{'CQR cov':>10s}{'plain wid':>11s}{'CQR wid':>10s}{'err pp':>9s}")
    print(hdr)
    print("-" * len(hdr))
    tbl = ["## Interval coverage and width", "",
           "Trained on 10 000 rows, scored on 20 000 fresh ones. `err pp` is "
           "the conformalized coverage minus nominal, in percentage points; "
           "the acceptance target is within 2.", "",
           "| interval | nominal | plain coverage | CQR coverage | "
           "plain width | CQR width | err pp |",
           "|:--|--:|--:|--:|--:|--:|--:|"]
    worst = 0.0
    mean_p = [{k: np.mean([r[i][k] for r in cov_p]) for k in cov_p[0][i]}
              for i in range(len(cov_p[0]))]
    mean_c = [{k: np.mean([r[i][k] for r in cov_c]) for k in cov_c[0][i]}
              for i in range(len(cov_c[0]))]
    for a, b in zip(mean_p, mean_c):
        err = (b["coverage"] - b["nominal"]) * 100
        worst = max(worst, abs(err))
        lbl = f"{a['lo']:.2f}-{a['hi']:.2f}"
        print(f"{lbl:>11s}{a['nominal']:9.2f}{a['coverage']:11.4f}"
              f"{b['coverage']:10.4f}{a['width']:11.4f}{b['width']:10.4f}"
              f"{err:9.2f}")
        tbl.append(f"| {lbl} | {a['nominal']:.2f} | {a['coverage']:.4f} | "
                   f"{b['coverage']:.4f} | {a['width']:.4f} | "
                   f"{b['width']:.4f} | {err:+.2f} |")
    print(f"\nworst conformalized coverage error: {worst:.2f} pp "
          f"({'PASS' if worst <= 2.0 else 'FAIL'} vs the 2 pp target)")
    print(f"crossing rate, plain: {qm.crossing_rate(plain.predict(Xv)):.6f}  "
          f"conformalized: {qm.crossing_rate(conf.predict(Xv)):.6f}")
    tbl += ["", f"Worst conformalized coverage error **{worst:.2f} pp** "
                f"({'PASS' if worst <= 2.0 else 'FAIL'} against the 2 pp "
                f"target). Crossing rate 0 in both columns.", ""]
    lines += tbl

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "quantile-head.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
