"""Interval-width probe for the multi-quantile head (LEAFTUNE P10-P13 re-run).

P12 found that the head's training-time narrowing budget saturates: the band
freezes within tens of rounds and the delivered intervals end 2x to 10x wider
than a calibrated band, so coverage runs far above nominal and pinball loses to
the dumbest possible quantile model. That harness lived on a worktree that no
longer exists; this rebuilds the parts that judge a fix.

Three arms on the three P12 datasets:

  head        ChimeraBoostQuantileRegressor, out of the box
  head+CQR    the same with conformalize=True (the feature's own remedy)
  offset      the rigid location shift: ONE squared-error fit plus the marginal
              quantiles of held-out residuals. Identical interval width for
              every row -- no conditional spread at all. This is the baseline
              the shipped head lost to on 3 of 3, and beating it is the bar.

Two questions, two sections:

  1. Quality  -- mean pinball over the grid, coverage and width of the central
                 interval, crossing rate. Lower pinball wins; coverage is read
                 next to width, since a wide enough band covers everything.
  2. Freeze   -- interval width by boosting round out to 3000. A width that is
                 identical at round 50 and round 3000 is the saturation
                 signature; a healthy fit keeps adapting.

Run alone -- one benchmark at a time.
    python benchmarks/probe_quantile_band.py --tag base
    python benchmarks/probe_quantile_band.py --tag fix
Writes: benchmarks/results/quantile-band-<tag>.md
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import chimeraboost                                            # noqa: E402
from chimeraboost import (ChimeraBoostQuantileRegressor,       # noqa: E402
                          ChimeraBoostRegressor)
from chimeraboost import quantile_metrics as qm                # noqa: E402
from chimeraboost.warmup import warmup                         # noqa: E402
import run_benchmarks as rb                                    # noqa: E402

rb._add_grinsztajn_datasets()

DATASETS = [
    "gr:reg_cat/Brazilian_houses",
    "gr:reg_num/cpu_act",
    "gr:reg_num/elevators",
]
SEEDS = (0, 1, 2)
# K=3 puts the 0.80 central interval on the grid, which is the interval every
# P10-P13 coverage number is quoted against. K=19 is the shipped default grid.
GRIDS = {"K3": np.array([0.1, 0.5, 0.9]),
         "K19": np.round(np.arange(0.05, 0.9501, 0.05), 10)}
FREEZE_ROUNDS = (50, 100, 300, 1000, 2000, 3000)


def _central(taus):
    """Index pair of the widest symmetric interval on the grid, and its
    nominal level."""
    lo, hi = 0, taus.size - 1
    return lo, hi, float(taus[hi] - taus[lo])


def _metrics(y, Q, taus):
    lo, hi, nominal = _central(taus)
    inside = (y >= Q[:, lo]) & (y <= Q[:, hi])
    return {
        "pinball": qm.crps(y, Q, taus),
        "coverage": float(inside.mean()),
        "width": float(np.mean(Q[:, hi] - Q[:, lo])),
        "crossing": qm.crossing_rate(Q),
        "nominal": nominal,
    }


def _rigid_offset(Xtr, ytr, Xte, cat, taus, seed):
    """One squared-error fit; the band is the marginal quantiles of residuals
    measured on a held-out fold, added to every row's point prediction.

    Held-out rather than in-sample: in-sample residuals are shrunk by the fit
    and understate the spread, which is the fix P10 gave this arm before it
    started winning."""
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=seed)
    m = ChimeraBoostRegressor(random_state=seed).fit(Xf, yf, cat_features=cat)
    resid = np.asarray(yc, dtype=np.float64) - m.predict(Xc)
    return m.predict(Xte)[:, None] + np.quantile(resid, taus)[None, :]


def run_quality(key, seed, taus):
    """One (dataset, seed): all three arms on an identical split."""
    X, y, cat, _ = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    yte = np.asarray(yte, dtype=np.float64)
    out = {}

    for name, kw in (("head", {}), ("head+CQR", {"conformalize": True})):
        t0 = time.perf_counter()
        m = ChimeraBoostQuantileRegressor(quantiles=taus, random_state=seed)
        m.set_params(**kw).fit(Xtr, ytr, cat_features=cat)
        secs = time.perf_counter() - t0
        out[name] = {**_metrics(yte, m.predict(Xte), taus), "secs": secs,
                     "rounds": len(m.model_.trees_)}

    t0 = time.perf_counter()
    Q = _rigid_offset(Xtr, ytr, Xte, cat, taus, seed)
    out["offset"] = {**_metrics(yte, Q, taus),
                     "secs": time.perf_counter() - t0, "rounds": np.nan}
    return out


def run_freeze(key, taus, seed=0):
    """Central-interval width by round, from ONE fit read through
    staged_predict. Early stopping off so the round axis is the real one."""
    X, y, cat, _ = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed)
    yte = np.asarray(yte, dtype=np.float64)
    m = ChimeraBoostQuantileRegressor(
        quantiles=taus, n_estimators=max(FREEZE_ROUNDS), random_state=seed,
        early_stopping=False).fit(Xtr, ytr, cat_features=cat)
    lo, hi, _ = _central(taus)
    wanted = set(FREEZE_ROUNDS)
    widths, covs = {}, {}
    for i, Q in enumerate(m.staged_predict(Xte), start=1):
        if i in wanted:
            widths[i] = float(np.mean(Q[:, hi] - Q[:, lo]))
            covs[i] = float(np.mean((yte >= Q[:, lo]) & (yte <= Q[:, hi])))
    return widths, covs


def _fmt_quality(rows, taus, grid_name, lines):
    lo, hi, nominal = _central(taus)
    hdr = (f"{'dataset':>26s}{'arm':>10s}{'pinball':>11s}{'coverage':>10s}"
           f"{'width':>12s}{'crossing':>10s}{'secs':>8s}")
    print(f"\n{'=' * len(hdr)}")
    print(f"QUALITY, grid {grid_name} ({taus.size} levels), central interval "
          f"{taus[lo]:.2f}-{taus[hi]:.2f}, nominal {nominal:.2f}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    lines += [f"## Quality, grid {grid_name}", "",
              f"Central interval {taus[lo]:.2f}-{taus[hi]:.2f}, nominal "
              f"coverage {nominal:.2f}. `pinball` is the mean over the grid "
              f"(lower is better), averaged over {len(SEEDS)} seeds.", "",
              "| dataset | arm | pinball | coverage | width | crossing | "
              "secs |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for key in DATASETS:
        for arm in ("head", "head+CQR", "offset"):
            r = rows[key][arm]
            short = key.split("/")[-1]
            print(f"{short:>26s}{arm:>10s}{r['pinball']:11.5f}"
                  f"{r['coverage']:10.4f}{r['width']:12.5f}"
                  f"{r['crossing']:10.4f}{r['secs']:8.2f}")
            lines.append(f"| {short} | {arm} | {r['pinball']:.5f} | "
                         f"{r['coverage']:.4f} | {r['width']:.5f} | "
                         f"{r['crossing']:.4f} | {r['secs']:.2f} |")
    lines.append("")

    # Head vs the trivial baseline: the bar the shipped version failed 0 of 3.
    print(f"\n{'head vs offset (pinball, + means the head is better)':>60s}")
    lines += ["### Head against the rigid offset", "",
              "Relative pinball, positive means the head wins. The shipped "
              "head lost on 3 of 3.", "",
              "| dataset | head vs offset | head+CQR vs offset |",
              "|:--|--:|--:|"]
    wins = 0
    for key in DATASETS:
        o = rows[key]["offset"]["pinball"]
        a = (o - rows[key]["head"]["pinball"]) / o * 100
        b = (o - rows[key]["head+CQR"]["pinball"]) / o * 100
        wins += int(max(a, b) > 0)
        print(f"{key.split('/')[-1]:>26s}{a:+11.2f}%{b:+11.2f}%")
        lines.append(f"| {key.split('/')[-1]} | {a:+.2f}% | {b:+.2f}% |")
    print(f"\nhead beats the offset on {wins} of {len(DATASETS)} datasets")
    lines += ["", f"Head (best of plain / CQR) beats the offset on **{wins} of "
                  f"{len(DATASETS)}** datasets.", ""]
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="label for the output file")
    ap.add_argument("--skip-freeze", action="store_true")
    ap.add_argument("--grids", default="K3,K19")
    args = ap.parse_args()

    print(f"chimeraboost: {chimeraboost.__file__}")
    print("warming JIT...", flush=True)
    warmup()

    lines = [f"# Quantile interval band probe ({args.tag})", "",
             f"Datasets {', '.join(d.split('/')[-1] for d in DATASETS)}; "
             f"seeds {list(SEEDS)}; arms head / head+CQR / rigid offset.", ""]
    verdict = {}

    for grid_name in args.grids.split(","):
        taus = GRIDS[grid_name]
        rows = {}
        for key in DATASETS:
            per_seed = [run_quality(key, s, taus) for s in SEEDS]
            rows[key] = {
                arm: {k: float(np.mean([p[arm][k] for p in per_seed]))
                      for k in per_seed[0][arm]}
                for arm in per_seed[0]}
            print(f"  done {key} [{grid_name}]", flush=True)
        verdict[grid_name] = _fmt_quality(rows, taus, grid_name, lines)

    if not args.skip_freeze:
        taus = GRIDS["K3"]
        hdr = f"{'dataset':>26s}" + "".join(f"{r:>10d}" for r in FREEZE_ROUNDS)
        print(f"\n{'=' * len(hdr)}")
        print("FREEZE CHECK: central-interval width by round (K3, no early "
              "stopping)")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        lines += ["## Freeze check", "",
                  "Central-interval width by boosting round, early stopping "
                  "off. Identical widths across the row are the saturation "
                  "signature; `ratio` is the last round over round 50.", "",
                  "| dataset | " + " | ".join(str(r) for r in FREEZE_ROUNDS)
                  + " | ratio |",
                  "|:--|" + "--:|" * (len(FREEZE_ROUNDS) + 1)]
        for key in DATASETS:
            w, c = run_freeze(key, taus)
            ratio = w[FREEZE_ROUNDS[-1]] / w[FREEZE_ROUNDS[0]]
            print(f"{key.split('/')[-1]:>26s}"
                  + "".join(f"{w[r]:10.5f}" for r in FREEZE_ROUNDS)
                  + f"   ratio {ratio:.4f}")
            print(f"{'  coverage':>26s}"
                  + "".join(f"{c[r]:10.4f}" for r in FREEZE_ROUNDS))
            lines.append(f"| {key.split('/')[-1]} | "
                         + " | ".join(f"{w[r]:.5f}" for r in FREEZE_ROUNDS)
                         + f" | {ratio:.4f} |")
            lines.append(f"| {key.split('/')[-1]} coverage | "
                         + " | ".join(f"{c[r]:.4f}" for r in FREEZE_ROUNDS)
                         + " | |")
        lines.append("")

    for grid_name, wins in verdict.items():
        print(f"\nVERDICT {grid_name}: head beats the rigid offset on "
              f"{wins} of {len(DATASETS)}")

    out = os.path.join(_HERE, "results", f"quantile-band-{args.tag}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
