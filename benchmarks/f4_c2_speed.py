"""F4 candidate C2 speed read: the `factorize` fast path for string columns.

The claim under test is that `_factorize_hashed` converts the 18% cProfile row
on hc:okcupid-stem into real fit time, on real data, without changing a single
prediction (that part is `identity_snapshot.py`'s job, not this script's).

Method -- a SAME-PROCESS A/B, which is why this is not a benchmark run and the
cross-run ~2% fit-time noise floor does not apply. One process loads the data
once, then fits the same estimator repeatedly, alternating:

    OFF: `target_encoding._factorize_hashed` monkeypatched to return None,
         so every string column falls into the dict loop -- the base behaviour,
         byte for byte, because that patch is the only difference.
    ON:  the fast path live.

Alternating ON/OFF within one warmed process removes JIT warmup, cache state
and machine drift from the comparison; the reported number is the median of
the per-repeat ratios, and the spread is printed so a too-small effect is
visible as such rather than rounded up into a claim.

It also censuses REACH, which the profile of one dataset could not: every
`factorize` call the fit actually makes is tagged with the path it took
(numeric fast / string fast / dict loop) and its rows, so "wide reach" stops
being an assumption.

Run:
    python benchmarks/f4_c2_speed.py                     # default panel
    python benchmarks/f4_c2_speed.py --datasets hc:kick --repeats 5
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402  (the repo's one dataset loader)

import chimeraboost.preprocessing as pp  # noqa: E402
import chimeraboost.target_encoding as te  # noqa: E402
from chimeraboost import (ChimeraBoostClassifier,  # noqa: E402
                          ChimeraBoostRegressor)

# One string-heavy multiclass set (the profiled one), one large high-card
# binary set, and one Grinsztajn categorical set -- the last is the control for
# the "Grinsztajn should not move" half of the I010 forecast.
PANEL = ["hc:okcupid-stem", "hc:kick", "gr:clf_cat/road-safety"]

_ORIG_HASHED = te._factorize_hashed
_ORIG_NUMERIC = te._factorize_numeric
_ORIG_FACTORIZE = pp.factorize
_CENSUS = {"calls": 0, "seconds": 0.0, "numeric": 0, "hashed": 0}


def _watch_numeric(col):
    out = _ORIG_NUMERIC(col)
    _CENSUS["numeric"] += out is not None
    return out


def _watch_hashed(col):
    out = _ORIG_HASHED(col)
    _CENSUS["hashed"] += out is not None
    return out


def _off_hashed(col):
    return None                       # base behaviour: everything hits the loop


def _timed_factorize(column):
    """Time every `factorize` call the fit makes.

    `preprocessing` imported the name directly, so patching it here is what the
    library actually calls. The path census rides on wrappers around the two
    fast paths themselves -- an earlier version of this script re-ran the fast
    path to ask which one fired, which charged the candidate arm for the work
    twice and made a 43%-faster path read as 25% SLOWER."""
    t0 = time.perf_counter()
    out = _ORIG_FACTORIZE(column)
    _CENSUS["seconds"] += time.perf_counter() - t0
    _CENSUS["calls"] += 1
    return out


def _fit_once(Est, Xtr, ytr, cat_idx, n_estimators):
    est = Est(n_estimators=n_estimators, random_state=0, early_stopping=True,
              early_stopping_rounds=50, validation_fraction=0.15)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr, cat_features=cat_idx)
    return time.perf_counter() - t0


def run_dataset(key, repeats, n_estimators):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if task != "regression" else None)
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, "
          f"n_features={Xtr.shape[1]}, cat_features="
          f"{len(cat_idx) if cat_idx else 0}) ===", flush=True)

    # Warm the JIT AND the download/bin caches before anything is timed.
    Est(n_estimators=5, random_state=0).fit(
        Xtr[:min(500, len(Xtr))], ytr[:min(500, len(Xtr))],
        cat_features=cat_idx)

    pp.factorize = _timed_factorize
    te._factorize_numeric = _watch_numeric
    off, on, fac = [], [], {}
    for i in range(repeats):
        for arm, patch, times in (("OFF", _off_hashed, off),
                                  ("ON", _watch_hashed, on)):
            te._factorize_hashed = patch
            for k in _CENSUS:
                _CENSUS[k] = 0 if k != "seconds" else 0.0
            times.append(_fit_once(Est, Xtr, ytr, cat_idx, n_estimators))
            fac[arm] = dict(_CENSUS)
        print(f"  repeat {i + 1}: OFF {off[-1]:6.2f}s   ON {on[-1]:6.2f}s   "
              f"ratio {on[-1] / off[-1]:.3f}", flush=True)
    pp.factorize = _ORIG_FACTORIZE
    te._factorize_numeric = _ORIG_NUMERIC
    te._factorize_hashed = _ORIG_HASHED

    ratios = sorted(o / f for o, f in zip(on, off))
    med = statistics.median(ratios)
    print(f"  MEDIAN ON/OFF = {med:.3f}  ({(med - 1) * 100:+.1f}% fit time), "
          f"per-repeat range {ratios[0]:.3f}..{ratios[-1]:.3f}")
    print(f"  OFF median {statistics.median(off):6.2f}s   "
          f"ON median {statistics.median(on):6.2f}s")
    for arm in ("OFF", "ON"):
        c = fac[arm]
        print(f"  {arm}: {c['calls']} factorize calls, {c['seconds']:.2f}s in "
              f"factorize ({c['numeric']} numeric-fast, {c['hashed']} hashed, "
              f"{c['calls'] - c['numeric'] - c['hashed']} loop)")
    return {"dataset": key, "task": task, "median_ratio": med,
            "off_median": statistics.median(off),
            "on_median": statistics.median(on),
            "factorize_off_s": fac["OFF"]["seconds"],
            "factorize_on_s": fac["ON"]["seconds"],
            "calls": fac["ON"]["calls"], "hashed": fac["ON"]["hashed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=2000)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    rows = [run_dataset(k, args.repeats, args.n_estimators)
            for k in args.datasets]
    print("\n| dataset | task | OFF s | ON s | fit-time change "
          "| factorize s OFF -> ON | calls (hashed) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['task']} | {r['off_median']:.2f} | "
              f"{r['on_median']:.2f} | {(r['median_ratio'] - 1) * 100:+.1f}% | "
              f"{r['factorize_off_s']:.2f} -> {r['factorize_on_s']:.2f} | "
              f"{r['calls']} ({r['hashed']}) |")


if __name__ == "__main__":
    main()
