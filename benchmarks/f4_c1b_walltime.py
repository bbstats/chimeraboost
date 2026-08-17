"""F4 candidate C1b ceiling read: what is left in `grad_hess` after C1?

C1 fused the softmax and took the multiclass fit down ~40%. C1b would go one
step further -- have the same kernel emit `P - Y` and `max(P*(1-P), 1e-6)` in
the pass it already makes, instead of running two more full (n, K) numpy passes
with an allocation each.

The point of this script is that C1b may no longer be worth anything, and that
has to be MEASURED rather than assumed in either direction. C1's own ceiling was
justified by a 40%-of-fit `grad_hess` row, but C1 removed most of what was
inside it, so the number C1b would be argued from no longer exists. The honest
prior is that the remainder is small.

The read separates the two halves of `grad_hess`: the softmax call (now fused
and fast) and the arithmetic after it, timed as the difference. Both are
reported as a share of a real fit, and the ceiling for C1b is the second half
alone.

Run: python benchmarks/f4_c1b_walltime.py [--datasets hc:okcupid-stem hc:cjs]
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

import chimeraboost.losses as losses  # noqa: E402
from chimeraboost import ChimeraBoostClassifier  # noqa: E402

PANEL = ["hc:okcupid-stem", "hc:cjs"]

_ORIG_SOFTMAX = losses._softmax
_ORIG_GH = losses.MultiSoftmax.grad_hess
_C = {"gh_s": 0.0, "gh_n": 0, "sm_s": 0.0, "sm_n": 0}


def _timed_softmax(F):
    t0 = time.perf_counter()
    out = _ORIG_SOFTMAX(F)
    _C["sm_s"] += time.perf_counter() - t0
    _C["sm_n"] += 1
    return out


def _timed_gh(self, Y, F):
    t0 = time.perf_counter()
    out = _ORIG_GH(self, Y, F)
    _C["gh_s"] += time.perf_counter() - t0
    _C["gh_n"] += 1
    return out


def run_dataset(key, n_estimators):
    X, y, cat_idx, task = rb.DATASETS[key](1, np.random.default_rng(0))
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=0,
                                      stratify=y)
    K = len(np.unique(ytr))
    print(f"\n=== {key}  ({task}, K={K}, n_train={len(Xtr)}) ===", flush=True)
    ChimeraBoostClassifier(n_estimators=5, random_state=0).fit(
        Xtr[:min(500, len(Xtr))], ytr[:min(500, len(Xtr))],
        cat_features=cat_idx)

    for k in _C:
        _C[k] = 0.0 if k.endswith("_s") else 0
    losses._softmax = _timed_softmax
    losses.MultiSoftmax.grad_hess = _timed_gh
    t0 = time.perf_counter()
    ChimeraBoostClassifier(n_estimators=n_estimators, random_state=0,
                           early_stopping=True, early_stopping_rounds=50,
                           validation_fraction=0.15).fit(
        Xtr, ytr, cat_features=cat_idx)
    fit_s = time.perf_counter() - t0
    losses._softmax = _ORIG_SOFTMAX
    losses.MultiSoftmax.grad_hess = _ORIG_GH

    # `grad_hess` calls `_softmax` once, so the arithmetic C1b would absorb is
    # what is left after subtracting the softmax time charged inside it. The
    # `eval` path also calls `_softmax`, hence the per-call apportioning rather
    # than a flat subtraction.
    sm_in_gh = _C["sm_s"] * (_C["gh_n"] / _C["sm_n"]) if _C["sm_n"] else 0.0
    rest = _C["gh_s"] - sm_in_gh
    print(f"  fit {fit_s:.2f}s")
    print(f"  grad_hess total {_C['gh_s']:.3f}s in {_C['gh_n']} calls "
          f"= {_C['gh_s'] / fit_s * 100:.1f}% of fit")
    print(f"    of which softmax (C1, fused) ~{sm_in_gh:.3f}s "
          f"= {sm_in_gh / fit_s * 100:.1f}% of fit")
    print(f"    C1b's object (grad + hess arithmetic) ~{rest:.3f}s "
          f"= {rest / fit_s * 100:.1f}% of fit   <-- the ceiling")
    return {"dataset": key, "fit_s": fit_s, "gh_s": _C["gh_s"],
            "rest": rest, "share": rest / fit_s * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--n-estimators", type=int, default=2000)
    args = ap.parse_args()
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    rows = [run_dataset(k, args.n_estimators) for k in args.datasets]

    print("\n| dataset | fit s | grad_hess s | C1b object s | ceiling |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['fit_s']:.2f} | {r['gh_s']:.3f} | "
              f"{r['rest']:.3f} | {r['share']:.1f}% |")
    worst = max(r["share"] for r in rows)
    print(f"\nCEILING: at best C1b can remove {worst:.1f}% of fit, and only if "
          f"the fused kernel makes those passes free. Read against the ~1% "
          f"same-process noise floor.")


if __name__ == "__main__":
    main()
