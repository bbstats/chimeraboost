"""Where does the replay refit's time go?

Splits a `refit_full="replay"` fit into the legs that make it up, so the
remaining speed headroom is visible: the donor prep transform of the full
matrix, the replayed rounds, the extra rounds that are still GROWN from
scratch (rounds are scaled by 1/(1-validation_fraction), so ~18% of the
refit's trees have no donor), and whatever is left over.

    python benchmarks/replay_attr.py [key-substring ...]
"""
import sys
import time
import collections

import numpy as np
from sklearn.model_selection import train_test_split

import chimeraboost
import chimeraboost.booster as bmod
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.warmup import warmup
import run_benchmarks as rb

T = collections.defaultdict(float)
N = collections.Counter()


def _wrap(mod, name, key):
    orig = getattr(mod, name)

    def w(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        T[key] += time.perf_counter() - t0
        N[key] += 1
        return r
    setattr(mod, name, w)
    return orig


_wrap(bmod, "replay_oblivious_tree", "replay_round")
_wrap(bmod, "build_oblivious_tree", "grow_round")

_orig_tf = bmod.FeaturePreprocessor.transform


def _tf(self, *a, **kw):
    t0 = time.perf_counter()
    r = _orig_tf(self, *a, **kw)
    T["prep_transform"] += time.perf_counter() - t0
    N["prep_transform"] += 1
    return r


bmod.FeaturePreprocessor.transform = _tf

PANEL = ["gr:reg_num/pol", "gr:clf_num/covertype", "gr:clf_cat/road-safety"]


def main():
    print("chimeraboost:", chimeraboost.__file__)
    rb._add_grinsztajn_datasets()
    warmup()
    panel = [k for k in PANEL
             if not sys.argv[1:] or any(a in k for a in sys.argv[1:])]
    for key in panel:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=0)
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        for mode in (True, "replay"):
            T.clear()
            N.clear()
            est = Est(random_state=0, refit_full=mode)
            t0 = time.perf_counter()
            est.fit(Xtr, ytr, cat_features=cat)
            total = time.perf_counter() - t0
            # Everything before the refit is shared by both arms; the refit's
            # own legs are what differ.
            print(f"\n{key}  refit_full={mode!r}   total {total:.2f}s")
            for k in ("grow_round", "replay_round", "prep_transform"):
                if N[k]:
                    print(f"   {k:16s} {T[k]:7.2f}s over {N[k]:5d} calls "
                          f"({T[k] / N[k] * 1000:6.3f} ms each, "
                          f"{T[k] / total * 100:4.1f}% of fit)")
            acct = sum(T[k] for k in ("grow_round", "replay_round",
                                      "prep_transform"))
            print(f"   {'unaccounted':16s} {total - acct:7.2f}s "
                  f"({(total - acct) / total * 100:4.1f}%)")


if __name__ == "__main__":
    main()
