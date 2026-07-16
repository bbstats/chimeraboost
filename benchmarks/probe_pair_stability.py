"""Pre-registered risk check for PARETO_PLAN step 2 (cheap selection).

The fallback design derives cross-feature candidate pairs from a ~100-round
truncated base fit's importances instead of a full fit's. Pairs are all
C(m,2) crosses of the top-m numeric features by split gain, so pair drift ==
top-m set drift. This probe measures it with ZERO library changes: the same
estimator fit (cross_features=False) run full vs truncated via a stop-at-k
fit callback, comparing the pair sets their importances generate.

Also re-confirms, through the real code path, that the truncated race picks
the same linear/const winner as the full fits (the offline race sim's claim).

Run:
    python benchmarks/probe_pair_stability.py [--k 100] [--seeds 3]
"""
import argparse
import collections

import numpy as np
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.sklearn_api import _cross_candidate_pairs
import run_benchmarks as rb

# The step-0 panel minus multiclass (no cross features) and minus
# wine-reviews (1 numeric column -> pairs structurally empty in both arms).
PANEL = [
    "gr:reg_num/cpu_act",
    "gr:reg_num/diamonds",
    "gr:reg_cat/nyc-taxi-green-dec-2016",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/Higgs",
    "gr:clf_cat/road-safety",
    "hc:kick",
]


def pair_set(model, cat, n_features):
    pairs = _cross_candidate_pairs(
        model.model_.feature_importances_, cat, n_features)
    return frozenset(pairs)


def _jac(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="*", default=[100, 200])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()

    def stopper(k):
        return lambda iteration, train_loss, val_loss, model: iteration + 1 >= k

    ks = args.ks
    print(f"Pair stability: full-fit pairs vs truncated-fit pairs at "
          f"k={ks} ({args.seeds} seeds). 'seed floor' = mean pairwise "
          f"Jaccard among the full fits across seeds (split noise).\n")
    head = " | ".join(f"jac@{k}" for k in ks)
    print(f"| dataset | seed | {head} | winner match "
          + "".join(f"@{k} " for k in ks) + "|")
    print("|---|--:|" + "--:|" * len(ks) + "---|")
    jaccards = {k: [] for k in ks}
    n_ident = {k: 0 for k in ks}
    n_total = 0
    winner = {k: [0, 0] for k in ks}
    floors = []
    for key in PANEL:
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
        Est = (ChimeraBoostRegressor if task == "regression"
               else ChimeraBoostClassifier)
        full_sets = []
        for seed in range(args.seeds):
            strat = y if task != "regression" else None
            Xtr, _, ytr, _ = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=strat)
            full = Est(cross_features=False, random_state=0).fit(
                Xtr, ytr, cat_features=cat)
            p_full = pair_set(full, cat, Xtr.shape[1])
            full_sets.append(p_full)
            n_total += 1
            cells, wcells = [], []
            for k in ks:
                trunc = Est(cross_features=False, random_state=0).fit(
                    Xtr, ytr, cat_features=cat, callbacks=[stopper(k)])
                p_trunc = pair_set(trunc, cat, Xtr.shape[1])
                jac = _jac(p_full, p_trunc)
                jaccards[k].append(jac)
                n_ident[k] += p_full == p_trunc
                cells.append(f"{jac:.2f}")
                if task == "regression" and full.linear_leaves_selected_ is not None:
                    m = (full.linear_leaves_selected_
                         == trunc.linear_leaves_selected_)
                    winner[k][0] += m
                    winner[k][1] += 1
                    wcells.append("Y" if m else "N")
                else:
                    wcells.append("-")
            print(f"| {key} | {seed} | " + " | ".join(cells)
                  + " | " + " ".join(wcells) + " |")
        fl = [_jac(a, b) for i, a in enumerate(full_sets)
              for b in full_sets[i + 1:]]
        if fl:
            floors.append(sum(fl) / len(fl))
            print(f"|   (seed floor {key}) |  | "
                  + " | ".join([f"{floors[-1]:.2f}"] * len(ks)) + " |  |")

    for k in ks:
        js = jaccards[k]
        print(f"\nk={k}: pair sets identical {n_ident[k]}/{n_total}, "
              f"mean Jaccard {sum(js)/len(js):.3f}, "
              f"ll-winner match {winner[k][0]}/{winner[k][1]}.")
    print(f"Cross-seed noise floor of the FULL fits: mean Jaccard "
          f"{sum(floors)/len(floors):.3f} (includes split noise, so an "
          f"optimistic upper bound on stability).")


if __name__ == "__main__":
    main()
