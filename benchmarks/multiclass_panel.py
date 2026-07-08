"""Multiclass characterization panel: log loss vs CatBoost/LightGBM/XGBoost,
single models AND TabArena-style 8-fold bags.

Motivation: the public TabArena leaderboard shows multiclass as ChimeraBoost's
weakest problem type. Our 2026-06-08 check found single-model parity on 8
OpenML sets — but TabArena scores 8-fold BAGGED models on LOG LOSS, neither of
which that check covered. This panel re-establishes ground truth on
independent suites (OpenML + PMLB multiclass; TabArena itself stays sealed).

Bag proxy = AutoGluon's bagging: stratified 8-fold split of the training data,
each child fits on 7/8 and early-stops on its held-out fold, test prediction
is the mean of the children's probabilities.

Usage:
    python benchmarks/multiclass_panel.py                # full panel
    python benchmarks/multiclass_panel.py --datasets oml:vehicle
Results append to benchmarks/results/multiclass_panel.jsonl (resumable);
the aggregate table prints at the end of every run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402  (dataset registry + single runners)

from chimeraboost import ChimeraBoostClassifier  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.path.join(RESULTS_DIR, "multiclass_panel.jsonl")

# OpenML multiclass suite + PMLB multiclass sets (pm names resolved to their
# fold-qualified registry keys at runtime). PMLB `nursery`/`segmentation`
# duplicate oml:nursery / oml:segment and are skipped.
OML = ["oml:vehicle", "oml:segment", "oml:optdigits", "oml:car", "oml:splice",
       "oml:nursery", "oml:satimage", "oml:pendigits", "oml:letter"]
PMLB = ["yeast", "contraceptive_method", "texture", "dna", "page_blocks",
        "ann_thyroid", "mfeat_factors", "krkopt"]

SINGLE_SEEDS = (0, 1, 2)
BAG_SEEDS = (0, 1)
N_BAG_FOLDS = 8


def _dataset_keys():
    rb._add_openml_datasets()
    rb._add_pmlb_datasets()
    keys = list(OML)
    for name in PMLB:
        match = [k for k in rb.DATASETS if k.startswith("pm:")
                 and k.endswith("/" + name)]
        keys.extend(sorted(match)[:1])
    return keys


def _aligned_proba(model, Xte, classes_global):
    """predict_proba mapped onto the global class set (a bag child may not
    have seen every class)."""
    p = model.predict_proba(Xte)
    child = np.asarray(model.classes_)
    out = np.zeros((p.shape[0], len(classes_global)))
    col = {c: j for j, c in enumerate(classes_global)}
    for i, c in enumerate(child):
        out[:, col[c]] = p[:, i]
    return out


def _fit_child(lib, Xf, yf, Xv, yv, Xte, cat):
    """Fit one bag child with (Xv, yv) as its early-stopping set; return the
    test proba and its classes_. Mirrors each library's conventions from
    run_benchmarks.py."""
    if lib == "ChimeraBoost":
        m = ChimeraBoostClassifier(n_estimators=rb.MAX_ITERS,
                                   early_stopping_rounds=rb.PATIENCE,
                                   thread_count=None, random_state=0)
        m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
        return m, Xte
    if lib == "CatBoost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(n_estimators=rb.MAX_ITERS,
                               early_stopping_rounds=rb.PATIENCE,
                               thread_count=-1, verbose=False, random_seed=0)
        m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
        return m, Xte
    if lib == "LightGBM":
        import lightgbm as lgb
        fit_kw = dict(callbacks=[lgb.early_stopping(rb.PATIENCE, verbose=False)])
        if cat is not None:
            Xf, Xv, Xte = rb._lgb_prepare(Xf, Xv, Xte, list(cat))
            fit_kw["categorical_feature"] = list(cat)
        fit_kw["eval_set"] = [(Xv, yv)]
        m = lgb.LGBMClassifier(n_estimators=rb.MAX_ITERS, n_jobs=-1,
                               random_state=0, verbosity=-1)
        m.fit(Xf, yf, **fit_kw)
        return m, Xte
    if lib == "XGBoost":
        import xgboost as xgb
        if cat is not None:
            Xf, Xv, Xte = rb._xgb_dataframes(Xf, Xv, Xte, list(cat))
            extra = {"enable_categorical": True}
        else:
            extra = {}
        m = xgb.XGBClassifier(n_estimators=rb.MAX_ITERS,
                              early_stopping_rounds=rb.PATIENCE,
                              n_jobs=-1, random_state=0, verbosity=0,
                              tree_method="hist", **extra)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        return m, Xte
    raise ValueError(lib)


def _run_bag(lib, Xtr, ytr, Xte, yte, cat):
    t = time.time()
    classes = np.unique(np.concatenate([ytr, yte]))
    min_count = np.bincount(ytr).min() if ytr.min() >= 0 else 2
    n_splits = int(min(N_BAG_FOLDS, max(2, min_count)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = np.zeros((len(yte), len(classes)))
    for tr_idx, va_idx in skf.split(Xtr, ytr):
        m, Xte_in = _fit_child(lib, Xtr[tr_idx], ytr[tr_idx],
                               Xtr[va_idx], ytr[va_idx], Xte, cat)
        proba += _aligned_proba(m, Xte_in, classes)
    proba /= n_splits
    ll = float(log_loss(yte, proba, labels=classes))
    f1 = float(f1_score(yte, classes[np.argmax(proba, axis=1)], average="macro"))
    return {"log_loss": ll, "f1_macro": f1}, time.time() - t


SINGLE_RUNNERS = {
    "ChimeraBoost": rb._run_chimera,
    "CatBoost": rb._run_catboost,
    "LightGBM": rb._run_lightgbm,
    "XGBoost": rb._run_xgboost,
}


def _load_done():
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done[(r["dataset"], r["model"], r["variant"], r["seed"])] = r
    return done


def _append(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+",
                    default=["ChimeraBoost", "CatBoost", "LightGBM", "XGBoost"])
    ap.add_argument("--table-only", action="store_true",
                    help="skip runs; just print the aggregate table")
    ap.add_argument("--out", default=None, metavar="NAME",
                    help="results filename under benchmarks/results/ (default "
                         "multiclass_panel.jsonl); use a separate file for "
                         "A/B runs against a modified library")
    args = ap.parse_args()
    if args.out:
        global RESULTS
        RESULTS = os.path.join(RESULTS_DIR, args.out)

    all_keys = _dataset_keys()          # also registers the dataset builders
    keys = args.datasets or all_keys
    done = _load_done()

    if not args.table_only:
        for ds in keys:
            X, y, cat, task = rb.DATASETS[ds](1.0, np.random.default_rng(0))
            assert task == "multiclass", ds
            for seed in sorted(set(SINGLE_SEEDS) | set(BAG_SEEDS)):
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=y)
                for model in args.models:
                    if seed in SINGLE_SEEDS:
                        k = (ds, model, "single", seed)
                        if k not in done:
                            out = SINGLE_RUNNERS[model](
                                task, Xtr, ytr, Xte, yte, cat, None)
                            if out is not None:
                                metrics, secs, _ = out
                                row = {"dataset": ds, "model": model,
                                       "variant": "single", "seed": seed,
                                       "log_loss": metrics["log_loss"],
                                       "f1_macro": metrics["f1_macro"],
                                       "fit_s": round(secs, 2)}
                                _append(row)
                                done[k] = row
                                print(f"{ds} {model} single s{seed}: "
                                      f"ll={metrics['log_loss']:.4f} "
                                      f"({secs:.0f}s)", flush=True)
                    if seed in BAG_SEEDS:
                        k = (ds, model, "bag8", seed)
                        if k not in done:
                            metrics, secs = _run_bag(
                                model, Xtr, ytr, Xte, yte, cat)
                            row = {"dataset": ds, "model": model,
                                   "variant": "bag8", "seed": seed,
                                   "log_loss": metrics["log_loss"],
                                   "f1_macro": metrics["f1_macro"],
                                   "fit_s": round(secs, 2)}
                            _append(row)
                            done[k] = row
                            print(f"{ds} {model} bag8   s{seed}: "
                                  f"ll={metrics['log_loss']:.4f} "
                                  f"({secs:.0f}s)", flush=True)

    _print_table(done, keys, args.models)


def _print_table(done, keys, models):
    """Per variant: mean %-of-best log loss across datasets (higher=better,
    100 = best model on that dataset) + W/L of each competitor vs ChimeraBoost."""
    for variant in ("single", "bag8"):
        per_model = {m: {} for m in models}
        for ds in keys:
            for m in models:
                vals = [r["log_loss"] for (d, mm, v, s), r in done.items()
                        if d == ds and mm == m and v == variant]
                if vals:
                    per_model[m][ds] = float(np.mean(vals))
        common = set.intersection(*[set(per_model[m]) for m in models
                                    if per_model[m]]) if models else set()
        if not common:
            continue
        print(f"\n== {variant} — log loss, {len(common)} datasets ==")
        print(f"| model | mean %-of-best | W/L vs ChimeraBoost |")
        print(f"|---|---|---|")
        for m in models:
            pct = np.mean([min(per_model[x][ds] for x in models) /
                           per_model[m][ds] * 100 for ds in sorted(common)])
            if m == "ChimeraBoost":
                wl = "—"
            else:
                w = sum(per_model[m][ds] < per_model["ChimeraBoost"][ds] - 1e-12
                        for ds in common)
                l = sum(per_model[m][ds] > per_model["ChimeraBoost"][ds] + 1e-12
                        for ds in common)
                wl = f"{w}W/{l}L (competitor wins)"
            print(f"| {m} | {pct:.1f}% | {wl} |")
        worst = sorted(common, key=lambda ds: min(per_model[x][ds] for x in models)
                       / per_model["ChimeraBoost"][ds])[:5]
        print("worst ChimeraBoost sets:",
              ", ".join(f"{ds} ({min(per_model[x][ds] for x in models) / per_model['ChimeraBoost'][ds] * 100:.0f}%)"
                        for ds in worst))


if __name__ == "__main__":
    main()
