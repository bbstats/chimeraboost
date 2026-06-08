"""Per-(dataset, seed) fitting: the fast (paired-curve) and promotion (true
test-metric) tiers, plus the shared-baseline cache.

Fast tier
    Fit baseline and variant with ``early_stopping=False``,
    ``n_estimators=FAST_HORIZON`` and an explicit ``eval_set`` so
    ``validation_history_`` is the COMPLETE validation-loss curve to the horizon
    (the stopper never fires -> never truncated). No test predictions. The
    decision signal is the paired curve comparison (see ``curves``).

Promotion tier
    Fit with normal early stopping on the val split, predict the held-out TEST
    split, and compute the true metric (RMSE / Brier / F1). The paired delta
    (variant - baseline) per dataset feeds the sign test in ``cascade``.

Shared-baseline cache
    The out-of-box baseline is fit once per (dataset, seed) and its curve / test
    metrics are cached to disk keyed by (dataset, seed, code_version). Every idea
    reuses it; only the variant refits.
"""

import hashlib
import os
import pickle
import subprocess

import numpy as np
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# Reuse the shared harness budget so a promotion fit == the shipped default.
import run_benchmarks as rb

FAST_HORIZON = 300   # fixed tree budget for the fast-tier validation curve
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cache", "baseline")


# ---------------------------------------------------------------------------
# Code versioning -- the baseline cache must invalidate when the source changes.
# ---------------------------------------------------------------------------
def code_version():
    """A short hash identifying the current ChimeraBoost source. Prefers the git
    HEAD (+ '-dirty' when the working tree is modified); falls back to a hash of
    the package source files when git is unavailable."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--", "chimeraboost"], cwd=root,
            stderr=subprocess.DEVNULL) != 0
        return head + ("-dirty" if dirty else "")
    except Exception:
        import chimeraboost
        pkg = os.path.dirname(chimeraboost.__file__)
        h = hashlib.sha1()
        for fn in sorted(os.listdir(pkg)):
            if fn.endswith(".py"):
                with open(os.path.join(pkg, fn), "rb") as f:
                    h.update(f.read())
        return "src-" + h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Splitting + metrics.
# ---------------------------------------------------------------------------
def three_way_split(X, y, task, seed, test_frac=0.2, val_frac=0.2):
    """Split into train / val / test. ``val_frac`` is a fraction of the
    post-test remainder. Stratified for classification."""
    strat = y if task != "regression" else None
    X_rest, X_te, y_rest, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=strat)
    strat2 = y_rest if task != "regression" else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac, random_state=seed, stratify=strat2)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _est(task, params, seed, threads):
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    kw = dict(n_estimators=rb.MAX_ITERS, early_stopping_rounds=rb.PATIENCE,
              random_state=seed, thread_count=threads)
    kw.update(params)
    return Est(**kw)


def test_metrics(task, model, X_te, y_te):
    """True held-out test metrics. ``primary`` is lower-is-better (RMSE for
    regression, Brier for classification) to match the cheap val-loss direction;
    ``f1`` (higher better) is reported alongside for classification."""
    if task == "regression":
        pred = np.asarray(model.predict(X_te), dtype=float)
        rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        return {"primary": rmse, "rmse": rmse}
    from sklearn.metrics import f1_score
    proba = np.asarray(model.predict_proba(X_te), dtype=float)
    classes = np.asarray(model.classes_)
    onehot = (np.asarray(y_te)[:, None] == classes[None, :]).astype(float)
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    f1 = float(f1_score(y_te, model.predict(X_te), average="macro"))
    return {"primary": brier, "brier": brier, "f1": f1}


# ---------------------------------------------------------------------------
# Fast tier: paired validation curves from a single fit each.
# ---------------------------------------------------------------------------
def fast_curve(task, params, X_tr, y_tr, X_val, y_val, cat, seed, threads,
               horizon=FAST_HORIZON):
    """Fit to a fixed horizon with no early stopping and return the full
    validation-loss curve (``validation_history_``)."""
    p = dict(params)
    p.update(n_estimators=horizon, early_stopping=False)
    m = _est(task, p, seed, threads)
    m.fit(X_tr, y_tr, cat_features=cat, eval_set=(X_val, y_val))
    return list(m.validation_history_)


# ---------------------------------------------------------------------------
# Promotion tier: true test metric with normal early stopping.
# ---------------------------------------------------------------------------
def promotion_metrics(task, params, X_tr, y_tr, X_val, y_val, X_te, y_te, cat,
                      seed, threads):
    """Fit with early stopping on the val split, then score the held-out test
    split. Returns (metrics_dict, n_trees)."""
    m = _est(task, params, seed, threads)
    m.fit(X_tr, y_tr, cat_features=cat, eval_set=(X_val, y_val))
    return test_metrics(task, m, X_te, y_te), int(m.best_iteration_)


# ---------------------------------------------------------------------------
# Shared-baseline cache (curve + promotion metrics), keyed by code version.
# ---------------------------------------------------------------------------
def _baseline_path(dataset, seed, kind, horizon, codever):
    safe = dataset.replace(":", "__").replace("/", "_")
    return os.path.join(_CACHE_DIR,
                        f"{safe}__seed{seed}__{kind}__h{horizon}__{codever}.pkl")


def cached_baseline_curve(dataset, task, X_tr, y_tr, X_val, y_val, cat, seed,
                          threads, horizon=FAST_HORIZON):
    """Baseline fast-tier curve, from cache when present (keyed by code version)."""
    path = _baseline_path(dataset, seed, "curve", horizon, code_version())
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    curve = fast_curve(task, {}, X_tr, y_tr, X_val, y_val, cat, seed, threads,
                       horizon)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(curve, f)
    return curve


def cached_baseline_promotion(dataset, task, X_tr, y_tr, X_val, y_val, X_te,
                              y_te, cat, seed, threads):
    """Baseline promotion-tier metrics, from cache when present."""
    path = _baseline_path(dataset, seed, "promo", 0, code_version())
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = promotion_metrics(task, {}, X_tr, y_tr, X_val, y_val, X_te, y_te, cat,
                            seed, threads)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out
