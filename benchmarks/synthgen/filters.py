"""Freeze-time dataset filters (sklearn allowed here; never at benchmark runtime).

TabICLv2-style quality control: drop degenerate, unlearnable, or intractable
candidates before a suite is frozen. Frozen ids therefore never need filtering
again -- benchmark runtime stays numpy-only.
"""
import numpy as np


def degeneracy_ok(X, y, task):
    if task == "regression":
        if np.asarray(y).std() < 1e-8:
            return False, "constant target"
        return True, ""
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    min_count = max(10, int(0.005 * len(y)))
    if counts.min() < min_count:
        return False, f"class count {counts.min()} < {min_count}"
    return True, ""


def _encode_for_sklearn(X, cat_idx):
    """Ordinal-encode cats, median-impute numeric NaN (filter use only)."""
    n, d = X.shape
    out = np.empty((n, d), dtype=np.float64)
    cat_set = set(cat_idx or [])
    for j in range(d):
        col = X[:, j]
        if j in cat_set:
            _, codes = np.unique(col.astype(str), return_inverse=True)
            out[:, j] = codes
        else:
            v = col.astype(np.float64)
            bad = ~np.isfinite(v)
            if bad.any():
                med = np.nanmedian(np.where(bad, np.nan, v))
                v = np.where(bad, med if np.isfinite(med) else 0.0, v)
            out[:, j] = v
    return out


def learnable(X, y, cat_idx, task, seed=0, max_rows=2000):
    """ExtraTrees must beat the constant baseline (TabICLv2 learnability gate)."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        X, y = X[idx], np.asarray(y)[idx]
    Xe = _encode_for_sklearn(X, cat_idx)
    strat = None if task == "regression" else y
    Xtr, Xte, ytr, yte = train_test_split(Xe, y, test_size=0.25, random_state=seed,
                                          stratify=strat)
    if task == "regression":
        model = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        ss_res = float(((yte - pred) ** 2).mean())
        ss_tot = float(((yte - ytr.mean()) ** 2).mean())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return r2 >= 0.05, {"r2": round(r2, 4)}
    model = ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(Xtr, ytr)
    acc = float((model.predict(Xte) == yte).mean())
    prior = float(np.bincount(np.asarray(ytr, dtype=np.int64)).max() / len(ytr))
    return acc >= prior + 0.03, {"acc": round(acc, 4), "prior": round(prior, 4)}


# "Provably done" bounds. Verified across the SAME 3 seeds the benchmark
# harness runs: a single-seed check at a loose 0.02 admitted sets with real
# multi-seed headroom (excess 0.009-0.025) that forced cat_combinations then
# genuinely captured -- a canary must have nothing left to capture.
CANARY_SEEDS = (0, 1, 2)
CANARY_XS_BRIER_MEAN = 0.005   # clf: mean excess Brier over the stored floor
CANARY_XS_BRIER_MAX = 0.01     # clf: worst single seed
CANARY_RMSE_RATIO = 1.1        # reg: mean RMSE vs the generative sigma


def at_ceiling(X, y, cat_idx, task, meta, seeds=CANARY_SEEDS):
    """Freeze-time canary verification: default ChimeraBoost must provably
    reach the known floor on the harness's own splits (test 25%, one per
    benchmark seed). Saturated sets that fail are genuinely hard
    (car-analogs), not canaries -- v1's canary-by-construction assumption is
    exactly what this replaces.
    """
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
    from sklearn.model_selection import train_test_split

    strat = None if task == "regression" else y
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    vals = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=seed, stratify=strat)
        m = Est(n_estimators=2000, early_stopping_rounds=50, random_state=0)
        m.fit(Xtr, ytr, cat_features=cat_idx)
        if task == "regression":
            rmse = float(np.sqrt(np.mean((yte - m.predict(Xte)) ** 2)))
            vals.append(rmse / max(float(meta["noise_sigma"] or 0.0), 1e-12))
        else:
            proba = m.predict_proba(Xte)
            classes = getattr(m, "classes_", np.unique(yte))
            onehot = (np.asarray(yte)[:, None]
                      == np.asarray(classes)[None, :]).astype(float)
            brier = float(np.mean(((proba - onehot) ** 2).sum(axis=1)))
            vals.append(brier - float(meta["bayes_brier"] or 0.0))
    mean, worst = float(np.mean(vals)), float(np.max(vals))
    if task == "regression":
        return mean <= CANARY_RMSE_RATIO, {"rmse_ratio": round(mean, 3)}
    ok = mean <= CANARY_XS_BRIER_MEAN and worst <= CANARY_XS_BRIER_MAX
    return ok, {"excess_brier": round(mean, 4), "excess_max": round(worst, 4)}


def tractable(meta):
    """Cap combinatorics so forced cat_combinations arms can't explode."""
    if meta["n_cat"] >= 2 and meta["cat_fraction"] >= 1.0:
        if meta["max_cardinality"] ** 2 > 4096:
            return False, f"all-cat pair cells {meta['max_cardinality']}^2 > 4096"
    return True, ""
