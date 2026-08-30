"""Bit-identity snapshot for output-identical refactors.

The golden panel (tests/test_no_regression.py) pins three losses at a 2%
band; this pins ~30 configs exactly, including n=6000 configs that arm the
cross-feature race / selection audition / forced-cross paths (the default-N
configs never reach CROSS_MIN_SAMPLES post-split) plus bagged and
conformalized-quantile fits. Usage:

    python benchmarks/identity_snapshot.py save    # once, at the base commit
    python benchmarks/identity_snapshot.py check   # after every refactor commit

`check` refits every config and asserts np.array_equal (exact, not allclose)
against the saved arrays. The snapshot lives in benchmarks/results/ (gitignored)
because it is machine- and commit-local, not a shared artifact. A verdict file
is written next to it -- trust the file over scrolled stdout on this machine.
"""
import sys
from pathlib import Path

import numpy as np

from chimeraboost import (ChimeraBoostClassifier, ChimeraBoostQuantileRegressor,
                          ChimeraBoostRegressor)

SNAP = Path(__file__).parent / "results" / "identity_baseline.npz"
VERDICT = Path(__file__).parent / "results" / "identity_check.txt"

N = 2500
N_EST = 200


def _data(seed, n=N, kind="reg", cats=False):
    rng = np.random.default_rng(seed)
    Xn = rng.normal(size=(n, 8))
    signal = (np.sin(Xn[:, 0] * 2) + Xn[:, 1] * Xn[:, 2]
              + (Xn[:, 3] > 0.5) * 2.0 + Xn[:, 4])
    if cats:
        c1 = rng.integers(0, 12, n)
        c2 = rng.choice(list("abcdefg"), n)
        signal = signal + (c1 % 3) * 1.5 + (c2 == "c") * 2.0
        X = np.empty((n, 10), dtype=object)
        X[:, :8] = Xn
        X[:, 8] = c1
        X[:, 9] = c2
        cat_idx = [8, 9]
    else:
        X, cat_idx = Xn, None
    noise = rng.normal(scale=0.5, size=n)
    if kind == "reg":
        y = signal + noise
    elif kind == "pos":
        y = np.exp((signal + noise) / 4.0)
    elif kind == "bin":
        y = (signal + noise > np.median(signal)).astype(int)
    else:  # 3-class
        q = np.quantile(signal + noise, [1 / 3, 2 / 3])
        y = np.digitize(signal + noise, q)
    w = rng.uniform(0.5, 2.0, n)
    cut = int(n * 0.8)
    return (X[:cut], y[:cut], w[:cut], X[cut:], cat_idx)


def _configs():
    """(name, estimator factory, data kwargs, fit uses weights?)"""
    R, C, Q = (ChimeraBoostRegressor, ChimeraBoostClassifier,
               ChimeraBoostQuantileRegressor)
    base = dict(n_estimators=N_EST, random_state=0)
    cfgs = []

    def add(name, cls, params, data, weighted=False):
        cfgs.append((name, cls, {**base, **params}, data, weighted))

    add("rmse", R, {}, dict(seed=1))
    add("rmse_w", R, {}, dict(seed=1), weighted=True)
    add("rmse_sub", R, {"subsample": 0.7}, dict(seed=2))
    add("rmse_sub_w", R, {"subsample": 0.7}, dict(seed=2), weighted=True)
    add("rmse_noes", R, {"early_stopping": False, "n_estimators": 80},
        dict(seed=3))
    add("mae", R, {"loss": "MAE"}, dict(seed=4))
    add("mae_w_sub", R, {"loss": "MAE", "subsample": 0.7}, dict(seed=4),
        weighted=True)
    add("quantile", R, {"loss": "Quantile", "alpha": 0.3}, dict(seed=5))
    add("quantile_w", R, {"loss": "Quantile", "alpha": 0.3}, dict(seed=5),
        weighted=True)
    add("huber", R, {"loss": "Huber"}, dict(seed=6))
    add("poisson", R, {"loss": "Poisson"}, dict(seed=7, kind="pos"))
    add("ordered", R, {"ordered_boosting": True}, dict(seed=8))
    add("linear", R, {"linear_leaves": True}, dict(seed=9))
    add("cats", R, {}, dict(seed=10, cats=True))
    add("logloss", C, {}, dict(seed=11, kind="bin"))
    add("logloss_w_sub", C, {"subsample": 0.7}, dict(seed=11, kind="bin"),
        weighted=True)
    add("multiclass", C, {}, dict(seed=12, kind="multi"))
    add("multiclass_cats", C, {}, dict(seed=13, kind="multi", cats=True))
    # The only config that routes MVS subsampling through the classifier's
    # vector-leaf round (booster._mvs_row_weights via the sketched gradient).
    # The other subsampled configs reach _mvs_row_weights only through the
    # quantile head, or use the scalar _maybe_subsample path.
    add("multiclass_sub", C, {"subsample": 0.7}, dict(seed=12, kind="multi"))
    add("mq3", Q, {"quantiles": [0.1, 0.5, 0.9]}, dict(seed=14))
    add("mq3_w_sub", Q, {"quantiles": [0.1, 0.5, 0.9], "subsample": 0.7},
        dict(seed=14), weighted=True)
    # The default N leaves 1600 post-ES-split rows -- below CROSS_MIN_SAMPLES
    # (2000), so none of the configs above ever runs the cross-feature race,
    # the selection-rounds audition, or forced cross. These n=6000 configs
    # (3840 post-split) arm those paths; _EXPECT asserts each one actually
    # fired, so gate coverage is proven at save time, not assumed.
    add("rmse_big", R, {}, dict(seed=15, n=6000))
    add("rmse_big_race", R, {"selection_rounds": None}, dict(seed=15, n=6000))
    add("rmse_forced", R, {"cross_features": "always"}, dict(seed=16, n=6000))
    add("rmse_norefit", R, {"refit_full": False}, dict(seed=15, n=6000))
    add("logloss_big", C, {}, dict(seed=17, kind="bin", n=6000))
    add("multiclass_big", C, {}, dict(seed=18, kind="multi", n=6000))
    add("bag", R, {"n_ensembles": 3}, dict(seed=19))
    add("mq3_conf", Q, {"quantiles": [0.1, 0.5, 0.9], "conformalize": True},
        dict(seed=20))
    return cfgs


# Path-activation proofs: fitting a config is not evidence the code path it
# exists for actually ran (the original suite silently skipped the cross race
# for years of commits). Each check raises at save/check time if its path
# went dark, e.g. because a constant or split fraction moved.
_EXPECT = {
    "rmse_big": lambda e: (e.linear_leaves_selected_ is not None
                           and e.cross_features_selected_ is not None),
    "rmse_big_race": lambda e: (e.linear_leaves_selected_ is not None
                                and e.cross_features_selected_ is not None),
    "rmse_forced": lambda e: e.cross_features_selected_ is True,
    "rmse_norefit": lambda e: e.linear_leaves_selected_ is not None,
    "logloss_big": lambda e: e.cross_features_selected_ is not None,
    "multiclass_big": lambda e: e.cross_features_selected_ is not None,
    "bag": lambda e: (e.estimators_ is not None and len(e.estimators_) == 3),
    "mq3_conf": lambda e: not np.all(e.conformal_scale_ == 1.0),
}


def _run_one(name, cls, params, data_kw, weighted):
    X, y, w, Xte, cat_idx = _data(**data_kw)
    est = cls(**params)
    fit_kw = {}
    if cat_idx is not None:
        fit_kw["cat_features"] = cat_idx
    if weighted:
        fit_kw["sample_weight"] = w
    est.fit(X, y, **fit_kw)

    expect = _EXPECT.get(name)
    if expect is not None and not expect(est):
        raise AssertionError(
            f"config {name!r} no longer exercises the path it exists to pin "
            "-- see _EXPECT; the gate would be silently weaker than saved.")

    out = {
        f"{name}__pred": np.asarray(est.predict(Xte), dtype=np.float64),
        f"{name}__imp": np.asarray(est.feature_importances_,
                                   dtype=np.float64),
    }
    if getattr(est, "estimators_", None) is not None:
        for i, m in enumerate(est.estimators_):
            out[f"{name}__m{i}_n_trees"] = np.asarray(len(m.model_.trees_))
            out[f"{name}__m{i}_valid_hist"] = np.asarray(
                m.model_.valid_history_, dtype=np.float64)
    else:
        out[f"{name}__n_trees"] = np.asarray(len(est.model_.trees_))
        out[f"{name}__valid_hist"] = np.asarray(est.model_.valid_history_,
                                                dtype=np.float64)
    # Calibration outputs are the direct products of the conformal /
    # temperature blocks -- pin them explicitly rather than only through pred.
    for attr in ("quantile_offset_", "temperature_", "conformal_scale_"):
        val = getattr(est, attr, None)
        if val is not None:
            out[f"{name}__{attr.rstrip('_')}"] = np.asarray(val,
                                                            dtype=np.float64)
    if hasattr(est, "predict_proba"):
        out[f"{name}__proba"] = est.predict_proba(Xte)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    arrays = {}
    for cfg in _configs():
        arrays.update(_run_one(*cfg))
        print(f"fitted {cfg[0]}", flush=True)

    if mode == "save":
        SNAP.parent.mkdir(exist_ok=True)
        np.savez_compressed(SNAP, **arrays)
        print(f"saved {len(arrays)} arrays -> {SNAP}")
        return 0

    base = np.load(SNAP, allow_pickle=False)
    lines, failed = [], []
    for key in sorted(set(base.files) | set(arrays)):
        if key not in base.files:
            ok, msg = False, "missing from baseline"
        elif key not in arrays:
            ok, msg = False, "missing from current run"
        else:
            ok = (base[key].shape == arrays[key].shape
                  and np.array_equal(base[key], arrays[key]))
            msg = "identical" if ok else (
                f"DIFFERS shape {base[key].shape} vs {arrays[key].shape}"
                if base[key].shape != arrays[key].shape else
                f"DIFFERS max|d|={np.max(np.abs(base[key] - arrays[key]))}")
        lines.append(f"{'PASS' if ok else 'FAIL'}  {key}: {msg}")
        if not ok:
            failed.append(key)
    summary = (f"{len(base.files) - len(failed)}/{len(base.files)} identical"
               + (f"; FAILED: {failed}" if failed else " -- bit-identical"))
    VERDICT.write_text("\n".join(lines + ["", summary]) + "\n",
                       encoding="utf-8")
    print(summary)
    print(f"verdict -> {VERDICT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
