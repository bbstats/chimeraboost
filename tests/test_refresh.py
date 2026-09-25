"""Stored training rows round-trip (issue #131 slice 1a2).

A fitted booster's rows are kept as binner bins (plain numerics), raw
floats (cross parents) and int codes (categoricals); ``rebuild_X`` turns
them back into a raw-equivalent array the existing replay refit consumes
unchanged. Every comparison here is bit for bit.

Slice 1b (below the line) covers the public ``store_training_data`` /
``refresh`` API; every comparison there is bit for bit too, never allclose.
"""

import inspect
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from chimeraboost import (ChimeraBoostClassifier, ChimeraBoostRegressor,
                          CustomObjective)
from chimeraboost.booster import GradientBoosting, _BaseBooster
from chimeraboost.preprocessing import FeaturePreprocessor
from chimeraboost.target_encoding import factorize
from chimeraboost.training_rows import TrainingRows

CAT = [6, 7]
N_NUM = 6


def _cats(rng, n, levels, p_nan):
    c = np.array(rng.choice(levels, n), dtype=object)
    c[rng.random(n) < p_nan] = np.nan
    return c


def _target(rng, num, hi, lo):
    x = np.where(np.isfinite(num), num, 0.0)
    base = (3.0 * (x[:, 0] > x[:, 1]) + x[:, 2] * x[:, 3]
            + 0.5 * x[:, 4] - 0.3 * x[:, 5])
    lo_eff = np.array([{"a": 0.0, "b": 0.5, "c": -0.5, "d": 1.0, "e": -1.0}.get(
        v, 0.0) for v in lo])
    hi_eff = np.zeros(len(lo))
    for k, v in enumerate(hi):
        if isinstance(v, str) and v.startswith("L"):
            try:
                hi_eff[k] = (int(v[1:]) % 10) * 0.2
            except ValueError:
                pass
    return base + lo_eff + hi_eff + 0.1 * rng.standard_normal(len(lo))


def _weights(rng, n):
    w = rng.uniform(0.5, 1.5, n)
    w[rng.random(n) < 0.01] = 0.0
    return w


def _frame(num, hi, lo):
    X = np.empty((len(hi), N_NUM + 2), dtype=object)
    X[:, :N_NUM] = num
    X[:, 6] = hi
    X[:, 7] = lo
    return X


def _make_data(seed=0):
    rng = np.random.default_rng(seed)
    n_full, n_held, n_new = 2600, 400, 300
    num = rng.standard_normal((n_full + n_held + n_new, N_NUM))
    num[rng.random(num.shape) < 0.05] = np.nan
    hi_levels = [f"L{i}" for i in range(350)]
    lo_levels = ["a", "b", "c", "d", "e"]
    hi_full = _cats(rng, n_full, hi_levels, 0.02)
    lo_full = _cats(rng, n_full, lo_levels, 0.02)
    hi_held = _cats(rng, n_held, hi_levels, 0.02)
    lo_held = _cats(rng, n_held, lo_levels, 0.02)
    hi_new = _cats(rng, n_new, hi_levels, 0.05)
    lo_new = _cats(rng, n_new, lo_levels, 0.05)
    unseen = rng.random(n_new) < 0.25
    hi_new[unseen] = rng.choice([f"NEW{i}" for i in range(10)], unseen.sum())
    lo_new[rng.random(n_new) < 0.15] = "z"
    n0, n1 = n_full, n_full + n_held
    d = {}
    d["X_full"] = _frame(num[:n_full], hi_full, lo_full)
    d["X_held"] = _frame(num[n0:n1], hi_held, lo_held)
    d["X_new"] = _frame(num[n1:], hi_new, lo_new)
    d["y_full"] = _target(rng, num[:n_full], hi_full, lo_full)
    d["y_held"] = _target(rng, num[n0:n1], hi_held, lo_held)
    d["y_new"] = _target(rng, num[n1:], hi_new, lo_new)
    d["w_full"] = _weights(rng, n_full)
    d["w_new"] = _weights(rng, n_new)
    return d


@pytest.fixture(scope="module")
def donor():
    """A realistic fitted donor: forced cross block, count column, combos."""
    d = _make_data()
    est = ChimeraBoostRegressor(random_state=0, n_estimators=100,
                               cross_features="always", cat_combinations=True,
                               linear_leaves=True)
    est.fit(d["X_full"], d["y_full"], cat_features=CAT,
            sample_weight=d["w_full"])
    b, prep = est.model_, est.model_.prep_
    assert len(b.trees_) > 0
    assert any(t.lin_coef is not None for t in b.trees_)
    assert prep.count_features_ == [6]
    assert len(prep.cat_maps_[0]) >= 256
    ops = {op for _, _, op in b.cross_pairs}
    assert ops == {"diff", "prod", "gdiff"}, b.cross_pairs
    # Default refit_full="replay" retrains on every row: the RMSE init is the
    # FULL-data weighted mean, not the 80% split's.
    assert b.init_ == pytest.approx(
        float(np.average(d["y_full"], weights=d["w_full"])))
    d["est"], d["booster"], d["prep"] = est, b, prep
    d["rows"] = TrainingRows.capture(prep, d["X_full"], d["y_full"],
                                     d["w_full"])
    assert d["rows"].plain_features and d["rows"].parent_features
    return d


def _pinned_prep(donor_booster, donor_prep):
    """A fresh prep with the replay-refit pinning (_prep_or_replay_matrices)."""
    fresh = FeaturePreprocessor(
        donor_booster.max_bins, donor_booster.cat_smoothing,
        donor_booster.random_state, donor_booster.cat_n_permutations,
        bool(donor_prep.combo_pairs_), donor_booster.cross_pairs,
        donor_booster.cat_count_features)
    fresh._pinned_count_features = list(donor_prep.count_features_)
    fresh._pinned_cat_counts = (donor_prep.cat_maps_, donor_prep.cat_counts_)
    return fresh


def _fit_pinned(donor, X, y, w):
    prep = _pinned_prep(donor["booster"], donor["prep"])
    mat = prep.fit_transform(X, [y], CAT, w,
                             binner=donor["prep"].binner_)
    return prep, mat


def _assert_prep_equal(p1, p2):
    assert p1.cat_maps_ == p2.cat_maps_
    assert p1.combo_pairs_ == p2.combo_pairs_
    assert p1.combo_maps_ == p2.combo_maps_
    assert len(p1.gdiff_maps_) == len(p2.gdiff_maps_)
    for (d1, g1), (d2, g2) in zip(p1.gdiff_maps_, p2.gdiff_maps_):
        assert d1 == d2
        assert g1 == g2
    assert p1.count_features_ == p2.count_features_
    assert len(p1.cat_counts_) == len(p2.cat_counts_)
    for c1, c2 in zip(p1.cat_counts_, p2.cat_counts_):
        np.testing.assert_array_equal(c1, c2)
    assert len(p1.encoders_) == len(p2.encoders_)
    for e1, e2 in zip(p1.encoders_, p2.encoders_):
        assert e1.prior_ == e2.prior_
        assert e1.n_cat_ == e2.n_cat_
        for a1, a2 in zip(e1.sums_, e2.sums_):
            np.testing.assert_array_equal(a1, a2)
        for a1, a2 in zip(e1.counts_, e2.counts_):
            np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(p1.feature_map_, p2.feature_map_)
    np.testing.assert_array_equal(p1.is_numeric_binned_, p2.is_numeric_binned_)
    np.testing.assert_array_equal(p1.n_bins_, p2.n_bins_)


def test_round_trip_same_rows(donor):
    """(a) Rebuilt X refits exactly like the raw rows, same rows."""
    rows = donor["rows"]
    assert rows.n_rows == len(donor["y_full"])
    X_rb = rows.rebuild_X(donor["prep"])
    assert X_rb.shape == donor["X_full"].shape
    assert X_rb.dtype == object
    w = GradientBoosting._normalize_weights(donor["w_full"],
                                            len(donor["y_full"]))
    p1, m1 = _fit_pinned(donor, X_rb, donor["y_full"], w)
    p2, m2 = _fit_pinned(donor, donor["X_full"], donor["y_full"], w)
    assert m1.dtype == m2.dtype
    np.testing.assert_array_equal(m1, m2)
    _assert_prep_equal(p1, p2)
    np.testing.assert_array_equal(p1.transform(donor["X_held"]),
                                 p2.transform(donor["X_held"]))


def test_round_trip_appended_rows(donor):
    """(b) The same with new rows appended through ``append``."""
    prep = donor["prep"]
    before = ([dict(m) for m in prep.cat_maps_],
              [(dict(d), g) for d, g in prep.gdiff_maps_],
              [b.copy() for b in prep.binner_.borders_])
    rows = donor["rows"]
    n_old = rows.n_rows
    rows2 = rows.append(prep, donor["X_new"], donor["y_new"], donor["w_new"])
    assert rows.n_rows == n_old
    assert rows2.n_rows == n_old + len(donor["y_new"])
    assert [dict(m) for m in prep.cat_maps_] == before[0]
    assert [(dict(d), g) for d, g in prep.gdiff_maps_] == before[1]
    for b, b0 in zip(prep.binner_.borders_, before[2]):
        np.testing.assert_array_equal(b, b0)
    X_cat = np.concatenate([donor["X_full"], donor["X_new"]])
    y_cat = np.concatenate([donor["y_full"], donor["y_new"]])
    w_cat = np.concatenate([donor["w_full"], donor["w_new"]])
    for j, f in enumerate(CAT):
        codes, cats = factorize(X_cat[:, f])
        np.testing.assert_array_equal(rows2.cat_codes[:, j], codes)
        assert list(rows2.cat_values[j]) == list(cats)
    X_rb = rows2.rebuild_X(prep)
    assert X_rb.shape == X_cat.shape
    w = GradientBoosting._normalize_weights(w_cat, len(y_cat))
    p1, m1 = _fit_pinned(donor, X_rb, y_cat, w)
    p2, m2 = _fit_pinned(donor, X_cat, y_cat, w)
    np.testing.assert_array_equal(m1, m2)
    _assert_prep_equal(p1, p2)
    np.testing.assert_array_equal(p1.transform(donor["X_held"]),
                                 p2.transform(donor["X_held"]))


def _fit_replay(donor, X, linear):
    kw = donor["booster"].replay_kwargs()
    kw["n_estimators"] = len(donor["booster"].trees_)
    kw["learning_rate"] = float(donor["booster"].lr_)
    kw["early_stopping_rounds"] = None
    kw["replay_donor"] = (donor["booster"].trees_, donor["prep"])
    if linear:
        kw["linear_leaves"] = True
    b = GradientBoosting(**kw)
    b.fit(X, donor["y_full"], cat_features=CAT,
          sample_weight=donor["w_full"])
    return b


def _assert_booster_equal(b1, b2):
    assert b1.init_ == b2.init_
    assert len(b1.trees_) == len(b2.trees_)
    for t1, t2 in zip(b1.trees_, b2.trees_):
        np.testing.assert_array_equal(t1.values, t2.values)
        if t1.lin_coef is None or t2.lin_coef is None:
            assert t1.lin_coef is None and t2.lin_coef is None
        else:
            np.testing.assert_array_equal(t1.lin_coef, t2.lin_coef)


@pytest.mark.parametrize("linear", [False, True])
def test_booster_replay_on_rebuilt(donor, linear):
    """(c) Replay refit on rebuilt X equals the one on raw rows.

    False is the donor as fitted (linear leaves), True forces
    ``linear_leaves=True``; replay refits ``lin_coef`` off a linear donor.
    """
    X_rb = donor["rows"].rebuild_X(donor["prep"])
    b1 = _fit_replay(donor, X_rb, linear)
    b2 = _fit_replay(donor, donor["X_full"], linear)
    assert any(t.lin_coef is not None for t in b1.trees_)
    _assert_booster_equal(b1, b2)
    np.testing.assert_array_equal(b1.predict_raw(donor["X_held"]),
                                 b2.predict_raw(donor["X_held"]))


def test_exhaustive_bin_round_trip(donor):
    """(d) Every plain-numeric bin maps to a value that bins back to it."""
    binner = donor["prep"].binner_
    pos = {f: k for k, f in enumerate(donor["prep"].num_features_)}
    for f in donor["rows"].plain_features:
        borders = binner.borders_[pos[f]]
        m = len(borders)
        vals = np.empty(m + 2, dtype=np.float64)
        if m == 0:
            vals[0], vals[1] = 0.0, np.nan
        else:
            vals[0] = np.nextafter(borders[0], -np.inf)
            vals[1:m + 1] = borders
            vals[m + 1] = np.nan
        feat = np.zeros((m + 2, len(binner.borders_)))
        feat[:, pos[f]] = vals
        got = binner.transform(feat)[:, pos[f]]
        np.testing.assert_array_equal(got, np.arange(m + 2))


def test_replay_kwargs_covers_init(donor):
    """(e) ``replay_kwargs`` covers every booster init parameter."""
    kw = donor["booster"].replay_kwargs()
    base = set(inspect.signature(_BaseBooster.__init__).parameters) - {"self"}
    assert set(kw) == base | {"loss", "loss_kwargs"}
    assert kw["loss"] == donor["booster"].loss_name
    assert kw["loss_kwargs"] == donor["booster"].loss_kwargs
    assert kw["cross_pairs"] == donor["booster"].cross_pairs
    assert kw["cross_pairs"] is not donor["booster"].cross_pairs
    assert kw["loss_kwargs"] is not donor["booster"].loss_kwargs
    b2 = GradientBoosting(**kw)
    assert b2.loss_name == donor["booster"].loss_name
    assert b2.cross_pairs == donor["booster"].cross_pairs


def test_training_rows_pickle_and_dtype(donor):
    """(f) ``TrainingRows`` pickles exactly; plain bins are uint8."""
    rows = donor["rows"]
    assert rows.plain_bins.dtype == np.uint8
    rt = pickle.loads(pickle.dumps(rows))
    assert rt.n_features == rows.n_features
    assert rt.cat_features == rows.cat_features
    assert rt.num_features == rows.num_features
    assert rt.plain_features == rows.plain_features
    assert rt.parent_features == rows.parent_features
    assert rt.plain_bins.dtype == rows.plain_bins.dtype
    np.testing.assert_array_equal(rt.plain_bins, rows.plain_bins)
    np.testing.assert_array_equal(rt.parent_raw, rows.parent_raw)
    np.testing.assert_array_equal(rt.cat_codes, rows.cat_codes)
    for v1, v2 in zip(rt.cat_values, rows.cat_values):
        np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(rt.y, rows.y)
    np.testing.assert_array_equal(rt.sample_weight, rows.sample_weight)
    assert rt.n_rows == rows.n_rows


# ---------------------------------------------------------------------------
# Slice 1b: the public store_training_data / refresh API (issue #131).
# ---------------------------------------------------------------------------


class _SqErr(CustomObjective):
    """Squared error through the public custom-objective hook.

    Module level, so instances are picklable.
    """

    def init(self, y, sample_weight=None):
        return float(np.average(y, weights=sample_weight))

    def grad_hess(self, y, raw):
        return raw - y, np.ones_like(raw)

    def eval(self, y, raw, sample_weight=None):
        return float(np.sqrt(np.average((raw - y) ** 2,
                                        weights=sample_weight)))


def _binary_labels(y):
    return np.where(np.asarray(y) > np.median(y), "pos", "neg")


def _snapshot(est, Xh):
    """Predictions, init and per-tree leaves to compare a refresh against."""
    snap = {"pred": est.predict(Xh), "init": est.model_.init_,
            "n": est.n_samples_trained_,
            "trees": [(t.values.copy(),
                       None if t.lin_coef is None else t.lin_coef.copy())
                      for t in est.model_.trees_]}
    if hasattr(est, "predict_proba"):
        snap["proba"] = est.predict_proba(Xh)
    return snap


def _assert_snapshot_equal(est, snap, Xh):
    np.testing.assert_array_equal(est.predict(Xh), snap["pred"])
    if "proba" in snap:
        np.testing.assert_array_equal(est.predict_proba(Xh), snap["proba"])
    assert est.model_.init_ == snap["init"]
    assert len(est.model_.trees_) == len(snap["trees"])
    for t, (v, lc) in zip(est.model_.trees_, snap["trees"]):
        np.testing.assert_array_equal(t.values, v)
        if lc is None or t.lin_coef is None:
            assert lc is None and t.lin_coef is None
        else:
            np.testing.assert_array_equal(t.lin_coef, lc)
    assert est.n_samples_trained_ == snap["n"]


# (a) estimator-kwarg overrides per configuration. Every fit also pins an
# integer random_state, store_training_data=True, and (regressor) the
# single-fit linear_leaves=False / cross_features=False unless the
# configuration says otherwise. NaNs ride in every configuration: _make_data
# salts 5% of the numerics and 2-5% of the categoricals with NaN.
_A_EST = {
    "refit-replay": dict(refit_full="replay"),
    "refit-true": dict(refit_full=True),
    "refit-false": dict(refit_full=False),
    "eval-set": {},
    "no-es": dict(early_stopping=False, n_estimators=20),
    "count": {},
    "cross-always": dict(cross_features="always"),
    "linear": dict(linear_leaves=True),
    "weighted": {},
    "ordered": dict(ordered_boosting=True),
    "mae": dict(loss="MAE"),
    "poisson": dict(loss="Poisson"),
    "custom": dict(loss=_SqErr()),
    "subsample": dict(subsample=0.8, colsample=0.7),
    "binary": {},
}


def _cfg_parts(cfg, d):
    """The (a) estimator, fit kwargs and fit target/weights for one config."""
    base = dict(random_state=0, n_estimators=50, store_training_data=True,
                linear_leaves=False, cross_features=False)
    base.update(_A_EST[cfg])
    fit_kw = dict(cat_features=CAT)
    y_fit, w_fit = d["y_full"], None
    cls = ChimeraBoostRegressor
    if cfg == "eval-set":
        fit_kw["eval_set"] = (d["X_held"], d["y_held"])
    if cfg in ("weighted", "binary"):
        fit_kw["sample_weight"] = d["w_full"]
        w_fit = d["w_full"]
    if cfg == "poisson":
        y_fit = np.exp((y_fit - y_fit.mean()) / y_fit.std())
    if cfg == "binary":
        cls = ChimeraBoostClassifier
        del base["linear_leaves"]
        y_fit = _binary_labels(d["y_full"])
    return cls(**base), fit_kw, y_fit, w_fit


@pytest.mark.parametrize("cfg", ["refit-replay", "refit-true", "refit-false",
                                 "eval-set", "no-es", "count", "cross-always",
                                 "linear", "weighted", "ordered", "mae",
                                 "poisson", "custom", "subsample", "binary"])
def test_zero_rows_is_identity(cfg):
    """(a) refresh() with zero new rows changes nothing, bit for bit."""
    d = _make_data()
    est, fit_kw, y_fit, w_fit = _cfg_parts(cfg, d)
    est.fit(d["X_full"], y_fit, **fit_kw)
    if cfg == "count":
        assert est.model_.prep_.count_features_ == [6]
    if cfg == "cross-always":
        ops = {op for _, _, op in est.model_.cross_pairs}
        assert ops == {"diff", "prod", "gdiff"}, est.model_.cross_pairs
    if cfg == "linear":
        assert any(t.lin_coef is not None for t in est.model_.trees_)
    snap = _snapshot(est, d["X_held"])
    w0 = None if w_fit is None else w_fit[:0]
    out = est.refresh(d["X_full"][:0], y_fit[:0], sample_weight=w0)
    assert out is est
    _assert_snapshot_equal(est, snap, d["X_held"])


def test_refresh_equals_manual_replay():
    """(b) refresh() equals a manual replay refit on the concatenated rows.

    The new rows carry unseen categories, NaN categories and NaN numerics
    (see _make_data).
    """
    d = _make_data()
    est = ChimeraBoostRegressor(random_state=0, n_estimators=60,
                                cross_features="always", linear_leaves=True,
                                store_training_data=True)
    est.fit(d["X_full"], d["y_full"], cat_features=CAT,
            sample_weight=d["w_full"])
    donor = est.model_
    assert donor.prep_.count_features_ == [6]
    est.refresh(d["X_new"], d["y_new"], sample_weight=d["w_new"])
    assert est.n_samples_trained_ == len(d["y_full"]) + len(d["y_new"])

    kw = donor.replay_kwargs()
    kw["n_estimators"] = len(donor.trees_)
    kw["learning_rate"] = float(donor.lr_)
    kw["early_stopping_rounds"] = None
    kw["replay_donor"] = (donor.trees_, donor.prep_)
    man = GradientBoosting(**kw)
    man.fit(np.concatenate([d["X_full"], d["X_new"]]),
            np.concatenate([d["y_full"], d["y_new"]]),
            cat_features=CAT,
            sample_weight=np.concatenate([d["w_full"], d["w_new"]]))
    _assert_booster_equal(est.model_, man)
    np.testing.assert_array_equal(est.predict_raw(d["X_held"]),
                                  man.predict_raw(d["X_held"]))


def _pinned_snapshot(est):
    """Every refresh-pinned quantity the (c) test compares."""
    m = est.model_
    return {
        "n_estimators": est.n_estimators,
        "best": est.best_iteration_,
        "lr": m.lr_,
        "depths": [t.depth for t in m.trees_],
        "feats": [np.asarray(t.splits_feat).copy() for t in m.trees_],
        "thrs": [np.asarray(t.splits_thr).copy() for t in m.trees_],
        "lin_feats": [(None if t.lin_feats is None
                       else np.asarray(t.lin_feats).copy())
                      for t in m.trees_],
        "borders": [b.copy() for b in m.prep_.binner_.borders_],
        "count": list(m.prep_.count_features_),
        "cross_pairs": list(m.cross_pairs),
        "valid": list(m.valid_history_),
        "n_features": est.n_features_in_,
        "names": getattr(est, "feature_names_in_", None),
        "ll_sel": getattr(est, "linear_leaves_selected_", None),
        "cx_sel": est.cross_features_selected_,
        "cx_pairs": est.cross_pairs_,
        "temp": getattr(est, "temperature_", None),
        "classes": getattr(est, "classes_", None),
    }


def _assert_pinned_equal(est, snap):
    m = est.model_
    assert est.n_estimators == snap["n_estimators"]
    assert est.best_iteration_ == snap["best"]
    assert m.lr_ == snap["lr"]
    assert [t.depth for t in m.trees_] == snap["depths"]
    assert len(m.trees_) == len(snap["feats"])
    for t, f, th, lf in zip(m.trees_, snap["feats"], snap["thrs"],
                            snap["lin_feats"]):
        np.testing.assert_array_equal(np.asarray(t.splits_feat), f)
        np.testing.assert_array_equal(np.asarray(t.splits_thr), th)
        if lf is None or t.lin_feats is None:
            assert lf is None and t.lin_feats is None
        else:
            np.testing.assert_array_equal(np.asarray(t.lin_feats), lf)
    for b, b0 in zip(m.prep_.binner_.borders_, snap["borders"]):
        np.testing.assert_array_equal(b, b0)
    assert list(m.prep_.count_features_) == snap["count"]
    assert list(m.cross_pairs) == snap["cross_pairs"]
    assert list(m.valid_history_) == snap["valid"]
    assert est.n_features_in_ == snap["n_features"]
    if snap["names"] is None:
        assert getattr(est, "feature_names_in_", None) is None
    else:
        np.testing.assert_array_equal(est.feature_names_in_, snap["names"])
    assert getattr(est, "linear_leaves_selected_", None) == snap["ll_sel"]
    assert est.cross_features_selected_ == snap["cx_sel"]
    assert est.cross_pairs_ == snap["cx_pairs"]
    if snap["temp"] is not None:
        assert est.temperature_ == snap["temp"]
    if snap["classes"] is not None:
        np.testing.assert_array_equal(est.classes_, snap["classes"])


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_refresh_keeps_pinned_state(kind):
    """(c) Nothing pinned moves; n_samples_trained_ grows by the new rows."""
    d = _make_data()
    if kind == "regression":
        est = ChimeraBoostRegressor(random_state=0, n_estimators=60,
                                    cross_features="always",
                                    linear_leaves=True,
                                    store_training_data=True)
        y_fit, y_new = d["y_full"], d["y_new"]
    else:
        est = ChimeraBoostClassifier(random_state=0, n_estimators=60,
                                     cross_features="always",
                                     store_training_data=True)
        y_fit = _binary_labels(d["y_full"])
        y_new = _binary_labels(d["y_new"])
    est.fit(d["X_full"], y_fit, cat_features=CAT, sample_weight=d["w_full"])
    assert est.model_.prep_.count_features_ == [6]
    est.shap_importances(d["X_full"][:20])
    assert est._shap_importances_cache_ is not None
    assert hasattr(est, "expected_value_")
    snap = _pinned_snapshot(est)
    n0 = est.n_samples_trained_
    est.refresh(d["X_new"], y_new, sample_weight=d["w_new"])
    _assert_pinned_equal(est, snap)
    assert est.n_samples_trained_ == n0 + len(y_new)
    assert est._shap_importances_cache_ is None
    assert not hasattr(est, "expected_value_")


def test_refresh_preserves_feature_names():
    """(c) feature_names_in_ survives a refresh (DataFrame fit)."""
    pd = pytest.importorskip("pandas")
    d = _make_data()
    num_cols = [f"n{i}" for i in range(N_NUM)]

    def frame(X):
        df = pd.DataFrame(np.asarray(X[:, :N_NUM], dtype=np.float64),
                          columns=num_cols)
        df["hi"] = list(X[:, 6])
        df["lo"] = list(X[:, 7])
        return df

    cols = num_cols + ["hi", "lo"]
    est = ChimeraBoostRegressor(random_state=0, n_estimators=20,
                                store_training_data=True)
    est.fit(frame(d["X_full"]), d["y_full"], cat_features=["hi", "lo"])
    assert list(est.feature_names_in_) == cols
    est.refresh(frame(d["X_new"]), d["y_new"])
    assert list(est.feature_names_in_) == cols
    assert est.n_samples_trained_ == len(d["y_full"]) + len(d["y_new"])


def test_refresh_ignores_post_fit_set_params():
    """refresh() replays the fitted configuration, not later set_params."""
    d = _make_data()
    est = ChimeraBoostRegressor(random_state=0, n_estimators=40,
                                store_training_data=True)
    est.fit(d["X_full"][:1500], d["y_full"][:1500], cat_features=CAT)
    snap = _snapshot(est, d["X_held"])
    est.set_params(n_estimators=5, learning_rate=0.5, max_bins=16, depth=2,
                   subsample=0.5, l2_leaf_reg=99.0)
    est.refresh(d["X_full"][:0], d["y_full"][:0])
    _assert_snapshot_equal(est, snap, d["X_held"])


def test_chained_refresh_equals_single():
    """(c2) refresh(A) then refresh(B) equals one refresh(A+B), bit for bit."""
    d = _make_data()

    def fresh():
        e = ChimeraBoostRegressor(random_state=0, n_estimators=50,
                                  cross_features="always", linear_leaves=True,
                                  store_training_data=True)
        e.fit(d["X_full"], d["y_full"], cat_features=CAT,
              sample_weight=d["w_full"])
        return e

    n_a = 120
    e1 = fresh()
    e1.refresh(d["X_new"][:n_a], d["y_new"][:n_a],
               sample_weight=d["w_new"][:n_a])
    e1.refresh(d["X_new"][n_a:], d["y_new"][n_a:],
               sample_weight=d["w_new"][n_a:])
    e2 = fresh()
    e2.refresh(d["X_new"], d["y_new"], sample_weight=d["w_new"])
    assert e1.n_samples_trained_ == e2.n_samples_trained_
    _assert_booster_equal(e1.model_, e2.model_)
    np.testing.assert_array_equal(e1.predict(d["X_held"]),
                                  e2.predict(d["X_held"]))


def test_pickle_round_trip():
    """(d) A refreshed-after-round-trip model equals the refreshed original."""
    d = _make_data()
    est = ChimeraBoostRegressor(random_state=0, n_estimators=50,
                                cross_features="always", linear_leaves=True,
                                store_training_data=True)
    est.fit(d["X_full"], d["y_full"], cat_features=CAT,
            sample_weight=d["w_full"])
    rt = pickle.loads(pickle.dumps(est))
    assert rt._training_data_.n_rows == est._training_data_.n_rows
    est.refresh(d["X_new"], d["y_new"], sample_weight=d["w_new"])
    rt.refresh(d["X_new"], d["y_new"], sample_weight=d["w_new"])
    _assert_booster_equal(rt.model_, est.model_)
    np.testing.assert_array_equal(rt.predict(d["X_held"]),
                                  est.predict(d["X_held"]))
    assert rt.n_samples_trained_ == est.n_samples_trained_
    assert est.model_.replay_donor is None
    assert rt.model_.replay_donor is None

    plain = ChimeraBoostRegressor(random_state=0, n_estimators=10).fit(
        d["X_full"][:500], d["y_full"][:500], cat_features=CAT)
    assert plain._training_data_ is None


def test_store_training_data_errors():
    """(e) Every slice-1 error message."""
    d = _make_data()
    X, y = d["X_full"][:800], d["y_full"][:800]

    for bad in ("yes", 1, None):
        with pytest.raises(ValueError, match="store_training_data must be"):
            ChimeraBoostRegressor(random_state=0,
                                  store_training_data=bad).fit(
                X, y, cat_features=CAT)

    with pytest.raises(NotImplementedError, match="n_ensembles > 1"):
        ChimeraBoostRegressor(random_state=0, n_ensembles=2,
                              store_training_data=True).fit(
            X, y, cat_features=CAT)

    with pytest.raises(NotImplementedError, match="n_ensembles > 1"):
        ChimeraBoostRegressor(random_state=0, quality=4,
                              store_training_data=True).fit(
            X, y, cat_features=CAT)

    y3 = np.where(y > np.quantile(y, 2 / 3), "c",
                  np.where(y > np.quantile(y, 1 / 3), "b", "a"))
    with pytest.raises(NotImplementedError, match="multiclass classification"):
        ChimeraBoostClassifier(random_state=0,
                               store_training_data=True).fit(
            X, y3, cat_features=CAT)

    with pytest.raises(NotImplementedError, match="loss='Quantile'"):
        ChimeraBoostRegressor(random_state=0, loss="Quantile",
                              store_training_data=True).fit(
            X, y, cat_features=CAT)

    groups = np.arange(len(y)) % 20
    with pytest.raises(NotImplementedError, match="random_effects=True"):
        ChimeraBoostRegressor(random_state=0, random_effects=True,
                              store_training_data=True).fit(
            X, y, cat_features=CAT, groups=groups)

    with pytest.raises(NotFittedError):
        ChimeraBoostRegressor().refresh(X[:10], y[:10])

    plain = ChimeraBoostRegressor(random_state=0, n_estimators=10).fit(
        X, y, cat_features=CAT)
    with pytest.raises(ValueError, match="store_training_data=False"):
        plain.refresh(X[:10], y[:10])


def test_refresh_weight_and_label_errors():
    """(e) Weighted/unweighted mismatches and unseen refresh labels."""
    d = _make_data()
    X, y = d["X_full"][:800], d["y_full"][:800]

    w = ChimeraBoostRegressor(random_state=0, n_estimators=10,
                              store_training_data=True).fit(
        X, y, cat_features=CAT, sample_weight=d["w_full"][:800])
    with pytest.raises(ValueError, match="needs sample_weight"):
        w.refresh(X[:5], y[:5])

    u = ChimeraBoostRegressor(random_state=0, n_estimators=10,
                              store_training_data=True).fit(
        X, y, cat_features=CAT)
    with pytest.raises(ValueError, match="cannot take sample_weight"):
        u.refresh(X[:5], y[:5], sample_weight=np.ones(5))

    yb = _binary_labels(y)
    clf = ChimeraBoostClassifier(random_state=0, n_estimators=10,
                                 store_training_data=True).fit(
        X, yb, cat_features=CAT)
    bad_y = np.array(["pos", "nope", "neg"], dtype=object)
    with pytest.raises(ValueError, match="cannot add classes"):
        clf.refresh(X[:3], bad_y)


@pytest.mark.parametrize("cls", [ChimeraBoostRegressor, ChimeraBoostClassifier])
def test_store_param_round_trip(cls):
    """(e) clone and get_params round-trip the new parameter."""
    est = cls(store_training_data=True)
    assert est.get_params()["store_training_data"] is True
    assert clone(est).store_training_data is True
    est.set_params(store_training_data=False)
    assert est.get_params()["store_training_data"] is False
    assert cls().get_params()["store_training_data"] is False


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_default_fit_unaffected(kind):
    """(e) The default fit is unaffected: flag False vs True agree exactly."""
    d = _make_data()
    if kind == "regression":
        cls, y = ChimeraBoostRegressor, d["y_full"]
    else:
        cls, y = ChimeraBoostClassifier, _binary_labels(d["y_full"])
    kw = dict(random_state=0, n_estimators=50)
    a = cls(store_training_data=False, **kw).fit(
        d["X_full"], y, cat_features=CAT, sample_weight=d["w_full"])
    b = cls(store_training_data=True, **kw).fit(
        d["X_full"], y, cat_features=CAT, sample_weight=d["w_full"])
    np.testing.assert_array_equal(a.predict(d["X_held"]), b.predict(d["X_held"]))
    if kind == "binary":
        np.testing.assert_array_equal(a.predict_proba(d["X_held"]),
                                      b.predict_proba(d["X_held"]))
    assert a._training_data_ is None
    assert b._training_data_ is not None
    assert a.n_samples_trained_ == b.n_samples_trained_


def test_refresh_usefulness_smoke():
    """(f) Usefulness smoke (not a gate): refresh lands between 60% and 90%.

    Fit on the first 60% of rows, refresh with the next 30%, and compare
    test RMSE against the 60% model and a full refit on 90%. Refresh is
    expected between the two, but that is not asserted.
    """
    d = _make_data()
    X60, y60 = d["X_full"][:1560], d["y_full"][:1560]
    X30, y30 = d["X_full"][1560:2340], d["y_full"][1560:2340]

    def rmse(est):
        return float(np.sqrt(np.mean((est.predict(d["X_held"]) - d["y_held"]) ** 2)))

    base = ChimeraBoostRegressor(random_state=0, n_estimators=200,
                                 store_training_data=True).fit(
        X60, y60, cat_features=CAT)
    rmse_base = rmse(base)
    base.refresh(X30, y30)
    rmse_ref = rmse(base)
    full = ChimeraBoostRegressor(random_state=0, n_estimators=200).fit(
        d["X_full"][:2340], d["y_full"][:2340], cat_features=CAT)
    rmse_full = rmse(full)
    print(f"(f) RMSE: 60%={rmse_base:.4f} refreshed={rmse_ref:.4f} "
          f"90%={rmse_full:.4f}")
    assert np.isfinite([rmse_base, rmse_ref, rmse_full]).all()
