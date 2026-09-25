"""Stored training rows round-trip (issue #131 slice 1a2).

A fitted booster's rows are kept as binner bins (plain numerics), raw
floats (cross parents) and int codes (categoricals); ``rebuild_X`` turns
them back into a raw-equivalent array the existing replay refit consumes
unchanged. Every comparison here is bit for bit.
"""

import inspect
import pickle

import numpy as np
import pytest

from chimeraboost import ChimeraBoostRegressor
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
