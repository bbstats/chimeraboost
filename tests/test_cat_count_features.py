"""Per-category count columns (cat_count_features; on by default in the regressor and classifier).

When on, every categorical column with at least CAT_COUNT_MIN_CARD training
categories gets one companion float column holding each row's category
count, stacked at the end of the numeric block -- invisible to the
cross-feature and linear-leaf machinery. When off, preprocessing is
bit-identical to before.
"""

import pickle

import numpy as np
from sklearn.base import clone

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.preprocessing import CAT_COUNT_MIN_CARD, FeaturePreprocessor


CATS = [0, 1]


def _mixed(n=3000, seed=0):
    """Two string categoricals (cardinality ~600 and ~40) + three numerics."""
    rng = np.random.default_rng(seed)
    hi = rng.integers(0, 600, n)
    lo = rng.integers(0, 40, n)
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = np.array([f"cat_{v}" for v in hi], dtype=object)
    X[:, 1] = np.array([f"g_{v}" for v in lo], dtype=object)
    X[:, 2:5] = rng.standard_normal((n, 3))
    return X


def _targets(X, seed=7):
    """Regression, binary, and 3-class targets sharing one signal."""
    rng = np.random.default_rng(seed)
    signal = (X[:, 2].astype(np.float64) - 0.5 * X[:, 3].astype(np.float64)
              + (X[:, 0] == "cat_0").astype(np.float64))
    yr = signal + 0.1 * rng.standard_normal(len(X))
    yb = (signal + 0.1 * rng.standard_normal(len(X))
          > np.median(signal)).astype(int)
    noisy = signal + 0.1 * rng.standard_normal(len(X))
    y3 = np.digitize(noisy, np.quantile(noisy, [1 / 3, 2 / 3]))
    return yr, yb, y3


def _cardinalities_ok(X):
    assert len(np.unique(X[:, 0])) >= CAT_COUNT_MIN_CARD
    assert len(np.unique(X[:, 1])) < CAT_COUNT_MIN_CARD


# ---- 4a: OFF is inert --------------------------------------------------------


def test_off_matches_old_positional_construction():
    X = _mixed()
    yr, _, _ = _targets(X)
    old = FeaturePreprocessor(64, 1.0, 0)
    off = FeaturePreprocessor(64, 1.0, 0, cat_count_features=False)
    old_binned = old.fit_transform(X, [yr], CATS)
    off_binned = off.fit_transform(X, [yr], CATS)
    assert off.count_features_ == []
    assert off.cat_counts_ == []
    np.testing.assert_array_equal(off_binned, old_binned)
    np.testing.assert_array_equal(off.is_numeric_binned_,
                                  old.is_numeric_binned_)
    np.testing.assert_array_equal(off.feature_map_, old.feature_map_)
    assert off.n_numeric_block_ == old.n_numeric_block_ == 3


def test_default_matches_on():
    X = _mixed()
    yr, yb, _ = _targets(X)
    r0 = ChimeraBoostRegressor(random_state=0).fit(X, yr, cat_features=CATS)
    r1 = ChimeraBoostRegressor(random_state=0, cat_count_features=True).fit(
        X, yr, cat_features=CATS)
    np.testing.assert_array_equal(r0.predict(X), r1.predict(X))
    c0 = ChimeraBoostClassifier(random_state=0).fit(X, yb, cat_features=CATS)
    c1 = ChimeraBoostClassifier(random_state=0, cat_count_features=True).fit(
        X, yb, cat_features=CATS)
    np.testing.assert_array_equal(c0.predict_proba(X), c1.predict_proba(X))


def test_off_differs_from_default_on_high_card():
    X = _mixed()
    yr, _, _ = _targets(X)
    off = ChimeraBoostRegressor(random_state=0, cat_count_features=False).fit(
        X, yr, cat_features=CATS)
    assert off.model_.prep_.count_features_ == []
    default = ChimeraBoostRegressor(random_state=0).fit(
        X, yr, cat_features=CATS)
    assert default.model_.prep_.count_features_ == [0]


def test_default_inert_without_high_card_column():
    n = 3000
    rng = np.random.default_rng(1)
    X = _mixed(n=n)
    X[:, 0] = np.array([f"cat_{v}" for v in rng.integers(0, 200, n)],
                       dtype=object)
    assert len(np.unique(X[:, 0])) < CAT_COUNT_MIN_CARD
    yr, yb, _ = _targets(X)
    r0 = ChimeraBoostRegressor(random_state=0).fit(X, yr, cat_features=CATS)
    r1 = ChimeraBoostRegressor(random_state=0, cat_count_features=False).fit(
        X, yr, cat_features=CATS)
    np.testing.assert_array_equal(r0.predict(X), r1.predict(X))
    c0 = ChimeraBoostClassifier(random_state=0).fit(X, yb, cat_features=CATS)
    c1 = ChimeraBoostClassifier(random_state=0, cat_count_features=False).fit(
        X, yb, cat_features=CATS)
    np.testing.assert_array_equal(c0.predict_proba(X), c1.predict_proba(X))


def test_param_round_trips_through_get_params_and_clone():
    assert (ChimeraBoostRegressor(cat_count_features=True)
            .get_params()["cat_count_features"] is True)
    assert (clone(ChimeraBoostRegressor(cat_count_features=True))
            .get_params()["cat_count_features"] is True)
    assert (ChimeraBoostClassifier(cat_count_features=True)
            .get_params()["cat_count_features"] is True)
    assert (clone(ChimeraBoostClassifier(cat_count_features=True))
            .get_params()["cat_count_features"] is True)


# ---- 4b: ON selects by cardinality -------------------------------------------


def test_on_selects_only_high_cardinality_column():
    X = _mixed()
    _cardinalities_ok(X)
    yr, _, _ = _targets(X)
    prep = FeaturePreprocessor(max_bins=64, random_state=0,
                               cat_count_features=True)
    prep.fit_transform(X, [yr], CATS)
    assert prep.count_features_ == [0]
    assert prep.n_numeric_block_ == 4
    # One target, no combos: [3 numerics | 1 count | 2 TS columns].
    np.testing.assert_array_equal(
        prep.is_numeric_binned_,
        np.array([True, True, True, False, False, False]))
    np.testing.assert_array_equal(prep.feature_map_, [2, 3, 4, 0, 0, 1])
    # Pre-binning position: the count column sits at len(num_features_).
    num, codes = prep._split_columns_fit(X, CATS)
    assert num.shape[1] == 4
    np.testing.assert_array_equal(
        num[:, 3], np.bincount(codes[:, 0])[codes[:, 0]])


# ---- 4c: values --------------------------------------------------------------


def test_count_values_seen_and_unseen():
    X = _mixed()
    yr, _, _ = _targets(X)
    on = FeaturePreprocessor(max_bins=64, random_state=0,
                             cat_count_features=True)
    on.fit_transform(X, [yr], CATS)
    off = FeaturePreprocessor(max_bins=64, random_state=0)
    off.fit_transform(X, [yr], CATS)

    # Training column equals bincount(codes)[codes].
    num, codes = on._split_columns_fit(X, CATS)
    np.testing.assert_array_equal(
        num[:, 3], np.bincount(codes[:, 0])[codes[:, 0]])

    # Transform: fit-time count for seen rows, 0.0 for unseen categories.
    Xt = X[:50].copy()
    Xt[::2, 0] = "never_seen_category_xyz"
    tf_codes = on._codes_for_transform(Xt)[:, 0]
    assert (tf_codes[::2] == -1).all()
    assert (tf_codes[1::2] >= 0).all()
    block = on._count_block(on._codes_for_transform(Xt))
    assert block.shape == (50, 1)
    expected = np.where(tf_codes >= 0,
                        on.cat_counts_[0][np.maximum(tf_codes, 0)], 0.0)
    np.testing.assert_array_equal(block[:, 0], expected)
    assert (block[::2, 0] == 0.0).all()
    assert (block[1::2, 0] > 0.0).all()

    # The count column is an insertion: numeric and TS bins do not move.
    b_on = on.transform(Xt)
    b_off = off.transform(Xt)
    assert b_on.shape[1] == b_off.shape[1] + 1
    np.testing.assert_array_equal(b_on[:, :3], b_off[:, :3])
    np.testing.assert_array_equal(b_on[:, 4:], b_off[:, 3:])


# ---- 4d: cross refit keeps the layout ----------------------------------------


def test_cross_refit_layout_with_counts():
    X = _mixed()
    yr, _, _ = _targets(X)
    base = FeaturePreprocessor(max_bins=64, random_state=0,
                               cat_count_features=True)
    base_binned = base.fit_transform(X, [yr], CATS)
    pairs = [(2, 3, "diff"), (2, 4, "prod")]
    aug, _, crossb = FeaturePreprocessor.from_base_with_cross(base, pairs, X)

    # [True x3, False x1, True x n_cross, False x n_TS].
    np.testing.assert_array_equal(
        aug.is_numeric_binned_,
        np.array([True, True, True, False, True, True, False, False]))
    # Count entry (original index 0) sits before the cross entries.
    np.testing.assert_array_equal(aug.feature_map_, [2, 3, 4, 0, 2, 2, 0, 1])

    # Spliced binned matrix matches the from-scratch fit exactly.
    scratch = FeaturePreprocessor(max_bins=64, random_state=0,
                                  cross_pairs=pairs, cat_count_features=True)
    scratch_binned = scratch.fit_transform(X, [yr], CATS)
    nb = base.n_numeric_block_
    spliced = np.hstack([base_binned[:, :nb], crossb, base_binned[:, nb:]])
    np.testing.assert_array_equal(spliced, scratch_binned)
    np.testing.assert_array_equal(aug.is_numeric_binned_,
                                  scratch.is_numeric_binned_)
    np.testing.assert_array_equal(aug.feature_map_, scratch.feature_map_)
    Xt = _mixed(n=200, seed=1)
    Xt[0, 0] = "unseen_cat"
    np.testing.assert_array_equal(aug.transform(Xt), scratch.transform(Xt))


def test_booster_with_cross_race_and_counts():
    X = _mixed()
    yr, _, _ = _targets(X)
    m = ChimeraBoostRegressor(cat_count_features=True, random_state=0)
    m.fit(X, yr, cat_features=CATS)
    assert m.cross_features_selected_ is not None  # the race engaged
    assert np.isfinite(m.predict(X)).all()
    assert len(m.feature_importances_) == m.n_features_in_ == 5


# ---- 4e: end to end ----------------------------------------------------------


def test_end_to_end_all_heads_with_pickle():
    X = _mixed(n=1200, seed=3)
    yr, yb, y3 = _targets(X, seed=11)
    reg = ChimeraBoostRegressor(cat_count_features=True, random_state=0)
    reg.fit(X, yr, cat_features=CATS)
    assert np.isfinite(reg.predict(X)).all()
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(reg)).predict(X),
                                  reg.predict(X))

    clf = ChimeraBoostClassifier(cat_count_features=True, random_state=0)
    clf.fit(X, yb, cat_features=CATS)
    assert np.isfinite(clf.predict_proba(X)).all()
    rt = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(rt.predict(X), clf.predict(X))
    np.testing.assert_array_equal(rt.predict_proba(X), clf.predict_proba(X))

    mc = ChimeraBoostClassifier(cat_count_features=True, random_state=0)
    mc.fit(X, y3, cat_features=CATS)
    assert mc.n_classes_ == 3
    assert np.isfinite(mc.predict_proba(X)).all()
    rt = pickle.loads(pickle.dumps(mc))
    np.testing.assert_array_equal(rt.predict(X), mc.predict(X))
    np.testing.assert_array_equal(rt.predict_proba(X), mc.predict_proba(X))


def test_bagged_fit_runs():
    X = _mixed(n=1200, seed=5)
    yr, _, _ = _targets(X, seed=13)
    m = ChimeraBoostRegressor(cat_count_features=True, random_state=0,
                              n_ensembles=3, ensemble_n_jobs=1)
    m.fit(X, yr, cat_features=CATS)
    assert len(m.estimators_) == 3
    assert np.isfinite(m.predict(X)).all()


# ---- 4f: boundary ------------------------------------------------------------


def test_cardinality_boundary_256_selected_255_not():
    n = 1200
    rng = np.random.default_rng(9)
    a = np.array([f"a_{i % 256}" for i in range(n)], dtype=object)
    b = np.array([f"b_{i % 255}" for i in range(n)], dtype=object)
    assert len(np.unique(a)) == 256
    assert len(np.unique(b)) == 255
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = a
    X[:, 1] = b
    X[:, 2] = rng.standard_normal(n)
    prep = FeaturePreprocessor(cat_count_features=True)
    prep.fit_transform(X, [rng.standard_normal(n)], [0, 1])
    assert prep.count_features_ == [0]


# ---- f5 follow-up: replay pinning + sample_weight ----------------------------


def test_replay_refit_adopts_donor_counts(monkeypatch):
    """The replay refit adopts the donor's count space, not fresh counts.

    The refit trains on ALL rows while the donor saw only the early-stopping
    ~80%: recounting there would scale every count ~1.25x and push each row
    into a higher bin than the one the replayed split was chosen for. The
    refit must reproduce the donor's per-category counts (0.0 for categories
    only the full data has) and bin identically through the adopted binner.
    """
    X = _mixed()
    yr, _, _ = _targets(X)
    captured = {}
    fit_inputs = {}
    orig_replay = GradientBoosting._prep_or_replay_matrices
    orig_fit_transform = FeaturePreprocessor.fit_transform
    orig_from_base = FeaturePreprocessor.from_base_with_cross

    def spy_replay(self, X_, y_, cat_features, eval_set, prep_cache, w):
        if self.replay_donor is not None:
            captured["donor_prep"] = self.replay_donor[1]
        return orig_replay(self, X_, y_, cat_features, eval_set,
                           prep_cache, w)

    def spy_fit_transform(self, X_, encode_targets, cat_features,
                          sample_weight=None, binner=None, cat_ctx=None):
        fit_inputs[id(self)] = np.array(X_, copy=True)
        return orig_fit_transform(self, X_, encode_targets, cat_features,
                                  sample_weight=sample_weight, binner=binner,
                                  cat_ctx=cat_ctx)

    def spy_from_base(cls, base, cross_pairs, X_, sample_weight=None,
                      cat_ctx=None):
        prep, cross_binner, crossb = orig_from_base(
            base, cross_pairs, X_, sample_weight, cat_ctx)
        fit_inputs[id(prep)] = np.array(X_, copy=True)
        return prep, cross_binner, crossb

    monkeypatch.setattr(GradientBoosting, "_prep_or_replay_matrices",
                        spy_replay)
    monkeypatch.setattr(FeaturePreprocessor, "fit_transform",
                        spy_fit_transform)
    # The winner may be the cross-augmented fit, whose prep never sees
    # fit_transform -- its rows arrive through from_base_with_cross.
    monkeypatch.setattr(FeaturePreprocessor, "from_base_with_cross",
                        classmethod(spy_from_base))

    m = ChimeraBoostRegressor(cat_count_features=True, refit_full="replay",
                              random_state=0)
    m.fit(X, yr, cat_features=CATS)

    assert "donor_prep" in captured  # the replay path engaged
    donor, refit = captured["donor_prep"], m.model_.prep_
    assert refit.count_features_ == donor.count_features_ == [0]

    # Same category -> same count; full-data-only categories read 0.0.
    dj, rj = donor.cat_features_.index(0), refit.cat_features_.index(0)
    dk, rk = donor.count_features_.index(0), refit.count_features_.index(0)
    dcounts, rcounts = donor.cat_counts_[dk], refit.cat_counts_[rk]
    dmap, rmap = donor.cat_maps_[dj], refit.cat_maps_[rj]
    new_cats = 0
    for cat, rcode in rmap.items():
        if cat in dmap:
            assert rcounts[rcode] == dcounts[dmap[cat]]
        else:
            assert rcounts[rcode] == 0.0
            new_cats += 1
    assert new_cats > 0

    # Same rows through the same borders: identical values bin identically
    # (the reviewer's 6.67 vs 8.31 mean-bin gap is the failure this pins).
    donor_rows = fit_inputs[id(donor)]
    pos = len(refit.num_features_) + rk
    assert pos == len(donor.num_features_) + dk
    refit_bins = refit.transform(donor_rows)[:, pos]
    donor_bins = donor.transform(donor_rows)[:, pos]
    np.testing.assert_array_equal(refit_bins, donor_bins)
    assert abs(float(refit_bins.mean()) - float(donor_bins.mean())) <= 0.05


def test_counts_respect_sample_weight():
    X = _mixed()
    yr, _, _ = _targets(X)
    w = np.ones(len(X))
    zero_rows = np.flatnonzero(X[:, 0] == "cat_0")[::2]  # half of one category
    assert len(zero_rows) > 0
    w[zero_rows] = 0.0

    prep = FeaturePreprocessor(max_bins=64, random_state=0,
                               cat_count_features=True)
    prep.fit_transform(X, [yr], CATS, sample_weight=w)
    assert prep.count_features_ == [0]
    j = prep.cat_features_.index(0)
    counts = prep.cat_counts_[0].copy()
    rmap = dict(prep.cat_maps_[j])

    # The zero-weighted category reads its weight total ...
    assert counts[rmap["cat_0"]] == w[X[:, 0] == "cat_0"].sum()
    # ... and no other category moved (still its plain row count).
    _, codes = prep._split_columns_fit(X, CATS)
    plain = np.bincount(codes[:, j], minlength=len(rmap))
    for cat, code in rmap.items():
        if cat != "cat_0":
            assert counts[code] == plain[code]

    # None weights are the exact integer counts.
    ref = FeaturePreprocessor(max_bins=64, random_state=0,
                              cat_count_features=True)
    ref.fit_transform(X, [yr], CATS)
    _, ref_codes = ref._split_columns_fit(X, CATS)
    rj = ref.cat_features_.index(0)
    np.testing.assert_array_equal(ref.cat_counts_[0],
                                  np.bincount(ref_codes[:, rj]))


def test_off_inert_with_sample_weight():
    """Flag off + weights: identical to the old construction, as without."""
    X = _mixed()
    yr, _, _ = _targets(X)
    w = np.random.default_rng(3).random(len(X)) + 0.1
    old = FeaturePreprocessor(64, 1.0, 0)
    off = FeaturePreprocessor(64, 1.0, 0, cat_count_features=False)
    old_binned = old.fit_transform(X, [yr], CATS, sample_weight=w)
    off_binned = off.fit_transform(X, [yr], CATS, sample_weight=w)
    np.testing.assert_array_equal(off_binned, old_binned)
    np.testing.assert_array_equal(off.transform(X), old.transform(X))
