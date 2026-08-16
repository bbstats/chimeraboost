"""Numeric cross features (``cross_features=True``): validation-selected
difference/product columns for top numeric feature pairs."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting, MulticlassBoosting
from chimeraboost.sklearn_api import FORCED_CROSS_TOP_M


def _interaction_reg(n=4000, seed=0):
    """Regression whose signal is a comparison + a product -- exactly what an
    oblivious tree staircases and a cross column captures in one split."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (3.0 * (X[:, 0] > X[:, 1]) + X[:, 2] * X[:, 3]
         + 0.1 * rng.standard_normal(n))
    return X, y


def _interaction_clf(n=6000, seed=0):
    """XOR of a comparison and a product sign: linear leaves can't express it,
    cross columns crack both parts with one split each."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    z = (X[:, 0] > X[:, 1]) != (X[:, 2] * X[:, 3] > 0)
    p = np.where(z, 0.9, 0.1)
    y = (rng.random(n) < p).astype(int)
    return X, y


def _interaction_mc(n=6000, seed=0):
    """3-class target whose class boundaries are a comparison and a product
    sign -- the same staircase-vs-one-split geometry, per softmax margin."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    a = X[:, 0] > X[:, 1]
    b = X[:, 2] * X[:, 3] > 0
    y = np.where(a & b, 0, np.where(a | b, 1, 2))
    flip = rng.random(n) < 0.05
    y[flip] = rng.integers(0, 3, flip.sum())
    return X, y


# ---- booster-level cross_pairs -------------------------------------------

def test_booster_cross_pairs_transform_roundtrip():
    X, y = _interaction_reg()
    pairs = [(0, 1, "diff"), (2, 3, "prod")]
    b = GradientBoosting(n_estimators=50, random_state=0, cross_pairs=pairs)
    b.fit(X, y)
    # transform reproduces the cross columns on new data of ORIGINAL width;
    # binned width = 5 numerics + 2 crosses.
    Xb = b.prep_.transform(X[:10])
    assert Xb.shape == (10, 7)
    assert b.prep_.is_numeric_binned_.shape == (7,)
    assert b.prep_.is_numeric_binned_.all()
    # feature map folds crosses into the lower-indexed parent.
    assert list(b.prep_.feature_map_) == [0, 1, 2, 3, 4, 0, 2]
    # importances stay in the ORIGINAL feature space.
    assert b.feature_importances_.shape == (5,)


def test_booster_cross_pairs_help_interaction_data():
    X, y = _interaction_reg()
    Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]
    plain = GradientBoosting(n_estimators=150, random_state=0)
    plain.fit(Xtr, ytr)
    crossed = GradientBoosting(n_estimators=150, random_state=0,
                               cross_pairs=[(0, 1, "diff"), (2, 3, "prod")])
    crossed.fit(Xtr, ytr)
    rmse = lambda m: np.sqrt(np.mean((yte - m.predict_raw(Xte)) ** 2))
    assert rmse(crossed) < rmse(plain) * 0.9


def test_multiclass_booster_cross_pairs_transform_roundtrip():
    X, y = _interaction_mc(n=2500)
    pairs = [(0, 1, "diff"), (2, 3, "prod")]
    b = MulticlassBoosting(n_estimators=20, random_state=0, cross_pairs=pairs)
    b.fit(X, y)
    # binned width = 5 numerics + 2 crosses; importances in ORIGINAL space.
    Xb = b.prep_.transform(X[:10])
    assert Xb.shape == (10, 7)
    assert list(b.prep_.feature_map_) == [0, 1, 2, 3, 4, 0, 2]
    assert b.feature_importances_.shape == (5,)
    assert b.predict_raw(X[:10]).shape == (10, 3)


def test_cross_block_nan_propagates():
    X, y = _interaction_reg(n=2500)
    X[5, 0] = np.nan
    b = GradientBoosting(n_estimators=20, random_state=0,
                         cross_pairs=[(0, 1, "diff")])
    b.fit(X, y)
    assert np.isfinite(b.predict_raw(X[:10])).all()


# ---- wrapper-level selection ---------------------------------------------

def test_regressor_selects_crosses_on_interaction_data():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is True
    assert m.cross_pairs_
    # predicts on ORIGINAL-width input, no user-side augmentation.
    assert m.predict(X[:20]).shape == (20,)
    base = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                 cross_features=False).fit(X, y)
    Xte, yte = _interaction_reg(seed=1)
    rmse = lambda mm: np.sqrt(np.mean((yte - mm.predict(Xte)) ** 2))
    assert rmse(m) < rmse(base)


def test_classifier_selects_crosses_on_interaction_data():
    X, y = _interaction_clf()
    m = ChimeraBoostClassifier(n_estimators=200, random_state=0,
                               cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is True
    proba = m.predict_proba(X[:20])
    assert proba.shape == (20, 2)


def test_default_is_auto_and_matches_explicit_true():
    X, y = _interaction_reg()
    assert ChimeraBoostRegressor().cross_features is None
    a = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              cross_features=True).fit(X, y)
    assert a.cross_features_selected_ in (True, False)
    np.testing.assert_array_equal(a.predict(X[:50]), b.predict(X[:50]))


def test_explicit_false_disables():
    X, y = _interaction_reg()
    off = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                                cross_features=False).fit(X, y)
    assert off.cross_features_selected_ is None
    default = ChimeraBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
    if default.cross_features_selected_:
        assert not np.array_equal(off.predict(X[:50]), default.predict(X[:50]))


def test_multiclass_default_runs_selection():
    # Noise multiclass above the row floor: selection RUNS (M1) and records a
    # verdict either way; the model stays usable.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2600, 4))
    y = rng.integers(0, 3, 2600)
    m = ChimeraBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
    assert m.cross_features_selected_ in (True, False)
    assert m.predict_proba(X[:5]).shape == (5, 3)


def test_multiclass_selects_crosses_on_interaction_data():
    X, y = _interaction_mc()
    m = ChimeraBoostClassifier(n_estimators=200, random_state=0,
                               cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is True
    assert m.cross_pairs_
    proba = m.predict_proba(X[:20])
    assert proba.shape == (20, 3)
    assert np.all(np.isfinite(proba))


def test_multiclass_default_matches_explicit_true():
    X, y = _interaction_mc()
    a = ChimeraBoostClassifier(n_estimators=100, random_state=0).fit(X, y)
    b = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                               cross_features=True).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X[:50]),
                                  b.predict_proba(X[:50]))


def test_multiclass_explicit_false_disables():
    X, y = _interaction_mc()
    off = ChimeraBoostClassifier(n_estimators=100, random_state=0,
                                 cross_features=False).fit(X, y)
    assert off.cross_features_selected_ is None
    assert off.predict_proba(X[:5]).shape == (5, 3)


def test_selection_can_reject_crosses():
    # Pure-noise target: crosses cannot beat the base on validation reliably;
    # whatever the verdict, the final model must be usable and recorded.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 4))
    y = rng.standard_normal(3000)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ in (True, False)
    if not m.cross_features_selected_:
        assert m.cross_pairs_ is None
    assert m.predict(X[:5]).shape == (5,)


def test_skipped_below_min_samples_and_without_validation():
    X, y = _interaction_reg(n=900)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is None

    X, y = _interaction_reg(n=3000)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features=True,
                              early_stopping=False).fit(X, y)
    assert m.cross_features_selected_ is None


def test_skipped_for_mae_loss():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0, loss="MAE",
                              cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is None


def test_multiclass_small_data_skips():
    # Below CROSS_MIN_SAMPLES no selection exists, even with an explicit True
    # (the binary/regression semantics, now shared by multiclass).
    X, y = _interaction_mc(n=900)
    m = ChimeraBoostClassifier(n_estimators=30, random_state=0,
                               cross_features=True).fit(X, y)
    assert m.cross_features_selected_ is None
    assert m.predict_proba(X[:5]).shape == (5, 3)


def test_crosses_skip_categorical_columns():
    rng = np.random.default_rng(0)
    n = 4000
    Xnum = rng.standard_normal((n, 3))
    cat = rng.integers(0, 4, n).astype(str)
    X = np.column_stack([Xnum.astype(object), cat.astype(object)])
    y = (2.0 * (Xnum[:, 0] > Xnum[:, 1]) + (cat == "2") * 1.5
         + 0.1 * rng.standard_normal(n))
    m = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              cross_features=True)
    m.fit(X, y, cat_features=[3])
    if m.cross_pairs_:
        # A categorical column may appear only as a gdiff group key (j of a
        # group-centered cross), never as a diff/prod arithmetic parent.
        for i, j, op in m.cross_pairs_:
            assert i != 3
            assert j != 3 or op == "gdiff"
    assert m.predict(X[:10]).shape == (10,)


# ---- forced mode (cross_features="always", SELECT_PLAN.md E2) -------------

def test_always_forces_crosses_on_rung1_config():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                              linear_leaves=True, refit_full=False,
                              cross_features="always").fit(X, y)
    assert m.cross_features_selected_ is True
    assert m.cross_pairs_
    base = ChimeraBoostRegressor(n_estimators=200, random_state=0,
                                 linear_leaves=True, refit_full=False,
                                 cross_features=False).fit(X, y)
    Xte, yte = _interaction_reg(seed=1)
    rmse = lambda mm: np.sqrt(np.mean((yte - mm.predict(Xte)) ** 2))
    assert rmse(m) < rmse(base)


def test_always_uses_the_narrow_block():
    # 8 numerics: the raced default would pair the top 6; the forced mode must
    # pair at most FORCED_CROSS_TOP_M (4) -- the width step 1 priced.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 8))
    y = (3.0 * (X[:, 0] > X[:, 1]) + X[:, 2] * X[:, 3]
         + 0.1 * rng.standard_normal(4000))
    m = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              linear_leaves=True,
                              cross_features="always").fit(X, y)
    assert m.cross_features_selected_ is True
    arith = {f for i, j, op in m.cross_pairs_ if op != "gdiff"
             for f in (i, j)}
    assert len(arith) <= FORCED_CROSS_TOP_M
    n_arith = sum(op != "gdiff" for _, _, op in m.cross_pairs_)
    assert n_arith <= FORCED_CROSS_TOP_M * (FORCED_CROSS_TOP_M - 1)


def test_always_keeps_the_linear_leaves_audition():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=100, random_state=0,
                              cross_features="always").fit(X, y)
    assert m.linear_leaves_selected_ in (True, False)
    assert m.cross_features_selected_ is True
    assert m.predict(X[:5]).shape == (5,)


def test_always_inert_where_gates_fail():
    # Below CROSS_MIN_SAMPLES, without a validation split, and for MAE the
    # forced mode must be exactly as inert as True is (B1: gates stay).
    X, y = _interaction_reg(n=900)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features="always").fit(X, y)
    assert m.cross_features_selected_ is None

    X, y = _interaction_reg(n=3000)
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_features="always",
                              early_stopping=False).fit(X, y)
    assert m.cross_features_selected_ is None

    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0, loss="MAE",
                              cross_features="always").fit(X, y)
    assert m.cross_features_selected_ is None


def test_always_rejected_on_classifier():
    X, y = _interaction_clf(n=2500)
    with pytest.raises(ValueError, match="always"):
        ChimeraBoostClassifier(n_estimators=30,
                               cross_features="always").fit(X, y)


def test_cross_features_bad_value_rejected():
    X, y = _interaction_reg(n=2500)
    with pytest.raises(ValueError, match="cross_features"):
        ChimeraBoostRegressor(n_estimators=30,
                              cross_features="sometimes").fit(X, y)


def test_shap_stays_in_original_feature_space():
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_features=True).fit(X, y)
    if m.cross_features_selected_:
        contrib = m.shap_values(X[:16])
        assert contrib.shape == (16, 5)
        recon = contrib.sum(axis=1) + m.expected_value_
        np.testing.assert_allclose(recon, m.predict(X[:16]), rtol=1e-6,
                                   atol=1e-8)


# ---- cross_top_columns: screen the candidate block ------------------------
# benchmarks/CAMPAIGN_PLAN.md F1. The screen ranks candidates by how well each
# explains the base fit's validation residuals and keeps the best k.

def _cat_reg(n=4000, seed=0):
    """The regression above with a categorical column, so the candidate set
    holds gdiff pairs as well as diff/prod ones."""
    X, y = _interaction_reg(n=n, seed=seed)
    Xo = np.empty((n, 6), dtype=object)
    Xo[:, :5] = X
    Xo[:, 5] = np.where(X[:, 4] > 0, "hi", "lo")
    return Xo, y


def test_none_leaves_every_path_untouched():
    # The default must not even change the candidate ORDER, let alone the model.
    X, y = _interaction_reg()
    a = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_top_columns=None).fit(X, y)
    assert a.cross_pairs_ == b.cross_pairs_
    np.testing.assert_array_equal(a.predict(X[:64]), b.predict(X[:64]))


def test_screen_caps_the_block_and_keeps_candidate_order():
    X, y = _interaction_reg()
    full = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(X, y)
    cut = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                                cross_top_columns=4).fit(X, y)
    assert full.cross_features_selected_ and cut.cross_features_selected_
    assert len(full.cross_pairs_) > 4
    assert len(cut.cross_pairs_) == 4
    # A kept pair is a candidate, and the surviving block stays in the
    # candidate order (column layout feeds split tie-breaks).
    assert set(cut.cross_pairs_) <= set(full.cross_pairs_)
    assert cut.cross_pairs_ == [p for p in full.cross_pairs_
                                if p in set(cut.cross_pairs_)]


def test_screen_finds_the_columns_that_carry_the_signal():
    # The signal is (x0 > x1) and x2 * x3; a k=4 screen must keep the diff of
    # the first pair and the product of the second. This is the test that killed
    # the |correlation| statistic: it kept four products and no comparison
    # column at all (CAMPAIGN_PLAN.md I006). Pair order follows base-fit
    # importance, so the diff may be emitted either way round.
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_top_columns=4).fit(X, y)
    kept = {(i, j, op) for i, j, op in m.cross_pairs_}
    assert kept & {(0, 1, "diff"), (1, 0, "diff")}
    assert kept & {(2, 3, "prod"), (3, 2, "prod")}


def test_screen_inert_below_the_row_gate():
    # B1: the applicability gates are untouched, so a sub-gate fit is
    # bit-identical with the knob set.
    X, y = _interaction_reg(n=900)
    a = ChimeraBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_top_columns=4).fit(X, y)
    assert a.cross_features_selected_ is None
    assert b.cross_features_selected_ is None
    np.testing.assert_array_equal(a.predict(X[:64]), b.predict(X[:64]))


def test_k_at_or_above_the_candidate_count_is_inert():
    X, y = _interaction_reg()
    a = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(X, y)
    b = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_top_columns=len(a.cross_pairs_)).fit(X, y)
    assert a.cross_pairs_ == b.cross_pairs_
    np.testing.assert_array_equal(a.predict(X[:64]), b.predict(X[:64]))


def test_screen_ranks_gdiff_candidates_too():
    X, y = _cat_reg()
    full = ChimeraBoostRegressor(n_estimators=80, random_state=0).fit(
        X, y, cat_features=[5])
    cut = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                                cross_top_columns=5).fit(X, y, cat_features=[5])
    assert any(op == "gdiff" for _, _, op in full.cross_pairs_)
    assert len(cut.cross_pairs_) == 5
    assert set(cut.cross_pairs_) <= set(full.cross_pairs_)


def test_screen_trims_the_forced_block():
    # cross_features="always" has no race, so the screen trims the block the
    # importance probe ranked.
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              cross_features="always",
                              cross_top_columns=3).fit(X, y)
    assert m.cross_features_selected_ is True
    assert len(m.cross_pairs_) == 3


def test_screen_on_the_unraced_path():
    # selection_rounds=None takes the post-fit cross branch, a different call
    # site from the audition race.
    X, y = _interaction_reg()
    m = ChimeraBoostRegressor(n_estimators=80, random_state=0,
                              selection_rounds=None,
                              cross_top_columns=4).fit(X, y)
    assert len(m.cross_pairs_) == 4


def test_screen_on_binary_and_multiclass_classifiers():
    X, y = _interaction_clf()
    m = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                               cross_top_columns=4).fit(X, y)
    assert m.cross_features_selected_ is not None
    if m.cross_features_selected_:
        assert len(m.cross_pairs_) == 4

    X, y = _interaction_mc()
    m = ChimeraBoostClassifier(n_estimators=80, random_state=0,
                               cross_top_columns=4).fit(X, y)
    assert m.cross_features_selected_ is not None
    if m.cross_features_selected_:
        assert len(m.cross_pairs_) == 4


def test_screen_survives_nan_and_overflow_in_a_candidate():
    # A NaN parent propagates into diff/prod columns and a huge parent
    # overflows the product; neither may crash or blank out the ranking.
    X, y = _interaction_reg()
    X = X.copy()
    X[::50, 0] = np.nan
    X[:, 4] *= 1e200
    m = ChimeraBoostRegressor(n_estimators=60, random_state=0,
                              cross_top_columns=4).fit(X, y)
    assert m.cross_pairs_ is None or len(m.cross_pairs_) == 4


def test_cross_top_columns_bad_value_rejected():
    X, y = _interaction_reg(n=2500)
    for bad in (0, -1, 2.5, "6"):
        with pytest.raises(ValueError, match="cross_top_columns"):
            ChimeraBoostRegressor(n_estimators=30,
                                  cross_top_columns=bad).fit(X, y)
