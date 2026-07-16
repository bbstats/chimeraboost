"""selection_rounds: capped audition fits for the internal selections
(benchmarks/PARETO_PLAN.md Track 1 step 2, fallback design)."""
import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


def _reg_linear(n=3000, seed=0):
    """Strong linear signal: linear leaves win the selection at any budget."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 4.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _bin_interaction(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 2.0 * X[:, 0] * X[:, 1] + X[:, 2]
    y = (logit + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _fit_segments(**fit_kw):
    """Fit while recording each internal booster fit's round count (fits run
    sequentially; a callback call with iteration 0 starts a new segment)."""
    segments = []

    def spy(iteration, train_loss, val_loss, model):
        if iteration == 0:
            segments.append(0)
        segments[-1] = iteration + 1

    fit_kw.setdefault("callbacks", [spy])
    return segments, fit_kw


def test_auditions_are_capped_and_winner_runs_full():
    X, y = _reg_linear()
    segments, kw = _fit_segments()
    m = ChimeraBoostRegressor(n_estimators=200, selection_rounds=25,
                              cross_features=False, random_state=0)
    m.fit(X, y, **kw)
    # const audition, linear audition, then the winner's full refit.
    assert len(segments) == 3
    assert segments[0] == 25 and segments[1] == 25
    # patience is 50, so a full fit always runs past the 25-round cap
    assert segments[2] > 25
    assert m.linear_leaves_selected_ is not None


def test_matches_full_selection_when_auditions_agree():
    X, y = _reg_linear()
    fast = ChimeraBoostRegressor(selection_rounds=50, cross_features=False,
                                 random_state=0).fit(X, y)
    full = ChimeraBoostRegressor(selection_rounds=None,
                                 cross_features=False,
                                 random_state=0).fit(X, y)
    # The linear signal is decisive at every budget, so both arms pick the
    # linear variant and ship the identical full fit.
    assert bool(fast.linear_leaves_selected_) is True
    assert bool(full.linear_leaves_selected_) is True
    np.testing.assert_array_equal(fast.predict(X), full.predict(X))


def test_binary_cross_selection_still_runs():
    X, y = _bin_interaction()
    m = ChimeraBoostClassifier(selection_rounds=50, random_state=0).fit(X, y)
    assert m.cross_features_selected_ is not None
    if m.cross_features_selected_:
        assert m.cross_pairs_
    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))


def test_binary_base_refit_full_when_cross_loses():
    # Pure-noise target: the augmented model has no interaction signal to win
    # with; whenever it loses, the shipped base model must be a full fit, not
    # the capped audition.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6))
    y = (rng.random(3000) > 0.5).astype(int)
    segments, kw = _fit_segments()
    m = ChimeraBoostClassifier(n_estimators=200, selection_rounds=25,
                               random_state=0)
    m.fit(X, y, **kw)
    if not m.cross_features_selected_:
        # audition (25), cross full, base full refit
        assert segments[0] == 25
        assert len(segments) == 3
        assert segments[-1] > 25


def test_uncapped_auditions_bit_identical_to_full_selection():
    # An audition that never hits the cap early-stops on its own, so it IS the
    # full fit; with the cap at the whole budget the fast path must reproduce
    # the default selection exactly, cross features included.
    Xr, yr = _reg_linear()
    fast_r = ChimeraBoostRegressor(selection_rounds=2000,
                                   random_state=0).fit(Xr, yr)
    full_r = ChimeraBoostRegressor(selection_rounds=None,
                                   random_state=0).fit(Xr, yr)
    np.testing.assert_array_equal(fast_r.predict(Xr), full_r.predict(Xr))
    assert bool(fast_r.cross_features_selected_) \
        == bool(full_r.cross_features_selected_)

    Xb, yb = _bin_interaction()
    fast_b = ChimeraBoostClassifier(selection_rounds=2000,
                                    random_state=0).fit(Xb, yb)
    full_b = ChimeraBoostClassifier(selection_rounds=None,
                                    random_state=0).fit(Xb, yb)
    np.testing.assert_array_equal(fast_b.predict_proba(Xb),
                                  full_b.predict_proba(Xb))


def test_multiclass_is_unaffected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    y = rng.integers(0, 3, size=1500)
    fast = ChimeraBoostClassifier(selection_rounds=50, random_state=0).fit(X, y)
    full = ChimeraBoostClassifier(selection_rounds=None,
                                  random_state=0).fit(X, y)
    np.testing.assert_array_equal(fast.predict_proba(X), full.predict_proba(X))


def test_small_data_is_bit_identical():
    # Below the linear-leaves (1000) and cross (2000) row floors no selection
    # exists, so selection_rounds must change nothing at all.
    X, y = _reg_linear(n=800)
    fast = ChimeraBoostRegressor(selection_rounds=100, random_state=0).fit(X, y)
    full = ChimeraBoostRegressor(selection_rounds=None,
                                 random_state=0).fit(X, y)
    np.testing.assert_array_equal(fast.predict(X), full.predict(X))


@pytest.mark.parametrize("bad", [0, -1, 0.5, "100"])
def test_selection_rounds_validated(bad):
    X, y = _reg_linear(n=200)
    with pytest.raises(ValueError, match="selection_rounds"):
        ChimeraBoostRegressor(selection_rounds=bad).fit(X, y)
