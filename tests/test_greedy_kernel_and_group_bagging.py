"""Regression tests for the two PR #53 follow-ups.

  * The `_greedy_borders` value sweep moved into a numba kernel and the
    unweighted `_feature_borders` path open-codes np.unique -- both promised
    bit-identical to the old pure-Python/numpy versions, pinned here against
    in-test reference transcriptions of the old code.
  * Bag members draw whole groups when ``groups`` is passed, so the
    group-disjoint OOB eval set `_member_oob_eval_indices` builds is non-empty
    by construction. Before, a typical 80% row draw touched essentially every
    group, the OOB set came back empty, and OOB early stopping was silently
    dead for every grouped bagged fit."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.binning import _feature_borders, _greedy_border_fill
from chimeraboost.sklearn_api import (_member_oob_eval_indices,
                                      _member_sample_indices)


# --- greedy-border kernel: bit-identity to the retired Python loop -----------

def _reference_greedy_fill(uniq, mass, heavy, target, budget):
    """The pure-Python sweep `_greedy_border_fill` replaced, verbatim."""
    borders = []
    acc = 0.0
    for i in range(uniq.size):
        if len(borders) >= budget:
            break
        if heavy[i]:
            if acc > 0.0:
                borders.append((uniq[i - 1] + uniq[i]) / 2.0)
                acc = 0.0
            if i < uniq.size - 1 and len(borders) < budget:
                borders.append((uniq[i] + uniq[i + 1]) / 2.0)
        else:
            acc += mass[i]
            if acc >= target and i < uniq.size - 1:
                borders.append((uniq[i] + uniq[i + 1]) / 2.0)
                acc = 0.0
    return np.asarray(borders, dtype=np.float64)


@pytest.mark.parametrize("seed", range(20))
def test_greedy_fill_kernel_matches_python_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 400))
    uniq = np.sort(rng.normal(size=n))
    mass = rng.uniform(0.0, 5.0, size=n)
    mass[rng.random(n) < 0.3] *= 100.0          # some heavy hitters
    heavy = mass >= np.sort(mass)[-max(1, n // 8)]
    target = float(mass[~heavy].sum() / max(1, int((~heavy).sum()) // 4 or 1))
    budget = int(rng.integers(1, 64))
    got = _greedy_border_fill(uniq, mass, heavy, target, budget)
    want = _reference_greedy_fill(uniq, mass, heavy, target, budget)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)


def _reference_feature_borders_unweighted(col, max_bins):
    """The np.unique + searchsorted + np.add.at path this change retired,
    verbatim; the new open-coded sort must be bit-identical to it."""
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    uniq = np.unique(finite)
    if uniq.size <= max_bins:
        return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    borders = np.unique(np.quantile(finite, qs))
    if borders.size == qs.size:
        return borders.astype(np.float64)
    counts = np.zeros(uniq.size)
    np.add.at(counts, np.searchsorted(uniq, finite), 1.0)
    from chimeraboost.binning import _greedy_borders
    return _greedy_borders(uniq, counts, max_bins)


@pytest.mark.parametrize("seed", range(10))
def test_unweighted_borders_bit_identical_to_old_path(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 5000))
    cols = [
        rng.normal(size=n),                                   # dense
        np.where(rng.random(n) < 0.9, 0.0, rng.normal(size=n)),   # dominated
        rng.poisson(1.5, size=n).astype(np.float64),          # few uniques
    ]
    zi = rng.normal(size=n)
    zi[rng.random(n) < 0.5] = 0.0
    cols.append(zi)                                           # zero-inflated
    nan_col = rng.normal(size=n)
    nan_col[rng.random(n) < 0.1] = np.nan
    cols.append(nan_col)
    for col in cols:
        for max_bins in (2, 3, 8, 15, 16, 32, 128):
            got = _feature_borders(col, max_bins)
            want = _reference_feature_borders_unweighted(col, max_bins)
            assert np.array_equal(got, want), (seed, max_bins)


# --- member sampling: rows without groups, whole groups with ----------------

def test_member_row_draw_byte_identical_to_old_recipe():
    """groups=None draws must not move: same rng construction, same single
    call as the inline code this helper replaced."""
    n = 1000
    for seed in (0, 7, 12345):
        for ms in (0.3, 0.8):
            m = max(1, int(round(ms * n)))
            want = np.random.default_rng(seed).choice(n, size=m, replace=False)
            got = _member_sample_indices(n, ms, seed, None)
            assert np.array_equal(got, want)
        want = np.random.default_rng(seed).integers(0, n, size=n)
        got = _member_sample_indices(n, 1.0, seed, None)
        assert np.array_equal(got, want)


@pytest.mark.parametrize("dtype", ["int", "str"])
@pytest.mark.parametrize("ms", [0.2, 0.5, 0.8])
def test_group_draw_takes_whole_groups_and_holds_some_out(dtype, ms):
    rng = np.random.default_rng(3)
    sizes = rng.integers(1, 30, size=25)
    groups = np.repeat(np.arange(sizes.size), sizes)
    if dtype == "str":
        groups = np.array([f"g{g:03d}" for g in groups])
    n = groups.size
    for seed in range(10):
        idx = _member_sample_indices(n, ms, seed, groups)
        drawn_groups = np.unique(groups[idx])
        # Whole groups: every row of a drawn group is in the sample.
        in_drawn = np.isin(groups, drawn_groups)
        assert np.array_equal(np.sort(idx), np.where(in_drawn)[0])
        # And at least one group is held out, so the OOB set is non-empty
        # and group-disjoint.
        oob = _member_oob_eval_indices(idx, n, groups)
        assert len(oob) > 0
        assert not np.isin(groups[oob], drawn_groups).any()


def test_group_cluster_bootstrap_at_full_size():
    """ms=1.0 with groups: n_groups draws with replacement, a group drawn k
    times contributes each of its rows exactly k times."""
    groups = np.repeat(np.arange(12), 5)
    n = groups.size
    idx = _member_sample_indices(n, 1.0, 42, groups)
    drawn = np.random.default_rng(42).integers(0, 12, size=12)
    mult = np.bincount(drawn, minlength=12)
    got_counts = np.bincount(groups[idx], minlength=12)
    assert np.array_equal(got_counts, mult * 5)
    # Every copy of a drawn group is complete (each row appears k times).
    row_counts = np.bincount(idx, minlength=n)
    assert np.array_equal(row_counts, mult[groups])


def test_single_group_falls_back_to_row_draw():
    """One group has nothing to hold out; the draw must equal the plain row
    draw so tiny grouped fits keep working exactly as before."""
    n = 100
    groups = np.zeros(n, dtype=int)
    for ms in (0.5, 1.0):
        got = _member_sample_indices(n, ms, 5, groups)
        want = _member_sample_indices(n, ms, 5, None)
        assert np.array_equal(got, want)


def test_tiny_group_count_still_leaves_one_out():
    """round(0.8 * 2) == 2 would draw both groups; the n_groups-1 cap keeps
    one out so the OOB set survives even at two groups."""
    groups = np.repeat([0, 1], 20)
    idx = _member_sample_indices(40, 0.8, 0, groups)
    assert np.unique(groups[idx]).size == 1


# --- end to end: grouped bagged fits get a live OOB eval set ----------------

def _grouped_data(seed=0, n_groups=20, group_size=12):
    """Group random-effects data: rows within a group share an offset."""
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    groups = np.repeat(np.arange(n_groups), group_size)
    X = rng.normal(size=(n, 4))
    effect = rng.normal(size=n_groups)[groups]
    y = X[:, 0] + effect + 0.3 * rng.normal(size=n)
    return X, y, groups


def test_grouped_bagged_members_get_nonempty_oob_evals(monkeypatch):
    """The point of the change: with groups, members no longer fall back to
    the auto-split -- the OOB set is non-empty for every member."""
    import chimeraboost.sklearn_api as api
    returned_sizes = []
    real = api._member_oob_eval_indices

    def spy(idx, n, groups):
        out = real(idx, n, groups)
        returned_sizes.append(len(out))
        return out

    monkeypatch.setattr(api, "_member_oob_eval_indices", spy)
    X, y, groups = _grouped_data()
    reg = ChimeraBoostRegressor(n_estimators=20, depth=3, n_ensembles=3,
                                ensemble_n_jobs=1, random_state=0)
    reg.fit(X, y, groups=groups)
    assert len(returned_sizes) == 3
    assert all(s > 0 for s in returned_sizes)
    assert np.all(np.isfinite(reg.predict(X[:5])))


def test_grouped_bagged_classifier_fits_and_predicts():
    X, y, groups = _grouped_data(seed=1)
    yc = (y > np.median(y)).astype(int)
    clf = ChimeraBoostClassifier(n_estimators=20, depth=3, n_ensembles=3,
                                 ensemble_n_jobs=1, random_state=0)
    clf.fit(X, yc, groups=groups)
    proba = clf.predict_proba(X[:7])
    assert proba.shape == (7, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_grouped_bagged_rare_class_guard_still_works():
    """A group draw can miss a class entirely (the rare class lives in one
    group); the donor patch must still rescue the member."""
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(6), 10)
    X = rng.normal(size=(n, 3))
    y = np.zeros(n, dtype=int)
    y[groups == 0] = 1          # the rare class is exactly one group
    clf = ChimeraBoostClassifier(n_estimators=5, depth=2, n_ensembles=5,
                                 max_samples=0.4, ensemble_n_jobs=1,
                                 random_state=0)
    clf.fit(X, y, groups=groups)
    proba = clf.predict_proba(X[:1])
    assert proba.shape == (1, 2)


def test_grouped_bagged_with_sample_weight():
    X, y, groups = _grouped_data(seed=2)
    sw = np.random.default_rng(2).uniform(0.5, 2.0, size=len(y))
    reg = ChimeraBoostRegressor(n_estimators=15, depth=3, n_ensembles=3,
                                ensemble_n_jobs=1, random_state=0)
    reg.fit(X, y, groups=groups, sample_weight=sw)
    assert np.all(np.isfinite(reg.predict(X[:5])))
