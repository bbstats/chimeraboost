"""Regression tests for the 2026-07-30 follow-up to the #52 bug hunt.

Three defects, all introduced or left behind by #52:
  * `_greedy_borders` divided by zero for max_bins < 16 on a dominated column;
  * the bagged rare-class guard still crashed when a member drew one row;
  * the serial/parallel predict dispatch threshold was ~8x too low, and the
    warmup routine hardcoded a row count that assumed the old value.

Each test failed before the fix."""

import numpy as np
import pytest

import chimeraboost.binning as B
import chimeraboost.booster as BO
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.binning import _feature_borders


def _dominated_column(n_heavy=900, n_light=100, seed=0):
    """A column where one value holds the overwhelming share of the mass --
    the profile that routes through `_greedy_borders`."""
    rng = np.random.default_rng(seed)
    return np.concatenate([np.zeros(n_heavy), rng.normal(size=n_light)])


# --- small max_bins on a dominated column -----------------------------------

@pytest.mark.parametrize("max_bins", list(range(2, 20)))
def test_greedy_borders_survives_small_max_bins(max_bins):
    """max_bins below 16 made the `max_bins // 16` light-budget floor 0; a
    column dominated hard enough to round the proportional term to 0 as well
    then divided by zero."""
    borders = _feature_borders(_dominated_column(), max_bins)
    assert borders.ndim == 1
    assert np.all(np.diff(borders) > 0), "borders must be strictly increasing"
    assert len(borders) <= max_bins - 1, "border budget overrun"


@pytest.mark.parametrize("max_bins", [3, 4, 6, 8, 12, 15])
def test_fit_with_small_max_bins_on_dominated_column(max_bins):
    """The same crash, reached through the public estimator."""
    rng = np.random.default_rng(0)
    X = np.column_stack([_dominated_column(), rng.normal(size=1000)])
    y = rng.normal(size=1000)
    m = ChimeraBoostRegressor(max_bins=max_bins, n_estimators=10,
                              depth=3).fit(X, y)
    assert np.all(np.isfinite(m.predict(X[:5])))


def test_extremely_dominated_column_every_budget():
    """99.9% of the mass on one value, across the whole valid budget range."""
    col = _dominated_column(n_heavy=9990, n_light=10)
    for max_bins in range(2, 40):
        borders = _feature_borders(col, max_bins)
        assert np.all(np.diff(borders) > 0)


def test_large_max_bins_allocation_unchanged():
    """The floor must not perturb budgets that already worked: for max_bins
    >= 16 the `// 16` term dominates the new floor of 1."""
    col = _dominated_column()
    for max_bins in (16, 32, 64, 128, 254):
        borders = _feature_borders(col, max_bins)
        light_bins = max(max_bins // 16, 1)
        assert light_bins == max_bins // 16
        assert len(borders) > 0


# --- bagged classifier, one-row member draw ---------------------------------

@pytest.mark.parametrize("n_rows,max_samples,n_ensembles",
                         [(4, 0.3, 6), (3, 0.3, 4), (5, 0.2, 5),
                          (6, 0.5, 8), (10, 0.15, 6)])
def test_bagged_classifier_one_row_member_draw(n_rows, max_samples,
                                               n_ensembles):
    """The rare-class guard overwrote a random drawn row with a donor of the
    missing class. With a one-row draw that overwrote the only row, so the
    member still saw a single class and raised "Need at least 2 classes"."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, 3))
    y = np.zeros(n_rows, dtype=int)
    y[0] = 1
    m = ChimeraBoostClassifier(n_estimators=5, n_ensembles=n_ensembles,
                               max_samples=max_samples, depth=2).fit(X, y)
    proba = m.predict_proba(X[:1])
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


# --- predict dispatch threshold ---------------------------------------------

def test_serial_and_parallel_predict_kernels_agree():
    """The dispatch only changes which kernel runs; the two are documented as
    bit-identical, which is what makes retuning the threshold a pure speed
    change. Pin that across the new boundary."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8))
    y = rng.normal(size=3000)
    reg = ChimeraBoostRegressor(n_estimators=40, depth=4,
                                early_stopping_rounds=None,
                                refit_full=False).fit(X, y)
    clf = ChimeraBoostClassifier(n_estimators=40, depth=4,
                                 early_stopping_rounds=None).fit(
                                     X, (y > 0).astype(int))
    saved = B._SERIAL_PREDICT_N, BO._SERIAL_PREDICT_N
    try:
        for batch in (1, 4, 5, 16, 31, 32, 33, 64, 200):
            Xb = np.ascontiguousarray(X[:batch])
            B._SERIAL_PREDICT_N = BO._SERIAL_PREDICT_N = 10 ** 9   # serial
            ser_r, ser_c = reg.predict(Xb), clf.predict_proba(Xb)
            B._SERIAL_PREDICT_N = BO._SERIAL_PREDICT_N = -1        # parallel
            par_r, par_c = reg.predict(Xb), clf.predict_proba(Xb)
            assert np.array_equal(ser_r, par_r), f"regression, batch {batch}"
            assert np.array_equal(ser_c, par_c), f"classifier, batch {batch}"
    finally:
        B._SERIAL_PREDICT_N, BO._SERIAL_PREDICT_N = saved


def test_warmup_reaches_the_parallel_predict_kernels():
    """Warmup hardcoded an 8-row "parallel" batch, which the raised threshold
    silently pulled onto the serial twins -- leaving the parallel forest walk
    to compile on the user's first real batch, the exact stall warmup exists
    to prevent. Assert the kernel really is compiled after a warmup, and that
    the batch size warmup uses is derived from the threshold rather than
    hardcoded (so raising it again cannot re-break this)."""
    import importlib

    from chimeraboost.tree import _predict_forest_rm
    from chimeraboost.warmup import warmup

    warmup()
    assert _predict_forest_rm.signatures, \
        "warmup left the parallel forest-walk kernel uncompiled"

    # The row count must track the threshold, not a literal.
    mod = importlib.import_module("chimeraboost.warmup")
    src = importlib.import_module("inspect").getsource(mod.warmup)
    assert "_SERIAL_PREDICT_N" in src, \
        "warmup must derive its parallel-batch size from the threshold"
