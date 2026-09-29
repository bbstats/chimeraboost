"""Contracts for the MISSING_PLAN bake-off arms (benchmarks/missing_arms.py)
and the harness wiring that runs them.

What makes the comparison fair: every arm keeps the original columns intact
(NaN included) and appends one NaN-free copy per NaN-bearing column, so all arms
have the same width; NaN-free data passes through untouched; nothing is learned
from test rows; the arms run the exact booster config the baseline runs; and
the novel masked imputer actually learns cross-column structure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

import missing_arms as ma  # noqa: E402
import run_benchmarks as rb  # noqa: E402

FAST = ["ind", "mia", "mean", "missforest", "cbmice", "masked"]
ALL = FAST + (["miceforest"] if ma.miceforest_status()[0] else [])
IMPUTERS = [a for a in ALL if a not in ("ind", "mia", "mean")]


def _planted(n=1500, rate=0.3, seed=0):
    """Column 2 is a near-exact linear function of columns 0 and 1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    X[:, 2] = 2 * X[:, 0] - X[:, 1] + 0.05 * rng.standard_normal(n)
    full = X.copy()
    M = rng.random(n) < rate
    X[M, 2] = np.nan
    X[rng.random(n) < rate, 3] = np.nan
    return X, full, M


@pytest.mark.parametrize("arm", ALL)
def test_no_nan_is_exact_passthrough(arm):
    X = np.random.default_rng(0).standard_normal((200, 3))
    tf = ma.ARMS[arm]()
    assert tf.fit_transform(X, None) is X
    Xte = X[:50].copy()
    assert tf.transform(Xte) is Xte


@pytest.mark.parametrize("arm", ALL)
def test_augment_keeps_originals_and_appends_nan_free_copy(arm):
    X, _, _ = _planted()
    tf = ma.ARMS[arm]()
    out = tf.fit_transform(X[:1000], None)
    te = tf.transform(X[1000:])
    # Same width for every arm: the 4 originals + one copy per NaN column.
    assert out.shape == (1000, 6) and te.shape == (500, 6)
    # Originals untouched, NaN included -- the booster's top bin still sees it.
    assert np.array_equal(out[:, :4], X[:1000], equal_nan=True)
    assert np.array_equal(te[:, :4], X[1000:], equal_nan=True)
    assert not np.isnan(out[:, 4:]).any() and not np.isnan(te[:, 4:]).any()
    # Observed cells are copied verbatim (the indicator arm aside).
    if arm != "ind":
        obs = ~np.isnan(X[:1000, 2:4])
        assert np.array_equal(out[:, 4:][obs], X[:1000, 2:4][obs])


def test_indicator_copy_is_the_mask():
    X, _, _ = _planted()
    out = ma.ARMS["ind"]().fit_transform(X, None)
    assert np.array_equal(out[:, 4:].astype(bool), np.isnan(X[:, 2:4]))


def test_mia_copy_sits_just_below_the_minimum():
    X, _, _ = _planted()
    out = ma.ARMS["mia"]().fit_transform(X, None)
    for j, c in ((2, 4), (3, 5)):
        miss = np.isnan(X[:, j])
        lo = np.nanmin(X[:, j])
        assert (out[miss, c] < lo).all()
        assert (out[miss, c] == np.nextafter(lo, -np.inf)).all()


def test_all_missing_column_gets_no_copy():
    X, _, _ = _planted(n=400)
    X[:, 1] = np.nan
    for arm in ALL:
        out = ma.ARMS[arm]().fit_transform(X, None)
        assert out.shape[1] == 4 + 2, arm      # copies for cols 2 and 3 only


@pytest.mark.parametrize("arm", [a for a in ALL if a != "miceforest"])
def test_test_rows_are_independent(arm):
    """Transforming a test row gives the same answer whatever rows accompany
    it -- nothing is learned from the test set."""
    X, _, _ = _planted()
    tf = ma.ARMS[arm]()
    tf.fit_transform(X[:1000], None)
    a = tf.transform(X[1000:])
    b = tf.transform(X[1000:1100])
    assert np.allclose(a[:100], b, equal_nan=True)


def test_categoricals_pass_through():
    X, _, _ = _planted(n=400)
    Xo = X.astype(object)
    Xo[:, 1] = "k"
    out = ma.ARMS["mean"]().fit_transform(Xo, [1])
    assert (out[:, 1] == "k").all()
    assert out.shape[1] == 4 + 2


@pytest.mark.parametrize("arm", IMPUTERS)
def test_model_imputers_beat_mean_on_planted_relation(arm):
    X, full, M = _planted()
    tr, te = slice(0, 1000), slice(1000, None)
    tf = ma.ARMS[arm]()
    got = tf.fit_transform(X[tr], None)[:, 4]
    got_te = tf.transform(X[te])[:, 4]
    mu = np.nanmean(X[tr, 2])

    def rmse(v, rows):
        m = M[rows]
        return np.sqrt(np.mean((v[m] - full[rows][m, 2]) ** 2))
    base = np.full(1000, mu), np.full(500, mu)
    assert rmse(got, tr) < 0.5 * rmse(base[0], tr)
    # Test rows too -- the miceforest new-data path once returned its random
    # initial fill here while the training fill looked fine.
    assert rmse(got_te, te) < 0.5 * rmse(base[1], te)


def test_masked_chunked_prediction_matches_unchunked():
    X, _, _ = _planted()
    tf = ma.ARMS["masked"]()
    tf.fit_transform(X, None)
    F = X[:, tf.cols_]
    Z = (F - tf.mu_) / tf.sd_
    rows, cols = np.nonzero(np.isnan(F[:, tf.nan_local_]))
    whole = tf._predict_cells(Z, rows, cols, chunk_rows=len(rows))
    parts = tf._predict_cells(Z, rows, cols, chunk_rows=37)
    assert np.array_equal(whole, parts)


def test_masked_pair_cap_respects_cell_budget(monkeypatch):
    monkeypatch.setattr(ma, "MASKED_CELL_BUDGET", 6_000)
    monkeypatch.setattr(ma, "MASKED_MIN_PAIRS", 10)
    seen = {}
    real = ma._lean_chimera

    def spy(*a, **k):
        m = real(*a, **k)
        fit = m.fit

        def fit_spy(Xd, y):
            seen["shape"] = Xd.shape
            return fit(Xd, y)
        m.fit = fit_spy
        return m
    monkeypatch.setattr(ma, "_lean_chimera", spy)
    X, _, _ = _planted()
    ma.ARMS["masked"]().fit_transform(X, None)
    rows, width = seen["shape"]
    assert width == 4 + 2 and rows * width <= 6_000


def test_miceforest_refused_on_new_lightgbm(monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("miceforest")
    monkeypatch.setattr(lgb, "__version__", "4.6.0")
    ok, reason = ma.miceforest_status()
    assert not ok and "lightgbm<4.6" in reason
    X, _, _ = _planted(n=300)
    with pytest.raises(RuntimeError, match="lightgbm<4.6"):
        ma.ARMS["miceforest"]().fit_transform(X, None)


# -- harness wiring ----------------------------------------------------------
def _cfg(**over):
    cfg = dict(lr=None, ordered_boosting=None, depth=6, subsample=1.0,
               colsample=None, mcw=None, cat_combinations=False,
               cat_count_features=False, cat_smoothing=None,
               leaf_estimation_iterations=None, linear_leaves=False,
               linear_lambda=1.0, cross_features=False, selection_rounds=None,
               quantize=False, refit_full=False)
    cfg.update(over)
    return cfg


def test_arms_run_the_baselines_booster_config():
    """On NaN-free data an arm is a pass-through, so under ANY --chimera-*
    config it must reproduce the baseline exactly. Before the fix the arms
    silently ran the function defaults while the baseline ran the CLI config."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(400)
    runners = rb._make_runners(["ChimeraBoost", "CBMeanImp", "CBMaskedImp"],
                               _cfg(depth=3, lr=0.2))
    res = {n: r("regression", X[:300], y[:300], X[300:], y[300:], None, 1)
           for n, r in runners.items()}
    base = res["ChimeraBoost"][0]["primary"]
    for n in ("CBMeanImp", "CBMaskedImp"):
        assert res[n][0]["primary"] == base, n
    default = rb._make_runners(["ChimeraBoost"], _cfg())["ChimeraBoost"](
        "regression", X[:300], y[:300], X[300:], y[300:], None, 1)
    assert default[0]["primary"] != base       # the config really changed


def test_miss_only_selection():
    assert rb._miss_only_key("gr:reg_num/cpu_act@mcar")
    assert rb._miss_only_key("gr:reg_num/cpu_act@mnar")
    assert rb._miss_only_key("hc:colleges")
    assert not rb._miss_only_key("gr:reg_num/cpu_act")
    assert not rb._miss_only_key("gr:reg_num/cpu_act@sus25")
    assert not rb._miss_only_key("hc:kick@time")


def _worker_task(ds, miss_only):
    return (ds, 0, 1.0, 1, [], _cfg(), rb.PATIENCE, rb.ENSEMBLE_N,
            False, False, False, False, False, False, False, miss_only)


def test_miss_only_skips_hc_sets_without_numeric_nan():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((200, 3))
    gappy = clean.copy()
    gappy[rng.random(200) < 0.3, 1] = np.nan
    y = rng.standard_normal(200)
    saved = dict(rb.DATASETS)
    try:
        rb.DATASETS["hc:clean"] = lambda s, r: (clean, y, None, "regression")
        rb.DATASETS["hc:gappy"] = lambda s, r: (gappy, y, None, "regression")
        assert rb._run_seed_task(_worker_task("hc:clean", True))[2]["n_train"] == 0
        assert rb._run_seed_task(_worker_task("hc:gappy", True))[2]["n_train"] > 0
        # Without --miss-only nothing is skipped.
        assert rb._run_seed_task(_worker_task("hc:clean", False))[2]["n_train"] > 0
    finally:
        rb.DATASETS.clear()
        rb.DATASETS.update(saved)


def test_seed_start_runs_fresh_seeds(monkeypatch):
    seen = []

    def fake_task(t):
        seen.append(t[1])
        return t[0], t[1], {"task": "regression", "n_train": 0, "n_total": 1,
                            "n_features": 1, "has_cats": False}, {}
    monkeypatch.setattr(rb, "_run_seed_task", fake_task)
    monkeypatch.setattr(sys, "argv", ["x", "--datasets", "diabetes",
                                      "--seed-start", "1", "--seeds", "3",
                                      "--jobs", "1", "--models", "ChimeraBoost"])
    rb.main()
    assert sorted(seen) == [1, 2, 3]
