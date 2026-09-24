"""The quantile decision tier (Q-B1): suite selection and the new arms.

Deterministic fixtures only -- no network, no downloads.
"""
import os
import sys

import numpy as np
import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import quantile_suite as qs  # noqa: E402
import run_benchmarks as rb  # noqa: E402
from chimeraboost import ChimeraBoostRegressor  # noqa: E402


def _stub_registry(monkeypatch, keys, tasks):
    """Replace the harness registry with fake keys; registration becomes a
    no-op and the task comes from ``tasks`` (base key -> task)."""
    monkeypatch.setattr(rb, "DATASETS", {k: None for k in keys})
    monkeypatch.setattr(rb, "_add_grinsztajn_datasets", lambda: None)
    monkeypatch.setattr(rb, "_add_highcard_datasets", lambda: None)
    monkeypatch.setattr(rb, "_add_variant_datasets", lambda keys: None)
    monkeypatch.setattr(rb, "_task_of",
                        lambda k: tasks[k.split(rb.VARIANT_SEP)[0]])


def test_decide_selection_returns_exactly_the_regression_keys(monkeypatch):
    keys = ["gr:reg_num/houses", "gr:clf_num/electricity",
            "hc:house_prices_nominal", "hc:kick",
            "hc:house_prices_nominal@time", "gr:reg_num/houses@sus25",
            "hc:kick@sus50"]
    tasks = {"gr:reg_num/houses": "regression",
             "gr:clf_num/electricity": "binary",
             "hc:house_prices_nominal": "regression",
             "hc:kick": "binary"}
    _stub_registry(monkeypatch, keys, tasks)
    assert qs.select_datasets(decide=True) == sorted([
        "gr:reg_num/houses", "hc:house_prices_nominal",
        "hc:house_prices_nominal@time", "gr:reg_num/houses@sus25"])


def test_default_selection_is_grinsztajn_regression_only(monkeypatch):
    keys = ["gr:reg_num/houses", "gr:clf_num/electricity",
            "hc:house_prices_nominal"]
    tasks = {"gr:reg_num/houses": "regression",
             "gr:clf_num/electricity": "binary",
             "hc:house_prices_nominal": "regression"}
    _stub_registry(monkeypatch, keys, tasks)
    assert qs.select_datasets(decide=False) == ["gr:reg_num/houses"]


def test_rigid_shift_offsets_match_validation_residuals():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 + rng.normal(size=n)
    split = (X[:40], X[40:50], y[:40], y[40:50])
    Xte = X[50:]
    taus = np.array([0.1, 0.5, 0.9])
    Q, _, _, _ = qs._fit_rigid_shift(split, Xte, None, 1, taus)
    # The reference fit: same point model, same rows.
    m = ChimeraBoostRegressor(n_estimators=rb.MAX_ITERS,
                              early_stopping_rounds=rb.PATIENCE,
                              thread_count=1, random_state=0)
    m.fit(X[:40], y[:40], eval_set=(X[40:50], y[40:50]))
    pred_val = np.asarray(m.predict(X[40:50]), dtype=np.float64).ravel()
    pred_test = np.asarray(m.predict(Xte), dtype=np.float64).ravel()
    offsets = np.quantile(y[40:50] - pred_val, taus)
    assert Q.shape == (len(Xte), len(taus))
    np.testing.assert_allclose(Q, pred_test[:, None] + offsets[None, :],
                               rtol=1e-9)
    assert np.all(np.diff(Q, axis=1) >= 0)  # one width for every row


def test_ngboost_arm_returns_ordered_grid():
    pytest.importorskip("ngboost")
    rng = np.random.default_rng(1)
    n = 80
    num = rng.normal(size=(n, 2))
    X = np.empty((n, 3), dtype=object)
    X[:, :2] = num
    X[:, 2] = rng.choice(["a", "b", "c"], size=n)
    y = num[:, 0] * 2.0 + rng.normal(size=n)
    split = (X[:50], X[50:65], y[:50], y[50:65])
    Xte = X[65:]
    taus = np.array([0.1, 0.5, 0.9])
    Q, _, _, _ = qs._fit_ngboost(split, Xte, [2], None, taus)
    assert Q.shape == (len(Xte), len(taus))
    assert np.all(np.diff(Q, axis=1) >= 0)


def test_ngboost_arm_predicts_at_best_round():
    ngboost = pytest.importorskip("ngboost")
    from ngboost.learners import default_tree_learner
    from sklearn.base import clone
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(size=(n, 4))
    y = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) + rng.normal(size=n)
    split = (X[:120], X[120:160], y[:120], y[120:160])
    Xte = X[160:]
    taus = np.array([0.1, 0.5, 0.9])
    Q, _, _, best = qs._fit_ngboost(split, Xte, None, None, taus)
    # The reference fit: the same booster on the same rows.
    base = clone(default_tree_learner).set_params(random_state=0)
    m = ngboost.NGBRegressor(Dist=ngboost.distns.Normal,
                             n_estimators=rb.MAX_ITERS,
                             early_stopping_rounds=rb.PATIENCE,
                             Base=base, random_state=0, verbose=False)
    m.fit(np.asarray(split[0], dtype=np.float64), split[2],
          X_val=np.asarray(split[1], dtype=np.float64), Y_val=split[3])
    assert m.best_val_loss_itr is not None
    assert len(m.base_models) < rb.MAX_ITERS  # early stopping triggered
    assert int(m.best_val_loss_itr) < len(m.base_models) - 1
    assert best == int(m.best_val_loss_itr)

    def _grid(dist):
        return np.column_stack(
            [np.asarray(dist.ppf(float(tau)), dtype=np.float64)
             for tau in taus])

    ref = _grid(m.pred_dist(np.asarray(Xte, dtype=np.float64),
                            max_iter=int(m.best_val_loss_itr) + 1))
    np.testing.assert_allclose(Q, ref, rtol=1e-9)
    all_trees = _grid(m.pred_dist(np.asarray(Xte, dtype=np.float64)))
    assert not np.allclose(Q, all_trees)


def test_explicit_datasets_register_needed_suites(monkeypatch):
    calls = []
    keys = ["hc:Moneyball", "hc:Moneyball@time"]
    monkeypatch.setattr(rb, "DATASETS", {k: None for k in keys})
    monkeypatch.setattr(rb, "_add_grinsztajn_datasets",
                        lambda: calls.append("gr"))
    monkeypatch.setattr(rb, "_add_highcard_datasets",
                        lambda: calls.append("hc"))
    monkeypatch.setattr(rb, "_add_variant_datasets",
                        lambda ks: calls.append("variants"))
    rc = qs.main(["--datasets", "hc:Moneyball", "hc:Moneyball@time",
                  "--list-datasets", "--seeds", "1",
                  "--models", "RigidShift"])
    assert rc == 0
    assert "hc" in calls
    assert "variants" in calls
    assert "gr" not in calls


def test_worker_registers_suites_for_its_key(monkeypatch):
    calls = []
    keys = ["hc:Moneyball@time"]
    monkeypatch.setattr(rb, "DATASETS", {k: None for k in keys})
    monkeypatch.setattr(rb, "_add_grinsztajn_datasets",
                        lambda: calls.append("gr"))
    monkeypatch.setattr(rb, "_add_highcard_datasets",
                        lambda: calls.append("hc"))
    monkeypatch.setattr(rb, "_add_variant_datasets",
                        lambda ks: calls.append("variants"))
    monkeypatch.setattr(qs, "run_one",
                        lambda *a: ({"task": "quantile"}, {}))
    ds, seed, meta, out = qs._run_seed_task_quantile(
        ("hc:Moneyball@time", 0, np.array([0.5]), 1, ["RigidShift"], False))
    assert (ds, seed) == ("hc:Moneyball@time", 0)
    assert meta == {"task": "quantile"}
    assert "hc" in calls
    assert "variants" in calls
    assert "gr" not in calls


def test_unknown_dataset_key_exits_before_running(monkeypatch):
    keys = ["hc:Moneyball", "hc:Moneyball@time"]
    tasks = {"hc:Moneyball": "regression"}
    _stub_registry(monkeypatch, keys, tasks)
    with pytest.raises(SystemExit):
        qs.main(["--datasets", "hc:Moneyball", "hc:bogus",
                 "--seeds", "1", "--models", "RigidShift"])


# ---------------------------------------------------------------------------
# Q0 probe battery: the four opt-in arms in `qs.PROBES` (no network).
# ---------------------------------------------------------------------------

def _q0_split(n=800, seed=0):
    """Small heteroscedastic regression split on the suite's data path."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (2.0 * X[:, 0] + np.sin(2.0 * X[:, 2])
         + np.exp(0.6 * X[:, 1]) * rng.standard_normal(n))
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25,
                                        random_state=seed)
    split = rb._val_split(Xtr, ytr, "regression", 0)
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    return split, Xte, taus


def _q0_capture_models(monkeypatch):
    """Record every head model the arms fit, for reading fitted attributes."""
    seen = []
    real = qs.ChimeraBoostQuantileRegressor

    class _Spy(real):
        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)
            seen.append(self)
            return self

    monkeypatch.setattr(qs, "ChimeraBoostQuantileRegressor", _Spy)
    return seen


def _q0_pinned_head(split, Xte, taus):
    """The pre-Q5 head the I057/I058 probes were built on: depth 4,
    raw grid. Returns (Q, best_iteration)."""
    m = qs._fit_head_model(split, None, 1, taus, rb.MAX_ITERS,
                           depth=4, conformalize=False)
    return m.predict(Xte), m.best_iteration_


def test_q0_probes_are_opt_in(monkeypatch):
    """Both suites default to the seven field arms; every probe parses."""
    import quantile_synth as qsyn

    assert len(qs.ARMS) == 7
    assert len(qs.PROBES) == 14
    _stub_registry(monkeypatch, ["gr:reg_num/houses"],
                   {"gr:reg_num/houses": "regression"})

    seen = {}
    monkeypatch.setattr(qs, "_print_dataset_list",
                        lambda n, a, t: seen.setdefault("suite", a.models))
    assert qs.main(["--list-datasets", "--seeds", "1"]) == 0
    assert seen["suite"] == list(qs.ARMS)
    assert qs.main(["--list-datasets", "--seeds", "1",
                    "--models"] + list(qs.PROBES)) == 0

    monkeypatch.setattr(qsyn, "_print_dataset_list",
                        lambda k, a, t: seen.setdefault("synth", a.models))
    assert qsyn.main(["--list-datasets", "--seeds", "1"]) == 0
    assert seen["synth"] == list(qs.ARMS)
    assert qsyn.main(["--list-datasets", "--seeds", "1",
                      "--models"] + list(qs.PROBES)) == 0


def test_q0_uncapped_matches_head_when_head_stops_early(monkeypatch):
    """Below the shared cap the uncapped probe is the pinned old head,
    bit for bit."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    Qh, best_h = _q0_pinned_head(split, Xte, taus)
    assert best_h is not None and best_h < rb.MAX_ITERS
    Qu, _, _, best_u = qs._fit_chimera_uncapped(split, Xte, None, 1, taus)
    assert best_u == best_h
    if not np.array_equal(Qu, Qh):
        i, j = np.argwhere(Qu != Qh)[0]
        pytest.fail(f"uncapped differs from head at [{i}, {j}]: "
                    f"head={Qh[i, j]!r} uncapped={Qu[i, j]!r}")
    assert np.array_equal(Qu, Qh)


def test_q0_recentred_sits_on_the_rigid_centre(monkeypatch):
    """Recentred is the pinned head's shape shifted onto RigidShift's
    median."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    from chimeraboost.quantile_api import _median_index

    split, Xte, taus = _q0_split()
    Qc, _, _, best_c = qs._fit_chimera_recentred(split, Xte, None, 1, taus)
    Qh, best_h = _q0_pinned_head(split, Xte, taus)
    Qs, _, _, _ = qs._fit_rigid_shift(split, Xte, None, 1, taus)
    mi, mw = _median_index(taus)
    assert mw == 0.0
    np.testing.assert_allclose(Qc[:, mi], Qs[:, mi], rtol=1e-12)
    d = Qc - Qh
    same = np.broadcast_to(d[:, [mi]], d.shape)  # constant along each row
    np.testing.assert_allclose(d, same)
    assert np.all(np.diff(Qc, axis=1) >= 0)
    assert best_c == best_h


def test_q0_valscaled_scales_about_the_median(monkeypatch):
    """ValScaled keeps the median and wears the library's own CQR factors."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    from chimeraboost.quantile_api import _cqr_scales

    split, Xte, taus = _q0_split()
    Xv, yv = split[1], np.asarray(split[3], dtype=np.float64)
    seen = _q0_capture_models(monkeypatch)
    Qv, _, _, _ = qs._fit_chimera_valscaled(split, Xte, None, 1, taus)
    Qh, _ = _q0_pinned_head(split, Xte, taus)
    assert len(seen) == 2
    m = seen[0]
    mi = int(np.argmin(np.abs(taus - 0.5)))
    assert taus[mi] == 0.5
    assert np.array_equal(Qv[:, mi], Qh[:, mi])
    saved = np.asarray(m.conformal_scale_)
    # The unscaled validation predictions the probe calibrated on: with the
    # factors reset to ones, predict returns them exactly.
    m.conformal_scale_ = np.ones_like(saved)
    ref = _cqr_scales(m.predict(Xv), yv, m.quantiles_, *m._median_idx_)
    assert np.array_equal(saved, ref)
    m.conformal_scale_ = saved
    Qvv = m.predict(Xv)
    assert taus[0] == 0.05 and taus[-1] == 0.95
    band = (yv >= Qvv[:, 0]) & (yv <= Qvv[:, -1])
    assert float(np.mean(band)) >= 0.90
    assert np.all(np.diff(Qv, axis=1) >= 0)


def test_q0_depth6_fits_depth6_trees(monkeypatch):
    """The Depth6 probe's fitted booster carries depth 6; the pinned
    head's, 4."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    seen = _q0_capture_models(monkeypatch)
    qs._fit_chimera_depth6(split, Xte, None, 1, taus)
    _q0_pinned_head(split, Xte, taus)
    assert [m.model_.depth for m in seen] == [6, 4]


# ---------------------------------------------------------------------------
# Q3 rate probes: two learning-rate arms and the depth-6 ValScaled arm.
# ---------------------------------------------------------------------------

def test_q3_rate_probes_resolve_learning_rate(monkeypatch):
    """The rate probes' fitted boosters run at 0.15 and 0.2; the pinned
    head's, at its 0.1 default. Read from the fitted booster, not the
    constructor."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    seen = _q0_capture_models(monkeypatch)
    qs.PROBES["ChimeraBoostQuantileLR15"](split, Xte, None, 1, taus)
    qs.PROBES["ChimeraBoostQuantileLR20"](split, Xte, None, 1, taus)
    _q0_pinned_head(split, Xte, taus)
    assert [m.model_.lr_ for m in seen] == [0.15, 0.2, 0.1]


def test_q3_depth6_valscaled_calibrates_depth6(monkeypatch):
    """Depth6ValScaled is depth 6 wearing the library's own CQR factors."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    from chimeraboost.quantile_api import _cqr_scales

    split, Xte, taus = _q0_split()
    Xv, yv = split[1], np.asarray(split[3], dtype=np.float64)
    seen = _q0_capture_models(monkeypatch)
    Qdv, _, _, _ = qs.PROBES["ChimeraBoostQuantileDepth6ValScaled"](
        split, Xte, None, 1, taus)
    Qd, _, _, _ = qs._fit_chimera_depth6(split, Xte, None, 1, taus)
    assert len(seen) == 2
    assert seen[0].model_.depth == 6
    mi = int(np.argmin(np.abs(taus - 0.5)))
    assert taus[mi] == 0.5
    assert np.array_equal(Qdv[:, mi], Qd[:, mi])
    m = seen[0]
    saved = np.asarray(m.conformal_scale_)
    # The unscaled validation predictions the probe calibrated on: with the
    # factors reset to ones, predict returns them exactly.
    m.conformal_scale_ = np.ones_like(saved)
    ref = _cqr_scales(m.predict(Xv), yv, m.quantiles_, *m._median_idx_)
    assert np.array_equal(saved, ref)
    m.conformal_scale_ = saved
    assert np.all(np.diff(Qdv, axis=1) >= 0)


def test_q5_field_arm_is_depth6_valscaled_bit_for_bit(monkeypatch):
    """The Q5 default reproduces the Depth6ValScaled probe exactly:
    depth 6 with the head's own CQR factors on the early-stopping
    rows."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    Qh, _, _, best_h = qs._fit_chimera_head(split, Xte, None, 1, taus)
    Qd, _, _, best_d = qs.PROBES["ChimeraBoostQuantileDepth6ValScaled"](
        split, Xte, None, 1, taus)
    assert best_h == best_d
    assert np.array_equal(Qh, Qd)


# ---------------------------------------------------------------------------
# Q2 CatBoost-knob probes: five one-knob arms off the current default head.
# ---------------------------------------------------------------------------

def test_q2_probes_carry_their_one_knob(monkeypatch):
    """Each Q2 probe's fitted booster carries its one change, read where the
    booster stores it, and keeps the library defaults otherwise: depth 6,
    128 bins, the resolved 0.1 rate, projection splits -- with the default
    "auto" calibration run (scales not all ones)."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    seen = _q0_capture_models(monkeypatch)
    names = ["ChimeraBoostQuantileBins254",
             "ChimeraBoostQuantileD6Uncapped",
             "ChimeraBoostQuantileLR03Uncapped",
             "ChimeraBoostQuantileDepth8",
             "ChimeraBoostQuantileExactSplits"]
    for name in names:
        qs.PROBES[name](split, Xte, None, 1, taus)
    assert len(seen) == 5
    bins, d6u, lr03u, d8, exact = seen

    # The one change each arm carries.
    assert bins.model_.max_bins == 254
    assert d6u.n_estimators == qs.UNCAPPED_ITERS
    assert lr03u.n_estimators == qs.UNCAPPED_ITERS
    assert lr03u.model_.lr_ == 0.03
    assert d8.model_.depth == 8
    assert exact.model_.exact_splits is True

    # Library defaults otherwise.
    for m in seen:
        if m is not d8:
            assert m.model_.depth == 6
        if m is not bins:
            assert m.model_.max_bins == 128
        if m is not lr03u:
            assert m.model_.lr_ == 0.1
        if m is not exact:
            assert m.model_.exact_splits is False
        assert not np.all(np.asarray(m.conformal_scale_) == 1.0)


def test_q2_d6_uncapped_matches_head_when_head_stops_early(monkeypatch):
    """Below the shared cap the D6Uncapped probe is the field arm, bit for
    bit: same constructor, same rows, only a larger round budget."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    Qh, _, _, best_h = qs._fit_chimera_head(split, Xte, None, 1, taus)
    assert best_h is not None and best_h < rb.MAX_ITERS
    Qu, _, _, best_u = qs.PROBES["ChimeraBoostQuantileD6Uncapped"](
        split, Xte, None, 1, taus)
    assert best_u == best_h
    if not np.array_equal(Qu, Qh):
        i, j = np.argwhere(Qu != Qh)[0]
        pytest.fail(f"d6uncapped differs from head at [{i}, {j}]: "
                    f"head={Qh[i, j]!r} d6uncapped={Qu[i, j]!r}")
    assert np.array_equal(Qu, Qh)


# ---------------------------------------------------------------------------
# Q6 audition: the H/B/R per-fit selection probe and its RecentredCal half.
# ---------------------------------------------------------------------------

def _q6_fit_raw(split, Xte, taus, **params):
    """One raw head plus its raw validation and test grids, on the suite's
    split -- the section-2 candidate recipe before calibration."""
    Xf, Xv, yf, yv = split
    m = qs._fit_head_model(split, None, 1, taus, rb.MAX_ITERS,
                           conformalize=False, **params)
    return m.predict(Xv), m.predict(Xte), m.best_iteration_


def _q6_point_centre(split, Xte):
    """The R centre: the RigidShift point fit plus the median validation
    residual, on the validation and test rows."""
    Xf, Xv, yf, yv = split
    pm = ChimeraBoostRegressor(n_estimators=rb.MAX_ITERS,
                              early_stopping_rounds=rb.PATIENCE,
                              thread_count=1, random_state=0)
    pm.fit(Xf, yf, eval_set=(Xv, yv))
    pv = np.asarray(pm.predict(Xv), dtype=np.float64).ravel()
    pt = np.asarray(pm.predict(Xte), dtype=np.float64).ravel()
    off = float(np.quantile(np.asarray(yv, dtype=np.float64) - pv, 0.5))
    return pv + off, pt + off


def test_q6_h_equals_field_arm_bit_for_bit(monkeypatch):
    """H -- the raw head plus the shared calibration -- is the field arm,
    bit for bit: the library's default does the same thing internally."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    Qv_raw, Qt_raw, best = _q6_fit_raw(split, Xte, taus)
    _, Qt = qs._calibrate(Qv_raw, Qt_raw, split[3], taus)
    Qh, _, _, best_h = qs._fit_chimera_head(split, Xte, None, 1, taus)
    assert best == best_h
    if not np.array_equal(Qt, Qh):
        i, j = np.argwhere(Qt != Qh)[0]
        pytest.fail(f"H differs from the field arm at [{i}, {j}]: "
                    f"H={Qt[i, j]!r} field={Qh[i, j]!r}")
    assert np.array_equal(Qt, Qh)


def test_q6_audition_picks_the_best_validation_crps(monkeypatch):
    """The audition returns the lowest validation-CRPS candidate, and its
    recorded fields match the recomputed scores."""
    from chimeraboost import quantile_metrics as qm
    from chimeraboost.quantile_api import _centre, _median_index

    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    Xf, Xv, yf, yv = split
    Qv_hr, Qt_hr, best_h = _q6_fit_raw(split, Xte, taus)
    Qv_h, Qt_h = qs._calibrate(Qv_hr, Qt_hr, yv, taus)
    Qv_br, Qt_br, best_b = _q6_fit_raw(split, Xte, taus, max_bins=254)
    Qv_b, Qt_b = qs._calibrate(Qv_br, Qt_br, yv, taus)
    cv, ct = _q6_point_centre(split, Xte)
    mi, mw = _median_index(taus)
    Qv_r, Qt_r = qs._calibrate(
        Qv_hr + (cv - _centre(Qv_hr, mi, mw))[:, None],
        Qt_hr + (ct - _centre(Qt_hr, mi, mw))[:, None], yv, taus)
    scores = [float(qm.crps(yv, Qv, taus)) for Qv in (Qv_h, Qv_b, Qv_r)]
    want = int(np.argmin(scores))

    got = qs._fit_chimera_audition(split, Xte, None, 1, taus)
    assert len(got) == 5
    Q, fit_s, pred_s, best, extra = got
    assert extra["audition_choice"] == want
    assert best == [best_h, best_b, best_h][want]
    assert fit_s > 0 and pred_s >= 0
    np.testing.assert_allclose(
        [extra["audition_val_crps_h"], extra["audition_val_crps_b"],
         extra["audition_val_crps_r"]], scores, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(Q, [Qt_h, Qt_b, Qt_r][want])


def test_q6_recentred_cal_sits_on_the_point_centre(monkeypatch):
    """RecentredCal's median column is the squared-error centre, and every
    returned grid has non-decreasing rows."""
    monkeypatch.setattr(rb, "MAX_ITERS", 300)
    split, Xte, taus = _q0_split()
    mi = int(np.argmin(np.abs(taus - 0.5)))
    assert taus[mi] == 0.5
    _, ct = _q6_point_centre(split, Xte)
    Qr, _, _, _ = qs._fit_chimera_recentred_cal(split, Xte, None, 1, taus)
    Qa, _, _, _, _ = qs._fit_chimera_audition(split, Xte, None, 1, taus)
    np.testing.assert_allclose(Qr[:, mi], ct, rtol=1e-12, atol=0)
    assert np.all(np.diff(Qr, axis=1) >= 0)
    assert np.all(np.diff(Qa, axis=1) >= 0)


def test_q6_extra_fields_land_in_metrics_in_both_harnesses(monkeypatch):
    """A stub arm returning a 5th dict gets its fields merged into the
    record's metrics in both run_one harnesses; a 4-value arm's record is
    exactly what score() returns."""
    import quantile_synth as qsyn
    from chimeraboost import quantile_metrics as qm
    from sklearn.model_selection import train_test_split

    taus = np.array([0.1, 0.5, 0.9])

    def _grid(n, k):
        base = np.linspace(-1.0, 1.0, n)
        return base[:, None] + np.linspace(-0.5, 0.5, k)[None, :]

    def _stub5(split, Xte, cat, threads, taus):
        return (_grid(Xte.shape[0], len(taus)), 0.1, 0.2, 7,
                {"probe_x": 1.5, "probe_n": 3})

    def _stub4(split, Xte, cat, threads, taus):
        return _grid(Xte.shape[0], len(taus)), 0.1, 0.2, 7

    monkeypatch.setattr(qs, "ARMS", {"Stub5": _stub5, "Stub4": _stub4})

    def _fake_builder(n_tasks, rng):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = X[:, 0] * 2.0 + rng.standard_normal(60)
        return X, y, None, {}

    monkeypatch.setattr(rb, "DATASETS", {"gr:reg_num/fake": _fake_builder})
    meta, out = qs.run_one("gr:reg_num/fake", 0, taus, 1,
                           ["Stub5", "Stub4"])
    assert meta["task"] == "quantile"
    m5, fit5, pred5, best5 = out["Stub5"]
    m4, _, _, _ = out["Stub4"]
    assert (fit5, pred5, best5) == (0.1, 0.2, 7)
    assert m5["probe_x"] == 1.5
    assert m5["probe_n"] == 3
    assert set(m5) == set(m4) | {"probe_x", "probe_n"}
    X, y, _, _ = _fake_builder(1, None)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
    split = rb._val_split(Xtr, ytr, "regression", 0)
    ref = qs.score(yte, _grid(Xte.shape[0], len(taus)), taus, split[2])
    assert m4 == ref

    meta_s, out_s = qsyn.run_one("location", 200, 0, taus, 1,
                                 ["Stub5", "Stub4"])
    assert meta_s["task"] == "quantile"
    ms5, _, _, _ = out_s["Stub5"]
    ms4, _, _, _ = out_s["Stub4"]
    assert ms5["probe_x"] == 1.5
    assert ms5["probe_n"] == 3
    assert set(ms5) == set(ms4) | {"probe_x", "probe_n"}
    Xn, yn, Q_or = qsyn.REGIMES["location"](200, 0, taus)
    idx_tr, idx_te = train_test_split(np.arange(200), test_size=0.25,
                                      random_state=0)
    Xtr, Xte = Xn[idx_tr], Xn[idx_te]
    ytr, yte = yn[idx_tr], yn[idx_te]
    split = rb._val_split(Xtr, ytr, "regression", 0)
    ref = qs.score(yte, _grid(Xte.shape[0], len(taus)), taus, split[2])
    ref["excess_crps"] = float(
        ref["crps"] - qm.crps(yte, Q_or[idx_te], taus))
    assert ms4 == ref
