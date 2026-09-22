"""Exactness of the predict-time direct categorical lookup (_direct_codes)."""
import os
import sys

import numpy as np
import pytest

import chimeraboost.preprocessing as pmod
from chimeraboost.preprocessing import (
    CatTransformCache,
    FeaturePreprocessor,
    _direct_codes,
    _remap_codes,
)
from chimeraboost.target_encoding import _factorize_numeric, factorize

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, BENCH)

import run_benchmarks as rb  # noqa: E402


class _RaisesNE:
    """Value whose self-inequality raises (missing by the factorize contract)."""

    def __ne__(self, other):
        raise TypeError("boom")

    def __eq__(self, other):
        raise TypeError("boom")


def _old_codes_for(col, mapping):
    c, cats = factorize(col)
    return _remap_codes(cats, mapping, -1)[c]


def _fit_prep(X_fit):
    n = X_fit.shape[0]
    prep = FeaturePreprocessor()
    prep.fit_transform(X_fit, [np.zeros(n)], [0, 1, 2])
    return prep


def _mixed_fit_test():
    fit_c0 = ["a", "b", True, 1, 1.0, "1.5", 1.5, -0.0, 0.0,
              np.int64(3), 3, "__nan__", "c", None, float("nan")]
    test_c0 = [float("nan"), np.nan, np.float32("nan"), None, True, 1, 1.0,
               "1.5", 1.5, -0.0, 0.0, np.int64(3), 3, "__nan__",
               "a", "unseen_0"]
    fit_c1 = ["x", "y", None, "z", "x"]
    test_c1 = ["x", _RaisesNE(), "y", None, "unseen_1"]
    fit_c2 = [None, float("nan"), None, np.nan, None]
    test_c2 = [None, np.nan, None, None, np.nan]
    # Pad columns to a rectangular matrix with each column's own first value.
    n_fit = max(len(fit_c0), len(fit_c1), len(fit_c2))
    n_test = max(len(test_c0), len(test_c1), len(test_c2))

    def _pad(vals, n):
        return list(vals) + [vals[0]] * (n - len(vals))

    X_fit = np.empty((n_fit, 3), dtype=object)
    X_fit[:, 0] = _pad(fit_c0, n_fit)
    X_fit[:, 1] = _pad(fit_c1, n_fit)
    X_fit[:, 2] = _pad(fit_c2, n_fit)
    X_test = np.empty((n_test, 3), dtype=object)
    X_test[:, 0] = _pad(test_c0, n_test)
    X_test[:, 1] = _pad(test_c1, n_test)
    X_test[:, 2] = _pad(test_c2, n_test)
    return X_fit, X_test


def test_edge_case_codes_equal_where_direct_applies():
    X_fit, X_test = _mixed_fit_test()
    prep = _fit_prep(X_fit)

    # Column 0 is the hashed string column: the direct path must apply.
    assert _direct_codes(X_test[:, 0], prep.cat_maps_[0]) is not None
    # Raising-__ne__ makes the mask unsafe; all-missing is numeric-accepted.
    assert _direct_codes(X_test[:, 1], prep.cat_maps_[1]) is None
    assert _direct_codes(X_test[:, 2], prep.cat_maps_[2]) is None

    for j, f in enumerate(prep.cat_features_):
        col = X_test[:, f]
        direct = _direct_codes(col, prep.cat_maps_[j])
        old = _old_codes_for(col, prep.cat_maps_[j])
        if direct is not None:
            assert direct.dtype == np.int64
            assert direct.shape == old.shape
            assert bool((direct == old).all())

    live = prep._codes_for_transform(X_test)
    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        off = prep._codes_for_transform(X_test)
    finally:
        pmod._direct_codes = orig
    assert live.dtype == np.int64 and off.dtype == np.int64
    assert live.shape == off.shape
    assert bool((live == off).all())

    # A shared context keeps the existing path and agrees bit-identically.
    shared = prep._codes_for_transform(X_test, CatTransformCache())
    assert bool((shared == off).all())


def test_empty_batch_codes_equal():
    X_fit, _ = _mixed_fit_test()
    prep = _fit_prep(X_fit)
    X_empty = np.empty((0, 3), dtype=object)
    for j, f in enumerate(prep.cat_features_):
        assert _direct_codes(X_empty[:, f], prep.cat_maps_[j]) is None
    live = prep._codes_for_transform(X_empty)
    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        off = prep._codes_for_transform(X_empty)
    finally:
        pmod._direct_codes = orig
    assert live.shape == (0, 3) and off.shape == (0, 3)
    assert bool((live == off).all())


def _has_non_numeric(Xte, cat_idx):
    for f in cat_idx:
        col = np.asarray(Xte[:, f], dtype=object)
        if _factorize_numeric(col) is None:
            return True
    return False


def _assert_prep_codes_match(prep, Xte):
    for j, f in enumerate(prep.cat_features_):
        col = np.asarray(Xte[:, f], dtype=object)
        direct = _direct_codes(col, prep.cat_maps_[j])
        old = _old_codes_for(col, prep.cat_maps_[j])
        if direct is not None:
            assert direct.shape == old.shape
            assert bool((direct == old).all())
    live = prep._codes_for_transform(Xte)
    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        off = prep._codes_for_transform(Xte)
    finally:
        pmod._direct_codes = orig
    assert live.shape == off.shape
    assert bool((live == off).all())


def test_real_data_codes_equal():
    from sklearn.model_selection import train_test_split

    rb._add_highcard_datasets()
    keys = sorted(k for k in rb.DATASETS if k.startswith("hc:"))
    tested, skipped = 0, []
    for key in keys:
        try:
            X, y, cat_idx, _task = rb.DATASETS[key](
                1.0, np.random.default_rng(0))
        except Exception:  # no network / no cache -> not a code failure
            skipped.append(key)
            continue
        if not cat_idx:
            continue
        Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0)
        if not _has_non_numeric(Xte, cat_idx):
            continue
        prep = FeaturePreprocessor()
        prep.fit_transform(Xtr, [np.zeros(Xtr.shape[0])], list(cat_idx))
        _assert_prep_codes_match(prep, Xte)
        tested += 1
    if tested == 0:
        pytest.skip(f"hc data cache absent (skipped {len(skipped)}/{len(keys)})")


def _string_frame(n, rng, unseen=False):
    cats = ["a", "b", "c", "d"] + (["unseen"] if unseen else [])
    c0 = [cats[i % len(cats)] for i in range(n)]
    c1 = [cats[(i * 3) % len(cats)] for i in range(n)]
    num = rng.normal(size=n)
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = c0
    X[:, 1] = c1
    X[:, 2] = list(num)
    return X


def test_end_to_end_bitidentical():
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    rng = np.random.default_rng(0)
    Xtr = _string_frame(200, rng)
    Xte = _string_frame(80, rng, unseen=True)
    y_clf = np.array([1 if v == "a" else 0 for v in Xtr[:, 0]])
    y_reg = np.asarray(Xtr[:, 2], dtype=np.float64) * 2.0 + 1.0

    clf = ChimeraBoostClassifier().fit(Xtr, y_clf, cat_features=[0, 1])
    pred_live = clf.predict(Xte)
    proba_live = clf.predict_proba(Xte)
    reg = ChimeraBoostRegressor().fit(Xtr, y_reg, cat_features=[0, 1])
    reg_live = reg.predict(Xte)

    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        pred_off = clf.predict(Xte)
        proba_off = clf.predict_proba(Xte)
        reg_off = reg.predict(Xte)
    finally:
        pmod._direct_codes = orig

    assert np.array_equal(pred_live, pred_off)
    assert np.array_equal(proba_live, proba_off)
    assert np.array_equal(reg_live, reg_off)


def _numeric_frame(n, rng, unseen=False):
    c0 = [int(i % 4) for i in range(n)]
    c1 = [float((i % 3) + 0.5) for i in range(n)]
    for i in range(0, n, 17):
        c0[i] = None
    for i in range(0, n, 19):
        c0[i] = float("nan")
    for i in range(0, n, 13):
        c1[i] = None
    for i in range(0, n, 15):
        c1[i] = float("nan")
    if unseen:
        c0[1] = 999
        c1[2] = 999.5
    num = rng.normal(size=n)
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = c0
    X[:, 1] = c1
    X[:, 2] = list(num)
    return X


def test_numeric_coded_bitidentical_single_factorize():
    from chimeraboost import ChimeraBoostClassifier

    import chimeraboost.target_encoding as tmod

    rng = np.random.default_rng(0)
    Xtr = _numeric_frame(200, rng)
    Xte = _numeric_frame(80, rng, unseen=True)
    y = np.array([i % 2 for i in range(200)])

    clf = ChimeraBoostClassifier(random_state=0).fit(Xtr, y, cat_features=[0, 1])
    pred_live = clf.predict(Xte)
    proba_live = clf.predict_proba(Xte)

    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        pred_off = clf.predict(Xte)
        proba_off = clf.predict_proba(Xte)
    finally:
        pmod._direct_codes = orig
    assert np.array_equal(pred_live, pred_off)
    assert np.array_equal(proba_live, proba_off)

    orig_p = pmod._factorize_numeric
    orig_t = tmod._factorize_numeric
    calls = []

    def count_p(col):
        calls.append(1)
        return orig_p(col)

    def count_t(col):
        calls.append(1)
        return orig_t(col)

    pmod._factorize_numeric = count_p
    tmod._factorize_numeric = count_t
    try:
        clf.predict(Xte)
    finally:
        pmod._factorize_numeric = orig_p
        tmod._factorize_numeric = orig_t
    assert len(calls) == 2


def _allcat_frame(n, unseen=False):
    cats = ["a", "b", "c", "d"] + (["unseen"] if unseen else [])
    c0 = [cats[i % len(cats)] for i in range(n)]
    c1 = [cats[(i * 2 + 1) % len(cats)] for i in range(n)]
    c2 = [cats[(i * 3 + 2) % len(cats)] for i in range(n)]
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = c0
    X[:, 1] = c1
    X[:, 2] = c2
    return X


def test_combos_bitidentical_old_path():
    from chimeraboost import ChimeraBoostClassifier

    Xtr = _allcat_frame(200)
    Xte = _allcat_frame(80, unseen=True)
    y = np.array([1 if v == "a" else 0 for v in Xtr[:, 0].tolist()])

    clf = ChimeraBoostClassifier(
        cat_combinations=True, random_state=0).fit(Xtr, y, cat_features=[0, 1, 2])
    assert len(clf.model_.prep_.combo_pairs_) > 0
    pred_live = clf.predict(Xte)
    proba_live = clf.predict_proba(Xte)

    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        pred_off = clf.predict(Xte)
        proba_off = clf.predict_proba(Xte)
    finally:
        pmod._direct_codes = orig
    assert np.array_equal(pred_live, pred_off)
    assert np.array_equal(proba_live, proba_off)

    orig_factorize = pmod.factorize
    orig_codes = FeaturePreprocessor._codes_for_transform
    orig_combo = FeaturePreprocessor._combo_codes_for_transform
    base_n = [0]
    combo_n = [0]
    phase = ["none"]

    def counting_factorize(col):
        if phase[0] == "base":
            base_n[0] += 1
        elif phase[0] == "combo":
            combo_n[0] += 1
        return orig_factorize(col)

    def codes_wrapper(self, X, cat_ctx=None, *a, **k):
        phase[0] = "base"
        try:
            return orig_codes(self, X, cat_ctx, *a, **k)
        finally:
            phase[0] = "none"

    def combo_wrapper(self, X, cat_ctx=None):
        phase[0] = "combo"
        try:
            return orig_combo(self, X, cat_ctx)
        finally:
            phase[0] = "none"

    pmod.factorize = counting_factorize
    FeaturePreprocessor._codes_for_transform = codes_wrapper
    FeaturePreprocessor._combo_codes_for_transform = combo_wrapper
    try:
        clf.predict(Xte)
    finally:
        pmod.factorize = orig_factorize
        FeaturePreprocessor._codes_for_transform = orig_codes
        FeaturePreprocessor._combo_codes_for_transform = orig_combo
    assert base_n[0] == 3
    assert combo_n[0] == 0


def test_gdiff_bitidentical():
    rng = np.random.default_rng(0)
    Xtr = _string_frame(200, rng)
    Xte = _string_frame(80, rng, unseen=True)
    prep = FeaturePreprocessor(cross_pairs=[(2, 0, "gdiff")])
    prep.fit_transform(Xtr, [np.zeros(200)], [0, 1])
    assert len(prep.gdiff_maps_) == 1
    live = prep.transform(Xte)
    orig = pmod._direct_codes
    pmod._direct_codes = lambda col, mapping: None
    try:
        off = prep.transform(Xte)
    finally:
        pmod._direct_codes = orig
    assert live.shape == off.shape
    assert bool((live == off).all())
