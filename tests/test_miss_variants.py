"""Contracts for the injected-missingness variants @mcar / @mar / @mnar.

They lock what makes the test bed honest: the advertised rate is hit, the
masks are reproducible per seed, categorical columns are never touched, MAR
missingness really is driven by fully observed columns, MNAR really is driven
by the masked value itself, and each mechanism is its own reporting stratum.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

import run_benchmarks as rb  # noqa: E402
import summarize  # noqa: E402


def _data(n_tr=4000, n_te=1000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tr, p)), rng.standard_normal((n_te, p))


@pytest.mark.parametrize("mech", rb.MISS_MECHANISMS)
def test_rate_hit_on_affected_columns(mech):
    Xtr, Xte = _data()
    out_tr, out_te, info = rb._inject_missing(Xtr, Xte, None, mech, seed=0)
    miss = np.isnan(out_tr).mean(axis=0)
    hit = miss[miss > 0]
    assert len(hit) == info["miss_cols"]
    assert np.allclose(hit, rb.MISS_RATE, atol=0.03), hit
    # Test rows come from the same process, so their rate is close too.
    te = np.isnan(out_te).mean(axis=0)
    assert np.allclose(te[miss > 0], rb.MISS_RATE, atol=0.06)


@pytest.mark.parametrize("mech", rb.MISS_MECHANISMS)
def test_seeded_and_non_mutating(mech):
    Xtr, Xte = _data()
    Xtr0, Xte0 = Xtr.copy(), Xte.copy()
    a = rb._inject_missing(Xtr, Xte, None, mech, seed=1)
    b = rb._inject_missing(Xtr, Xte, None, mech, seed=1)
    c = rb._inject_missing(Xtr, Xte, None, mech, seed=2)
    assert np.array_equal(Xtr, Xtr0) and np.array_equal(Xte, Xte0)
    assert np.array_equal(np.isnan(a[0]), np.isnan(b[0]))
    assert not np.array_equal(np.isnan(a[0]), np.isnan(c[0]))


def test_mechanisms_differ_for_one_seed():
    Xtr, Xte = _data()
    masks = [np.isnan(rb._inject_missing(Xtr, Xte, None, m, seed=0)[0])
             for m in rb.MISS_MECHANISMS]
    assert not np.array_equal(masks[0], masks[2])


def test_categoricals_untouched():
    Xtr, Xte = _data(p=4)
    Xtr = Xtr.astype(object)
    Xte = Xte.astype(object)
    Xtr[:, 1] = "a"
    Xte[:, 1] = "b"
    for mech in rb.MISS_MECHANISMS:
        out_tr, out_te, _ = rb._inject_missing(Xtr, Xte, [1], mech, seed=0)
        assert (out_tr[:, 1] == "a").all() and (out_te[:, 1] == "b").all()


def test_mar_drivers_observed_and_informative():
    Xtr, Xte = _data(n_tr=20000)
    out, _, info = rb._inject_missing(Xtr, Xte, None, "mar", seed=0)
    M = np.isnan(out)
    observed = ~M.any(axis=0)
    assert observed.sum() == Xtr.shape[1] - info["miss_cols"] >= 1
    # Missingness in a masked column is predictable from the driver columns
    # and NOT from the column's own (hidden) value beyond that.
    drivers = Xtr[:, observed]
    for j in np.flatnonzero(~observed):
        r = max(abs(np.corrcoef(drivers[:, d], M[:, j])[0, 1])
                for d in range(drivers.shape[1]))
        assert r > 0.1


def test_mnar_depends_on_own_value():
    Xtr, Xte = _data(n_tr=20000)
    out, _, _ = rb._inject_missing(Xtr, Xte, None, "mnar", seed=0)
    M = np.isnan(out)
    for j in range(Xtr.shape[1]):
        assert abs(np.corrcoef(Xtr[:, j], M[:, j])[0, 1]) > 0.3


def test_mar_needs_two_numeric_columns():
    Xtr, Xte = _data(p=1)
    assert rb._inject_missing(Xtr, Xte, None, "mar", seed=0) is None
    assert rb._inject_missing(Xtr, Xte, None, "mcar", seed=0) is not None


def test_registration_is_grinsztajn_only():
    saved = dict(rb.DATASETS)
    try:
        rb.DATASETS.clear()
        f = lambda scale, rng: None  # noqa: E731
        rb.DATASETS.update({"gr:a/b": f, "hc:c": f, "gr:a/b@sus25": f})
        rb._add_miss_datasets(list(rb.DATASETS))
        added = set(rb.DATASETS) - {"gr:a/b", "hc:c", "gr:a/b@sus25"}
        assert added == {f"gr:a/b@{m}" for m in rb.MISS_MECHANISMS}
    finally:
        rb.DATASETS.clear()
        rb.DATASETS.update(saved)


def test_each_mechanism_is_its_own_stratum():
    keys = ["gr:a/b"] + [f"gr:a/b@{m}" for m in rb.MISS_MECHANISMS]
    strata = summarize.split_strata(keys)
    assert len(strata) == 4
    for m in rb.MISS_MECHANISMS:
        assert m in summarize.VARIANT_LABELS
