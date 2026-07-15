"""SynthGen generator contracts: determinism, goldens, meta, structure, suites.

The golden-hash test is the numpy-RNG-stream-drift tripwire: if it fails after
a numpy upgrade (NEP 19 does not guarantee distribution-method streams), bump
synthgen.recipe.VERSION and re-freeze the suite -- never re-pin goldens under
the same version.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import synthgen
from synthgen import filters
from synthgen.suites import CANARIES, SUITES

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_synthgen.json")


@pytest.fixture(autouse=True)
def _flat_memory():
    yield
    synthgen.build_dataset.cache_clear()


def test_key_roundtrip():
    key = synthgen.key_for(31)
    assert key == f"syn:{synthgen.VERSION}/031"
    version, did = synthgen.parse_key(key)
    assert (version, did) == (synthgen.VERSION, 31)
    with pytest.raises(ValueError):
        synthgen.parse_key("syn:v0/031")
    with pytest.raises(ValueError):
        synthgen.parse_key("gr:clf_num/pol")


def test_determinism_and_arg_independence():
    key = synthgen.key_for(3)
    h1 = synthgen.hash_dataset(key)
    synthgen.build_dataset.cache_clear()
    h2 = synthgen.hash_dataset(key)
    assert h1 == h2
    builder = synthgen.make_builder(key)
    X1, y1, c1, t1 = builder(1.0, np.random.default_rng(0))
    X2, y2, c2, t2 = builder(9.9, np.random.default_rng(12345))
    assert (t1, c1) == (t2, c2)
    assert np.array_equal(y1, y2)
    if X1.dtype == object:
        same = all((a == b) or (a != a and b != b)
                   for a, b in zip(X1.ravel(), X2.ravel()))
        assert same
    else:
        assert np.array_equal(X1, X2, equal_nan=True)


def test_golden_hashes():
    goldens = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    assert goldens, "golden_synthgen.json is empty"
    for key, expected in goldens.items():
        assert synthgen.hash_dataset(key) == expected, (
            f"{key} content drifted -- if this is a numpy upgrade, bump "
            "synthgen VERSION and re-freeze (see module docstring)")
        synthgen.build_dataset.cache_clear()


def test_meta_and_dtype_contracts():
    for did in range(12):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        assert X.shape == (meta["n"], meta["d"])
        assert task == meta["task"] == synthgen.task_of(key)
        json.dumps(meta)  # JSON-safe
        if cat:
            assert X.dtype == object
            for j in cat:
                col = X[:100, j]
                assert all(isinstance(v, str) for v in col)
                assert not any(v.replace(".", "").lstrip("-").isdigit()
                               for v in col if v != "__nan__")
        else:
            assert X.dtype == np.float64
        if task == "regression":
            assert meta["noise_sigma"] and meta["noise_sigma"] > 0
            assert meta["bayes_brier"] is None
        else:
            assert y.dtype == np.int64
            counts = np.bincount(y)
            assert len(counts) == meta["n_classes"]
            assert 0.0 <= meta["bayes_brier"] <= 2.0
            if not meta["degenerate"]:
                assert counts.min() >= max(10, int(0.005 * meta["n"]))
        synthgen.build_dataset.cache_clear()


def test_saturated_floors_are_zero():
    seen = 0
    for did in range(40):
        rec = synthgen.sample_recipe(did)
        if not rec.saturated or rec.task == "regression":
            continue
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(did))
        assert meta["bayes_brier"] == 0.0
        synthgen.build_dataset.cache_clear()
        seen += 1
    assert seen >= 1


def test_classification_floor_matches_recomputation():
    # bayes_brier stored must equal sum-form Brier of the true p against y --
    # verified indirectly: floor <= observed Brier of the *ideal* constant
    # predictor and >= 0; exact recomputation happens inside emit. Here we
    # check the invariant bayes_brier < Brier(constant class prior).
    for did in range(20):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        if task == "regression" or meta["degenerate"]:
            synthgen.build_dataset.cache_clear()
            continue
        k = meta["n_classes"]
        prior = np.bincount(y, minlength=k) / len(y)
        onehot = np.zeros((len(y), k))
        onehot[np.arange(len(y)), y] = 1.0
        const_brier = float(((prior[None, :] - onehot) ** 2).sum(axis=1).mean())
        # expected Brier of the true p is <= any constant's; the REALIZED gap
        # can go slightly the other way on weak-signal data (O(1/sqrt(n)))
        assert meta["bayes_brier"] <= const_brier + 0.01
        synthgen.build_dataset.cache_clear()


def test_regression_noise_floor_below_target_std():
    for did in range(20):
        key = synthgen.key_for(did)
        _, y, _, task, meta = synthgen.build_dataset(key)
        if task == "regression":
            assert meta["noise_sigma"] < float(np.std(y))
        synthgen.build_dataset.cache_clear()


def test_irrelevant_columns_uncorrelated():
    # a dataset with irrelevant numeric columns: max |corr(x_j, y)| over the
    # weakest columns should be small on a decent-n dataset
    for did in range(40):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        if (task == "regression" and meta["irrelevant_fraction"] > 0.3
                and not cat and meta["n"] >= 3000):
            yz = (y - y.mean()) / y.std()
            cors = []
            for j in range(X.shape[1]):
                v = X[:, j].astype(float)
                m = np.isfinite(v)
                vz = v[m] - v[m].mean()
                s = vz.std()
                if s > 0:
                    cors.append(abs(float(np.mean(vz / s * yz[m]))))
            cors = np.sort(cors)
            n_irr = int(round(meta["irrelevant_fraction"] * meta["d"]))
            assert cors[: max(1, n_irr // 2)].max() < 0.15
            synthgen.build_dataset.cache_clear()
            return
        synthgen.build_dataset.cache_clear()
    pytest.skip("no qualifying dataset in probe range")


def test_entity_column_mechanism():
    from synthgen import emit
    rng = np.random.default_rng(7)
    codes, card, effect_rows, sigma_e = emit._entity_column(
        5000, np.log(10.0), 64, rng)
    assert codes.shape == (5000,) and codes.max() == card - 1
    assert 0.3 <= sigma_e <= 1.0
    counts = np.bincount(codes, minlength=card)
    # Zipf-ish frequencies: the heaviest level dwarfs the median level
    assert counts.max() > 5 * max(1.0, float(np.median(counts)))
    # singleton rare levels exist (the unseen-at-train stress)
    assert (counts == 1).sum() >= 2
    # per-row effect is a per-level lookup: constant within each level
    for lvl in range(min(card, 5)):
        v = effect_rows[codes == lvl]
        if len(v):
            assert np.allclose(v, v[0])


def test_entity_cats_present_in_meta():
    seen = 0
    for did in range(30):
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(did))
        if meta["n_cat_entity"] > 0:
            seen += 1
            assert meta["entity_strength"] > 0
            assert meta["n_cat_entity"] <= meta["n_cat"]
        synthgen.build_dataset.cache_clear()
        if seen >= 3:
            break
    assert seen >= 1


def test_canaries_contract():
    # canary status is earned at freeze time and frozen into suites.py
    assert CANARIES, "CANARIES empty -- freeze.py paste missing"
    assert set(CANARIES) <= set(SUITES["full"])
    for i in sorted(CANARIES)[:5]:
        assert synthgen.sample_recipe(i).saturated
    n_cat_bearing = 0
    for i in sorted(set(CANARIES) & set(SUITES["screen"])):
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(i))
        n_cat_bearing += meta["n_cat"] > 0
        synthgen.build_dataset.cache_clear()
    assert n_cat_bearing >= 3, "screen needs >=3 cat-bearing verified canaries"


def test_suite_nesting_and_registration():
    assert set(SUITES["smoke"]) <= set(SUITES["screen"]) <= set(SUITES["full"])
    keys = synthgen.frozen_keys("smoke")
    assert keys and all(k.startswith(f"syn:{synthgen.VERSION}/") for k in keys)
    assert len(synthgen.all_frozen_keys()) == len(set(synthgen.all_frozen_keys()))


def test_harness_registration():
    import run_benchmarks as rb
    rb._add_synth_datasets()
    keys = synthgen.all_frozen_keys()
    assert all(k in rb.DATASETS for k in keys)
    assert all(rb._task_of(k) in ("regression", "binary", "multiclass") for k in keys)


def test_filters_behave():
    ok, why = filters.degeneracy_ok(np.zeros((100, 3)), np.zeros(100), "regression")
    assert not ok
    y = np.array([0] * 995 + [1] * 5)
    ok, _ = filters.degeneracy_ok(np.zeros((1000, 3)), y, "binary")
    assert not ok
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    y = X[:, 0] * 2.0 + rng.normal(size=1500) * 0.1
    ok, detail = filters.learnable(X, y, None, "regression")
    assert ok, detail
    ok, _ = filters.tractable({"n_cat": 3, "cat_fraction": 1.0, "max_cardinality": 80})
    assert not ok
    ok, _ = filters.tractable({"n_cat": 3, "cat_fraction": 1.0, "max_cardinality": 16})
    assert ok
