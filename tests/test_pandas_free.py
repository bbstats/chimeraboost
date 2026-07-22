"""The library must run without pandas, and the shared-factorization bagged
predict must be bit-identical to per-member prediction.

pandas was dropped as a dependency (sklearn does not pull it in): frames are
consumed through their own to_numpy/columns attributes, and the categorical
machinery (factorize, code mapping, gdiff group means) is numpy/numba only.
"""

import sys

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.preprocessing import CatTransformCache, FeaturePreprocessor
from chimeraboost.target_encoding import factorize


def _cat_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = rng.choice(["a", "b", "c", "d"], n)
    X[:, 1] = rng.normal(size=n)
    X[:, 2] = rng.choice(["x", "y", "z"], n)
    X[rng.random(n) < 0.05, 0] = np.nan
    y = ((X[:, 0] == "a").astype(int) ^ (X[:, 2] == "x").astype(int))
    return X, y


def test_factorize_first_appearance_and_missing():
    codes, cats = factorize(np.array(["b", None, "a", np.nan, "b", "a"],
                                     dtype=object))
    assert cats.tolist() == ["b", "__nan__", "a"]
    assert codes.tolist() == [0, 1, 2, 1, 0, 2]


class _BlockPandas:
    """Meta-path finder that makes `import pandas` fail as if not installed."""

    def find_spec(self, name, path=None, target=None):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas is blocked for this test")
        return None


def test_library_runs_without_pandas(monkeypatch):
    """Categorical fit + predict (incl. bagging, combos, and gdiff crosses)
    in a simulated pandas-less environment: the module is absent from
    sys.modules and any fresh import raises ImportError."""
    for mod in [m for m in sys.modules
                if m == "pandas" or m.startswith("pandas.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(sys, "meta_path", [_BlockPandas()] + sys.meta_path)
    X, y = _cat_data()

    clf = ChimeraBoostClassifier(n_estimators=40, random_state=0,
                                 cat_combinations=True)
    clf.fit(X, y, cat_features=[0, 2])
    assert clf.predict_proba(X).shape == (len(y), 2)

    with pytest.warns(UserWarning, match="member defaults"):
        bag = ChimeraBoostRegressor(n_estimators=30, random_state=0,
                                    n_ensembles=3)
        bag.fit(X, y.astype(float), cat_features=[0, 2])
    assert bag.predict(X).shape == (len(y),)

    prep = FeaturePreprocessor(max_bins=32, random_state=0,
                               cross_pairs=[(1, 0, "gdiff")])
    Xb = prep.fit_transform(X, [y.astype(np.float64)], [0, 2])
    assert prep.transform(X).shape == Xb.shape


def test_bagged_predict_matches_member_average_bitwise():
    """The shared conversion + CatTransformCache path must reproduce the
    per-member public predictions exactly, DataFrame input included."""
    pd = pytest.importorskip("pandas")
    X, y = _cat_data()
    df = pd.DataFrame({"c1": X[:, 0], "n1": X[:, 1].astype(float),
                       "c2": X[:, 2]})

    reg = ChimeraBoostRegressor(n_estimators=40, random_state=0, n_ensembles=3)
    with pytest.warns(UserWarning, match="member defaults"):
        reg.fit(df, y.astype(float), cat_features=["c1", "c2"])
    manual = np.mean([m.predict(df) for m in reg.estimators_], axis=0)
    assert np.array_equal(reg.predict(df), manual)

    clf = ChimeraBoostClassifier(n_estimators=40, random_state=0, n_ensembles=3)
    with pytest.warns(UserWarning, match="member defaults"):
        clf.fit(df, y, cat_features=["c1", "c2"])
    acc = np.zeros((len(y), clf.n_classes_))
    for m in clf.estimators_:
        acc[:, np.searchsorted(clf.classes_, m.classes_)] += m.predict_proba(df)
    assert np.array_equal(clf.predict_proba(df), acc / len(clf.estimators_))


def test_cat_transform_cache_shares_factorizations():
    """A shared cache factorizes each column once; repeated transforms through
    it produce codes identical to uncached transforms."""
    X, y = _cat_data()
    prep = FeaturePreprocessor(max_bins=32, random_state=0)
    prep.fit_transform(X, [y.astype(np.float64)], [0, 2])

    ctx = CatTransformCache()
    first = prep.transform(X, ctx)
    assert set(ctx._columns) == {0, 2}
    codes_obj = [id(v) for v in ctx._columns.values()]
    again = prep.transform(X, ctx)
    assert [id(v) for v in ctx._columns.values()] == codes_obj  # reused, not rebuilt
    assert np.array_equal(first, again)
    assert np.array_equal(first, prep.transform(X))