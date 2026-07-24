"""Pair-code categorical combinations (2026-07-23 semi-bug hunt).

The old combo encoding concatenated stringified values with "_x_", which
(a) aliased distinct pairs whose values contain the delimiter, (b) turned a
real missing value into the literal string "nan" (bypassing factorize's
``__nan__`` sentinel), and (c) split int/float spellings of one value.
Combos are now pairs of the parents' canonical categories; models pickled
with the old string-keyed maps keep the legacy path.
"""

import numpy as np

from chimeraboost import ChimeraBoostClassifier
from chimeraboost.preprocessing import CatTransformCache, FeaturePreprocessor


def _combo_codes(X):
    """Canonical combo codes and categories for columns (0, 1) of X."""
    return CatTransformCache().combo(X, 0, 1)


def test_delimiter_collision_pairs_stay_distinct():
    """("a_x", "b") and ("a", "x_b") both stringified to "a_x_x_b" and were
    one category; as pairs they are two."""
    X = np.array([["a_x", "b"],
                  ["a", "x_b"],
                  ["a_x", "b"]], dtype=object)
    codes, cats = _combo_codes(X)
    assert codes[0] != codes[1]
    assert codes[0] == codes[2]
    assert len(cats) == 2


def test_missing_value_distinct_from_literal_nan_string():
    """A real NaN routes to the ``__nan__`` sentinel; the string "nan" is an
    ordinary category. The old path merged them ("nan_x_b")."""
    X = np.array([[np.nan, "b"],
                  ["nan", "b"]], dtype=object)
    codes, cats = _combo_codes(X)
    assert codes[0] != codes[1]
    assert ("__nan__", "b") in cats
    assert ("nan", "b") in cats


def test_combo_codes_stable_across_batches():
    """Predict-time combo codes come from the fit-time maps regardless of the
    batch's own row order; unseen pairs of seen values fall back to -1."""
    Xfit = np.array([["a", "p"], ["b", "q"], ["a", "q"]], dtype=object)
    prep = FeaturePreprocessor(cat_combinations=True)
    prep._split_columns_fit(Xfit, [0, 1])
    # Reversed batch order + an unseen pair ("b","p") of two seen values.
    Xnew = np.array([["b", "p"], ["a", "q"], ["b", "q"], ["a", "p"]],
                    dtype=object)
    out = prep._combo_codes_for_transform(Xnew)[:, 0]
    fit_codes = prep._combo_codes_for_transform(Xfit)[:, 0]
    assert list(fit_codes) == [0, 1, 2]        # fit-time first-appearance
    assert out[0] == -1                        # unseen pair -> prior fallback
    assert list(out[1:]) == [2, 1, 0]


def test_legacy_string_keyed_maps_keep_predicting():
    """A model whose combo maps are old-style strings (pre-change pickle)
    predicts through the legacy string-concat path, identically."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.choice(["u", "v", "w"], size=n)
    b = rng.choice(["p", "q"], size=n)
    X = np.column_stack([a, b]).astype(object)
    y = ((a == "u") ^ (b == "p")).astype(int)   # combo-only signal
    clf = ChimeraBoostClassifier(n_estimators=30, cat_combinations=True,
                                 random_state=0)
    clf.fit(X, y, cat_features=[0, 1])
    prep = clf.model_.prep_
    assert prep.combo_pairs_
    before = clf.predict_proba(X)
    # Rewrite the maps exactly as the old fit stored them: "a_x_b" strings.
    prep.combo_maps_ = [
        {f"{va}_x_{vb}": code for (va, vb), code in m.items()}
        for m in prep.combo_maps_]
    after = clf.predict_proba(X)
    np.testing.assert_array_equal(before, after)
