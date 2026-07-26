"""Contracts for the SUS and temporal variant families (issue #37).

These lock the properties that make the variants honest rather than just
present: the frozen assignment matches the committed doc, a SUS twin is scored
on exactly its parent's test rows, a temporal split never lets a later row train
a model that is tested on an earlier one, and every family lands in its own
reporting stratum so no sign test silently pools a twin with its parent.
"""
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

import run_benchmarks as rb  # noqa: E402
import summarize  # noqa: E402

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmarks", "VARIANTS.md")


def _doc_tables():
    """{dataset: variant} parsed from the frozen tables in VARIANTS.md."""
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()
    sus = dict(re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*`@(\w+)`\s*\|$",
                          text, re.M))
    temporal = dict(re.findall(r"^\|\s*`((?:hc|pub):[^`]+)`\s*\|\s*`([^`]+)`\s*\|",
                               text, re.M))
    return sus, temporal


def test_sus_assignment_matches_doc():
    """The committed table is the source of truth for which datasets get twins."""
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    rb._add_public_datasets()
    doc_sus, _ = _doc_tables()

    code = {}
    for prefix in ("gr:", "hc:", "pub:"):
        keys = sorted(k for k in rb.DATASETS
                      if k.startswith(prefix) and rb.VARIANT_SEP not in k)
        code.update(rb._sus_assignment(keys))

    assert code == doc_sus


def test_temporal_registry_matches_doc():
    _, doc_time = _doc_tables()
    assert rb.TEMPORAL_COLUMNS == doc_time


def test_sus_proportions_near_target():
    """20% of each suite at 25% size, 10% at 50% -- within rounding."""
    rb._add_grinsztajn_datasets()
    for prefix, tol in (("gr:", 0.02), ("hc:", 0.06)):
        keys = sorted(k for k in rb.DATASETS
                      if k.startswith(prefix) and rb.VARIANT_SEP not in k)
        assign = rb._sus_assignment(keys)
        n = len(keys)
        f25 = sum(v == "sus25" for v in assign.values()) / n
        f50 = sum(v == "sus50" for v in assign.values()) / n
        assert abs(f25 - 0.20) <= tol, (prefix, f25)
        assert abs(f50 - 0.10) <= tol, (prefix, f50)


def test_sus_picks_are_disjoint():
    """No dataset may receive two twins -- the strides must not collide."""
    keys = [f"x:{i:03d}" for i in range(200)]
    assign = rb._sus_assignment(keys)
    both = [k for k in keys
            if assign.get(k) == "sus25" and assign.get(k) == "sus50"]
    assert not both
    assert set(assign.values()) <= {"sus25", "sus50"}


def test_sus_twin_keeps_parent_test_rows():
    """The twin must be scored on exactly the parent's test rows.

    This is what lets a twin be read as a point on the parent's learning curve;
    if the test set moved too, the comparison would confound less training data
    with a different (smaller, noisier) evaluation.
    """
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] > 0).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1,
                                          stratify=y)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25,
                                              random_state=1, stratify=y)
    Xs, ys = rb._subsample_train(Xtr2, ytr2, 0.25, "binary")

    assert np.array_equal(Xte, Xte2) and np.array_equal(yte, yte2)
    assert len(ys) < len(ytr)
    assert abs(len(ys) - 0.25 * len(ytr)) <= 1
    # every retained training row came from the parent's training rows
    assert {tuple(r) for r in Xs} <= {tuple(r) for r in Xtr}


def test_subsample_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    a = rb._subsample_train(X, y, 0.5, "regression")
    b = rb._subsample_train(X, y, 0.5, "regression")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_temporal_split_trains_on_the_past():
    """Every training row must precede every test row, for all seeds."""
    X = np.arange(1000).reshape(-1, 1).astype(float)   # already time-ordered
    y = np.arange(1000).astype(float)
    for seed in range(3):
        Xtr, Xte, ytr, yte = rb._temporal_split(X, y, seed, "regression")
        assert ytr.max() < yte.min(), seed
        assert len(ytr) and len(yte)


def test_temporal_seeds_differ():
    """Rolling origin must actually roll, or extra seeds buy nothing."""
    X = np.arange(1000).reshape(-1, 1).astype(float)
    y = np.arange(1000).astype(float)
    cuts = {len(rb._temporal_split(X, y, s, "regression")[0]) for s in range(3)}
    assert len(cuts) == 3, cuts


def test_temporal_drops_unseen_classes():
    """A label absent from the training window cannot be scored.

    Brier builds a one-hot row per test sample; for an unseen label that row is
    all zeros, so a confident wrong prediction would score WELL. Such rows are
    dropped instead.
    """
    # Training window (first 65%) sees classes 0 and 1; class 2 appears only
    # later, so it must be dropped from the test window rather than scored.
    y = np.array([0] * 400 + [1] * 300 + [2] * 300)
    X = np.arange(len(y)).reshape(-1, 1).astype(float)
    Xtr, Xte, ytr, yte = rb._temporal_split(X, y, 0, "multiclass")
    assert set(np.unique(ytr)) == {0, 1}
    assert set(np.unique(yte)) <= set(np.unique(ytr))
    assert 2 not in yte
    assert len(yte) == len(Xte)


def test_temporal_rejects_degenerate_window():
    """Single-class training window -> no variant, rather than a fake score."""
    y = np.array([0] * 900 + [1] * 100)
    X = np.arange(len(y)).reshape(-1, 1).astype(float)
    # cut 0.65 leaves only class 0 in training and only class 0 in the window
    out = rb._temporal_split(X, y, 0, "binary")
    assert out is None or len(np.unique(out[2])) >= 2


@pytest.mark.parametrize("col,expected_first", [
    (["09/22/1986", "05/05/2014", "11/19/1989"], "09/22/1986"),
    (["2014", "2003", "2010"], "2003"),
    ([1260144000, 1200000000, 1300000000], 1200000000),
])
def test_time_sort_key_orders_correctly(col, expected_first):
    """MM/DD/YYYY must not sort lexicographically, and year-like columns must
    sort numerically whatever dtype they arrive as."""
    pd = pytest.importorskip("pandas")
    s = pd.Series(col)
    key = rb._time_sort_key(s)
    assert key is not None
    assert s.iloc[key.argsort(kind="stable").iloc[0]] == expected_first


def test_time_sort_key_handles_unordered_category():
    pd = pytest.importorskip("pandas")
    s = pd.Series(pd.Categorical(["2014", "2003", "2010"], ordered=False))
    key = rb._time_sort_key(s)
    assert key is not None
    assert key.tolist() == [2014.0, 2003.0, 2010.0]


def test_time_sort_key_handles_mixed_timezones():
    """Timestamps carrying more than one UTC offset (kickstarter_projects does)
    make pandas raise rather than coerce, on the fallback call too -- so without
    utc=True this kills the run instead of ordering it."""
    pd = pytest.importorskip("pandas")
    s = pd.Series(["2013-06-01T12:00:00+02:00", "2013-06-01T09:00:00+00:00",
                   "2013-06-01T05:00:00-05:00"])
    key = rb._time_sort_key(s)
    assert key is not None
    # 10:00, 09:00 and 10:00 UTC: the +00:00 row is genuinely first.
    assert key.argsort(kind="stable").iloc[0] == 1


def test_task_of_strips_variant():
    rb._add_highcard_datasets()
    assert rb._task_of("hc:Moneyball@sus25") == rb._task_of("hc:Moneyball")
    assert rb._task_of("hc:kick@time") == rb._task_of("hc:kick")


def test_variants_get_their_own_strata():
    """A twin must never share a stratum with its parent -- that is what keeps
    a sign test from counting the same rows twice."""
    parent = summarize.stratum_of("hc:Moneyball")
    for v in ("sus25", "sus50", "time"):
        assert summarize.stratum_of(f"hc:Moneyball@{v}") != parent
    assert summarize.stratum_of("gr:x/y@sus25") != summarize.stratum_of("hc:z@sus25")


def test_split_strata_separates_families():
    keys = ["gr:a/b", "gr:a/c@sus25", "hc:d", "hc:d@time", "hc:e@sus50"]
    strata = summarize.split_strata(keys)
    assert len(strata) == 5
    # full-size suites are listed before variant families
    assert [s[1] for s in strata][:2] == ["", ""]


def test_base_key_and_variant_of():
    assert summarize.base_key("gr:reg_num/sulfur@sus25") == "gr:reg_num/sulfur"
    assert summarize.variant_of("gr:reg_num/sulfur@sus25") == "sus25"
    assert summarize.variant_of("gr:reg_num/sulfur") == ""
