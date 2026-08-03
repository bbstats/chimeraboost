"""Public suite contracts: registration, frozen-list<->doc agreement, and the
overlap gate.

The public suite is the one the published chart runs on, so its overlap gate is
strictly harder than HC's: it checks the HC decision suite too, by OpenML id AND
by name, because charting a dataset we tune on would make the figure in-sample.

The suite itself is NOT sealed (changed 2026-07-27) -- it is validation, read
freely, and never blocks a ship. TabArena's full run is the sealed holdout.

PUBLIC_DATASETS is frozen at 22 datasets. The registration and composition tests
skip if it is ever emptied; test_freeze_is_all_or_nothing fails on a half-freeze
(code populated but doc not, or the reverse) either way.
"""
import os
import re
import sys

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import run_benchmarks as rb  # noqa: E402
import suite_overlap  # noqa: E402

PLAN_MD = os.path.join(BENCH, "PUBLIC_PLAN.md")


def _parse_frozen_doc():
    """Rows of the form `| pub:<name> | <id> | <task> | ...` in PUBLIC_PLAN.md."""
    out = {}
    with open(PLAN_MD, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\|\s*pub:([\w.\-]+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), m.group(3))
    return out


# --------------------------------------------------------------------------
# the freeze is atomic
# --------------------------------------------------------------------------
def test_freeze_is_all_or_nothing():
    """Code and doc must be populated together. A frozen table with no code
    behind it (or the reverse) is a half-shipped suite, which is how a chart
    ends up running on something nobody audited."""
    doc, code = _parse_frozen_doc(), rb.PUBLIC_DATASETS
    assert bool(doc) == bool(code), (
        f"PUBLIC_PLAN.md has {len(doc)} frozen `| pub:... |` rows but "
        f"PUBLIC_DATASETS has {len(code)} entries — freeze both or neither")


def test_frozen_matches_doc():
    doc = _parse_frozen_doc()
    code = {name: (spec["data_id"], spec["task"])
            for name, spec in rb.PUBLIC_DATASETS.items()}
    assert doc == code, (
        "PUBLIC_DATASETS and the PUBLIC_PLAN.md frozen table disagree — "
        f"only in doc: {set(doc) - set(code)}; only in code: {set(code) - set(doc)}; "
        f"mismatched: {{k: (doc[k], code[k]) for k in set(doc)&set(code) if doc[k]!=code[k]}}")


# --------------------------------------------------------------------------
# the hard gate
# --------------------------------------------------------------------------
def test_no_suite_overlap():
    # include_hc=True: unlike HC's own test, the public suite must also clear
    # the HC decision suite, by id and by name.
    pools = suite_overlap.exclusion_pools(include_hc=True)
    failures = []
    for name, spec in rb.PUBLIC_DATASETS.items():
        failures += suite_overlap.overlap_failures(name, spec["data_id"], pools)
    assert not failures, (
        "public suite overlaps a sealed/decision suite:\n" + "\n".join(failures))


def test_overlap_gate_actually_catches_a_hit():
    """The gate is only worth having if it fires. Feed it a known member of each
    pool and require a failure -- otherwise an empty PUBLIC_DATASETS would make
    test_no_suite_overlap pass for the wrong reason forever."""
    pools = suite_overlap.exclusion_pools(include_hc=True)
    # 1590 is adult's OpenML id, from the retired oml: gate's registry. Decorative
    # here -- adult is in suite_overlap.CONSUMED_ELSEWHERE, so the name alone fires.
    assert suite_overlap.overlap_failures("adult", 1590, pools)
    assert suite_overlap.overlap_failures("Amazon_employee_access", None, pools)
    assert suite_overlap.overlap_failures("wine-reviews", None, pools)   # HC
    assert not suite_overlap.overlap_failures("a-name-in-no-suite", 10**9, pools)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def test_registration_idempotent_and_tasks():
    if not rb.PUBLIC_DATASETS:
        pytest.skip("public suite not frozen yet (see PUBLIC_PLAN.md)")
    rb._add_public_datasets()
    keys = [k for k in rb.DATASETS if k.startswith("pub:")]
    rb._add_public_datasets()   # second call must not duplicate
    assert keys == [k for k in rb.DATASETS if k.startswith("pub:")]
    assert len(keys) == len(rb.PUBLIC_DATASETS) >= 12
    for name, spec in rb.PUBLIC_DATASETS.items():
        key = f"pub:{name}"
        assert key in rb.DATASETS
        assert rb._task_of(key) == spec["task"] == rb.PUBLIC_TASKS[key]
        assert spec["task"] in ("binary", "multiclass", "regression")
        if spec.get("time_col"):
            assert rb.TEMPORAL_COLUMNS[key] == spec["time_col"]


def test_composition_meets_plan_targets():
    if not rb.PUBLIC_DATASETS:
        pytest.skip("public suite not frozen yet (see PUBLIC_PLAN.md)")
    tasks = [s["task"] for s in rb.PUBLIC_DATASETS.values()]
    assert len(rb.PUBLIC_DATASETS) >= 12
    assert tasks.count("regression") >= 2
    assert tasks.count("multiclass") >= 3
    assert tasks.count("binary") >= 3
    n_time = sum(1 for s in rb.PUBLIC_DATASETS.values() if s.get("time_col"))
    assert n_time >= 3, "the @time variant family needs >= 3 datasets to run on"


def test_public_flag_errors_while_empty(monkeypatch):
    """Until the freeze, --public must refuse rather than run an empty suite."""
    if rb.PUBLIC_DATASETS:
        pytest.skip("public suite is frozen; the guard no longer applies")
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--public", "--seeds", "1"])
    with pytest.raises(SystemExit):
        rb.main()


# --------------------------------------------------------------------------
# drop_cols: what near-uniqueness cannot catch
# --------------------------------------------------------------------------
def test_drop_cols_removes_named_columns(monkeypatch):
    """rossmann ships a train/test/valid `Set` marker and freMTPL2freq a numeric
    IDpol row id; neither is near-unique enough for the categorical filter."""
    pd = pytest.importorskip("pandas")
    import numpy as np

    frame = pd.DataFrame({
        "keep": np.arange(50.0),
        "Set": ["train"] * 30 + ["test"] * 20,
        "IDpol": np.arange(50.0),
        "y": np.arange(50.0),
    })

    # Both builders now read a parquet off disk, so stub both halves of that
    # seam: the path (no download) and the read (no file, no pyarrow needed).
    seen = []

    def _fake_path(data_id):
        seen.append(data_id)
        return "<stub parquet>"

    monkeypatch.setattr(rb, "_public_parquet_path", _fake_path)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())

    spec = dict(data_id=-1, task="regression", target="y",
                drop_cols=("Set", "IDpol", "absent"))
    builder = rb._make_highcard_builder(spec, max_rows=None)
    X, y, cat, task = builder(1.0, np.random.default_rng(0))
    assert seen == [-1], "the loader must go through the cached-parquet path"
    assert X.shape[1] == 1, "only `keep` should survive drop_cols"
    assert task == "regression" and len(y) == 50


# --------------------------------------------------------------------------
# weighted aggregation
# --------------------------------------------------------------------------
def test_facet_metadata_covers_the_suite():
    """Every frozen dataset needs facets, or the weighting is not reproducible."""
    assert set(rb.PUBLIC_FACETS) == set(rb.PUBLIC_DATASETS), (
        f"only in facets: {set(rb.PUBLIC_FACETS) - set(rb.PUBLIC_DATASETS)}; "
        f"only in datasets: {set(rb.PUBLIC_DATASETS) - set(rb.PUBLIC_FACETS)}")
    for name, m in rb.PUBLIC_FACETS.items():
        assert m["rows"] >= 50_000, f"{name} is below the suite's size floor"
        assert m["maxcard"] >= 0


def test_weights_sum_to_one_and_respect_the_cap():
    import suite_weights as sw
    facets = sw.dataset_facets(rb.PUBLIC_DATASETS, rb.PUBLIC_FACETS)
    w = sw.raked_weights(facets)
    n = len(w)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    lo, hi = sw.WEIGHT_CAP
    for k, v in w.items():
        assert lo / n - 1e-9 <= v <= hi / n + 1e-9, f"{k} escaped the cap at {v * n:.2f}x"


def test_weighting_actually_balances_and_costs_ess():
    """Raking must move every facet toward equal mass, and the cost of doing so
    must be visible: effective sample size drops below n."""
    import suite_weights as sw
    facets = sw.dataset_facets(rb.PUBLIC_DATASETS, rb.PUBLIC_FACETS)
    w = sw.raked_weights(facets)
    n = len(w)
    equal = {k: 1.0 / n for k in w}

    for facet, acc in sw.facet_balance(facets, w).items():
        target = 1.0 / len(acc)
        before = sw.facet_balance(facets, equal)[facet]
        worst_after = max(abs(v - target) for v in acc.values())
        worst_before = max(abs(v - target) for v in before.values())
        assert worst_after <= worst_before + 1e-9, f"{facet} got less balanced"

    ess = sw.effective_sample_size(w)
    assert ess < n, "weighting that costs no ESS is not weighting"
    assert ess > 0.6 * n, f"ESS {ess:.1f} of {n} -- the cap should stop this"


def test_average_rank_handles_ties_as_midranks():
    import suite_weights as sw
    scores = {"d1": {"a": 1.0, "b": 1.0, "c": 2.0}}
    r = sw.average_rank(scores, ["a", "b", "c"])
    assert r["a"] == r["b"] == 1.5, "a two-way tie must share the midrank"
    assert r["c"] == 3.0


def test_weighted_median_respects_weight():
    import suite_weights as sw
    # 1 carries almost all the mass, so the median must be 1 despite three 9s.
    assert sw.weighted_median([1, 9, 9, 9], [0.97, 0.01, 0.01, 0.01]) == 1
    assert sw.weighted_median([1, 2, 3], [1, 1, 1]) == 2


def test_competitor_relative_rank_is_composition_stable():
    """Adding one of our rungs must not move another rung's rank. Plain
    average rank over the whole field fails this -- the new rung becomes an
    opponent -- which is exactly why the chart does not use it."""
    import suite_weights as sw
    scores = {
        "d1": {"r1": 0.5, "r2": 0.4, "CatBoost": 0.45, "LightGBM": 0.6},
        "d2": {"r1": 0.3, "r2": 0.2, "CatBoost": 0.35, "LightGBM": 0.25},
        "d3": {"r1": 0.9, "r2": 0.8, "CatBoost": 0.7, "LightGBM": 0.95},
    }
    comps = ["CatBoost", "LightGBM"]
    one = sw.competitor_relative_rank(scores, ["r1"], comps)
    two = sw.competitor_relative_rank(scores, ["r1", "r2"], comps)
    assert abs(one["r1"] - two["r1"]) < 1e-12, "adding a rung moved another rung"

    # and the plain full-field rank demonstrably does NOT have this property
    a = sw.average_rank(scores, ["r1"] + comps)["r1"]
    b = sw.average_rank(scores, ["r1", "r2"] + comps)["r1"]
    assert abs(a - b) > 1e-9, "full-field rank was expected to be unstable here"


def test_no_rung_ranks_against_a_sibling():
    """A rung's rank must be computed in a 3-way field, so with two competitors
    it can never exceed 3.0 no matter how many rungs are charted."""
    import suite_weights as sw
    scores = {"d1": {"r1": 9.0, "r2": 8.0, "CatBoost": 1.0, "LightGBM": 2.0}}
    r = sw.competitor_relative_rank(scores, ["r1", "r2"], ["CatBoost", "LightGBM"])
    assert r["r1"] == 3.0 and r["r2"] == 3.0, (
        "worst-of-three is 3.0; a larger value means siblings competed")
