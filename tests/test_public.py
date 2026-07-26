"""Public suite contracts: registration, frozen-list<->doc agreement, and the
sealed-holdout overlap gate.

The public suite is the one the published chart runs on, so its overlap gate is
strictly harder than HC's: it checks the HC decision suite too, by OpenML id AND
by name, because charting a dataset we tune on would make the headline figure
in-sample.

PUBLIC_DATASETS is empty until the audit's remaining metadata checks can run
(benchmarks/PUBLIC_PLAN.md, "What remains"). These tests pass vacuously while it
is empty and start biting the moment it is populated -- except
test_freeze_is_all_or_nothing, which fails on a half-freeze right now.
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
    adult_id = rb.OPENML_SUITE["adult"]["data_id"]
    assert suite_overlap.overlap_failures("adult", adult_id, pools)
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
def test_drop_cols_removes_named_columns():
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

    class _DS:
        pass

    ds = _DS()
    ds.frame = frame
    ds.target = frame["y"]
    spec = dict(data_id=-1, task="regression", drop_cols=("Set", "IDpol", "absent"))
    builder = rb._make_highcard_builder(spec, max_rows=None)
    import sklearn.datasets as skds
    orig = skds.fetch_openml
    skds.fetch_openml = lambda **kw: ds
    try:
        X, y, cat, task = builder(1.0, np.random.default_rng(0))
    finally:
        skds.fetch_openml = orig
    assert X.shape[1] == 1, "only `keep` should survive drop_cols"
    assert task == "regression" and len(y) == 50
