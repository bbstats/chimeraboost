"""Fast robustness guard: a curated subset of benchmarks/fuzz_inputs.py.

The full boundary x pathology x task cross-product (255 cases) lives in the
on-demand `benchmarks/fuzz_inputs.py`. This module reuses that harness's case
generators (single source of truth) but runs a curated, fast slice every test
cycle, so a regression that lets a bad input crash -- or a bad param slip through
without a clean error -- fails here immediately.

Each case is asserted to be either PASS (ran, finite, right shape; proba sums to
1) or OK-REJECTED (a clean named ValueError/TypeError), never CRASH (an ugly
exception, non-finite output, or a bad input silently accepted).
"""
import os
import sys

import pytest

# Reuse the fuzz harness as the source of truth (repo convention: benchmarks add
# themselves to sys.path; here a test reaches into benchmarks/ the same way).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
fz = pytest.importorskip("fuzz_inputs")


# Highest-value input pathologies to keep in the fast guard (one or two per
# category). The full set runs in the on-demand sweep.
_KEEP_INPUT = {
    "nullable:Int64+NA", "nullable:all-NA", "dtype:object-numeric", "dtype:float32",
    "cat:pandas-category", "cat:pandas-string", "cat:datetime",
    "cat:high-cardinality-unique", "cat:unseen-at-predict",
    "cat:category-no-cat_features", "shape:1-row", "shape:1-feature",
    "shape:wide-n<<p", "content:all-nan-column", "content:all-constant-column",
    "content:all-nan-row", "content:duplicate-columns", "content:huge-target-1e300",
    "content:+inf", "content:nan-only-at-predict", "predict:wrong-feature-count",
    "predict:column-reorder", "target:single-class",
}


def _id(case):
    return f"{case['task']}-{case['name']}"


_INPUT = [c for c in fz.iter_input_cases() if c.get("name") in _KEEP_INPUT]
# All parameter cases: the reject cases guard _validate_hyperparams and are
# near-instant; the pass boundaries (depth=1/16, max_bins=2, etc.) are cheap too.
_PARAM = list(fz.iter_param_cases())
_FITOPT = list(fz.iter_fit_option_cases())


@pytest.mark.parametrize("case", _INPUT, ids=[_id(c) for c in _INPUT])
def test_input_pathology(case):
    outcome, detail = fz.run_case(case)
    assert outcome != fz.Outcome.CRASH, f"{_id(case)}: {detail}"


@pytest.mark.parametrize("case", _PARAM, ids=[_id(c) for c in _PARAM])
def test_parameter_boundaries(case):
    outcome, detail = fz.run_case(case)
    assert outcome != fz.Outcome.CRASH, f"{_id(case)}: {detail}"


@pytest.mark.parametrize("case", _FITOPT, ids=[_id(c) for c in _FITOPT])
def test_fit_options(case):
    outcome, detail = fz.run_case(case)
    assert outcome != fz.Outcome.CRASH, f"{_id(case)}: {detail}"


def test_curated_subset_is_nonempty():
    # Guard against a rename in the harness silently emptying the subset.
    assert len(_INPUT) >= 15
    assert any(c["expect"] == "reject" for c in _PARAM)
    assert any(c["expect"] == "pass" for c in _PARAM)
