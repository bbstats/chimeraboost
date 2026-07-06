"""Accuracy + timing regression guards against tests/golden_metrics.json.

Reuses benchmarks/make_golden.py (single source of truth for the measurement
config and logic) to recompute metrics on the current code, then compares to the
committed golden baseline.

* Accuracy is fully deterministic (fixed seed, single thread, early_stopping off),
  so the metric is bit-identical run-to-run on the same code; the tolerance only
  absorbs cross-machine float drift. A real accuracy regression moves it well
  past the band.
* Timing is normalized by an in-process pure-numpy calibration (see make_golden),
  making it machine-independent. It is inherently noisier, so the default bound is
  generous (catch gross slowdowns, never flake); set CHIMERA_STRICT_TIMING=1 for a
  tight bound when investigating a suspected slowdown on a fixed machine. For
  precise timing work use benchmarks/profile_fit.py / scaling_*.py.

Regenerate the golden (and commit it) with `python benchmarks/make_golden.py`
when a change is intended to move the numbers.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
mg = pytest.importorskip("make_golden")

ACC_RTOL = 0.02                                   # 2% relative band on the metric
STRICT = os.environ.get("CHIMERA_STRICT_TIMING") == "1"
TIME_FACTOR = 1.25 if STRICT else 1.6             # fail if current > golden*factor
TIME_ABS_SLACK = 0.05                             # ignore wobble on tiny ratios


@pytest.fixture(scope="module")
def measured():
    records, _cal = mg.measure_all()
    return records


@pytest.fixture(scope="module")
def golden():
    with open(mg.GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)["records"]


def test_golden_covers_panel(golden):
    assert set(golden) == set(mg.PANEL), "golden file is stale vs the panel"


@pytest.mark.parametrize("ds", mg.PANEL)
def test_accuracy_no_regression(ds, measured, golden):
    cur, base = measured[ds]["metric"], golden[ds]["metric"]
    # Metrics are lower-is-better; flag drift in EITHER direction (a drop is also
    # a signal that something moved and the golden should be intentionally reset).
    assert cur == pytest.approx(base, rel=ACC_RTOL), (
        f"{ds}: metric {cur:.6f} drifted from golden {base:.6f} "
        f"(>{ACC_RTOL:.0%}); if intended, rerun make_golden.py")


@pytest.mark.parametrize("kind", ["fit_ratio", "predict_ratio"])
@pytest.mark.parametrize("ds", mg.PANEL)
def test_timing_no_regression(ds, kind, measured, golden):
    cur, base = measured[ds][kind], golden[ds][kind]
    limit = base * TIME_FACTOR + TIME_ABS_SLACK
    assert cur <= limit, (
        f"{ds} {kind}: {cur:.3f} exceeds {limit:.3f} "
        f"(golden {base:.3f} x{TIME_FACTOR} + {TIME_ABS_SLACK} slack) -- "
        f"possible timing regression")
