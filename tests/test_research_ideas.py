"""Guards for the research cascade's idea registry (issue #81).

Every idea with ``implemented=True`` must set only params the current library
accepts; retired ideas stay unimplemented and are refused before any fit runs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))
from research import cascade, ideas  # noqa: E402

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


def test_implemented_ideas_use_live_flags():
    """The test that would have caught #81: an implemented idea's params must
    be accepted by BOTH estimators (the cascade builds either from the same
    params, regressor or classifier per dataset task)."""
    reg_keys = ChimeraBoostRegressor().get_params()
    clf_keys = ChimeraBoostClassifier().get_params()
    live = [n for n, s in ideas.IDEAS.items() if s.get("implemented")]
    assert live, "expected at least one implemented idea (self-test anchors)"
    for name in live:
        for k in ideas.IDEAS[name]["params"]:
            assert k in reg_keys, f"{name}: {k!r} not a regressor param"
            assert k in clf_keys, f"{name}: {k!r} not a classifier param"


def test_retired_ideas_are_not_implemented():
    for name, spec in ideas.IDEAS.items():
        if "retired" in spec:
            assert spec["implemented"] is False, name


def _no_fit(*args, **kwargs):
    raise AssertionError("no tier runner may run for a refused idea")


def test_retired_idea_refuses_before_any_fit(monkeypatch):
    monkeypatch.setattr(cascade, "run_fast_tier", _no_fit)
    monkeypatch.setattr(cascade, "run_promo_tier", _no_fit)
    with pytest.raises(SystemExit) as exc:
        cascade.cascade("C1_onehot_low_card", ["T0"], 0, None, 1)
    assert ideas.IDEAS["C1_onehot_low_card"]["retired"] in str(exc.value)


def test_unknown_flag_refuses_before_any_fit(monkeypatch):
    monkeypatch.setattr(cascade, "run_fast_tier", _no_fit)
    monkeypatch.setattr(cascade, "run_promo_tier", _no_fit)
    monkeypatch.setitem(ideas.IDEAS, "throwaway_bad_flag", dict(
        params={"no_such_flag_xyz": True},
        category="test",
        implemented=True,
        direction="lower_better",
        hypothesis="throwaway entry for the unknown-flag guard",
    ))
    with pytest.raises(SystemExit) as exc:
        cascade.cascade("throwaway_bad_flag", ["T0"], 0, None, 1)
    assert "no_such_flag_xyz" in str(exc.value)
