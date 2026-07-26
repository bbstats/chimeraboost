"""Contracts for the competitor-relative win rate used by the public chart.

The published number must mean "how often does this configuration beat the
competitors", not "how often does it beat a field we chose, most of which is
ourselves". These tests pin the two properties that buys: correctness against a
hand-computed case, and stability when a sibling rung is added.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

import summarize  # noqa: E402

COMPETITORS = ["CatBoost", "LightGBM"]


def _primary(**per_ds):
    return dict(per_ds)


def test_matches_hand_computed_rate():
    # ours beats both competitors on d1, loses to both on d2, splits on d3
    primary = _primary(
        d1={"ChimeraBoost": 1.0, "CatBoost": 2.0, "LightGBM": 3.0},
        d2={"ChimeraBoost": 9.0, "CatBoost": 2.0, "LightGBM": 3.0},
        d3={"ChimeraBoost": 2.5, "CatBoost": 2.0, "LightGBM": 3.0},
    )
    r = summarize.winrate_vs_opponents(primary, COMPETITORS)
    # 6 matchups: win, win, lose, lose, lose, win -> 3/6
    assert r["ChimeraBoost"] == pytest.approx(50.0)


def test_ties_count_one_half():
    primary = _primary(d1={"ChimeraBoost": 1.0, "CatBoost": 1.0})
    r = summarize.winrate_vs_opponents(primary, ["CatBoost"])
    assert r["ChimeraBoost"] == pytest.approx(50.0)


def test_model_is_not_scored_against_itself():
    """A competitor listed as an opponent must not play itself."""
    primary = _primary(d1={"CatBoost": 1.0, "LightGBM": 2.0})
    r = summarize.winrate_vs_opponents(primary, COMPETITORS)
    assert r["CatBoost"] == pytest.approx(100.0)
    assert r["LightGBM"] == pytest.approx(0.0)


def test_adding_a_sibling_rung_does_not_move_other_rows():
    """The stability property that field-relative win rate lacks.

    Adding another ChimeraBoost operating point must leave every existing
    model's published number untouched -- otherwise each release would silently
    restate the previous release's claims.
    """
    base = _primary(
        d1={"ChimeraBoost": 1.0, "CatBoost": 2.0, "LightGBM": 3.0},
        d2={"ChimeraBoost": 4.0, "CatBoost": 2.0, "LightGBM": 5.0},
    )
    with_rung = {ds: dict(s) for ds, s in base.items()}
    with_rung["d1"]["ChimeraBoostEns8"] = 0.5
    with_rung["d2"]["ChimeraBoostEns8"] = 1.0

    a = summarize.winrate_vs_opponents(base, COMPETITORS)
    b = summarize.winrate_vs_opponents(with_rung, COMPETITORS)
    for m in a:
        assert a[m] == pytest.approx(b[m]), m

    # the field-relative version does NOT have this property, which is why the
    # public chart does not use it
    fa = summarize.winrate_vs_field(base)
    fb = summarize.winrate_vs_field(with_rung)
    assert fa["ChimeraBoost"] != pytest.approx(fb["ChimeraBoost"])


def test_ci_brackets_the_point_estimate():
    primary = _primary(
        **{f"d{i}": {"ChimeraBoost": 1.0 if i % 3 else 5.0,
                     "CatBoost": 2.0, "LightGBM": 3.0} for i in range(12)})
    r = summarize.winrate_vs_opponents(primary, COMPETITORS)
    ci = summarize.bootstrap_winrate_vs_opponents_ci(
        primary, COMPETITORS, n_boot=500)
    lo, hi = ci["ChimeraBoost"]
    assert lo <= r["ChimeraBoost"] <= hi


def test_missing_competitor_on_a_dataset_is_skipped_not_counted():
    """A dataset where a competitor did not run must not silently score as a win."""
    primary = _primary(
        d1={"ChimeraBoost": 1.0, "CatBoost": 2.0},
        d2={"ChimeraBoost": 1.0},                 # nobody to beat here
    )
    r = summarize.winrate_vs_opponents(primary, COMPETITORS)
    assert r["ChimeraBoost"] == pytest.approx(100.0)   # 1 matchup, won
