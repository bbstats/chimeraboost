"""compare_runs' near-solved guard.

A dataset every model solves to a practically-zero loss carries no information
about a change, but it destroys a RELATIVE mean: the ratio of two tiny numbers
is numerical noise. On a real historical screen one such dataset (syn:v2/117)
contributed -12555% alone and dragged an 88-dataset mean to -144.7% while the
sign test read 54 wins / 31 losses.

Contracts pinned here:
  * near-solved datasets are excluded from the mean, by the same thresholds the
    rest of the analysis stack uses (summarize.NEAR_SOLVED_NRMSE / 1e-3 Brier);
  * they are NAMED in the output, never dropped silently;
  * the sign test and its PASS/FAIL bar still count every shared dataset, so
    the guard cannot silently flip a verdict recorded in an older plan file;
  * --keep-near-solved reproduces the old unguarded arithmetic for auditing.

Deterministic fixtures only -- no benchmark runs, no network.
"""
import json
import os
import sys

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import compare_runs  # noqa: E402


def _write(tmp_path, name, brier_solved, brier_real, rmse_solved, rmse_real):
    """One run JSON: two classification sets (one solved) and two regression
    sets (one solved). y_std makes reg_solved's NRMSE 0.005 and reg_real's 0.1."""
    datasets = {
        "bin_solved": {"task": "binary"},
        "bin_real": {"task": "binary"},
        "reg_solved": {"task": "regression", "y_std": 100.0},
        "reg_real": {"task": "regression", "y_std": 10.0},
    }
    records = [
        {"dataset": "bin_solved", "model": "M", "seed": 0, "fit_time": 1.0,
         "metrics": {"primary": 1.0, "brier": brier_solved}},
        {"dataset": "bin_real", "model": "M", "seed": 0, "fit_time": 1.0,
         "metrics": {"primary": 0.8, "brier": brier_real}},
        {"dataset": "reg_solved", "model": "M", "seed": 0, "fit_time": 1.0,
         "metrics": {"primary": -rmse_solved, "rmse": rmse_solved}},
        {"dataset": "reg_real", "model": "M", "seed": 0, "fit_time": 1.0,
         "metrics": {"primary": -rmse_real, "rmse": rmse_real}},
    ]
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datasets": datasets, "records": records}, fh)
    return path


@pytest.fixture
def pair(tmp_path):
    base = _write(tmp_path, "base.json", 1e-9, 0.20, 0.5, 1.0)
    # bin_solved moves from 1e-9 to 1e-6: a 100000% relative "regression" that
    # is entirely numerical noise. Everything else moves by a sane amount.
    new = _write(tmp_path, "new.json", 1e-6, 0.19, 0.6, 0.9)
    return base, new


def test_near_solved_identified_by_shared_thresholds(pair):
    base, new = pair
    _, rmse_b, brier_b, meta = compare_runs.load_run(base, "M", "brier")
    _, rmse_n, brier_n, _ = compare_runs.load_run(new, "M", "brier")
    flag = lambda ds: compare_runs.is_near_solved(  # noqa: E731
        ds, meta, rmse_b, rmse_n, brier_b, brier_n)
    assert flag("bin_solved")      # Brier 1e-9 < 1e-3
    assert not flag("bin_real")    # Brier 0.20
    assert flag("reg_solved")      # NRMSE 0.005 < 0.02
    assert not flag("reg_real")    # NRMSE 0.1


def test_the_two_rules_degrade_differently_without_metadata(pair):
    """On an older JSON with no dataset metadata the rules must differ, by
    design: the regression rule cannot form an NRMSE without y_std and so
    excludes nothing, while the Brier rule needs no metadata and still bites."""
    base, new = pair
    _, rmse_b, brier_b, _ = compare_runs.load_run(base, "M", "brier")
    _, rmse_n, brier_n, _ = compare_runs.load_run(new, "M", "brier")
    flag = lambda ds: compare_runs.is_near_solved(  # noqa: E731
        ds, {}, rmse_b, rmse_n, brier_b, brier_n)
    assert flag("bin_solved")        # Brier 1e-9 is solved, metadata or not
    assert not flag("reg_solved")    # no y_std -> cannot judge -> kept


def _main(capsys, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["compare_runs.py"] + argv)
    compare_runs.main()
    return capsys.readouterr().out


def test_solved_dataset_excluded_from_mean_and_named(pair, capsys, monkeypatch):
    base, new = pair
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M", "--metric", "brier"])
    assert "near-solved: excluded from mean" in out
    assert "bin_solved" in out            # named, not silently dropped
    # Only bin_real survives, at +5% (0.20 -> 0.19 Brier).
    assert "+5.000%" in out
    assert "n=1" in out


def test_unguarded_mode_reproduces_the_old_blowup(pair, capsys, monkeypatch):
    base, new = pair
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M", "--metric", "brier",
                 "--keep-near-solved"])
    assert "near-solved" not in out
    # bin_solved's -99900% swamps bin_real's +5% -> a hugely negative mean.
    line = [l for l in out.splitlines() if "mean relative change" in l][0]
    assert "-49" in line or "-50" in line  # ~ -49947%


def test_sign_test_bar_still_counts_every_dataset(pair, capsys, monkeypatch):
    """The guard must not silently flip a verdict recorded in an older plan."""
    base, new = pair
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M", "--metric", "brier"])
    # 2 shared classification sets: bin_solved loses, bin_real wins.
    assert "1 wins / 1 losses / 0 ties  (of 2 datasets)" in out
    # ...and the retained-only sign test is reported alongside as a diagnostic.
    assert "excluding near-solved: 1 wins / 0 losses" in out


def test_regression_guard_uses_nrmse_not_raw_rmse(pair, capsys, monkeypatch):
    """reg_solved has the SMALLER raw RMSE but is excluded on NRMSE; reg_real
    has a larger RMSE and is kept. A raw-magnitude rule would get this wrong."""
    base, new = pair
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M"])
    rows = {l.split()[0]: l for l in out.splitlines()
            if l and l.split()[0].startswith("reg_")}
    assert "near-solved" in rows["reg_solved"]
    assert "near-solved" not in rows["reg_real"]
