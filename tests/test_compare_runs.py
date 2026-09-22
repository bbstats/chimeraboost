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


# --------------------------------------------------------------------------
# POINTER label (GATE_ROBUSTNESS.md #2): a stratum with fewer than 8 decided
# datasets cannot separate a regression from noise, so every sign-test line
# says so. Print-only -- the PASS/FAIL word never changes.
# --------------------------------------------------------------------------
def _write_many(tmp_path, name, n_decided, n_ties, shift):
    """One run JSON with ``n_decided`` binary sets that move by ``shift`` and
    ``n_ties`` that do not; Brier 0.2 keeps every set clear of near-solved."""
    datasets, records = {}, []
    for i in range(n_decided + n_ties):
        ds = f"set{i:02d}"
        datasets[ds] = {"task": "binary"}
        b = 0.20 + (shift if i < n_decided else 0.0)
        records.append({"dataset": ds, "model": "M", "seed": 0, "fit_time": 1.0,
                        "metrics": {"primary": 1.0 - b, "brier": b}})
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datasets": datasets, "records": records}, fh)
    return path


def test_pointer_label_marks_small_strata_and_keeps_the_verdict(
        tmp_path, capsys, monkeypatch):
    base = _write_many(tmp_path, "b.json", n_decided=4, n_ties=6, shift=0.0)
    new = _write_many(tmp_path, "n.json", n_decided=4, n_ties=6, shift=-0.01)
    out = _main(capsys, monkeypatch, [base, new, "BASE", "NEW", "--model", "M"])
    bar = [l for l in out.splitlines() if l.startswith("sign-test bar")][0]
    # 4 wins of 10 is under the bar: the word is FAIL exactly as before...
    assert "FAIL" in bar
    # ...and the line says it is a pointer, with the decided count (ties are
    # the inert slice, so they are not decided).
    assert "[POINTER, not a gate: 4 decided < 8]" in bar
    engaged = [l for l in out.splitlines() if "engaged only" in l][0]
    assert "PASS" in engaged and "4 decided < 8" in engaged


def test_pointer_label_absent_at_eight_decided(tmp_path, capsys, monkeypatch):
    base = _write_many(tmp_path, "b.json", n_decided=8, n_ties=0, shift=0.0)
    new = _write_many(tmp_path, "n.json", n_decided=8, n_ties=0, shift=-0.01)
    out = _main(capsys, monkeypatch, [base, new, "BASE", "NEW", "--model", "M"])
    assert "POINTER" not in out
    assert compare_runs.pointer_label(8) == ""
    assert "7 decided < 8" in compare_runs.pointer_label(7)


# --------------------------------------------------------------------------
# The engaged slice is measured (CAMPAIGN_PLAN I031, H(1)): median relative
# change with a bootstrap CI over datasets, and per-seed agreement. Print-only.
# --------------------------------------------------------------------------
def _write_seeded(tmp_path, name, briers):
    """One run JSON: ``briers`` maps dataset -> {seed: brier}; binary sets at
    Brier ~0.2 stay clear of near-solved."""
    datasets = {ds: {"task": "binary"} for ds in briers}
    records = [{"dataset": ds, "model": "M", "seed": s, "fit_time": 1.0,
                "metrics": {"primary": 1.0 - b, "brier": b}}
               for ds, per_seed in briers.items() for s, b in per_seed.items()]
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datasets": datasets, "records": records}, fh)
    return path


def test_engaged_slice_median_and_seed_agreement(tmp_path, capsys, monkeypatch):
    base = _write_seeded(tmp_path, "b.json", {
        "una": {0: 0.20, 1: 0.20, 2: 0.20},   # every seed improves: unanimous
        "spl": {0: 0.20, 1: 0.20, 2: 0.20},   # 2 seeds improve, 1 regresses
        "tie": {0: 0.20, 1: 0.20, 2: 0.20},   # exact tie: inert, not engaged
    })
    new = _write_seeded(tmp_path, "n.json", {
        "una": {0: 0.19, 1: 0.19, 2: 0.19},   # -5% Brier = +5% relative
        "spl": {0: 0.19, 1: 0.19, 2: 0.23},   # mean 0.2033 -> -1.67% relative
        "tie": {0: 0.20, 1: 0.20, 2: 0.20},
    })
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M", "--metric", "brier"])
    slice_line = [l for l in out.splitlines() if "engaged slice" in l][0]
    # Two engaged datasets, relative changes +5% and -1.67%: the median is
    # their midpoint, and a CI over two points is bracketed by them.
    assert "engaged slice (2 scored)" in slice_line
    assert "+1.667%" in slice_line and "95% bootstrap CI" in slice_line
    assert "-1.667%..+5.000%" in slice_line
    assert "2 decided < 8" in slice_line          # the I030 label rides along
    seeds_line = [l for l in out.splitlines() if "per-seed agreement" in l][0]
    assert "1 of 2 engaged datasets unanimous" in seeds_line
    assert "1 split (spl)" in seeds_line
    # The tie is the inert slice, not an engaged dataset: it appears nowhere
    # in the engaged reads.
    assert "tie" not in slice_line and "tie" not in seeds_line
    # And the bar is untouched by any of it.
    assert "sign-test bar" in out


def test_engaged_slice_single_seed_says_so(tmp_path, capsys, monkeypatch):
    base = _write_seeded(tmp_path, "b.json", {"a": {0: 0.20}, "b": {0: 0.20}})
    new = _write_seeded(tmp_path, "n.json", {"a": {0: 0.19}, "b": {0: 0.21}})
    out = _main(capsys, monkeypatch,
                [base, new, "BASE", "NEW", "--model", "M", "--metric", "brier"])
    assert "per-seed agreement: single seed -- nothing to agree on" in out


def test_engaged_slice_helpers_are_deterministic_and_bracket_the_median():
    rels = [0.01, -0.02, 0.03, 0.005, -0.001]
    med, lo, hi = compare_runs.engaged_slice_stats(rels, n_boot=2000, seed=0)
    assert med == 0.005
    assert lo <= med <= hi
    assert compare_runs.engaged_slice_stats(rels, n_boot=2000, seed=0) == (med, lo, hi)
    assert compare_runs.engaged_slice_stats([]) is None
    # Agreement: unanimous needs every shared seed on the mean's side; a
    # dataset with one shared seed is skipped, not counted either way.
    sb = {"x": {0: 1.0, 1: 1.0}, "y": {0: 1.0, 1: 1.0}, "z": {0: 1.0}}
    sn = {"x": {0: 1.1, 1: 1.2}, "y": {0: 1.1, 1: 0.5}, "z": {0: 2.0}}
    assert compare_runs.seed_agreement(["x", "y", "z"], sb, sn) == (1, ["y"], 2)
