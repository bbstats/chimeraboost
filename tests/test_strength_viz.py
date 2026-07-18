"""Strength-viz contracts (benchmarks/STRENGTH_VIZ_PLAN.md): the head-to-head
primary-metric machinery in summarize (primary scores, ranks with average
ties, win rates with ties = 1/2, near-solved exclusions, bootstrap CI) and
make_pareto's win-rate scoring + frontier. Deterministic fixtures only — no
benchmark runs, no network, no images.
"""
import os
import sys

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import summarize  # noqa: E402
import make_pareto  # noqa: E402


def _rec(ds, model, ft=1.0, **mt):
    return {"dataset": ds, "model": model, "seed": 0,
            "metrics": {"primary": 0.0, **mt}, "fit_time": ft}


def _data():
    """3 models x 4 datasets with known ranks/win rates.

    gr:reg1  scored regression: A < B < C (best NRMSE 0.1, kept)
    gr:reg0  near-solved regression (best NRMSE 0.005 < 2%): excluded
    gr:bin1  scored binary with an exact A==B Brier tie
    gr:bin0  near-solved-Brier binary (best < 1e-3): excluded
    Fit times make C the fastest everywhere: A 2x, B 4x, C 1x.
    """
    datasets = {
        "gr:reg1": {"task": "regression", "y_std": 10.0},
        "gr:reg0": {"task": "regression", "y_std": 100.0},
        "gr:bin1": {"task": "binary"},
        "gr:bin0": {"task": "binary"},
    }
    ft = {"A": 1.0, "B": 2.0, "C": 0.5}
    rmse1 = {"A": 1.0, "B": 2.0, "C": 3.0}
    rmse0 = {"A": 0.5, "B": 0.6, "C": 0.7}
    brier1 = {"A": 0.20, "B": 0.20, "C": 0.30}
    brier0 = {"A": 0.0005, "B": 0.0006, "C": 0.0007}
    records = []
    for m in "ABC":
        records.append(_rec("gr:reg1", m, ft[m], rmse=rmse1[m]))
        records.append(_rec("gr:reg0", m, ft[m], rmse=rmse0[m]))
        records.append(_rec("gr:bin1", m, ft[m], brier=brier1[m],
                            f1_macro=0.8, calibration_mcb=0.01))
        records.append(_rec("gr:bin0", m, ft[m], brier=brier0[m],
                            f1_macro=0.9, calibration_mcb=0.01))
    return {"config": {"seeds": 1}, "datasets": datasets, "records": records}


def test_primary_scores_exclusions():
    p = summarize.primary_scores(_data())
    assert set(p) == {"gr:reg1", "gr:bin1"}
    assert p["gr:reg1"] == {"A": 1.0, "B": 2.0, "C": 3.0}
    assert p["gr:bin1"] == {"A": 0.20, "B": 0.20, "C": 0.30}


def test_ranks_average_ties_and_mean_rank():
    p = summarize.primary_scores(_data())
    r = summarize.per_dataset_ranks(p)
    assert r["gr:reg1"] == {"A": 1.0, "B": 2.0, "C": 3.0}
    assert r["gr:bin1"] == {"A": 1.5, "B": 1.5, "C": 3.0}
    assert summarize.mean_rank(r) == {"A": 1.25, "B": 1.75, "C": 3.0}


def test_winrate_vs_field_and_rank_identity():
    p = summarize.primary_scores(_data())
    wr = summarize.winrate_vs_field(p)
    # A: wins both reg1 matchups + 1.5 of 2 on bin1 -> 3.5/4; C loses all.
    assert wr == {"A": 87.5, "B": 62.5, "C": 0.0}
    # With every model scored on every dataset, win rate IS mean rank
    # rescaled: (k - mean_rank) / (k - 1).
    mr = summarize.mean_rank(summarize.per_dataset_ranks(p))
    for m in wr:
        assert wr[m] == pytest.approx((3 - mr[m]) / 2 * 100.0)


def test_winrate_matrix_ties_half():
    p = summarize.primary_scores(_data())
    models, mat = summarize.winrate_matrix(p)
    assert models == ["A", "B", "C"]  # best-first by vs-field win rate
    idx = {m: i for i, m in enumerate(models)}
    assert mat[idx["A"]][idx["B"]] == 75.0   # win reg1, tie bin1
    assert mat[idx["B"]][idx["A"]] == 25.0
    assert mat[idx["A"]][idx["C"]] == 100.0
    assert mat[idx["C"]][idx["A"]] == 0.0
    assert all(mat[i][i] is None for i in range(3))
    # Row mean of the matrix == the vs-field axis scalar (complete data).
    wr = summarize.winrate_vs_field(p)
    for m in models:
        row = [v for v in mat[idx[m]] if v is not None]
        assert sum(row) / len(row) == pytest.approx(wr[m])
    assert summarize.n_tied_matchups(p) == 1


def test_bootstrap_ci_deterministic_and_bracketing():
    p = summarize.primary_scores(_data())
    ci1 = summarize.bootstrap_winrate_ci(p, n_boot=500, seed=0)
    ci2 = summarize.bootstrap_winrate_ci(p, n_boot=500, seed=0)
    assert ci1 == ci2
    wr = summarize.winrate_vs_field(p)
    for m, (lo, hi) in ci1.items():
        assert 0.0 <= lo <= wr[m] <= hi <= 100.0
    assert ci1["C"] == (0.0, 0.0)  # loses every matchup in every resample


def test_single_model_dataset_adds_no_matchups():
    data = _data()
    data["datasets"]["gr:bin2"] = {"task": "binary"}
    data["records"].append(_rec("gr:bin2", "A", 1.0, brier=0.1,
                                f1_macro=0.8, calibration_mcb=0.01))
    wr = summarize.winrate_vs_field(summarize.primary_scores(data))
    assert wr == {"A": 87.5, "B": 62.5, "C": 0.0}


def test_f1_never_moves_the_ranking():
    # D2: classification ranks on Brier only; F1 stays a table diagnostic.
    data = _data()
    for r in data["records"]:
        if "f1_macro" in r["metrics"]:
            r["metrics"]["f1_macro"] = 0.999 if r["model"] == "C" else 0.01
    wr = summarize.winrate_vs_field(summarize.primary_scores(data))
    assert wr == {"A": 87.5, "B": 62.5, "C": 0.0}


def test_score_models_and_winrate_frontier():
    scored, meta, primary = make_pareto.score_models(_data(), n_boot=200)
    assert meta["n_h2h"] == 2 and meta["n_ties"] == 1
    assert scored["A"]["winrate"] == 87.5
    assert scored["A"]["mean_rank"] == 1.25
    assert scored["C"]["slowdown"] == pytest.approx(1.0)
    front = make_pareto.pareto_frontier(scored, key="winrate")
    assert front == {"A", "C"}  # B slower than A and weaker -> dominated
    # The winrate table + matrix render without error and carry the numbers.
    txt = make_pareto.format_text(scored, meta, primary, metric="winrate")
    assert "87.5" in txt and "MeanRank" in txt and "vs field" in txt
