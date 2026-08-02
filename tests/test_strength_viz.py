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


# ---------------------------------------------------------------------------
# Skill-score axis (headline since 2026-08-02): Brier skill for classification,
# R2 for regression, split into two panels. Pinned here because it now decides
# what images/pareto.png shows.
# ---------------------------------------------------------------------------

def _skill_data():
    """Same shape as _data(), plus the class priors the BSS reference needs.

    gr:bin1 is a balanced binary set, so the no-skill Brier under the harness's
    sum-over-classes convention is sum_k p_k(1-p_k) = 0.5. gr:reg1 has
    y_std = 10, so an RMSE of 1.0 is R2 = 1 - (1/10)^2 = 0.99.
    """
    data = _data()
    data["datasets"]["gr:bin1"]["class_prior"] = [0.5, 0.5]
    data["datasets"]["gr:bin0"]["class_prior"] = [0.5, 0.5]
    return data


def test_skill_scores_match_their_definitions():
    skill = make_pareto.skill_scores(_skill_data())
    # R2 = 1 - (rmse / y_std)^2, averaged over the two regression datasets.
    # A: reg1 1 - (1.0/10)^2 = 0.99 ; reg0 1 - (0.5/100)^2 = 0.999975
    assert skill["regression"]["A"]["strength"] == pytest.approx(
        (0.99 + 0.999975) / 2)
    # BSS = 1 - brier / 0.5. A: bin1 1 - 0.20/0.5 = 0.6 ; bin0 1 - 0.0005/0.5
    assert skill["classification"]["A"]["strength"] == pytest.approx(
        (0.6 + (1 - 0.0005 / 0.5)) / 2)


def test_skill_axis_keeps_near_solved_datasets():
    """The whole point of a bounded score: no exclusion, nothing to explode."""
    skill = make_pareto.skill_scores(_skill_data())
    assert skill["regression"]["A"]["n"] == 2      # reg1 AND near-solved reg0
    assert skill["classification"]["A"]["n"] == 2  # bin1 AND near-solved bin0


def test_skill_slowdown_is_per_task():
    """Each panel's x-axis is scored on its own datasets only."""
    skill = make_pareto.skill_scores(_skill_data())
    for task in ("classification", "regression"):
        assert skill[task]["C"]["slowdown"] == pytest.approx(1.0)  # fastest
        assert skill[task]["A"]["slowdown"] == pytest.approx(2.0)
        assert skill[task]["B"]["slowdown"] == pytest.approx(4.0)


def test_skill_is_not_field_relative():
    """Dropping an arm must not move anyone else -- the property win rate lacks."""
    full = make_pareto.skill_scores(_skill_data())
    data = _skill_data()
    data["records"] = [r for r in data["records"] if r["model"] != "B"]
    pruned = make_pareto.skill_scores(data)
    for task in ("classification", "regression"):
        assert (pruned[task]["A"]["strength"]
                == pytest.approx(full[task]["A"]["strength"]))


def test_skill_degrades_without_class_prior():
    """Pre-0.30.0 runs carry no class_prior; classification drops out rather
    than silently inventing a reference, and regression still scores."""
    data = _data()                       # no class_prior anywhere
    skill = make_pareto.skill_scores(data)
    assert skill["classification"] == {}
    assert skill["regression"]["A"]["n"] == 2


def test_skill_frontier_and_table():
    skill = make_pareto.skill_scores(_skill_data())
    front = make_pareto.pareto_frontier(skill["regression"], key="strength")
    assert front == {"A", "C"}           # B is slower than A and weaker
    txt = make_pareto.format_skill_text(skill, "# fixture")
    assert "Classification" in txt and "Regression" in txt and "Pareto" in txt


def test_short_names_never_collide():
    """m[:5] used to map every unlisted ChimeraBoost arm to "Chime", so two
    different models shared a chart label and a matrix column header."""
    arms = ["ChimeraBoost", "ChimeraBoostNoRefit", "ChimeraBoostOneLin",
            "ChimeraBoostEns5", "ChimeraBoostEns8", "ChimeraBoostEns5RM",
            "ChimeraBoostSel25", "ChimeraBoostFlatLR", "ChimeraBoostRefit",
            "CatBoost", "LightGBM", "sklearn_HGB"]
    shorts = [make_pareto._short(a) for a in arms]
    assert len(set(shorts)) == len(arms), dict(zip(arms, shorts))


def test_r2_prefers_the_test_split_target_scale():
    """y_std_test is where RMSE is measured; y_std (full target) is the
    fallback for runs recorded before that field existed."""
    data = _skill_data()
    only_full = make_pareto.skill_scores(data)["regression"]["A"]["strength"]
    for ds in ("gr:reg1", "gr:reg0"):
        data["datasets"][ds]["y_std_test"] = data["datasets"][ds]["y_std"] * 2
    with_test = make_pareto.skill_scores(data)["regression"]["A"]["strength"]
    assert with_test != pytest.approx(only_full)
    # rmse 1.0 / (y_std_test 20) -> 1 - 0.0025 ; rmse 0.5 / 200 -> 1 - 6.25e-6
    assert with_test == pytest.approx(((1 - 0.0025) + (1 - 6.25e-06)) / 2)
