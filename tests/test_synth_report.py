"""synth_report.py must recover planted per-slice effects from results JSONs."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import synth_report


def _fake_run(tmp_path, name, primary_by_ds, metas):
    data = {
        "config": {"seeds": 1},
        "datasets": metas,
        "records": [
            {"dataset": ds, "model": "ChimeraBoost", "seed": 0,
             "metrics": {"primary": p, "brier": 0.2, "rmse": 1.0},
             "fit_time": 0.1, "best_iter": 10}
            for ds, p in primary_by_ds.items()
        ] + [
            {"dataset": ds, "model": "LightGBM", "seed": 0,
             "metrics": {"primary": p * 0.9, "brier": 0.25, "rmse": 1.1},
             "fit_time": 0.05, "best_iter": 10}
            for ds, p in primary_by_ds.items()
        ],
    }
    path = os.path.join(tmp_path, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _metas():
    metas = {}
    for i in range(8):
        deep = i < 4
        metas[f"syn:v1/{i:03d}"] = {
            "task": "binary", "n_train": 1500, "n_total": 2000, "n_features": 10,
            "has_cats": False,
            "synth": {"gen_version": "v1", "recipe_id": i, "task": "binary",
                      "n": 3000, "d": 10, "interaction_depth": 3 if deep else 1,
                      "cat_fraction": 0.0, "n_cat": 0, "max_cardinality": 0,
                      "irrelevant_fraction": 0.2, "noise_level": 0.3,
                      "missing_fraction": 0.0, "imbalance": 0.5,
                      "saturated": False, "func_dominant": "neural",
                      "bayes_brier": 0.15, "noise_sigma": None, "n_classes": 2},
        }
    return metas


def test_ab_attribution_recovers_planted_slice(tmp_path, capsys):
    metas = _metas()
    base = {ds: 0.70 for ds in metas}
    # plant: +5% only on the deep-interaction half
    new = {ds: (0.735 if metas[ds]["synth"]["interaction_depth"] >= 3 else 0.70)
           for ds in metas}
    pb = _fake_run(str(tmp_path), "base.json", base, metas)
    pn = _fake_run(str(tmp_path), "new.json", new, metas)

    m, per_b = synth_report.load_run(pb)
    _, per_n = synth_report.load_run(pn)
    rows = synth_report.ab_report(
        m, synth_report.primary_means(per_b, "ChimeraBoost"),
        synth_report.primary_means(per_n, "ChimeraBoost"))
    out = capsys.readouterr().out
    by_label = {r[0]: r for r in rows}
    label, n, w, l, t, mean_d, p = by_label["depth>=3"]
    assert (w, l) == (4, 0) and mean_d > 0.04
    label, n, w, l, t, mean_d, p = by_label["depth<=2"]
    assert (w, l) == (0, 0)
    assert "interaction_depth" in out


def test_model_filter_isolates_chimera(tmp_path):
    metas = _metas()
    base = {ds: 0.70 for ds in metas}
    pb = _fake_run(str(tmp_path), "b.json", base, metas)
    _, per = synth_report.load_run(pb)
    prim_c = synth_report.primary_means(per, "ChimeraBoost")
    prim_l = synth_report.primary_means(per, "LightGBM")
    assert all(abs(prim_c[ds] - 0.70) < 1e-12 for ds in prim_c)
    assert all(abs(prim_l[ds] - 0.63) < 1e-12 for ds in prim_l)
