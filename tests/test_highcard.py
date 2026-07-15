"""HC suite contracts: registration, frozen-list<->doc agreement, the sealed-
holdout overlap regression test, loader smoke, and summarize's multiclass columns.

The overlap test embeds the TabArena-51 dataset NAMES (names only, no results of
any kind — using the membership list to AVOID contamination is the sanctioned use
per benchmarks/HIGHCARD_PLAN.md and the sealed-holdout vow). Source of the names:
tabarena/nips2025_utils/metadata/curated_tabarena_dataset_metadata.csv.
"""
import os
import re
import sys

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import run_benchmarks as rb  # noqa: E402
import summarize  # noqa: E402

PLAN_MD = os.path.join(BENCH, "HIGHCARD_PLAN.md")

# TabArena 51 (openml_dataset_name column). NAMES ONLY — never any result.
TABARENA_51 = [
    "airfoil_self_noise", "Amazon_employee_access", "anneal",
    "Another-Dataset-on-used-Fiat-500", "APSFailure", "bank-marketing",
    "Bank_Customer_Churn", "Bioresponse", "blood-transfusion-service-center",
    "churn", "coil2000_insurance_policies", "concrete_compressive_strength",
    "credit-g", "credit_card_clients_default", "customer_satisfaction_in_airline",
    "diabetes", "Diabetes130US", "diamonds", "E-CommereShippingData",
    "Fitness_Club", "Food_Delivery_Time", "GiveMeSomeCredit",
    "hazelnut-spread-contaminant-detection", "healthcare_insurance_expenses",
    "heloc", "hiva_agnostic", "houses", "HR_Analytics_Job_Change_of_Data_Scientists",
    "in_vehicle_coupon_recommendation", "Is-this-a-good-customer",
    "kddcup09_appetency", "Marketing_Campaign", "maternal_health_risk",
    "miami_housing", "MIC", "NATICUSdroid", "online_shoppers_intention",
    "physiochemical_protein", "polish_companies_bankruptcy", "qsar-biodeg",
    "QSAR-TID-11", "QSAR_fish_toxicity", "SDSS17", "seismic-bumps", "splice",
    "students_dropout_and_academic_success", "superconductivity",
    "taiwanese_bankruptcy_prediction", "website_phishing", "wine_quality", "jm1",
]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _name_hit(name, pool):
    """(kind, matched) if `name` exact- or substring-matches any pool name
    (shorter side >= 6 chars to avoid generic-token false hits), else None."""
    n = _norm(name)
    for other in pool:
        o = _norm(other)
        if n == o:
            return ("exact", other)
        short, long_ = (n, o) if len(n) <= len(o) else (o, n)
        if len(short) >= 6 and short in long_:
            return ("contains", other)
    return None


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def test_registration_idempotent_and_tasks():
    rb._add_highcard_datasets()
    keys = [k for k in rb.DATASETS if k.startswith("hc:")]
    rb._add_highcard_datasets()  # second call must not duplicate
    keys2 = [k for k in rb.DATASETS if k.startswith("hc:")]
    assert keys == keys2
    assert len(keys) == len(rb.HC_DATASETS) >= 12
    for name, spec in rb.HC_DATASETS.items():
        key = f"hc:{name}"
        assert key in rb.DATASETS
        assert rb._task_of(key) == spec["task"] == rb.HC_TASKS[key]
        assert spec["task"] in ("binary", "multiclass", "regression")


def test_composition_meets_plan_targets():
    tasks = [s["task"] for s in rb.HC_DATASETS.values()]
    assert tasks.count("regression") >= 2          # >= 2 reg-with-cats
    assert tasks.count("multiclass") >= 3          # multiclass columns land
    assert len(rb.HC_DATASETS) >= 12


# --------------------------------------------------------------------------
# frozen list <-> doc agreement
# --------------------------------------------------------------------------
def _parse_frozen_doc():
    """Rows of the form `| hc:<name> | <id> | <task> | ...` in HIGHCARD_PLAN.md."""
    out = {}
    with open(PLAN_MD, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\|\s*hc:([\w.\-]+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), m.group(3))
    return out


def test_frozen_matches_doc():
    doc = _parse_frozen_doc()
    code = {name: (spec["data_id"], spec["task"])
            for name, spec in rb.HC_DATASETS.items()}
    assert doc, "no frozen `| hc:... |` table rows found in HIGHCARD_PLAN.md"
    assert doc == code, (
        "HC_DATASETS and the HIGHCARD_PLAN.md frozen table disagree — "
        f"only in doc: {set(doc) - set(code)}; only in code: {set(code) - set(doc)}; "
        f"mismatched: {{k: (doc[k], code[k]) for k in set(doc)&set(code) if doc[k]!=code[k]}}")


# --------------------------------------------------------------------------
# sealed-holdout overlap regression test (the hard gate)
# --------------------------------------------------------------------------
def test_no_suite_overlap():
    grinsztajn = [n for names in rb.GRINSZTAJN_DATASETS.values() for n in names]
    gate_names = list(rb.OPENML_SUITE)
    gate_ids = {spec["data_id"] for spec in rb.OPENML_SUITE.values()}
    pmlb = [n for items in rb.PMLB_DATASETS.values() for n, _ in items]

    failures = []
    for name, spec in rb.HC_DATASETS.items():
        did = spec["data_id"]
        if did in gate_ids:
            failures.append(f"{name}: OpenML id {did} is in the gate suite")
        for tag, pool in (("TabArena", TABARENA_51), ("Grinsztajn", grinsztajn),
                          ("gate", gate_names), ("PMLB", pmlb)):
            hit = _name_hit(name, pool)
            if hit:
                failures.append(f"{name}: {tag} {hit[0]} match '{hit[1]}'")
    assert not failures, "HC suite overlaps a sealed/decision suite:\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# loader smoke (2 smallest sets); skips if OpenML/cache is unavailable
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["eucalyptus", "Moneyball"])
def test_loader_smoke_smallest(name):
    import numpy as np
    spec = rb.HC_DATASETS[name]
    try:
        X, y, cat, task = rb.DATASETS[f"hc:{name}"](1.0, np.random.default_rng(0))
    except Exception as e:  # no network / no A: cache -> not a code failure
        pytest.skip(f"OpenML fetch unavailable for {name}: {e}")
    assert X.shape[0] == len(y) > 0
    assert task == spec["task"]
    assert cat, f"{name} should expose categorical columns"
    assert X.dtype == object
    for j in cat[:3]:
        assert all(isinstance(v, str) for v in X[:20, j])


def test_id_columns_dropped():
    # employee_salaries carries a `full_name` column (99.9% unique). The HC
    # builder must drop near-unique categoricals, so no surviving cat column has
    # cardinality > _HIGHCARD_ID_FRAC * n.
    import numpy as np
    try:
        X, y, cat, task = rb.DATASETS["hc:employee_salaries"](
            1.0, np.random.default_rng(0))
    except Exception as e:
        pytest.skip(f"OpenML fetch unavailable: {e}")
    n = X.shape[0]
    for j in cat:
        card = len({v for v in X[:, j] if v == v})
        assert card <= rb._HIGHCARD_ID_FRAC * n, (
            f"cat col {j} card {card} exceeds id-drop threshold "
            f"{rb._HIGHCARD_ID_FRAC} * {n}")


# --------------------------------------------------------------------------
# summarize: multiclass columns render only when multiclass records exist
# --------------------------------------------------------------------------
def _mini(with_mc):
    def rec(ds, model, **mt):
        return {"dataset": ds, "model": model, "seed": 0,
                "metrics": {"primary": 0.0, **mt}, "fit_time": mt.get("_ft", 1.0)}
    datasets = {"gr:bin1": {"task": "binary"}}
    records = [rec("gr:bin1", "ChimeraBoost", f1_macro=.9, brier=.2, calibration_mcb=.01),
               rec("gr:bin1", "CatBoost", f1_macro=.88, brier=.22, calibration_mcb=.02, _ft=2.0)]
    if with_mc:
        datasets["hc:mc1"] = {"task": "multiclass"}
        records += [rec("hc:mc1", "ChimeraBoost", f1_macro=.7, brier=.5, calibration_mcb=.03),
                    rec("hc:mc1", "CatBoost", f1_macro=.75, brier=.45, calibration_mcb=.02, _ft=2.0)]
    return {"config": {"seeds": 1}, "datasets": datasets, "records": records}


def test_summarize_multiclass_columns_conditional():
    no_mc = summarize.format_table(_mini(False))
    assert "Multi F1%" not in no_mc and "Multi Brier%" not in no_mc

    with_mc = summarize.format_table(_mini(True))
    assert "Multi F1%" in with_mc and "Multi Brier%" in with_mc
    # every table row is the same width (alignment holds with the extra columns)
    rows = [l for l in with_mc.splitlines()
            if l.startswith(("Model", "ChimeraBoost", "CatBoost"))]
    assert len({len(r) for r in rows}) == 1

    # aggregate exposes the multiclass columns; make_pareto blend ignores them
    cols, _ = summarize.aggregate(_mini(True))
    assert "Multi F1%" in cols and "Multi Brier%" in cols
