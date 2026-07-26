"""Shared machinery for the sealed/decision-suite overlap gate.

Three callers assert the same rule and must not drift apart:

  tests/test_highcard.py   HC must not overlap a sealed holdout or decision suite
  tests/test_public.py     the same for the sealed public suite (plus HC itself)
  benchmarks/public_audit.py   applies the rule while shortlisting candidates

TABARENA_51 is dataset NAMES ONLY -- never a result of any kind. Using the
membership list to AVOID contamination is the sanctioned use per
benchmarks/HIGHCARD_PLAN.md and the sealed-holdout vow. Source of the names:
tabarena/nips2025_utils/metadata/curated_tabarena_dataset_metadata.csv, column
`openml_dataset_name`.
"""
import os
import re
import sys

_BENCH = os.path.dirname(os.path.abspath(__file__))

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


def norm(s):
    """Lowercase and strip everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def name_hit(name, pool):
    """(kind, matched) if `name` exact- or substring-matches any pool name
    (shorter side >= 6 chars to avoid generic-token false hits), else None."""
    n = norm(name)
    for other in pool:
        o = norm(other)
        if n == o:
            return ("exact", other)
        short, long_ = (n, o) if len(n) <= len(o) else (o, n)
        if len(short) >= 6 and short in long_:
            return ("contains", other)
    return None


def _rb():
    """Import run_benchmarks lazily (it pulls numpy/sklearn)."""
    if _BENCH not in sys.path:
        sys.path.insert(0, _BENCH)
    import run_benchmarks
    return run_benchmarks


def exclusion_pools(include_hc=True, extra=()):
    """The pools a candidate suite must not touch, as [(tag, ids, names), ...].

    `ids` is a set of OpenML dataset ids and may be empty: Grinsztajn loads from
    a HuggingFace mirror and PMLB from a Git-LFS URL, so neither registry
    carries ids, and TABARENA_51 is deliberately names-only. Name matching is
    therefore the load-bearing check for those three.

    include_hc adds the HC decision suite, checked by id AND name -- correct for
    the public suite, wrong for HC's own test (a suite cannot overlap itself).
    `extra` takes further (tag, ids, names) triples, e.g. a doc-sourced list of
    datasets already consumed elsewhere.
    """
    rb = _rb()
    pools = [
        ("TabArena", set(), list(TABARENA_51)),
        ("Grinsztajn", set(),
         [n for names in rb.GRINSZTAJN_DATASETS.values() for n in names]),
        ("gate", {s["data_id"] for s in rb.OPENML_SUITE.values()},
         list(rb.OPENML_SUITE)),
        ("PMLB", set(),
         [n for items in rb.PMLB_DATASETS.values() for n, _ in items]),
    ]
    if include_hc:
        pools.append(("HC", {s["data_id"] for s in rb.HC_DATASETS.values()},
                      list(rb.HC_DATASETS)))
    pools.extend((tag, set(ids), list(names)) for tag, ids, names in extra)
    return pools


def overlap_failures(name, data_id, pools):
    """Human-readable reasons `name`/`data_id` may not join a new suite ([] = clean)."""
    out = []
    for tag, ids, names in pools:
        if data_id is not None and data_id in ids:
            out.append(f"{name}: OpenML id {data_id} is in the {tag} suite")
        hit = name_hit(name, names)
        if hit:
            out.append(f"{name}: {tag} {hit[0]} match '{hit[1]}'")
    return out
