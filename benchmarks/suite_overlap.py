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


# Datasets spent elsewhere that no registry pool catches by name. Two kinds:
# well-known sets already consumed by a suite whose own list spells them
# differently, and verified aliases found by reading columns rather than names.
# This pool is why retiring the OpenML gate did not un-protect `adult` and
# friends -- they were previously covered only by the gate's id set.
CONSUMED_ELSEWHERE = [
    # large sets already inside Grinsztajn / TabArena / HC under other spellings
    "covertype", "Higgs", "Allstate_Claims_Severity", "road-safety",
    "nyc-taxi-green-dec-2016", "diamonds", "house_sales", "Airlines_DepDelay_1M",
    "MiniBooNE", "jannis", "albert", "delays_zurich_transport", "medical_charges",
    "particulate-matter-ukair-2017", "seattlecrime6", "SGEMM_GPU_kernel_performance",
    "Diabetes130US", "superconduct", "APSFailure", "kddcup09_appetency", "adult",
    "electricity", "letter",
    # verified aliases -- each caught by reading columns, not by the matcher:
    #   uci_diabetes_p (42106) is Diabetes130US, same 101,766 rows
    #   Winedata (43651) is the wine-reviews data that is hc:wine-reviews (41275)
    #   Amazon_access (4135) is TabArena's Amazon_employee_access
    "uci_diabetes_p", "Winedata", "Amazon_access",
    # concatenations that smuggle a consumed dataset in as a column block:
    #   AirlinesCodrnaAdult (1240) contains adult
    #   CovPokElec (149) contains covertype and electricity
    "AirlinesCodrnaAdult", "CovPokElec",
]

# Re-uploads of datasets ALREADY IN the public suite. These are discovery-time
# filters only and must NOT join the exclusion pool: `rossmann_store_sales` is a
# frozen member, and the substring matcher would flag it against its own
# re-upload and fail the overlap test.
PUB_REUPLOADS = ["hcdr_main", "rossmann_store_sales_processed"]


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

    The OpenML one-shot gate is NOT a pool: it was retired 2026-07-27 (see
    OPENML_SUITE) and no longer needs protecting from contamination. Its
    datasets that mattered are covered anyway -- eight were exact Grinsztajn
    members and four are in TabArena, so they still fail those pools.

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
        ("PMLB", set(),
         [n for items in rb.PMLB_DATASETS.values() for n, _ in items]),
        ("consumed", set(), list(CONSUMED_ELSEWHERE)),
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
