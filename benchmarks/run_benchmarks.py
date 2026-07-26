"""ChimeraBoost benchmark harness.

Runs ChimeraBoost against whatever competitors are installed (scikit-learn
HistGradientBoosting is always available; CatBoost, XGBoost and LightGBM are
auto-detected and skipped if absent) across a fixed suite of regression and
classification tasks, including categorical-heavy ones.

Every task is run over multiple seeds and reported as mean +/- std so that a
real improvement can be told apart from noise. This is the tool we use to
decide whether any future change (ordered boosting, feature combinations, ...)
actually helps before it goes in.

Usage:
    python benchmarks/run_benchmarks.py                 # default scale
    python benchmarks/run_benchmarks.py --scale 3       # ~3x bigger datasets
    python benchmarks/run_benchmarks.py --seeds 10      # more seeds
    python benchmarks/run_benchmarks.py --only classification
    python benchmarks/run_benchmarks.py --threads 8     # ChimeraBoost threads
"""

import argparse
import json as _json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402  (sibling module; scoring shared with the charts)

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, f1_score
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


# --------------------------------------------------------------------------
# Optional competitors: detected once, skipped silently if not installed.
# --------------------------------------------------------------------------
def _detect():
    have = {}
    try:
        import catboost  # noqa
        have["catboost"] = True
    except Exception:
        have["catboost"] = False
    try:
        import xgboost  # noqa
        have["xgboost"] = True
    except Exception:
        have["xgboost"] = False
    try:
        import lightgbm  # noqa
        have["lightgbm"] = True
    except Exception:
        have["lightgbm"] = False
    return have


HAVE = _detect()


# --------------------------------------------------------------------------
# Dataset builders. Each returns (X, y, cat_features, task).
# `scale` multiplies the synthetic sample counts.
# --------------------------------------------------------------------------
def _ds_diabetes(scale, rng):
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    return X, y, None, "regression"


def _ds_friedman(scale, rng):
    from sklearn.datasets import make_friedman1
    n = int(2000 * scale)
    X, y = make_friedman1(n_samples=n, noise=1.0, random_state=int(rng.integers(1e9)))
    return X, y, None, "regression"


def _ds_synthetic_reg(scale, rng):
    from sklearn.datasets import make_regression
    n = int(8000 * scale)
    X, y = make_regression(n_samples=n, n_features=30, n_informative=20,
                           noise=20.0, random_state=int(rng.integers(1e9)))
    return X, y, None, "regression"


def _ds_breast_cancer(scale, rng):
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, None, "binary"


def _ds_wine(scale, rng):
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    return X, y, None, "multiclass"


def _ds_categorical_binary(scale, rng):
    """High-cardinality + low-cardinality categoricals driving a binary target."""
    n = int(6000 * scale)
    hi = rng.integers(0, 150, n)              # high-card categorical
    lo = rng.integers(0, 5, n)                # low-card categorical
    num = rng.normal(size=(n, 3))
    hi_eff = rng.normal(0, 1.5, 150)[hi]
    lo_eff = np.array([-1.0, -0.3, 0.2, 0.8, 1.5])[lo]
    logit = hi_eff + lo_eff + 0.6 * num[:, 0] - 0.4 * num[:, 1] + rng.normal(0, 1, n)
    y = (logit > np.median(logit)).astype(int)
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = np.array([f"h{c}" for c in hi], dtype=object)
    X[:, 1] = np.array([f"l{c}" for c in lo], dtype=object)
    X[:, 2:] = num
    return X, y, [0, 1], "binary"


def _ds_categorical_multiclass(scale, rng):
    n = int(5000 * scale)
    region = rng.choice(["N", "S", "E", "W"], n)
    tier = rng.choice(["a", "b", "c"], n)
    num = rng.normal(size=(n, 3))
    score = (np.select([region == "N", region == "S", region == "E"],
                       [1.5, -1.0, 0.3], 0.0)
             + np.select([tier == "a", tier == "b"], [1.0, -0.5], 0.0)
             + 0.5 * num[:, 0] + rng.normal(0, 0.5, n))
    y = np.digitize(score, [-0.5, 1.0])
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = region
    X[:, 1] = tier
    X[:, 2:] = num
    return X, y, [0, 1], "multiclass"


DATASETS = {
    "diabetes": _ds_diabetes,
    "friedman1": _ds_friedman,
    "synthetic_reg": _ds_synthetic_reg,
    "breast_cancer": _ds_breast_cancer,
    "wine": _ds_wine,
    "cat_binary": _ds_categorical_binary,
    "cat_multiclass": _ds_categorical_multiclass,
}

# Task type per synthetic dataset, so selection/filtering needn't build them.
SYNTH_TASKS = {
    "diabetes": "regression", "friedman1": "regression",
    "synthetic_reg": "regression", "breast_cancer": "binary",
    "wine": "multiclass", "cat_binary": "binary", "cat_multiclass": "multiclass",
}


# --------------------------------------------------------------------------
# Variant families (issue #37). A variant is a derived VIEW of a parent dataset
# that probes one named failure regime. Keys are "<parent>@<variant>", so every
# downstream consumer (summarize, compare_runs, the charts) keeps working, and
# summarize.stratum_of puts each family in its own stratum -- a variant reuses
# its parent's rows, so pooling the two would inflate a sign test's sample size.
#
#   @sus25 / @sus50 -- supplemental under-sampling: train on 25% / 50% of the
#       training rows, TEST SET UNCHANGED. Isolates data volume: the twin and
#       its parent are scored on identical rows, so the pair reads as a learning
#       curve, and the small arm doesn't gain variance from a smaller test set.
#       Datasets grow over time; this asks how the models rank earlier in that life.
#   @time -- temporal split: rows ordered by a real observation timestamp, train
#       on earlier rows and test on later ones. Isolates distribution shift, the
#       most common way a deployed model fails, and the one regime every other
#       split in this harness is blind to (they are all random).
#
# The two are kept apart on purpose: "earlier and smaller" is their intersection,
# and blurring them would make a result impossible to attribute to either cause.
VARIANT_SEP = "@"
SUS_FRACTIONS = {"sus25": 0.25, "sus50": 0.50}

# Frozen selection, per suite, over that suite's sorted keys: every 5th dataset
# gets a 25% twin (20% of the suite) and every 10th at offset 3 gets a 50% twin
# (10%). The offset keeps the two picks disjoint (i % 10 == 3 implies i % 5 != 0).
# Deterministic and spread across the suite rather than clustered; the resulting
# assignment is committed in benchmarks/VARIANTS.md and locked by a test.
SUS_STRIDE_25, SUS_STRIDE_50, SUS_OFFSET_50 = 5, 10, 3

# Rolling origin: seed s trains on the earliest cut and tests the window that
# follows. A single fixed cut would be deterministic, and since every model runs
# with random_state=0 all seeds would reproduce one number exactly -- replication
# in name only. Three cuts give three genuinely different reads.
TEMPORAL_CUTS = (0.65, 0.70, 0.75)
TEMPORAL_TEST_FRAC = 0.25

# Datasets with a genuine observation timestamp, audited by hand: the column
# must be a real time of record (not merely year-shaped), must not be the
# target, and must survive the near-unique-categorical drop. Verified against
# the cached frames. Grinsztajn is absent by nature -- the HuggingFace mirror
# ships pre-transformed numeric CSVs with no recoverable time column, so this
# regime simply has no expression there, and we declare that rather than fake it.
TEMPORAL_COLUMNS = {
    "hc:kick": "PurchDate",                  # epoch seconds, vehicle purchase
    "hc:sf-police-incidents": "Year",        # incident year (unordered category)
    "hc:Traffic_violations": "Year",         # violation year (0.6% unparseable)
    "hc:house_prices_nominal": "YrSold",     # year the sale closed (Ames)
    "hc:Moneyball": "Year",                  # baseball season
    "hc:employee_salaries": "date_first_hired",   # MM/DD/YYYY string
    "hc:eucalyptus": "Year",                 # measurement year
}


def _time_sort_key(series):
    """Sortable key for a time column, or None if it can't be ordered.

    Coercion order matters. Numeric first, so year columns and epoch seconds
    sort correctly whether they arrive as int, float, or an UNORDERED pandas
    category (sf-police ships Year as a category whose order is not guaranteed
    chronological). Datetime second, for real date strings -- employee_salaries
    stores MM/DD/YYYY, where a lexicographic sort would order by month.
    """
    import pandas as pd
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:
        return num
    try:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        # format="mixed" needs pandas >= 2.0; the dev extras declare >= 1.3, so
        # fall back to plain inference rather than failing on an older resolve.
        dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.5:
        return dt
    return None


def _sus_assignment(keys):
    """{dataset key: sus variant} for one suite, from its sorted key order."""
    out = {}
    for i, k in enumerate(sorted(keys)):
        if i % SUS_STRIDE_25 == 0:
            out[k] = "sus25"
        elif i % SUS_STRIDE_50 == SUS_OFFSET_50:
            out[k] = "sus50"
    return out


def _add_variant_datasets(base_keys):
    """Register @sus and @time twins for `base_keys`. Idempotent."""
    by_suite = defaultdict(list)
    for k in base_keys:
        if VARIANT_SEP in k:
            continue
        by_suite[k.split(":", 1)[0] if ":" in k else ""].append(k)

    for keys in by_suite.values():
        for key, variant in _sus_assignment(keys).items():
            vkey = f"{key}{VARIANT_SEP}{variant}"
            if vkey not in DATASETS:
                # Same builder as the parent; the shrink happens after the split
                # in _run_seed_task so the twin keeps the parent's test rows.
                DATASETS[vkey] = DATASETS[key]

    for key, col in TEMPORAL_COLUMNS.items():
        if key not in DATASETS:
            continue
        vkey = f"{key}{VARIANT_SEP}time"
        if vkey in DATASETS:
            continue
        # Both suites load real OpenML frames through the same builder, so both
        # can carry a temporal twin. Grinsztajn cannot (see VARIANTS.md).
        if key.startswith("hc:"):
            spec, cap = HC_DATASETS[key[len("hc:"):]], _HIGHCARD_MAX_ROWS
        elif key.startswith("pub:"):
            spec, cap = PUBLIC_DATASETS[key[len("pub:"):]], _PUBLIC_MAX_ROWS
        else:
            raise ValueError(
                "temporal variants need a builder that keeps column names; "
                f"{key!r} is not from the hc: or pub: suites")
        DATASETS[vkey] = _make_highcard_builder(spec, time_col=col, max_rows=cap)


def _task_of(ds_name):
    """Task type of a dataset by name, without building it."""
    ds_name = ds_name.split(VARIANT_SEP, 1)[0]     # variants inherit the parent's task
    if ds_name.startswith("oml:"):
        return OPENML_SUITE[ds_name[4:]]["task"]
    if ds_name.startswith("gr:"):
        return GRINSZTAJN_TASKS[ds_name]
    if ds_name.startswith("pm:"):
        return PMLB_TASKS[ds_name]
    if ds_name.startswith("hc:"):
        return HC_TASKS[ds_name]
    if ds_name.startswith("pub:"):
        return PUBLIC_TASKS[ds_name]
    if ds_name.startswith("syn:"):
        return SYN_TASKS[ds_name]
    return SYNTH_TASKS[ds_name]


# --------------------------------------------------------------------------
# Real external datasets via OpenML (the standard tabular-ML benchmark repo).
# These are fetched on demand with --openml and cached by sklearn. They exist
# so that decisions about defaults rest on many real datasets, not a handful of
# synthetic ones hand-picked here. Each entry: (openml_name_or_id, task,
# cat_feature_indices_or_None). Categoricals are auto-detected from dtype when
# the index list is "auto".
#
# This list is intentionally broad and editable -- add the datasets you care
# about. IDs are OpenML dataset IDs (stable); names can drift, so IDs preferred.
# --------------------------------------------------------------------------
OPENML_SUITE = {
    # classification (binary)
    "credit-g":      dict(data_id=31,    task="binary",     cats="auto"),
    "adult":         dict(data_id=1590,  task="binary",     cats="auto"),
    "bank-marketing":dict(data_id=1461,  task="binary",     cats="auto"),
    "kc1":           dict(data_id=1067,  task="binary",     cats=None),
    "phoneme":       dict(data_id=1489,  task="binary",     cats=None),
    "electricity":   dict(data_id=151,   task="binary",     cats=None),
    "magic":         dict(data_id=1120,  task="binary",     cats=None),
    "spambase":      dict(data_id=44,    task="binary",     cats=None),
    "kc2":           dict(data_id=1063,  task="binary",     cats=None),
    "sick":          dict(data_id=38,    task="binary",     cats="auto"),
    "mushroom":      dict(data_id=24,    task="binary",     cats="auto"),
    "kr-vs-kp":      dict(data_id=3,     task="binary",     cats="auto"),
    # high-cardinality real categoricals -- where one-hot vs ordered TS and fresh
    # permutations actually differ (the CatBoost-gap research targets). Added for
    # the categorical tier (Part B); all-categorical, no pre-encoding.
    "Amazon_access": dict(data_id=4135,  task="binary",     cats="auto"),
    # classification (multiclass)
    "vehicle":       dict(data_id=54,    task="multiclass", cats=None),
    "segment":       dict(data_id=40984, task="multiclass", cats=None),
    "optdigits":     dict(data_id=28,    task="multiclass", cats=None),
    "car":           dict(data_id=40975, task="multiclass", cats="auto"),
    "splice":        dict(data_id=46,    task="multiclass", cats="auto"),
    "nursery":       dict(data_id=26,    task="multiclass", cats="auto"),
    "satimage":      dict(data_id=182,   task="multiclass", cats=None),
    "pendigits":     dict(data_id=32,    task="multiclass", cats=None),
    "letter":        dict(data_id=6,     task="multiclass", cats=None),
    # regression
    "cpu_act":       dict(data_id=197,   task="regression", cats=None),
    "wine_quality":  dict(data_id=287,   task="regression", cats=None),
    "boston":        dict(data_id=531,   task="regression", cats=None),
    "elevators":     dict(data_id=216,   task="regression", cats=None),
    "ailerons":      dict(data_id=296,   task="regression", cats=None),
    "abalone":       dict(data_id=183,   task="regression", cats="auto"),
    "house_16H":     dict(data_id=574,   task="regression", cats=None),
}


def _is_categorical_dtype(dtype):
    """True for a pandas dtype that should be treated as categorical. pandas 3.0
    returns free-text columns as the new "str" dtype (repr "str"); object,
    category, and pyarrow "string[...]" all mean categorical too. Missing "str"
    silently routed text columns to the numeric branch -> astype(float) crash on
    high-card free-text datasets."""
    s = str(dtype).lower()
    return s in ("category", "object", "str") or s.startswith("string")


def _frame_to_dataset(X_df, y, cats, task):
    """Turn a (features DataFrame, target Series) into (X, y, cat_idx, task).
    Shared by the OpenML and Grinsztajn/HuggingFace builders.

    `cats` is "auto" (detect object/category/string columns), an explicit index
    list, or None. Categorical NaNs become the "__nan__" string: CatBoost rejects
    float NaN in cat_features, and ChimeraBoost maps "__nan__" to its missing
    bucket, so both see missing the same way. Numerics stay float.
    """
    if cats == "auto":
        cat_idx = [i for i, c in enumerate(X_df.columns)
                   if _is_categorical_dtype(X_df[c].dtype)]
    else:
        cat_idx = cats

    if task == "regression":
        y = y.astype(float).to_numpy()
    else:
        y = y.astype("category").cat.codes.to_numpy()

    if cat_idx:
        import pandas as pd
        cat_cols = set(cat_idx)
        cols = []
        for i, c in enumerate(X_df.columns):
            s = X_df[c]
            if i in cat_cols:
                cols.append(s.astype(object).where(s.notna(), "__nan__"))
            else:
                cols.append(s.astype(float))
        X = pd.concat(cols, axis=1).to_numpy(dtype=object)
    else:
        X = X_df.to_numpy(dtype=float)
    return X, y, (cat_idx or None), task


def _make_openml_builder(spec):
    """Build a dataset-builder closure for one OpenML spec (fetched by data_id)."""
    def builder(scale, rng):
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=spec["data_id"], as_frame=True)
        X_df = ds.frame.drop(columns=[ds.target.name])
        return _frame_to_dataset(X_df, ds.target, spec["cats"], spec["task"])
    return builder


def _add_openml_datasets():
    for name, spec in OPENML_SUITE.items():
        DATASETS[f"oml:{name}"] = _make_openml_builder(spec)


# The Grinsztajn et al. 2022 tabular benchmark ("Why do tree-based models still
# outperform deep learning on tabular data?"), the standard reference for this
# question. Loaded from the official inria-soda HuggingFace mirror (the exact
# transformed CSVs from the paper) rather than OpenML's flaky `study` API, so the
# dataset membership is hardcoded below and only the working CSV download is
# needed. Binary-classification + regression only (no multiclass). The target is
# always the last column. Folder -> (task, has-categoricals).
GRINSZTAJN_HF = ("https://huggingface.co/datasets/inria-soda/tabular-benchmark/"
                 "resolve/main")
GRINSZTAJN_FOLDERS = {
    "clf_num": ("binary", False),
    "clf_cat": ("binary", True),
    "reg_num": ("regression", False),
    "reg_cat": ("regression", True),
}
# Dataset membership per folder (from the HuggingFace mirror's file tree). Names
# repeat across folders (e.g. electricity is in both clf_num and clf_cat, with
# different feature sets), so the folder is part of the DATASETS key.
GRINSZTAJN_DATASETS = {
    "clf_num": ["Bioresponse", "Diabetes130US", "Higgs", "MagicTelescope",
                "MiniBooNE", "bank-marketing", "california", "covertype",
                "credit", "default-of-credit-card-clients", "electricity",
                "eye_movements", "heloc", "house_16H", "jannis", "pol"],
    "clf_cat": ["albert", "compas-two-years", "covertype",
                "default-of-credit-card-clients", "electricity",
                "eye_movements", "road-safety"],
    "reg_num": ["Ailerons", "Bike_Sharing_Demand", "Brazilian_houses",
                "MiamiHousing2016", "abalone", "cpu_act",
                "delays_zurich_transport", "diamonds", "elevators", "house_16H",
                "house_sales", "houses", "medical_charges",
                "nyc-taxi-green-dec-2016", "pol", "sulfur", "superconduct",
                "wine_quality", "yprop_4_1"],
    "reg_cat": ["Airlines_DepDelay_1M", "Allstate_Claims_Severity",
                "Bike_Sharing_Demand", "Brazilian_houses",
                "Mercedes_Benz_Greener_Manufacturing",
                "SGEMM_GPU_kernel_performance", "abalone", "analcatdata_supreme",
                "delays_zurich_transport", "diamonds", "house_sales",
                "medical_charges", "nyc-taxi-green-dec-2016",
                "particulate-matter-ukair-2017", "seattlecrime6", "topo_2_1",
                "visualizing_soil"],
}
GRINSZTAJN_TASKS = {}   # "gr:<folder>/<name>" -> task, filled at registration
# Cap rows so the largest datasets (Higgs, nyc-taxi, ...) stay tractable, in the
# spirit of the paper's size caps. Seeded subsample for reproducibility.
_GRINSZTAJN_MAX_ROWS = 50000


_GRINSZTAJN_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data_cache", "grinsztajn")


def _grinsztajn_local_csv(folder, name):
    """Download-once local copy of one Grinsztajn CSV. HuggingFace rate-limits
    anonymous bursts (parallel workers saw HTTP 401s), so each file is fetched
    at most once ever: raw bytes to a temp file, then an atomic rename that is
    race-safe across worker processes. Retries with backoff for flaky links."""
    import time as _time
    import urllib.request
    path = os.path.join(_GRINSZTAJN_CACHE_DIR, f"{folder}__{name}.csv")
    if os.path.exists(path):
        return path
    os.makedirs(_GRINSZTAJN_CACHE_DIR, exist_ok=True)
    url = f"{GRINSZTAJN_HF}/{folder}/{name}.csv"
    tmp = f"{path}.{os.getpid()}.tmp"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, path)
            return path
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(path):   # another worker won the race
                return path
            if attempt == 3:
                raise
            _time.sleep(3 * (attempt + 1))
    return path


def _make_grinsztajn_builder(folder, name, task, has_cats):
    def builder(scale, rng):
        import pandas as pd
        df = pd.read_csv(_grinsztajn_local_csv(folder, name))
        if len(df) > _GRINSZTAJN_MAX_ROWS:
            df = df.sample(_GRINSZTAJN_MAX_ROWS, random_state=0).reset_index(drop=True)
        return _frame_to_dataset(df.iloc[:, :-1], df.iloc[:, -1],
                                 "auto" if has_cats else None, task)
    return builder


def _add_grinsztajn_datasets():
    """Register the Grinsztajn benchmark (HuggingFace mirror) into DATASETS as
    gr:<folder>/<name>. Idempotent so workers can call it once cheaply."""
    if any(k.startswith("gr:") for k in DATASETS):
        return
    for folder, (task, has_cats) in GRINSZTAJN_FOLDERS.items():
        for name in GRINSZTAJN_DATASETS[folder]:
            key = f"gr:{folder}/{name}"
            DATASETS[key] = _make_grinsztajn_builder(folder, name, task, has_cats)
            GRINSZTAJN_TASKS[key] = task


# The Penn Machine Learning Benchmarks (PMLB, EpistasisLab) — a fourth, fully
# INDEPENDENT suite, distinct from Grinsztajn / the OpenML-34 suite / TabArena.
# Purpose: a TUNING benchmark, kept separate so the report suites above stay
# pure out-of-sample. Data is fetched as gzipped TSV from the GitHub LFS media
# mirror (no `pmlb` dependency, same spirit as the Grinsztajn CSV loader); every
# file has a `target` last-or-named column. Two design notes specific to PMLB:
#   * All columns are integer-encoded (no string dtype), so `cats="auto"` would
#     find nothing — we load everything as numeric. Nominal signal in the
#     "categorical" columns is therefore not exploited (acceptable for tuning).
#   * PMLB regression is dominated by synthetic Friedman/BNG families; we curate
#     a real-world subset only. PMLB's value-add is MULTICLASS coverage, which
#     the Grinsztajn suite lacks entirely.
# The curated subset is split into a `tune` fold and a `holdout` fold so a tuned
# knob can be checked for within-suite generalization (tune -> holdout) before it
# ever touches the report suites. Keys are "pm:<fold>/<name>".
PMLB_MEDIA = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/"
              "master/datasets")
# (name, task) per fold. Deduped against Grinsztajn + the OpenML-34 suite;
# real-world only (no fri_c*/BNG_*/2dplanes/mv synthetics). reg = regression
# (some are ordinal targets, treated as continuous), bin = binary, mc = multiclass.
PMLB_DATASETS = {
    "tune": [
        ("1028_SWD", "regression"), ("1029_LEV", "regression"),
        ("503_wind", "regression"), ("225_puma8NH", "regression"),
        ("218_house_8L", "regression"),
        ("churn", "binary"), ("hypothyroid", "binary"), ("coil2000", "binary"),
        ("yeast", "multiclass"), ("contraceptive_method", "multiclass"),
        ("segmentation", "multiclass"), ("texture", "multiclass"),
        ("nursery", "multiclass"),
    ],
    "holdout": [
        ("1030_ERA", "regression"),
        ("4544_GeographicalOriginalofMusic", "regression"),
        ("solar_flare", "regression"), ("529_pollen", "regression"),
        ("dis", "binary"), ("clean2", "binary"), ("titanic", "binary"),
        ("dna", "multiclass"), ("page_blocks", "multiclass"),
        ("ann_thyroid", "multiclass"), ("mfeat_factors", "multiclass"),
        ("krkopt", "multiclass"),
    ],
}
PMLB_TASKS = {}   # "pm:<fold>/<name>" -> task, filled at registration
_PMLB_MAX_ROWS = 50000


def _make_pmlb_builder(name, task):
    def builder(scale, rng):
        import pandas as pd
        df = pd.read_csv(f"{PMLB_MEDIA}/{name}/{name}.tsv.gz",
                         sep="\t", compression="gzip")
        if len(df) > _PMLB_MAX_ROWS:
            df = df.sample(_PMLB_MAX_ROWS, random_state=0).reset_index(drop=True)
        y = df["target"]
        X_df = df.drop(columns=["target"])
        # All-numeric integer encoding -> no categoricals to detect (cats=None).
        return _frame_to_dataset(X_df, y, None, task)
    return builder


def _add_pmlb_datasets():
    """Register the curated PMLB tuning suite into DATASETS as pm:<fold>/<name>.
    Idempotent so workers can call it once cheaply."""
    if any(k.startswith("pm:") for k in DATASETS):
        return
    for fold, items in PMLB_DATASETS.items():
        for name, task in items:
            key = f"pm:{fold}/{name}"
            DATASETS[key] = _make_pmlb_builder(name, task)
            PMLB_TASKS[key] = task


# --------------------------------------------------------------------------
# SynthGen: frozen prior-sampled synthetic suite (decision tier 1). Content is
# deterministic per key (harness seed only moves the split); recipe factors +
# Bayes floors ride in the per-dataset meta. See benchmarks/synthgen/.
# --------------------------------------------------------------------------
SYN_TASKS = {}    # "syn:<ver>/<id>" -> task, filled at registration


def _add_synth_datasets():
    """Register every frozen synthgen id as syn:<version>/<id>. Idempotent;
    registration builds closures only (no data generation)."""
    if any(k.startswith("syn:") for k in DATASETS):
        return
    import synthgen
    for key in synthgen.all_frozen_keys():
        DATASETS[key] = synthgen.make_builder(key)
        SYN_TASKS[key] = synthgen.task_of(key)


# --------------------------------------------------------------------------
# HC: REAL high-cardinality-categorical datasets (decision tier). Added so the
# decision stack has a real-data expression of the entity-cat / high-card regime
# where the CatBoost Brier gap lives: Grinsztajn curates high-card cats OUT and
# has 0 multiclass, so no lever targeting that gap can clear the protocol on
# Grinsztajn alone (see benchmarks/HIGHCARD_PLAN.md + synthgen/PAYOFF.md). This
# suite is NEITHER synthetic (the generator's prior must not vote on ships) NOR a
# Grinsztajn split (both halves would inherit the same blind spot).
#
# Every dataset passed a HARD overlap audit: zero intersection with TabArena (the
# sealed holdout -- its DATASETS must never enter a decision suite), the
# Grinsztajn 59, the OpenML one-shot gate (29), or the PMLB tuning suite (25),
# matched by OpenML id AND normalized/substring name. Selection was by DATA
# PROPERTIES ONLY (n, cardinality, task, missingness) measured WITHOUT fitting any
# model; degenerate sets (undefined/empty target, zero-support classes, pure-ID
# columns, structural target leakage, inactive OpenML versions) were dropped. The
# audit matrix + rationale live in benchmarks/HIGHCARD_PLAN.md. The frozen list
# only changes with a version bump + re-audit (synthgen-freeze discipline).
#
# Loaded via fetch_openml(as_frame=True); cats auto-detected from dtype; a 100k
# deterministic subsample (random_state=0, NOT the harness seed -- the train/test
# split stays the only seed-dependent step). Keys are "hc:<name>".
# --------------------------------------------------------------------------
HC_DATASETS = {
    # binary, high-card real categoricals
    "kick":                  dict(data_id=41162, task="binary"),      # Model card 1063
    "porto-seguro":          dict(data_id=42742, task="binary"),      # ps_car_11_cat 104
    "sf-police-incidents":   dict(data_id=42344, task="binary"),      # Address 15165
    "kdd_ipums_la_97-small": dict(data_id=993,   task="binary"),      # occ/ind codes 191
    # multiclass, the regime Grinsztajn lacks entirely
    "okcupid-stem":          dict(data_id=42734, task="multiclass"),  # speaks 7019 (3-class)
    "Traffic_violations":    dict(data_id=42345, task="multiclass"),  # Model 3830 (3-class)
    "cjs":                   dict(data_id=473,   task="multiclass"),  # TREE 57 (6-class)
    "eucalyptus":            dict(data_id=188,   task="multiclass"),  # Sp 27 (5-class)
    # regression with categoricals
    "wine-reviews":          dict(data_id=41275, task="regression"),  # winery 15633
    "colleges":              dict(data_id=42727, task="regression"),  # zip 6039 / state 59
    "house_prices_nominal":  dict(data_id=42563, task="regression"),  # Ames, Neighborhood 25
    "black_friday":          dict(data_id=41540, task="regression"),  # low-card cats
    "employee_salaries":     dict(data_id=42125, task="regression"),  # department 37
    "Moneyball":             dict(data_id=41021, task="regression"),  # Team 39
}
HC_TASKS = {}   # "hc:<name>" -> task, filled at registration

# --------------------------------------------------------------------------
# PUBLIC suite (issue #37) -- the sealed suite behind the published chart.
#
# SEALED: report-only. No result from it, aggregate or per-task, may influence a
# source change. It exists because the published chart must not run on
# Grinsztajn or HC -- we tune against those, so charting them would be in-sample
# and would contradict the north star ("true generalization, never faked from
# data"). Decisions keep running on synth -> Grinsztajn + HC -> OpenML gate.
#
# EMPTY PENDING AUDIT. The selection criteria, the overlap gate and the
# procedure are in benchmarks/PUBLIC_PLAN.md. The list is deliberately not
# populated from remembered OpenML ids: HIGHCARD_PLAN.md's step 0 requires every
# candidate be verified in-session, and the OpenML API was returning HTTP 504
# throughout the session that built this machinery. Filling it in is one line
# per dataset once the audit runs.
PUBLIC_DATASETS = {}
PUBLIC_TASKS = {}   # "pub:<name>" -> task, filled at registration
_PUBLIC_MAX_ROWS = 200000   # higher cap than HC: the speed axis is the point
_HIGHCARD_MAX_ROWS = 100000
# Categorical columns whose nunique/n exceeds this are row identifiers / free
# text (e.g. wine-reviews' `description`/`title`, ~94% unique). They carry no
# repeated-level signal -- the opposite of the entity-cat regime this suite
# targets -- and only add noise + fit cost, so the loader drops them. The
# threshold sits well above every genuine high-card cat in the frozen suite
# (the highest is colleges' zip at ~0.855), so no real categorical is dropped.
_HIGHCARD_ID_FRAC = 0.9
# sklearn's OpenML cache lands on C: by default, which has ~4 GB free while
# porto/sf-police/wine-reviews are 100-500 MB fetches, so default the cache to A:
# (override with the standard SCIKIT_LEARN_DATA env var on a box where C: is fine).
_HIGHCARD_DATA_HOME = os.environ.get("SCIKIT_LEARN_DATA") or r"A:\code\sklearn_data"


def _make_highcard_builder(spec, time_col=None, max_rows=None):
    """Build a dataset-builder closure for one HC spec (fetched by data_id, with
    a deterministic 100k subsample). Mirrors the Grinsztajn/PMLB builders.

    With `time_col`, rows are returned in ascending time order and rows whose
    timestamp won't parse are dropped, so _run_seed_task can cut a temporal
    split positionally. The time column stays a FEATURE: a model meeting unseen
    later timestamps is precisely the deployment failure the variant measures.
    """
    def builder(scale, rng):
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=spec["data_id"], as_frame=True,
                          data_home=_HIGHCARD_DATA_HOME)
        frame = ds.frame
        target = ds.target.name
        # Deterministic subsample BEFORE the split (fixed seed 0, ignores the
        # harness rng), so the train/test split stays the only seed-dependent step.
        cap = max_rows or _HIGHCARD_MAX_ROWS
        if len(frame) > cap:
            frame = frame.sample(cap, random_state=0).reset_index(drop=True)
        if time_col is not None:
            key = _time_sort_key(frame[time_col])
            if key is None:
                raise ValueError(
                    f"time column {time_col!r} could not be ordered as numeric "
                    "or datetime")
            frame = (frame.assign(_t=key).dropna(subset=["_t"])
                     .sort_values("_t", kind="stable")
                     .drop(columns=["_t"]).reset_index(drop=True))
        X_df = frame.drop(columns=[target])
        # Drop near-unique categorical columns (row identifiers / free text).
        n = len(X_df)
        id_cols = [c for c in X_df.columns
                   if _is_categorical_dtype(X_df[c].dtype)
                   and X_df[c].nunique(dropna=True) > _HIGHCARD_ID_FRAC * n]
        if id_cols:
            X_df = X_df.drop(columns=id_cols)
        return _frame_to_dataset(X_df, frame[target], "auto", spec["task"])
    return builder


def _add_highcard_datasets():
    """Register the frozen HC suite into DATASETS as hc:<name>. Idempotent so
    workers can call it once cheaply."""
    if any(k.startswith("hc:") for k in DATASETS):
        return
    for name, spec in HC_DATASETS.items():
        key = f"hc:{name}"
        DATASETS[key] = _make_highcard_builder(spec)
        HC_TASKS[key] = spec["task"]


def _add_public_datasets():
    """Register the sealed public suite as pub:<name>. Idempotent.

    Reuses the HC builder (fetch by OpenML id, deterministic row cap, drop
    near-unique categorical id columns) with a larger cap, and honours a
    per-dataset `time_col` so a public dataset can carry a temporal variant.
    """
    if any(k.startswith("pub:") for k in DATASETS):
        return
    for name, spec in PUBLIC_DATASETS.items():
        key = f"pub:{name}"
        DATASETS[key] = _make_highcard_builder(
            spec, time_col=None, max_rows=_PUBLIC_MAX_ROWS)
        PUBLIC_TASKS[key] = spec["task"]
        if spec.get("time_col"):
            TEMPORAL_COLUMNS[key] = spec["time_col"]


# --------------------------------------------------------------------------
# Model runners. Each returns (metrics_dict, fit_seconds, best_iter). The
# metrics dict always includes "primary" (higher=better; -RMSE for regression,
# F1-macro for classification) which the summary/sign-test logic uses. For
# classification it also includes "log_loss" so table cuts can report both.
# Returns None if the model can't run the task (e.g. competitor without
# native categorical support we skip).
# --------------------------------------------------------------------------
def _compute_metrics(task, y_true, model, X_test):
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, model.predict(X_test))))
        return {"primary": -rmse, "rmse": rmse}
    f1 = float(f1_score(y_true, model.predict(X_test), average="macro"))
    proba = model.predict_proba(X_test)
    classes = getattr(model, "classes_", np.unique(y_true))
    # log_loss needs labels= for safety when a class is missing from y_true.
    ll = float(log_loss(y_true, proba, labels=classes))
    # Multiclass Brier: mean over samples of sum_k (p_k - onehot_k)^2. Bounded,
    # outlier-robust, and a proper scoring rule like log loss, but it aggregates
    # far more stably across datasets (no unbounded tail). Used for binary too
    # (the K=2 sum form), so the two tasks share one definition.
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    # Miscalibration (MCB), the CORP / Dimitriadis-Gneiting-Jordan calibration
    # measure: how much a *monotone* recalibration improves the (per-class) Brier
    # score. Refit each class's probabilities to its outcomes with ascending,
    # clipped isotonic regression (the optimal calibration map) and take the gap
    # MCB_k = Brier_k(p_k) - Brier_k(isotonic(p_k)); average over classes. 0 means
    # already perfectly calibrated, higher = worse. In-sample isotonic on the test
    # fold is the standard CORP diagnostic. Binary -> just the class-1 curve.
    from sklearn.isotonic import IsotonicRegression
    mcb_k = []
    for k in range(proba.shape[1]):
        p = proba[:, k]
        yk = onehot[:, k]
        recal = IsotonicRegression(increasing=True, out_of_bounds="clip"
                                   ).fit_transform(p, yk)
        mcb_k.append(np.mean((p - yk) ** 2) - np.mean((recal - yk) ** 2))
    mcb = float(np.mean(mcb_k))
    return {"primary": f1, "f1_macro": f1, "log_loss": ll, "brier": brier,
            "calibration_mcb": mcb}


def _finish(task, y_true, model, X_test, t_fit_start):
    """Stop the fit clock, then score, timing the scoring separately.

    Returns (metrics, fit_seconds, predict_seconds).

    Every runner used to end with
        return _compute_metrics(...), time.time() - t, best_iter
    and Python evaluates tuple elements left to right, so metric computation ran
    BEFORE the clock was read and landed inside "fit_time". For classification
    that is a full `predict` pass, a full `predict_proba` pass, and one isotonic
    regression fit per class (the CORP calibration term) — all charged to the
    model's fit.

    Measured on the built-in panel at the time of the fix: scoring was 31% of the
    old "fit time" for ChimeraBoost on breast_cancer and 14% for LightGBM (7% and
    3% on diabetes/regression). Because our predict is absolutely slower than
    LightGBM's, the old convention OVERSTATED our slowdown — the ChimeraBoost /
    LightGBM ratio on breast_cancer was 2.08x before and 1.69x after. Those are
    tiny datasets where prediction dominates; expect a smaller effect on the 50k-
    100k-row suites. Keep the clock read on its own line.
    """
    fit_s = time.time() - t_fit_start
    t_pred = time.time()
    metrics = _compute_metrics(task, y_true, model, X_test)
    return metrics, fit_s, time.time() - t_pred


def _val_split(Xtr, ytr, task, seed):
    """Carve an internal validation set from training data for early stopping.
    Never touches the test set."""
    strat = ytr if task != "regression" else None
    return train_test_split(Xtr, ytr, test_size=0.2, random_state=seed,
                            stratify=strat)


# Shared early-stopping budget for every model, so the comparison is fair.
MAX_ITERS = 2000
PATIENCE = 50


def _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads, lr=None,
                 ordered_boosting=None, depth=6, subsample=1.0, colsample=None,
                 mcw=None,
                 cat_combinations=None, leaf_estimation_iterations=None,
                 linear_leaves=False, linear_lambda=1.0, cross_features=False,
                 cat_smoothing=None, selection_rounds=None, quantize=False,
                 refit_full=False):
    t = time.time()
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    # None = use the class default. For ordered_boosting that's False (Reg) /
    # False (Clf); for min_child_weight it's the classifier's size-adaptive auto.
    # An explicit value overrides (e.g. --no-ordered-boosting, --chimera-mcw 0).
    kw = {} if ordered_boosting is None else {"ordered_boosting": ordered_boosting}
    if quantize:
        kw["quantize_gradients"] = True
    # "off" forces the full-data refit off (default-ON since 0.25.0); a plain
    # False means "don't override the class default", as for the other knobs.
    if refit_full == "off":
        kw["refit_full"] = False
    elif refit_full:
        kw["refit_full"] = True
    if leaf_estimation_iterations is not None:
        kw["leaf_estimation_iterations"] = leaf_estimation_iterations
    if mcw is not None:
        kw["min_child_weight"] = mcw
    if cat_smoothing is not None:
        kw["cat_smoothing"] = cat_smoothing
    # None = class default (now 100). "full" = explicit selection_rounds=None
    # (every variant to full early stopping -- the pre-0.15 ablation arm).
    if selection_rounds == "full":
        kw["selection_rounds"] = None
    elif selection_rounds is not None:
        kw["selection_rounds"] = selection_rounds
    # None = use the class auto-default (on for all-categorical data); only force
    # it on when --chimera-cat-combinations is passed.
    if cat_combinations:
        kw["cat_combinations"] = True
    if linear_leaves == "auto":
        # Regressor: linear_leaves=None = validation-selected (fit both, keep
        # the val winner). The classifier's None default is already its own
        # auto rule, so only regression needs the override.
        if task == "regression":
            kw["linear_leaves"] = None
            kw["linear_lambda"] = linear_lambda
    elif linear_leaves == "off":
        # force constant leaves everywhere (ablation arm vs the class defaults)
        kw["linear_leaves"] = False
    elif linear_leaves:
        # multiclass doesn't support linear leaves yet; fall back to constant
        # leaves there so a full-suite run doesn't crash on multiclass tasks.
        is_multiclass = task != "regression" and len(np.unique(ytr)) > 2
        if not is_multiclass:
            kw["linear_leaves"] = True
            kw["linear_lambda"] = linear_lambda
    # "off" forces cross_features off (ablation arm vs the shipped default
    # None = on where applicable, multiclass included since M1).
    if cross_features == "off":
        kw["cross_features"] = False
    elif cross_features:
        kw["cross_features"] = True
    # IMPORTANT: this measures OUT-OF-BOX DEFAULT behavior. We call fit(Xtr, ytr)
    # with NO explicit eval_set, so ChimeraBoost performs its own internal
    # early-stopping split (early_stopping=True, validation_fraction default) —
    # exactly what a user gets from `ChimeraBoostX().fit(X, y)`. n_estimators and
    # early_stopping_rounds come from the shared harness budget, which equals the
    # class defaults for the headline run, so the benchmark == the default.
    m = Est(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
            learning_rate=lr, depth=depth,
            subsample=subsample, colsample=colsample,
            thread_count=threads, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    return (*_finish(task, yte, m, Xte, t), m.best_iteration_)


# Bagged ChimeraBoost: train N members on bootstrap resamples, each early-stopping
# on its own bootstrap, and average. ensemble_n_jobs=1 (members sequential, each
# using the job's thread budget) so we don't nest a joblib pool inside the
# harness's --jobs ProcessPool. For a faster dedicated sweep, run with --jobs 1.
ENSEMBLE_N = 10


def _chimera_ens(n, task, Xtr, ytr, Xte, yte, cat, threads, lr=None,
                 subsample=1.0, colsample=None, quantize=False):
    """Shared implementation for all bagged-ChimeraBoost runners.

    ensemble_n_jobs is left at the shipped default (parallel members inside
    the task's thread budget) so the chart measures the shipped config
    (BAGGING_PLAN.md B4; same core budget as every other model). lr /
    subsample / colsample / quantize forward to the members (the B3
    member-defaults grid rides the same --chimera-* flags as the single arm)."""
    t = time.time()
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    kw = {"quantize_gradients": True} if quantize else {}
    m = Est(n_estimators=MAX_ITERS, early_stopping=True, early_stopping_rounds=PATIENCE,
            n_ensembles=n, learning_rate=lr,
            subsample=subsample, colsample=colsample,
            thread_count=threads, random_state=0, **kw)
    m.fit(Xtr, ytr, cat_features=cat)
    return (*_finish(task, yte, m, Xte, t), m.best_iteration_)


def _run_chimera_ensemble(task, Xtr, ytr, Xte, yte, cat, threads, **kw):
    return _chimera_ens(ENSEMBLE_N, task, Xtr, ytr, Xte, yte, cat, threads, **kw)


def _run_chimera_ensemble_2(task, Xtr, ytr, Xte, yte, cat, threads, **kw):
    return _chimera_ens(2, task, Xtr, ytr, Xte, yte, cat, threads, **kw)


def _run_chimera_ensemble_5(task, Xtr, ytr, Xte, yte, cat, threads, **kw):
    return _chimera_ens(5, task, Xtr, ytr, Xte, yte, cat, threads, **kw)


def _run_chimera_ensemble_8(task, Xtr, ytr, Xte, yte, cat, threads, **kw):
    # The blessed bagged config (BAGGING_PLAN.md B3): K=8 with the library's
    # bagged-member defaults.
    return _chimera_ens(8, task, Xtr, ytr, Xte, yte, cat, threads, **kw)


# --- SELECT program arms (benchmarks/SELECT_PLAN.md) ------------------------
# Fixed-config ChimeraBoost variants that trade model selection for fit time,
# so the empty 1x-5x stretch of the strength/slowdown Pareto can be measured.
# They deliberately ignore the --chimera-* CLI knobs: each is one pinned
# operating point, and several run side by side in a single benchmark.
#
# Passing the STRING "off" is what forces a knob off; the runner's plain
# `False` defaults mean "don't override the class default" (see _run_chimera).
# Disabling both decisions statically collapses the search to exactly one
# booster fit: `select_ll` needs linear_leaves to be None (sklearn_api.py:1436),
# `cross_ok` and the classifier's `fast` audition both need cross_features to
# be anything but False (sklearn_api.py:1443, :2101).


def _run_chimera_one(task, Xtr, ytr, Xte, yte, cat, threads):
    """Speed floor: no selection at all, constant leaves. One booster fit."""
    return _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads,
                        linear_leaves="off", cross_features="off",
                        refit_full="off")


def _run_chimera_one_lin(task, Xtr, ytr, Xte, yte, cat, threads):
    """Rung 1 (quality=1): one booster fit, linear leaves pinned, no refit."""
    return _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads,
                        linear_leaves=True, cross_features="off",
                        refit_full="off")


def _run_chimera_norefit(task, Xtr, ytr, Xte, yte, cat, threads):
    """Rung 2 (quality=2): the full search, without the default's refit."""
    return _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads,
                        refit_full="off")


def _run_chimera_sel25(task, Xtr, ytr, Xte, yte, cat, threads):
    """Full search, quarter-length auditions (selection_rounds 100 -> 25)."""
    return _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads,
                        selection_rounds=25)


def _run_chimera_refit(task, Xtr, ytr, Xte, yte, cat, threads):
    """Defaults + refit_full: retrain the ES winner on 100% of the rows.

    The ladder rung above the default. Note it does NOT stack with the
    bagged rungs -- refit_full is a deliberate no-op inside ensemble
    members, whose OOB rows already act as an eval set (REFIT_PLAN.md), so
    ChimeraBoostEns5/Ens8 sit on top of the plain default, not on top of
    this arm.
    """
    return _run_chimera(task, Xtr, ytr, Xte, yte, cat, threads,
                        refit_full=True)


def _run_sklearn(task, Xtr, ytr, Xte, yte, cat, threads):
    """sklearn HGB with native categorical support.

    HGB requires integer-encoded categoricals; we ordinal-encode them here so
    the comparison is fair (same information given to all models).
    """
    from sklearn.preprocessing import OrdinalEncoder
    t = time.time()
    if cat is not None:
        cat_idx = list(cat)
        Xtr = np.array(Xtr, dtype=object)
        Xte = np.array(Xte, dtype=object)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[:, cat_idx])
        Xtr[:, cat_idx] = enc.transform(Xtr[:, cat_idx])
        Xte[:, cat_idx] = enc.transform(Xte[:, cat_idx])
        Xtr = Xtr.astype(float)
        Xte = Xte.astype(float)
    else:
        cat_idx = None
    # HGB has built-in early stopping via a validation fraction. NOTE: HGB has
    # NO n_jobs constructor arg (it parallelises via OpenMP, set through
    # OMP_NUM_THREADS / threadpoolctl), so we must not pass `threads` here —
    # doing so raised TypeError and silently dropped the sklearn column.
    common = dict(max_iter=MAX_ITERS, early_stopping=True,
                  validation_fraction=0.2, n_iter_no_change=PATIENCE,
                  categorical_features=cat_idx,
                  random_state=0)
    Est = (HistGradientBoostingRegressor if task == "regression"
           else HistGradientBoostingClassifier)
    m = Est(**common)
    m.fit(Xtr, ytr)
    return (*_finish(task, yte, m, Xte, t), m.n_iter_)


def _run_catboost(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["catboost"]:
        return None
    from catboost import CatBoostRegressor, CatBoostClassifier
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
                  thread_count=threads or -1, verbose=False, random_seed=0)
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = Est(**common)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return (*_finish(task, yte, m, Xte, t), m.best_iteration_)


def _xgb_dataframes(Xtr, Xval, Xte, cat_idx):
    """Build three pandas DataFrames sharing the same category sets per cat
    column. Categories absent from training become NaN at predict-time, which
    XGBoost handles via its default missing direction. This avoids the
    'category not in the training set' error on unseen values."""
    import pandas as pd
    cat_set = set(cat_idx)

    def _to_df(X):
        df = pd.DataFrame(X)
        for i in range(df.shape[1]):
            if i not in cat_set:
                df[i] = pd.to_numeric(df[i], errors="coerce")
        return df

    df_tr = _to_df(Xtr)
    df_va = _to_df(Xval)
    df_te = _to_df(Xte)
    for i in cat_idx:
        df_tr[i] = df_tr[i].astype("category")
        cats = df_tr[i].cat.categories
        df_va[i] = pd.Categorical(df_va[i], categories=cats)
        df_te[i] = pd.Categorical(df_te[i], categories=cats)
    return df_tr, df_va, df_te


def _run_xgboost(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["xgboost"]:
        return None
    import xgboost as xgb
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=MAX_ITERS, early_stopping_rounds=PATIENCE,
                  n_jobs=threads or -1, random_state=0, verbosity=0,
                  tree_method="hist")
    if cat is not None:
        common["enable_categorical"] = True
        Xf_in, Xv_in, Xte_in = _xgb_dataframes(Xf, Xv, Xte, list(cat))
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte
    Est = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    m = Est(**common)
    m.fit(Xf_in, yf, eval_set=[(Xv_in, yv)], verbose=False)
    best = getattr(m, "best_iteration", None)
    return (*_finish(task, yte, m, Xte_in, t), best)


def _lgb_prepare(Xtr, Xval, Xte, cat_idx):
    """Ordinal-encode the cat columns to ints using a single encoder fit on
    training. Validation and test reuse it; unseen categories become -1.
    LightGBM expects integer-coded categoricals when categorical_feature is set.
    """
    from sklearn.preprocessing import OrdinalEncoder
    Xtr = np.array(Xtr, dtype=object)
    Xval = np.array(Xval, dtype=object)
    Xte = np.array(Xte, dtype=object)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[:, cat_idx])
    Xtr[:, cat_idx] = enc.transform(Xtr[:, cat_idx])
    Xval[:, cat_idx] = enc.transform(Xval[:, cat_idx])
    Xte[:, cat_idx] = enc.transform(Xte[:, cat_idx])
    return Xtr.astype(float), Xval.astype(float), Xte.astype(float)


def _run_lightgbm(task, Xtr, ytr, Xte, yte, cat, threads):
    if not HAVE["lightgbm"]:
        return None
    import lightgbm as lgb
    Xf, Xv, yf, yv = _val_split(Xtr, ytr, task, 0)
    t = time.time()
    common = dict(n_estimators=MAX_ITERS, n_jobs=threads or -1,
                  random_state=0, verbosity=-1)
    fit_kw = dict(callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    if cat is not None:
        Xf_in, Xv_in, Xte_in = _lgb_prepare(Xf, Xv, Xte, list(cat))
        fit_kw["categorical_feature"] = list(cat)
    else:
        Xf_in, Xv_in, Xte_in = Xf, Xv, Xte
    fit_kw["eval_set"] = [(Xv_in, yv)]
    Est = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    m = Est(**common)
    m.fit(Xf_in, yf, **fit_kw)
    return (*_finish(task, yte, m, Xte_in, t), m.best_iteration_)


RUNNERS = {
    "ChimeraBoost": _run_chimera,
    "ChimeraBoostEns2": _run_chimera_ensemble_2,
    "ChimeraBoostEns5": _run_chimera_ensemble_5,
    "ChimeraBoostEns8": _run_chimera_ensemble_8,
    "ChimeraBoostEns10": _run_chimera_ensemble,
    "ChimeraBoostOne": _run_chimera_one,
    "ChimeraBoostOneLin": _run_chimera_one_lin,
    "ChimeraBoostSel25": _run_chimera_sel25,
    "ChimeraBoostRefit": _run_chimera_refit,
    "ChimeraBoostNoRefit": _run_chimera_norefit,
    "sklearn_HGB": _run_sklearn,
    "CatBoost": _run_catboost,
    "XGBoost": _run_xgboost,
    "LightGBM": _run_lightgbm,
}

# Always available (hard deps); the rest are gated on _detect(). Ensemble variants
# are also dep-free but N× slower, so they're selectable via --models but off by
# default (like XGBoost).
_ALWAYS = ("ChimeraBoost", "sklearn_HGB")
_OFF_BY_DEFAULT = ("XGBoost", "ChimeraBoostEns2", "ChimeraBoostEns5",
                   "ChimeraBoostEns8", "ChimeraBoostEns10",
                   "ChimeraBoostOne", "ChimeraBoostOneLin",
                   "ChimeraBoostSel25", "ChimeraBoostRefit",
                   "ChimeraBoostNoRefit")
_OPTIONAL = ("CatBoost", "XGBoost", "LightGBM")


def _make_runners(model_names, chimera_cfg):
    """Build the runner dict for `model_names`, wiring ChimeraBoost's CLI knobs.
    The bagged arms honor the member-HP subset of the knobs (lr / subsample /
    colsample) so the B3 grid runs ride the same flags."""
    import functools
    runners = dict(RUNNERS)
    runners["ChimeraBoost"] = functools.partial(_run_chimera, **chimera_cfg)
    ens_cfg = {k: chimera_cfg[k]
               for k in ("lr", "subsample", "colsample", "quantize")}
    for name in ("ChimeraBoostEns2", "ChimeraBoostEns5", "ChimeraBoostEns8",
                 "ChimeraBoostEns10"):
        runners[name] = functools.partial(runners[name], **ens_cfg)
    return {name: runners[name] for name in model_names}


def _subsample_train(Xtr, ytr, frac, task):
    """Keep `frac` of the TRAINING rows, stratified for classification.

    Fixed random_state=0 -- the same house convention as the suite row caps, so
    the train/test split stays the only seed-dependent step and a twin's shrink
    is reproducible across runs.
    """
    n = int(round(len(ytr) * frac))
    if n >= len(ytr) or n < 2:
        return Xtr, ytr
    strat = ytr if task != "regression" else None
    try:
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n,
                                        random_state=0, stratify=strat)
    except ValueError:
        # Too few members of some class to stratify at this size; fall back to
        # an unstratified draw rather than dropping the variant.
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, random_state=0)
    return Xs, ys


def _temporal_split(X, y, seed, task):
    """Rolling-origin split of time-ordered rows, or None if degenerate.

    X/y arrive sorted ascending by the dataset's timestamp (see
    _make_highcard_builder). Seed s takes cut TEMPORAL_CUTS[s % 3]: train on
    everything before it, test on the window that follows. Test rows whose class
    never appears in training are dropped -- the model cannot emit a probability
    for an unseen label, and scoring them would credit confident wrong answers
    (the one-hot row would be all zeros, so Brier would reward low probabilities
    on every real class).
    """
    n = len(y)
    cut = TEMPORAL_CUTS[seed % len(TEMPORAL_CUTS)]
    i = int(n * cut)
    j = min(n, i + int(n * TEMPORAL_TEST_FRAC))
    if i < 2 or j - i < 2:
        return None
    Xtr, ytr, Xte, yte = X[:i], y[:i], X[i:j], y[i:j]
    if task != "regression":
        seen = np.unique(ytr)
        if len(seen) < 2:
            return None
        keep = np.isin(yte, seen)
        if keep.sum() < 2:
            return None
        Xte, yte = Xte[keep], yte[keep]
    return Xtr, Xte, ytr, yte


def _run_seed_task(task):
    """Fit every requested model on one (dataset, seed) draw. Top-level and
    picklable so it can run in a worker process. Returns
    (ds_name, seed, meta, {model: (metrics, secs, best_iter) or None})."""
    global PATIENCE, ENSEMBLE_N
    (ds_name, seed, scale, threads, model_names, chimera_cfg, patience,
     ensemble_n, need_openml, need_grinsztajn, need_pmlb, need_synth,
     need_highcard, need_public, need_variants) = task
    PATIENCE = patience
    ENSEMBLE_N = ensemble_n
    if need_openml:
        _add_openml_datasets()
    if need_grinsztajn:
        _add_grinsztajn_datasets()
    if need_pmlb:
        _add_pmlb_datasets()
    if need_synth:
        _add_synth_datasets()
    if need_highcard:
        _add_highcard_datasets()
    if need_public:
        _add_public_datasets()
    if need_variants:
        _add_variant_datasets(list(DATASETS))

    rng = np.random.default_rng(1000 + seed)
    X, y, cat, ttype = DATASETS[ds_name](scale, rng)
    variant = ds_name.split(VARIANT_SEP, 1)[1] if VARIANT_SEP in ds_name else ""

    if variant == "time":
        split = _temporal_split(X, y, seed, ttype)
        if split is None:
            # Degenerate window (e.g. a class the training period never saw at
            # all). Reported as a skip rather than silently scored.
            print(f"  [skip] {ds_name} (seed {seed}): temporal window "
                  "left training data with fewer than 2 classes")
            return ds_name, seed, {"task": ttype, "n_train": 0, "n_total": int(len(y)),
                                   "n_features": int(X.shape[1]),
                                   "has_cats": bool(cat)}, {}
        Xtr, Xte, ytr, yte = split
    else:
        strat = y if ttype != "regression" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=strat)
        if variant in SUS_FRACTIONS:
            # Shrink TRAINING rows only. The test set is identical to the
            # parent's for this seed, so the twin reads as a point on the
            # parent's learning curve rather than a noisier separate dataset.
            Xtr, ytr = _subsample_train(Xtr, ytr, SUS_FRACTIONS[variant], ttype)

    meta = {"task": ttype, "n_train": int(Xtr.shape[0]),
            "n_total": int(X.shape[0]), "n_features": int(X.shape[1]),
            "has_cats": bool(cat), "variant": variant or None}
    # Target scale, so the table layer can flag "near-solved" regression datasets
    # (best NRMSE = best_RMSE / y_std below a threshold), where the "% vs best"
    # RMSE ratio explodes a negligible absolute gap. See summarize.NEAR_SOLVED_NRMSE.
    if ttype == "regression":
        meta["y_std"] = float(np.std(y))
    if ds_name.startswith("syn:"):
        import synthgen
        meta["synth"] = synthgen.recipe_meta(ds_name)  # LRU hit: builder just ran

    out = {}
    for name, runner in _make_runners(model_names, chimera_cfg).items():
        try:
            out[name] = runner(ttype, Xtr, ytr, Xte, yte, cat, threads)
        except Exception as e:
            # A model that structurally cannot handle a dataset -- e.g. sklearn
            # HGB's hard 255-category cap on high-cardinality categoricals, or a
            # competitor OOM on a huge cat -- is recorded as SKIPPED (None), not
            # allowed to abort the whole run. Downstream aggregation already
            # treats None as "did not run" (like an uninstalled competitor). The
            # HC suite deliberately exercises regimes that break some native-cat
            # paths (the plan's "record, don't fix"); the print keeps them visible.
            print(f"  [skip] {name} on {ds_name} (seed {seed}): "
                  f"{type(e).__name__}: {e}")
            out[name] = None
    return ds_name, seed, meta, out


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def _pairwise_winrate(primary, ours, theirs, n_boot=2000):
    """(win rate %, lo, hi, wins, losses, ties, median relative gap %) for
    `ours` against `theirs`, on the per-dataset primary metric (lower = better).

    Restricting `primary` to the two models and reusing summarize's field-wide
    helpers makes the two-model "field" exactly the pairwise matchup, so the win
    rate and its bootstrap CI come from the same tested code that draws the
    Pareto chart -- no second implementation to drift.
    """
    pair = {ds: {m: v for m, v in scores.items() if m in (ours, theirs)}
            for ds, scores in primary.items()}
    pair = {ds: s for ds, s in pair.items() if len(s) == 2}
    if not pair:
        return None
    rate = summarize.winrate_vs_field(pair).get(ours)
    lo, hi = summarize.bootstrap_winrate_ci(pair, n_boot=n_boot).get(
        ours, (None, None))
    wins = losses = ties = 0
    gaps = []
    for s in pair.values():
        o, t = s[ours], s[theirs]
        if o < t:
            wins += 1
        elif o > t:
            losses += 1
        else:
            ties += 1
        # primary is lower-better, so a positive gap means we are better.
        gaps.append(100.0 * (t - o) / t)
    return rate, lo, hi, wins, losses, ties, float(np.median(gaps))


def _verdict(competitor, winrate):
    """Ship-facing read of a pairwise win rate (% of datasets we win, ties 1/2).

    Replaces the old mean-relative-gap verdict, which averaged ratios and so
    blew up on near-solved datasets (the failure PR #31 fixed in compare_runs
    but never here), and which judged classification on F1 while every ship
    decision and the Pareto chart judge it on Brier.
    """
    if winrate is None:
        return "no shared datasets"
    if competitor == "sklearn_HGB":
        return "PASS: beats sklearn" if winrate > 50.0 else "FAIL: must beat sklearn"
    if competitor == "CatBoost":
        if winrate >= 50.0:
            return "PASS: matches/beats CatBoost"
        if winrate >= 45.0:
            return "PASS: close to CatBoost"
        return f"GAP: behind CatBoost ({winrate:.1f}% of matchups)"
    return "better" if winrate > 50.0 else "behind"


class _Progress:
    """Writes a JSON sidecar tracking run progress so another process (the
    /bench status command) can poll it. One file per run, located next to the
    saved results when --save is given, else results/_running.progress.

    Schema: {status, completed, total, pct, started, updated, elapsed_s,
             eta_s, models, config, results_json}. `status` is "running" while
    tasks complete and "done" once finished. Best-effort: any write error is
    swallowed so progress tracking never breaks a benchmark run.
    """

    def __init__(self, results_path, total, model_names, args):
        import datetime
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        if results_path:
            self.path = results_path.rsplit(".", 1)[0] + ".progress"
            self.results_json = results_path.rsplit(".", 1)[0] + ".json"
        else:
            self.path = os.path.join(results_dir, "_running.progress")
            self.results_json = None
        self.total = total
        self.start = time.time()
        lei = args.leaf_estimation_iterations
        self.config = (f"seeds={args.seeds} jobs={args.jobs} "
                       f"grinsztajn={args.grinsztajn} "
                       f"pmlb={args.pmlb}"
                       f"{('/' + args.pmlb_fold) if args.pmlb_fold else ''} "
                       f"highcard={args.highcard} "
                       f"depth={args.chimera_depth} "
                       f"lei={'default' if lei is None else lei} "
                       f"ob={'off' if not args.ordered_boosting else 'on'}"
                       f"{' quantize=on' if args.chimera_quantize else ''}")
        self.models = list(model_names)
        self.started_iso = datetime.datetime.now().isoformat(timespec="seconds")
        self._write("running", 0)

    def _write(self, status, completed):
        try:
            elapsed = time.time() - self.start
            rate = completed / elapsed if elapsed > 0 and completed else 0
            remaining = (self.total - completed) / rate if rate > 0 else None
            payload = {
                "status": status, "completed": completed, "total": self.total,
                "pct": round(100.0 * completed / self.total, 1) if self.total else 0,
                "started": self.started_iso,
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(remaining, 1) if remaining is not None else None,
                "models": self.models, "config": self.config,
                "results_json": self.results_json,
            }
            with open(self.path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
        except Exception:
            pass

    def update(self, completed):
        self._write("running", completed)

    def finish(self):
        self._write("done", self.total)


def main():
    global PATIENCE, ENSEMBLE_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier for synthetic dataset sizes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None,
                    help="total thread budget across all parallel jobs "
                         "(None = all cores).")
    ap.add_argument("--jobs", type=int, default=5,
                    help="(dataset, seed) tasks to run in parallel processes; "
                         "each gets threads/jobs threads. GBDT thread scaling is "
                         "sublinear, so spreading seeds beats piling threads on "
                         "one fit (default: 5). Use 1 to run inline.")
    ap.add_argument("--with-xgboost", action="store_true",
                    help="include XGBoost (off by default; it tracks LightGBM "
                         "closely and roughly doubles competitor runtime).")
    ap.add_argument("--only", choices=["regression", "classification"],
                    default=None)
    ap.add_argument("--openml", action="store_true",
                    help="include real OpenML benchmark datasets (downloads + caches)")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="run ONLY the OpenML datasets (implies --openml)")
    ap.add_argument("--grinsztajn", action="store_true",
                    help="run the Grinsztajn et al. 2022 tabular benchmark "
                         "(binary + regression), loaded from the HuggingFace mirror.")
    ap.add_argument("--pmlb", action="store_true",
                    help="run the curated PMLB tuning suite (reg + binary + "
                         "multiclass; independent of the report suites). Use "
                         "--pmlb-fold to restrict to tune/holdout.")
    ap.add_argument("--pmlb-fold", choices=["tune", "holdout"], default=None,
                    help="with --pmlb, run only this fold (default: both).")
    ap.add_argument("--highcard", action="store_true",
                    help="run the frozen HC suite (real high-cardinality-"
                         "categorical datasets; decision tier 2 alongside "
                         "Grinsztajn). Fetched from OpenML, cached on A: "
                         "(see SCIKIT_LEARN_DATA). See benchmarks/HIGHCARD_PLAN.md.")
    ap.add_argument("--public", action="store_true",
                    help="run the SEALED public suite behind the published "
                         "chart (report-only -- never read it to justify a "
                         "source change; see benchmarks/PUBLIC_PLAN.md).")
    ap.add_argument("--decide", action="store_true",
                    help="run the full decision tier in one go: Grinsztajn + HC "
                         "(+ their SUS/temporal variants unless --no-variants). "
                         "Results are reported and sign-tested per stratum, "
                         "never pooled -- see /experiment.")
    ap.add_argument("--synth", action="store_true",
                    help="run the frozen synthgen prior-sampled suite "
                         "(decision tier 1; see benchmarks/synthgen/).")
    ap.add_argument("--synth-suite", choices=["smoke", "screen", "full"],
                    default="screen",
                    help="with --synth, which frozen suite (default: screen).")
    ap.add_argument("--synth-n", type=int, default=None,
                    help="with --synth, run only the first N suite ids "
                         "(deterministic prefix, pairing-safe).")
    ap.add_argument("--variants", action="store_true", default=None,
                    help="add the SUS under-sampled twins and the temporal-split "
                         "variants for the selected suites (default: on for "
                         "--decide, off otherwise).")
    ap.add_argument("--no-variants", dest="variants", action="store_false",
                    help="suppress the variant families (see --variants).")
    ap.add_argument("--list-datasets", action="store_true",
                    help="print the datasets this run WOULD use, grouped by "
                         "stratum, then exit. Registration is lazy, so this "
                         "downloads nothing.")
    ap.add_argument("--models", nargs="+", default=None,
                    metavar="MODEL",
                    help=("limit to specific runners, e.g. "
                          "--models ChimeraBoost CatBoost sklearn_HGB. "
                          f"Available: {list(RUNNERS)}"))
    ap.add_argument("--lr", type=float, default=None,
                    help="ChimeraBoost learning rate (default: auto).")
    ap.add_argument("--chimera-depth", type=int, default=6,
                    help="ChimeraBoost tree depth (default: 6).")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience for ALL models "
                         "(default: %d)." % PATIENCE)
    ap.add_argument("--ensemble-n", type=int, default=None, dest="ensemble_n",
                    help="number of members for the ChimeraBoostEns10 bagged "
                         "runner (default: %d)." % ENSEMBLE_N)
    ap.add_argument("--no-ordered-boosting", dest="ordered_boosting",
                    action="store_false", default=True,
                    help="disable ChimeraBoost's LOO leaf correction.")
    ap.add_argument("--chimera-ordered-boosting", action="store_true",
                    dest="force_ordered",
                    help="force ordered_boosting=True for BOTH estimator classes "
                         "(default: each class default; backtest arm).")
    ap.add_argument("--chimera-subsample", type=float, default=1.0,
                    dest="chimera_subsample",
                    help="ChimeraBoost row subsample fraction; "
                         "MVS sampling when < 1.0 (default: 1.0 = off).")
    ap.add_argument("--chimera-colsample", type=float, default=None,
                    dest="chimera_colsample",
                    help="ChimeraBoost per-tree column subsample fraction "
                         "(default: None = the library auto default, i.e. "
                         "1.0 single / 0.85 bagged members). Applies to the "
                         "bagged arms too (B3 member grid).")
    ap.add_argument("--chimera-mcw", type=float, default=None,
                    dest="chimera_mcw",
                    help="ChimeraBoost min_child_weight (default: None = the "
                         "classifier's size-adaptive auto rule).")
    ap.add_argument("--chimera-cat-combinations", action="store_true",
                    default=False, dest="cat_combinations",
                    help="enable 2-way categorical feature combinations "
                         "(default: off).")
    ap.add_argument("--chimera-cat-smoothing", type=float, default=None,
                    dest="cat_smoothing",
                    help="Bayesian pseudocount in the ordered target-statistic "
                         "denominator (default: None = class default 1.0).")
    ap.add_argument("--chimera-lei", type=int, default=None,
                    dest="leaf_estimation_iterations",
                    help="leaf estimation iterations: additional Newton steps to "
                         "refine leaf values after tree structure is fixed. "
                         "None = use each class default (Regressor=1, Classifier=3). "
                         "Only applies on the non-LOO path.")
    ap.add_argument("--chimera-linear-leaves", action="store_true",
                    dest="linear_leaves",
                    help="enable per-leaf linear models (regression + binary; "
                         "multiclass falls back to constant leaves). Default off.")
    ap.add_argument("--chimera-linear-leaves-auto", action="store_true",
                    dest="linear_leaves_auto",
                    help="regressor linear_leaves=None: fit constant + linear "
                         "variants and keep the validation winner (~2x fit). "
                         "Classifier keeps its own auto default.")
    ap.add_argument("--chimera-linear-lambda", type=float, default=1.0,
                    dest="linear_lambda",
                    help="ridge penalty on per-leaf linear slopes (default 1.0).")
    ap.add_argument("--chimera-cross-features", action="store_true",
                    dest="cross_features",
                    help="force cross_features=True (regression + binary; "
                         "multiclass skips). Default: class default (None = "
                         "on where applicable).")
    ap.add_argument("--chimera-no-cross-features", action="store_true",
                    dest="no_cross_features",
                    help="force cross_features=False (ablation arm vs the "
                         "shipped on-where-applicable default).")
    ap.add_argument("--chimera-no-linear-leaves", action="store_true",
                    dest="no_linear_leaves",
                    help="force linear_leaves=False for both classes (ablation "
                         "arm vs the class defaults).")
    ap.add_argument("--chimera-refit-full", action="store_true",
                    dest="refit_full",
                    help="refit_full=True on the single ChimeraBoost arm: "
                         "retrain the ES winner on 100%% of train at the "
                         "selected budget (REFIT_PLAN.md A/B arm).")
    ap.add_argument("--chimera-selection-rounds", type=int, default=None,
                    dest="selection_rounds",
                    help="cap the internal selection fits at this many rounds; "
                         "default None = class default (100).")
    ap.add_argument("--chimera-full-selection", action="store_true",
                    dest="full_selection",
                    help="selection_rounds=None: every selection variant runs "
                         "to full early stopping (pre-0.15 ablation arm).")
    ap.add_argument("--chimera-quantize", action="store_true",
                    dest="chimera_quantize",
                    help="quantize_gradients=True on every ChimeraBoost arm "
                         "(single + bagged): packed-int64 quantized-gradient "
                         "histograms (QUANT_PLAN.md).")
    ap.add_argument("--datasets", nargs="+", default=None,
                    metavar="DS",
                    help=("run only these datasets, e.g. --datasets diabetes "
                          "oml:phoneme boston. Names must match keys in DATASETS "
                          "(after --openml datasets are added)."))
    ap.add_argument("--save", nargs="?", const="auto", default=None,
                    metavar="PATH",
                    help=("Also write the full benchmark output to a file. "
                          "Pass a path, or no argument for a timestamped file "
                          "under benchmarks/results/."))
    args = ap.parse_args()

    # --decide is the decision tier: both suites in one run. They stay separate
    # STRATA in every report (CLAUDE.md requires them sign-tested apart); the
    # flag only removes the need to launch two runs by hand.
    if args.decide:
        args.grinsztajn = True
        args.highcard = True

    # Optional tee: mirror stdout to a results file so runs are inspectable
    # later. Default location is benchmarks/results/YYYYMMDD-HHMMSS.txt.
    tee = None
    if args.save is not None:
        import sys, datetime
        if args.save == "auto":
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(results_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(results_dir, f"{stamp}.txt")
        else:
            save_path = args.save
        tee_file = open(save_path, "w", encoding="utf-8")
        real_stdout = sys.stdout
        class _Tee:
            def write(self, s):
                real_stdout.write(s); tee_file.write(s); tee_file.flush()
            def flush(self):
                real_stdout.flush(); tee_file.flush()
        sys.stdout = _Tee()
        tee = (tee_file, save_path)
        print(f"# Benchmark results will be saved to: {save_path}")

    if args.patience is not None:
        PATIENCE = args.patience
    if args.ensemble_n is not None:
        ENSEMBLE_N = args.ensemble_n

    # Each requested suite contributes a "keep this key" predicate; the actual
    # pruning is ONE pass over the union at the end. Previously every suite
    # deleted all keys that weren't its own the moment it registered, so asking
    # for two suites at once (--grinsztajn --highcard, the decision pair) left
    # nothing to run -- whichever registered last wiped the other.
    keepers = []

    need_openml = (args.openml or args.no_synthetic or bool(
        args.datasets and any(d.startswith("oml:") for d in args.datasets)))
    if need_openml:
        _add_openml_datasets()
    if args.no_synthetic:
        keepers.append(lambda k: k.startswith("oml:"))

    need_grinsztajn = args.grinsztajn or bool(
        args.datasets and any(d.startswith("gr:") for d in args.datasets))
    if need_grinsztajn:
        _add_grinsztajn_datasets()
        if args.grinsztajn:
            keepers.append(lambda k: k.startswith("gr:"))

    # PMLB tuning suite; --pmlb-fold narrows to one fold.
    need_pmlb = args.pmlb or bool(
        args.datasets and any(d.startswith("pm:") for d in args.datasets))
    if need_pmlb:
        _add_pmlb_datasets()
        if args.pmlb:
            keep_fold = (f"pm:{args.pmlb_fold}/" if args.pmlb_fold else "pm:")
            keepers.append(lambda k, p=keep_fold: k.startswith(p))

    need_highcard = args.highcard or bool(
        args.datasets and any(d.startswith("hc:") for d in args.datasets))
    if need_highcard:
        _add_highcard_datasets()
        if args.highcard:
            keepers.append(lambda k: k.startswith("hc:"))

    need_public = args.public or bool(
        args.datasets and any(d.startswith("pub:") for d in args.datasets))
    if need_public:
        if not PUBLIC_DATASETS:
            ap.error("the public suite is empty pending its overlap audit -- "
                     "see benchmarks/PUBLIC_PLAN.md")
        _add_public_datasets()
        if args.public:
            keepers.append(lambda k: k.startswith("pub:"))

    need_synth = args.synth or bool(
        args.datasets and any(d.startswith("syn:") for d in args.datasets))
    if need_synth:
        _add_synth_datasets()
        if args.synth:
            import synthgen
            keep = set(synthgen.frozen_keys(args.synth_suite)[: args.synth_n])
            if not keep:
                ap.error(f"--synth suite {args.synth_suite!r} is empty -- "
                         "freeze it first (benchmarks/synthgen/freeze.py).")
            keepers.append(lambda k, s=keep: k in s)

    if keepers and not args.datasets:
        for k in [k for k in DATASETS
                  if not any(keep_it(k) for keep_it in keepers)]:
            del DATASETS[k]

    # Variants are derived from whatever survived the prune, so they are
    # registered last. Default on for --decide, off elsewhere, so a plain
    # --grinsztajn / --highcard run stays comparable with every run in history.
    need_variants = args.variants if args.variants is not None else bool(args.decide)
    if need_variants:
        _add_variant_datasets(list(DATASETS))
    elif args.datasets and any(VARIANT_SEP in d for d in args.datasets):
        # Explicitly naming a variant key implies wanting it.
        _add_variant_datasets(list(DATASETS))
        need_variants = True

    # Resolve the model set. Competitors are gated on install; XGBoost is off
    # by default (it tracks LightGBM). --models overrides everything.
    available = (list(_ALWAYS)
                 + [m for m in _OFF_BY_DEFAULT if not HAVE.get(m.lower(), False)]
                 + [m for m in _OPTIONAL if HAVE[m.lower()]])
    if args.models:
        unknown = set(args.models) - set(RUNNERS)
        if unknown:
            ap.error(f"Unknown models: {unknown}. Available: {list(RUNNERS)}")
        model_names = [m for m in args.models if m in available]
    else:
        model_names = [m for m in available if m not in _OFF_BY_DEFAULT
                       or (m == "XGBoost" and args.with_xgboost)]
    if "ChimeraBoost" not in model_names:
        ap.error("ChimeraBoost must be one of the models (it is the baseline).")

    # None = use each class's default (Regressor=False, Classifier=True).
    # --no-ordered-boosting forces False for both; --chimera-ordered-boosting
    # forces True for both (backtest arm).
    ob_override = True if args.force_ordered else (
        None if args.ordered_boosting else False)
    chimera_cfg = dict(lr=args.lr, ordered_boosting=ob_override,
                       depth=args.chimera_depth, subsample=args.chimera_subsample,
                       colsample=args.chimera_colsample,
                       mcw=args.chimera_mcw, cat_combinations=args.cat_combinations,
                       cat_smoothing=args.cat_smoothing,
                       leaf_estimation_iterations=args.leaf_estimation_iterations,
                       linear_leaves="auto" if args.linear_leaves_auto
                       else ("off" if args.no_linear_leaves
                             else args.linear_leaves),
                       linear_lambda=args.linear_lambda,
                       cross_features="off" if args.no_cross_features
                       else args.cross_features,
                       selection_rounds="full" if args.full_selection
                       else args.selection_rounds,
                       quantize=args.chimera_quantize,
                       refit_full=args.refit_full)

    # Split the thread budget across parallel jobs: GBDT thread scaling is
    # sublinear, so running J seeds at threads/J each beats one fit at all cores.
    total_threads = args.threads or os.cpu_count() or 1
    jobs = max(1, args.jobs)
    threads_per = max(1, total_threads // jobs)

    selected = [ds for ds in DATASETS
                if not (args.datasets and ds not in args.datasets)
                and not (args.only == "regression" and _task_of(ds) != "regression")
                and not (args.only == "classification" and _task_of(ds) == "regression")]

    if args.list_datasets:
        strata = summarize.split_strata(selected)
        for stratum, ds_names in strata.items():
            print(f"\n{summarize.stratum_label(stratum)}  ({len(ds_names)})")
            for ds in ds_names:
                print(f"  {ds}  [{_task_of(ds)}]")
        n_str = len(strata)
        print(f"\ntotal: {len(selected)} datasets in {n_str} "
              f"{'stratum' if n_str == 1 else 'strata'}")
        return

    print("Detected competitors:",
          ", ".join(k for k, v in HAVE.items() if v) or "none (sklearn only)")
    print(f"scale={args.scale}  seeds={args.seeds}  jobs={jobs}  "
          f"threads/job={threads_per}  max_iter={MAX_ITERS}  patience={PATIENCE}  "
          f"models={model_names}"
          + (f"  chimera_lr={args.lr}" if args.lr else "")
          + ("  ordered_boosting=off" if not args.ordered_boosting else "")
          + (f"  subsample={args.chimera_subsample}" if args.chimera_subsample < 1.0 else "")
          + ("  cat_combinations=on" if args.cat_combinations else "")
          + (f"  cat_smoothing={args.cat_smoothing}"
             if args.cat_smoothing is not None else "")
          + (f"  synth={args.synth_suite}" if args.synth else "")
          + "\n")

    # Run every (dataset, seed) draw, in parallel processes unless jobs == 1.
    tasks = [(ds, s, args.scale, threads_per, model_names, chimera_cfg,
              PATIENCE, ENSEMBLE_N, need_openml, need_grinsztajn, need_pmlb,
              need_synth, need_highcard, need_public, need_variants)
             for ds in selected for s in range(args.seeds)]
    total_tasks = len(tasks)

    # Live-progress sidecar so `bench_status.py` (the /bench command) can report
    # how far along a run is from another process. Written next to the results
    # file when --save is on; otherwise to results/_running.progress. It records
    # completed/total task counts (one task = one dataset-seed draw) and the
    # config line, and is marked done at the end / removed on a clean exit.
    prog = _Progress(tee[1] if tee else None, total_tasks, model_names, args)

    collected = defaultdict(dict)   # collected[ds][seed] = (meta, out)
    done = 0
    if jobs == 1:
        for t in tasks:
            ds, seed, meta, out = _run_seed_task(t)
            collected[ds][seed] = (meta, out)
            done += 1
            prog.update(done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_run_seed_task, t) for t in tasks]
            for fut in as_completed(futs):
                ds, seed, meta, out = fut.result()
                collected[ds][seed] = (meta, out)
                done += 1
                prog.update(done)
    prog.finish()

    metric_name = {"regression": "RMSE (lower better)",
                   "binary": "F1 macro shown, Brier scored (lower better)",
                   "multiclass": "F1 macro shown, Brier scored (lower better)"}
    # {competitor: {dataset: fit-time ratio vs ChimeraBoost}}; keyed by dataset
    # so the summary can average within a stratum rather than across the run.
    speed_acc = {m: {} for m in model_names if m != "ChimeraBoost"}
    raw_records = []      # one row per (dataset, model, seed); feeds make_tables
    dataset_meta = {}

    for ds_name in selected:
        seed_map = collected.get(ds_name)
        if not seed_map:
            continue
        dataset_meta[ds_name] = seed_map[next(iter(seed_map))][0]
        task = dataset_meta[ds_name]["task"]

        results = {m: [] for m in model_names}
        times = {m: [] for m in model_names}
        iters = {m: [] for m in model_names}
        briers = {m: [] for m in model_names}
        for s in range(args.seeds):
            if s not in seed_map:
                continue
            for name, res in seed_map[s][1].items():
                if res is None:
                    continue
                metrics, secs, pred_secs, best_it = res
                results[name].append(metrics["primary"])
                times[name].append(secs)
                if metrics.get("brier") is not None:
                    briers[name].append(metrics["brier"])
                if best_it is not None:
                    iters[name].append(best_it)
                raw_records.append({
                    "dataset": ds_name, "model": name, "seed": s,
                    "metrics": metrics, "fit_time": secs,
                    "predict_time": pred_secs,
                    "best_iter": int(best_it) if best_it is not None else None,
                })

        print(f"### {ds_name}  [{task}]  metric={metric_name[task]}")
        for name in model_names:
            if not results[name]:
                continue
            sc = np.array(results[name])
            tm = np.array(times[name])
            disp = (-sc if task == "regression" else sc)
            it_str = f"  trees~{int(np.mean(iters[name]))}" if iters[name] else ""
            star = " <-- ours" if name == "ChimeraBoost" else ""
            # Show Brier next to F1 on classification: Brier is what the summary
            # win rate and every ship decision actually score, so it should be
            # visible on the row rather than inferred.
            br = (f"  brier {np.mean(briers[name]):.4f}"
                  if briers[name] else "")
            print(f"  {name:14s} {disp.mean():8.4f} +/- {disp.std():.4f}{br}"
                  f"   fit {tm.mean():6.2f}s{it_str}{star}")

        if results["ChimeraBoost"]:
            our_time = np.mean(times["ChimeraBoost"])
            for name in speed_acc:
                if results[name]:
                    speed_acc[name][ds_name] = (
                        np.mean(times[name]) / max(our_time, 1e-9))
        print()

    # ---- summary verdict ----
    # Scored on the per-dataset primary metric (RMSE regression / Brier
    # classification, both lower=better) via summarize, so this agrees with the
    # Pareto chart and the ship gate by construction. The old summary averaged
    # relative percentage gaps on F1 with no near-solved guard -- the statistic
    # that produced this project's -144% and -8e21% readings.
    run_data = {"config": {}, "datasets": dataset_meta, "records": raw_records}
    strata = summarize.split_strata(dataset_meta)
    for stratum, ds_names in strata.items():
        sub = summarize.subset(run_data, ds_names)
        primary = summarize.primary_scores(sub)
        n_excluded = len(ds_names) - len(primary)
        title = (f"SUMMARY [{summarize.stratum_label(stratum)}]"
                 if len(strata) > 1 else "SUMMARY")
        print("=" * 78)
        print(f"{title} -- head-to-head win rate "
              "(RMSE reg / Brier clf)")
        print("=" * 78)
        for rname in speed_acc:
            pw = _pairwise_winrate(primary, "ChimeraBoost", rname)
            if pw is None:
                continue
            rate, lo, hi, w, l, t, med = pw
            sp = [speed_acc[rname][ds] for ds in ds_names
                  if ds in speed_acc[rname]]
            ci = f" [{lo:.0f}-{hi:.0f}]" if lo is not None else ""
            # speed ratio >1 means ChimeraBoost is faster
            speed = f"x{np.mean(sp):.2f}" if sp else "--"
            print(f"  vs {rname:12s}  win {rate:5.1f}%{ci}  (W{w}-L{l}-T{t})  "
                  f"median gap {med:+6.2f}%   speed {speed}   "
                  f"-> {_verdict(rname, rate)}")
        print(f"\n  scored on {len(primary)} of {len(ds_names)} datasets"
              + (f" ({n_excluded} near-solved, excluded)" if n_excluded else "")
              + "; ties count 1/2; CI = 95% bootstrap over datasets.")
        print()
    if len(strata) > 1:
        print("Strata are reported separately and never pooled -- the decision "
              "suites answer\ndifferent questions, and a variant reuses its "
              "parent's rows. Sign-test each\nwith: compare_runs.py BASE NEW "
              "--by-suite\n")
    if tee is not None:
        import sys
        # Sidecar JSON: every metric for every (dataset, model, seed), plus
        # dataset metadata (task, size, has_cats). Used by make_tables.py.
        json_path = tee[1].rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            _json.dump({
                "config": {
                    "seeds": args.seeds, "max_iters": MAX_ITERS,
                    "patience": PATIENCE, "ensemble_n": ENSEMBLE_N,
                    "threads_per_model": threads_per,
                    "total_threads": total_threads,
                    "jobs": jobs,
                    # Runs written before this marker charged predict + metric
                    # computation to fit_time (see _finish). Their speed columns
                    # are NOT comparable with these; summarize warns on a mix.
                    "timing": "fit_only",
                },
                "datasets": dataset_meta,
                "records": raw_records,
            }, jf, indent=2)
        print(f"# Saved results to: {tee[1]}")
        print(f"# Saved raw data to: {json_path}")
        sys.stdout = real_stdout
        tee[0].close()


if __name__ == "__main__":
    main()