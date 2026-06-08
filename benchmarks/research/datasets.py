"""Dataset tiers for the cascade, with a download-once persistent cache.

Reuses the loaders already wired in ``run_benchmarks`` (OpenML, Grinsztajn,
PMLB) -- never re-implements a fetch. Each tier is a list of ``run_benchmarks``
DATASETS keys; ``load(key)`` returns ``(X, y, cat, task)`` and caches the result
to ``research/cache/data/`` keyed by the dataset key, so a dataset is fetched at
most once ever (not once per idea, not once per seed).

Tiers (categorical-first, never TabArena):
  T0  handful (~8)  -- fast go/kill on paired curves. Seconds per fit-pair.
  T1  medium (~)    -- OpenML real-categorical breadth (the categorical-aware
                       gate; OpenML is used ONCE as a gate, never iterated).
  T2  large         -- Grinsztajn-59 numeric breadth + OpenML + the categorical
                       tier -> full sign test + blended Pareto.
  HOLDOUT           -- PMLB holdout fold, out-of-sample generalization.

GUARDRAIL: no tier may reference TabArena. See feedback_tabarena_lite_is_sealed_holdout.
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_benchmarks as rb  # noqa: E402

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cache", "data")


# ---------------------------------------------------------------------------
# Tier membership. Keys are run_benchmarks DATASETS keys; the loaders for the
# oml:/gr:/pm: namespaces are registered lazily by _ensure_registered().
# ---------------------------------------------------------------------------

# T0: a handful spanning the known levers' sweet-spots -- high-signal binary, an
# interaction-heavy regression, real string-categorical sets (low + high card),
# and multiclass. Row-capped (see T0_MAX_ROWS) so a fit-pair is seconds.
T0_HANDFUL = [
    "oml:electricity",        # binary, high-signal numeric
    "gr:clf_num/covertype",   # binary, large numeric (capped)
    "gr:reg_num/pol",         # regression, interaction-heavy
    "oml:kr-vs-kp",           # binary, all real categoricals
    "oml:adult",              # binary, mixed real categoricals (high card)
    "oml:bank-marketing",     # binary, mixed real categoricals
    "oml:car",                # multiclass, all real categoricals
    "oml:splice",             # multiclass, real categoricals
]
T0_MAX_ROWS = 8000   # cap rows on T0 only, to keep go/kill in well under a minute

# T1: OpenML real-categorical breadth -- the categorical-aware medium tier. Used
# ONCE as a gate (not an iteration target). The cats="auto" members of the
# (Part-B-extended) OpenML suite.
def _openml_cat_keys():
    return [f"oml:{name}" for name, spec in rb.OPENML_SUITE.items()
            if spec.get("cats") == "auto"]


# T2 large: full numeric breadth (Grinsztajn) + the whole OpenML suite + the
# categorical tier. Built lazily (membership depends on registration).
def _grinsztajn_keys():
    return [k for k in rb.DATASETS if k.startswith("gr:")]


def _openml_keys():
    return [f"oml:{name}" for name in rb.OPENML_SUITE]


def _pmlb_holdout_keys():
    return [k for k in rb.DATASETS if k.startswith("pm:holdout/")]


def _ensure_registered():
    """Register every namespace loader (idempotent, cheap)."""
    rb._add_openml_datasets()
    rb._add_grinsztajn_datasets()
    rb._add_pmlb_datasets()


def tier_keys(tier):
    """Return the dataset keys for a named tier. ``tier`` in
    {T0, T1, T2, HOLDOUT}."""
    _ensure_registered()
    tier = tier.upper()
    if tier == "T0":
        return list(T0_HANDFUL)
    if tier == "T1":
        return _openml_cat_keys()
    if tier == "T2":
        # De-dup while preserving order: Grinsztajn numeric breadth, then the
        # OpenML suite (includes the categorical tier from Part B).
        seen, out = set(), []
        for k in _grinsztajn_keys() + _openml_keys():
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out
    if tier == "HOLDOUT":
        return _pmlb_holdout_keys()
    raise ValueError(f"unknown tier {tier!r}; expected T0/T1/T2/HOLDOUT.")


# ---------------------------------------------------------------------------
# Persistent cache: fetch each dataset at most once, ever.
# ---------------------------------------------------------------------------

def _cache_path(key):
    safe = key.replace(":", "__").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")


def load(key, max_rows=None):
    """Return ``(X, y, cat, task)`` for a dataset key, from cache when present.

    ``max_rows`` (e.g. for T0) seeded-subsamples after load and is NOT part of
    the cache key -- the full dataset is cached once, then capped per call."""
    if "tabarena" in key.lower():
        raise ValueError("TabArena is a sealed holdout; the cascade must not "
                         "load it.")
    _ensure_registered()
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            X, y, cat, task = pickle.load(f)
    else:
        builder = rb.DATASETS[key]
        X, y, cat, task = builder(1.0, np.random.default_rng(0))
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((X, y, cat, task), f)
    if max_rows is not None and len(y) > max_rows:
        idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y, cat, task
