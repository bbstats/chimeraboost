"""Dataset tiers for the cascade, with a download-once persistent cache.

Reuses the loaders already wired in ``run_benchmarks`` (Grinsztajn, high-card,
PMLB) -- never re-implements a fetch. Each tier is a list of ``run_benchmarks``
DATASETS keys; ``load(key)`` returns ``(X, y, cat, task)`` and caches the result
to ``research/cache/data/`` keyed by the dataset key, so a dataset is fetched at
most once ever (not once per idea, not once per seed).

Tiers (categorical-first, never TabArena):
  T0  handful (8)   -- fast go/kill on paired curves. Seconds per fit-pair.
  T1  medium (14)   -- the hc: high-cardinality suite: real-categorical breadth,
                       the categorical-aware promotion gate.
  T2  large (73)    -- Grinsztajn-59 numeric breadth + the hc: suite -> full
                       sign test + blended Pareto. The expensive tier.
  HOLDOUT           -- PMLB holdout fold, out-of-sample generalization.

TIER MEMBERSHIP CHANGED 2026-08-02 (issue #71). Every tier used to be built on
the 29-dataset OpenML one-shot gate, which was retired on 2026-07-27 (eight of
its datasets were exact-name Grinsztajn members, so it partly re-scored the data
it was meant to check) and whose registry has now been deleted outright, along
with the last fetch_openml call in the harness. T1 moved to hc:, the project's
audited real-categorical suite and what decisions actually run on today. T0 kept
its two gr: members and refilled the other six by role, not by name: one numeric
binary, two categorical binaries, two categorical multiclass, plus a categorical
regression the old handful lacked. The cascade's own verdicts predate the swap
and are not comparable dataset-for-dataset -- see SUMMARY.md.

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
# gr:/hc:/pm: namespaces are registered lazily by _ensure_registered().
# ---------------------------------------------------------------------------

# T0: a handful spanning the known levers' sweet-spots -- high-signal binary, an
# interaction-heavy regression, real categorical sets (low + high card), and
# multiclass. Row-capped (see T0_MAX_ROWS) so a fit-pair is seconds; the hc:
# members are the small end of that suite so the first download is cheap too
# (the largest, kick, is a 2.6 MB parquet).
#
# Every categorical slot is filled from hc:, not from Grinsztajn's clf_cat /
# reg_cat folders. Those folders are named for the paper's *original* feature
# types, but the HuggingFace mirror serves the paper's transformed CSVs, in which
# the categoricals are already numerically encoded -- cats="auto" detects zero
# string columns in every one of them. A gr:clf_cat key exercises nothing this
# tier is here to screen.
T0_HANDFUL = [
    "gr:clf_num/electricity",    # binary, high-signal numeric
    "gr:clf_num/covertype",      # binary, large numeric (capped)
    "gr:reg_num/pol",            # regression, interaction-heavy
    "hc:kick",                   # binary, high-card categoricals (Model, 1063)
    "hc:kdd_ipums_la_97-small",  # binary, mid-card code categoricals (191)
    "hc:Moneyball",              # regression with categoricals (Team, 39)
    "hc:eucalyptus",             # multiclass (5), real categoricals
    "hc:cjs",                    # multiclass (6), high-card categoricals
]
T0_MAX_ROWS = 8000   # cap rows on T0 only, to keep go/kill in well under a minute


# T1: the frozen hc: suite -- 14 audited real-categorical datasets, the
# categorical-aware promotion gate.
def _highcard_keys():
    return [k for k in rb.DATASETS if k.startswith("hc:")]


# T2 large: full numeric breadth (Grinsztajn) + the categorical tier. Built
# lazily (membership depends on registration).
def _grinsztajn_keys():
    return [k for k in rb.DATASETS if k.startswith("gr:")]


def _pmlb_holdout_keys():
    return [k for k in rb.DATASETS if k.startswith("pm:holdout/")]


def _ensure_registered():
    """Register every namespace loader (idempotent, cheap)."""
    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    rb._add_pmlb_datasets()


def tier_keys(tier):
    """Return the dataset keys for a named tier. ``tier`` in
    {T0, T1, T2, HOLDOUT}."""
    _ensure_registered()
    tier = tier.upper()
    if tier == "T0":
        return list(T0_HANDFUL)
    if tier == "T1":
        return _highcard_keys()
    if tier == "T2":
        # De-dup while preserving order: Grinsztajn numeric breadth, then the
        # categorical tier. The two suites are audited disjoint, but the de-dup
        # stays -- it is what makes T2 safe to extend with another namespace.
        seen, out = set(), []
        for k in _grinsztajn_keys() + _highcard_keys():
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
