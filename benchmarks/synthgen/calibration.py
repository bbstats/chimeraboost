"""Bootstrap sampler over the harvested real-dataset marginals.

`corpus_marginals.json` (checked in; produced by harvest_metadata.py) holds one
row per real public dataset: [n, d, task, cat_frac, n_classes, missing_rate,
majority_frac, max_card, curated]. Sampling a row jointly preserves the real
correlations between size, width, task type and categorical make-up. TabArena
members were excluded at harvest time (sealed holdout).
"""
import json
import os

_SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "corpus_marginals.json")
_cache = None

# Minimal fallback so generation never dies without the snapshot (shape only —
# rough echoes of the in-repo decision suites). The checked-in snapshot is the
# real calibration; a missing snapshot is a packaging bug, not a normal path.
_FALLBACK_ROWS = [
    [3000, 10, 0, 0.0, 0, 0.0, 0.0, 0, 1], [8000, 30, 0, 0.0, 0, 0.0, 0.0, 0, 1],
    [15000, 20, 0, 0.3, 0, 0.02, 0.0, 12, 1], [1500, 8, 0, 0.0, 0, 0.0, 0.0, 0, 0],
    [6000, 15, 1, 0.2, 2, 0.0, 0.7, 8, 1], [3000, 36, 1, 1.0, 2, 0.0, 0.52, 3, 1],
    [45000, 14, 1, 0.5, 2, 0.01, 0.76, 42, 1], [800, 20, 1, 0.0, 2, 0.0, 0.65, 0, 0],
    [2000, 25, 2, 0.4, 4, 0.0, 0.4, 10, 1], [5000, 40, 2, 0.0, 6, 0.0, 0.3, 0, 0],
    [1700, 6, 2, 1.0, 4, 0.0, 0.7, 4, 1], [10000, 16, 2, 0.1, 3, 0.05, 0.5, 20, 0],
]


def _load():
    global _cache
    if _cache is None:
        if os.path.exists(_SNAPSHOT):
            snap = json.load(open(_SNAPSHOT, encoding="utf-8"))
            rows = snap["rows"]
        else:
            snap = {"source": "built-in fallback (snapshot missing!)"}
            rows = _FALLBACK_ROWS
        curated = [r for r in rows if r[8] == 1]
        broad = [r for r in rows if r[8] == 0] or curated
        _cache = {"curated": curated, "broad": broad, "source": snap.get("source", "?")}
    return _cache


def sample_row(rng, prefer_curated=0.5):
    """Draw one real-dataset metadata row (jointly, preserving correlations).

    Returns dict(n, d, task, cat_frac, n_classes, missing_rate, majority_frac,
    max_card). task in {"regression", "binary", "multiclass"}.
    """
    corpus = _load()
    pool = corpus["curated"] if rng.random() < prefer_curated else corpus["broad"]
    row = pool[int(rng.integers(0, len(pool)))]
    task = ("regression", "binary", "multiclass")[int(row[2])]
    return {
        "n": int(row[0]), "d": int(row[1]), "task": task,
        "cat_frac": float(row[3]), "n_classes": int(row[4]),
        "missing_rate": float(row[5]), "majority_frac": float(row[6]),
        "max_card": int(row[7]),
    }


def source_note():
    return _load()["source"]
