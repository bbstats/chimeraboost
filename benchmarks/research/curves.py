"""Paired validation-curve comparison -- the cascade's cheap go/kill signal.

Given a baseline curve ``b`` and a variant curve ``v`` recorded on the SAME
train/val split (so the comparison is paired, not across-split), summarize how
the variant's per-iteration validation loss trajectory differs from baseline.
Lower loss is better throughout (logloss / RMSE-space), so negative deltas favor
the variant.

These are deliberately cheap scalar reductions of two equal-or-unequal-length
loss arrays; the runner computes them from ``validation_history_`` with no extra
fits. The cascade thresholds them (see ``cascade``).
"""

import numpy as np


def _as_array(curve):
    a = np.asarray(curve, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("curve must be a non-empty 1-D sequence of losses.")
    return a


def best_val(curve):
    """The minimum validation loss along the curve -- what early stopping picks."""
    return float(np.min(_as_array(curve)))


def best_val_delta(baseline, variant):
    """``min(variant) - min(baseline)``: the primary fast signal. Negative means
    the variant reaches a lower best validation loss than baseline (better)."""
    return best_val(variant) - best_val(baseline)


def best_val_delta_pct(baseline, variant):
    """``best_val_delta`` as a fraction of the baseline's best loss (signed).
    Comparable across datasets of different loss scales."""
    bb = best_val(baseline)
    return best_val_delta(baseline, variant) / (abs(bb) + 1e-12)


def _aligned(baseline, variant):
    """Truncate both curves to their common length for pointwise comparison.
    Curves can differ in length when a depth-0 tree stops one early."""
    b, v = _as_array(baseline), _as_array(variant)
    n = min(b.size, v.size)
    return b[:n], v[:n]


def dominance(baseline, variant):
    """Fraction of iterations where ``variant <= baseline`` (variant at least as
    good pointwise). 1.0 = variant dominates everywhere (strong promote); ~0.0 =
    dominated everywhere (fast kill); 0.5 = curves interleave."""
    b, v = _aligned(baseline, variant)
    return float(np.mean(v <= b + 1e-12))


def early_signal(baseline, variant, k=50):
    """Sign of the mean ``variant - baseline`` gap over the first ``k`` iterations.
    Lets us kill before convergence when the early trajectory is already clearly
    worse. Returns the signed mean gap (negative favors the variant)."""
    b, v = _aligned(baseline, variant)
    k = min(k, b.size)
    return float(np.mean(v[:k] - b[:k]))


def area_between(baseline, variant):
    """Signed mean gap ``variant - baseline`` over the common length -- a
    magnitude estimate of how far apart the trajectories run. Negative favors the
    variant."""
    b, v = _aligned(baseline, variant)
    return float(np.mean(v - b))


def compare(baseline, variant, k=50):
    """All paired-curve statistics for one (baseline, variant) curve pair."""
    return {
        "best_val_baseline": best_val(baseline),
        "best_val_variant": best_val(variant),
        "best_val_delta": best_val_delta(baseline, variant),
        "best_val_delta_pct": best_val_delta_pct(baseline, variant),
        "dominance": dominance(baseline, variant),
        "early_signal": early_signal(baseline, variant, k),
        "area_between": area_between(baseline, variant),
        "len_baseline": int(_as_array(baseline).size),
        "len_variant": int(_as_array(variant).size),
    }
