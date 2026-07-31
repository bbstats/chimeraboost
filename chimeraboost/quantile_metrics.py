"""Scoring for a predicted quantile grid.

Everything here takes ``Q`` of shape (n_samples, n_quantiles) -- the matrix
`ChimeraBoostQuantileRegressor.predict` returns -- alongside the ``taus`` it
was fitted on. Nothing here is used during fitting; these are for judging a
fitted model.

The three questions worth asking of a predictive distribution, and the
function that answers each:

  * Is each quantile in the right place?      `pinball_loss` (per level)
  * Is the distribution as a whole right?     `crps`
  * Do the intervals hold what they claim?    `interval_coverage` (plus width,
    because a wide enough interval covers everything and says nothing)
"""

import numpy as np


def _check(y, Q, taus):
    y = np.asarray(y, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2-D (n_samples, n_quantiles); got shape "
                         f"{Q.shape}.")
    if Q.shape[0] != y.shape[0]:
        raise ValueError(f"Q has {Q.shape[0]} rows but y has {y.shape[0]}.")
    if Q.shape[1] != taus.shape[0]:
        raise ValueError(f"Q has {Q.shape[1]} columns but taus has "
                         f"{taus.shape[0]} entries.")
    return y, Q, taus


def pinball_loss(y, Q, taus, sample_weight=None, average=False):
    """Pinball (quantile) loss, one value per level.

    ``average=True`` collapses to the mean over levels. Returns a ``(K,)``
    array otherwise -- the per-level breakdown is usually the interesting
    part, since a model can be excellent at the median and useless in the
    tails."""
    y, Q, taus = _check(y, Q, taus)
    r = y[:, None] - Q
    loss = np.maximum(taus * r, (taus - 1.0) * r)
    per_level = np.average(loss, axis=0, weights=sample_weight)
    return float(per_level.mean()) if average else per_level


def crps(y, Q, taus, sample_weight=None):
    """Continuous ranked probability score, approximated as the mean pinball
    loss across the grid.

    Convention note: CRPS is exactly ``2 * integral of pinball over tau``, so
    this returns half the textbook CRPS. The factor is constant, so rankings
    and relative comparisons are unaffected; it is the same convention the
    booster's early-stopping metric uses, which keeps the two numbers directly
    comparable. Grid resolution bounds the accuracy -- a 19-point grid
    approximates the integral, it does not evaluate it."""
    return pinball_loss(y, Q, taus, sample_weight, average=True)


def interval_coverage(y, Q, taus, sample_weight=None):
    """Empirical coverage and mean width of every symmetric (lo, hi) pair.

    Pairs the k-th level with the (K-1-k)-th and keeps those whose nominal
    levels are symmetric about the median, so a grid like 0.05...0.95 yields
    the 90%, 80%, ... intervals. Returns a list of dicts with keys
    ``lo``, ``hi``, ``nominal``, ``coverage``, ``width``.

    Coverage alone is not a score: an interval from minus to plus infinity
    covers everything. Read the two together."""
    y, Q, taus = _check(y, Q, taus)
    K = taus.shape[0]
    out = []
    for k in range(K // 2):
        j = K - 1 - k
        nominal = taus[j] - taus[k]
        if abs((taus[k] + taus[j]) - 1.0) > 1e-9:
            continue                      # not a symmetric pair; skip
        inside = ((y >= Q[:, k]) & (y <= Q[:, j])).astype(np.float64)
        out.append({
            "lo": float(taus[k]),
            "hi": float(taus[j]),
            "nominal": float(nominal),
            "coverage": float(np.average(inside, weights=sample_weight)),
            "width": float(np.average(Q[:, j] - Q[:, k],
                                      weights=sample_weight)),
        })
    return out


def crossing_rate(Q):
    """Fraction of adjacent quantile pairs that come out in the wrong order.

    Zero for `ChimeraBoostQuantileRegressor`, whose delivered rows are sorted;
    this function exists to prove that rather than to fix anything.
    Independently fitted per-level models are not zero."""
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape[1] < 2:
        return 0.0
    d = np.diff(Q, axis=1)
    return float((d < 0).sum()) / float(d.size)


def quantile_report(y, Q, taus, sample_weight=None):
    """Everything above in one dict: ``crps``, ``pinball`` (per level),
    ``taus``, ``intervals``, ``crossing_rate``."""
    y, Q, taus = _check(y, Q, taus)
    return {
        "crps": crps(y, Q, taus, sample_weight),
        "pinball": pinball_loss(y, Q, taus, sample_weight),
        "taus": taus,
        "intervals": interval_coverage(y, Q, taus, sample_weight),
        "crossing_rate": crossing_rate(Q),
    }


def format_report(report, title=None):
    """Render `quantile_report` as a fixed-width text table."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"CRPS (mean pinball over the grid): {report['crps']:.6f}")
    lines.append(f"crossing rate: {report['crossing_rate']:.6f}")
    lines.append("")
    lines.append(f"{'tau':>8s}{'pinball':>12s}")
    for t, p in zip(report["taus"], report["pinball"]):
        lines.append(f"{t:8.2f}{p:12.6f}")
    if report["intervals"]:
        lines.append("")
        lines.append(f"{'interval':>10s}{'nominal':>10s}{'coverage':>10s}"
                     f"{'width':>10s}")
        for iv in report["intervals"]:
            lines.append(f"{iv['lo']:.2f}-{iv['hi']:.2f}".rjust(10)
                         + f"{iv['nominal']:10.2f}{iv['coverage']:10.4f}"
                           f"{iv['width']:10.4f}")
    return "\n".join(lines)
