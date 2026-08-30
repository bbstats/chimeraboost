"""Scoring for a predicted quantile grid.

Everything here takes ``Q`` of shape (n_samples, n_quantiles) -- the matrix
`ChimeraBoostQuantileRegressor.predict` returns -- alongside the ``taus`` it
was fitted on. Nothing here is used during fitting; these are for judging a
fitted model.

The questions worth asking of a predictive distribution, and the function
that answers each:

  * Is each quantile in the right place?      `pinball_loss` (per level)
  * Is the distribution as a whole right?     `crps`
  * Is it right by a useful margin?           `quantile_skill_score` -- 1 is
    perfect, 0 is no better than ignoring every feature
  * Do the intervals hold what they claim?    `interval_coverage` (plus width,
    because a wide enough interval covers everything and says nothing)
  * Coverage and width in ONE number?         `interval_score`, the proper
    rule that trades them off; `sharpness` is the width half alone
  * Where is the model wrong?                 `pit_values` / `pit_histogram`.
    Coverage says a band is too narrow; PIT says where.
  * Do the levels come out in order?          `crossing_rate`

`quantile_report` runs the lot and `format_report` prints it.
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


def _symmetric_pairs(taus):
    """Yield ``(k, j, nominal)`` for each pair of levels symmetric about the
    median, outermost first. A grid like 0.05...0.95 gives the 90%, 80%, ...
    intervals; levels with no partner are skipped.

    Shared by every interval-shaped metric so they can never disagree about
    which pairs exist.
    """
    K = taus.shape[0]
    for k in range(K // 2):
        j = K - 1 - k
        if abs((taus[k] + taus[j]) - 1.0) > 1e-9:
            continue
        yield k, j, float(taus[j] - taus[k])


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


def crps(y, Q, taus, sample_weight=None, convention="half"):
    """Continuous ranked probability score, approximated as the mean pinball
    loss across the grid.

    Convention: CRPS is exactly ``2 * integral of pinball over tau``, so the
    default ``"half"`` returns half the textbook value. The factor is
    constant, so rankings and relative comparisons are unaffected, and it
    matches the booster's early-stopping metric, which keeps the two numbers
    directly comparable. Pass ``convention="full"`` for the textbook scale
    when comparing against another library -- `properscoring` and
    `scoringrules` both report the full value.

    Grid resolution bounds the accuracy either way: a 19-point grid
    approximates the integral, it does not evaluate it."""
    if convention not in ("half", "full"):
        raise ValueError(
            f'convention must be "half" or "full"; got {convention!r}.')
    v = pinball_loss(y, Q, taus, sample_weight, average=True)
    return v if convention == "half" else 2.0 * v


def interval_coverage(y, Q, taus, sample_weight=None):
    """Empirical coverage and mean width of every symmetric (lo, hi) pair.

    Pairs the k-th level with the (K-1-k)-th and keeps those whose nominal
    levels are symmetric about the median, so a grid like 0.05...0.95 yields
    the 90%, 80%, ... intervals. Returns a list of dicts with keys
    ``lo``, ``hi``, ``nominal``, ``coverage``, ``width``.

    Coverage alone is not a score: an interval from minus to plus infinity
    covers everything. Read the two together."""
    y, Q, taus = _check(y, Q, taus)
    out = []
    for k, j, nominal in _symmetric_pairs(taus):
        inside = ((y >= Q[:, k]) & (y <= Q[:, j])).astype(np.float64)
        out.append({
            "lo": float(taus[k]),
            "hi": float(taus[j]),
            "nominal": nominal,
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


def interval_score(y, Q, taus, sample_weight=None):
    """Winkler interval score for every symmetric (lo, hi) pair.

    Coverage and width are each only half an answer -- an infinitely wide
    interval covers everything, a zero-width one covers nothing -- and this is
    the proper scoring rule that combines them. For a nominal ``1 - alpha``
    interval it charges the width, plus a penalty of ``2/alpha`` times how far
    outside the interval the outcome fell:

        (hi - lo) + 2/alpha * (lo - y) if y < lo
                  + 2/alpha * (y - hi) if y > hi

    Lower is better, and it is minimized by the true quantiles, so it cannot be
    gamed by widening. Returns a list of dicts with keys ``lo``, ``hi``,
    ``nominal``, ``score``, matching `interval_coverage`'s shape.
    """
    y, Q, taus = _check(y, Q, taus)
    out = []
    for k, j, nominal in _symmetric_pairs(taus):
        alpha = 1.0 - nominal
        lo, hi = Q[:, k], Q[:, j]
        s = ((hi - lo)
             + (2.0 / alpha) * np.maximum(lo - y, 0.0)
             + (2.0 / alpha) * np.maximum(y - hi, 0.0))
        out.append({
            "lo": float(taus[k]),
            "hi": float(taus[j]),
            "nominal": float(nominal),
            "score": float(np.average(s, weights=sample_weight)),
        })
    return out


def sharpness(Q, taus, sample_weight=None):
    """Mean width of every symmetric interval, ignoring ``y`` entirely.

    Sharpness is the half of forecast quality that does not depend on the
    outcome: of two models that are equally calibrated, the sharper one is
    better. Read it against `interval_coverage` -- narrow is only a virtue at
    the nominal coverage.
    """
    Q = np.asarray(Q, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    return [{
        "lo": float(taus[k]),
        "hi": float(taus[j]),
        "nominal": float(nominal),
        "width": float(np.average(Q[:, j] - Q[:, k], weights=sample_weight)),
    } for k, j, nominal in _symmetric_pairs(taus)]


def pit_values(y, Q, taus):
    """Probability integral transform: for each row, roughly which quantile
    level the outcome actually landed on.

    Interpolates the predicted quantile function at ``y``, so a perfectly
    calibrated model returns values uniform on (0, 1). Outcomes past either
    end of the grid are clamped to ``0.0`` or ``1.0``, since a finite grid
    cannot say where in the tail they fell -- the mass at exactly 0 and 1 is
    itself the signal that the grid is too narrow.

    This is the diagnostic `interval_coverage` cannot give: coverage tells you
    a band is too narrow, PIT tells you *where* the model is wrong.
    """
    y, Q, taus = _check(y, Q, taus)
    Qs = np.sort(Q, axis=1)             # interpolation needs a monotone grid
    out = np.empty(y.shape[0])
    for i in range(y.shape[0]):
        out[i] = np.interp(y[i], Qs[i], taus, left=0.0, right=1.0)
    return out


def pit_histogram(y, Q, taus, bins=10):
    """`pit_values` binned on (0, 1) as relative frequencies, plus the bin
    edges: ``(freq, edges)``.

    Flat is calibrated. A U-shape means the intervals are too narrow (too many
    outcomes in the tails), a hump means too wide, and a tilt means the whole
    distribution is biased. `ChimeraBoostQuantileRegressor` runs slightly
    narrow before conformalization, so expect a mild U.
    """
    p = pit_values(y, Q, taus)
    freq, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    return freq / max(freq.sum(), 1), edges


def marginal_grid(y_train, taus, n_rows=1, sample_weight=None):
    """The no-skill forecast: the unconditional quantiles of ``y_train``,
    repeated for every row. This is what `quantile_skill_score` scores
    against -- a model that ignores its features entirely.
    """
    taus = np.asarray(taus, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    if sample_weight is None:
        q = np.quantile(y_train, taus)
    else:
        from .losses import _weighted_quantile
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        q = np.array([_weighted_quantile(y_train, w, t) for t in taus])
    return np.tile(q, (n_rows, 1))


def quantile_skill_score(y, Q, taus, baseline=None, sample_weight=None):
    """CRPS skill against a no-skill forecast: ``1 - crps(model)/crps(base)``.

    1.0 is perfect, 0.0 is no better than the baseline, negative is worse.
    ``baseline`` is either a ``(n_samples, n_quantiles)`` grid or a 1-D array
    of training targets to take unconditional quantiles from; the default uses
    ``y`` itself, which scores against the best possible constant forecast and
    is therefore the conservative reading.

    Same construction as the Brier skill and R2 the project's north-star chart
    uses -- zero at no-skill, one at perfect -- so a quantile model can be read
    on the same scale as a regressor or a classifier.
    """
    y, Q, taus = _check(y, Q, taus)
    if baseline is None:
        baseline = y
    B = np.asarray(baseline, dtype=np.float64)
    if B.ndim == 1:
        B = marginal_grid(B, taus, y.shape[0], sample_weight)
    if B.shape != Q.shape:
        raise ValueError(f"baseline grid has shape {B.shape}, but Q has "
                         f"{Q.shape}.")

    num = crps(y, Q, taus, sample_weight)
    den = crps(y, B, taus, sample_weight)
    if den <= 0.0:
        raise ValueError(
            "the baseline forecast is already perfect (CRPS 0), so skill "
            "against it is undefined -- every target is identical.")
    return 1.0 - num / den


def quantile_report(y, Q, taus, sample_weight=None, baseline=None):
    """Everything above in one dict: ``crps``, ``skill``, ``pinball`` (per
    level), ``taus``, ``intervals`` (coverage, width and interval score for
    each symmetric pair), ``pit`` and ``pit_edges``, and ``crossing_rate``.

    ``baseline`` is passed through to `quantile_skill_score`; the default
    scores against the unconditional quantiles of ``y`` itself.
    """
    y, Q, taus = _check(y, Q, taus)
    intervals = interval_coverage(y, Q, taus, sample_weight)
    for iv, sc in zip(intervals, interval_score(y, Q, taus, sample_weight)):
        iv["score"] = sc["score"]
    freq, edges = pit_histogram(y, Q, taus)
    return {
        "crps": crps(y, Q, taus, sample_weight),
        "skill": quantile_skill_score(y, Q, taus, baseline, sample_weight),
        "pinball": pinball_loss(y, Q, taus, sample_weight),
        "taus": taus,
        "intervals": intervals,
        "pit": freq,
        "pit_edges": edges,
        "crossing_rate": crossing_rate(Q),
    }


def format_report(report, title=None):
    """Render `quantile_report` as a fixed-width text table."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"CRPS (mean pinball over the grid): {report['crps']:.6f}")
    if "skill" in report:
        lines.append(f"CRPS skill vs the marginal grid:   "
                     f"{report['skill']:.4f}   (1 perfect, 0 no better)")
    lines.append(f"crossing rate: {report['crossing_rate']:.6f}")
    lines.append("")
    lines.append(f"{'tau':>8s}{'pinball':>12s}")
    for t, p in zip(report["taus"], report["pinball"]):
        lines.append(f"{t:8.2f}{p:12.6f}")
    if report["intervals"]:
        lines.append("")
        scored = "score" in report["intervals"][0]
        head = (f"{'interval':>10s}{'nominal':>10s}{'coverage':>10s}"
                f"{'width':>10s}")
        lines.append(head + f"{'score':>12s}" if scored else head)
        for iv in report["intervals"]:
            row = (f"{iv['lo']:.2f}-{iv['hi']:.2f}".rjust(10)
                   + f"{iv['nominal']:10.2f}{iv['coverage']:10.4f}"
                     f"{iv['width']:10.4f}")
            lines.append(row + f"{iv['score']:12.4f}" if scored else row)

    if "pit" in report:
        # A flat row is calibrated; a U means the bands are too narrow.
        lines.append("")
        lines.append("PIT histogram (flat = calibrated, U = too narrow)")
        lines.append("  " + " ".join(f"{v:5.3f}" for v in report["pit"]))
    return "\n".join(lines)
