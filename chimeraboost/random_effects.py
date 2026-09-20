"""Random intercepts for grouped data (issue #109, slice 1).

The model is ``y = F(X) + b_g + eps`` with ``b ~ N(0, s2b)``: trees learn the
global part while each group carries one shrunk intercept. Two routines:

- ``solve_intercepts``: the closed-form empirical-Bayes intercepts given the
  variance ratio. Random intercepts are block-diagonal (one block per group),
  so this is one groupby, never a matrix inverse.
- ``estimate_ratio_reml``: the noise-to-group variance ratio by 1-D REML over
  the same sufficient statistics. Golden-section on the log ratio, fixed
  iterations, no scipy (the library is numpy + numba + sklearn only).

``codes_for_labels`` maps predict-time group labels back to the fit codes,
with unseen labels (and anything that fails to match) falling back to -1,
whose intercept is exactly 0.
"""

import numpy as np

# The ratio search lives on the log10 scale: 1e-8 is no-pooling for any
# realistic group size, 1e8 is complete pooling. Fixed iterations keep the
# search deterministic (no data-dependent stopping).
_LOG_RATIO_LO = -8.0
_LOG_RATIO_HI = 8.0
_GOLDEN_ITERS = 100
_GOLDEN_GR = (np.sqrt(5.0) - 1.0) / 2.0


def _suff_stats(resid, codes, n_groups, weights):
    """Per-group sums/counts plus the pooled totals, all weighted."""
    w = (np.ones_like(resid) if weights is None
         else np.asarray(weights, dtype=np.float64))
    wr = resid * w
    sums = np.bincount(codes, weights=wr, minlength=n_groups)
    counts = np.bincount(codes, weights=w, minlength=n_groups)
    return sums, counts, float(wr.sum()), float((resid * wr).sum()), float(
        w.sum())


def solve_intercepts(resid, codes, n_groups, ratio, weights=None):
    """Empirical-Bayes intercepts for ``resid`` given the variance ratio.

    ``ratio`` is noise variance over group variance (lambda = s2e / s2b);
    group j's intercept is ``S_j / (W_j + ratio)``: a huge group tends to
    its own mean, a one-row group shrinks to ~0. ``ratio=inf`` (no group
    signal) returns exact zeros; empty groups are exactly 0 at any ratio.
    """
    resid = np.asarray(resid, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != resid.shape:
        raise ValueError("resid and codes must share a shape; got "
                         f"{resid.shape} and {codes.shape}.")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1; got {n_groups}.")
    if codes.size and (codes.min() < 0 or codes.max() >= n_groups):
        raise ValueError("codes out of range for "
                         f"n_groups={n_groups}.")
    if not (ratio >= 0.0):
        raise ValueError(f"ratio must be >= 0; got {ratio}.")

    sums, counts, _, _, _ = _suff_stats(resid, codes, n_groups, weights)
    denom = counts + ratio
    out = np.zeros(n_groups, dtype=np.float64)
    nz = denom > 0.0
    out[nz] = sums[nz] / denom[nz]
    return out


def _reml_from_stats(sums, counts, total, q_ss, w_sum, log_ratio):
    """-2 REML (up to an additive constant) at ``log10(ratio)``.

    Intercept-only model, profiled over the grand mean and the noise
    variance. With ``a_j = 1 / (W_j + lambda)``:

        Q    = q_ss - S' diag(a) S            (residual quadratic form)
        c    = W - W' diag(a) W               (intercept information)
        d    = T - W' diag(a) S               (intercept score)
        Qc   = Q - d^2 / c                    (contrasts quadratic form)

    -2 REML = (W-1) log Qc + sum log(1 + W_j/lambda) + log c.
    """
    lam = 10.0 ** log_ratio
    a = 1.0 / (counts + lam)
    q_full = q_ss - float((sums * sums * a).sum())
    c = w_sum - float((counts * counts * a).sum())
    d = total - float((counts * sums * a).sum())
    if c <= 0.0:
        return np.inf
    qc = q_full - d * d / c
    if qc <= 0.0:
        return np.inf
    det = float(np.log1p(counts / lam).sum())
    return (w_sum - 1.0) * np.log(qc) + det + np.log(c)


def estimate_ratio_reml(resid, codes, n_groups, weights=None):
    """REML estimate of the noise-to-group variance ratio.

    Returns ``inf`` (fit no intercepts) when the data cannot separate the
    two variances: fewer than two non-empty groups, no within-group
    degrees of freedom, or zero within-group variation.
    """
    resid = np.asarray(resid, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != resid.shape:
        raise ValueError("resid and codes must share a shape; got "
                         f"{resid.shape} and {codes.shape}.")
    if codes.size and (codes.min() < 0 or codes.max() >= n_groups):
        # np.bincount would silently extend past n_groups instead.
        raise ValueError("codes out of range for "
                         f"n_groups={n_groups}.")

    sums, counts, total, q_ss, w_sum = _suff_stats(resid, codes, n_groups,
                                                  weights)
    nonempty = counts > 0.0
    n_nonempty = int(nonempty.sum())
    if n_nonempty < 2 or w_sum - float(n_nonempty) <= 0.0:
        return np.inf
    within_ss = q_ss - float((sums[nonempty] ** 2
                              / counts[nonempty]).sum())
    if within_ss <= 0.0:
        # Every group is constant: no-pooling is the answer, and the
        # criterion's minimum sits at the boundary.
        return 10.0 ** _LOG_RATIO_LO

    stats = (sums, counts, total, q_ss, w_sum)

    def f(t):
        return _reml_from_stats(*stats, t)

    lo, hi = _LOG_RATIO_LO, _LOG_RATIO_HI
    c = hi - _GOLDEN_GR * (hi - lo)
    d = lo + _GOLDEN_GR * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(_GOLDEN_ITERS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN_GR * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN_GR * (hi - lo)
            fd = f(d)
    return float(10.0 ** ((lo + hi) / 2.0))


def _normalize_label(v):
    """The ``factorize`` missing rule: None, NaN-like, or refusing
    self-comparison all become the "__nan__" category."""
    if v is None:
        return "__nan__"
    try:
        if v != v:
            return "__nan__"
    except (TypeError, ValueError):
        return "__nan__"
    return v


def codes_for_labels(labels, categories):
    """Map group labels to the fit's codes; unseen labels get -1.

    ``categories`` is the fit-time ``factorize`` output. Missing values
    follow the same "__nan__" rule, so a NaN group at predict matches the
    NaN group at fit. Anything unhashable cannot have been a fit category
    either, so it also maps to -1 rather than raising.
    """
    pos = {}
    for i, c in enumerate(np.asarray(categories, dtype=object).tolist()):
        if c not in pos:
            try:
                pos[c] = i
            except TypeError:
                continue
    out = np.empty(len(labels), dtype=np.int64)
    for i, v in enumerate(np.asarray(labels, dtype=object).tolist()):
        v = _normalize_label(v)
        try:
            out[i] = pos.get(v, -1)
        except TypeError:
            out[i] = -1
    return out
