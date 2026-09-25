"""Random intercepts for grouped data (issue #109, slice 1), plus random
slopes on one numeric column (issue #113, slice 2).

Slice 1 fits ``y = F(X) + b_g + eps`` with ``b ~ N(0, s2b)``: trees learn the
global part while each group carries one shrunk intercept. Two routines:

- ``solve_intercepts``: the closed-form empirical-Bayes intercepts given the
  variance ratio. Random intercepts are block-diagonal (one block per group),
  so this is one groupby, never a matrix inverse.
- ``estimate_ratio_reml``: the noise-to-group variance ratio by 1-D REML over
  the same sufficient statistics. Golden-section on the log ratio, fixed
  iterations, no scipy (the library is numpy + numba + sklearn only).

Slice 2 fits ``y = F(X) + b_g + a_g * z + eps`` with independent
``b ~ N(0, s2b)`` and ``a ~ N(0, s2a)``, where ``z = x - centre`` on one
user-named numeric column (a NaN x contributes ``z = 0``). Two more
routines over five weighted per-group bincounts:

- ``solve_slopes``: each group's posterior mean from its 2x2 ridge system.
- ``estimate_slope_ratios_reml``: both noise-to-group variance ratios by
  profiled REML with fixed effects [1, z], via cyclic golden-section on the
  log ratios with slice 1's bounds and constants (fixed iterations,
  deterministic, no scipy -- it matches slice 1).

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


# --- random slopes, slice 2 (issue #113) ------------------------------------

# Full intercept/slope alternations in estimate_slope_ratios_reml. Each 1-D
# leg runs slice 1's 100-iteration golden-section; the third cycle moves each
# ratio under 0.01 decades on gate-shaped probes, so it is confirmation, not
# search. Fixed count: deterministic.
_SLOPE_REML_CYCLES = 3


def _slope_suff_stats(resid, codes, n_groups, z, weights):
    """Five weighted per-group bincounts plus the pooled totals.

    ``z`` is ``x - centre`` with NaN (a missing x) mapped to 0, so a
    missing slope value contributes nothing anywhere. Returns
    ``(S_w, S_z, S_zz, S_r, S_zr, q_ss, w_sum)``.
    """
    w = (np.ones_like(resid) if weights is None
         else np.asarray(weights, dtype=np.float64))
    zc = np.where(np.isnan(z), 0.0, z)
    wz = w * zc
    wr = resid * w
    s_w = np.bincount(codes, weights=w, minlength=n_groups)
    s_z = np.bincount(codes, weights=wz, minlength=n_groups)
    s_zz = np.bincount(codes, weights=wz * zc, minlength=n_groups)
    s_r = np.bincount(codes, weights=wr, minlength=n_groups)
    s_zr = np.bincount(codes, weights=wr * zc, minlength=n_groups)
    return (s_w, s_z, s_zz, s_r, s_zr, float((resid * wr).sum()),
            float(w.sum()))


def solve_slopes(resid, codes, n_groups, ratio_b, ratio_a, z, weights=None):
    """Empirical-Bayes intercepts and slopes for ``resid`` given both ratios.

    ``ratio_b``/``ratio_a`` are noise over group variance (``s2/s2b`` and
    ``s2/s2a`` under independent priors); group j's ``(b, a)`` solves its
    2x2 ridge system ``[[S_w + ratio_b, S_z], [S_z, S_zz + ratio_a]]``.
    ``z`` is ``x - centre`` (NaN maps to 0: a missing x carries no slope
    information). ``ratio_a=inf`` is exactly intercept-only -- ``a = 0``
    and ``b`` bit-identical to ``solve_intercepts``; empty groups are
    exactly (0, 0). Returns the ``(b, a)`` pair, each ``(n_groups,)``.
    """
    resid = np.asarray(resid, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    z = np.asarray(z, dtype=np.float64)
    if codes.shape != resid.shape or z.shape != resid.shape:
        raise ValueError("resid, codes and z must share a shape; got "
                         f"{resid.shape}, {codes.shape} and {z.shape}.")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1; got {n_groups}.")
    if codes.size and (codes.min() < 0 or codes.max() >= n_groups):
        raise ValueError("codes out of range for "
                         f"n_groups={n_groups}.")
    if not (ratio_b >= 0.0):
        raise ValueError(f"ratio_b must be >= 0; got {ratio_b}.")
    if not (ratio_a >= 0.0):
        raise ValueError(f"ratio_a must be >= 0; got {ratio_a}.")

    if np.isinf(ratio_a):
        # No slopes: delegate the intercepts so they match solve_intercepts
        # bit for bit (the 1e-12 reduction the plan requires, with margin).
        return (solve_intercepts(resid, codes, n_groups, ratio_b, weights),
                np.zeros(n_groups, dtype=np.float64))

    s_w, s_z, s_zz, s_r, s_zr, _, _ = _slope_suff_stats(
        resid, codes, n_groups, z, weights)
    if np.isinf(ratio_b):
        b = np.zeros(n_groups, dtype=np.float64)
        a = np.zeros(n_groups, dtype=np.float64)
        denom = s_zz + ratio_a
        nz = denom > 0.0
        a[nz] = s_zr[nz] / denom[nz]
        return b, a

    m11 = s_w + ratio_b
    m22 = s_zz + ratio_a
    det = m11 * m22 - s_z * s_z
    b = np.zeros(n_groups, dtype=np.float64)
    a = np.zeros(n_groups, dtype=np.float64)
    ok = det > 0.0
    b[ok] = (m22[ok] * s_r[ok] - s_z[ok] * s_zr[ok]) / det[ok]
    a[ok] = (m11[ok] * s_zr[ok] - s_z[ok] * s_r[ok]) / det[ok]
    # Singular at zero penalty (a slope no observation identifies, e.g. a
    # singleton with ratio 0): fall back to intercept-only, a = 0.
    bad = ~ok
    if bad.any():
        denom = s_w + ratio_b
        nz = bad & (denom > 0.0)
        b[nz] = s_r[nz] / denom[nz]
    return b, a


def _slope_reml_from_stats(s_w, s_z, s_zz, s_r, s_zr, q_ss, w_sum,
                           log_ratio_b, log_ratio_a):
    """-2 REML (up to an additive constant) at the log10 ratios.

    Fixed effects [1, z] (a global intercept and slope), the noise variance
    profiled out, V block-diagonal over groups. Per group, with
    ``M = [[S_w + lb, S_z], [S_z, S_zz + la]]`` and ``T`` the same matrix
    unpenalized, Woodbury gives ``Sigma_g^-1 = D - D U M^-1 U' D`` and the
    matrix determinant lemma ``log|Sigma_g| = log|M| - log lb - log la``
    (up to the weights constant):

        Q    = q_ss - sum v' M^-1 v          (residual quadratic form)
        XtX  = sum (T - T M^-1 T)            (fixed-effects information)
        d    = sum (v - T M^-1 v)            (fixed-effects score)
        Qc   = Q - d' XtX^-1 d               (contrasts quadratic form)

    -2 REML = (W-2) log Qc + sum log|M| - G log lb - G log la + log|XtX|,
    the p = 2 form of slice 1's criterion.
    """
    lb = 10.0 ** log_ratio_b
    la = 10.0 ** log_ratio_a
    m11 = s_w + lb
    m22 = s_zz + la
    detm = m11 * m22 - s_z * s_z
    if np.any(detm <= 0.0) or not np.all(np.isfinite(detm)):
        return np.inf
    i11 = m22 / detm
    i12 = -s_z / detm
    i22 = m11 / detm
    # Residual quadratic form: v' M^-1 v per group.
    mv1 = i11 * s_r + i12 * s_zr
    mv2 = i12 * s_r + i22 * s_zr
    q_full = q_ss - float((mv1 * s_r + mv2 * s_zr).sum())
    # T M^-1 T and T M^-1 v per group, accumulated into XtX and d.
    t11 = s_w
    t12 = s_z
    t22 = s_zz
    tm11 = t11 * i11 + t12 * i12
    tm12 = t11 * i12 + t12 * i22
    tm21 = t12 * i11 + t22 * i12
    tm22 = t12 * i12 + t22 * i22
    x11 = float((t11 - (tm11 * t11 + tm12 * t12)).sum())
    x12 = float((t12 - (tm11 * t12 + tm12 * t22)).sum())
    x22 = float((t22 - (tm21 * t12 + tm22 * t22)).sum())
    d1 = float((s_r - (tm11 * s_r + tm12 * s_zr)).sum())
    d2 = float((s_zr - (tm21 * s_r + tm22 * s_zr)).sum())
    detx = x11 * x22 - x12 * x12
    if not detx > 0.0:
        return np.inf
    qc = q_full - (x22 * d1 * d1 - 2.0 * x12 * d1 * d2
                   + x11 * d2 * d2) / detx
    if not qc > 0.0:
        return np.inf
    det = float(np.log(detm).sum() - s_w.size * (np.log(lb) + np.log(la)))
    return (w_sum - 2.0) * np.log(qc) + det + np.log(detx)


def _golden_minimize(f, lo=_LOG_RATIO_LO, hi=_LOG_RATIO_HI):
    """Golden-section minimum of ``f`` on [lo, hi] (slice 1's constants)."""
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
    return (lo + hi) / 2.0


def estimate_slope_ratios_reml(resid, codes, n_groups, z, weights=None):
    """REML estimates of the (intercept, slope) noise-to-group ratios.

    Profiled REML with fixed effects [1, z] over the slope sufficient
    statistics; both ratios searched on the log10 scale in slice 1's
    bounds by cyclic golden-section (``_SLOPE_REML_CYCLES`` full
    intercept/slope alternations from slice 1's intercept ratio with no
    slopes). Returns ``(ratio_b, ratio_a)``. Degenerate data returns inf
    for the ratio with no support, as slice 1 does: one group or all
    singletons give ``(inf, inf)``; x constant within every group (no
    within-group slope information) gives slice 1's intercept ratio with
    ``ratio_a = inf``.
    """
    resid = np.asarray(resid, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    z = np.asarray(z, dtype=np.float64)
    if codes.shape != resid.shape or z.shape != resid.shape:
        raise ValueError("resid, codes and z must share a shape; got "
                         f"{resid.shape}, {codes.shape} and {z.shape}.")
    if codes.size and (codes.min() < 0 or codes.max() >= n_groups):
        # np.bincount would silently extend past n_groups instead.
        raise ValueError("codes out of range for "
                         f"n_groups={n_groups}.")

    stats = _slope_suff_stats(resid, codes, n_groups, z, weights)
    s_w, s_z, s_zz, _, _, _, w_sum = stats
    nonempty = s_w > 0.0
    n_nonempty = int(nonempty.sum())
    if n_nonempty < 2 or w_sum - float(n_nonempty) <= 0.0:
        return (np.inf, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        within_z = s_zz - s_z * s_z / s_w
    if not float(within_z[nonempty].sum()) > 0.0:
        # x constant within every group: slopes have nothing to fit.
        return (estimate_ratio_reml(resid, codes, n_groups, weights),
                np.inf)

    def f(tb, ta):
        return _slope_reml_from_stats(*stats, tb, ta)

    start = estimate_ratio_reml(resid, codes, n_groups, weights)
    tb = float(np.log10(start)) if np.isfinite(start) else _LOG_RATIO_HI
    tb = min(max(tb, _LOG_RATIO_LO), _LOG_RATIO_HI)
    ta = _LOG_RATIO_HI  # start with no slopes
    for _ in range(_SLOPE_REML_CYCLES):
        tb = _golden_minimize(lambda t: f(t, ta))
        ta = _golden_minimize(lambda t: f(tb, t))
    return (float(10.0 ** tb), float(10.0 ** ta))
