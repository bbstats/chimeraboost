"""Ordered target statistics for categorical features.

This is CatBoost's key trick for categoricals. Instead of plain mean-target
encoding (which leaks the label of each row into its own feature and overfits),
we fix a random permutation of the rows and encode each row using only the rows
that come *before* it in that permutation:

    ctr_i = (sum_y_before(category_i) + prior * a) / (count_before(category_i) + a)

where `prior` is the global target mean and `a` is a smoothing weight. A row
never sees its own target, which removes the leakage / prediction shift that
makes naive target encoding so fragile.

At prediction time there is no "before", so we use the full training totals:

    ctr   = (sum_y_total(category) + prior * a) / (count_total(category) + a)

Unseen categories fall back to the prior.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _ordered_ts(codes, y, perm, n_cat, prior, a):
    """Single-permutation ordered target statistic.

    Returns the encoded column plus the full per-category totals (reused at
    predict time).
    """
    sums = np.zeros(n_cat)
    counts = np.zeros(n_cat)
    out = np.empty(codes.shape[0], dtype=np.float64)
    for pos in range(perm.shape[0]):
        i = perm[pos]
        c = codes[i]
        out[i] = (sums[c] + prior * a) / (counts[c] + a)
        sums[c] += y[i]
        counts[c] += 1.0
    return out, sums, counts


@njit(cache=True)
def _ordered_ts_weighted(codes, y, w, perm, n_cat, prior, a):
    """Sample-weighted ordered target statistic.

    Each row contributes its weight ``w[i]`` to the running per-category sum and
    count, so a zero-weight row is a ghost: it neither shifts the statistic used
    by later rows nor (via the returned totals) the predict-time encoding. With
    ``w`` all ones this is bit-identical to ``_ordered_ts``; the unweighted
    kernel is kept as the fast path for that case.
    """
    sums = np.zeros(n_cat)
    counts = np.zeros(n_cat)
    out = np.empty(codes.shape[0], dtype=np.float64)
    for pos in range(perm.shape[0]):
        i = perm[pos]
        c = codes[i]
        out[i] = (sums[c] + prior * a) / (counts[c] + a)
        sums[c] += w[i] * y[i]
        counts[c] += w[i]
    return out, sums, counts


class OrderedTargetEncoder:
    """Encodes one or more categorical columns into numeric ctr columns.

    Categorical inputs are expected as integer codes in [0, n_categories).
    Use `factorize` to turn arbitrary (string/object) columns into codes.

    n_permutations: number of random orderings to average during fit.
    Averaging reduces the variance of the encoded values the same way
    bagging reduces variance — each permutation is an independent noisy
    estimate of the leave-one-out target statistic. CatBoost uses 4 by
    default; more is strictly better but with diminishing returns past ~8.
    """

    def __init__(self, smoothing=1.0, random_state=None, n_permutations=4):
        self.smoothing = float(smoothing)
        self.random_state = random_state
        self.n_permutations = int(n_permutations)
        self.prior_ = None
        self.sums_ = None       # list per column
        self.counts_ = None     # list per column
        self.n_cat_ = None      # list per column

    def fit_transform(self, codes_matrix, y, sample_weight=None):
        """codes_matrix: (n_samples, n_cat_features) int array of codes.

        ``sample_weight`` (mean-1 normalized, ``None`` == uniform) weights each
        row's contribution to the prior and the per-category statistics, so a
        zero-weight row never shapes another row's encoding. ``None`` takes the
        unweighted fast path, bit-identical to before this argument existed."""
        codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        w = None if sample_weight is None else np.ascontiguousarray(
            sample_weight, dtype=np.float64)
        n_samples, n_cols = codes_matrix.shape
        rng = np.random.default_rng(self.random_state)

        self.prior_ = (float(np.mean(y)) if w is None
                       else float(np.average(y, weights=w)))
        self.sums_, self.counts_, self.n_cat_ = [], [], []
        out = np.zeros((n_samples, n_cols), dtype=np.float64)

        for j in range(n_cols):
            codes = np.ascontiguousarray(codes_matrix[:, j])
            n_cat = int(codes.max()) + 1 if codes.size else 1
            acc = np.zeros(n_samples, dtype=np.float64)
            sums = counts = None
            for _ in range(self.n_permutations):
                perm = rng.permutation(n_samples)
                if w is None:
                    enc, sums, counts = _ordered_ts(
                        codes, y, perm, n_cat, self.prior_, self.smoothing)
                else:
                    enc, sums, counts = _ordered_ts_weighted(
                        codes, y, w, perm, n_cat, self.prior_, self.smoothing)
                acc += enc
            out[:, j] = acc / self.n_permutations
            # sums/counts are full-data totals: identical across permutations.
            self.sums_.append(sums)
            self.counts_.append(counts)
            self.n_cat_.append(n_cat)
        return out

    def transform(self, codes_matrix):
        codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
        n_samples, n_cols = codes_matrix.shape
        out = np.empty((n_samples, n_cols), dtype=np.float64)
        a = self.smoothing
        for j in range(n_cols):
            codes = codes_matrix[:, j]
            sums, counts, n_cat = self.sums_[j], self.counts_[j], self.n_cat_[j]
            enc = np.full(n_samples, self.prior_, dtype=np.float64)
            valid = (codes >= 0) & (codes < n_cat)
            c = codes[valid]
            enc[valid] = (sums[c] + self.prior_ * a) / (counts[c] + a)
            out[:, j] = enc
        return out


def factorize(column):
    """Map an arbitrary 1D column to integer codes in [0, K), in first-appearance
    order. NaN / None map to a dedicated "__nan__" category. Returns
    (codes, categories).

    Codes are internal labels; the ordered target encoder is invariant to their
    particular values.
    """
    import pandas as pd
    col = np.asarray(column, dtype=object)
    na = pd.isna(col)
    if na.any():
        col = col.copy()
        col[na] = "__nan__"
    codes, categories = pd.factorize(col, sort=False)   # first-appearance order
    return codes.astype(np.int64), np.asarray(categories, dtype=object)
