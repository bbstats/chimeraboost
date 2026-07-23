"""Quantization of numeric features into integer bins.

Borders are learned once on the training data (quantile based). Every feature
is mapped to a small integer bin index, which is what the tree builder consumes.
NaNs are routed to a dedicated bin so a split can isolate missing values, the
way CatBoost/LightGBM do.

Bin layout per feature:
    real values -> 0 .. n_borders        (via searchsorted on borders)
    NaN         -> n_borders + 1          (the highest bin, "missing")
The histogram width for a feature is therefore (n_borders + 2).
"""

import numpy as np
from numba import njit, prange

BIN_DTYPE = np.uint16
# uint16 max is 65535; we reserve one slot for NaN, so the cap is 65534.
# In practice 128-256 bins is the useful range; this guard just catches typos.
_MAX_SUPPORTED_BINS = np.iinfo(BIN_DTYPE).max - 1


@njit(cache=True, parallel=True)
def _bin_matrix(X, borders_flat, offsets, out):
    """Map every (row, feature) of X to its integer bin, in parallel over rows.

    Equivalent to, per feature f with borders b = borders_flat[offsets[f]:offsets[f+1]]:
        finite v -> np.searchsorted(b, v, side="right")   (count of borders <= v)
        non-finite v (NaN / +-inf) -> len(b) + 1          (the NaN/missing bin)
    Parallelised over rows so each thread reads a contiguous X row and writes a
    contiguous `out` row (cache-friendly on the row-major matrices), replacing the
    per-column single-threaded np.searchsorted loop.
    """
    n, nf = X.shape
    for i in prange(n):
        for f in range(nf):
            lo = offsets[f]
            hi = offsets[f + 1]
            m = hi - lo                      # number of borders for feature f
            v = X[i, f]
            if not np.isfinite(v):
                out[i, f] = m + 1            # NaN / inf -> missing bin
            else:
                # rightmost insertion point == count of borders <= v.
                a = lo
                b = hi
                while a < b:
                    mid = (a + b) // 2
                    if borders_flat[mid] <= v:
                        a = mid + 1
                    else:
                        b = mid
                out[i, f] = a - lo


@njit(cache=True)
def _bin_matrix_serial(X, borders_flat, offsets, out):
    """Serial twin of `_bin_matrix` for tiny predict batches, where the
    OpenMP fork/join costs more than the whole pass (~20us vs ~1us for a
    1-row batch). Every write is independent, so the two are bit-identical;
    `Binner.transform` dispatches on `_SERIAL_PREDICT_N`."""
    n, nf = X.shape
    for i in range(n):
        for f in range(nf):
            lo = offsets[f]
            hi = offsets[f + 1]
            m = hi - lo
            v = X[i, f]
            if not np.isfinite(v):
                out[i, f] = m + 1
            else:
                a = lo
                b = hi
                while a < b:
                    mid = (a + b) // 2
                    if borders_flat[mid] <= v:
                        a = mid + 1
                    else:
                        b = mid
                out[i, f] = a - lo


# Predict batches at or below this many rows take the serial kernels: the
# measured fork/join cost (~20us on 12 threads) exceeds the whole serial pass
# there, and the parallel walk overtakes serial by n~5 on a mid-size forest.
# Both sides of the dispatch are bit-identical, so this only affects speed.
_SERIAL_PREDICT_N = 4


def _weighted_quantiles(values, weights, qs):
    """Weighted quantiles at levels ``qs`` (sorted values, midpoint plotting
    position). Reduces to the ordinary midpoint quantile when weights are
    equal; used only on the sample-weighted binning path."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w)
    total = cumw[-1]
    # Position of each value on [0, 1]: cumulative weight up to its midpoint.
    pos = (cumw - 0.5 * w) / total
    return np.interp(qs, pos, v)


def _feature_borders(col, max_bins, weights=None):
    """Quantile borders for one numeric column, ignoring NaNs.

    ``weights`` (per row, aligned with ``col``) makes the borders sample-weight
    aware: zero-weight rows are dropped outright and fractional weights steer the
    quantiles, so a row the caller zeroed out cannot place a bin edge. ``None``
    is the unweighted fast path, unchanged from before this argument existed."""
    finite_mask = np.isfinite(col)
    finite = col[finite_mask]
    if weights is None:
        if finite.size == 0:
            return np.array([], dtype=np.float64)
        uniq = np.unique(finite)
        if uniq.size <= max_bins:
            # Few distinct values: put a border between each pair.
            return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
        qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        borders = np.quantile(finite, qs)
        return np.unique(borders).astype(np.float64)
    # Weighted path: a zero-weight row does not exist for border purposes.
    fw = weights[finite_mask]
    pos = fw > 0.0
    finite, fw = finite[pos], fw[pos]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    uniq = np.unique(finite)
    if uniq.size <= max_bins:
        return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    borders = _weighted_quantiles(finite, fw, qs)
    return np.unique(borders).astype(np.float64)


class Binner:
    """Learns per-feature borders and maps a float matrix to bins."""

    def __init__(self, max_bins=128):
        # max_bins is a scalar (uniform budget) or a per-feature array (C4
        # cat-aware binning: a larger budget for target-encoded categorical
        # columns). Validate either form against the dtype cap and the >=2 floor.
        arr = np.atleast_1d(np.asarray(max_bins))
        if (arr > _MAX_SUPPORTED_BINS).any():
            raise ValueError(
                f"max_bins={max_bins} exceeds {_MAX_SUPPORTED_BINS} "
                f"(BIN_DTYPE={BIN_DTYPE.__name__}); use a smaller value."
            )
        if (arr.astype(np.int64) < 2).any():
            raise ValueError(f"max_bins={max_bins} must be >= 2.")
        # Keep a scalar scalar (back-compat) or an int per-feature array.
        self.max_bins = (int(max_bins) if np.isscalar(max_bins)
                         or arr.size == 1 and np.ndim(max_bins) == 0
                         else arr.astype(np.int64))
        self.borders_ = None       # list of np.ndarray, one per feature
        self.n_bins_ = None        # np.ndarray int, width per feature
        self.bin_centers_ = None   # list of np.ndarray: representative value/bin
        self._borders_flat = None  # contiguous borders for the numba kernel
        self._offsets = None       # int64 (n_features+1) prefix offsets into flat

    def _max_bins_for(self, f):
        """Per-feature bin budget: a scalar applies to all features, an array
        gives feature f its own budget."""
        return int(self.max_bins) if np.ndim(self.max_bins) == 0 \
            else int(self.max_bins[f])

    @staticmethod
    def _centers_for(borders):
        """A representative continuous value for each bin of one feature.

        Bin layout is bins 0..m (the searchsorted buckets for m borders) plus a
        trailing NaN bin. Interior bins use the midpoint of their border pair;
        the two edge bins extrapolate by half the adjacent gap; the NaN bin gets
        NaN (callers using these for a linear term map it to the feature mean).
        Used by the optional linear-leaf models to evaluate a within-leaf slope.
        """
        m = len(borders)
        centers = np.empty(m + 2, dtype=np.float64)
        if m == 0:
            centers[:] = 0.0
            centers[1] = np.nan
            return centers
        if m == 1:
            centers[0] = borders[0]
            centers[1] = borders[0]
        else:
            centers[0] = borders[0] - 0.5 * (borders[1] - borders[0])
            centers[1:m] = 0.5 * (borders[:-1] + borders[1:])
            centers[m] = borders[m - 1] + 0.5 * (borders[m - 1] - borders[m - 2])
        centers[m + 1] = np.nan                     # NaN bin
        return centers

    def fit(self, X, sample_weight=None):
        """Learn quantile borders for each column from training data.

        ``sample_weight`` (per row, ``None`` == uniform) makes the borders
        weight-aware; ``None`` is bit-identical to the pre-weight behavior."""
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        w = None if sample_weight is None else np.asarray(
            sample_weight, dtype=np.float64)
        self.borders_ = [
            _feature_borders(X[:, f], self._max_bins_for(f), w)
            for f in range(n_features)
        ]
        # +1 for the searchsorted upper bucket, +1 for the NaN bucket.
        self.n_bins_ = np.array(
            [len(b) + 2 for b in self.borders_], dtype=np.int64
        )
        self.bin_centers_ = [self._centers_for(b) for b in self.borders_]
        self._build_flat_borders()
        return self

    def _build_flat_borders(self):
        """Flatten the ragged per-feature borders into one contiguous array plus
        offsets, so the numba kernel can index feature f's borders as
        borders_flat[offsets[f]:offsets[f+1]]. Cached on the instance."""
        lens = [len(b) for b in self.borders_]
        self._offsets = np.zeros(len(self.borders_) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(lens)
        self._borders_flat = (np.concatenate(self.borders_).astype(np.float64)
                              if self.borders_ else np.zeros(0, dtype=np.float64))

    def transform(self, X):
        """Map a float matrix to integer bin indices; NaNs go to the top bin."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        # Lazily (re)build the flat border layout if missing — e.g. when borders_
        # were set without going through fit().
        if getattr(self, "_borders_flat", None) is None or \
                len(self._offsets) != n_features + 1:
            self._build_flat_borders()
        out = np.empty((n_samples, n_features), dtype=BIN_DTYPE)
        if n_samples:
            kernel = (_bin_matrix_serial if n_samples <= _SERIAL_PREDICT_N
                      else _bin_matrix)
            kernel(X, self._borders_flat, self._offsets, out)
        return out

    def fit_transform(self, X, sample_weight=None):
        return self.fit(X, sample_weight).transform(X)
