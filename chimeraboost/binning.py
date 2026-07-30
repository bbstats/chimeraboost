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

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import get_num_threads, njit, prange

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
# fork/join cost exceeds the whole serial pass there.
#
# Re-measured 2026-07-30 by forcing each kernel on the same packed forest.
# The old value of 4 assumed the parallel walk overtakes serial by n~5; it
# actually overtakes between 32 and 64, so every 5-to-32-row predict was
# paying fork/join for nothing -- up to 1.26x on a 100-tree forest:
#
#   rows   serial   parallel        (100 trees, depth 6, 25 features)
#      4   38.4us     48.9us   serial 1.27x
#      8   39.5us     49.6us   serial 1.26x
#     16   41.7us     49.1us   serial 1.18x
#     32   47.7us     50.1us   serial 1.05x
#     64   56.1us     51.5us   parallel 1.09x
#
# 300- and 500-tree forests cross over in the same place. 32 is the last row
# count serial still wins; the penalty for being wrong at the boundary on a
# wider machine is the 1.05x there, against the 1.26x recovered below it.
# Both sides of the dispatch are bit-identical, so this only affects speed.
_SERIAL_PREDICT_N = 32


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


@njit(cache=True)
def _greedy_border_fill(uniq, mass, heavy, target, budget):
    """Compiled twin of the border-building sweep in `_greedy_borders`.

    A straight transcription of the former Python loop: same statement order,
    same float64 arithmetic, so the borders are bit-identical. Only the sweep
    lives here -- the mass sums and heavy-value marking stay in numpy above,
    because numba's reductions do not match numpy's pairwise summation
    bit-for-bit and `target` must not move."""
    n = uniq.size
    borders = np.empty(budget, dtype=np.float64)
    nb = 0
    acc = 0.0
    for i in range(n):
        if nb >= budget:
            break
        if heavy[i]:
            if acc > 0.0:
                # Close the light run before the heavy value so it gets a
                # bin of its own.
                borders[nb] = (uniq[i - 1] + uniq[i]) / 2.0
                nb += 1
                acc = 0.0
            if i < n - 1 and nb < budget:
                borders[nb] = (uniq[i] + uniq[i + 1]) / 2.0
                nb += 1
        else:
            acc += mass[i]
            if acc >= target and i < n - 1:
                borders[nb] = (uniq[i] + uniq[i + 1]) / 2.0
                nb += 1
                acc = 0.0
    return borders[:nb].copy()


def _greedy_borders(uniq, mass, max_bins):
    """Borders over distinct values when even-mass quantiles have collapsed.

    A value holding at least an even bin's share of mass gets isolated between
    the midpoints to its neighbors, and the budget it does not consume is
    re-spread over the remaining values by mass. Pure even-mass splitting
    cannot do this: every quantile level lands on the heavy value, the borders
    dedup to one edge equal to that value, and (with the "border <= v goes
    right" convention) that edge separates nothing -- the feature silently
    dies. Borders always sit strictly between distinct values, like the
    few-uniques branch."""
    total = float(mass.sum())
    # Mark heavy values iteratively: removing one from the pool lowers the
    # even share of what remains, which can expose the next. Capped so the
    # border budget below cannot overflow even on pathological mass profiles.
    heavy = np.zeros(uniq.size, dtype=np.bool_)
    max_heavy = (max_bins - 1) // 2
    while int(heavy.sum()) < max_heavy:
        rest = total - float(mass[heavy].sum())
        thr = rest / (max_bins - int(heavy.sum()))
        new = (~heavy) & (mass >= thr)
        if not new.any():
            break
        if int(heavy.sum()) + int(new.sum()) > max_heavy:
            cand = np.where(new)[0]
            room = max_heavy - int(heavy.sum())
            heavy[cand[np.argsort(mass[cand])[::-1][:room]]] = True
            break
        heavy |= new
    n_heavy = int(heavy.sum())
    # The light mass keeps its MASS-PROPORTIONAL share of the bin budget --
    # the allocation plain quantile borders produce, and the one the library
    # defaults are tuned around -- with a floor so a dominated column can
    # never collapse to nothing. Redistributing the heavy values' whole freed
    # budget into the light region instead measurably over-resolves sparse
    # tails (decision-tier A/B 2026-07-30: Grinsztajn 21W-32L, median -0.03%).
    light_total = total - float(mass[heavy].sum())
    # Floor of 1: below max_bins=16 the `max_bins // 16` floor is itself 0, and
    # a column dominated hard enough to round the proportional term to 0 too
    # divided by zero here. For max_bins >= 16 the // 16 term already dominates,
    # so every budget that worked before keeps its exact allocation.
    light_bins = max(1, max_bins // 16,
                     int(round((max_bins - n_heavy) * (light_total / total))))
    target = light_total / light_bins
    # The O(n_distinct) sweep is compiled: in pure Python it made Binner.fit
    # ~4x slower on zero-inflated columns (1.34 s vs 0.29 s dense at 200k
    # rows x 30 features).
    return _greedy_border_fill(
        np.ascontiguousarray(uniq, dtype=np.float64),
        np.ascontiguousarray(mass, dtype=np.float64),
        heavy, float(target), int(max_bins - 1))


def _feature_borders(col, max_bins, weights=None):
    """Quantile borders for one numeric column, ignoring NaNs.

    ``weights`` (per row, aligned with ``col``) makes the borders sample-weight
    aware: zero-weight rows are dropped outright and fractional weights steer the
    quantiles, so a row the caller zeroed out cannot place a bin edge. ``None``
    is the unweighted fast path, unchanged from before this argument existed.

    Columns whose quantile levels all land on distinct borders keep the plain
    even-mass quantile borders, bit-identical to before. Colliding levels mean
    a value holds more than an even bin's share of mass; those columns switch
    to `_greedy_borders`, which isolates such values instead of letting the
    collapsed edge kill the column."""
    finite_mask = np.isfinite(col)
    finite = col[finite_mask]
    if weights is None:
        if finite.size == 0:
            return np.array([], dtype=np.float64)
        # Open-coded np.unique (one sort + a run-start mask) so the greedy
        # path below can read its per-value mass off the run lengths for
        # free. np.unique's counts would cost every dense column a little,
        # and the old searchsorted + np.add.at pass cost the greedy columns
        # more than the greedy sweep itself. Counts are exact integers any
        # way they are computed, so borders are bit-identical.
        sv = np.sort(finite)
        run_start = np.empty(sv.size, dtype=np.bool_)
        run_start[0] = True
        np.not_equal(sv[1:], sv[:-1], out=run_start[1:])
        uniq = sv[run_start]
        if uniq.size <= max_bins:
            # Few distinct values: put a border between each pair.
            return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
        qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        # Quantiles of the sorted copy, partitioned in place: identical values
        # (quantile is order-agnostic), but skips np.quantile's internal copy
        # and partitions near-instantly. Only sv.size is read below this line,
        # so scrambling sv is fine.
        borders = np.unique(np.quantile(sv, qs, overwrite_input=True))
        if borders.size == qs.size:
            return borders.astype(np.float64)
        starts = np.nonzero(run_start)[0]
        counts = np.empty(starts.size, dtype=np.float64)
        counts[:-1] = np.diff(starts)
        counts[-1] = sv.size - starts[-1]
        return _greedy_borders(uniq, counts, max_bins)
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
    borders = np.unique(_weighted_quantiles(finite, fw, qs))
    if borders.size == qs.size:
        return borders.astype(np.float64)
    wmass = np.zeros(uniq.size)
    np.add.at(wmass, np.searchsorted(uniq, finite), fw)
    return _greedy_borders(uniq, wmass, max_bins)


# Matrices with fewer cells than this fit their borders serially: below the
# crossover the thread-pool spawn/join costs more than the whole pass.
#
# Measured 2026-07-30 (12 threads, 128 bins, dense unweighted, best of 5):
#
#      shape      cells    serial   parallel
#    5000x10      50000    1.79ms     3.09ms   serial 1.7x
#    5000x30     150000    5.31ms     7.92ms   serial 1.5x
#   20000x10     200000    3.64ms     3.39ms   parallel 1.08x
#   50000x10     500000    7.43ms     3.84ms   parallel 1.9x
#  100000x30    3000000   40.24ms    12.23ms   parallel 3.3x
#  200000x30    6000000   96.0 ms    22.3 ms   parallel 4.3x
#
# Zero-inflated columns gain 3.2x at 200k x 30 and the weighted path 5.5x
# (its argsort/cumsum/interp all release the GIL). 200k cells is the first
# size parallel wins; the penalty for being wrong near the boundary is a few
# ms either way. Both dispatch arms run identical per-column code, so the
# constant only affects speed, never borders.
_PARALLEL_FIT_MIN_CELLS = 200_000


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
        # Every column's borders are independent (the invariant
        # from_base_with_cross splices on), so wide-enough fits farm the
        # columns out to a thread pool running the *same* per-column code --
        # numpy's sort/partition/argsort release the GIL, and nothing here
        # accumulates across features, so the borders are bit-identical to
        # the serial loop regardless of scheduling. Pool width follows
        # numba's thread count: fit already runs inside _thread_limit, so
        # thread_count=1 and bagged-worker budget shares are honored.
        n_threads = min(get_num_threads(), n_features)
        if n_threads > 1 and X.size >= _PARALLEL_FIT_MIN_CELLS:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                self.borders_ = list(ex.map(
                    lambda f: _feature_borders(X[:, f], self._max_bins_for(f), w),
                    range(n_features)))
        else:
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
