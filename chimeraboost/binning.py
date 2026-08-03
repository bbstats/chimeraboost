"""Quantization of numeric features into integer bins.

Borders are learned once on the training data, from quantiles. Every feature
becomes a small integer bin index, which is what the tree builder consumes.
NaNs get a bin of their own so a split can isolate missing values, the way
CatBoost and LightGBM do.

Bin layout per feature:

    real values -> 0 .. n_borders        (searchsorted on the borders)
    NaN         -> n_borders + 1         (the highest bin, "missing")

So a feature's histogram is (n_borders + 2) wide.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import get_num_threads, njit, prange

BIN_DTYPE = np.uint16

# uint16 tops out at 65535 and one slot is reserved for NaN, so the cap is
# 65534. The useful range is 128-256 bins; this guard only catches typos.
_MAX_SUPPORTED_BINS = np.iinfo(BIN_DTYPE).max - 1


@njit(cache=True, parallel=True)
def _bin_matrix(X, borders_flat, offsets, out):
    """Map every (row, feature) of X to its integer bin, in parallel over rows.

    For feature f with borders b = borders_flat[offsets[f]:offsets[f+1]], this is:

        finite v     -> np.searchsorted(b, v, side="right")  (borders <= v)
        NaN or +-inf -> len(b) + 1                           (the missing bin)

    Splitting the work by row, not by column, lets each thread read a contiguous
    X row and write a contiguous `out` row, which is cache-friendly on these
    row-major matrices. It replaces a single-threaded per-column searchsorted.
    """
    n, nf = X.shape

    for i in prange(n):
        for f in range(nf):
            lo = offsets[f]
            hi = offsets[f + 1]
            m = hi - lo                      # borders for feature f
            v = X[i, f]

            if not np.isfinite(v):
                out[i, f] = m + 1            # NaN / inf -> missing bin
            else:
                # Binary search for the rightmost insertion point.
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
    """Serial twin of `_bin_matrix`, for tiny predict batches.

    On a one-row batch the OpenMP fork/join costs about 20us against about 1us
    for the whole pass. Every write is independent, so the two kernels are
    bit-identical; `Binner.transform` picks one on `_SERIAL_PREDICT_N`.
    """
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


# Predict batches this small or smaller take the serial kernels, where
# fork/join costs more than the whole serial pass.
#
# Re-measured 2026-07-30 by forcing each kernel on the same packed forest. The
# old value of 4 assumed parallel overtook serial around n=5; it actually
# overtakes between 32 and 64, so every 5-to-32-row predict paid fork/join for
# nothing -- up to 1.26x on a 100-tree forest:
#
#   rows   serial   parallel        (100 trees, depth 6, 25 features)
#      4   38.4us     48.9us   serial 1.27x
#      8   39.5us     49.6us   serial 1.26x
#     16   41.7us     49.1us   serial 1.18x
#     32   47.7us     50.1us   serial 1.05x
#     64   56.1us     51.5us   parallel 1.09x
#
# 300- and 500-tree forests cross over in the same place. 32 is the last row
# count serial still wins. A wider machine could move the crossover, but being
# wrong at the boundary costs the 1.05x there, against the 1.26x recovered
# below it. Both kernels are bit-identical, so this only affects speed.
_SERIAL_PREDICT_N = 32


def _weighted_quantiles(values, weights, qs):
    """Weighted quantiles at levels ``qs``, using the midpoint plotting position.

    With equal weights this is the ordinary midpoint quantile. Only the
    sample-weighted binning path uses it.
    """
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

    A straight transcription of the old Python loop: same statement order, same
    float64 arithmetic, so the borders come out bit-identical.

    Only the sweep lives here. The mass sums and the heavy-value marking stay in
    numpy in the caller, because numba's reductions do not match numpy's
    pairwise summation bit-for-bit and `target` must not move.
    """
    n = uniq.size
    borders = np.empty(budget, dtype=np.float64)
    nb = 0
    acc = 0.0

    for i in range(n):
        if nb >= budget:
            break

        if heavy[i]:
            if acc > 0.0:
                # Close the light run first, so the heavy value gets its own bin.
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

    A value holding at least an even bin's share of mass ("heavy") is isolated
    between the midpoints to its neighbors, and the budget it does not use is
    re-spread over the remaining values by mass. As in the few-uniques branch,
    borders always sit strictly between distinct values.

    Even-mass splitting cannot do this. Every quantile level lands on the heavy
    value, the borders dedup to a single edge equal to that value, and under the
    "border <= v goes right" convention that edge separates nothing -- the
    feature silently dies.

    The light mass keeps its mass-proportional share of the bin budget. That is
    the allocation plain quantile borders produce, and the one the library
    defaults are tuned around. Handing the light region the heavy values' whole
    freed budget instead measurably over-resolves sparse tails (decision-tier
    A/B 2026-07-30: Grinsztajn 21W-32L, median -0.03%).
    """
    total = float(mass.sum())

    # Mark heavy values one pass at a time: removing one from the pool lowers
    # the even share of what remains, which can expose the next. Capped so the
    # border budget below cannot overflow on a pathological mass profile.
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
    light_total = total - float(mass[heavy].sum())

    # The floor of 1 keeps a dominated column from collapsing to nothing. Below
    # max_bins=16 the `max_bins // 16` term is itself 0, and a column dominated
    # hard enough to round the proportional term to 0 too divided by zero here.
    # From max_bins=16 up, the // 16 term already dominates, so every budget
    # that worked before keeps its exact allocation.
    light_bins = max(1, max_bins // 16,
                     int(round((max_bins - n_heavy) * (light_total / total))))
    target = light_total / light_bins

    # The O(n_distinct) sweep is compiled: in pure Python it made Binner.fit ~4x
    # slower on zero-inflated columns (1.34 s against 0.29 s dense, at 200k rows
    # by 30 features).
    return _greedy_border_fill(
        np.ascontiguousarray(uniq, dtype=np.float64),
        np.ascontiguousarray(mass, dtype=np.float64),
        heavy, float(target), int(max_bins - 1))


def _feature_borders(col, max_bins, weights=None):
    """Quantile borders for one numeric column, ignoring NaNs.

    ``weights`` (per row, aligned with ``col``) makes the borders sample-weight
    aware: zero-weight rows are dropped outright and fractional weights steer
    the quantiles, so a row the caller zeroed out cannot place a bin edge.
    ``None`` is the unweighted fast path, unchanged from before this argument
    existed.

    When every quantile level lands on a distinct border, the column keeps the
    plain even-mass quantile borders, bit-identical to before. Colliding levels
    mean some value holds more than an even bin's share of the mass, and the
    column falls through to `_greedy_borders`, which isolates that value rather
    than let the collapsed edge kill the feature.
    """
    finite_mask = np.isfinite(col)
    finite = col[finite_mask]

    if weights is None:
        if finite.size == 0:
            return np.array([], dtype=np.float64)

        # np.unique open-coded as one sort plus a run-start mask, so the greedy
        # path below gets its per-value mass off the run lengths for free.
        # np.unique's own counts would tax every dense column, and the old
        # searchsorted + np.add.at pass cost the greedy columns more than the
        # greedy sweep itself. Counts are exact integers however they are
        # computed, so borders stay bit-identical.
        sv = np.sort(finite)
        run_start = np.empty(sv.size, dtype=np.bool_)
        run_start[0] = True
        np.not_equal(sv[1:], sv[:-1], out=run_start[1:])
        uniq = sv[run_start]

        if uniq.size <= max_bins:
            # Few distinct values: put a border between each pair.
            return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)

        qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]

        # Quantiles are order-agnostic, so partitioning the sorted copy in place
        # gives identical values while skipping np.quantile's internal copy.
        # Nothing below reads sv except its size, so scrambling it is safe.
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
# At 200k x 30, zero-inflated columns gain 3.2x and the weighted path 5.5x (its
# argsort, cumsum and interp all release the GIL). 200k cells is the first size
# parallel wins, and being wrong near the boundary costs a few ms either way.
# Both arms run identical per-column code, so this only affects speed, never
# the borders themselves.
_PARALLEL_FIT_MIN_CELLS = 200_000


class Binner:
    """Learns per-feature borders and maps a float matrix to bins."""

    def __init__(self, max_bins=128):
        # max_bins is either a scalar (one budget for every feature) or a
        # per-feature array (C4 cat-aware binning gives target-encoded
        # categorical columns a bigger budget). Both forms get checked against
        # the dtype cap and the floor of 2.
        arr = np.atleast_1d(np.asarray(max_bins))

        if (arr > _MAX_SUPPORTED_BINS).any():
            raise ValueError(
                f"max_bins={max_bins} exceeds {_MAX_SUPPORTED_BINS} "
                f"(BIN_DTYPE={BIN_DTYPE.__name__}); use a smaller value."
            )

        if (arr.astype(np.int64) < 2).any():
            raise ValueError(f"max_bins={max_bins} must be >= 2.")

        # A scalar stays a scalar, for back-compat; anything else becomes an
        # int array with one entry per feature.
        self.max_bins = (int(max_bins) if np.isscalar(max_bins)
                         or arr.size == 1 and np.ndim(max_bins) == 0
                         else arr.astype(np.int64))

        self.borders_ = None       # list of np.ndarray, one per feature
        self.n_bins_ = None        # np.ndarray int, width per feature
        self.bin_centers_ = None   # list of np.ndarray: representative value/bin
        self._borders_flat = None  # contiguous borders for the numba kernel
        self._offsets = None       # int64 (n_features+1) prefix offsets into flat

    def _max_bins_for(self, f):
        """Bin budget for feature f: the scalar, or its own entry of the array."""
        return int(self.max_bins) if np.ndim(self.max_bins) == 0 \
            else int(self.max_bins[f])

    @staticmethod
    def _centers_for(borders):
        """A representative continuous value for each bin of one feature.

        The layout is bins 0..m (the searchsorted buckets for m borders) plus a
        trailing NaN bin. An interior bin takes the midpoint of its border pair,
        the two edge bins extrapolate by half the adjacent gap, and the NaN bin
        gets NaN -- callers building a linear term map that to the feature mean.

        The optional linear-leaf models use these to fit a within-leaf slope.
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

        ``sample_weight`` is per row, ``None`` meaning uniform. Passing ``None``
        is bit-identical to the behavior from before weights existed.

        A wide enough fit farms the columns out to a thread pool. That is safe
        because every column's borders are independent -- the same invariant
        `from_base_with_cross` splices on -- and both arms run the same
        per-column code. Numpy's sort, partition and argsort release the GIL,
        and nothing accumulates across features, so the borders match the serial
        loop bit-for-bit whatever order the threads finish in. The pool is sized
        from numba's thread count, since fit already runs inside `_thread_limit`
        and that keeps thread_count=1 and bagged-worker budget shares honored.
        """
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        w = None if sample_weight is None else np.asarray(
            sample_weight, dtype=np.float64)

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
        """Flatten the ragged per-feature borders into one contiguous array.

        With the accompanying offsets, the numba kernel reads feature f's
        borders as borders_flat[offsets[f]:offsets[f+1]]. Cached on the instance.
        """
        lens = [len(b) for b in self.borders_]

        self._offsets = np.zeros(len(self.borders_) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(lens)

        self._borders_flat = (np.concatenate(self.borders_).astype(np.float64)
                              if self.borders_ else np.zeros(0, dtype=np.float64))

    def transform(self, X):
        """Map a float matrix to integer bin indices; NaNs go to the top bin."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Rebuild the flat layout if it is missing or stale, which happens when
        # borders_ was set directly instead of through fit().
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
