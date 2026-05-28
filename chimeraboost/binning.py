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

BIN_DTYPE = np.uint16
# uint16 max is 65535; we reserve one slot for NaN, so the cap is 65534.
# In practice 128-256 bins is the useful range; this guard just catches typos.
_MAX_SUPPORTED_BINS = np.iinfo(BIN_DTYPE).max - 1


def _feature_borders(col, max_bins):
    """Quantile borders for one numeric column, ignoring NaNs."""
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    uniq = np.unique(finite)
    if uniq.size <= max_bins:
        # Few distinct values: put a border between each pair.
        return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    borders = np.quantile(finite, qs)
    return np.unique(borders).astype(np.float64)


class Binner:
    """Learns per-feature borders and maps a float matrix to bins."""

    def __init__(self, max_bins=128):
        if int(max_bins) > _MAX_SUPPORTED_BINS:
            raise ValueError(
                f"max_bins={max_bins} exceeds {_MAX_SUPPORTED_BINS} "
                f"(BIN_DTYPE={BIN_DTYPE.__name__}); use a smaller value."
            )
        if int(max_bins) < 2:
            raise ValueError(f"max_bins={max_bins} must be >= 2.")
        self.max_bins = int(max_bins)
        self.borders_ = None       # list of np.ndarray, one per feature
        self.n_bins_ = None        # np.ndarray int, width per feature

    def fit(self, X):
        """Learn quantile borders for each column from training data."""
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        self.borders_ = [
            _feature_borders(X[:, f], self.max_bins) for f in range(n_features)
        ]
        # +1 for the searchsorted upper bucket, +1 for the NaN bucket.
        self.n_bins_ = np.array(
            [len(b) + 2 for b in self.borders_], dtype=np.int64
        )
        return self

    def transform(self, X):
        """Map a float matrix to integer bin indices; NaNs go to the top bin."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        out = np.empty((n_samples, n_features), dtype=BIN_DTYPE)
        for f in range(n_features):
            col = X[:, f]
            borders = self.borders_[f]
            nan_bin = len(borders) + 1
            binned = np.searchsorted(borders, col, side="right").astype(BIN_DTYPE)
            binned[~np.isfinite(col)] = nan_bin
            out[:, f] = binned
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
