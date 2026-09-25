"""Compact storage of the rows a fitted booster's leaves came from.

Plain numeric columns are kept as bins of the fitted binner (feature-major),
cross parents as raw float64, categoricals as int32 codes plus per-column
categories in code order, y as float64 and the raw weights. ``rebuild_X``
turns the store back into a raw-equivalent array that the existing replay
refit consumes unchanged -- there is deliberately no second preprocessing
path here.
"""

import numpy as np

from .binning import BIN_DTYPE
from .preprocessing import as_model_array
from .target_encoding import factorize

# What factorize() turns a missing categorical (None, NaN, or refusing
# self-comparison) into; rebuild_X maps this category back to np.nan.
_MISSING = "__nan__"


def _split_plain_parent(prep):
    """Split ``prep.num_features_`` into plain (binned) and parent (raw).

    Every numeric column appearing in any of ``prep.cross_pairs`` stays raw,
    whatever the op: diff/prod crosses are computed from raw floats and gdiff
    group means are refit on the replay rows. Any future preprocessing input
    that reads raw numeric values beyond binning (as crosses do) must be
    added to the raw set here, or refresh loses exactness.
    """
    num = list(prep.num_features_)
    inset = set(num)
    parents = set()
    for i, j, _op in prep.cross_pairs:
        if i in inset:
            parents.add(i)
        if j in inset:
            parents.add(j)
    return [f for f in num if f not in parents], [f for f in num if f in parents]


def _values_for_bins(bins, borders):
    """A value per bin that bins back identically (binning.py:82-93)."""
    m = len(borders)
    out = np.empty(len(bins), dtype=np.float64)
    if m == 0:
        out[:] = 0.0
        out[bins == 1] = np.nan
        return out
    miss = bins == m + 1
    zero = bins == 0
    out[zero] = np.nextafter(borders[0], -np.inf)
    mid = ~(miss | zero)
    out[mid] = borders[bins[mid].astype(np.int64) - 1]
    out[miss] = np.nan
    return out


def _extend_codes(stored_vals, new_col, stored_codes):
    """Codes for [stored; new] with unseen categories appended in order."""
    ext = list(stored_vals)
    idx = {v: i for i, v in enumerate(ext)}
    ncodes, ncats = factorize(new_col)
    loc = np.empty(len(ncats), dtype=np.int64)
    for li, nc in enumerate(ncats):
        if nc in idx:
            loc[li] = idx[nc]
        else:
            idx[nc] = len(ext)
            ext.append(nc)
            loc[li] = len(ext) - 1
    new_codes = loc[ncodes].astype(np.int32)
    full = np.concatenate([stored_codes, new_codes])
    return full, np.asarray(ext, dtype=object)


class TrainingRows:
    """The rows a fitted booster's leaves came from, in its row order."""

    def __init__(self, n_features, cat_features, num_features, plain_features,
                 parent_features, plain_bins, parent_raw, cat_codes,
                 cat_values, y, sample_weight):
        self.n_features = int(n_features)
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.plain_features = list(plain_features)
        self.parent_features = list(parent_features)
        self.plain_bins = plain_bins
        self.parent_raw = parent_raw
        self.cat_codes = cat_codes
        self.cat_values = cat_values
        self.y = y
        self.sample_weight = sample_weight

    @property
    def n_rows(self):
        return int(self.y.shape[0])

    @classmethod
    def capture(cls, prep, X, y, sample_weight=None):
        """Capture raw rows with a FITTED ``FeaturePreprocessor``."""
        X = as_model_array(X, bool(prep.cat_features_))
        y = np.asarray(y, dtype=np.float64)
        w = (None if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float64))
        plain, parents = _split_plain_parent(prep)
        pos = {f: k for k, f in enumerate(prep.num_features_)}
        cols = [pos[f] for f in plain]
        nb = prep.binner_.n_bins_
        dt = (np.uint8 if all(nb[k] <= 256 for k in cols) else BIN_DTYPE)
        if cols:
            bins = np.ascontiguousarray(prep.transform(X)[:, cols].T, dtype=dt)
        else:
            bins = np.empty((0, X.shape[0]), dtype=dt)
        pcols = [pos[f] for f in parents]
        num = prep._numeric_block(X)
        if pcols:
            raw = np.ascontiguousarray(num[:, pcols], dtype=np.float64)
        else:
            raw = np.empty((X.shape[0], 0), dtype=np.float64)
        codes = prep._codes_for_transform(X).astype(np.int32)
        vals = []
        for m in prep.cat_maps_:
            arr = np.empty(len(m), dtype=object)
            for v, c in m.items():
                arr[c] = v
            vals.append(arr)
        return cls(X.shape[1], prep.cat_features_, prep.num_features_,
                   plain, parents, bins, raw, codes, vals, y, w)

    def append(self, prep, X_new, y_new, sample_weight_new=None):
        """A NEW ``TrainingRows`` for [stored; new]; neither mutates."""
        X_new = as_model_array(X_new, bool(self.cat_features))
        y_new = np.asarray(y_new, dtype=np.float64)
        w_new = (None if sample_weight_new is None
                 else np.asarray(sample_weight_new, dtype=np.float64))
        if X_new.shape[1] != self.n_features:
            raise ValueError("X_new has %d columns, stored %d"
                             % (X_new.shape[1], self.n_features))
        if (self.sample_weight is None) != (w_new is None):
            raise ValueError("weighted/unweighted append mismatch")
        pos = {f: k for k, f in enumerate(prep.num_features_)}
        cols = [pos[f] for f in self.plain_features]
        if cols:
            nb = np.ascontiguousarray(
                prep.transform(X_new)[:, cols].T,
                dtype=self.plain_bins.dtype)
            bins = np.concatenate([self.plain_bins, nb], axis=1)
        else:
            bins = np.empty((0, self.n_rows + len(y_new)),
                            dtype=self.plain_bins.dtype)
        pcols = [pos[f] for f in self.parent_features]
        num = prep._numeric_block(X_new)
        praw = np.concatenate(
            [self.parent_raw,
             np.ascontiguousarray(num[:, pcols], dtype=np.float64)
             if pcols else np.empty((len(y_new), 0))])
        codes, vals = [], []
        for j, f in enumerate(self.cat_features):
            c, v = _extend_codes(self.cat_values[j], X_new[:, f],
                                 self.cat_codes[:, j])
            codes.append(c)
            vals.append(v)
        stacked = (np.column_stack(codes).astype(np.int32) if codes
                   else np.empty((self.n_rows + len(y_new), 0),
                                 dtype=np.int32))
        y = np.concatenate([self.y, y_new])
        w = (None if w_new is None
             else np.concatenate([self.sample_weight, w_new]))
        return TrainingRows(self.n_features, self.cat_features,
                            self.num_features, self.plain_features,
                            self.parent_features, bins, praw, stacked,
                            vals, y, w)

    def rebuild_X(self, prep):
        """The raw-equivalent array in the original column layout."""
        n = self.n_rows
        X = (np.empty((n, self.n_features), dtype=object)
             if self.cat_features
             else np.empty((n, self.n_features), dtype=np.float64))
        pos = {f: k for k, f in enumerate(prep.num_features_)}
        for p, f in enumerate(self.plain_features):
            X[:, f] = _values_for_bins(
                self.plain_bins[p], prep.binner_.borders_[pos[f]])
        for j, f in enumerate(self.parent_features):
            X[:, f] = self.parent_raw[:, j]
        for j, f in enumerate(self.cat_features):
            col = self.cat_values[j][self.cat_codes[:, j].astype(np.int64)]
            col[col == _MISSING] = np.nan
            X[:, f] = col
        return X
