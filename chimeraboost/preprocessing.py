"""Shared feature preprocessing for every ChimeraBoost estimator.

Turns a raw (possibly mixed numeric/categorical, possibly object-dtype) matrix
into integer bins ready for the tree builder, and remembers everything needed to
reproduce the same transform at predict time.

Categoricals are encoded with ordered target statistics. The encoder is fit
against a *list* of target vectors:
  * regression / binary -> one target (y, or the 0/1 label)
  * multiclass          -> K one-hot targets (one ordered-TS column per class)
This is why a single categorical column can expand into K numeric columns for
multiclass, exactly like CatBoost's per-class target statistics.

`feature_map_` maps each combined-matrix column back to its original input
column index, so importances can be aggregated in the user's feature space.
"""

import numpy as np

from .binning import Binner
from .target_encoding import OrderedTargetEncoder, factorize


class FeaturePreprocessor:
    """Converts raw mixed-type input into integer bins for the tree builder.

    Numeric columns are quantile-binned; categorical columns are ordered-target
    encoded (one encoded column per target supplied to `fit_transform`) and then
    binned alongside the numerics. The fitted state needed to reproduce the
    transform at predict time is retained, along with `feature_map_` mapping each
    output column back to its original input column for importances.
    """

    def __init__(self, max_bins=128, cat_smoothing=1.0, random_state=None):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state

    # ---- helpers -------------------------------------------------------------
    def _split_columns_fit(self, X, cat_features):
        """Split input into a numeric matrix and an integer-code matrix for the
        categorical columns, learning the category->code maps on the way."""
        n_features = X.shape[1]
        cat_set = set(cat_features or [])
        self.cat_features_ = sorted(cat_set)
        self.num_features_ = [f for f in range(n_features) if f not in cat_set]

        num = (np.asarray(X[:, self.num_features_], dtype=np.float64)
               if self.num_features_ else np.empty((X.shape[0], 0)))

        if self.cat_features_:
            codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
            self.cat_maps_ = []
            for j, f in enumerate(self.cat_features_):
                c, cats = factorize(X[:, f])
                codes[:, j] = c
                self.cat_maps_.append({v: i for i, v in enumerate(cats)})
        else:
            codes = np.empty((X.shape[0], 0), dtype=np.int64)
            self.cat_maps_ = []
        return num, codes

    def _codes_for_transform(self, X):
        """Map categorical columns to the codes learned at fit time; unseen
        categories get -1 (the encoder then falls back to the prior)."""
        if not self.cat_features_:
            return np.empty((X.shape[0], 0), dtype=np.int64)
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        for j, f in enumerate(self.cat_features_):
            m = self.cat_maps_[j]
            col = X[:, f]
            for i in range(X.shape[0]):
                v = col[i]
                if v is None or (isinstance(v, float) and v != v):
                    v = "__nan__"
                codes[i, j] = m.get(v, -1)   # unseen -> prior fallback
        return codes

    # ---- fit / transform -----------------------------------------------------
    def fit_transform(self, X, encode_targets, cat_features):
        """encode_targets: list of 1D arrays used for ordered TS (len T)."""
        num, codes = self._split_columns_fit(X, cat_features)

        encoded_blocks = []
        self.encoders_ = []
        if codes.shape[1]:
            for t, target in enumerate(encode_targets):
                enc = OrderedTargetEncoder(
                    self.cat_smoothing,
                    None if self.random_state is None else self.random_state + t,
                )
                encoded_blocks.append(enc.fit_transform(codes, target))
                self.encoders_.append(enc)

        feat = self._stack(num, encoded_blocks)
        self._build_feature_map(num.shape[1], codes.shape[1], len(encode_targets))

        self.binner_ = Binner(self.max_bins)
        X_binned = self.binner_.fit_transform(feat)
        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X):
        """Apply the fitted binning + categorical encoding to new data."""
        num = (np.asarray(X[:, self.num_features_], dtype=np.float64)
               if self.num_features_ else np.empty((X.shape[0], 0)))
        encoded_blocks = []
        if self.cat_features_:
            codes = self._codes_for_transform(X)
            for enc in self.encoders_:
                encoded_blocks.append(enc.transform(codes))
        feat = self._stack(num, encoded_blocks)
        return self.binner_.transform(feat)

    # ---- internals -----------------------------------------------------------
    @staticmethod
    def _stack(num, encoded_blocks):
        mats = [m for m in ([num] + encoded_blocks) if m.shape[1]]
        if not mats:
            return num
        return np.hstack(mats) if len(mats) > 1 else mats[0]

    def _build_feature_map(self, n_num, n_cat, n_targets):
        """Combined column index -> original input column index."""
        fmap = list(self.num_features_)            # numeric block
        for _ in range(n_targets):                 # each TS target adds a block
            fmap.extend(self.cat_features_)        # one col per cat feature
        self.feature_map_ = np.array(fmap, dtype=np.int64)
        self.n_input_features_ = (
            (max(self.num_features_) if self.num_features_ else -1)
        )
        if self.cat_features_:
            self.n_input_features_ = max(self.n_input_features_,
                                         max(self.cat_features_))
        self.n_input_features_ += 1
