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

    cat_combinations : bool
        When True, generate all C(n_cat, 2) pairwise categorical feature
        combinations as additional synthetic columns (e.g. "buying_x_maint")
        before target encoding. Mirrors CatBoost's feature combination step;
        gives the tree access to interaction effects that individual categoricals
        can't capture. Only active when ≥2 categorical columns are present.
    onehot_low_card : bool
        When True, every categorical column with at most ``onehot_max_card``
        distinct levels also gets a one-hot indicator block (one 0/1 column per
        level), stacked alongside the ordered-TS encoding. The indicators let the
        tree make EXACT subset splits on individual rare categories -- which a
        single monotone TS column, where rare levels are shrunk toward the prior,
        cannot. CatBoost-proven; cheap for low cardinality. High-cardinality
        columns keep TS only (one-hot would explode the column count).
    onehot_max_card : int
        Cardinality ceiling (inclusive) for one-hot encoding a column.
    """

    def __init__(self, max_bins=128, cat_smoothing=1.0, random_state=None,
                 cat_n_permutations=4, cat_combinations=False,
                 onehot_low_card=False, onehot_max_card=8):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state
        self.cat_n_permutations = int(cat_n_permutations)
        self.cat_combinations = bool(cat_combinations)
        self.onehot_low_card = bool(onehot_low_card)
        self.onehot_max_card = int(onehot_max_card)

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _combo_values(X, f_a, f_b):
        """The synthetic "val_a_x_val_b" string column for a feature pair."""
        col_a = np.asarray(X[:, f_a], dtype=str)
        col_b = np.asarray(X[:, f_b], dtype=str)
        return np.char.add(np.char.add(col_a, "_x_"), col_b)

    def _split_columns_fit(self, X, cat_features):
        """Split input into a numeric matrix and an integer-code matrix for the
        categorical columns, learning the category->code maps on the way.
        When cat_combinations is True, appends combo codes after the base codes."""
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

        # 2-way combinations: each pair becomes a new categorical column of
        # "val_a_x_val_b" strings, target-encoded like any other cat column, so
        # the tree sees interaction effects single columns can't express.
        self.combo_pairs_ = []
        self.combo_maps_ = []
        if self.cat_combinations and len(self.cat_features_) >= 2:
            combo_cols = []
            for a in range(len(self.cat_features_)):
                for b in range(a + 1, len(self.cat_features_)):
                    f_a, f_b = self.cat_features_[a], self.cat_features_[b]
                    c, cats = factorize(self._combo_values(X, f_a, f_b))
                    self.combo_pairs_.append((f_a, f_b))
                    self.combo_maps_.append({v: i for i, v in enumerate(cats)})
                    combo_cols.append(c)
            codes = np.hstack([codes,
                               np.column_stack(combo_cols).astype(np.int64)])
        return num, codes

    def _compute_onehot_specs(self):
        """Pick the base categorical columns to one-hot: those with at least 2
        and at most ``onehot_max_card`` distinct levels. Stores
        ``onehot_specs_`` as a list of (col-index-into-cat_features_, n_levels).
        A no-op (empty list) unless ``onehot_low_card`` is set."""
        self.onehot_specs_ = []
        if not self.onehot_low_card:
            return
        for j in range(len(self.cat_features_)):
            n_levels = len(self.cat_maps_[j])
            if 2 <= n_levels <= self.onehot_max_card:
                self.onehot_specs_.append((j, n_levels))

    def _onehot_block(self, base_codes):
        """Build the (n, sum n_levels) one-hot 0/1 matrix from the base-cat code
        matrix. Column for level ``l`` of cat ``j`` is ``base_codes[:, j] == l``;
        an unseen category (code -1 from transform) matches no level, so its row
        is all-zeros -- the right behavior (the TS column still carries the prior
        fallback for unseen values)."""
        if not self.onehot_specs_:
            return np.empty((base_codes.shape[0], 0))
        cols = []
        for j, n_levels in self.onehot_specs_:
            cj = base_codes[:, j]
            for l in range(n_levels):
                cols.append((cj == l).astype(np.float64))
        return np.column_stack(cols)

    def _codes_for_transform(self, X):
        """Map categorical columns to the codes learned at fit time; unseen
        categories get -1 (the encoder then falls back to the prior).
        Vectorized via pandas."""
        if not self.cat_features_:
            return np.empty((X.shape[0], 0), dtype=np.int64)
        import pandas as pd
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        for j, f in enumerate(self.cat_features_):
            s = pd.Series(X[:, f], dtype=object)
            s = s.where(~pd.isna(s), "__nan__")
            mapped = s.map(self.cat_maps_[j])          # unseen -> NaN
            codes[:, j] = mapped.fillna(-1).astype(np.int64).to_numpy()
        return codes

    def _combo_codes_for_transform(self, X):
        """Reconstruct combination codes for transform using stored combo maps."""
        combo_codes = np.full((X.shape[0], len(self.combo_pairs_)), -1, dtype=np.int64)
        for k, (f_a, f_b) in enumerate(self.combo_pairs_):
            m = self.combo_maps_[k]
            vals = self._combo_values(X, f_a, f_b)
            combo_codes[:, k] = [m.get(v, -1) for v in vals.tolist()]
        return combo_codes

    # ---- fit / transform -----------------------------------------------------
    def fit_transform(self, X, encode_targets, cat_features):
        """encode_targets: list of 1D arrays used for ordered TS (len T)."""
        num, codes = self._split_columns_fit(X, cat_features)

        # One-hot indicator block for low-cardinality cats (built from the base
        # cat codes, before any combo columns). Stacked between num and the TS
        # blocks; non-numeric (binary, no ordinal meaning for linear leaves).
        self._compute_onehot_specs()
        onehot = self._onehot_block(codes[:, :len(self.cat_features_)])

        encoded_blocks = []
        self.encoders_ = []
        if codes.shape[1]:
            for t, target in enumerate(encode_targets):
                enc = OrderedTargetEncoder(
                    self.cat_smoothing,
                    None if self.random_state is None else self.random_state + t,
                    self.cat_n_permutations,
                )
                encoded_blocks.append(enc.fit_transform(codes, target))
                self.encoders_.append(enc)

        feat = self._stack(num, [onehot] + encoded_blocks)
        self._build_feature_map(len(encode_targets))
        # Block order is [numeric | one-hot | per-target TS]. Only true numeric
        # columns carry an ordinal meaning usable by linear-leaf models; mark them
        # so the booster can pick linear-term features. One-hot indicators are 0/1
        # (the tree still splits on them exactly) and the TS blocks are excluded.
        self.is_numeric_binned_ = np.zeros(feat.shape[1], dtype=bool)
        self.is_numeric_binned_[:num.shape[1]] = True

        self.binner_ = Binner(self.max_bins)
        X_binned = self.binner_.fit_transform(feat)
        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X):
        """Apply the fitted binning + categorical encoding to new data."""
        num = (np.asarray(X[:, self.num_features_], dtype=np.float64)
               if self.num_features_ else np.empty((X.shape[0], 0)))
        onehot = np.empty((X.shape[0], 0))
        encoded_blocks = []
        if self.cat_features_:
            codes = self._codes_for_transform(X)
            onehot = self._onehot_block(codes)   # from base codes, pre-combo
            if self.combo_pairs_:
                combo_codes = self._combo_codes_for_transform(X)
                codes = np.hstack([codes, combo_codes])
            for enc in self.encoders_:
                encoded_blocks.append(enc.transform(codes))
        feat = self._stack(num, [onehot] + encoded_blocks)
        return self.binner_.transform(feat)

    # ---- internals -----------------------------------------------------------
    @staticmethod
    def _stack(num, encoded_blocks):
        mats = [m for m in ([num] + encoded_blocks) if m.shape[1]]
        if not mats:
            return num
        return np.hstack(mats) if len(mats) > 1 else mats[0]

    def _build_feature_map(self, n_targets):
        """Map each combined-matrix column back to its original input column.
        Block order is [numeric | one-hot | per-target (cat + combo)]. Each
        one-hot indicator maps to its source categorical column (repeated per
        level), and combo columns map to the lower-indexed feature of their pair,
        so their split gains fold into the right importance bucket."""
        combo_orig = [min(i, j) for i, j in self.combo_pairs_]
        onehot_orig = [self.cat_features_[j]
                       for j, n_levels in self.onehot_specs_
                       for _ in range(n_levels)]
        fmap = list(self.num_features_) + onehot_orig
        for _ in range(n_targets):
            fmap.extend(self.cat_features_)
            fmap.extend(combo_orig)
        self.feature_map_ = np.array(fmap, dtype=np.int64)

        max_idx = max(self.num_features_, default=-1)
        if self.cat_features_:
            max_idx = max(max_idx, max(self.cat_features_))
        self.n_input_features_ = max_idx + 1
