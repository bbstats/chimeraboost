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


def as_model_array(X, want_object):
    """Convert a raw feature matrix to the numpy array the model consumes.

    ``want_object`` -> object dtype (categoricals present, decoded downstream);
    otherwise -> float64. pandas nullable dtypes (Int64/Float64/boolean and the
    ``string`` dtype) store missing values as ``pd.NA``/``NAType``, which neither
    casts to float nor compares like ``np.nan`` -- a plain
    ``np.asarray(df, dtype=float)`` raises a cryptic
    "float() argument must be ... not 'NAType'". Routing through pandas'
    ``to_numpy(na_value=np.nan)`` maps every flavor of NA to ``np.nan`` (the
    missing value the binner/encoder already understand). Inputs without a
    na_value-aware ``to_numpy`` (plain ndarrays, polars frames) fall back to
    ``np.asarray`` unchanged.
    """
    dtype = object if want_object else np.float64
    to_numpy = getattr(X, "to_numpy", None)
    if to_numpy is not None and hasattr(X, "dtypes"):  # pandas DataFrame
        try:
            return to_numpy(dtype=dtype, na_value=np.nan)
        except TypeError:
            pass  # older pandas / polars: no na_value kwarg -> plain cast below
    return np.asarray(X, dtype=dtype)


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
    cross_pairs : list[(int, int, str)] | None
        Numeric cross features: each (i, j, op) appends the column
        ``X[:, i] - X[:, j]`` (op="diff") or ``X[:, i] * X[:, j]`` (op="prod"),
        binned like any numeric column. Oblivious trees can only approximate a
        numeric interaction with a depth-limited staircase (the same split is
        applied to every leaf of a level); a cross column turns e.g. the
        ``x_i < x_j`` boundary into a single split. Indices refer to ORIGINAL
        input columns and must be numeric (not in ``cat_features``).
    """

    def __init__(self, max_bins=128, cat_smoothing=1.0, random_state=None,
                 cat_n_permutations=4, cat_combinations=False,
                 cross_pairs=None):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state
        self.cat_n_permutations = int(cat_n_permutations)
        self.cat_combinations = bool(cat_combinations)
        self.cross_pairs = list(cross_pairs) if cross_pairs else []

    # ---- helpers -------------------------------------------------------------
    def _numeric_block(self, X):
        """The numeric columns as float64. When every column is numeric (the
        no-categoricals case) `num_features_` is exactly range(n_features), so
        plain asarray suffices — the fancy-index gather `X[:, list]` would copy
        the whole matrix (a large predict-time tax on wide batches)."""
        if not self.num_features_:
            return np.empty((X.shape[0], 0))
        if len(self.num_features_) == X.shape[1]:
            return np.asarray(X, dtype=np.float64)
        return np.asarray(X[:, self.num_features_], dtype=np.float64)

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

        num = self._numeric_block(X)

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
        n_cat = len(self.cat_features_)
        if self.cat_combinations and n_cat >= 2:
            combo_cols = []
            for a in range(n_cat):
                for b in range(a + 1, n_cat):
                    f_a, f_b = self.cat_features_[a], self.cat_features_[b]
                    c, cats = factorize(self._combo_values(X, f_a, f_b))
                    self.combo_pairs_.append((f_a, f_b))
                    self.combo_maps_.append({v: i for i, v in enumerate(cats)})
                    combo_cols.append(c)
            if combo_cols:
                codes = np.hstack(
                    [codes, np.column_stack(combo_cols).astype(np.int64)])
        return num, codes

    def _cross_block(self, X):
        """Compute the numeric cross-feature columns (float64) from raw input.
        NaN in either parent propagates to the cross (binned to the missing
        bucket like any numeric NaN)."""
        if not self.cross_pairs:
            return np.empty((X.shape[0], 0))
        cols = []
        for i, j, op in self.cross_pairs:
            a = np.asarray(X[:, i], dtype=np.float64)
            b = np.asarray(X[:, j], dtype=np.float64)
            cols.append(a - b if op == "diff" else a * b)
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
        cross = self._cross_block(X)
        if cross.shape[1]:
            num = np.hstack([num, cross]) if num.shape[1] else cross

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

        feat = self._stack(num, encoded_blocks)
        self._build_feature_map(len(encode_targets))
        # Block order is [numeric | per-target TS]. Only true numeric columns
        # carry an ordinal meaning usable by linear-leaf models; mark them so the
        # booster can pick linear-term features (the TS blocks are excluded).
        self.is_numeric_binned_ = np.zeros(feat.shape[1], dtype=bool)
        self.is_numeric_binned_[:num.shape[1]] = True

        self.binner_ = Binner(self.max_bins)
        X_binned = self.binner_.fit_transform(feat)
        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X):
        """Apply the fitted binning + categorical encoding to new data."""
        num = self._numeric_block(X)
        cross = self._cross_block(X)
        if cross.shape[1]:
            num = np.hstack([num, cross]) if num.shape[1] else cross
        encoded_blocks = []
        if self.cat_features_:
            codes = self._codes_for_transform(X)
            if self.combo_pairs_:
                combo_codes = self._combo_codes_for_transform(X)
                codes = np.hstack([codes, combo_codes])
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

    def _build_feature_map(self, n_targets):
        """Map each combined-matrix column back to its original input column.
        Block order is [numeric | cross | per-target (cat + combo)]. Cross and
        combo columns map to the lower-indexed feature of their pair, so their
        split gains fold into the right importance bucket."""
        combo_orig = [min(i, j) for i, j in self.combo_pairs_]
        fmap = list(self.num_features_)
        fmap.extend(min(i, j) for i, j, _op in self.cross_pairs)
        for _ in range(n_targets):
            fmap.extend(self.cat_features_)
            fmap.extend(combo_orig)
        self.feature_map_ = np.array(fmap, dtype=np.int64)

        max_idx = max(self.num_features_, default=-1)
        if self.cat_features_:
            max_idx = max(max_idx, max(self.cat_features_))
        self.n_input_features_ = max_idx + 1
