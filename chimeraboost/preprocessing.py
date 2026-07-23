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
from numba import njit

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


@njit(cache=True)
def _grouped_kahan_sum(codes, vals, n_groups):
    """Per-group Kahan-compensated sums (row order, sequential).

    Kahan matters for reproducibility, not just accuracy: the gdiff group
    means were historically computed by pandas' groupby, whose kernel is
    Kahan-compensated, and existing fitted models / identity goldens embed
    those exact floats. A naive accumulation drifts in the last ulp on large
    groups, so a bit-identical replacement has to compensate the same way.
    """
    out = np.zeros(n_groups)
    comp = np.zeros(n_groups)
    for i in range(codes.shape[0]):
        c = codes[i]
        y = vals[i] - comp[c]
        t = out[c] + y
        comp[c] = (t - out[c]) - y
        out[c] = t
    return out


class CatTransformCache:
    """Canonical factorizations of one input matrix's categorical columns
    (and combo string columns), computed once per fit/predict call.

    Bagged members each learned their own category->code map on their own
    bootstrap, so fit-time codes can't be shared across the bag -- but
    hashing every row of the batch is member-independent. The bagged parent
    passes one cache into every member's transform; the first member pays the
    per-row factorize and each further member maps only the ~n_unique
    canonical categories through its own dict and gathers. Callers must pass
    one cache only across transforms of the *same* matrix ``X`` (entries are
    keyed by column index alone).
    """

    def __init__(self):
        self._columns = {}
        self._combos = {}

    def column(self, X, f):
        """(codes, categories) of raw column ``f``, first-appearance order."""
        out = self._columns.get(f)
        if out is None:
            out = self._columns[f] = factorize(X[:, f])
        return out

    def combo(self, X, f_a, f_b):
        """(codes, categories) of the synthetic combo column for a pair."""
        out = self._combos.get((f_a, f_b))
        if out is None:
            out = self._combos[(f_a, f_b)] = factorize(
                FeaturePreprocessor._combo_values(X, f_a, f_b))
        return out


def _remap_codes(categories, mapping, default):
    """Vectorize a fit-time {category value -> code/float} dict over canonical
    categories: the returned array, gathered by canonical codes, equals the
    row-wise dict lookup with ``default`` for unseen categories."""
    dtype = np.int64 if isinstance(default, int) else np.float64
    return np.fromiter((mapping.get(u, default) for u in categories.tolist()),
                       dtype=dtype, count=len(categories))


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
        Cross features: each (i, j, op) appends the column
        ``X[:, i] - X[:, j]`` (op="diff"), ``X[:, i] * X[:, j]`` (op="prod"),
        or ``X[:, i] - mean_fit(X[:, i] | X[:, j])`` (op="gdiff"), binned like
        any numeric column. Oblivious trees can only approximate an
        interaction with a depth-limited staircase (the same split is applied
        to every leaf of a level); a cross column turns e.g. the ``x_i < x_j``
        boundary -- or "above this row's own category's average" -- into a
        single split. Indices refer to ORIGINAL input columns; for diff/prod
        both must be numeric, for gdiff ``i`` is numeric and ``j`` is a
        ``cat_features`` column. gdiff group means are learned from the fit
        rows only (they use no target values, so the same map serves fit and
        predict); unseen categories fall back to the global mean of column i.
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

    def _split_columns_fit(self, X, cat_features, cat_ctx=None):
        """Split input into a numeric matrix and an integer-code matrix for the
        categorical columns, learning the category->code maps on the way.
        When cat_combinations is True, appends combo codes after the base codes."""
        n_features = X.shape[1]
        cat_set = set(cat_features or [])
        self.cat_features_ = sorted(cat_set)
        self.num_features_ = [f for f in range(n_features) if f not in cat_set]
        if cat_ctx is None:
            cat_ctx = CatTransformCache()

        num = self._numeric_block(X)

        if self.cat_features_:
            codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
            self.cat_maps_ = []
            for j, f in enumerate(self.cat_features_):
                c, cats = cat_ctx.column(X, f)
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
                    c, cats = cat_ctx.combo(X, f_a, f_b)
                    self.combo_pairs_.append((f_a, f_b))
                    self.combo_maps_.append({v: i for i, v in enumerate(cats)})
                    combo_cols.append(c)
            if combo_cols:
                codes = np.hstack(
                    [codes, np.column_stack(combo_cols).astype(np.int64)])
        return num, codes

    def _fit_gdiff(self, X, sample_weight=None, cat_ctx=None):
        """Learn the per-category means backing the gdiff cross columns:
        for each (i, j, "gdiff") pair a {category value -> mean of X[:, i]}
        map plus the global-mean fallback for categories unseen at fit.
        Means are computed over rows with a finite X[:, i] (a NaN numeric
        contributes nothing, mirroring the binner's quantile treatment) and,
        when ``sample_weight`` is given, weighted by it so zero-weight rows
        never shape another row's centering."""
        self.gdiff_maps_ = []
        pairs = [(i, j) for i, j, op in self.cross_pairs if op == "gdiff"]
        if not pairs:
            return
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        for i, j in pairs:
            a = np.asarray(X[:, i], dtype=np.float64)
            ok = np.isfinite(a)
            if ok.all():
                codes, cats = cat_ctx.column(X, j)
                v = a
            else:
                # Factorize the finite rows only: category (= summation)
                # order is first appearance among the contributing rows.
                codes, cats = factorize(np.asarray(X[:, j], dtype=object)[ok])
                v = a[ok]
            w = (np.ones(v.shape[0]) if sample_weight is None
                 else np.asarray(sample_weight, dtype=np.float64)[ok])
            vsum = _grouped_kahan_sum(codes, v * w, len(cats))
            wsum = _grouped_kahan_sum(codes, w, len(cats))
            tot_w = float(np.sum(wsum))
            global_mean = (float(np.sum(vsum) / tot_w) if tot_w > 0 else 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = vsum / wsum
            means = np.where(np.isfinite(means), means, global_mean)
            self.gdiff_maps_.append(
                (dict(zip(cats.tolist(), means.tolist())), global_mean))

    def _cross_block(self, X, cat_ctx=None, num=None):
        """Compute the cross-feature columns (float64) from raw input. NaN in
        a numeric parent propagates to the cross (binned to the missing bucket
        like any numeric NaN); gdiff maps a NaN category to its own "__nan__"
        group and an unseen category to the global mean.

        Numeric parents are read from ``num`` (the float64 numeric block;
        pass the one already built for this matrix, else it is computed
        here): one cast per input column instead of one per pair. On object
        arrays (categoricals present) the per-pair element-wise casts were
        the dominant predict-time cost of cross features."""
        if not self.cross_pairs:
            return np.empty((X.shape[0], 0))
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        if num is None:
            num = self._numeric_block(X)
        pos = {f: k for k, f in enumerate(self.num_features_)}
        cols = []
        g = 0
        for i, j, op in self.cross_pairs:
            a = num[:, pos[i]]
            if op == "gdiff":
                means, global_mean = self.gdiff_maps_[g]
                g += 1
                codes, cats = cat_ctx.column(X, j)
                cols.append(a - _remap_codes(cats, means, global_mean)[codes])
                continue
            b = num[:, pos[j]]
            cols.append(a - b if op == "diff" else a * b)
        return np.column_stack(cols)

    def _codes_for_transform(self, X, cat_ctx=None):
        """Map categorical columns to the codes learned at fit time; unseen
        categories get -1 (the encoder then falls back to the prior). Each
        column is factorized once and only its unique values pass through the
        fit-time dict; with a shared ``cat_ctx`` the factorization is also
        reused across bagged members."""
        if not self.cat_features_:
            return np.empty((X.shape[0], 0), dtype=np.int64)
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        for j, f in enumerate(self.cat_features_):
            c, cats = cat_ctx.column(X, f)
            codes[:, j] = _remap_codes(cats, self.cat_maps_[j], -1)[c]
        return codes

    def _combo_codes_for_transform(self, X, cat_ctx=None):
        """Reconstruct combination codes for transform using stored combo maps."""
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        combo_codes = np.empty((X.shape[0], len(self.combo_pairs_)), dtype=np.int64)
        for k, (f_a, f_b) in enumerate(self.combo_pairs_):
            c, cats = cat_ctx.combo(X, f_a, f_b)
            combo_codes[:, k] = _remap_codes(cats, self.combo_maps_[k], -1)[c]
        return combo_codes

    # ---- fit / transform -----------------------------------------------------
    def fit_transform(self, X, encode_targets, cat_features, sample_weight=None):
        """encode_targets: list of 1D arrays used for ordered TS (len T).

        ``sample_weight`` (mean-1 normalized, ``None`` == uniform) is forwarded to
        the ordered-target encoder and the binner so zero-weight rows shape
        neither the categorical statistics nor the bin borders. ``None`` is the
        unweighted path, bit-identical to before this argument existed."""
        cat_ctx = CatTransformCache()
        num, codes = self._split_columns_fit(X, cat_features, cat_ctx)
        self._fit_gdiff(X, sample_weight, cat_ctx)
        cross = self._cross_block(X, cat_ctx, num=num)
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
                encoded_blocks.append(
                    enc.fit_transform(codes, target, sample_weight))
                self.encoders_.append(enc)

        feat = self._stack(num, encoded_blocks)
        self._build_feature_map(len(encode_targets))
        # Block order is [numeric | per-target TS]. Only true numeric columns
        # carry an ordinal meaning usable by linear-leaf models; mark them so the
        # booster can pick linear-term features (the TS blocks are excluded).
        self.is_numeric_binned_ = np.zeros(feat.shape[1], dtype=bool)
        self.is_numeric_binned_[:num.shape[1]] = True

        self.binner_ = Binner(self.max_bins)
        X_binned = self.binner_.fit_transform(feat, sample_weight)
        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X, cat_ctx=None):
        """Apply the fitted binning + categorical encoding to new data.
        ``cat_ctx`` (internal) shares the per-column canonical factorizations
        across the members of a bagged ensemble -- see CatTransformCache."""
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        num = self._numeric_block(X)
        cross = self._cross_block(X, cat_ctx, num=num)
        if cross.shape[1]:
            num = np.hstack([num, cross]) if num.shape[1] else cross
        encoded_blocks = []
        if self.cat_features_:
            codes = self._codes_for_transform(X, cat_ctx)
            if self.combo_pairs_:
                combo_codes = self._combo_codes_for_transform(X, cat_ctx)
                codes = np.hstack([codes, combo_codes])
            for enc in self.encoders_:
                encoded_blocks.append(enc.transform(codes))
        feat = self._stack(num, encoded_blocks)
        return self.binner_.transform(feat)

    @classmethod
    def from_base_with_cross(cls, base, cross_pairs, X, sample_weight=None):
        """A fitted preprocessor equal to refitting ``base``'s configuration
        with ``cross_pairs`` added, built by reusing ``base``'s fitted state.

        Every fit artifact is computed independently per column -- category
        maps, TS encodings, quantile borders, bin indices -- and appending
        cross columns leaves the base columns' inputs untouched, so the base
        results are shared by reference and only the cross columns are
        computed here. Bit-identical to the from-scratch fit with the same
        ``cross_pairs``; ``base`` must itself have no cross features.

        Returns ``(prep, cross_binner, cross_binned)``: the fitted augmented
        preprocessor, the binner covering only the cross columns (for binning
        eval-set cross blocks), and the binned cross block for ``X``'s rows.
        The caller splices ``cross_binned`` into the base binned matrix at
        column offset ``len(base.num_features_)`` (stacked column order is
        [numeric | cross | TS blocks]).
        """
        if base.cross_pairs:
            raise ValueError("base preprocessor already has cross features")
        prep = cls(base.max_bins, base.cat_smoothing, base.random_state,
                   base.cat_n_permutations, base.cat_combinations, cross_pairs)
        prep.cat_features_ = base.cat_features_
        prep.num_features_ = base.num_features_
        prep.cat_maps_ = base.cat_maps_
        prep.combo_pairs_ = base.combo_pairs_
        prep.combo_maps_ = base.combo_maps_
        prep.encoders_ = base.encoders_

        cat_ctx = CatTransformCache()
        prep._fit_gdiff(X, sample_weight, cat_ctx)
        cross = prep._cross_block(X, cat_ctx)
        cross_binner = Binner(base.max_bins).fit(cross, sample_weight)
        nb = len(base.num_features_)
        bb = base.binner_
        binner = Binner(base.max_bins)
        binner.borders_ = (bb.borders_[:nb] + cross_binner.borders_
                           + bb.borders_[nb:])
        binner.n_bins_ = np.concatenate(
            [bb.n_bins_[:nb], cross_binner.n_bins_, bb.n_bins_[nb:]])
        binner.bin_centers_ = (bb.bin_centers_[:nb] + cross_binner.bin_centers_
                               + bb.bin_centers_[nb:])
        binner._build_flat_borders()
        prep.binner_ = binner
        prep.n_bins_ = binner.n_bins_
        prep.is_numeric_binned_ = np.zeros(len(binner.borders_), dtype=bool)
        prep.is_numeric_binned_[:nb + cross.shape[1]] = True
        prep._build_feature_map(max(1, len(base.encoders_)))
        return prep, cross_binner, cross_binner.transform(cross)

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
        # gdiff is a recentering of its numeric parent i, so its gain belongs
        # there; diff/prod keep the established min(i, j) convention.
        fmap.extend(i if op == "gdiff" else min(i, j)
                    for i, j, op in self.cross_pairs)
        for _ in range(n_targets):
            fmap.extend(self.cat_features_)
            fmap.extend(combo_orig)
        self.feature_map_ = np.array(fmap, dtype=np.int64)

        max_idx = max(self.num_features_, default=-1)
        if self.cat_features_:
            max_idx = max(max_idx, max(self.cat_features_))
        self.n_input_features_ = max_idx + 1
