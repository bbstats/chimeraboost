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
    cat_combinations_selective : bool
        Selective variant of ``cat_combinations``: instead of all C(n_cat, 2)
        pairs, keep only those whose target mutual information beats BOTH parent
        columns (the combo adds interaction signal beyond its marginals), ranked
        and capped at ``cat_combinations_max_pairs``. Unlike the plain flag it is
        meant for MIXED data -- selecting a few high-signal interactions instead
        of flooding the tree with combos that crowd out the numeric features.
    cat_combinations_max_pairs : int
        Maximum number of selected combo columns (top-k by target MI).
    """

    def __init__(self, max_bins=128, cat_smoothing=1.0, random_state=None,
                 cat_n_permutations=4, cat_combinations=False,
                 onehot_low_card=False, onehot_max_card=8,
                 cat_combinations_selective=False, cat_combinations_max_pairs=20,
                 cat_aware_binning=False, cat_max_bins=254):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state
        self.cat_n_permutations = int(cat_n_permutations)
        self.cat_combinations = bool(cat_combinations)
        self.onehot_low_card = bool(onehot_low_card)
        self.onehot_max_card = int(onehot_max_card)
        self.cat_combinations_selective = bool(cat_combinations_selective)
        self.cat_combinations_max_pairs = int(cat_combinations_max_pairs)
        self.cat_aware_binning = bool(cat_aware_binning)
        self.cat_max_bins = int(cat_max_bins)

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _combo_values(X, f_a, f_b):
        """The synthetic "val_a_x_val_b" string column for a feature pair."""
        col_a = np.asarray(X[:, f_a], dtype=str)
        col_b = np.asarray(X[:, f_b], dtype=str)
        return np.char.add(np.char.add(col_a, "_x_"), col_b)

    @staticmethod
    def _assoc_labels(encode_targets):
        """Reduce the ordered-TS encode targets to a single integer label vector
        for measuring categorical->target association (combo selection only).

        * multiclass (K one-hot targets) -> the class index (argmax)
        * binary / already-discrete single target -> the values as labels
        * regression (continuous single target) -> decile bins
        Selection uses the full training target, like any mutual-info feature
        screen; the ordered encoder still prevents value leakage at encode time.
        """
        if len(encode_targets) > 1:
            return np.argmax(np.column_stack(encode_targets), axis=1)
        t = np.asarray(encode_targets[0], dtype=np.float64)
        uniq = np.unique(t)
        if uniq.size <= 20:                       # already categorical/binary
            return np.searchsorted(uniq, t)
        # Continuous: decile-bin (robust, dependency-free).
        edges = np.quantile(t, np.linspace(0, 1, 11)[1:-1])
        return np.searchsorted(edges, t)

    @staticmethod
    def _mutual_info(codes, labels):
        """Mutual information I(codes; labels) in nats, both integer-coded. A
        dependency-free contingency-table estimate; used to rank candidate combo
        columns by how much they explain the target."""
        n = labels.shape[0]
        if n == 0:
            return 0.0
        a = codes - codes.min() if codes.size else codes
        b = labels - labels.min() if labels.size else labels
        na, nb = int(a.max()) + 1, int(b.max()) + 1
        joint = np.zeros((na, nb), dtype=np.float64)
        np.add.at(joint, (a, b), 1.0)
        joint /= n
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        denom = pa @ pb
        nz = joint > 0
        return float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))

    def _split_columns_fit(self, X, cat_features, assoc_labels=None):
        """Split input into a numeric matrix and an integer-code matrix for the
        categorical columns, learning the category->code maps on the way.
        When cat_combinations is True, appends combo codes after the base codes.
        ``assoc_labels`` (integer target labels) enables selective combos: only
        the top pairs by target mutual information that beat both parents."""
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
        n_cat = len(self.cat_features_)
        selective = self.cat_combinations_selective and assoc_labels is not None
        enable_combos = (self.cat_combinations or selective) and n_cat >= 2
        if enable_combos:
            # Factorize every candidate pair once.
            cand = []   # (f_a, f_b, codes, cats)
            for a in range(n_cat):
                for b in range(a + 1, n_cat):
                    f_a, f_b = self.cat_features_[a], self.cat_features_[b]
                    c, cats = factorize(self._combo_values(X, f_a, f_b))
                    cand.append((f_a, f_b, a, b, c, cats))
            if selective:
                # Keep only combos whose target MI beats BOTH parents (they add
                # interaction signal beyond their marginals), ranked by MI and
                # capped at cat_combinations_max_pairs. Works on MIXED data --
                # the numeric columns are untouched, so combos no longer crowd
                # them out indiscriminately (the C1/auto-combo lesson).
                parent_mi = [self._mutual_info(codes[:, j], assoc_labels)
                             for j in range(n_cat)]
                scored = []
                for (f_a, f_b, a, b, c, cats) in cand:
                    mi = self._mutual_info(c, assoc_labels)
                    if mi > max(parent_mi[a], parent_mi[b]):
                        scored.append((mi, f_a, f_b, c, cats))
                scored.sort(key=lambda t: t[0], reverse=True)
                cand = [(f_a, f_b, None, None, c, cats)
                        for (_mi, f_a, f_b, c, cats)
                        in scored[:self.cat_combinations_max_pairs]]
            combo_cols = []
            for (f_a, f_b, _a, _b, c, cats) in cand:
                self.combo_pairs_.append((f_a, f_b))
                self.combo_maps_.append({v: i for i, v in enumerate(cats)})
                combo_cols.append(c)
            if combo_cols:
                codes = np.hstack(
                    [codes, np.column_stack(combo_cols).astype(np.int64)])
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
        assoc_labels = (self._assoc_labels(encode_targets)
                        if self.cat_combinations_selective else None)
        num, codes = self._split_columns_fit(X, cat_features, assoc_labels)

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

        # C4: cat-aware binning gives the target-encoded categorical columns
        # (everything that isn't a raw numeric) a larger bin budget, so the tree
        # can make sharper splits on the categorical target statistic. Off by
        # default -> uniform max_bins (scalar, byte-identical to before).
        if self.cat_aware_binning and not self.is_numeric_binned_.all():
            bins = np.where(self.is_numeric_binned_, self.max_bins,
                            self.cat_max_bins).astype(np.int64)
            self.binner_ = Binner(bins)
        else:
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
