"""Shared feature preprocessing for every ChimeraBoost estimator.

Turns a raw matrix -- numeric, categorical, or an object-dtype mixture -- into
integer bins for the tree builder, and keeps everything needed to reproduce the
same transform at predict time.

Categoricals are encoded with ordered target statistics. The encoder is fit
against a *list* of target vectors:

  * regression / binary -> one target (y, or the 0/1 label)
  * multiclass          -> K one-hot targets (one ordered-TS column per class)

So one categorical column can expand into K numeric columns for multiclass,
exactly like CatBoost's per-class target statistics.

``feature_map_`` maps each combined-matrix column back to its original input
column index, so importances can be aggregated in the user's feature space.

Stacked column order is [numeric | count | cross | per-target (cat + combo)]:
the optional per-category count columns extend the numeric block, and the
cross block follows them.
"""

import itertools

import numpy as np
from numba import njit

from .binning import Binner
from .target_encoding import OrderedTargetEncoder, _factorize_numeric, factorize


# A count column only pays above the binner's resolution: below it the target
# statistic already resolves categories one by one, and a count column added
# noise where it was measured.
CAT_COUNT_MIN_CARD = 256


def as_model_array(X, want_object):
    """Convert a raw feature matrix to the numpy array the model consumes.

    ``want_object`` selects object dtype (categoricals present, decoded
    downstream); otherwise float64.

    pandas nullable dtypes (Int64/Float64/boolean and ``string``) store missing
    values as ``pd.NA``/``NAType``, which neither casts to float nor compares
    like ``np.nan``: a plain ``np.asarray(df, dtype=float)`` raises the cryptic
    "float() argument must be ... not 'NAType'". ``to_numpy(na_value=np.nan)``
    maps every flavor of NA to ``np.nan``, which the binner and encoder already
    handle. Inputs with no na_value-aware ``to_numpy`` (plain ndarrays, polars
    frames) fall back to ``np.asarray``.
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
    """Per-group Kahan-compensated sums, in row order, sequential.

    Kahan is here for reproducibility, not just accuracy. The gdiff group means
    used to come from pandas' groupby, whose kernel is Kahan-compensated, and
    fitted models plus the identity goldens embed those exact floats. Naive
    accumulation drifts in the last ulp on large groups, so a bit-identical
    replacement has to compensate the same way.
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


def _cast_numeric_block(X, num_features):
    """The numeric columns at positions ``num_features`` as float64.

    When every column is numeric, ``num_features`` is exactly
    range(n_features), so a plain asarray suffices -- the fancy-index
    gather ``X[:, list]`` would copy the whole matrix, a large predict-time
    tax on wide batches.
    """
    if not num_features:
        return np.empty((X.shape[0], 0))

    if len(num_features) == X.shape[1]:
        return np.asarray(X, dtype=np.float64)

    return np.asarray(X[:, num_features], dtype=np.float64)


class CatTransformCache:
    """Canonical factorizations of one matrix's categorical and combo columns.

    Computed once per fit/predict call, then shared across a bagged ensemble.
    Each member learns its own category->code map on its own bootstrap, so
    fit-time codes can't be shared across the bag -- but hashing every row of
    the batch is member-independent. The parent passes one cache to every
    member: the first pays the per-row factorize, each later one maps only the
    ~n_unique canonical categories through its own dict and gathers.

    One cache is valid for one matrix ``X`` only -- entries are keyed by column
    index alone.

    Child form: ``CatTransformCache(parent=..., parent_X=..., rows=...)`` is a
    view of a ``parent`` cache over the full matrix ``parent_X``, restricted to
    the row-index array ``rows``. A leg's factorization is derived from the one
    parent pass by gathering and re-numbering in first-appearance order, so a
    fit factorizes each categorical column once instead of once per leg. A
    row-count guard falls back to a plain ``factorize`` whenever the matrix at
    hand is not the leg this child was built for.

    The cache also carries the matrix's float64 numeric block, cast once per
    fit on the parent and gathered per leg by row index. Consumers read the
    block and never write into it -- that invariant is what makes the one
    shared copy safe.
    """

    def __init__(self, parent=None, parent_X=None, rows=None):
        self._columns = {}
        self._combos = {}
        self._numeric = {}
        self._parent = parent
        self._parent_X = parent_X
        self._rows = rows

    def column(self, X, f):
        """(codes, categories) of raw column ``f``, first-appearance order."""
        out = self._columns.get(f)
        if out is None:
            parent, pX, rows = self._parent, self._parent_X, self._rows
            if (parent is not None and pX is not None and rows is not None
                    and X.shape[0] == rows.shape[0]):
                pc, pcats = parent.column(pX, f)
                codes, keys = _rerank_first_appearance(
                    np.ascontiguousarray(pc[rows]), len(pcats))
                if not isinstance(pcats, np.ndarray):
                    pcats = np.asarray(pcats, dtype=object)
                out = self._columns[f] = (codes, pcats[keys])
            else:
                out = self._columns[f] = factorize(X[:, f])
        return out

    def combo(self, X, f_a, f_b):
        """(codes, categories) of the synthetic combo column for a pair.

        A combo category is the PAIR of the parents' canonical categories, not
        the concatenated string it used to be. "a_x_b" aliased distinct pairs
        whose values contain the delimiter, stringified NaN to "nan" (bypassing
        factorize's ``__nan__`` sentinel, so a real missing value merged with
        the literal string "nan"), and split int/float spellings of one value
        ("1" vs "1.0"). Pair codes inherit the parents' factorize semantics
        exactly and skip the per-row string building.

        Categories are (val_a, val_b) tuples in first-appearance order, like
        every factorize.
        """
        out = self._combos.get((f_a, f_b))
        if out is None:
            ca, cats_a = self.column(X, f_a)
            cb, cats_b = self.column(X, f_b)
            kb = len(cats_b)
            pcodes, pkeys = _factorize_int(ca * np.int64(kb) + cb)
            cats = [(cats_a[k // kb], cats_b[k % kb]) for k in pkeys]
            out = self._combos[(f_a, f_b)] = (pcodes, cats)
        return out

    def combo_str(self, X, f_a, f_b):
        """Legacy string-concat combo factorization.

        Kept only so models pickled before the pair-code change (string-keyed
        ``combo_maps_``) keep predicting identically.
        """
        out = self._combos.get(("str", f_a, f_b))
        if out is None:
            out = self._combos[("str", f_a, f_b)] = factorize(
                FeaturePreprocessor._combo_values(X, f_a, f_b))
        return out

    def numeric(self, X, num_features):
        """The float64 numeric block of ``X`` at positions ``num_features``.

        A child derives its leg's block from the one parent cast by gathering
        ``rows`` -- the object-to-float64 cast is element-wise, so gathering
        after the cast is bit-identical to casting the leg's own rows. The
        row-count guard falls back to a plain cast whenever the matrix at hand
        is not the leg this child was built for. Callers must not write into
        the returned block.
        """
        key = tuple(num_features)
        out = self._numeric.get(key)
        if out is None:
            parent, pX, rows = self._parent, self._parent_X, self._rows
            if (parent is not None and pX is not None and rows is not None
                    and X.shape[0] == rows.shape[0]):
                out = self._numeric[key] = parent.numeric(
                    pX, num_features)[rows]
            else:
                out = self._numeric[key] = _cast_numeric_block(X, num_features)
        return out


def _factorize_int(vals):
    """First-appearance factorize of an int64 array (combo pair keys).

    Returns (codes, keys) -- the same contract as ``factorize``, except keys
    stay a plain list, because wrapping tuples derived from them in
    ``np.asarray`` would build a 2-D array.

    Vectorized: np.unique's sorted ids are remapped onto first-appearance ranks
    (return_index gives each value's FIRST position), which is exactly the
    insertion order the old dict loop produced.
    """
    su, first, inv = np.unique(vals, return_index=True, return_inverse=True)

    order = np.argsort(first, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)

    codes = np.ascontiguousarray(rank[inv], dtype=np.int64)
    keys = [int(v) for v in su[order]]
    return codes, keys


@njit(cache=True)
def _rerank_first_appearance(codes, n_parent):
    """Re-number parent codes in first-appearance order over a row subset.

    ``codes`` holds the parent matrix's codes gathered at the subset rows
    (values in ``[0, n_parent)``). Returns ``(out, keys)``: the subset's own
    first-appearance codes plus the parent code behind each one, so
    ``parent_categories[keys]`` is the subset's category array -- exactly what
    ``factorize`` on the subset's rows returns. One pass with a lookup table
    mapping parent code -> subset code.
    """
    if codes.shape[0] == 0 or n_parent <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lut = np.full(n_parent, np.int64(-1), np.int64)
    keys = np.empty(n_parent, dtype=np.int64)
    out = np.empty(codes.shape[0], dtype=np.int64)
    nxt = 0
    for i in range(codes.shape[0]):
        c = codes[i]
        if lut[c] < 0:
            lut[c] = nxt
            keys[nxt] = c
            nxt += 1
        out[i] = lut[c]
    return out, keys[:nxt]


def _remap_codes(categories, mapping, default):
    """Vectorize a fit-time {category -> code/float} dict over canonical categories.

    The returned array, gathered by canonical codes, equals the row-wise dict
    lookup, with ``default`` for unseen categories. ``categories`` is an
    ndarray, or a plain list for combo pair tuples.
    """
    cats = categories if isinstance(categories, list) else categories.tolist()
    dtype = np.int64 if isinstance(default, int) else np.float64
    return np.fromiter((mapping.get(u, default) for u in cats),
                       dtype=dtype, count=len(cats))


def _direct_hashed(col, mapping):
    """Per-row codes for a column already ruled non-numeric, or None.

    ``col`` is an object ndarray the caller probed with ``_factorize_numeric``
    (refused). Same missing-mask contract as ``_factorize_hashed``.
    """
    try:
        miss = np.not_equal(col, col) | np.equal(col, None)
    except (TypeError, ValueError):
        return None
    if not isinstance(miss, np.ndarray) or miss.dtype != np.bool_:
        return None
    if miss.any():
        col = col.copy()
        col[miss] = "__nan__"
    try:
        return np.fromiter(
            map(mapping.get, col.tolist(), itertools.repeat(-1)),
            dtype=np.int64, count=col.shape[0])
    except TypeError:
        return None


def _direct_codes(col, mapping):
    """Per-row fit-time codes via one dict lookup each, or None for factorize.

    Exact for hashed (string) columns: the fit-time dict defines the equality
    classes, so a direct ``mapping.get`` agrees with factorize-then-remap.
    None for numeric-accepted columns (float grouping merges ints past 2**53),
    an unsafe missing mask (the ``_factorize_hashed`` contract), or unhashable
    values -- the caller uses the existing path.

    The single monkeypatch point disabling the direct path: the speed script
    and tests replace this with ``lambda col, mapping: None``.
    """
    col = np.asarray(col, dtype=object)
    if _factorize_numeric(col) is not None:
        return None
    return _direct_hashed(col, mapping)


_DIRECT_CODES_ORIG = _direct_codes


class FeaturePreprocessor:
    """Converts raw mixed-type input into integer bins for the tree builder.

    Numeric columns are quantile-binned. Categorical columns are ordered-target
    encoded -- one encoded column per target passed to ``fit_transform`` -- and
    then binned alongside the numerics. The state needed to reproduce the
    transform at predict time is kept, along with ``feature_map_``, which maps
    each output column back to its original input column for importances.

    cat_combinations : bool
        Generate all C(n_cat, 2) pairwise categorical combinations as extra
        synthetic columns (e.g. "buying_x_maint") before target encoding.
        Mirrors CatBoost's feature combination step: it gives the tree the
        interaction effects individual categoricals can't capture. Only active
        with 2 or more categorical columns.
    cross_pairs : list[(int, int, str)] | None
        Cross features. Each (i, j, op) appends one column -- ``X[:, i] -
        X[:, j]`` (op="diff"), ``X[:, i] * X[:, j]`` (op="prod"), or
        ``X[:, i] - mean_fit(X[:, i] | X[:, j])`` (op="gdiff") -- binned like
        any numeric column.

        Oblivious trees can only approximate an interaction with a
        depth-limited staircase, since the same split is applied to every leaf
        of a level. A cross column turns e.g. the ``x_i < x_j`` boundary -- or
        "above this row's own category's average" -- into a single split.

        Indices refer to ORIGINAL input columns. diff/prod need both parents
        numeric; gdiff needs ``i`` numeric and ``j`` in ``cat_features``. gdiff
        group means are learned from the fit rows only (they use no target
        values, so one map serves fit and predict); unseen categories fall back
        to the global mean of column i.
    cat_count_features : bool, default False
        For every categorical column with at least CAT_COUNT_MIN_CARD (256)
        training categories, append one float column holding each row's
        category count (the weight total when ``sample_weight`` is given)
        at fit time (unseen categories read 0.0 at transform). The count
        columns extend the numeric block but stay
        invisible to cross-feature candidacy and linear-leaf term selection.
        Off by default; when off, preprocessing is bit-identical to before.
    """

    def __init__(self, max_bins=128, cat_smoothing=1.0, random_state=None,
                 cat_n_permutations=4, cat_combinations=False,
                 cross_pairs=None, cat_count_features=False):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state
        self.cat_n_permutations = int(cat_n_permutations)
        self.cat_combinations = bool(cat_combinations)
        self.cross_pairs = list(cross_pairs) if cross_pairs else []
        self.cat_count_features = bool(cat_count_features)

    # ---- helpers -------------------------------------------------------------

    def _numeric_block(self, X, cat_ctx=None):
        """The numeric columns as float64.

        With a ``cat_ctx`` the block comes from the per-fit cache -- cast once
        on the parent matrix, gathered per leg -- else it is cast directly
        (see ``_cast_numeric_block``). Callers must not write into it.
        """
        if cat_ctx is not None:
            return cat_ctx.numeric(X, self.num_features_)
        return _cast_numeric_block(X, self.num_features_)

    @staticmethod
    def _combo_values(X, f_a, f_b):
        """The synthetic "val_a_x_val_b" string column for a feature pair.

        LEGACY: only serves models pickled with string-keyed combo maps (see
        ``CatTransformCache.combo_str``).
        """
        col_a = np.asarray(X[:, f_a], dtype=str)
        col_b = np.asarray(X[:, f_b], dtype=str)
        return np.char.add(np.char.add(col_a, "_x_"), col_b)

    def _split_columns_fit(self, X, cat_features, cat_ctx=None,
                           sample_weight=None):
        """Split input into a numeric matrix and a categorical code matrix.

        Learns the category->code maps on the way. When cat_combinations is
        True, combo codes are appended after the base codes. ``sample_weight``
        (mean-1 normalized, ``None`` == uniform) reaches only the count
        tables; the encoder and binner take it separately in fit_transform.
        """
        n_features = X.shape[1]
        cat_set = set(cat_features or [])
        self.cat_features_ = sorted(cat_set)
        self.num_features_ = [f for f in range(n_features) if f not in cat_set]

        if cat_ctx is None:
            cat_ctx = CatTransformCache()

        num = self._numeric_block(X, cat_ctx)

        if self.cat_features_:
            codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
            self.cat_maps_ = []
            all_cats = []
            for j, f in enumerate(self.cat_features_):
                c, cats = cat_ctx.column(X, f)
                codes[:, j] = c
                self.cat_maps_.append({v: i for i, v in enumerate(cats)})
                all_cats.append(cats)
        else:
            codes = np.empty((X.shape[0], 0), dtype=np.int64)
            self.cat_maps_ = []
            all_cats = []

        # Opt-in per-category count columns (see CAT_COUNT_MIN_CARD): one
        # float column per qualifying categorical, stacked at the end of the
        # numeric block. They bin like any numeric column but stay invisible
        # to cross candidacy (pairs reference raw input columns) and to
        # linear-leaf term selection (is_numeric_binned_ is False for them).
        count_block = self._fit_count_tables(codes, all_cats, sample_weight)
        if count_block.shape[1]:
            num = (np.hstack([num, count_block]) if num.shape[1]
                   else count_block)
        self.n_numeric_block_ = (len(self.num_features_)
                                 + len(self.count_features_))

        # 2-way combinations: each pair becomes a categorical column whose
        # categories are (val_a, val_b) pairs, target-encoded like any other
        # cat column, so the tree sees interactions single columns can't hold.
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
        """Learn the per-category means backing the gdiff cross columns.

        For each (i, j, "gdiff") pair: a {category value -> mean of X[:, i]}
        map, plus the global-mean fallback for categories unseen at fit.

        Means use only rows with a finite X[:, i] -- a NaN numeric contributes
        nothing, mirroring the binner's quantile treatment -- and, when
        ``sample_weight`` is given, are weighted by it so zero-weight rows never
        shape another row's centering.
        """
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
        """Compute the cross-feature columns (float64) from raw input.

        A NaN in a numeric parent propagates to the cross column and bins to
        the missing bucket like any numeric NaN. gdiff maps a NaN category to
        its own "__nan__" group and an unseen category to the global mean.

        Numeric parents come from ``num``, the float64 numeric block -- pass the
        one already built for this matrix, else it is built here. That is one
        cast per input column instead of one per pair; on object arrays
        (categoricals present) the per-pair element-wise casts were the dominant
        predict-time cost of cross features.
        """
        if not self.cross_pairs:
            return np.empty((X.shape[0], 0))

        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        if num is None:
            num = self._numeric_block(X, cat_ctx)

        pos = {f: k for k, f in enumerate(self.num_features_)}

        # Write each cross column straight into the output block: the ``out=``
        # ufunc form skips both the per-pair temporary and a final column_stack.
        out = np.empty((X.shape[0], len(self.cross_pairs)))

        g = 0
        for k, (i, j, op) in enumerate(self.cross_pairs):
            a = num[:, pos[i]]

            if op == "gdiff":
                means, global_mean = self.gdiff_maps_[g]
                g += 1
                codes, cats = cat_ctx.column(X, j)
                np.subtract(a, _remap_codes(cats, means, global_mean)[codes],
                            out=out[:, k])
                continue

            b = num[:, pos[j]]
            if op == "diff":
                np.subtract(a, b, out=out[:, k])
            else:
                np.multiply(a, b, out=out[:, k])

        return out

    def _direct_ineligible(self):
        """Categorical columns the direct path must skip in this transform.

        A column read again later through the cache -- a parent of any combo
        pair or the ``j`` of any gdiff pair -- fills the cache once via the
        old factorize path so the later block reuses it, instead of paying a
        dict pass plus a factorize.
        """
        skip = set()
        for f_a, f_b in getattr(self, "combo_pairs_", []):
            skip.add(f_a)
            skip.add(f_b)
        for _i, j, op in getattr(self, "cross_pairs", []):
            if op == "gdiff":
                skip.add(j)
        return skip

    def _try_direct_column(self, X, f, j, codes):
        """Try the direct path for one eligible column; True when handled.

        Calls ``_factorize_numeric`` once: on success its result is remapped
        directly (``factorize`` returns exactly that, so no second call); on
        refusal the hashed direct lookup runs with no second numeric probe.
        A patched ``_direct_codes`` (speed OFF arm) takes the legacy route.
        """
        if _direct_codes is not _DIRECT_CODES_ORIG:
            direct = _direct_codes(X[:, f], self.cat_maps_[j])
            if direct is None:
                return False
            codes[:, j] = direct
            return True
        col = np.asarray(X[:, f], dtype=object)
        num_res = _factorize_numeric(col)
        if num_res is not None:
            c, cats = num_res
            codes[:, j] = _remap_codes(cats, self.cat_maps_[j], -1)[c]
            return True
        direct = _direct_hashed(col, self.cat_maps_[j])
        if direct is None:
            return False
        codes[:, j] = direct
        return True

    def _codes_with_direct(self, X, cat_ctx):
        """Base codes allowing direct for columns no later block re-reads."""
        skip = self._direct_ineligible()
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        for j, f in enumerate(self.cat_features_):
            if f not in skip and self._try_direct_column(X, f, j, codes):
                continue
            c, cats = cat_ctx.column(X, f)
            codes[:, j] = _remap_codes(cats, self.cat_maps_[j], -1)[c]
        return codes

    def _codes_for_transform(self, X, cat_ctx=None, allow_direct=None):
        """Map categorical columns to the codes learned at fit time.

        Unseen categories get -1, and the encoder falls back to the prior.
        Single-model transforms allow the direct path (``allow_direct``) for
        columns no later block re-reads; a shared ``cat_ctx`` factorizes once
        and reuses it across bagged members too.
        """
        if not self.cat_features_:
            return np.empty((X.shape[0], 0), dtype=np.int64)
        if allow_direct is None:
            allow_direct = cat_ctx is None
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        if allow_direct:
            return self._codes_with_direct(X, cat_ctx)
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        for j, f in enumerate(self.cat_features_):
            c, cats = cat_ctx.column(X, f)
            codes[:, j] = _remap_codes(cats, self.cat_maps_[j], -1)[c]
        return codes

    def _fit_count_tables(self, codes, all_cats=None, sample_weight=None):
        """Select the count-feature categoricals and build their train block.

        A categorical qualifies when its training cardinality reaches
        CAT_COUNT_MIN_CARD -- or, on the replay-refit path, when the donor
        prep selected it (``_pinned_count_features``): the adopted binner's
        borders and the replayed splits address columns by position, so the
        layout must match the donor's even if a near-threshold column's
        cardinality drifted across the cutoff between the two row samples.

        On the replay-refit path the counts themselves come from the donor
        too (``_pinned_cat_counts``): the donor's per-category counts,
        re-indexed onto this fit's codes, 0.0 for categories the donor never
        saw. Refitting them on these rows would rescale every count by the
        row-count ratio and push each row into a higher bin than the one the
        replayed split was chosen for -- the binner is held still precisely
        so replayed thresholds keep their meaning, and the count lookups
        must be held still with it.

        Otherwise the count is each category's training-row count -- the
        weight total when ``sample_weight`` (mean-1 normalized, ``None`` ==
        uniform) is given, so zero-weight rows shape neither the counts
        nor, downstream, the borders.

        Sets ``count_features_`` (original column indices, in
        ``cat_features_`` order) and ``cat_counts_`` (one per-category
        count lookup per selected column). ``codes`` holds the base
        categorical codes, before any combo columns are appended, and
        ``all_cats`` the matching per-column category arrays in code order
        (read only on the pinned path). Returns the (n, k) float64 training
        block, (n, 0) when the flag is off or nothing qualifies.
        """
        self.count_features_ = []
        self.cat_counts_ = []
        if not self.cat_count_features:
            return np.empty((codes.shape[0], 0))

        pinned = getattr(self, "_pinned_count_features", None)
        pinned_counts = getattr(self, "_pinned_cat_counts", None)
        w = (None if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float64))
        cols = []
        for j, f in enumerate(self.cat_features_):
            n_cats = len(self.cat_maps_[j])
            if pinned is not None:
                selected = f in pinned
            else:
                selected = n_cats >= CAT_COUNT_MIN_CARD
            if not selected:
                continue
            c = codes[:, j]
            if pinned_counts is not None:
                counts = self._pinned_count_column(j, f, all_cats, pinned,
                                                   pinned_counts)
            elif w is None:
                counts = np.bincount(c, minlength=n_cats).astype(np.float64)
            else:
                counts = np.bincount(c, weights=w,
                                     minlength=n_cats).astype(np.float64)
            self.cat_counts_.append(counts)
            self.count_features_.append(f)
            cols.append(counts[c])

        if not cols:
            return np.empty((codes.shape[0], 0))
        return np.column_stack(cols)

    def _pinned_count_column(self, j, f, all_cats, pinned, pinned_counts):
        """One replay-refit count lookup: the donor's counts re-indexed onto
        this fit's codes.

        ``j``/``f`` are this fit's categorical position and original column
        index, ``all_cats`` the per-column category arrays in code order.
        ``pinned`` is the donor's
        ``count_features_`` (selection order matches ``cat_counts_`` order);
        ``pinned_counts`` is ``(donor_cat_maps, donor_cat_counts)``. Both
        fits share the same ``cat_features_``, so position ``j`` addresses
        the same column on the donor. Categories the donor never saw read
        0.0, so ``transform`` -- which indexes by this fit's codes -- needs
        no change: predict-time counts equal the donor's.
        """
        donor_maps, donor_counts_list = pinned_counts
        donor_counts = np.asarray(donor_counts_list[pinned.index(f)],
                                  dtype=np.float64)
        dc_by_code = _remap_codes(all_cats[j], donor_maps[j], -1)
        return np.where(dc_by_code >= 0,
                        donor_counts[np.maximum(dc_by_code, 0)], 0.0)

    def _count_block(self, codes):
        """Gather the count columns for base-categorical ``codes``.

        Each selected categorical contributes its fit-time per-category row
        count; code -1 (a category unseen at fit) reads 0.0. ``codes`` holds
        the base categorical columns in ``cat_features_`` order, before any
        combo codes are appended. Empty (n, 0) when nothing was selected.
        """
        cols = []
        for k, f in enumerate(getattr(self, "count_features_", [])):
            code = codes[:, self.cat_features_.index(f)]
            counts = self.cat_counts_[k]
            cols.append(np.where(code >= 0, counts[np.maximum(code, 0)], 0.0))
        if not cols:
            return np.empty((codes.shape[0], 0))
        return np.column_stack(cols)

    def _combo_codes_for_transform(self, X, cat_ctx=None):
        """Reconstruct combination codes for transform from the stored maps.

        Models pickled before the pair-code change carry string-keyed maps; they
        stay on the legacy string-concat path so their predictions don't move.
        """
        if cat_ctx is None:
            cat_ctx = CatTransformCache()

        combo_codes = np.empty((X.shape[0], len(self.combo_pairs_)), dtype=np.int64)
        for k, (f_a, f_b) in enumerate(self.combo_pairs_):
            legacy = (self.combo_maps_[k]
                      and isinstance(next(iter(self.combo_maps_[k])), str))
            c, cats = (cat_ctx.combo_str(X, f_a, f_b) if legacy
                       else cat_ctx.combo(X, f_a, f_b))
            combo_codes[:, k] = _remap_codes(cats, self.combo_maps_[k], -1)[c]

        return combo_codes

    # ---- fit / transform -----------------------------------------------------

    def fit_transform(self, X, encode_targets, cat_features, sample_weight=None,
                      binner=None, cat_ctx=None):
        """Fit on ``X`` and return the binned matrix.

        ``encode_targets`` is the list of T 1-D arrays used for ordered TS.

        ``sample_weight`` (mean-1 normalized, ``None`` == uniform) is forwarded
        to the ordered-target encoder and the binner, so zero-weight rows shape
        neither the categorical statistics nor the bin borders. ``None`` is the
        unweighted path, bit-identical to before this argument existed.

        ``binner`` (internal, for the replay refit) adopts an already-fitted
        ``Binner`` instead of fitting new borders. Everything upstream --
        categories, gdiff group means, ordered target statistics -- is still fit
        on ``X``, so the returned matrix carries proper fit-time (non-leaky)
        encodings; only the bin borders are held still, because the replayed
        split thresholds are bin INDICES into them. ``None`` fits a binner as
        before, bit-identically.
        """
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        num, codes = self._split_columns_fit(X, cat_features, cat_ctx,
                                             sample_weight)

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

        # Block order is [numeric | count | cross | per-target TS]. Only true
        # numeric columns carry an ordinal meaning a linear-leaf model can
        # use, so mark the raw numerics and the cross columns for the
        # booster's linear-term selection; the count columns (a per-category
        # lookup, not a measurement) and the TS blocks are excluded.
        self.is_numeric_binned_ = np.zeros(feat.shape[1], dtype=bool)
        n_num = len(self.num_features_)
        n_nocross = n_num + len(self.count_features_)
        self.is_numeric_binned_[:n_num] = True
        self.is_numeric_binned_[n_nocross:n_nocross + cross.shape[1]] = True

        if binner is None:
            self.binner_ = Binner(self.max_bins)
            X_binned = self.binner_.fit_transform(feat, sample_weight)
        else:
            self.binner_ = binner
            X_binned = binner.transform(feat)

        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X, cat_ctx=None):
        """Apply the fitted binning and categorical encoding to new data.

        ``cat_ctx`` (internal) shares the per-column canonical factorizations
        across the members of a bagged ensemble -- see CatTransformCache.
        """
        if cat_ctx is None:
            cat_ctx = CatTransformCache()
            base_direct = True
        else:
            base_direct = False

        # Hoisted above the numeric block: the same fit-time-code lookup
        # the encoder block below needs (bit-identical: same inputs, same
        # function). The count block reads its columns from it too.
        tf_codes = None
        if self.cat_features_:
            tf_codes = self._codes_for_transform(
                X, cat_ctx, allow_direct=base_direct)

        num = self._numeric_block(X, cat_ctx)
        if getattr(self, "count_features_", []):
            count_block = self._count_block(tf_codes)
            if count_block.shape[1]:
                num = (np.hstack([num, count_block]) if num.shape[1]
                       else count_block)
        cross = self._cross_block(X, cat_ctx, num=num)
        if cross.shape[1]:
            num = np.hstack([num, cross]) if num.shape[1] else cross

        encoded_blocks = []
        if self.cat_features_:
            codes = tf_codes
            if self.combo_pairs_:
                combo_codes = self._combo_codes_for_transform(X, cat_ctx)
                codes = np.hstack([codes, combo_codes])
            for enc in self.encoders_:
                encoded_blocks.append(enc.transform(codes))

        feat = self._stack(num, encoded_blocks)
        return self.binner_.transform(feat)

    @classmethod
    def from_base_with_cross(cls, base, cross_pairs, X, sample_weight=None,
                             cat_ctx=None):
        """Refit ``base``'s configuration with ``cross_pairs`` added, cheaply.

        Every fit artifact is per-column -- category maps, TS encodings,
        quantile borders, bin indices -- and appending cross columns leaves the
        base columns' inputs untouched, so base results are shared by reference
        and only the cross columns are computed here. Bit-identical to the
        from-scratch fit with the same ``cross_pairs``; ``base`` must itself
        have no cross features.

        Returns ``(prep, cross_binner, cross_binned)``: the fitted augmented
        preprocessor, the binner covering only the cross columns (for binning
        eval-set cross blocks), and the binned cross block for ``X``'s rows.
        The caller splices ``cross_binned`` into the base binned matrix at
        column offset ``base.n_numeric_block_`` -- stacked column order is
        [numeric | count | cross | TS blocks].
        """
        if base.cross_pairs:
            raise ValueError("base preprocessor already has cross features")

        prep = cls(base.max_bins, base.cat_smoothing, base.random_state,
                   base.cat_n_permutations, base.cat_combinations, cross_pairs)
        prep.cat_features_ = base.cat_features_
        prep.num_features_ = base.num_features_
        prep.count_features_ = base.count_features_
        prep.cat_counts_ = base.cat_counts_
        prep.n_numeric_block_ = base.n_numeric_block_
        prep.cat_count_features = getattr(base, "cat_count_features", False)
        prep.cat_maps_ = base.cat_maps_
        prep.combo_pairs_ = base.combo_pairs_
        prep.combo_maps_ = base.combo_maps_
        prep.encoders_ = base.encoders_

        if cat_ctx is None:
            cat_ctx = CatTransformCache()
        prep._fit_gdiff(X, sample_weight, cat_ctx)
        cross = prep._cross_block(X, cat_ctx)
        cross_binner = Binner(base.max_bins).fit(cross, sample_weight)

        # Splice the cross borders in between the base numeric block
        # (raw numerics plus count columns) and the TS block.
        nb = getattr(base, "n_numeric_block_", len(base.num_features_))
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
        n_num = len(base.num_features_)
        prep.is_numeric_binned_[:n_num] = True
        prep.is_numeric_binned_[nb:nb + cross.shape[1]] = True
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

        Block order is [numeric | count | cross | per-target (cat + combo)].
        Count columns map to their categorical's original index; cross and
        combo columns map to the lower-indexed feature of their pair, so their
        split gains fold into the right importance bucket.
        """
        combo_orig = [min(i, j) for i, j in self.combo_pairs_]
        fmap = list(self.num_features_)
        fmap.extend(getattr(self, "count_features_", []))

        # gdiff recenters its numeric parent i, so its gain belongs there;
        # diff/prod keep the established min(i, j) convention.
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
