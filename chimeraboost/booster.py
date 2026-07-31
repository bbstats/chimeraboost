"""The gradient boosting core: builds the full additive model.

Two boosters share the same machinery (FeaturePreprocessor, oblivious trees):
  * GradientBoosting     -> scalar output (regression, binary classification)
  * MulticlassBoosting   -> K simultaneous outputs (softmax multiclass)
"""

import time
from contextlib import contextmanager

import numpy as np

from .losses import LOSSES, MAE, MultiSoftmax, MultiQuantile, Quantile
from .preprocessing import FeaturePreprocessor, as_model_array
from .binning import _SERIAL_PREDICT_N
from .tree import (build_oblivious_tree, replay_oblivious_tree,
                   build_oblivious_tree_exact, alloc_exact_hist, _SMALL_N,
                   _leaf_quantiles_vec, _leaf_quantiles_vec_serial,
                   _leaf_quantiles_vec_w, _leaf_quantiles_vec_w_serial,
                   _project_pinball, _add_leaf_values,
                   _loo_leaf_step, _leaf_values,
                   _leaf_values_vec, _linear_predict,
                   _predict_forest_rm, _predict_forest_rm_serial,
                   pack_forest, pack_forest_vec,
                   _predict_forest_vec_rm, _predict_forest_vec_rm_serial,
                   _predict_forest_linear,
                   _predict_forest_linear_rm, _predict_forest_linear_rm_serial,
                   pack_forest_linear, _shap_forest_linear,
                   _mvs_lambda_scan, _mvs_weights, _mvs_weights_serial)


def _uniform_to_none(w):
    """Collapse uniform (all-equal) weights to None: they are the unweighted
    case, and routing them through None keeps every weight-aware path (grad/hess,
    loss init, target encoder, binner, validation metric) bit-identical to
    sample_weight=None. None passes through unchanged."""
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    if w.size and np.all(w == w[0]):
        return None
    return w


def _run_callbacks(callbacks, iteration, train_loss, val_loss, model):
    """Invoke each fit callback for one boosting round. A callback is
    ``cb(iteration, train_loss, val_loss, model)``; returning True requests an
    early stop. Returns True if any callback asked to stop. ``callbacks`` may be
    a single callable or an iterable of them; None is a no-op."""
    if not callbacks:
        return False
    cbs = callbacks if isinstance(callbacks, (list, tuple)) else (callbacks,)
    stop = False
    for cb in cbs:
        if cb(iteration, train_loss, val_loss, model) is True:
            stop = True
    return stop


def _callbacks_need_train_loss(callbacks):
    """Whether any fit callback wants the per-round train loss. Internal
    callbacks (selection-race auditions) never read it and are tagged with
    ``_cb_needs_train_loss = False``; evaluating the loss on the full training
    set every round is the single biggest non-tree cost of an audition fit, so
    the fit loop skips it when nobody will look at the value. Untagged (user)
    callbacks keep receiving the train loss as documented."""
    if not callbacks:
        return False
    cbs = callbacks if isinstance(callbacks, (list, tuple)) else (callbacks,)
    return any(getattr(cb, "_cb_needs_train_loss", True) for cb in cbs)


def _factorials(n):
    """Factorials 0!..n! as a float array (Shapley coalition weights)."""
    f = np.empty(n + 1)
    f[0] = 1.0
    for i in range(1, n + 1):
        f[i] = f[i - 1] * i
    return f


# Below this many training rows, per-leaf linear models overfit (noisy small
# data has too little signal per leaf to support a stable slope), so linear
# leaves silently fall back to constant leaves. Matches the codebase's recurring
# "sub-~1k rows is the small-data danger zone" boundary (cf. _auto_min_child_weight,
# the max_bins sub-1k overfit). Validated: protects kc2 (~313 train) from a -4.6%
# Brier loss while keeping the wins on larger sets (sick/spambase/electricity).
LINEAR_LEAVES_MIN_SAMPLES = 1000

# Default number of training rows retained as the SHAP background distribution.
SHAP_BACKGROUND_SIZE = 200


@contextmanager
def _thread_limit(thread_count):
    """Apply an explicit ``thread_count`` for the duration of one fit or
    predict call, restoring the caller's numba thread setting on exit.

    ``numba.set_num_threads`` is process-global; without the restore, one
    model's ``thread_count=1`` would silently cap every later numba call in
    the process (other models' fits and predicts included). ``None`` / ``-1``
    leaves the ambient setting untouched entirely, so a user's own
    ``numba.set_num_threads`` (or ``NUMBA_NUM_THREADS``) is respected.
    Yields the effective thread count so callers can record it.
    """
    import numba
    if thread_count is None or int(thread_count) < 0:
        yield numba.get_num_threads()
        return
    prev = numba.get_num_threads()
    n = max(1, min(int(thread_count), numba.config.NUMBA_NUM_THREADS))
    if n == prev:
        # Already there: skip the switch entirely. A changed thread count is
        # NOT free -- the omp layer charges ~1 ms to re-team on the next
        # parallel region -- so processes that align the ambient setting
        # (NUMBA_NUM_THREADS / numba.set_num_threads) predict at full speed.
        yield n
        return
    numba.set_num_threads(n)
    try:
        yield n
    finally:
        numba.set_num_threads(prev)


def _eval_advance(Fv, tree, Xvb):
    """Add one tree's eval-set contribution to ``Fv`` in place (scalar path).
    Constant-leaf trees fuse the value gather into `_add_leaf_values` via the
    parallel leaf assignment; linear-leaf trees keep ``tree.predict`` -- their
    step lives in ``lin_coef``, not ``values``. Both arms are the same
    elementwise adds `Fv += tree.predict(Xvb)` performed."""
    if tree.lin_coef is None:
        _add_leaf_values(Fv.reshape(-1, 1), tree.values.reshape(-1, 1),
                         tree.apply(Xvb))
    else:
        np.add(Fv, tree.predict(Xvb), out=Fv)


def _auto_learning_rate(n_estimators, early_stopping):
    """Default learning rate when the user did not specify one.

    With early stopping, 0.1 (the field-standard default) lets early stopping
    pick the ensemble size; it converges in ~half the trees of a smaller rate
    with no measured accuracy cost, which speeds up both fit and predict.
    Otherwise the rate scales inversely with the iteration budget so short runs
    still cover enough ground.
    """
    if early_stopping:
        return 0.1
    return float(np.clip(20.0 / max(n_estimators, 1), 0.03, 0.2))


class _EarlyStopper:
    """Tracks the best validation score and signals when patience runs out."""

    def __init__(self, patience):
        self.patience = patience
        self.best_score = np.inf
        self.best_iter = 0

    def step(self, score, m):
        """Record the round-*m* score; return True if training should stop."""
        if not np.isfinite(score):
            # A NaN/inf score never compares below best, so without this guard
            # best_iter would silently freeze at 0 and the fitted model would
            # be truncated to a single tree.
            raise ValueError(
                f"Validation score at round {m} is {score!r} and cannot drive "
                "early stopping. Check eval_set y for NaN/inf or values "
                "outside the loss's domain (e.g. zeros with loss='Gamma'), "
                "and any custom eval_metric's return value.")
        if score < self.best_score - 1e-9:
            self.best_score, self.best_iter = score, m
            return False
        return bool(self.patience) and (m - self.best_iter >= self.patience)


class _BaseBooster:
    """Shared machinery for the scalar and multiclass boosters.

    Holds the common hyperparameters and the helpers both subclasses use:
    histogram-buffer allocation, column subsampling, row subsampling, feature
    preprocessing, and split-gain feature importances. Subclasses implement
    `fit` and `predict_raw`.
    """

    def __init__(self, n_estimators=500, learning_rate=None, depth=6,
                 l2_leaf_reg=1.0, max_bins=128, subsample=1.0,
                 colsample=1.0, cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None, min_child_weight=1.0,
                 thread_count=None, random_state=None, verbose=False,
                 ordered_boosting=False, cat_combinations=False,
                 leaf_estimation_iterations=1,
                 linear_leaves=False, linear_lambda=1.0, cross_pairs=None,
                 quantize_gradients=True, eval_metric=None,
                 replay_donor=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.max_bins = int(max_bins)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.cat_smoothing = float(cat_smoothing)
        self.cat_n_permutations = int(cat_n_permutations)
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = float(min_child_weight)
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = bool(ordered_boosting)
        self.cat_combinations = bool(cat_combinations)
        self.leaf_estimation_iterations = int(leaf_estimation_iterations)
        self.linear_leaves = bool(linear_leaves)
        self.linear_lambda = float(linear_lambda)
        self.cross_pairs = list(cross_pairs) if cross_pairs else []
        self.quantize_gradients = bool(quantize_gradients)
        self.eval_metric = eval_metric
        # Structure-transfer refit (see tree.replay_oblivious_tree): a
        # (trees, preprocessor) pair whose splits are replayed instead of
        # re-grown. None == ordinary fit, and every path below is unchanged.
        self.replay_donor = replay_donor

    def _alloc_hist_buffers(self, n_features, n_bins):
        """Allocate the reusable histogram buffer once per fit.

        Float path: (n_features, 2**depth, max_bins, 2); the last axis
        interleaves grad and hess so each scatter write hits one cache line.
        Quantized path: int64 (n_features, 2**depth, max_bins) — grad and
        hess ride packed in one cell (see tree._quantize_pack), halving the
        footprint. Reused for every tree and level (the kernel zeroes the
        active slice each call), avoiding thousands of reallocations over a
        long boosting run.
        """
        max_leaves = 1 << self.depth
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        if self.quantize_gradients:
            return np.zeros((n_features, max_leaves, max_bins),
                            dtype=np.int64)
        return np.zeros((n_features, max_leaves, max_bins, 2))

    def _build_centers_std(self, Xb, n_bins):
        """Per-(feature, bin) table of STANDARDIZED bin-center values for the
        optional linear-leaf models. Standardizing per feature over the training
        distribution makes the linear ridge penalty scale-fair across features.
        Non-numeric (target-encoded) columns get zeros (never used as linear
        terms); the NaN/missing bin keeps NaN (treated as 0 = mean downstream)."""
        n_features = Xb.shape[0]
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        centers_std = np.zeros((n_features, max_bins))
        is_num = self.prep_.is_numeric_binned_
        bc = self.prep_.binner_.bin_centers_
        for f in range(n_features):
            if not is_num[f]:
                continue
            c = bc[f]
            per_sample = c[Xb[f]]
            finite = per_sample[np.isfinite(per_sample)]
            if finite.size == 0:
                continue
            mu = float(finite.mean())
            sd = float(finite.std())
            if sd <= 0.0:
                sd = 1.0
            centers_std[f, :c.shape[0]] = (c - mu) / sd
        return centers_std

    def _feature_mask(self, n_cols, rng):
        """0/1 mask selecting a random subset of columns for one tree."""
        if self.colsample >= 1.0:
            return None
        k = max(1, int(round(self.colsample * n_cols)))
        mask = np.zeros(n_cols, dtype=np.int64)
        mask[rng.choice(n_cols, size=k, replace=False)] = 1
        return mask

    def _new_preprocessor(self):
        """Build a FeaturePreprocessor configured from this booster's params."""
        return FeaturePreprocessor(self.max_bins, self.cat_smoothing,
                                   self.random_state, self.cat_n_permutations,
                                   self.cat_combinations, self.cross_pairs)

    def fit(self, X, y, *args, **kwargs):
        """Fit under this model's thread limit (restored on exit)."""
        with _thread_limit(self.thread_count) as n:
            self.n_threads_ = n
            return self._fit_impl(X, y, *args, **kwargs)

    def _val_score(self, yv, Fv, wv):
        """One entry of the validation series (early stopping and the sklearn
        layer's selections take the min of this). Default: the training loss
        on raw scores. With a custom ``eval_metric``, the metric on transformed
        predictions instead — negated when the metric declares
        ``greater_is_better = True`` so the lower-is-better machinery is
        untouched (``valid_history_`` then holds the negated values)."""
        if self.eval_metric is None:
            return self.loss_.eval(yv, Fv, wv)
        pred = self.loss_.transform(Fv)
        # Two-arg call when there are no validation weights, so plain
        # ``lambda y, p: ...`` metrics work.
        val = float(self.eval_metric(yv, pred) if wv is None
                    else self.eval_metric(yv, pred, wv))
        return -val if getattr(self.eval_metric, "greater_is_better", False) \
            else val

    def _round_epilogue(self, m, F, y, w, Fv, yv, wv, stopper, callbacks,
                        cb_train_loss, advance_fv):
        """Per-round tail shared by the three boosters: train-history append,
        eval-set scoring with early stopping, the progress print, and the
        callback hook. ``advance_fv`` adds the new tree's contribution to
        ``Fv`` in place; it runs only when there is an eval set, so callers
        must not precompute it. Returns True when training should stop -- the
        caller breaks, which keeps the ``for/else`` no-break truncation
        exactly as before."""
        if self.verbose:
            self.train_history_.append(self.loss_.eval(y, F, w))

        if Fv is not None:
            advance_fv()
            val = self._val_score(yv, Fv, wv)   # wv=None -> unweighted
            self.valid_history_.append(val)
            if stopper.step(val, m):
                if self.verbose:
                    print(f"Early stop at {m} (best {stopper.best_iter})")
                self.trees_ = self.trees_[: stopper.best_iter + 1]
                return True

        if self.verbose and (m % max(1, self.n_estimators // 10) == 0):
            msg = f"[{m}] train {self.train_history_[-1]:.5f}"
            if Fv is not None:
                msg += f"  val {self.valid_history_[-1]:.5f}"
            print(msg)

        if callbacks:
            tl = self.train_history_[-1] if self.train_history_ \
                else (self.loss_.eval(y, F, w) if cb_train_loss else None)
            vl = self.valid_history_[-1] if self.valid_history_ else None
            if _run_callbacks(callbacks, m, tl, vl, self):
                return True
        return False

    def predict_raw(self, X, cat_ctx=None):
        """Predict under this model's thread limit (restored on exit).
        ``cat_ctx`` (internal) shares categorical factorizations across the
        members of a bagged ensemble -- see CatTransformCache."""
        with _thread_limit(self.thread_count):
            return self._predict_raw_impl(X, cat_ctx)

    def __getstate__(self):
        """Pickle without the lazily-built packed-forest caches: they are
        redundant with ``trees_`` (roughly doubling the payload) and are
        rebuilt on the first predict after load."""
        state = self.__dict__.copy()
        for cache in ("_forest_", "_forests_"):
            if cache in state:
                state[cache] = None
        return state

    def _prep_matrices(self, X, encode_targets, cat_features, eval_set,
                       prep_cache, sample_weight=None):
        """Fit ``self.prep_`` and return the feature-major binned train/eval
        matrices ``(Xb, Xvb)`` (``Xvb`` is None without an eval set).
        ``sample_weight`` (mean-1 normalized train weights, ``None`` == uniform)
        is forwarded to the encoder/binner so zero-weight rows shape neither.
        ``encode_targets`` is the list of TS-encoding targets: ``[y]`` for the
        scalar booster, the K one-hot columns for multiclass.

        ``prep_cache`` (internal) is a dict shared across the booster fits of
        one sklearn-level fit -- the selection auditions, the cross-augmented
        candidate, and the winner refit -- which all see identical
        ``(X, y, cat_features, eval_set)`` and identical prep parameters
        except ``cross_pairs``. Entries are keyed by ``cross_pairs``: a
        repeat fit reuses the cached preprocessor and matrices outright, and
        the cross-augmented fit builds on the no-cross base entry, computing
        only the cross columns (``FeaturePreprocessor.from_base_with_cross``;
        bit-identical either way because every prep artifact is per-column).
        Callers that cannot guarantee identical inputs must pass None.
        """
        key = tuple(self.cross_pairs) if self.cross_pairs else ()
        if prep_cache is not None and key in prep_cache:
            self.prep_, Xb, Xvb = prep_cache[key]
            if eval_set is not None and Xvb is None:
                Xv = as_model_array(eval_set[0], bool(cat_features))
                Xvb = np.ascontiguousarray(self.prep_.transform(Xv).T)
            return Xb, Xvb

        base = prep_cache.get(()) if (prep_cache is not None and key) else None
        if base is not None:
            base_prep, base_Xb, base_Xvb = base
            self.prep_, cross_binner, crossb = \
                FeaturePreprocessor.from_base_with_cross(
                    base_prep, list(self.cross_pairs), X, sample_weight)
            nb = len(self.prep_.num_features_)
            # Stacked column order is [numeric | cross | TS]; splice the new
            # cross rows into the feature-major base matrix (concatenate
            # returns a fresh C-contiguous array, as the kernels require).
            Xb = np.concatenate([base_Xb[:nb], crossb.T, base_Xb[nb:]], axis=0)
            Xvb = None
            if eval_set is not None:
                Xv = as_model_array(eval_set[0], bool(cat_features))
                if base_Xvb is not None:
                    crossvb = cross_binner.transform(
                        self.prep_._cross_block(Xv))
                    Xvb = np.concatenate(
                        [base_Xvb[:nb], crossvb.T, base_Xvb[nb:]], axis=0)
                else:
                    Xvb = np.ascontiguousarray(self.prep_.transform(Xv).T)
        else:
            self.prep_ = self._new_preprocessor()
            # Tree kernels consume a feature-major matrix; transpose once here.
            Xb = np.ascontiguousarray(
                self.prep_.fit_transform(
                    X, encode_targets, cat_features, sample_weight).T)
            Xvb = None
            if eval_set is not None:
                Xv = as_model_array(eval_set[0], bool(cat_features))
                Xvb = np.ascontiguousarray(self.prep_.transform(Xv).T)
        if prep_cache is not None:
            prep_cache[key] = (self.prep_, Xb, Xvb)
        return Xb, Xvb

    @staticmethod
    def _normalize_weights(sample_weight, n_samples):
        """Scale weights to mean 1 so the gradient magnitude matches the
        unweighted case. None passes through unchanged. Uniform weights collapse
        to None so the whole fit takes the exact unweighted path (see
        ``_uniform_to_none``)."""
        if sample_weight is None:
            return None
        w = np.asarray(sample_weight, dtype=np.float64)
        return _uniform_to_none(w * (n_samples / w.sum()))

    def _resolve_lr(self, eval_set):
        if self.learning_rate is not None:
            return float(self.learning_rate)
        es = self.early_stopping_rounds is not None and eval_set is not None
        return _auto_learning_rate(self.n_estimators, es)

    def _loo_update(self, tree, leaf, g, h):
        """Leave-one-out leaf step: each row's training update uses its leaf's
        grad/hess totals with its own contribution removed, which curbs the
        self-reinforcement of plain boosting. ``tree.values`` keeps the standard
        Newton values for inference; only the training scores use this. Rows
        subsampled out (g=h=0) reduce to the standard leaf value. `leaf` is the
        training assignment returned by build_oblivious_tree."""
        return _loo_leaf_step(leaf, g, h, tree.values.shape[0],
                              self.l2_leaf_reg, self.lr_)

    def _refine_leaf_values(self, tree, leaf, F, y, w):
        """Apply ``leaf_estimation_iterations - 1`` extra Newton steps to a
        tree's leaf values, recomputing grad/hess at the updated residuals after
        each step. Mutates ``tree.values`` in place. A no-op when
        ``leaf_estimation_iterations == 1``."""
        for _ in range(self.leaf_estimation_iterations - 1):
            F_tmp = F + tree.values[leaf]
            g2, h2 = self.loss_.grad_hess(y, F_tmp)
            if w is not None:
                g2, h2 = g2 * w, h2 * w
            n_lv = tree.values.shape[0]
            tree.values += _leaf_values(leaf, g2, h2, n_lv,
                                        self.l2_leaf_reg, self.lr_)

    def _mvs_threshold(self, abs_g, target):
        """MVS: find threshold λ s.t. sum(min(|g_i|/λ, 1)) = target.

        Sort once, then hand the cutoff search to `_mvs_lambda_scan`, which
        fuses the prefix/suffix/argmax pass into one early-exiting loop.
        Returns λ=0 to signal "use uniform fallback" (degenerate cases).

        The sort and its sum stay here in numpy deliberately: they are the two
        pairwise-summed operations in this path, and porting them would move λ
        in the last ulp -- see `_mvs_lambda_scan`'s docstring.
        """
        n = len(abs_g)
        if target >= n:
            return 0.0
        sorted_g = np.sort(abs_g)[::-1]  # descending
        total = sorted_g.sum()
        if total < 1e-12:
            return 0.0
        return _mvs_lambda_scan(sorted_g, total, target)

    def _mvs_weights_for(self, grad, prob_src, lam):
        """The shared 1/p weight vector both MVS callers build: importance
        weight where the row survives its draw, 0 where it does not.

        `prob_src` is the booster's `rng.random(n)` draw. It is made by the
        caller so the Generator stream keeps its golden-frozen position."""
        max_w = 1.0 / max(self.subsample, 1e-3)
        kernel = (_mvs_weights_serial if grad.shape[0] < _SMALL_N
                  else _mvs_weights)
        return kernel(grad, prob_src, lam, max_w)

    def _maybe_subsample(self, grad, hess, rng):
        """MVS (Minimum Variance Sampling): gradient-weighted row subsampling.

        Rows with larger |grad| are sampled with higher probability and
        reweighted by 1/p to keep the leaf gradient sum unbiased. Reduces
        tree-to-tree correlation while concentrating capacity on uncertain
        samples — CatBoost's approach. Falls back to uniform when subsample=1.
        """
        if self.subsample >= 1.0:
            return grad, hess
        n = grad.shape[0]
        target = self.subsample * n
        abs_g = np.abs(grad)
        lam = self._mvs_threshold(abs_g, target)
        if lam == 0.0:
            # degenerate or all rows selected: uniform fallback
            mask = rng.random(n) < self.subsample
            return np.where(mask, grad, 0.0), np.where(mask, hess, 0.0)
        # importance weight = 1/p; capped at 1/subsample to avoid blowup on
        # near-zero-gradient rows (whose effective contribution g_i/p_i = λ)
        w = self._mvs_weights_for(grad, rng.random(n), lam)
        return grad * w, hess * w

    def _mvs_row_weights(self, grad, rng):
        """Per-row MVS weights (1/p importance weights, 0 = dropped), or None
        when subsample >= 1. The vector-leaf multiclass round derives ONE
        row selection from its sketched gradient and applies it to the
        sketch AND the per-class grad/hess (leaf values must see the same
        rows the split search saw). Shares the threshold and the 1/p weight
        vector with `_maybe_subsample`; the scalar path differs only in
        applying that vector to grad/hess itself."""
        if self.subsample >= 1.0:
            return None
        n = grad.shape[0]
        target = self.subsample * n
        abs_g = np.abs(grad)
        lam = self._mvs_threshold(abs_g, target)
        if lam == 0.0:
            # degenerate or all rows selected: uniform fallback
            return (rng.random(n) < self.subsample).astype(np.float64)
        return self._mvs_weights_for(grad, rng.random(n), lam)

    @property
    def feature_importances_(self):
        """Total split gain per ORIGINAL input column, normalized to sum 1.

        Computed lazily from the RETAINED trees, so trees discarded by early
        stopping (built past the best iteration, then truncated) contribute
        nothing. The old running accumulator counted them -- with patience 50,
        up to 50 dead trees' gains skewed the ranking."""
        feats, gains = [], []
        for item in self.trees_:
            # scalar booster stores trees; multiclass stores rounds of K trees
            for tree in (item if isinstance(item, list) else (item,)):
                feats.append(tree.splits_feat)
                gains.append(tree.gains)
        if not feats:
            return np.zeros(self.prep_.n_input_features_)
        # One bincount over every split in iteration order: its sequential C
        # accumulation reproduces the old per-split Python loop's addition
        # order exactly, at none of its interpreter cost.
        fmap = self.prep_.feature_map_
        imp = np.bincount(fmap[np.concatenate(feats)],
                          weights=np.concatenate(gains),
                          minlength=self.prep_.n_input_features_)
        s = imp.sum()
        return imp / s if s > 0 else imp


class GradientBoosting(_BaseBooster):
    """Scalar booster: regression and binary classification."""

    def __init__(self, loss="RMSE", loss_kwargs=None, **kw):
        super().__init__(**kw)
        self.loss_name = loss
        self.loss_kwargs = loss_kwargs or {}

    def _fit_impl(self, X, y, cat_features=None, eval_set=None,
                  sample_weight=None, callbacks=None, prep_cache=None):
        """Fit the additive model. Optionally pass `cat_features` (column indices
        to target-encode) and `eval_set=(X_val, y_val)` for early stopping.
        `sample_weight` is a 1-D array of per-sample weights; None means uniform.
        Weights are normalized to mean 1 internally so the gradient scale stays
        comparable to the no-weight case. `prep_cache` is internal -- see
        `_prep_matrices`. (Called via the base ``fit``, which owns the thread
        limit.)"""
        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        w = self._normalize_weights(sample_weight, n_samples)

        # A non-string loss is a user objective instance implementing the
        # losses.py protocol (see losses.CustomObjective); used as-is.
        self.loss_ = (LOSSES[self.loss_name](**self.loss_kwargs)
                      if isinstance(self.loss_name, str) else self.loss_name)
        self.lr_ = self._resolve_lr(eval_set)

        donor_trees = None
        if self.replay_donor is not None:
            donor_trees, donor_prep = self.replay_donor
            # Drop the reference once consumed: the donor holds a whole forest,
            # and this booster is the one that gets pickled.
            self.replay_donor = None
            # Refit every data-dependent statistic on these rows exactly as a
            # from-scratch refit would -- categories, gdiff group means, ordered
            # target statistics -- but ADOPT THE DONOR'S BINNER, because the
            # replayed split thresholds are bin indices into its borders.
            #
            # The donor's `transform` would be wrong here, not merely
            # approximate: it applies the INFERENCE-time target statistics, in
            # which a category's mean includes the label of the very row being
            # encoded. On the training matrix that is straight target leakage
            # (caught by tests/test_chimeraboost.py::test_ordered_ts_resists_
            # leakage, where it handed a pure-noise 2500-level column 54% of the
            # model's importance). Ordered TS exists precisely to prevent it.
            #
            # `cat_combinations` is pinned to what the donor actually built
            # rather than re-resolved at the new row count: it decides how many
            # TS columns exist, and the transferred splits address columns by
            # position.
            self.prep_ = FeaturePreprocessor(
                self.max_bins, self.cat_smoothing, self.random_state,
                self.cat_n_permutations, bool(donor_prep.combo_pairs_),
                self.cross_pairs)
            Xb = np.ascontiguousarray(
                self.prep_.fit_transform(X, [y], cat_features, w,
                                         binner=donor_prep.binner_).T)
            Xvb = (np.ascontiguousarray(self.prep_.transform(
                as_model_array(eval_set[0], bool(cat_features))).T)
                if eval_set is not None else None)
        else:
            Xb, Xvb = self._prep_matrices(X, [y], cat_features, eval_set,
                                          prep_cache, w)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(Xb.shape[0], n_bins)
        # Packed quantized grad/hess scratch, reused across trees (see
        # build_oblivious_tree's quantize path).
        qbuf = (np.empty(n_samples, dtype=np.int64)
                if self.quantize_gradients else None)
        # Keep a small sample of the (binned) training rows as the default SHAP
        # background -- the reference distribution interventional TreeSHAP
        # integrates over. Capped so it never bloats the pickled model.
        bg_n = min(n_samples, SHAP_BACKGROUND_SIZE)
        bg_idx = np.random.default_rng(self.random_state).choice(
            n_samples, bg_n, replace=False)
        self._shap_background_ = np.ascontiguousarray(Xb[:, bg_idx])

        yv = Fv = wv = None
        if eval_set is not None:
            yv = np.asarray(eval_set[1], dtype=np.float64)
            # eval_set may carry per-row validation weights as a 3rd element
            # (auto-split / OOB folds); uniform or absent -> unweighted metric.
            if len(eval_set) > 2:
                wv = _uniform_to_none(eval_set[2])

        self.init_ = self.loss_.init(y, w)
        F = np.full(n_samples, self.init_, dtype=np.float64)
        if yv is not None:
            Fv = np.full(len(yv), self.init_)

        adjusts_leaves = getattr(self.loss_, "adjusts_leaves", False)
        # Linear leaves are incompatible with the median/quantile leaf override
        # (adjusts_leaves) and with ordered boosting (a linear LOO step is not
        # implemented); they take over the leaf value otherwise. Below
        # LINEAR_LEAVES_MIN_SAMPLES rows they overfit, so fall back to constant.
        ll_active = (self.linear_leaves and not adjusts_leaves
                     and n_samples >= LINEAR_LEAVES_MIN_SAMPLES)
        self._centers_std_ = (self._build_centers_std(Xb, n_bins)
                              if ll_active else None)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self._forest_ = None   # packed-forest cache; built lazily on predict
        self.train_history_, self.valid_history_ = [], []
        stopper = _EarlyStopper(self.early_stopping_rounds)
        cb_train_loss = _callbacks_need_train_loss(callbacks)
        t0 = time.time()

        for m in range(self.n_estimators):
            grad, hess = self.loss_.grad_hess(y, F)
            if w is not None:
                grad, hess = grad * w, hess * w
            g, h = self._maybe_subsample(grad, hess, rng)
            fmask = self._feature_mask(Xb.shape[0], rng)
            # Fresh rounding seed per tree: decorrelates the stochastic
            # rounding noise across boosting rounds (drawn only on the
            # quantized path, so the float path's rng stream is unchanged).
            qseed = (int(rng.integers(1 << 63))
                     if self.quantize_gradients else 0)
            if donor_trees is not None and m < len(donor_trees):
                # Replay round: structure fixed, leaf values (and linear-leaf
                # coefficients) refit against this round's full-data gradients.
                tree, leaf = replay_oblivious_tree(
                    donor_trees[m], Xb, g, h, self.l2_leaf_reg, self.lr_,
                    linear_leaves=ll_active, centers_std=self._centers_std_,
                    linear_lambda=self.linear_lambda)
            else:
                tree, leaf = build_oblivious_tree(
                    Xb, g, h, n_bins, self.depth,
                    self.l2_leaf_reg, self.lr_,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=hist_buffers,
                    linear_leaves=ll_active,
                    centers_std=self._centers_std_,
                    is_numeric=self.prep_.is_numeric_binned_,
                    linear_lambda=self.linear_lambda,
                    quantize=self.quantize_gradients,
                    qbuf=qbuf, qseed=qseed)
            # A depth-0 tree found no legal split; the next round on the same
            # gradients would too, so stop rather than bank empty trees. Keep
            # the best prefix exactly as the other exits do -- without this,
            # trees grown after the validation optimum survived on this path.
            if tree.depth == 0:
                if Fv is not None and stopper.patience:
                    self.trees_ = self.trees_[: stopper.best_iter + 1]
                break
            if adjusts_leaves:
                self._correct_leaves(tree, leaf, y, F, w)
            self.trees_.append(tree)
            # Ordered boosting and leaf adjustment are mutually exclusive: the
            # former rewrites the training step, the latter the leaf value.
            if ll_active and tree.lin_coef is not None:
                # Linear-leaf path: training update is the leaf's local linear
                # model (no ordered boosting / no leaf_estimation refinement).
                F += _linear_predict(leaf, tree.lin_feats, tree.lin_coef,
                                     self._centers_std_, Xb)
            elif self.ordered_boosting and not adjusts_leaves:
                # Ordered boosting owns the training step (leaf-estimation Newton
                # refinement is for the plain path only).
                F += self._loo_update(tree, leaf, g, h)
            else:
                # Additional Newton steps refine the leaf values using the updated
                # residuals after each step. For constant-hessian losses (RMSE)
                # this converges in a few steps; for Logloss it reaches a better
                # per-leaf approximation than the single first-order step. Not
                # for MAE/Quantile: _correct_leaves already set the exact
                # minimizer (median/quantile), and a sign-gradient Newton step
                # on top would corrupt it -- the sklearn layer's "no effect"
                # warning for those losses is honest only with this guard.
                if not adjusts_leaves:
                    self._refine_leaf_values(tree, leaf, F, y, w)
                # Fused in-place add: F += tree.values[leaf] without the
                # n-length gather temporary. The (n, 1) views are C-contiguous,
                # so this reuses the vector boosters' compiled signature.
                _add_leaf_values(F.reshape(-1, 1), tree.values.reshape(-1, 1),
                                 leaf)
            if self._round_epilogue(
                    m, F, y, w, Fv, yv, wv, stopper, callbacks, cb_train_loss,
                    lambda: _eval_advance(Fv, tree, Xvb)):
                break
        else:
            # No break: the tree budget ran out before patience could fire.
            # Keep the best prefix exactly as the mid-training stop would
            # have. Callback stops are excluded on purpose -- the selection
            # auditions read feature_importances_ off callback-capped fits.
            if Fv is not None and stopper.patience:
                self.trees_ = self.trees_[: stopper.best_iter + 1]

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        return self

    def _correct_leaves(self, tree, leaf, y, F, sample_weight=None):
        """Override Newton leaf values with the loss-appropriate residual
        statistic (median for MAE, alpha-quantile for Quantile). The tree
        structure was chosen by the gradient; this fixes the step size.
        `leaf` is the training assignment from build_oblivious_tree.

        The library losses dispatch to the multi-quantile head's compiled
        leaf kernels with K=1: one parallel quickselect pass replaces a
        per-round stable argsort plus a Python-level np.quantile per leaf.
        The values are exactly unchanged -- `_quantile_slice` matches
        np.quantile bit for bit, `_weighted_quantile_slice` reproduces
        `losses._weighted_quantile`, `_leaf_row_index` groups rows in the
        same ascending order the argsort did (all three pinned by
        tests/test_quantile_head.py and the scalar oracle in
        tests/test_bitident_refactors.py). Custom adjusts-leaves losses keep
        the generic path below."""
        n_leaves = tree.values.shape[0]
        if type(self.loss_) in (MAE, Quantile):
            alpha = 0.5 if type(self.loss_) is MAE else self.loss_.alpha
            taus = np.array([alpha])
            F2 = F[:, None]        # (n, 1) view of contiguous F: C-contiguous
            small = leaf.shape[0] <= _SMALL_N
            if sample_weight is None:
                kern = (_leaf_quantiles_vec_serial if small
                        else _leaf_quantiles_vec)
                vals = kern(leaf, y, F2, taus, n_leaves, self.lr_)
            else:
                kern = (_leaf_quantiles_vec_w_serial if small
                        else _leaf_quantiles_vec_w)
                vals = kern(leaf, y, F2, sample_weight, taus, n_leaves,
                            self.lr_)
            tree.values = vals[:, 0]
            return
        residuals = y - F
        order = np.argsort(leaf, kind="stable")
        counts = np.bincount(leaf, minlength=n_leaves)
        stop = np.cumsum(counts)
        r_sorted = residuals[order]
        w_sorted = sample_weight[order] if sample_weight is not None else None
        for l in range(n_leaves):
            lo, hi = stop[l] - counts[l], stop[l]
            w = w_sorted[lo:hi] if w_sorted is not None else None
            tree.values[l] = self.lr_ * self.loss_.leaf_value(r_sorted[lo:hi], w)

    def _predict_raw_impl(self, X, cat_ctx=None):
        """Return raw additive scores (pre-link): the regression prediction, or
        the log-odds for binary classification."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        # The fused predict kernels consume the binner's row-major output
        # directly (no feature-major transpose; each sample's bins stay in
        # one or two cache lines for the whole forest walk).
        Xb = self.prep_.transform(X, cat_ctx)
        if not self.trees_:
            return np.full(Xb.shape[0], self.init_, dtype=np.float64)
        # Tiny batches take the serial kernel twins: the OpenMP fork/join
        # costs more than the whole walk there (both sides bit-identical).
        small = Xb.shape[0] <= _SERIAL_PREDICT_N
        if getattr(self, "_centers_std_", None) is not None:
            # Linear-leaf path: a dedicated fused kernel walks the whole forest
            # in one parallel pass (constant trees ride along as k=0).
            if self._forest_ is None:
                self._forest_ = pack_forest_linear(self.trees_, self.depth)
            feats, thrs, depths, lin_k, foff, lidx, coff, coef = self._forest_
            kernel = (_predict_forest_linear_rm_serial if small
                      else _predict_forest_linear_rm)
            return kernel(Xb, feats, thrs, depths, lin_k, foff, lidx, coff,
                          coef, self._centers_std_, self.init_)
        if self._forest_ is None:
            self._forest_ = pack_forest(self.trees_, self.depth)
        feats, thrs, depths, vals, voff = self._forest_
        kernel = _predict_forest_rm_serial if small else _predict_forest_rm
        return kernel(Xb, feats, thrs, depths, vals, voff, self.init_)

    def staged_predict_raw(self, X):
        """Yield the cumulative raw prediction after each tree (1..n_trees)."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        F = np.full(Xb.shape[1], self.init_, dtype=np.float64)
        for tree in self.trees_:
            F += tree.predict(Xb)
            yield F.copy()

    def shap_values(self, X, background=None, max_background=SHAP_BACKGROUND_SIZE,
                    random_state=0):
        """Exact interventional TreeSHAP, in raw-score (margin) space.

        Returns ``(phi, expected_value)`` where ``phi`` has shape
        ``(n_samples, n_input_features)`` and, for every row,
        ``phi.sum(axis=1) + expected_value == predict_raw(X)`` to floating-point
        tolerance (Shapley efficiency). Each ``phi[i, f]`` is feature f's signed
        additive contribution to the raw score of row i -- the regression target,
        or the binary log-odds. Linear-leaf slope terms are included exactly.

        ``background`` is the reference distribution SHAP integrates over
        (defaults to the training-data sample captured at fit); ``max_background``
        subsamples it for speed (cost is linear in the background size)."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        n_orig = self.prep_.n_input_features_
        if not self.trees_:
            return np.zeros((Xb.shape[1], n_orig)), float(self.init_)
        if background is None:
            Rb = self._shap_background_
        else:
            bg = as_model_array(background, bool(self.prep_.cat_features_))
            Rb = np.ascontiguousarray(self.prep_.transform(bg).T)
        if Rb.shape[1] > max_background:
            sel = np.random.default_rng(random_state).choice(
                Rb.shape[1], max_background, replace=False)
            Rb = np.ascontiguousarray(Rb[:, sel])
        cs = getattr(self, "_centers_std_", None)
        if cs is not None:
            # Linear-leaf model: predict caches the same linear-packed forest
            # in _forest_; build it once and share (constant-leaf models cache
            # the non-linear pack there instead, so those repack here).
            if self._forest_ is None:
                self._forest_ = pack_forest_linear(self.trees_, self.depth)
            feats, thrs, depths, lin_k, foff, lidx, coff, coef = self._forest_
        else:
            feats, thrs, depths, lin_k, foff, lidx, coff, coef = \
                pack_forest_linear(self.trees_, self.depth)
            cs = np.zeros((1, 1))   # unused: every tree is constant (k=0)
        fact = _factorials(self.depth)
        phi = _shap_forest_linear(Xb, Rb, feats, thrs, depths, lin_k, foff, lidx,
                                  coff, coef, cs, self.prep_.feature_map_, n_orig,
                                  fact)
        base = _predict_forest_linear(Rb, feats, thrs, depths, lin_k, foff, lidx,
                                      coff, coef, cs, self.init_)
        return phi, float(base.mean())


class MulticlassBoosting(_BaseBooster):
    """Softmax multiclass booster: one VECTOR-LEAF tree per round.

    Each round grows a single oblivious tree whose leaves hold K-vectors
    (one Newton value per class) on a shared structure. The split search
    runs on a 1-d sketch of the K gradient columns — a fresh Rademacher
    (±1) projection per round (benchmarks/A1_PLAN.md; SketchBoost's Random
    Projections, k=1): g_i = Σ_k r_k·grad_ik, and since r_k² = 1 the
    projected curvature rᵀdiag(H_i)r is exactly the row hessian sum, so
    (g, h) is a principled scalar Newton pair and the scalar split kernels
    (quantized path included) are reused verbatim. Models fitted before
    0.25.0 stored K trees per round (one per class); `predict_raw` keeps a
    fallback for those unpickled forests."""

    def _fit_impl(self, X, y, cat_features=None, eval_set=None,
                  sample_weight=None, callbacks=None, prep_cache=None):
        """Fit one vector-leaf tree per boosting round under softmax loss.
        Same `cat_features` / `eval_set` / `sample_weight` semantics as the
        scalar booster; `prep_cache` is internal -- see `_prep_matrices`.
        (Called via the base ``fit``, which owns the thread limit.)"""
        X = as_model_array(X, bool(cat_features))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        self.n_classes_ = K
        y_idx = np.searchsorted(self.classes_, y)
        Y = np.eye(K)[y_idx]                      # one-hot (n, K)
        n_samples = X.shape[0]
        w = self._normalize_weights(sample_weight, n_samples)

        self.loss_ = MultiSoftmax(K)
        self.lr_ = self._resolve_lr(eval_set)

        # One ordered-TS target per class (CatBoost-style per-class statistics).
        Xb, Xvb = self._prep_matrices(X, [Y[:, k] for k in range(K)],
                                      cat_features, eval_set, prep_cache, w)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(Xb.shape[0], n_bins)
        # Packed quantized grad/hess scratch, reused across every class tree.
        qbuf = (np.empty(n_samples, dtype=np.int64)
                if self.quantize_gradients else None)

        Yv = Fv = yv_idx = wv = None
        if eval_set is not None:
            yv_idx = np.searchsorted(self.classes_, np.asarray(eval_set[1]))
            Yv = np.eye(K)[yv_idx]
            if len(eval_set) > 2:
                wv = _uniform_to_none(eval_set[2])

        self.init_ = self.loss_.init(Y, w)         # (K,)
        F = np.tile(self.init_, (n_samples, 1))    # (n, K)
        if Yv is not None:
            Fv = np.tile(self.init_, (len(yv_idx), 1))

        coupling = (K - 1) / K   # softmax Hessian has rank K-1, not K
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []                           # flat list of vector trees
        self._forest_ = None    # packed vector-forest cache (lazy on predict)
        self._forests_ = None   # legacy per-class cache (pre-0.25.0 pickles)
        self.train_history_, self.valid_history_ = [], []
        stopper = _EarlyStopper(self.early_stopping_rounds)
        cb_train_loss = _callbacks_need_train_loss(callbacks)
        t0 = time.time()

        for m in range(self.n_estimators):
            grad, hess = self.loss_.grad_hess(Y, F)   # (n, K) each
            if w is not None:
                grad, hess = grad * w[:, None], hess * w[:, None]
            # 1-d Newton sketch for the split search: fresh CENTERED
            # Rademacher projection of the gradient columns. Softmax
            # gradient rows sum to zero, so the all-ones direction is the
            # null space — an uncentered all-equal draw (prob 2^(1-K)) gives
            # an identically-zero sketch and a spurious permanent stop
            # (caught in the A1 smoke; see A1_PLAN.md implementation log).
            # Centering removes that dead component; the rescale keeps
            # sum(r^2) = K so the projected-curvature mass stays on the
            # row-hessian-sum scale every round (min_child_weight and l2
            # semantics don't wobble with the draw).
            r = rng.integers(0, 2, size=K).astype(np.float64) * 2.0 - 1.0
            r -= r.mean()
            while not np.any(r):        # all-equal draw centered to zero
                r = rng.integers(0, 2, size=K).astype(np.float64) * 2.0 - 1.0
                r -= r.mean()
            r *= np.sqrt(K / (r @ r))
            g_s = grad @ r
            # Exact projected curvature r^T diag(H_i) r, coupled like the
            # per-class path's hessians.
            h_s = (hess @ (r * r)) * coupling
            # One MVS row selection per round, derived from the sketch and
            # applied to the per-class matrices too, so the leaf values see
            # exactly the rows (and importance weights) the split search saw.
            mw = self._mvs_row_weights(g_s, rng)
            if mw is not None:
                g_s, h_s = g_s * mw, h_s * mw
                grad, hess = grad * mw[:, None], hess * mw[:, None]
            fmask = self._feature_mask(Xb.shape[0], rng)
            qseed = (int(rng.integers(1 << 63))
                     if self.quantize_gradients else 0)
            tree, leaf = build_oblivious_tree(Xb, g_s, h_s, n_bins, self.depth,
                                              self.l2_leaf_reg, self.lr_,
                                              feature_mask=fmask,
                                              min_child_weight=self.min_child_weight,
                                              hist_buffers=hist_buffers,
                                              quantize=self.quantize_gradients,
                                              qbuf=qbuf, qseed=qseed)
            # No legal split on the sketched gradients: stop like the scalar
            # booster, keeping the best prefix exactly as the other exits do.
            if tree.depth == 0:
                if Fv is not None and stopper.patience:
                    self.trees_ = self.trees_[: stopper.best_iter + 1]
                break
            # Replace the sketch's scalar leaf values with the K-vector
            # Newton values on the shared partition (coupling applied
            # inside, per element — see _leaf_values_vec).
            tree.values = _leaf_values_vec(leaf, grad, hess, coupling,
                                           tree.values.shape[0],
                                           self.l2_leaf_reg, self.lr_)
            if self.ordered_boosting:
                # Per-class LOO on the shared leaf assignment.
                for k in range(K):
                    F[:, k] += _loo_leaf_step(
                        leaf, np.ascontiguousarray(grad[:, k]),
                        np.ascontiguousarray(hess[:, k]) * coupling,
                        tree.values.shape[0], self.l2_leaf_reg, self.lr_)
            else:
                _add_leaf_values(F, tree.values, leaf)
            self.trees_.append(tree)
            if self._round_epilogue(
                    m, F, Y, w, Fv, Yv, wv, stopper, callbacks, cb_train_loss,
                    lambda: _add_leaf_values(Fv, tree.values, tree.apply(Xvb))):
                break
        else:
            # No break: the tree budget ran out before patience could fire.
            # Keep the best prefix exactly as the mid-training stop would
            # have. Callback stops are excluded on purpose -- the selection
            # auditions read feature_importances_ off callback-capped fits.
            if Fv is not None and stopper.patience:
                self.trees_ = self.trees_[: stopper.best_iter + 1]

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        return self

    def _predict_raw_impl(self, X, cat_ctx=None):
        """Return the (n_samples, n_classes) matrix of raw per-class scores
        (pre-softmax)."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        # Row-major binned matrix straight from the binner (see the scalar
        # predict_raw).
        Xb = self.prep_.transform(X, cat_ctx)
        if not self.trees_:
            return np.tile(self.init_, (Xb.shape[0], 1))
        if isinstance(self.trees_[0], list):
            # Pre-0.25.0 pickle: rounds of K per-class trees.
            return self._predict_raw_ktrees(Xb)
        if getattr(self, "_forest_", None) is None:
            self._forest_ = pack_forest_vec(self.trees_, self.depth)
        feats, thrs, depths, vals, voff, K = self._forest_
        kernel = (_predict_forest_vec_rm_serial
                  if Xb.shape[0] <= _SERIAL_PREDICT_N
                  else _predict_forest_vec_rm)
        return kernel(Xb, feats, thrs, depths, vals, voff, K, self.init_)

    def _predict_raw_ktrees(self, Xb):
        """Legacy predict for models fitted before 0.25.0 (K trees per
        round): one packed forest per class, K walks over the same rows."""
        # Every column is fully overwritten below; no need to tile the init.
        F = np.empty((Xb.shape[0], self.n_classes_), dtype=np.float64)
        if getattr(self, "_forests_", None) is None:
            # One packed forest per class: class k's trees are round_trees[k]
            # across every round.
            self._forests_ = [
                pack_forest([rt[k] for rt in self.trees_], self.depth)
                for k in range(self.n_classes_)]
        kernel = (_predict_forest_rm_serial
                  if Xb.shape[0] <= _SERIAL_PREDICT_N else _predict_forest_rm)
        for k in range(self.n_classes_):
            feats, thrs, depths, vals, voff = self._forests_[k]
            F[:, k] = kernel(Xb, feats, thrs, depths, vals, voff,
                             self.init_[k])
        return F


# Rows sampled to estimate the gradient covariance for the split direction.
# The top eigenvector is a direction, not a precise quantity, and it does not
# need every row; capping keeps the Gram a few percent of round cost on large
# data instead of growing with n.
_GRAM_MAX_ROWS = 16384

# Which contrast each round uses under ``split_projection="rotate"``, cycled.
# Two rounds of location per round of spread: location carries most of the
# available gain, but a pure location diet is blind to spread entirely.
# Measured across location-only, spread-only, mixed and extreme
# heteroscedastic regimes at both K=3 and K=19; this beat uniform cycling,
# location-only, and every schedule that also spent rounds on skew. See
# benchmarks/QUANTILE_PLAN.md.
_ROTATE_PATTERN = (0, 0, 1)


def _fixed_contrasts(taus, n_basis=2):
    """Orthonormal split directions for ``split_projection`` of "sum" (the
    first only) and "rotate" (cycled by `_ROTATE_PATTERN`).

    Why polynomials in tau. Every row's pinball gradient is
    ``e(r_i) - taus``, where ``r_i`` is the row's PIT rank -- how many of its
    own K estimates the target falls above -- and ``e(r)`` is the step vector
    that turns on at r. A row's whole gradient is therefore one number, so
    projecting on a direction ``c`` is exactly scoring that rank by
    ``phi(r) = sum(c[r:])``. Taking c polynomial in tau of degree 0 and 1
    makes phi degree 1 and 2 in the rank:

    * degree 1 (c constant) -- location. Finds regions where the whole
      predictive distribution sits in the wrong place. This is the plain
      channel sum, and it is the ONLY direction "sum" ever uses.
    * degree 2 (c linear) -- spread. Finds regions whose width is wrong, which
      the location contrast cannot see at all: on a symmetric grid the tau and
      1-tau pushes are equal and opposite, so a correctly-centred but
      too-narrow region sums to exactly zero.

    A degree-3 (skew) contrast is available by raising ``n_basis`` but is not
    used by default: across every regime measured it took rounds away from
    location and spread without paying them back.

    Gram-Schmidt keeps the directions orthonormal on the actual grid (they are
    already near-orthogonal on a uniform one), and unit norm keeps the
    projected curvature at exactly 1 per row.
    """
    K = taus.shape[0]
    s = 2.0 * taus - 1.0                       # tau -> [-1, 1]
    raw = [np.ones(K), s, 1.5 * s * s - 0.5]   # Legendre P0, P1, P2
    basis = []
    for v in raw[:max(1, n_basis)]:
        v = v.astype(np.float64).copy()
        for b in basis:
            v -= (v @ b) * b
        nrm = float(np.sqrt(v @ v))
        if nrm > 1e-9:                # degenerate on very short grids
            basis.append(v / nrm)
    return basis


def _null_whitener(grad, taus):
    """Cholesky factor of the gradient covariance expected under NO x-signal.

    A row's gradient is ``e(r_i) - taus`` for its PIT rank r, so when the rank
    carries no information about x the covariance is the Brownian-bridge form
    ``min(p_j, p_k) - p_j p_k`` -- fixed by the realized marginal rates p
    alone, no fitting involved. Whitening by it is what stops the split
    direction being chosen by noise: the raw gradient variance in channel k is
    ``tau_k(1 - tau_k)``, four times larger at the median than at the 5%
    tail, so the largest-energy direction is the median one whether or not
    anything about the median depends on x.

    Estimated from the realized rates rather than from ``taus`` so that a
    globally miscalibrated model does not leak into the whitener.
    """
    p = np.clip(grad.mean(axis=0) + taus, 1e-6, 1.0 - 1e-6)
    N = np.minimum(p[:, None], p[None, :]) - np.outer(p, p)
    # Ridge for a strictly positive-definite factorization; scaled to the
    # matrix so it is negligible relative to the real structure.
    N += np.eye(p.shape[0]) * (1e-8 * float(np.trace(N)) / p.shape[0] + 1e-300)
    try:
        return np.linalg.cholesky(N)
    except np.linalg.LinAlgError:
        return None


def _gram_direction(grad, taus, basis, rng):
    """Best signal-to-noise split direction WITHIN the Legendre subspace.

    With unit hessians the exact summed-across-tau split gain is
    ``‖G_L‖²/(n_L+l2) + ‖G_R‖²/(n_R+l2) - ‖G_P‖²/(n_P+l2)``, which by Parseval
    equals the sum of projected gains over any orthonormal basis. Projecting
    onto one direction is therefore the rank-1 truncation of the exact gain
    rather than a different algorithm, and the question is only which
    direction to keep.

    Same subspace "rotate" cycles through -- location, spread, skew (see
    `_fixed_contrasts`) -- but weighted by what this round's gradients
    actually show instead of taken in turn. Maximizes ``c'Cc / c'Nc`` over the
    subspace, C being the centered gradient covariance and N the covariance
    expected with no x-signal at all (`_null_whitener`). That is a 3x3
    generalized eigenproblem, solved exactly.

    Three corrections separate a useful direction from a useless one, and each
    was measured rather than assumed:

    * Centering. A gradient component shared by every row contributes exactly
      zero gain (``n_L + n_R - n = 0``), so the raw second moment can point
      somewhere no split is able to exploit.
    * Whitening. Raw energy is dominated by per-row Bernoulli noise, largest
      at the median; without whitening this collapses onto the location
      contrast and inherits its blindness to spread.
    * The subspace. Whitening across all of R^K is ill-conditioned -- the
      Brownian-bridge null has eigenvalues falling like 1/k², so its inverse
      amplifies exactly the high-frequency directions that carry only sampling
      noise. Restricting to the low-order polynomials keeps the ratio
      well-posed. Measured: unrestricted whitening was four times worse than
      cycling on location-driven data.

    Returns None on a degenerate covariance, leaving the caller on the
    location contrast.
    """
    n = grad.shape[0]
    G = grad
    if n > _GRAM_MAX_ROWS:
        # With replacement: this estimates a direction only, and it avoids the
        # permutation cost of sampling without replacement on large n.
        G = grad[rng.integers(0, n, size=_GRAM_MAX_ROWS)]
    L = _null_whitener(G, taus)
    if L is None:
        return None
    B = np.stack(basis, axis=1)                  # (K, n_basis), orthonormal
    Gc = G - G.mean(axis=0)
    GB = Gc @ B
    Cb = GB.T @ GB                               # B'CB
    LB = L.T @ B
    Nb = LB.T @ LB                               # B'NB
    Nb += np.eye(Nb.shape[0]) * (1e-12 * float(np.trace(Nb)) + 1e-300)
    try:
        M = np.linalg.solve(Nb, Cb)
        vals, vecs = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    w = np.real(vecs[:, int(np.argmax(np.real(vals)))])
    c = B @ w
    nrm = float(np.sqrt(c @ c))
    if nrm < 1e-250:
        return None
    c = c / nrm                        # unit norm keeps min_child_weight exact
    # Pin the sign (largest-magnitude component positive) so repeated fits are
    # reproducible. Gain is even in the direction, so this changes no split.
    if c[np.argmax(np.abs(c))] < 0.0:
        c = -c
    return c


class MultiQuantileBoosting(_BaseBooster):
    """Multi-quantile booster: one VECTOR-LEAF tree per round, K quantiles.

    Each round grows a single oblivious tree whose leaves hold a K-vector, one
    entry per level in ``quantiles``. Two things make this more than K boosters
    stapled together:

    * The split search runs on a 1-d projection of the K pinball-gradient
      columns, so the scalar split kernels (quantized path included) are reused
      verbatim -- one histogram per round rather than K. `_gram_direction`
      explains why any single projection is the rank-1 truncation of the exact
      summed-across-tau gain rather than a different algorithm;
      `_fixed_contrasts` explains which directions are worth truncating to,
      and why the plain channel sum is the worst available choice.
    * Leaf values are the EXACT per-tau empirical quantile of the leaf's
      residuals (`tree._leaf_quantiles_vec`), not a Newton step -- the vector
      form of the override the scalar MAE/Quantile path already applies.

    Leaf channels are independent steps and nothing constrains them against one
    another during the fit. Ordering is imposed on DELIVERED predictions, per
    row, by monotone rearrangement: `_predict_raw_impl` and
    `staged_predict_raw` sort each row's K-vector before returning it, so
    predictions cannot cross at any stage. Rearrangement is free of charge in
    accuracy terms -- Chernozhukov, Fernandez-Val & Galichon (2010) show that
    sorting a crossing quantile curve never increases pinball loss at any
    level, for any row -- and it is exact per row rather than a bound that has
    to hold for every row at once.

    Two rules this class depends on, both learned the expensive way:

    * **Never sort ``F`` in place, or anything that feeds back into it.** An
      earlier design sorted each leaf vector as it was committed. Sorted values
      then re-entered the accumulated scores and self-reinforced, and the fit
      diverged to a measured pinball of 2.9e7 against an oracle of 0.35.
      Sorting delivered output has no such path back into training.
    * Nothing may assume a row of ``F`` is ordered. The pinball gradient is
      per-channel independent (`losses.MultiQuantile.grad_hess` compares only
      ``F[:, k]`` against ``y``), so crossing scores cost the fit nothing, but
      code that reads a row as if it were sorted is silently wrong -- see the
      scan in `tree._project_pinball`, which used to binary-search a PIT rank.

    The predecessor to all this was a training-time "narrowing budget" charged
    at the worst-case leaf each round. It was a valid bound and a bad one: one
    aggressively-narrowing leaf spent budget on behalf of every row, so it
    saturated within tens of rounds and froze the interval width, leaving bands
    2x to 10x wider than calibrated (benchmarks/LEAFTUNE_PLAN.md, P12).

    Known limitation, by construction: within one round, gradient signal
    orthogonal to the chosen direction cannot drive a split. The exact leaf
    refit still expresses that shape within whatever partition it is handed,
    so the failure mode is graceful, and the direction is re-measured every
    round -- once a mode is fit its energy drops and the next mode takes over
    without any schedule. ``exact_splits=True`` removes the limitation at K
    times the histogram cost.
    """

    def __init__(self, quantiles=None, split_projection="rotate",
                 exact_splits=False, **kw):
        super().__init__(**kw)
        taus = (np.arange(0.05, 0.9501, 0.05) if quantiles is None
                else np.asarray(quantiles, dtype=np.float64))
        self.quantiles = np.ascontiguousarray(taus, dtype=np.float64)
        self.split_projection = split_projection
        self.exact_splits = bool(exact_splits)

    def _split_direction(self, grad, m, rng):
        """Unit-norm direction to project this round's gradient columns onto.

        Unit norm matters beyond tidiness: the pinball hessian is 1, so the
        projected curvature is exactly 1 per row, which keeps
        ``min_child_weight`` and ``l2_leaf_reg`` meaning precisely what they
        mean on today's scalar Quantile path.

        "rotate" is the default because it measured best of the cheap arms --
        see benchmarks/QUANTILE_PLAN.md. Cycling guarantees each contrast a
        fixed share of rounds; picking the strongest one greedily ("gram")
        sounds better and is not, because on data whose only signal is spread
        it drifts onto the location contrast and never comes back."""
        basis = self._contrasts_
        if self.split_projection == "gram":
            v = _gram_direction(grad, self.quantiles, basis, rng)
            return basis[0] if v is None else v
        if self.split_projection == "rotate":
            i = _ROTATE_PATTERN[m % len(_ROTATE_PATTERN)]
            return basis[min(i, len(basis) - 1)]
        return basis[0]         # "sum": the literal channel sum

    def _fit_impl(self, X, y, cat_features=None, eval_set=None,
                  sample_weight=None, callbacks=None, prep_cache=None):
        """Fit one vector-leaf quantile tree per boosting round. Same
        `cat_features` / `eval_set` / `sample_weight` semantics as the scalar
        booster. (Called via the base ``fit``, which owns the thread limit.)"""
        X = as_model_array(X, bool(cat_features))
        y = np.ascontiguousarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        w = self._normalize_weights(sample_weight, n_samples)

        taus = self.quantiles
        K = taus.shape[0]
        self.loss_ = MultiQuantile(taus)
        self.lr_ = self._resolve_lr(eval_set)
        self._contrasts_ = _fixed_contrasts(taus)

        # One TS-encoding target: every quantile shares the same y, unlike
        # multiclass where each class gets its own one-hot column.
        Xb, Xvb = self._prep_matrices(X, [y], cat_features, eval_set,
                                      prep_cache, w)
        n_bins = self.prep_.n_bins_
        # The exact arm needs its own K-deep buffers and never touches the
        # scalar ones; allocate exactly one of the two.
        if self.exact_splits:
            hist_buffers, qbuf = None, None
            exact_hist = alloc_exact_hist(Xb.shape[0], self.depth, n_bins, K)
        else:
            exact_hist = None
            hist_buffers = self._alloc_hist_buffers(Xb.shape[0], n_bins)
            qbuf = (np.empty(n_samples, dtype=np.int64)
                    if self.quantize_gradients else None)

        yv = Fv = wv = None
        if eval_set is not None:
            yv = np.asarray(eval_set[1], dtype=np.float64)
            if len(eval_set) > 2:
                wv = _uniform_to_none(eval_set[2])

        self.init_ = self.loss_.init(y, w)          # (K,), sorted
        F = np.tile(self.init_, (n_samples, 1))     # (n, K)
        if yv is not None:
            Fv = np.tile(self.init_, (yv.shape[0], 1))

        # Scratch reused every round: the projected gradient, and a row-weight
        # vector that is all ones when unweighted (exactly 1.0, so that path
        # stays unchanged).
        g_buf = np.empty(n_samples)
        w_row = np.ones(n_samples) if w is None else w
        # Only two arms ever need the (n, K) gradient itself: "gram" measures
        # its covariance, and the exact split search scatters all K channels.
        need_grad = self.exact_splits or self.split_projection == "gram"

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self._forest_ = None
        self.train_history_, self.valid_history_ = [], []
        stopper = _EarlyStopper(self.early_stopping_rounds)
        cb_train_loss = _callbacks_need_train_loss(callbacks)
        # `_SMALL_N` is the scalar path's fork/join break-even, in units of
        # per-row work. The leaf refit does K quantile selections per leaf, so
        # the same amount of work is reached at K times fewer rows -- comparing
        # n alone would leave the parallel kernel unused on every ordinary
        # dataset. Both branches are bit-identical, so this only moves speed.
        small = n_samples * K <= _SMALL_N
        t0 = time.time()

        for m in range(self.n_estimators):
            # The default arms know their direction up front, so they read the
            # projection straight off each row's PIT rank and never build the
            # (n, K) gradient at all -- see `tree._project_pinball`.
            grad = None
            if need_grad:
                grad, _ = self.loss_.grad_hess(y, F)
                if w is not None:
                    grad = grad * w[:, None]
            direction = self._split_direction(grad, m, rng)
            if grad is not None and self.split_projection == "gram":
                g_s = grad @ direction
            else:
                _project_pinball(y, F, direction, float(direction @ taus),
                                 w_row, g_buf)
                g_s = g_buf
            # Unit-norm direction + unit hessian => projected curvature is
            # exactly 1 per row -- times the row's sample weight, exactly as
            # the projected gradient above is weighted. An unweighted hessian
            # here would score weighted gradient sums against row counts, so
            # the structure would optimize a different objective than the
            # leaf values are fit to (and min_child_weight would count rows
            # instead of weight mass). w_row already holds exactly these
            # values, and the build path never writes to its hessian, so the
            # hoisted buffer is safe to share across rounds.
            h_s = w_row
            # One MVS row selection per round, taken from the projection and
            # reused for the leaf refit, so leaf values see exactly the rows
            # the split search saw.
            mw = self._mvs_row_weights(g_s, rng)
            if mw is not None:
                g_s, h_s = g_s * mw, h_s * mw
            # Row weight seen by BOTH the split search and the leaf refit:
            # sample weights times any MVS importance weight. None == uniform,
            # which keeps the unweighted path on the fast kernels.
            rw = w if mw is None else (mw if w is None else w * mw)
            fmask = self._feature_mask(Xb.shape[0], rng)

            if self.exact_splits:
                tree, leaf = build_oblivious_tree_exact(
                    Xb, grad if mw is None else grad * mw[:, None],
                    n_bins, self.depth, self.l2_leaf_reg,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    row_weight=rw, hist_buffers=exact_hist)
            else:
                qseed = (int(rng.integers(1 << 63))
                         if self.quantize_gradients else 0)
                tree, leaf = build_oblivious_tree(
                    Xb, g_s, h_s, n_bins, self.depth, self.l2_leaf_reg,
                    self.lr_, feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=hist_buffers,
                    quantize=self.quantize_gradients, qbuf=qbuf, qseed=qseed)

            # No legal split: stop like the other boosters, keeping the best
            # prefix exactly as the other exits do.
            if tree.depth == 0:
                if Fv is not None and stopper.patience:
                    self.trees_ = self.trees_[: stopper.best_iter + 1]
                break

            # Replace the projection's scalar leaf values with the exact
            # per-tau residual quantiles on the shared partition. Each channel
            # is its own quantile of the same residuals; they are not
            # constrained against one another, which is what lets a leaf
            # narrow its interval as far as its rows warrant.
            n_lv = tree.values.shape[0]
            if rw is None:
                kern = (_leaf_quantiles_vec_serial if small
                        else _leaf_quantiles_vec)
                tree.values = kern(leaf, y, F, taus, n_lv, self.lr_)
            else:
                kern = (_leaf_quantiles_vec_w_serial if small
                        else _leaf_quantiles_vec_w)
                tree.values = kern(leaf, y, F, rw, taus, n_lv, self.lr_)

            _add_leaf_values(F, tree.values, leaf)
            self.trees_.append(tree)
            if self._round_epilogue(
                    m, F, y, w, Fv, yv, wv, stopper, callbacks, cb_train_loss,
                    lambda: _add_leaf_values(Fv, tree.values, tree.apply(Xvb))):
                break
        else:
            # No break: the tree budget ran out before patience could fire.
            if Fv is not None and stopper.patience:
                self.trees_ = self.trees_[: stopper.best_iter + 1]

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        return self

    def _val_score(self, yv, Fv, wv):
        """Score the eval set on REARRANGED scores, which is what a user
        receives. Sorting can only lower the pinball loss, so stopping on the
        raw accumulator would pick the best round for a prediction this class
        never returns. ``np.sort`` copies -- ``Fv`` itself must stay untouched,
        since the fit keeps adding to it."""
        return super()._val_score(yv, np.sort(Fv, axis=1), wv)

    def _predict_raw_impl(self, X, cat_ctx=None):
        """Return the (n_samples, n_quantiles) matrix of quantile estimates,
        each row sorted (see the class docstring: monotone rearrangement is
        where the non-crossing guarantee is enforced)."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        Xb = self.prep_.transform(X, cat_ctx)       # row-major
        if not self.trees_:
            return np.tile(self.init_, (Xb.shape[0], 1))
        if getattr(self, "_forest_", None) is None:
            self._forest_ = pack_forest_vec(self.trees_, self.depth)
        feats, thrs, depths, vals, voff, K = self._forest_
        kernel = (_predict_forest_vec_rm_serial
                  if Xb.shape[0] <= _SERIAL_PREDICT_N
                  else _predict_forest_vec_rm)
        return np.sort(
            kernel(Xb, feats, thrs, depths, vals, voff, K, self.init_), axis=1)

    def staged_predict_raw(self, X):
        """Yield the (n, K) quantile matrix after each successive tree.

        Trees are accumulated in the same order as `_predict_forest_vec_rm`,
        so the final stage equals ``predict_raw``. Each stage is rearranged on
        the way out and every one of them is non-crossing.

        ``np.sort`` returns a copy, which is load-bearing twice over: it is the
        per-stage copy the caller needs, and it keeps the running ``F`` in raw
        accumulated form. Sorting ``F`` in place would feed rearranged scores
        back into the next stage and reproduce the divergence recorded in the
        class docstring."""
        X = as_model_array(X, bool(self.prep_.cat_features_))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)   # feature-major
        F = np.tile(self.init_, (Xb.shape[1], 1))
        for tree in self.trees_:
            F += tree.values[tree.apply(Xb)]
            yield np.sort(F, axis=1)
