"""Oblivious (symmetric) decision trees, numba-accelerated.

An oblivious tree of depth D uses the SAME (feature, bin-threshold) split at
every node of a given level. A row's leaf is therefore just a D-bit number, one
bit per level: bit_d = 1 if X[feature_d] > threshold_d else 0. This makes:

  * prediction a handful of comparisons + an array lookup (very fast), and
  * the model strongly regularized (only D splits per tree, shared across the
    whole level), which is a big part of why the defaults don't overfit.

We grow level by level. At each level we build per-(feature, current-leaf, bin)
gradient/hessian histograms and pick the single split that maximizes the summed
XGBoost-style gain over all current leaves.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _build_histograms_into(X_binned, grad, hess, leaf, n_leaves, hg, hh):
    """Fill per-feature gradient/hessian histograms into pre-allocated buffers.

    hg, hh are caller-owned arrays of shape (n_features, max_leaves, max_bins),
    reused across every tree and level to avoid reallocating gigabytes over a
    long boosting run. We zero only the (n_leaves) slice we are about to write,
    then accumulate. Parallelized over features so each thread owns a disjoint
    slice -- no write races, no locks.
    """
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in prange(n_features):
        # Zero this feature's active region (only the leaves/bins we will use).
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]


@njit(cache=True, parallel=True)
def _best_split(hg, hh, n_bins_per_feature, l2, feat_mask, min_child_weight,
                n_leaves):
    """Find the (feature, threshold) with the highest total gain.

    hg/hh may be oversized buffers (shape max_leaves); `n_leaves` says how many
    leaf rows are actually active at this level, so we only read those.

    For a candidate threshold t, bins <= t go left and bins > t go right, the
    same way in every current leaf. Gain is summed across leaves. Features with
    feat_mask[f] == 0 are skipped (column subsampling).

    A threshold is only legal if EVERY non-empty leaf keeps at least
    `min_child_weight` hessian mass on both sides of the split. This stops the
    tree from carving off near-empty leaves whose gradient statistics are pure
    noise -- the main cause of oblivious trees overfitting at higher depth.
    """
    n_features = hg.shape[0]
    max_bins = hg.shape[2]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        # Totals per leaf for this feature (same regardless of threshold).
        Gt = np.zeros(n_leaves)
        Ht = np.zeros(n_leaves)
        for l in range(n_leaves):
            for b in range(nb):
                Gt[l] += hg[f, l, b]
                Ht[l] += hh[f, l, b]

        GL = np.zeros(n_leaves)
        HL = np.zeros(n_leaves)
        best_g = -np.inf
        best_t = -1
        # Threshold t means "left = bins [0..t]". Last bin can't be a threshold.
        for t in range(nb - 1):
            # Pass 1: advance the running prefix sums for every leaf and check
            # the min_child_weight constraint. The sums must be advanced for all
            # leaves regardless of legality, since GL/HL carry into the next
            # threshold and an early exit would desync them.
            legal = True
            for l in range(n_leaves):
                GL[l] += hg[f, l, t]
                HL[l] += hh[f, l, t]
                if Ht[l] > 0.0:
                    if HL[l] < min_child_weight or (Ht[l] - HL[l]) < min_child_weight:
                        legal = False
            if not legal:
                continue
            # Pass 2: accumulate the summed gain for this legal threshold. A
            # non-empty leaf failing min_child_weight disqualifies the shared
            # split, giving the tree a data-driven depth limit.
            gain = 0.0
            for l in range(n_leaves):
                if Ht[l] > 0.0:
                    gr = Gt[l] - GL[l]
                    hr = Ht[l] - HL[l]
                    gain += (
                        GL[l] * GL[l] / (HL[l] + l2)
                        + gr * gr / (hr + l2)
                        - Gt[l] * Gt[l] / (Ht[l] + l2)
                    )
            if gain > best_g:
                best_g = gain
                best_t = t
        feat_gain[f] = best_g
        feat_thr[f] = best_t

    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    return best_f, feat_thr[best_f], best_gain


@njit(cache=True)
def _assign_leaves(X_binned, splits_feat, splits_thr):
    """Map every row to its leaf index given the chosen splits."""
    n = X_binned.shape[0]
    depth = splits_feat.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = 0
        for d in range(depth):
            f = splits_feat[d]
            t = splits_thr[d]
            bit = 1 if X_binned[i, f] > t else 0
            idx = idx * 2 + bit
        leaf[i] = idx
    return leaf


@njit(cache=True)
def _leaf_values(leaf, grad, hess, n_leaves, l2, lr):
    """Newton leaf values: value = -G / (H + l2), scaled by learning rate."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for i in range(leaf.shape[0]):
        G[leaf[i]] += grad[i]
        H[leaf[i]] += hess[i]
    values = np.zeros(n_leaves)
    for l in range(n_leaves):
        if H[l] > 0.0:
            values[l] = -lr * G[l] / (H[l] + l2)
    return values


@njit(cache=True)
def _predict_tree(X_binned, splits_feat, splits_thr, values):
    """Route each row to its leaf and return the leaf value for each row."""
    leaf = _assign_leaves(X_binned, splits_feat, splits_thr)
    out = np.empty(X_binned.shape[0], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


class ObliviousTree:
    """A single symmetric tree. Stores its splits and leaf values."""

    __slots__ = ("splits_feat", "splits_thr", "values", "gains", "depth")

    def __init__(self, splits_feat, splits_thr, values, gains=None):
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.values = values
        self.gains = gains if gains is not None else np.zeros(len(splits_feat))
        self.depth = len(splits_feat)

    def apply(self, X_binned):
        """Return the leaf index of each row."""
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.int64)
        return _assign_leaves(X_binned, self.splits_feat, self.splits_thr)

    def predict(self, X_binned):
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.float64)
        return _predict_tree(X_binned, self.splits_feat, self.splits_thr, self.values)


def build_oblivious_tree(X_binned, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None):
    """Grow one oblivious tree level by level and return an ObliviousTree.

    feature_mask: optional 0/1 array over features; 0 disables a feature for
    this tree (column subsampling). None means all features are eligible.
    min_child_weight: minimum hessian mass each side of a split must retain in
    every non-empty leaf. Stops the tree growing once no legal split remains,
    which prevents sparse-leaf overfitting at higher depth.
    hist_buffers: optional (hg, hh) arrays of shape (n_features, 2**max_depth,
    max_bins) reused across trees to avoid per-level allocation. If None, they
    are allocated here (convenient for one-off calls and tests).
    """
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
    if feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    if hist_buffers is None:
        max_leaves = 1 << max_depth
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
    else:
        hg, hh = hist_buffers
    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(X_binned.shape[0], dtype=np.int64)

    for d in range(max_depth):
        n_leaves = 1 << d
        _build_histograms_into(X_binned, grad, hess, leaf, n_leaves, hg, hh)
        f, t, gain = _best_split(hg, hh, n_bins_per_feature, l2, feature_mask,
                                 min_child_weight, n_leaves)
        if gain <= min_gain or t < 0:
            break
        splits_feat.append(f)
        splits_thr.append(t)
        splits_gain.append(gain)
        sf = np.array(splits_feat, dtype=np.int64)
        st = np.array(splits_thr, dtype=np.int64)
        leaf = _assign_leaves(X_binned, sf, st)

    sf = np.array(splits_feat, dtype=np.int64)
    st = np.array(splits_thr, dtype=np.int64)
    n_leaves = 1 << len(splits_feat)
    values = _leaf_values(leaf, grad, hess, n_leaves, l2, lr)
    return ObliviousTree(sf, st, values, np.array(splits_gain, dtype=np.float64))
