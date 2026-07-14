"""Turn a propagated SCM into a benchmark dataset (X, y, cats, meta).

Features are *views* of SCM nodes (rescaled / warped / discretized), so
categorical encoding and missingness destroy information realistically. The
target is emitted in node space, which makes the stored Bayes floors exact
generative lower bounds:
  regression      y = warp(node)*scale + offset + N(0, sigma); floor = sigma
  classification  p = softmax(-||phi - c_k||^2 / tau + log pi); y ~ Cat(p);
                  floor = sum-form Brier of p vs y (the harness convention)
When feature views quantize information the floor may be unattainable -- excess
metrics stay >= 0 and comparable across arms, which is all the suite needs.
"""
import numpy as np

from . import scm


def _zscore(v):
    std = v.std()
    return (v - v.mean()) / (std if std > 1e-12 else 1.0)


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------
def _select_informative(dag, recipe, rng):
    """Node indices exposed as informative features.

    Eligible: not the target, not a descendant of the target (keeps the Bayes
    floor honest), and causally tied to the target (an ancestor of it, or
    sharing an ancestor with it) -- the TabICL x<->y dependence constraint.
    """
    anc = dag.ancestor_masks()
    t = dag.target
    t_anc = anc[t]
    eligible = []
    for j in range(dag.n_nodes):
        if j == t or (anc[j] >> t) & 1:
            continue
        is_ancestor = (t_anc >> j) & 1
        shares = anc[j] & t_anc
        if is_ancestor or shares:
            eligible.append(j)
    take = min(recipe.d_informative, len(eligible))
    picked = list(rng.choice(eligible, size=take, replace=False)) if take else []
    return [int(j) for j in picked], anc


def _irrelevant_columns(n, count, rng):
    """Nuisance columns: half a correlated factor block, half iid noise."""
    cols = np.empty((n, count))
    kinds = []
    n_corr = count // 2
    if n_corr:
        k = max(1, min(3, n_corr))
        Z = rng.normal(size=(n, k))
        W = rng.normal(size=(k, n_corr))
        cols[:, :n_corr] = np.einsum("ik,kj->ij", Z, W) + 0.5 * rng.normal(
            size=(n, n_corr))
        kinds += ["irr_corr"] * n_corr
    for j in range(n_corr, count):
        u = rng.random()
        if u < 0.4:
            cols[:, j] = rng.normal(size=n)
        elif u < 0.7:
            cols[:, j] = rng.uniform(-1.7, 1.7, size=n)
        else:
            cols[:, j] = _zscore(rng.lognormal(0.0, 1.0, size=n))
        kinds.append("irr_noise")
    return cols, kinds


def _discretize(col, card, mode, rng):
    """Return integer codes in [0, realized_card) for one column."""
    if mode == "quantile_shuffled":
        probs = np.linspace(0, 1, card + 1)[1:-1]
        probs = np.clip(probs + rng.uniform(-0.4, 0.4, size=probs.size) / card, 0.02, 0.98)
        edges = np.unique(np.quantile(col, np.sort(probs)))
        codes = np.searchsorted(edges, col)
        perm = rng.permutation(edges.size + 1)
        codes = perm[codes]
    else:  # nearest_ref: non-monotone partition
        refs = np.sort(col[rng.choice(col.size, size=card, replace=False)])
        codes = np.abs(col[:, None] - refs[None, :]).argmin(axis=1)
    # compact to consecutive codes
    uniq, codes = np.unique(codes, return_inverse=True)
    return codes, uniq.size


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def _class_priors(n_classes, majority_frac, rng):
    if n_classes == 2:
        pi = np.array([majority_frac, 1.0 - majority_frac])
    else:
        lo, hi = 1e-4, 1.0
        for _ in range(60):  # geometric profile with p0 = majority_frac
            r = 0.5 * (lo + hi)
            p0 = (1 - r) / (1 - r ** n_classes) if r < 1 else 1.0 / n_classes
            if p0 > majority_frac:
                lo = r
            else:
                hi = r
        pi = r ** np.arange(n_classes) * (1 - r) / (1 - r ** n_classes)
    pi = np.clip(pi, 0.02, None)
    pi = pi / pi.sum()
    return pi[rng.permutation(n_classes)] if n_classes > 2 else pi


def _emit_classification(phi, recipe, rng):
    n, k = phi.shape[0], recipe.n_classes
    min_count = max(10, int(0.005 * n))
    pi = _class_priors(k, recipe.majority_frac, rng)
    tau = recipe.noise_level
    refs = phi[rng.choice(n, size=k, replace=False)]
    for _ in range(8):
        D = ((phi[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
        scale = np.median(D) + 1e-12
        logits = -D / (tau * scale) + np.log(pi)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
        counts = np.bincount(y, minlength=k)
        if counts.min() >= min_count:
            break
        tau *= 1.7                      # soften -> more class mixing
        pi = 0.7 * pi + 0.3 / k         # and drift priors toward uniform
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    bayes_brier = float(((p - onehot) ** 2).sum(axis=1).mean())
    degenerate = bool(counts.min() < min_count)
    return y.astype(np.int64), bayes_brier, degenerate, float(counts.max() / n)


def _saturated_target(cols, cat_codes, cat_idx, recipe, rng):
    """kr-vs-kp analog: y is a deterministic CELL RULE over 2-4 final feature
    columns -- a cat-cross lookup when categoricals exist, else axis-aligned
    threshold cells (exactly representable by an oblivious tree). The baseline
    must sit near the ceiling here, so added structure can only inject variance.

    Returns (y, meta_updates, degenerate).
    """
    n, k = cols.shape[0], recipe.n_classes

    def _cells():
        if len(cat_idx) >= 2:
            by_card = sorted(cat_idx, key=lambda j: int(cat_codes[j].max()) + 1)
            picked, prod = [], 1
            for j in by_card:
                card = int(cat_codes[j].max()) + 1
                if len(picked) == 4 or prod * card > 512:
                    break
                picked.append(j)
                prod *= card
            if prod >= max(k, 4) and len(picked) >= 2:
                cell = np.zeros(n, dtype=np.int64)
                for j in picked:
                    cell = cell * (int(cat_codes[j].max()) + 1) + cat_codes[j]
                return cell, prod, [int(j) for j in picked], "cat_cross"
        numeric = [j for j in range(cols.shape[1]) if j not in cat_codes]
        n_bits = 3 if k > 4 or len(numeric) >= 3 else 2
        n_bits = min(n_bits, len(numeric))
        if 2 ** n_bits < k or n_bits == 0:
            return None, 0, [], "none"
        picked = [int(j) for j in rng.choice(numeric, size=n_bits, replace=False)]
        cell = np.zeros(n, dtype=np.int64)
        for bit, j in enumerate(picked):
            thr = np.quantile(cols[:, j], rng.uniform(0.35, 0.65))
            cell |= (cols[:, j] > thr).astype(np.int64) << bit
        return cell, 2 ** n_bits, picked, "axis_cells"

    cell, n_cells, rule_cols, rule_kind = _cells()
    if cell is None:
        return None, {}, True

    if recipe.task == "regression":
        levels = rng.normal(size=n_cells)
        f = _zscore(levels[cell])
        y_scale = float(np.exp(rng.uniform(np.log(0.5), np.log(200.0))))
        y_clean = f * y_scale + float(rng.normal() * y_scale)
        sigma = recipe.noise_level * y_scale
        y = y_clean + rng.normal(0.0, sigma, size=n)
        meta = {"noise_sigma": round(float(sigma), 6), "y_scale": round(y_scale, 4),
                "warp": "cells", "bayes_brier": None, "imbalance": 0.0,
                "rule_kind": rule_kind, "rule_cols": rule_cols}
        return y, meta, False

    min_count = max(10, int(0.005 * n))
    masses = np.bincount(cell, minlength=n_cells)
    order = np.argsort(-masses)                      # heaviest cells first
    for _ in range(8):
        classes = rng.permutation(k)
        cell_class = np.empty(n_cells, dtype=np.int64)
        for rank, c in enumerate(order):             # round-robin by mass
            cell_class[c] = classes[rank % k]
        y = cell_class[cell]
        counts = np.bincount(y, minlength=k)
        if counts.min() >= min_count:
            break
    meta = {"bayes_brier": 0.0, "noise_sigma": None,
            "imbalance": round(float(counts.max() / n), 4),
            "rule_kind": rule_kind, "rule_cols": rule_cols}
    return y.astype(np.int64), meta, bool(counts.min() < min_count)


_WARPS = ("identity", "sinh", "expskew", "cube")


def _emit_regression(phi, recipe, rng):
    f = _zscore(phi[:, 0])
    warp = _WARPS[int(rng.choice(len(_WARPS), p=[0.5, 0.2, 0.2, 0.1]))]
    if warp == "sinh":
        f = np.sinh(rng.uniform(0.5, 1.2) * f)
    elif warp == "expskew":
        f = np.exp(np.clip(rng.uniform(0.6, 1.1) * f, -6, 6)) * (
            1.0 if rng.random() < 0.5 else -1.0)
    elif warp == "cube":
        f = f ** 3
    f = _zscore(f)
    y_scale = float(np.exp(rng.uniform(np.log(0.5), np.log(200.0))))
    offset = float(rng.normal() * y_scale)
    y_clean = f * y_scale + offset
    sigma = recipe.noise_level * y_scale
    y = y_clean + rng.normal(0.0, sigma, size=f.size)
    return y, float(sigma), y_scale, warp


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def emit_dataset(recipe, seed_seq):
    """Build one dataset. Returns (X, y, cat_idx_or_None, task, meta)."""
    ss_dag, ss_nodes, ss_feat, ss_irr, ss_cat, ss_target, ss_miss = seed_seq.spawn(7)
    dag = scm.build_dag(recipe.n_nodes, recipe.interaction_depth,
                        recipe.max_in_degree, np.random.default_rng(ss_dag))
    M, node_kind = scm.propagate(dag, recipe, recipe.n, ss_nodes)

    feat_rng = np.random.default_rng(ss_feat)
    informative, anc = _select_informative(dag, recipe, feat_rng)
    d_inf_real = len(informative)
    n_irr = recipe.d - d_inf_real
    irr, irr_kinds = _irrelevant_columns(recipe.n, n_irr, np.random.default_rng(ss_irr))

    cols = np.empty((recipe.n, recipe.d))
    kinds = []
    for j, node in enumerate(informative):
        cols[:, j] = M[:, node]
        kinds.append("informative")
    if n_irr:
        cols[:, d_inf_real:] = irr
        kinds += irr_kinds

    order = feat_rng.permutation(recipe.d)
    cols = cols[:, order]
    kinds = [kinds[i] for i in order]

    # per-column affine + heavy-tail warps (views only; target already fixed
    # in node space, so these never touch the Bayes floor)
    n_heavy = int(round(recipe.heavy_tail_cols * recipe.d))
    heavy_ix = set(feat_rng.choice(recipe.d, size=n_heavy, replace=False).tolist()
                   ) if n_heavy else set()
    for j in range(recipe.d):
        v = _zscore(cols[:, j])
        if j in heavy_ix:
            v = np.exp(np.clip(v * feat_rng.uniform(0.7, 1.3), -8, 8))
            v = _zscore(v)
        scale = np.exp(feat_rng.uniform(np.log(0.5), np.log(200.0)))
        cols[:, j] = v * scale + feat_rng.normal() * scale

    # categoricals BEFORE the target: the saturated cell-rule builds on the
    # final encoded views. Streams are independent, so this ordering leaves the
    # non-saturated path draw-for-draw unchanged.
    cat_rng = np.random.default_rng(ss_cat)
    cat_idx = sorted(cat_rng.choice(recipe.d, size=min(recipe.n_cat, recipe.d),
                                    replace=False).tolist()) if recipe.n_cat else []
    cat_codes, cards = {}, []
    for j in cat_idx:
        card = int(np.clip(round(cat_rng.lognormal(recipe.log_card_mu, 0.3)),
                           2, recipe.max_cardinality))
        codes, realized = _discretize(cols[:, j], card, recipe.cat_encode_mode, cat_rng)
        cat_codes[j] = codes
        cards.append(realized)

    # target: node-space readout normally; deterministic cell rule when saturated
    target_rng = np.random.default_rng(ss_target)
    t_parents = dag.parents[dag.target]
    r = int(target_rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
    phi_nodes = [dag.target] + list(t_parents[: r - 1])
    phi = np.column_stack([_zscore(M[:, j]) for j in phi_nodes])

    meta = {
        "gen_version": recipe.version, "recipe_id": recipe.id, "task": recipe.task,
        "n": recipe.n, "d": recipe.d, "d_informative": d_inf_real,
        "irrelevant_fraction": round(1.0 - d_inf_real / recipe.d, 4),
        "interaction_depth": recipe.interaction_depth,
        "root_dist": recipe.root_dist, "saturated": recipe.saturated,
        "noise_level": round(recipe.noise_level, 5),
        "heavy_tail_cols": round(recipe.heavy_tail_cols, 4),
        "n_classes": recipe.n_classes,
        "cat_encode_mode": recipe.cat_encode_mode,
        "degenerate": False,
    }
    # dominant function class on the target's ancestry (the mechanism label)
    t_anc_nodes = [j for j in range(dag.n_nodes) if (anc[dag.target] >> j) & 1]
    anc_kinds = [node_kind[j] for j in t_anc_nodes if node_kind[j] != "root"]
    anc_kinds.append(node_kind[dag.target])
    vals, cnt = np.unique(anc_kinds, return_counts=True)
    meta["func_dominant"] = str(vals[cnt.argmax()])

    if recipe.saturated:
        y, sat_meta, degenerate = _saturated_target(
            cols, cat_codes, cat_idx, recipe, target_rng)
        if degenerate or y is None:
            meta["degenerate"] = True
            y = (np.zeros(recipe.n) if recipe.task == "regression"
                 else np.zeros(recipe.n, dtype=np.int64)) if y is None else y
        meta.update(sat_meta or dict(bayes_brier=None, noise_sigma=None,
                                     imbalance=0.0))
        meta["func_dominant"] = "cellrule"
    elif recipe.task == "regression":
        y, sigma, y_scale, warp = _emit_regression(phi, recipe, target_rng)
        meta.update(noise_sigma=round(sigma, 6), y_scale=round(y_scale, 4),
                    warp=warp, bayes_brier=None, imbalance=0.0)
        if y.std() < 1e-9:
            meta["degenerate"] = True
    else:
        y, bayes_brier, degenerate, maj_real = _emit_classification(
            phi, recipe, target_rng)
        meta.update(bayes_brier=round(bayes_brier, 6), noise_sigma=None,
                    imbalance=round(maj_real, 4), degenerate=degenerate)

    meta.update(n_cat=len(cat_idx),
                cat_fraction=round(len(cat_idx) / recipe.d, 4),
                max_cardinality=int(max(cards)) if cards else 0,
                card_gmean=round(float(np.exp(np.mean(np.log(cards)))), 2)
                if cards else 0.0)

    # missingness (MCAR, per-column rates around the calibrated fraction)
    miss_rng = np.random.default_rng(ss_miss)
    miss_mask = np.zeros((recipe.n, recipe.d), dtype=bool)
    if recipe.missing_fraction > 0:
        for j in range(recipe.d):
            rate = min(0.5, recipe.missing_fraction * miss_rng.uniform(0.3, 1.7))
            miss_mask[:, j] = miss_rng.random(recipe.n) < rate
    meta["missing_fraction"] = round(float(miss_mask.mean()), 4)

    # assembly
    if cat_idx:
        X = np.empty((recipe.n, recipe.d), dtype=object)
        for j in range(recipe.d):
            if j in cat_codes:
                labels = np.array([f"c{c:02d}" for c in range(int(cat_codes[j].max()) + 1)],
                                  dtype=object)
                colv = labels[cat_codes[j]]
                colv[miss_mask[:, j]] = "__nan__"
                X[:, j] = colv
            else:
                v = cols[:, j].copy()
                v[miss_mask[:, j]] = np.nan
                X[:, j] = v
    else:
        X = cols.copy()
        X[miss_mask] = np.nan

    y = y.astype(np.float64) if recipe.task == "regression" else y
    return X, y, (cat_idx or None), recipe.task, meta
