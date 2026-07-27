"""Metadata-balanced dataset weights for the public aggregate.

A benchmark suite is skewed by accident of what exists. The public suite runs
9 binary / 9 multiclass but only 4 regression, 11 medium-sized against 5 large,
and 7 high-cardinality against 4 low. Averaging over datasets unweighted lets
whichever stratum happens to be over-collected vote twice.

The fix is **raking** (iterative proportional fitting): rescale weights until
every level of every facet holds an equal share of the total, cycling across
facets until it converges, then normalise to 1.

Why not weight by the crossed cell `task x size x card`? Because 3x3x4 = 36
cells over 22 datasets leaves most of them empty, and a lone dataset in a rare
cell would carry an enormous weight. Raking matches all three marginals at once
without ever asking a cell to be non-empty.

Why cap the weights? Because uncapped raking on this suite hands freMTPL2freq
3.17x an average dataset's weight and internet_firewall 0.20x -- one dataset
deciding the regression column. The cap is the difference between an effective
sample size of 14.3 and 17.1 out of 22:

    equal weights          ESS 22.0 / 22    spread  1.0x
    raked, uncapped        ESS 14.3 / 22    spread 15.9x
    raked, capped [.5, 2]  ESS 17.1 / 22    spread  4.0x

Balancing is not free: error bars widen by sqrt(22/17.1), about 13%. That is the
honest price of not letting an over-represented stratum count twice, and
`effective_sample_size` exists so the price is always quoted.
"""
import math

# Facet thresholds. Deliberately round numbers, not tuned: `size` splits at the
# 100k floor the suite already requires and at 500k, `card` at the 10/100-level
# boundaries the HC work uses for "does this behave like an entity column".
SIZE_BINS = ((100_000, "small"), (500_000, "med"), (float("inf"), "large"))
CARD_BINS = ((1, "none"), (10, "low"), (100, "med"), (float("inf"), "high"))

WEIGHT_CAP = (0.5, 2.0)   # multiples of the equal weight
_ITERS = 200


def _bin(value, bins):
    for edge, label in bins:
        if value < edge:
            return label
    return bins[-1][1]


def facets_of(task, rows, maxcard):
    """The facet levels one dataset occupies."""
    return {"task": task,
            "size": _bin(rows, SIZE_BINS),
            "card": _bin(maxcard, CARD_BINS)}


def dataset_facets(specs, facet_meta):
    """{name: {facet: level}} from PUBLIC_DATASETS + PUBLIC_FACETS."""
    out = {}
    for name, spec in specs.items():
        meta = facet_meta.get(name)
        if meta is None:
            raise KeyError(f"no frozen facet metadata for {name!r} -- add it to "
                           "PUBLIC_FACETS or the weighting is not reproducible")
        out[name] = facets_of(spec["task"], meta["rows"], meta["maxcard"])
    return out


def raked_weights(facets, cap=WEIGHT_CAP, iters=_ITERS):
    """{name: weight} summing to 1, with every facet's levels at equal mass.

    `facets` is {name: {facet: level}}. Levels are counted from the data, so a
    facet level nobody occupies simply does not exist and cannot claim mass.
    """
    names = sorted(facets)
    if not names:
        return {}
    n = len(names)
    w = {k: 1.0 / n for k in names}
    facet_names = sorted({f for v in facets.values() for f in v})

    for _ in range(iters):
        for facet in facet_names:
            groups = {}
            for k in names:
                groups.setdefault(facets[k][facet], []).append(k)
            target = 1.0 / len(groups)
            for members in groups.values():
                mass = sum(w[k] for k in members)
                if mass > 0:
                    scale = target / mass
                    for k in members:
                        w[k] *= scale
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        if cap:
            w = _enforce_cap(w, cap, n)
    return w


def _enforce_cap(w, cap, n):
    """Clip to the cap and renormalise until BOTH hold simultaneously.

    A single clip-then-normalise does not do it: clipping the big weights down
    lowers the sum, so renormalising scales everything back up and the clipped
    ones drift over the cap again (2.02x instead of 2.00x). Alternating the two
    converges -- each round hands the removed mass to the uncapped weights,
    which is exactly the fixed point where capped items sit at the cap and the
    rest sum to the remainder.
    """
    lo, hi = cap[0] / n, cap[1] / n
    for _ in range(200):
        clipped = {k: min(max(v, lo), hi) for k, v in w.items()}
        total = sum(clipped.values())
        if total <= 0:
            return clipped
        nxt = {k: v / total for k, v in clipped.items()}
        if max(abs(nxt[k] - w[k]) for k in w) < 1e-15:
            w = nxt
            break
        w = nxt
    # Final clip so the invariant holds exactly; the residual is spread over the
    # uncapped weights, keeping the sum at 1 to within float error.
    w = {k: min(max(v, lo), hi) for k, v in w.items()}
    free = [k for k, v in w.items() if lo < v < hi]
    slack = 1.0 - sum(w.values())
    if free and abs(slack) > 0:
        share = slack / len(free)
        for k in free:
            w[k] = min(max(w[k] + share, lo), hi)
    return w


def effective_sample_size(weights):
    """Kish ESS: 1 / sum(w^2). Equals n for equal weights, drops as they skew."""
    s = sum(v * v for v in weights.values())
    return (1.0 / s) if s > 0 else 0.0


def facet_balance(facets, weights):
    """{facet: {level: mass}} -- what the weights actually achieved."""
    out = {}
    for facet in sorted({f for v in facets.values() for f in v}):
        acc = {}
        for name, lv in facets.items():
            acc[lv[facet]] = acc.get(lv[facet], 0.0) + weights.get(name, 0.0)
        out[facet] = acc
    return out


# --------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------
def weighted_mean(values, weights):
    tot = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / tot if tot > 0 else float("nan")


def weighted_median(values, weights):
    """Lower weighted median: the smallest value whose cumulative weight >= half."""
    pairs = sorted(zip(values, weights))
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return float("nan")
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def average_rank(scores, field, weights=None):
    """{model: weighted mean rank} (1 = best) over datasets scoring all of `field`.

    `scores` is {dataset: {model: value}} with LOWER meaning better, which is the
    convention summarize.primary_scores already uses (RMSE / Brier).
    Ties share the midrank, so a two-way tie scores 1.5 rather than silently
    handing one model the win.
    """
    acc = {m: [] for m in field}
    wts = {m: [] for m in field}
    for ds, s in scores.items():
        if any(m not in s for m in field):
            continue
        vals = [(s[m], m) for m in field]
        order = sorted(vals)
        ranks = {}
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
                j += 1
            mid = (i + j) / 2.0 + 1.0        # midrank, 1-based
            for k in range(i, j + 1):
                ranks[order[k][1]] = mid
            i = j + 1
        w = 1.0 if weights is None else weights.get(ds, 0.0)
        for m in field:
            acc[m].append(ranks[m])
            wts[m].append(w)
    return {m: weighted_mean(acc[m], wts[m]) for m in field if acc[m]}


def competitor_relative_rank(scores, ours, competitors, weights=None):
    """Average rank with our OWN rungs taken out of the field.

    Plain average rank over the whole field counts sibling rungs as opponents,
    so the stronger rung banks a free point every time it beats the weaker one
    -- which is us beating ourselves, not evidence about anybody else. Here each
    of our rungs is ranked in its own contest against the competitors alone:

        contest_r = {rung r} U competitors,  for each r in `ours`

    A rung's rank is its rank in its own contest. A competitor appears in every
    contest, so its rank is the mean across them. Adding or removing a rung
    therefore cannot move another rung's number -- the same stability property
    winrate_vs_opponents has, and for the same reason.
    """
    out = {}
    per_competitor = {c: [] for c in competitors}
    for r in ours:
        field = [r] + [c for c in competitors if c != r]
        if len(field) < 2:
            continue
        sub = average_rank(scores, field, weights)
        if r in sub:
            out[r] = sub[r]
        for c in competitors:
            if c in sub:
                per_competitor[c].append(sub[c])
    for c, vals in per_competitor.items():
        if vals:
            out[c] = sum(vals) / len(vals)
    return out


def bootstrap_competitor_relative_rank_ci(scores, ours, competitors,
                                          weights=None, n_boot=2000, seed=0):
    """95% CI for competitor_relative_rank, resampling DATASETS."""
    import numpy as np
    all_models = list(ours) + list(competitors)
    ds_list = sorted(d for d, s in scores.items()
                     if any(m in s for m in all_models))
    if not ds_list:
        return {m: (None, None) for m in all_models}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds_list), size=(n_boot, len(ds_list)))
    draws = {m: [] for m in all_models}
    for row in idx:
        sub, subw = {}, {}
        for c, i in enumerate(row):
            d = ds_list[i]
            key = f"{d}#{c}"
            sub[key] = scores[d]
            subw[key] = 1.0 if weights is None else weights.get(d, 0.0)
        r = competitor_relative_rank(sub, ours, competitors, subw)
        for m in all_models:
            if m in r and not math.isnan(r[m]):
                draws[m].append(r[m])
    out = {}
    for m in all_models:
        if draws[m]:
            arr = np.array(draws[m])
            out[m] = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
        else:
            out[m] = (None, None)
    return out


def bootstrap_average_rank_ci(scores, field, weights=None, n_boot=10000, seed=0):
    """95% CI for average_rank, resampling DATASETS (not seeds) with replacement.

    Weights ride along with the resampled datasets, so the interval reflects both
    dataset-sampling noise and the concentration the weighting introduces.
    """
    import numpy as np
    ds_list = sorted(d for d, s in scores.items() if all(m in s for m in field))
    if not ds_list:
        return {m: (None, None) for m in field}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds_list), size=(n_boot, len(ds_list)))
    draws = {m: [] for m in field}
    for row in idx:
        sub = {}
        subw = {}
        for c, i in enumerate(row):
            d = ds_list[i]
            key = f"{d}#{c}"          # keep duplicates distinct
            sub[key] = scores[d]
            subw[key] = 1.0 if weights is None else weights.get(d, 0.0)
        r = average_rank(sub, field, subw)
        for m in field:
            if m in r and not math.isnan(r[m]):
                draws[m].append(r[m])
    out = {}
    for m in field:
        if draws[m]:
            arr = np.array(draws[m])
            out[m] = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
        else:
            out[m] = (None, None)
    return out
