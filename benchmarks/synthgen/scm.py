"""Structural-causal-model core: layered DAG + random node functions.

The generative vocabulary follows the TabPFN/TabICLv2 prior family, numpy-only:
per-node function classes {linear, neural (rich activations), oblivious tree
ensemble, product, plateau}, multi-parent aggregation {sum, product, max,
logsumexp}, z-scored nodes with small inner noise.

Determinism: every function takes explicit `np.random.Generator`s derived from
the dataset's SeedSequence. Dense transforms avoid BLAS (einsum) so content
does not depend on the BLAS build.
"""
import dataclasses

import numpy as np

FUNC_KINDS = ("linear", "neural", "tree", "product", "plateau")
AGG_KINDS = ("sum", "product", "max", "logsumexp")


@dataclasses.dataclass
class Dag:
    parents: list          # list[list[int]] per node
    layer: list            # layer index per node (roots = 0)
    target: int            # target node index (deepest layer)
    n_nodes: int

    def ancestor_masks(self):
        """Bitmask of ancestors per node (python ints, cheap for <=200 nodes)."""
        anc = [0] * self.n_nodes
        for i in range(self.n_nodes):        # topological: parents precede children
            m = 0
            for p in self.parents[i]:
                m |= anc[p] | (1 << p)
            anc[i] = m
        return anc


def build_dag(n_nodes, depth, max_in_degree, rng):
    """Layered DAG with exactly `depth` layers above the roots.

    Every non-root draws >=1 parent from the previous layer (guarantees its
    layer = its longest path) plus optional skip edges from any earlier layer.
    The target node is the last node, alone responsible for layer == depth.
    """
    n_nodes = max(n_nodes, depth + 3)
    # distribute nodes over layers 0..depth; roots get the biggest share
    sizes = [1] * (depth + 1)
    sizes[0] = max(2, int(round(n_nodes * 0.35)))
    remaining = n_nodes - sum(sizes)
    for _ in range(max(0, remaining)):
        sizes[int(rng.integers(0, depth + 1))] += 1
    layer_of, parents = [], []
    layers = []
    node = 0
    for li, sz in enumerate(sizes):
        layers.append(list(range(node, node + sz)))
        for _ in range(sz):
            layer_of.append(li)
            node += 1
    n_nodes = node
    for i in range(n_nodes):
        li = layer_of[i]
        if li == 0:
            parents.append([])
            continue
        k = 1 + int(rng.binomial(max_in_degree - 1, 0.5))
        prev = layers[li - 1]
        ps = {int(rng.choice(prev))}
        earlier = [j for j in range(i) if layer_of[j] < li]
        while len(ps) < min(k, len(earlier)):
            ps.add(int(rng.choice(earlier)))
        parents.append(sorted(ps))
    target = layers[depth][-1]
    return Dag(parents=parents, layer=layer_of, target=target, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# node functions
# ---------------------------------------------------------------------------
_ACTIVATIONS = ("tanh", "relu", "sine", "abs", "sqrt", "clipexp", "step", "rbf")


def _apply_act(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sine":
        return np.sin(x)
    if name == "abs":
        return np.abs(x)
    if name == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    if name == "clipexp":
        return np.exp(np.clip(x, -4.0, 4.0))
    if name == "step":
        return (x > 0.0).astype(np.float64)
    if name == "rbf":
        return np.exp(-np.clip(x, -30.0, 30.0) ** 2)
    raise ValueError(name)


def _zscore(v):
    std = v.std()
    if std < 1e-12:
        return v - v.mean()
    return (v - v.mean()) / std


def _aggregate(P, kind):
    """Collapse an (n, k) parent matrix to (n,)."""
    if P.shape[1] == 1:
        return P[:, 0]
    if kind == "sum":
        return P.sum(axis=1) / np.sqrt(P.shape[1])
    if kind == "product":
        return np.prod(np.clip(P, -5.0, 5.0), axis=1)
    if kind == "max":
        return P.max(axis=1)
    if kind == "logsumexp":
        m = P.max(axis=1)
        return m + np.log(np.exp(P - m[:, None]).sum(axis=1))
    raise ValueError(kind)


def make_node_function(kind, k, rng):
    """Build f(P: (n,k) float64) -> (n,) float64 with weights drawn from rng."""
    if kind == "linear":
        w = rng.normal(size=k) / np.sqrt(k)
        b = rng.normal() * 0.3
        return lambda P: np.einsum("ij,j->i", P, w) + b
    if kind == "neural":
        h = int(rng.integers(4, 17))
        W1 = rng.normal(size=(k, h)) / np.sqrt(k)
        b1 = rng.normal(size=h) * 0.5
        w2 = rng.normal(size=h) / np.sqrt(h)
        act = _ACTIVATIONS[int(rng.integers(0, len(_ACTIVATIONS)))]
        return lambda P: np.einsum(
            "ih,h->i", _apply_act(act, np.einsum("ij,jh->ih", P, W1) + b1), w2)
    if kind == "tree":
        n_trees = int(rng.integers(3, 9))
        specs = []
        for _ in range(n_trees):
            dep = int(rng.integers(2, 4))
            feats = rng.integers(0, k, size=dep)
            qs = rng.uniform(0.2, 0.8, size=dep)
            leaves = rng.normal(size=2 ** dep)
            specs.append((feats, qs, leaves))

        def _trees(P, specs=specs):
            out = np.zeros(P.shape[0])
            for feats, qs, leaves in specs:
                idx = np.zeros(P.shape[0], dtype=np.int64)
                for bit, (f, q) in enumerate(zip(feats, qs)):
                    thr = np.quantile(P[:, f], q)
                    idx |= (P[:, f] > thr).astype(np.int64) << bit
                out += leaves[idx]
            return out
        return _trees
    if kind == "product":
        if k == 1:
            c = rng.normal()
            return lambda P: np.clip(P[:, 0], -5.0, 5.0) * np.abs(
                np.clip(P[:, 0], -5.0, 5.0)) * np.sign(c)
        cols = rng.permutation(k)[:2]
        return lambda P: (np.clip(P[:, cols[0]], -5.0, 5.0)
                          * np.clip(P[:, cols[1]], -5.0, 5.0))
    if kind == "plateau":
        n_bins = int(rng.integers(3, 9))
        levels = rng.normal(size=n_bins)
        qs = np.sort(rng.uniform(0.05, 0.95, size=n_bins - 1))
        agg = AGG_KINDS[int(rng.integers(0, len(AGG_KINDS)))]

        def _plateau(P, levels=levels, qs=qs, agg=agg):
            v = _aggregate(P, agg)
            edges = np.quantile(v, qs)
            return levels[np.searchsorted(edges, v)]
        return _plateau
    raise ValueError(kind)


def _sample_root(dist, n, rng):
    if dist == "normal":
        return rng.normal(size=n)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n)
    if dist == "mog":
        n_comp = int(rng.integers(2, 4))
        means = rng.normal(size=n_comp) * 2.0
        sds = rng.uniform(0.3, 1.0, size=n_comp)
        comp = rng.integers(0, n_comp, size=n)
        return _zscore(rng.normal(size=n) * sds[comp] + means[comp])
    if dist == "heavy":
        if rng.random() < 0.5:
            return _zscore(rng.standard_t(3.0, size=n))
        return _zscore(rng.lognormal(0.0, 1.0, size=n))
    raise ValueError(dist)


def propagate(dag, recipe, n, seed_seq):
    """Compute the (n, n_nodes) node-value matrix.

    Per-node child streams: a recipe edit that changes one node's function
    cannot shift the draws of other nodes.
    """
    kinds = list(FUNC_KINDS)
    kind_w = np.array([recipe.func_weights[k] for k in kinds], dtype=float)
    kind_w = kind_w / kind_w.sum()
    agg_w = np.array([recipe.agg_weights[a] for a in AGG_KINDS], dtype=float)
    agg_w = agg_w / agg_w.sum()

    node_seeds = seed_seq.spawn(dag.n_nodes)
    M = np.empty((n, dag.n_nodes), dtype=np.float64)
    node_kind = [""] * dag.n_nodes
    for i in range(dag.n_nodes):
        rng = np.random.default_rng(node_seeds[i])
        ps = dag.parents[i]
        if not ps:
            M[:, i] = _sample_root(recipe.root_dist, n, rng)
            node_kind[i] = "root"
            continue
        kind = kinds[int(rng.choice(len(kinds), p=kind_w))]
        P = M[:, ps]
        if kind in ("linear",) and len(ps) >= 2 and rng.random() < 0.5:
            # sometimes route multi-parent linears through a nonlinear aggregator
            agg = AGG_KINDS[int(rng.choice(len(AGG_KINDS), p=agg_w))]
            P = _aggregate(P, agg)[:, None]
        f = make_node_function(kind, P.shape[1], rng)
        v = _zscore(f(P))
        sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
        M[:, i] = v + rng.normal(0.0, sigma, size=n)
        node_kind[i] = kind
    return M, node_kind
