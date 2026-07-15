"""Recipe = every factor of one synthetic dataset, drawn from the meta-distribution.

Observable marginals (n, d, task, categorical make-up, missingness, imbalance)
are bootstrapped jointly from the harvested real-dataset corpus (calibration.py).
Latent DGP factors (interaction depth, function mix, noise, aggregation) are
literature-informed priors (TabPFN/TabICLv2/Mitra), tuned only ever by the
backtest gate -- never by any sealed benchmark.

A recipe is fully determined by (VERSION, dataset id).
"""
import dataclasses

import numpy as np

from . import calibration

VERSION = "v2"
VERSION_SEED = 0x53594E32  # "SYN2"; bump alongside VERSION on ANY recipe change

N_CAP = 32000       # full-tier row cap (screen selection additionally caps at 8000)
D_CAP = 100
CARD_CAP = 64
CARD_CAP_ALLCAT = 16   # keeps forced cat_combinations arms tractable

_SATURATED_MOD = 10    # id % 10 == 7 -> saturated canary (~10% of id space)


@dataclasses.dataclass(frozen=True)
class Recipe:
    id: int
    version: str
    # size / task (calibrated)
    task: str                  # "regression" | "binary" | "multiclass"
    n: int
    n_classes: int             # 1 reg, 2 binary, 3..8 multiclass
    d: int
    d_informative: int
    irrelevant_fraction: float
    # graph (latent)
    n_nodes: int
    max_in_degree: int
    interaction_depth: int
    root_dist: str
    func_weights: dict
    agg_weights: dict
    # categoricals (calibrated count/cardinality, latent encoding)
    n_cat: int
    log_card_mu: float
    max_cardinality: int
    cat_encode_mode: str       # "quantile_shuffled" | "nearest_ref"
    entity_cat_fraction: float  # share of cat columns that are latent entities
    # target (latent noise, calibrated imbalance)
    noise_level: float         # reg: sigma/std(y_clean); clf: softmax temperature
    majority_frac: float       # target majority-class share (clf)
    saturated: bool
    # robustness axes (calibrated missingness, latent tails)
    missing_fraction: float
    heavy_tail_cols: float


def _dirichlet_around(base, strength, rng):
    keys = sorted(base)
    alpha = np.array([base[k] for k in keys], dtype=float) * strength
    w = rng.dirichlet(alpha)
    return {k: float(v) for k, v in zip(keys, w)}


def sample_recipe(dataset_id, rng):
    """Draw the full factor set for `dataset_id` using the provided Generator."""
    row = calibration.sample_row(rng)
    task = row["task"]
    n = int(np.clip(row["n"], 600, N_CAP))
    d = int(np.clip(row["d"], 5, D_CAP))

    saturated = (dataset_id % _SATURATED_MOD) == 7

    irrelevant_fraction = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.1, 0.6))
    d_informative = max(2, d - int(round(irrelevant_fraction * d)))
    irrelevant_fraction = (d - d_informative) / d

    # v2: shifted deep (v1's {.15,.35,.35,.15} over 1-4 let depth-4 ablations
    # WIN the backtest -- targets were too easy for the default capacity)
    interaction_depth = int(rng.choice([1, 2, 3, 4, 5],
                                       p=[0.10, 0.30, 0.35, 0.15, 0.10]))
    root_dist = str(rng.choice(["normal", "uniform", "mog", "heavy"],
                               p=[0.40, 0.20, 0.25, 0.15]))
    func_weights = _dirichlet_around(
        {"linear": 0.20, "neural": 0.35, "tree": 0.20, "product": 0.12,
         "plateau": 0.13}, strength=25.0, rng=rng)
    agg_weights = _dirichlet_around(
        {"sum": 0.55, "product": 0.15, "max": 0.15, "logsumexp": 0.15},
        strength=30.0, rng=rng)

    # categoricals: calibrated fraction & cardinality center
    n_cat = int(round(row["cat_frac"] * d))
    all_cat = n_cat >= d
    max_cardinality = CARD_CAP_ALLCAT if all_cat else CARD_CAP
    card_center = np.clip(row["max_card"], 2, max_cardinality)
    log_card_mu = float(np.log(card_center) * rng.uniform(0.55, 0.95))
    cat_encode_mode = str(rng.choice(["quantile_shuffled", "nearest_ref"]))
    # v2: ~40% of cat columns are latent ENTITIES (Zipf frequencies, per-level
    # target effect) instead of discretized views of smooth latents -- the
    # mechanism real high-card cats have and v1 lacked (realism CHECK failed)
    entity_cat_fraction = float(rng.uniform(0.2, 0.6))

    if task == "regression":
        n_classes = 1
        noise_level = float(np.exp(rng.uniform(np.log(0.03), np.log(0.5))))
        majority_frac = 0.0
    else:
        n_classes = 2 if task == "binary" else int(np.clip(row["n_classes"], 3, 8))
        noise_level = float(np.exp(rng.uniform(np.log(0.08), np.log(2.0))))
        default_maj = 1.0 / n_classes
        majority_frac = row["majority_frac"] or default_maj
        majority_frac = float(np.clip(majority_frac, default_maj, 0.95))

    missing_fraction = row["missing_rate"]
    if missing_fraction < 0.005:
        missing_fraction = 0.0
    missing_fraction = float(min(missing_fraction, 0.2))
    heavy_tail_cols = 0.0 if rng.random() < 0.6 else float(rng.uniform(0.1, 0.4))

    if saturated:
        # kr-vs-kp analog: exactly-representable, near-noiseless. Added model
        # complexity must HURT here (variance injection), never help.
        noise_level = 0.01 if task == "regression" else 0.02
        missing_fraction = 0.0
        heavy_tail_cols = 0.0
        func_weights = {"linear": 0.15, "neural": 0.05, "tree": 0.50,
                        "product": 0.05, "plateau": 0.25}
        if task != "regression":
            # keep classes comfortably populated so the class-presence retry
            # loop never has to soften the near-zero temperature (the floor
            # must stay near 0 for the canary to bite)
            majority_frac = float(min(majority_frac, 0.65))

    n_nodes = int(round(1.5 * d_informative)) + 4

    return Recipe(
        id=dataset_id, version=VERSION, task=task, n=n, n_classes=n_classes,
        d=d, d_informative=d_informative, irrelevant_fraction=irrelevant_fraction,
        n_nodes=n_nodes, max_in_degree=int(rng.integers(2, 6)),
        interaction_depth=interaction_depth, root_dist=root_dist,
        func_weights=func_weights, agg_weights=agg_weights,
        n_cat=n_cat, log_card_mu=log_card_mu, max_cardinality=max_cardinality,
        cat_encode_mode=cat_encode_mode,
        entity_cat_fraction=entity_cat_fraction, noise_level=noise_level,
        majority_frac=majority_frac, saturated=saturated,
        missing_fraction=missing_fraction, heavy_tail_cols=heavy_tail_cols)
