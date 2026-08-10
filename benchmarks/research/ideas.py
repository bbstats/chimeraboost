"""The pre-registered research queue.

Each idea is a default-off variant expressed as ``params`` (kwargs that override
the out-of-box baseline), plus a PRE-REGISTERED hypothesis (direction + which
datasets it should help) recorded BEFORE any results are seen. Lead categorical-
first; the cheap-tier signal is allowed to reprioritize.

``implemented=True`` means the variant's flag already exists in the library and
can be run through the cascade today. The CatBoost-gap levers (C*/G*) are listed
here as the pre-registered agenda; each flips to implemented as its default-off
flag lands (one change at a time).

Two ideas with KNOWN outcomes are kept for the engine self-test:
  * ``crossfeat_off``  -- a known LARGE SIGNAL (removing a shipped default must
    visibly hurt on the numeric sets where cross features apply).
  * ``noop``           -- an exact no-op (no overrides at all), which must come
    back bit-identical.
The harness must re-confirm both, proving it discriminates a real effect from
nothing.

Both anchors are chosen to be **drift-proof**, because the previous pair was not
(see cascade.selftest): ``linear_leaves`` stopped being off-by-default when the
regressor began auditioning it, and ``patience300`` turned out not to be inert
under ``early_stopping=False``. An anchor whose premise is "this flag is
currently off" expires the moment a default moves. These two cannot: one removes
a default that would have to be un-shipped for the anchor to go stale, and the
other overrides nothing at all.
"""

# direction: "lower_better" means the variant should DECREASE the primary
# loss/metric (RMSE/Brier/val-loss) where the hypothesis says it helps.
IDEAS = {
    # --- self-test anchors (known truths) ----------------------------------
    "crossfeat_off": dict(
        params={"cross_features": False},
        category="selftest",
        implemented=True,
        direction="higher_worse",
        hypothesis="Cross features ship on by default and are worth ~1.5% on "
                   "Grinsztajn (51W/8L when added). Removing them must visibly "
                   "HURT on the numeric sets where pairs exist, and be inert "
                   "where they do not. Self-test anchor: must re-confirm a LARGE "
                   "SIGNAL. Drift-proof: going stale would mean cross features "
                   "had been un-shipped.",
    ),
    "noop": dict(
        params={},
        category="selftest",
        implemented=True,
        direction="none",
        hypothesis="No overrides at all, so the variant IS the baseline. Every "
                   "curve must come back bit-identical. Self-test anchor: must "
                   "re-confirm EXACT FLATNESS -- this is the harness's own "
                   "reproducibility control, and the one anchor no library "
                   "change can invalidate.",
    ),

    # --- C: categorical levers (lead the queue) ----------------------------
    "C1_onehot_low_card": dict(
        params={"onehot_low_card": True},
        category="categorical",
        implemented=True,   # flipped on in Part C
        direction="lower_better",
        hypothesis="One-hot encoding low-cardinality categoricals (<= ~8 levels) "
                   "alongside ordered TS lets the tree make exact subset splits "
                   "on rare discrete categories. HELPS mixed / high-card "
                   "categorical sets (adult, bank-marketing, kr-vs-kp, car); "
                   "NEUTRAL on the numeric sets (electricity, covertype, pol).",
    ),
    "C2_pertree_ts_permutation": dict(
        params={"cat_pertree_permutation": True},
        category="categorical",
        implemented=False,   # DEFERRED -- see note below
        direction="lower_better",
        hypothesis="A fresh TS permutation per tree (per block) approximates "
                   "CatBoost's per-snapshot adaptivity; helps high-signal "
                   "categorical sets. DEFERRED: this requires RE-ENCODING the "
                   "categorical columns every boosting round, a fundamental break "
                   "from ChimeraBoost's encode-once-then-bin architecture (the "
                   "binned matrix the tree kernel consumes is built once at fit). "
                   "Not justified for a near-certain kill given 6/6 other levers "
                   "killed; would need a dedicated re-encode-per-round redesign.",
    ),
    "C3_selective_cat_combinations": dict(
        params={"cat_combinations_selective": True},
        category="categorical",
        implemented=True,
        direction="lower_better",
        hypothesis="Selecting cat-combinations by target association (mutual "
                   "info / gain) instead of all C(n,2), and allowing them on "
                   "MIXED data, extends the car-type interaction win without "
                   "crowding numeric splits.",
    ),
    "C4_cat_aware_binning": dict(
        params={"cat_aware_binning": True},
        category="categorical",
        implemented=True,
        direction="lower_better",
        hypothesis="More/different bins for encoded categorical columns gives "
                   "sharper categorical splits.",
    ),

    # --- G: general default-accuracy levers ---------------------------------
    "G1_forest_joint_leaf_refit": dict(
        params={"forest_leaf_refit": True},
        category="general",
        implemented=True,
        # Post-fit pass: it rewrites leaf values AFTER boosting, so the per-round
        # validation_history_ curve cannot see it. The fast (curve) tier is blind
        # to it -> evaluate directly at the promotion tier (true test metric).
        post_fit=True,
        direction="lower_better",
        hypothesis="A post-fit ridge over all leaves couples redundant oblivious "
                   "splits and recovers sharpness -- the highest-upside Brier "
                   "lever. De-risk with a slow reference impl first.",
    ),
    "G2_adaptive_leaf_shrinkage": dict(
        params={"adaptive_leaf_shrinkage": 1.0},
        category="general",
        implemented=True,
        direction="lower_better",
        hypothesis="Per-leaf L2 scaled by leaf mass regularizes low-mass leaves "
                   "harder; small broad Brier gain.",
    ),
    "G3_adaptive_leaf_estimation": dict(
        params={"adaptive_leaf_estimation": True},
        category="general",
        implemented=True,
        direction="lower_better",
        hypothesis="Scaling leaf_estimation_iterations by data size/signal helps "
                   "where the single Newton step underfits.",
    ),
    "G4_ordered_plus_leaf_estimation": dict(
        params={"ordered_boosting": True, "ordered_leaf_estimation": True},
        category="general",
        implemented=True,
        direction="lower_better",
        hypothesis="Reconciling ordered_boosting WITH leaf estimation (mutually "
                   "exclusive today) gives CatBoost's 'ordered boosting AND leaf "
                   "machinery together' -- what the TabArena gap analysis points "
                   "at directly.",
    ),
}


def get(name):
    if name not in IDEAS:
        raise KeyError(f"unknown idea {name!r}; known: {sorted(IDEAS)}")
    return IDEAS[name]
