"""Custom TabArena model wrapping ChimeraBoost (a CatBoost-inspired, numba-backed
oblivious gradient-boosting library: https://github.com/bbstats/chimeraboost).

Mirrors the official custom_random_forest_model.py template. NOTE: the model class
MUST live in its own file (not the run script) because TabArena pickles it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


class ChimeraBoostModel(AbstractModel):
    """ChimeraBoost as an AutoGluon/TabArena model (scikit-learn style API)."""

    ag_key = "CHIMERA"
    ag_name = "ChimeraBoost"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """Label-encode categoricals and hand the model an object matrix while
        recording which columns are categorical, so ChimeraBoost applies its
        NATIVE ordered-target-statistics encoding (and auto cat_combinations on
        all-categorical data) rather than seeing cats as plain numbers."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
            # Positions of the categorical columns in the final column order,
            # passed verbatim as cat_features to ChimeraBoost. Computed once on
            # train and reused at predict (column order is identical).
            self._cat_indices = [X.columns.get_loc(c)
                                 for c in self._feature_generator.features_in]
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        # object dtype: cat columns stay integer-coded categories (ChimeraBoost
        # factorizes them), numeric columns stay float. fillna(0) for numerics;
        # the label encoder already maps unseen/NaN cats to a reserved code.
        return X.fillna(0).to_numpy(dtype=object)

    def _fit(self, X, y, num_cpus: int = 1, **kwargs):
        if self.problem_type in ["regression"]:
            from chimeraboost import ChimeraBoostRegressor

            model_cls = ChimeraBoostRegressor
        else:  # 'binary' and 'multiclass'
            from chimeraboost import ChimeraBoostClassifier

            model_cls = ChimeraBoostClassifier

        X = self.preprocess(X, y=y, is_train=True)
        params = self._get_model_params()
        # Tuned configs may sample linear_leaves=True, which raises on multiclass;
        # None = auto (binary on, multiclass off) keeps the config valid everywhere.
        if self.problem_type == "multiclass" and params.get("linear_leaves") is True:
            params["linear_leaves"] = None
        self.model = model_cls(**params)
        # early_stopping=True with no eval_set => ChimeraBoost auto-splits a
        # validation fraction internally and stops on it. cat_features triggers
        # native ordered-TS encoding + (on all-cat data) auto cat_combinations.
        cat = self._cat_indices or None
        self.model.fit(X, y, cat_features=cat)

    def _set_default_params(self):
        default_params = {
            "n_estimators": 500,
            "early_stopping": True,
            "thread_count": -1,
            "random_state": 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params


class ChimeraBoostE10Model(ChimeraBoostModel):
    """Deliberate variant: the exact default config + n_ensembles=10 (internal
    averaging of 10 boosting runs). A distinct method so it sits ALONGSIDE the
    default ChimeraBoost entry on the leaderboard, never replacing it."""

    ag_key = "CHIMERAE10"
    ag_name = "ChimeraBoost_e10"

    def _set_default_params(self):
        super()._set_default_params()  # identical default config ...
        self._set_default_param_value("n_ensembles", 10)  # ... + this one knob
        # ensemble_n_jobs left at 1: the 10 members run sequentially, each using
        # all threads (thread_count=-1); avoids core oversubscription. ~10x train time.


def get_configs_for_chimera_e10():
    """Single experiment: default config + n_ensembles=10, bagged, Lite."""
    from autogluon.common.space import Int

    from tabarena.utils.config_utils import ConfigGenerator

    gen = ConfigGenerator(
        model_cls=ChimeraBoostE10Model,
        manual_configs=[{}],
        search_space={"n_estimators": Int(100, 1000)},
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=0, fold_fitting_strategy="sequential_local"
    )


def get_configs_for_chimera(*, num_random_configs: int = 0):
    """Experiment configs for ChimeraBoost. Default: a single default config (no HPO),
    which is the cheapest way to get a first Elo placement on TabArena-Lite."""
    from autogluon.common.space import Int

    from tabarena.utils.config_utils import ConfigGenerator

    manual_configs = [{}]  # one default config
    search_space = {"n_estimators": Int(100, 1000)}  # only sampled if num_random_configs>0

    gen = ConfigGenerator(
        model_cls=ChimeraBoostModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )


class ChimeraBoostTunedModel(ChimeraBoostModel):
    """Tuned variant — distinct ag_name so it appears as a separate leaderboard entry."""

    ag_key = "CHIMERATUNED"
    ag_name = "ChimeraBoost_tuned"

    def _set_default_params(self):
        # Fixed tree budget for every tuned config; early stopping picks the
        # effective count, so low-lr configs aren't truncated by the 500 cap
        # the default entry uses.
        self._set_default_param_value("n_estimators", 1500)
        super()._set_default_params()


def get_configs_for_chimera_tuned(*, num_random_configs: int = 200):
    """Random HP configs for the tuned TabArena entry (default 200, the
    TabArena convention for tuned methods).

    Search-space design (pre-registered from dev-side evidence only — Grinsztajn
    / OpenML / PMLB studies; never from TabArena results):
    * Core capacity/regularization knobs with known per-dataset variance:
      learning_rate, depth, l2_leaf_reg, min_child_weight, subsample, colsample,
      leaf_estimation_iterations, max_bins.
    * The default-OFF research flags that were KILLED as defaults precisely
      because each helps only a narrow slice of datasets (see
      benchmarks/research/SUMMARY.md) — per-task HPO is the regime where
      narrow-sweet-spot levers earn their keep: linear_leaves (regression wins
      pol/abalone; default only auto-on for binary), onehot_low_card,
      cat_aware_binning, cat_combinations_selective (hard-capped at 20 pairs;
      the plain cat_combinations flag is deliberately EXCLUDED — explicit True
      bypasses the auto-rule's <=1000-pair resource guard and would explode on
      high-cardinality tasks), and hs_lambda (null at depth 6 but designed to
      make deeper trees safe — searchable jointly with depth here).
    * n_estimators stays fixed (1500 cap + early stopping): searching a budget
      cap under ES only adds noise.
    """
    from autogluon.common.space import Categorical, Int, Real

    from tabarena.utils.config_utils import ConfigGenerator

    manual_configs = [{}]  # include the default config too
    search_space = {
        "learning_rate": Real(0.03, 0.3, log=True),
        "depth": Int(4, 8),
        "l2_leaf_reg": Real(0.1, 10.0, log=True),
        "min_child_weight": Real(0.0, 8.0),
        "subsample": Real(0.5, 1.0),
        "colsample": Real(0.5, 1.0),
        "leaf_estimation_iterations": Int(1, 5),
        "max_bins": Categorical(128, 254),
        "linear_leaves": Categorical(False, True),
        "onehot_low_card": Categorical(False, True),
        "cat_aware_binning": Categorical(False, True),
        "cat_combinations_selective": Categorical(False, True),
        "hs_lambda": Real(0.0, 4.0),
    }

    gen = ConfigGenerator(
        model_cls=ChimeraBoostTunedModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
