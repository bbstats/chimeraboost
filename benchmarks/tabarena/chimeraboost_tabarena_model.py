"""Custom TabArena model wrapping ChimeraBoost (a CatBoost-inspired, numba-backed
oblivious gradient-boosting library: https://github.com/bbstats/chimeraboost).

Mirrors the official custom_random_forest_model.py template. NOTE: the model class
MUST live in its own file (not the run script) because TabArena pickles it.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


class ChimeraBoostModel(AbstractModel):
    """ChimeraBoost as an AutoGluon/TabArena model (scikit-learn style API)."""

    ag_key = "CHIMERA"
    ag_name = "ChimeraBoost"
    seed_name = "random_state"  # AutoGluon injects the framework seed here

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        """Pass the frame straight to ChimeraBoost with categoricals marked by
        name. ChimeraBoost factorizes them with its native ordered-target-
        statistics encoding and routes NaN to a dedicated missing bin, so we do
        NOT label-encode or impute here (both would discard signal)."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            # category-dtype columns (AutoGluon marks them; valid_raw_types keeps
            # them as 'category'); recorded once and reused at predict.
            self._cat_col_names = list(X.select_dtypes(include="category").columns)
        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()
        if self.problem_type in ["regression"]:
            from chimeraboost import ChimeraBoostRegressor

            model_cls = ChimeraBoostRegressor
        else:  # 'binary' and 'multiclass'
            from chimeraboost import ChimeraBoostClassifier

            model_cls = ChimeraBoostClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        # Run on the CPU budget TabArena allocates (thread_count<0 => all cores).
        params["thread_count"] = num_cpus
        # Tuned configs may sample linear_leaves=True, which raises on multiclass;
        # None = auto (binary on, multiclass off) keeps the config valid everywhere.
        if self.problem_type == "multiclass" and params.get("linear_leaves") is True:
            params["linear_leaves"] = None
        self.model = model_cls(**params)

        cat = self._cat_col_names or None
        # Use TabArena's validation split for early stopping when provided (don't
        # carve a second holdout out of the training data); else ChimeraBoost
        # auto-splits internally via early_stopping=True.
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            eval_set = (X_val, y_val)

        fit_kwargs = {}
        # Stop boosting once TabArena's fit budget runs out, leaving 5% headroom.
        # Skipped for the E10 variant: ChimeraBoost disallows callbacks when
        # n_ensembles>1 (members fit in worker processes).
        if time_limit is not None and params.get("n_ensembles") in (None, 1):
            deadline = start_time + 0.95 * time_limit

            def _time_stop(iteration, train_loss, val_loss, model):
                return time.time() >= deadline

            fit_kwargs["callbacks"] = _time_stop

        self.model.fit(X, y, cat_features=cat, eval_set=eval_set, **fit_kwargs)

    def _set_default_params(self):
        default_params = {
            # Cap only: early stopping picks the real count and the auto learning
            # rate is pinned at 0.1 under ES, so a high cap is LR-neutral headroom.
            "n_estimators": 10000,
            "early_stopping": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Physical cores only (matches RealMLP/XRFM); ChimeraBoost is CPU-only.
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        return num_cpus, 0

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = 1,
        **kwargs,
    ) -> int:
        """Conservative peak-fit RAM estimate (bytes) for fold-parallel scheduling.

        ChimeraBoost is a CPU GBDT; peak memory is dominated by O(n_samples *
        n_features) terms — the input matrix, the quantized bin codes, and a few
        per-row stat buffers (gradients, hessians, predictions, the validation
        copy). The oblivious trees themselves are negligible (2**depth leaves *
        n_estimators * a few bytes). We deliberately over-estimate (a 3x factor
        on the input matrix + a 1 GB baseline) so the scheduler packs fewer folds
        rather than risk OOM.
        """
        n, p = int(X.shape[0]), int(X.shape[1])
        k = max(int(num_classes or 1), 1)
        cell = 8  # float64 / object-pointer width
        data = n * p * cell          # input matrix (object array when cats present)
        binned = n * p * 2           # quantized bin codes (uint8/uint16)
        stats = n * k * cell * 6     # grad / hess / pred / weight / val buffers
        hist = p * 256 * 2 * cell    # transient per-level histograms
        baseline = 1_000_000_000     # python + numba + autogluon overhead
        return int(baseline + 3 * data + binned + stats + hist)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}


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
        # all allocated threads; avoids core oversubscription. ~10x train time.


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
    """Tuned variant — distinct ag_name so it appears as a separate leaderboard entry.

    Inherits the n_estimators=10000 cap from the base; early stopping picks the
    effective count, so low-lr tuned configs aren't truncated."""

    ag_key = "CHIMERATUNED"
    ag_name = "ChimeraBoost_tuned"


def get_configs_for_chimera_tuned(*, num_random_configs: int = 200):
    """Random HP configs for the tuned TabArena entry (default 200, the
    TabArena convention for tuned methods).

    Search-space design (pre-registered from dev-side evidence only — Grinsztajn
    / OpenML / PMLB studies; never from TabArena results):
    * Core capacity/regularization knobs with known per-dataset variance:
      learning_rate, depth, l2_leaf_reg, min_child_weight, subsample, colsample,
      leaf_estimation_iterations, max_bins.
    * Categorical-handling knobs that exist in current source: cat_smoothing
      (ordered-TS pseudocount) and cat_n_permutations (anti-leakage averaging).
      The raw cat_combinations flag is deliberately EXCLUDED — explicit True
      bypasses the auto-rule's <=1000-pair resource guard and would explode on
      high-cardinality tasks; combinations stay on the adaptive auto default.
    * linear_leaves (regression wins pol/abalone; default only auto-on for
      binary) searched jointly with its regularizer linear_lambda.
    * ordered_boosting (CatBoost-style ordered target stats) — default off, a
      per-task lever for leakage-sensitive small data.
    * n_estimators stays fixed (10000 cap + early stopping): searching a budget
      cap under ES only adds noise.

    NOTE: the 8 default-off research flags (onehot_low_card, cat_aware_binning,
    cat_combinations_selective, hs_lambda, …) were REMOVED from the model in the
    June de-slop pass (benchmarks/research/SUMMARY.md), so they are no longer
    searchable — passing them now raises TypeError at construction.
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
        "linear_lambda": Real(0.1, 10.0, log=True),
        "cat_smoothing": Real(0.1, 10.0, log=True),
        "cat_n_permutations": Int(1, 8),
        "ordered_boosting": Categorical(False, True),
    }

    gen = ConfigGenerator(
        model_cls=ChimeraBoostTunedModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
