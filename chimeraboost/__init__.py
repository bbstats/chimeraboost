"""ChimeraBoost: a CatBoost-inspired gradient boosting library in pure Python.

Borrowed from CatBoost (Prokhorenkova et al. 2018):
  * Ordered target statistics for categorical features (anti-leakage encoding)
  * Oblivious / symmetric trees (fast, strongly regularized -> good defaults)

Borrowed from LightGBM:
  * Histogram-based split finding on pre-binned features (Ke et al. 2017)
  * Quantized gradient histograms (Shi, Ke et al. 2022)

docs/attribution.md maps every technique to its source, and says which of them
originated here.

Public API:
  >>> from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier
  >>> model = ChimeraBoostClassifier().fit(X, y, cat_features=[0, 3])
  >>> proba = model.predict_proba(X_test)
"""

import os as _os

from .sklearn_api import (
    ChimeraBoostRegressor,
    ChimeraBoostClassifier,
)
from .quantile_api import ChimeraBoostQuantileRegressor
from .losses import CustomObjective
from . import quantile_metrics
from .warmup import warmup, _warmup_from_env

# CHIMERABOOST_WARMUP=1 -> compile the numba kernels at import ("background"
# uses a daemon thread instead). For short-lived workers where the first
# fit/predict would otherwise pay the JIT. See warmup().
_warmup_from_env(_os.environ.get("CHIMERABOOST_WARMUP"))

__all__ = [
    "ChimeraBoostRegressor",
    "ChimeraBoostClassifier",
    "ChimeraBoostQuantileRegressor",
    "CustomObjective",
    "quantile_metrics",
    "warmup",
]
__version__ = "0.31.0"
