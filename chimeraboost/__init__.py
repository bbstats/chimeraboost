"""ChimeraBoost: a CatBoost-inspired gradient boosting library in pure Python.

Key ingredients borrowed from CatBoost:
  * Ordered target statistics for categorical features (anti-leakage encoding)
  * Oblivious / symmetric trees (fast, strongly regularized -> good defaults)
  * Histogram-based quantized splitting (numba accelerated)

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
from .warmup import warmup, _warmup_from_env

# CHIMERABOOST_WARMUP=1 -> compile the numba kernels in a background daemon
# thread at import ("sync" blocks instead). For short-lived workers where the
# first fit/predict would otherwise pay the JIT. See warmup().
_warmup_from_env(_os.environ.get("CHIMERABOOST_WARMUP"))

__all__ = [
    "ChimeraBoostRegressor",
    "ChimeraBoostClassifier",
    "warmup",
]
__version__ = "0.14.1"
