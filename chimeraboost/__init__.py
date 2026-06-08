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

from .sklearn_api import (
    ChimeraBoostRegressor,
    ChimeraBoostClassifier,
)

__all__ = [
    "ChimeraBoostRegressor",
    "ChimeraBoostClassifier",
]
__version__ = "0.11.0"
