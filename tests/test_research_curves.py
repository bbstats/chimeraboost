"""Unit tests for the cascade's paired-curve comparison (cheap go/kill signal)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))
from research import curves  # noqa: E402


def test_variant_uniformly_better():
    b = np.linspace(1.0, 0.5, 50)
    v = b - 0.05                       # 0.05 lower everywhere
    c = curves.compare(b, v)
    assert c["best_val_delta"] < 0
    assert c["dominance"] == 1.0       # dominates everywhere -> strong promote
    assert c["early_signal"] < 0
    assert c["area_between"] < 0


def test_variant_uniformly_worse():
    b = np.linspace(1.0, 0.5, 50)
    v = b + 0.05
    c = curves.compare(b, v)
    assert c["best_val_delta"] > 0
    assert c["dominance"] == 0.0       # dominated everywhere -> fast kill
    assert c["early_signal"] > 0


def test_identical_curves_are_flat():
    b = np.linspace(1.0, 0.4, 40)
    c = curves.compare(b, b.copy())
    assert c["best_val_delta"] == 0.0
    assert c["best_val_delta_pct"] == 0.0
    assert c["dominance"] == 1.0       # <= holds with equality everywhere


def test_unequal_length_curves_align():
    b = np.linspace(1.0, 0.5, 50)
    v = np.linspace(1.0, 0.5, 30) - 0.02
    c = curves.compare(b, v)
    assert c["len_baseline"] == 50 and c["len_variant"] == 30
    # best_val still uses each curve's own minimum
    assert np.isclose(c["best_val_variant"], v.min())


def test_best_val_delta_pct_normalizes():
    b = np.array([2.0, 1.0])
    v = np.array([2.0, 0.9])           # best 0.9 vs 1.0 -> -10%
    assert np.isclose(curves.best_val_delta_pct(b, v), -0.1, atol=1e-9)
