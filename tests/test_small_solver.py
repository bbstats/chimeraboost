"""Tests for the hand-rolled `_solve_small` LU kernel that replaced the single
`np.linalg.solve` call in `_linear_leaf_fit` (dropping numba's LAPACK-binding
JIT cost from the cold-start fit). numpy stays a dependency, so comparing the
jitted kernel against `np.linalg.solve` remains valid indefinitely."""

import numpy as np

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier
from chimeraboost.tree import _solve_small


def test_solve_small_matches_numpy_on_ridge_gram_and_perturbations():
    """~500 random tiny systems, d in 1..11, of the exact shape `_linear_leaf_fit`
    builds: SPD ridge-Gram matrices (X^T X + diag(uniform) + 1e-9 I) plus
    asymmetric perturbations of the same. Must agree with np.linalg.solve to a
    tight relative bound. `_solve_small` mutates its inputs, so pass copies."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        d = int(rng.integers(1, 12))
        m = int(rng.integers(d, 3 * d + 1))
        X = rng.normal(size=(m, d))
        A = X.T @ X
        A += np.diag(rng.uniform(0.1, 5.0, size=d))
        A += 1e-9 * np.eye(d)
        if rng.random() < 0.5:
            # asymmetric perturbation of the same well-conditioned system
            A = A + 0.01 * rng.normal(size=(d, d))
        b = rng.normal(size=d)

        x_ref = np.linalg.solve(A, b)
        x = _solve_small(A.copy(), b.copy())

        err = np.max(np.abs(x - x_ref)) / max(1.0, np.max(np.abs(x_ref)))
        worst = max(worst, err)
    assert worst < 1e-12, f"max relative diff {worst:g} exceeds 1e-12"


def test_solve_small_singular_returns_nan_without_raising():
    """A zero matrix has no valid pivot; the kernel must signal that with an
    all-NaN vector (the caller's fallback trigger) rather than raise or divide."""
    x = _solve_small(np.zeros((3, 3)), np.ones(3))
    assert x.shape == (3,)
    assert np.all(np.isnan(x))


def test_linear_leaves_binary_fit_is_finite_and_reduces_loss():
    """Tiny binary fit (linear leaves auto-on): predictions finite and train
    logloss beats the constant base-rate baseline."""
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)

    m = ChimeraBoostClassifier(n_estimators=60, thread_count=1, random_state=0).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert np.all(np.isfinite(p))

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    model_ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    base = np.clip(y.mean(), eps, 1 - eps)
    base_ll = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
    assert model_ll < base_ll


def test_linear_leaves_regression_fit_is_finite_and_reduces_loss():
    """Tiny regression fit with linear_leaves=True: predictions finite and train
    MSE beats predicting the target mean."""
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(size=(n, 6))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)

    m = ChimeraBoostRegressor(
        n_estimators=60, linear_leaves=True, thread_count=1, random_state=0
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred))

    model_mse = np.mean((y - pred) ** 2)
    base_mse = np.mean((y - y.mean()) ** 2)
    assert model_mse < base_mse
