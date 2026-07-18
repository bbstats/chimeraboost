"""warmup() must compile every default-path kernel without side effects."""

import threading

import numpy as np

from chimeraboost import ChimeraBoostClassifier, warmup
from chimeraboost.warmup import _warmup_from_env


def test_warmup_compiles_all_default_path_kernels():
    elapsed = warmup()
    assert elapsed > 0

    from chimeraboost.binning import _bin_matrix
    from chimeraboost.losses import _sigmoid
    from chimeraboost.target_encoding import _ordered_ts
    from chimeraboost.tree import (
        _build_split_descend,
        _leaf_values,
        _linear_leaf_fit,
        _linear_predict,
        _loo_leaf_step,
        _predict_forest_linear_rm,
        _predict_forest_rm,
        _predict_tree,
    )

    for kernel in (_bin_matrix, _sigmoid, _ordered_ts, _build_split_descend,
                   _leaf_values, _linear_leaf_fit, _linear_predict,
                   _loo_leaf_step, _predict_tree, _predict_forest_rm,
                   _predict_forest_linear_rm):
        assert kernel.signatures, f"{kernel.py_func.__name__} not compiled by warmup()"


def test_background_warmup_returns_daemon_thread_and_finishes():
    t = warmup(background=True)
    assert isinstance(t, threading.Thread) and t.daemon
    t.join(timeout=300)
    assert not t.is_alive()


def test_warmup_env_dispatch():
    assert _warmup_from_env(None) is None
    assert _warmup_from_env("") is None
    assert _warmup_from_env("0") is None
    assert isinstance(_warmup_from_env("1"), float)  # "1" = blocking
    t = _warmup_from_env("background")
    assert isinstance(t, threading.Thread)
    t.join(timeout=300)


def test_warmup_does_not_disturb_global_rng_or_model_output():
    rng_before = np.random.get_state()[1].copy()

    X = np.random.default_rng(7).standard_normal((300, 4))
    y = (X[:, 0] > 0).astype(int)
    ref = ChimeraBoostClassifier(n_estimators=20, random_state=3).fit(X, y)
    p_ref = ref.predict_proba(X)

    warmup()

    rng_after = np.random.get_state()[1]
    assert np.array_equal(rng_before, rng_after)

    again = ChimeraBoostClassifier(n_estimators=20, random_state=3).fit(X, y)
    np.testing.assert_array_equal(p_ref, again.predict_proba(X))
