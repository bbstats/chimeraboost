"""warmup() must compile every default-path kernel without side effects."""

import importlib
import json
import os
import pathlib
import subprocess
import sys
import threading

import numpy as np
import pytest

import chimeraboost
from chimeraboost import ChimeraBoostClassifier, warmup
from chimeraboost.warmup import _cache_is_cold, _warmup_from_env, main

# `chimeraboost.warmup` is the *function* -- the package re-exports it over the
# module of the same name -- so reach the module through sys.modules to patch it.
warmup_module = importlib.import_module("chimeraboost.warmup")

# Kernels warmup() deliberately leaves uncompiled, each with the reason.
# The coverage test asserts this set EXACTLY (after dropping CACHE_DEPENDENT,
# below): a new kernel that nobody warms fails it, and so does an entry here
# that quietly became covered (meaning the reason below is stale). The
# previous hand-written list missed two real holes.
NOT_WARMED = {
    # Non-quantized tree build: quantize_gradients has been default-on since
    # 0.18.0, so the default path uses the _q twin.
    "chimeraboost.tree._build_split_descend",
    # Reference oracles for the fused kernels, called only by
    # tests/test_tree_kernels.py -- on no runtime path at all.
    "chimeraboost.tree._build_histograms_into",
    "chimeraboost.tree._best_split",
    "chimeraboost.tree._build_and_split",
    # Column-major predict twins; the row-major kernels are the default path.
    # (`_descend_leaves` itself is warmed: it is the large-n arm of
    # ObliviousTree.apply. Its serial twin stays test-only.)
    "chimeraboost.tree._descend_leaves_serial",
    "chimeraboost.tree._predict_forest",
    # SHAP, and the column-major predictor its path uses: opt-in via
    # warmup(shap=True), ~3.7 s of compile most callers never need.
    "chimeraboost.tree._shap_forest_linear",
    "chimeraboost.tree._predict_forest_linear",
    # ... and its vector-leaf twin, serving the multiclass and multi-quantile
    # heads. Same opt-in reasoning; `_predict_forest_vec_rm` is not listed
    # because the ordinary multiclass/quantile stages already warm it.
    "chimeraboost.tree._shap_forest_vec",
    # MVS gradient row sampling: only reached when subsample < 1, and the
    # default is 1.0. Warming them would put their compile on the critical
    # path of every startup for a knob most callers never set -- the same
    # reasoning as the SHAP and exact-split kernels above. Measured 0.74 s
    # cold / 0.26 s warm-cache, paid inside the first subsampled fit, which
    # saves about 0.9 s at 200k rows -- so it repays within that same fit.
    "chimeraboost.tree._mvs_lambda_scan",
    "chimeraboost.tree._mvs_weights",
    "chimeraboost.tree._mvs_weights_serial",
    # Exact multi-quantile split search: opt-in via exact_splits=True, a
    # reference arm costing K histogram channels per feature. Same reasoning
    # as SHAP -- most callers never pay for it.
    "chimeraboost.tree._build_split_descend_vec",
}

# Kernels called only from inside another njit kernel, never from Python.
# Whether they carry a signature after warmup() depends on cache state: a
# caller that actually *compiles* registers its inner callees, but a caller
# loaded from a warm cache does not. Listing them in NOT_WARMED (as an older
# version of this file did) made the coverage test fail on the first run
# after any edit to tree.py and pass on the second -- so they are excluded
# from the comparison entirely. A new inner-only kernel still surfaces: on a
# warm cache it appears as uncovered and fails the test until documented here.
CACHE_DEPENDENT = {
    "chimeraboost.tree._solve_small",
    "chimeraboost.tree._leaf_row_index",
    "chimeraboost.tree._lerp_np",
    "chimeraboost.tree._quantile_slice",
    "chimeraboost.tree._select_kth",
    "chimeraboost.tree._sort_pairs",
    "chimeraboost.tree._weighted_quantile_slice",
}

# warmup(shap=True) additionally covers these.
WARMED_BY_SHAP = {
    "chimeraboost.tree._shap_forest_linear",
    "chimeraboost.tree._predict_forest_linear",
    "chimeraboost.tree._shap_forest_vec",
}

# Enumerate every @njit kernel in the package and report which warmup() left
# uncompiled. Run as a subprocess so no earlier test can compile a kernel as a
# side effect and mask a real coverage hole -- which is exactly how the old
# hand-written version of this test passed while kernels went unwarmed.
_PROBE = '''
import importlib, json, pkgutil, sys
from numba.core.dispatcher import Dispatcher
import chimeraboost

def kernels():
    """Deduplicated by defining module + qualname: booster.py re-exports a
    dozen of tree.py's dispatchers, so a raw scan double-counts them."""
    out = {}
    for mod_info in pkgutil.iter_modules(chimeraboost.__path__):
        if mod_info.name.startswith("__"):
            continue            # never import __main__ during enumeration
        mod = importlib.import_module("chimeraboost." + mod_info.name)
        for obj in vars(mod).values():
            if isinstance(obj, Dispatcher):
                f = obj.py_func
                out[f"{f.__module__}.{f.__qualname__}"] = obj
    return out

found = kernels()
chimeraboost.warmup(shap=json.loads(sys.argv[1]))
uncovered = sorted(n for n, d in found.items() if not d.signatures)
print(json.dumps({"total": len(found), "uncovered": uncovered,
                  "file": chimeraboost.__file__}))
'''


def _probe(tmp_path, shap):
    """Run the enumeration probe in a clean interpreter.

    Written to a file rather than passed with ``-c``: quoting is unreliable on
    the Windows box this repo is developed on. PYTHONPATH is pinned to the
    package root pytest itself imported -- the script lives in a tmp dir, so
    without it the child would resolve chimeraboost through the editable
    install and silently test a different checkout.
    """
    script = tmp_path / "kernel_probe.py"
    script.write_text(_PROBE)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(chimeraboost.__file__).parent.parent)
    out = subprocess.run([sys.executable, str(script), json.dumps(bool(shap))],
                         capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["file"] == chimeraboost.__file__, (
        f"probe tested {result['file']}, not {chimeraboost.__file__}")
    return result


def test_warmup_covers_every_kernel_except_the_documented_exclusions(tmp_path):
    result = _probe(tmp_path, shap=False)
    assert result["total"] >= 32, "kernel enumeration found suspiciously few"
    assert set(result["uncovered"]) - CACHE_DEPENDENT == NOT_WARMED


def test_warmup_shap_covers_the_shap_kernels(tmp_path):
    result = _probe(tmp_path, shap=True)
    assert set(result["uncovered"]) - CACHE_DEPENDENT == NOT_WARMED - WARMED_BY_SHAP


def test_serial_predict_twin_is_warmed():
    """The small-batch constant-leaf predictor: a warmed serving process must
    not stall on its first single-row request."""
    warmup()
    from chimeraboost.tree import _predict_forest_rm_serial
    assert _predict_forest_rm_serial.signatures


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


def test_main_runs_and_reports(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "kernels ready in" in out and chimeraboost.__version__ in out


def test_main_quiet_prints_nothing(capsys):
    assert main(["--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_module_entry_points_run():
    """python -m chimeraboost.warmup and python -m chimeraboost -- the only way
    to catch a broken __main__ block or a bad entry-point target."""
    for args in (["-m", "chimeraboost.warmup"], ["-m", "chimeraboost"]):
        out = subprocess.run([sys.executable, *args, "--quiet"],
                             capture_output=True, text=True, timeout=900)
        assert out.returncode == 0, out.stderr


def test_cache_cold_probe_reaches_numba_internals():
    """_cache_is_cold() reads private numba attributes and swallows failures,
    so this asserts the attribute chain still resolves. Without it a numba
    upgrade would silently disable the notice instead of failing here."""
    from chimeraboost.binning import _bin_matrix
    assert _bin_matrix._cache._cache_file._load_index() is not None
    assert isinstance(_cache_is_cold(), bool)


def test_notice_prints_once_when_cold(monkeypatch, capsys):
    w = warmup_module
    monkeypatch.delenv("CHIMERABOOST_NO_NOTICE", raising=False)
    monkeypatch.delenv("CHIMERABOOST_WARMUP", raising=False)
    monkeypatch.setattr(w, "_cache_is_cold", lambda: True)
    monkeypatch.setattr(w, "_NOTICE_DONE", False)

    w._maybe_notice_cold_compile()
    assert capsys.readouterr().err.count("compiling numba kernels") == 1

    w._maybe_notice_cold_compile()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("env", ["CHIMERABOOST_NO_NOTICE", "CHIMERABOOST_WARMUP"])
def test_notice_suppressed_by_env(monkeypatch, capsys, env):
    w = warmup_module
    monkeypatch.setenv(env, "1")
    monkeypatch.setattr(w, "_cache_is_cold", lambda: True)
    monkeypatch.setattr(w, "_NOTICE_DONE", False)
    w._maybe_notice_cold_compile()
    assert capsys.readouterr().err == ""


def test_warm_cache_fit_is_silent(capsys):
    """The regression guard against the notice becoming chatty: a normal warm
    fit must say nothing at all."""
    X = np.random.default_rng(1).standard_normal((80, 3))
    y = (X[:, 0] > 0).astype(int)
    ChimeraBoostClassifier(n_estimators=5, random_state=0).fit(X, y)
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""
