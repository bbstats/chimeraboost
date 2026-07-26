"""Measure what a cold numba cache actually costs.

ChimeraBoost's kernels are JIT-compiled on first use and cached on disk. A
fresh machine -- or any machine right after `pip install -U`, which resets the
cache -- pays that compile once, inside whatever call happens to be first.
This script measures it instead of guessing, in three states:

  cold    brand-new NUMBA_CACHE_DIR, fresh process: everything compiles
  warm    same cache dir, fresh process: kernels load from disk
  steady  a second call inside the warm process: no cache work at all

Every measurement runs in a subprocess with NUMBA_CACHE_DIR pointed at a
scratch directory, which is the only honest way to get a stone-cold cache
without deleting files out of the installed package.

    python benchmarks/cold_start.py                # the standard table
    python benchmarks/cold_start.py --kernels      # + per-kernel compile cost
    python benchmarks/cold_start.py --ladder       # + quality=1..5 timings

Numbers in docs/deployment.md come from here; regenerate them rather than
editing the table by hand.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("import", "warmup", "fit-default", "predict-1row")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def make_data(n=3000, seed=0):
    """Mixed numeric + categorical binary task -- the default fit path,
    including the ordered target statistics and cross features."""
    rng = np.random.default_rng(seed)
    Xn = rng.standard_normal((n, 8))
    c1 = rng.integers(0, 40, size=n)      # high-ish cardinality
    c2 = rng.integers(0, 7, size=n)
    X = np.column_stack([Xn, c1, c2])
    y = (Xn[:, 0] + 0.5 * Xn[:, 1] + (c1 % 3) + rng.standard_normal(n) * 0.5 > 1)
    return X, y.astype(int)


# --------------------------------------------------------------------------
# child: one scenario, one process
# --------------------------------------------------------------------------

def run_child(scenario, out_path, kernels=False):
    result = {}
    t0 = time.perf_counter()
    import chimeraboost
    result["import"] = time.perf_counter() - t0
    result["file"] = chimeraboost.__file__
    result["version"] = chimeraboost.__version__

    timers = _install_kernel_timers() if kernels else None

    if scenario == "import":
        result["first"] = result["import"]

    elif scenario == "warmup":
        t0 = time.perf_counter()
        chimeraboost.warmup()
        result["first"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chimeraboost.warmup()
        result["steady"] = time.perf_counter() - t0

    elif scenario == "fit-default":
        from chimeraboost import ChimeraBoostClassifier
        X, y = make_data()
        t0 = time.perf_counter()
        ChimeraBoostClassifier(random_state=0, cat_features=[8, 9]).fit(X, y)
        result["first"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ChimeraBoostClassifier(random_state=0, cat_features=[8, 9]).fit(X, y)
        result["steady"] = time.perf_counter() - t0

    elif scenario == "predict-1row":
        # The serving shape: a process that only unpickles a model and scores
        # single rows never calls fit, so it meets the compile on request one.
        with open(os.environ["CHIMERA_MODEL_PATH"], "rb") as fh:
            model = pickle.load(fh)
        X, _ = make_data()
        t0 = time.perf_counter()
        model.predict_proba(X[:1])
        result["first"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.predict_proba(X[:1])
        result["steady"] = time.perf_counter() - t0

    elif scenario == "ladder":
        from chimeraboost import ChimeraBoostClassifier
        from sklearn.metrics import roc_auc_score
        X, y = make_data(n=4000)
        Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
        rungs = {}
        for q in (1, 2, 3, 4, 5):
            t0 = time.perf_counter()
            m = ChimeraBoostClassifier(random_state=0, cat_features=[8, 9],
                                       quality=q).fit(Xtr, ytr)
            elapsed = time.perf_counter() - t0
            rungs[q] = {"seconds": elapsed,
                        "auc": float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))}
        result["rungs"] = rungs
        result["first"] = sum(r["seconds"] for r in rungs.values())

    else:
        raise SystemExit(f"unknown scenario: {scenario}")

    if timers is not None:
        result["kernels"] = _collect_kernel_timers(timers)

    Path(out_path).write_text(json.dumps(result))


def _install_kernel_timers():
    """Time every kernel's compile by wrapping its dispatcher.

    `_compile_for_args` is numba's entry point for compiling a new
    specialization, so wrapping it attributes wall time per kernel. Cache
    *loads* go through a different path and are correctly excluded.
    """
    import importlib
    from numba.core.dispatcher import Dispatcher
    import chimeraboost
    import pkgutil

    timings = []
    for mod_info in pkgutil.iter_modules(chimeraboost.__path__):
        if mod_info.name.startswith("__"):
            continue
        mod = importlib.import_module("chimeraboost." + mod_info.name)
        for obj in vars(mod).values():
            if not isinstance(obj, Dispatcher):
                continue
            if getattr(obj, "_chimera_timed", False):
                continue          # booster.py re-exports tree.py's kernels
            obj._chimera_timed = True
            name = obj.py_func.__name__

            def wrap(dispatcher=obj, name=name):
                original = dispatcher._compile_for_args

                def timed(*args, **kwargs):
                    t0 = time.perf_counter()
                    out = original(*args, **kwargs)
                    timings.append((time.perf_counter() - t0, name))
                    return out
                return timed

            obj._compile_for_args = wrap()
    return timings


def _collect_kernel_timers(timings):
    agg = {}
    for seconds, name in timings:
        agg[name] = agg.get(name, 0.0) + seconds
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
# parent: drive the subprocesses
# --------------------------------------------------------------------------

def _launch(scenario, cache_dir, kernels=False, model_path=None):
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    # Pin the package under test: the child must not resolve chimeraboost
    # through an editable install pointing at some other checkout.
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("CHIMERABOOST_WARMUP", None)
    env["CHIMERABOOST_NO_NOTICE"] = "1"
    if model_path:
        env["CHIMERA_MODEL_PATH"] = str(model_path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out_path = fh.name
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--child", scenario, "--out", out_path]
    if kernels:
        cmd.append("--kernels")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"child '{scenario}' failed:\n{proc.stderr}")
    # Read results from the file, not stdout: terminal output is unreliable
    # on this box, the file is not.
    data = json.loads(Path(out_path).read_text())
    os.unlink(out_path)
    return data


def _build_model(tmp_root):
    """Fit one model in this process and pickle it, so the predict scenario
    can measure a fresh process that never fits."""
    from chimeraboost import ChimeraBoostClassifier
    X, y = make_data()
    model = ChimeraBoostClassifier(random_state=0, cat_features=[8, 9]).fit(X, y)
    path = tmp_root / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def _fmt(seconds):
    if seconds is None:
        return "-"
    if seconds < 0.01:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--kernels", action="store_true",
                        help="also report per-kernel cold compile cost")
    parser.add_argument("--ladder", action="store_true",
                        help="also time quality=1..5 on a cold and warm cache")
    args = parser.parse_args(argv)

    if args.child:
        return run_child(args.child, args.out, kernels=args.kernels) or 0

    scenarios = list(SCENARIOS) + (["ladder"] if args.ladder else [])
    tmp_root = Path(tempfile.mkdtemp(prefix="chimera_coldstart_"))
    try:
        model_path = _build_model(tmp_root)
        rows, kernel_costs, meta = [], None, None
        for scenario in scenarios:
            cache_dir = tmp_root / f"cache_{scenario}"
            cache_dir.mkdir()
            want_kernels = args.kernels and scenario == "warmup"
            cold = _launch(scenario, cache_dir, kernels=want_kernels,
                           model_path=model_path)
            warm = _launch(scenario, cache_dir, model_path=model_path)
            meta = meta or cold
            if want_kernels:
                kernel_costs = cold.get("kernels")
            rows.append((scenario, cold, warm))

        _print_table(rows, meta, kernel_costs,
                     ladder=[r for r in rows if r[0] == "ladder"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


def _print_table(rows, meta, kernel_costs, ladder):
    import numba
    print()
    print(f"chimeraboost {meta['version']}  from {meta['file']}")
    print(f"numba {numba.__version__}  numpy {np.__version__}  "
          f"{numba.config.NUMBA_NUM_THREADS} threads")
    print()
    print(f"{'scenario':<16}{'cold':>10}{'warm':>10}{'steady':>10}"
          f"{'cold/warm':>12}")
    print("-" * 58)
    for scenario, cold, warm in rows:
        c, w = cold.get("first"), warm.get("first")
        ratio = f"{c / w:.1f}x" if c and w else "-"
        print(f"{scenario:<16}{_fmt(c):>10}{_fmt(w):>10}"
              f"{_fmt(warm.get('steady')):>10}{ratio:>12}")
    print()

    if kernel_costs:
        total = sum(kernel_costs.values())
        print(f"cold compile by kernel ({total:.2f} s over "
              f"{len(kernel_costs)} kernels)")
        print("-" * 40)
        for name, seconds in list(kernel_costs.items())[:12]:
            print(f"  {name:<32}{seconds:6.2f}")
        rest = list(kernel_costs.items())[12:]
        if rest:
            print(f"  {'(%d more)' % len(rest):<32}{sum(s for _, s in rest):6.2f}")
        print()

    for scenario, cold, warm in ladder:
        print(f"{'quality':<10}{'cold s':>10}{'warm s':>10}{'AUC':>10}")
        print("-" * 40)
        for q in ("1", "2", "3", "4", "5"):
            c = cold["rungs"].get(q, {})
            w = warm["rungs"].get(q, {})
            print(f"{q:<10}{c.get('seconds', 0):10.2f}{w.get('seconds', 0):10.2f}"
                  f"{c.get('auc', 0):10.4f}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
