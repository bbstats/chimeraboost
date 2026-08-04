"""Guard against config drift in the TabArena tuned search space.

The de-slop pass removed several research flags from the model (e.g. hs_lambda,
onehot_low_card). The tuned wrapper's search space referenced them and would have
crashed every HPO config with TypeError at construction. This test re-derives the
search-space keys straight from the wrapper source (via AST, so it needs neither
autogluon nor tabarena installed) and asserts each is a real constructor parameter
of the ChimeraBoost estimators -- so any future param rename/removal fails here
instead of mid-benchmark.
"""
import ast
import inspect
from pathlib import Path

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

_TABARENA = Path(__file__).resolve().parent.parent / "benchmarks" / "tabarena"
WRAPPER = _TABARENA / "chimeraboost_tabarena_model.py"
# Verbatim mirror of the package that upstream TabArena actually runs. Guarding
# it here means a param rename fails on our own test suite at release time,
# rather than on their cluster after we have asked for a rerun.
UPSTREAM_HPO = _TABARENA / "upstream" / "chimeraboost" / "hpo.py"
UPSTREAM_MODEL = _TABARENA / "upstream" / "chimeraboost" / "model.py"


def _tuned_search_space_keys() -> list[str]:
    """Extract the string keys of the `search_space = {...}` dict literal inside
    get_configs_for_chimera_tuned, without executing the module."""
    tree = ast.parse(WRAPPER.read_text(encoding="utf-8"), filename=str(WRAPPER))
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name == "get_configs_for_chimera_tuned")
    for node in ast.walk(func):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "search_space"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("search_space dict literal not found in wrapper")


def _valid_params() -> set[str]:
    return (set(inspect.signature(ChimeraBoostRegressor.__init__).parameters)
            | set(inspect.signature(ChimeraBoostClassifier.__init__).parameters))


def test_tuned_search_space_keys_are_real_params():
    keys = _tuned_search_space_keys()
    assert keys, "expected a non-empty tuned search space"
    unknown = sorted(set(keys) - _valid_params())
    assert not unknown, (
        f"tuned search space references params absent from the model: {unknown}. "
        "Update get_configs_for_chimera_tuned after a param rename/removal.")


def test_raw_cat_combinations_flag_stays_excluded():
    """Explicit cat_combinations=True bypasses the auto-rule's resource guard and
    can explode on high-cardinality tasks; it must stay out of the search space."""
    assert "cat_combinations" not in _tuned_search_space_keys()


def _upstream_module_level_dict_keys(path: Path, name: str) -> list[str]:
    """String keys of a module-level `name = {...}` dict literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{name} dict literal not found in {path.name}")


def test_upstream_search_space_keys_are_real_params():
    keys = _upstream_module_level_dict_keys(UPSTREAM_HPO, "_SEARCH_SPACE")
    assert keys, "expected a non-empty upstream search space"
    unknown = sorted(set(keys) - _valid_params())
    assert not unknown, (
        f"the search space registered in upstream TabArena references params "
        f"absent from the model: {unknown}. Every HPO config would raise "
        "TypeError on their cluster. Open a PR against autogluon/tabarena and "
        "re-sync benchmarks/tabarena/upstream/.")


def _upstream_wrapper_params() -> set[str]:
    """Constructor params the upstream wrapper passes, however it passes them:
    the `_set_default_params` defaults, any `params["..."] = ...` assignment in
    `_fit`, and the class-level `seed_name` AutoGluon injects the seed into."""
    tree = ast.parse(UPSTREAM_MODEL.read_text(encoding="utf-8"),
                     filename=str(UPSTREAM_MODEL))
    found = set()
    for node in ast.walk(tree):
        # params["thread_count"] = ... / params.get("linear_leaves")
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "params"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            found.add(node.slice.value)
        # seed_name = "random_state"
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "seed_name"
                        for t in node.targets)
                and isinstance(node.value, ast.Constant)):
            found.add(node.value.value)
        # default_params = {...} inside _set_default_params
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "default_params"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            found.update(k.value for k in node.value.keys
                         if isinstance(k, ast.Constant))
    return found


def test_upstream_wrapper_params_are_real_params():
    """The upstream wrapper's own hyperparameters must survive a param rename too
    -- these reach the constructor on every task, not just tuned configs."""
    keys = _upstream_wrapper_params()
    assert {"n_estimators", "early_stopping", "thread_count", "random_state"} <= keys, (
        f"expected the known upstream wrapper params, extracted {sorted(keys)}")
    unknown = sorted(keys - _valid_params())
    assert not unknown, (
        f"the upstream TabArena wrapper passes params absent from the model: "
        f"{unknown}. Every fit on their cluster would fail.")
