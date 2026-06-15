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

WRAPPER = (Path(__file__).resolve().parent.parent
           / "benchmarks" / "tabarena" / "chimeraboost_tabarena_model.py")


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
