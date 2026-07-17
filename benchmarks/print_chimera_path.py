"""Print which chimeraboost the current environment resolves (A/B trap check).

Run with the same PYTHONPATH as the benchmark to verify worktree resolution:
    PYTHONPATH=<worktree> python benchmarks/print_chimera_path.py
"""
import chimeraboost

print(chimeraboost.__file__)
