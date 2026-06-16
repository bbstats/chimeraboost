"""TabArena/AutoGluon default model unit test for ChimeraBoostModel.

This is the Part-A integration gate ("must pass the default unit test for
TabArena models"): AutoGluon's FitHelper loads a toy dataset, fits the model
through a TabularPredictor, and validates fit -> predict -> refit_full ->
save/load for each problem type.

Run in the tabarena venv from this directory:
    & A:\\code\\tabarena\\.venv\\Scripts\\python.exe model_unittest.py
Not named test_*.py on purpose: it needs autogluon, which the base-env pytest
suite (tests/) does not have.
"""
from __future__ import annotations

import os

# Windows shim: AutoGluon/tabarena call os.sched_getaffinity (Linux-only).
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))

from autogluon.tabular.testing import FitHelper

from chimeraboost_tabarena_model import ChimeraBoostModel

DATASETS = ["toy_binary", "toy_multiclass", "toy_regression"]


def main() -> int:
    model_cls = ChimeraBoostModel
    fit_args = dict(hyperparameters={model_cls: {}})
    failures = []
    for ds in DATASETS:
        try:
            FitHelper.fit_and_validate_dataset(
                dataset_name=ds,
                fit_args=fit_args,
                refit_full=True,
                raise_on_model_failure=True,
            )
            print(f"PASS: {ds}")
        except Exception as e:  # noqa: BLE001 - report-all harness
            failures.append((ds, repr(e)))
            print(f"FAIL: {ds} -> {e!r}")
    print("=" * 60)
    if failures:
        print(f"UNIT TEST FAILED ({len(failures)}/{len(DATASETS)}):")
        for ds, err in failures:
            print(f"  {ds}: {err}")
        return 1
    print(f"UNIT TEST PASSED ({len(DATASETS)}/{len(DATASETS)} problem types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
