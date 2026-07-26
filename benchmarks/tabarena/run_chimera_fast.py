"""Run TabArena-Lite for ChimeraBoost_fast (quality=1, the fast rung).

All 51 tasks are already cached locally, so this is pure offline compute.
Cheaper than the default run: quality=1 fits one booster instead of the
default's two-to-four (benchmarks/SELECT_PLAN.md).
"""
from __future__ import annotations

from pathlib import Path

import openml
import pandas as pd
from chimeraboost_tabarena_model import get_configs_for_chimera_fast

from tabarena.benchmark.experiment import run_experiments_new

OUTPUT_DIR = r"A:\code\tabarena_out\chimera_fast"


def _task_ids_from_csv() -> list[int]:
    import tabarena as _ta
    csv = Path(_ta.__file__).parent / "nips2025_utils" / "metadata" / "task_metadata_tabarena51.csv"
    return pd.read_csv(csv)["tid"].tolist()


def main():
    openml.config.set_root_cache_directory(r"A:\code\openml")
    task_ids = _task_ids_from_csv()
    run_experiments_new(
        output_dir=OUTPUT_DIR,
        model_experiments=get_configs_for_chimera_fast(),
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )


if __name__ == "__main__":
    main()
