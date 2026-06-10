"""Run TabArena-Lite for the custom ChimeraBoost model.

Usage (from this directory, with the tabarena venv):
    python run_chimera_lite.py --limit 2     # smoke test on 2 tasks
    python run_chimera_lite.py               # full 51-task TabArena-Lite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import openml
import pandas as pd
from chimeraboost_tabarena_model import get_configs_for_chimera

from tabarena.benchmark.experiment import run_experiments_new

OUTPUT_DIR = r"A:\code\tabarena_out\chimera"

# Local task metadata ships with the repo — use it instead of the live API
# (openml.study.get_suite("tabarena-v0.1") 503/502s transiently).
def _task_ids_from_csv() -> list[int]:
    import tabarena as _ta
    csv = Path(_ta.__file__).parent / "nips2025_utils" / "metadata" / "task_metadata_tabarena51.csv"
    return pd.read_csv(csv)["tid"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N tasks (smoke test); default = all 51")
    args = ap.parse_args()

    # Keep the OpenML dataset cache off C: (only ~4 GB free there).
    openml.config.set_root_cache_directory(r"A:\code\openml")

    task_ids = _task_ids_from_csv()
    if args.limit:
        task_ids = task_ids[: args.limit]

    model_experiments = get_configs_for_chimera(num_random_configs=0)

    run_experiments_new(
        output_dir=OUTPUT_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )


if __name__ == "__main__":
    main()
