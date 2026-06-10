"""Run FULL TabArena (all folds/repeats) with 200 random HP configs (tuned entry).

Usage (from this directory, with the tabarena venv + A: env vars set):
    python run_chimera_tuned.py --limit 2    # smoke test: 2 tasks
    python run_chimera_tuned.py              # full 51-task run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import openml
import pandas as pd
from chimeraboost_tabarena_model import get_configs_for_chimera_tuned

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata

OUTPUT_DIR = r"A:\code\tabarena_out\chimera_tuned"


def _task_ids_from_csv() -> list[int]:
    import tabarena as _ta
    csv = Path(_ta.__file__).parent / "nips2025_utils" / "metadata" / "task_metadata_tabarena51.csv"
    return pd.read_csv(csv)["tid"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N tasks (smoke test); default = all 51")
    ap.add_argument("--num-random-configs", type=int, default=200,
                    help="number of random HP configs (default 200)")
    args = ap.parse_args()

    openml.config.set_root_cache_directory(r"A:\code\openml")

    task_ids = _task_ids_from_csv()
    if args.limit:
        task_ids = task_ids[: args.limit]

    tasks_metadata = load_curated_task_metadata()

    model_experiments = get_configs_for_chimera_tuned(num_random_configs=args.num_random_configs)
    print(f"Running {len(model_experiments)} model configs on {len(task_ids)} tasks "
          f"(repetitions_mode='TabArena' = all folds/repeats)")

    run_experiments_new(
        output_dir=OUTPUT_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena",
        tasks_metadata=tasks_metadata,
    )


if __name__ == "__main__":
    main()
