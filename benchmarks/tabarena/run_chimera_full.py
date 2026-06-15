"""Run FULL TabArena (all folds/repeats) with the single DEFAULT config.

This is the default-entry counterpart to run_chimera_tuned.py — the run that
produces the leaderboard row for tabarena.ai (the Lite scripts are pre-flight
only). Writes to its OWN output dir so it never collides with the Lite cache.

Usage (from this directory, with the tabarena venv + A: env vars set):
    python run_chimera_full.py --limit 1    # smoke: 1 task, full repetitions
    python run_chimera_full.py              # full 51-task run
Safe to kill and relaunch: each (task, fold, repeat) result is cached; a re-run
skips completed work and refits only what's missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import openml
import pandas as pd
from chimeraboost_tabarena_model import get_configs_for_chimera

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata

OUTPUT_DIR = r"A:\code\tabarena_out\chimera_full"


def _task_ids_from_csv() -> list[int]:
    import tabarena as _ta
    csv = Path(_ta.__file__).parent / "nips2025_utils" / "metadata" / "task_metadata_tabarena51.csv"
    return pd.read_csv(csv)["tid"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N tasks (smoke test); default = all 51")
    args = ap.parse_args()

    openml.config.set_root_cache_directory(r"A:\code\openml")

    task_ids = _task_ids_from_csv()
    if args.limit:
        task_ids = task_ids[: args.limit]

    model_experiments = get_configs_for_chimera(num_random_configs=0)
    print(f"Running {len(model_experiments)} config on {len(task_ids)} tasks "
          f"(repetitions_mode='TabArena' = all folds/repeats)")

    run_experiments_new(
        output_dir=OUTPUT_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena",
        tasks_metadata=load_curated_task_metadata(),
    )


if __name__ == "__main__":
    main()
