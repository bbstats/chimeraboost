"""Evaluate ChimeraBoost_tuned (Lite) against the TabArena leaderboard -> Elo.

Run AFTER run_chimera_tuned_lite.py has produced result artifacts.
"""
from __future__ import annotations

import os
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))

from tabarena.nips2025_utils.end_to_end_single import (
    EndToEndResultsSingle,
    EndToEndSingle,
)
from tabarena.website.website_format import format_leaderboard

PATH_RAW = r"A:\code\tabarena_out\chimera_tuned_lite"
FIG_OUTPUT_DIR = r"A:\code\tabarena_out\evals\chimera_tuned_lite"
METHOD = "ChimeraBoost_tuned"

if __name__ == "__main__":
    end_to_end = EndToEndSingle.from_path_raw(path_raw=PATH_RAW)
    _ = end_to_end.to_results()

    end_to_end_results = EndToEndResultsSingle.from_cache(method=METHOD)
    leaderboard = end_to_end_results.compare_on_tabarena(
        only_valid_tasks=True, output_dir=FIG_OUTPUT_DIR
    )
    leaderboard_website = format_leaderboard(leaderboard)
    md = leaderboard_website.to_markdown(index=False)
    out_md = r"A:\code\tabarena_out\evals\chimera_tuned_lite_leaderboard.md"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    leaderboard_website.to_csv(
        r"A:\code\tabarena_out\evals\chimera_tuned_lite_leaderboard.csv", index=False
    )
    print(f"leaderboard written to {out_md}")
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
