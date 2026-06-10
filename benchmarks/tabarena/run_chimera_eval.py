"""Evaluate ChimeraBoost against the TabArena(-Lite) leaderboard -> Elo.

Run AFTER run_chimera_lite.py has produced result artifacts in OUTPUT_DIR.
Mirrors the official run_evaluate_model.py.
"""
from __future__ import annotations

# Windows shim: TabArena calls os.sched_getaffinity (Linux-only) to count CPUs.
import os
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))

from tabarena.nips2025_utils.end_to_end_single import (
    EndToEndResultsSingle,
    EndToEndSingle,
)
from tabarena.website.website_format import format_leaderboard

PATH_RAW = r"A:\code\tabarena_out\chimera"
FIG_OUTPUT_DIR = r"A:\code\tabarena_out\evals\chimera"
METHOD = "ChimeraBoost"  # == ChimeraBoostModel.ag_name

if __name__ == "__main__":
    # 1. Load raw artifacts, process + cache results for our method.
    end_to_end = EndToEndSingle.from_path_raw(path_raw=PATH_RAW)
    _ = end_to_end.to_results()

    # 2. Compare against the TabArena baseline pool -> leaderboard with Elo.
    end_to_end_results = EndToEndResultsSingle.from_cache(method=METHOD)
    leaderboard = end_to_end_results.compare_on_tabarena(
        only_valid_tasks=True, output_dir=FIG_OUTPUT_DIR
    )
    leaderboard_website = format_leaderboard(leaderboard)
    md = leaderboard_website.to_markdown(index=False)
    out_md = r"A:\code\tabarena_out\evals\chimera_leaderboard.md"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    # Save CSV too for easy ChimeraBoost-row extraction.
    leaderboard_website.to_csv(r"A:\code\tabarena_out\evals\chimera_leaderboard.csv", index=False)
    print(f"leaderboard written to {out_md}")
    # Console print guarded against Windows charmap codec.
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
