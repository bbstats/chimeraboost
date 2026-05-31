"""Status reporter for the /bench command.

Designed to be glanced at from a phone. Reports, in order of preference:

  1. If a run is in progress (a `.progress` sidecar with status "running" that
     is still being updated): a one-screen progress block -- percent done,
     tasks completed, elapsed, ETA, and the run's config. If that run has
     produced partial results we can't show them (results json is written only
     at the end), so we just show progress.

  2. Otherwise: the full aggregate table for the most recent completed results
     json (via summarize.format_table).

"Running" is decided by freshness: a progress file whose `status` is "running"
and whose file mtime is within STALE_S seconds is treated as live; older ones
are assumed to be from a crashed/killed run and ignored in favor of the latest
results table.

CLI:
    python benchmarks/bench_status.py
"""
import glob
import os
import time

import summarize

RESULTS_DIR = summarize.RESULTS_DIR
# A running progress file not touched in this many seconds is considered stale
# (the run probably died). One dataset-seed draw can take a couple minutes on
# the big datasets, so give it generous headroom.
STALE_S = 600


def _fmt_dur(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _latest_progress():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.progress"))
    return max(files, key=os.path.getmtime) if files else None


def _load_json(path):
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _progress_block(prog, path):
    pct = prog.get("pct", 0)
    done = prog.get("completed", 0)
    total = prog.get("total", 0)
    bar_n = 24
    filled = int(round(bar_n * pct / 100.0))
    bar = "#" * filled + "." * (bar_n - filled)
    lines = [
        f"BENCHMARK RUNNING  [{bar}] {pct:.0f}%",
        f"  tasks     {done}/{total} dataset-seed draws",
        f"  elapsed   {_fmt_dur(prog.get('elapsed_s'))}"
        f"   eta ~{_fmt_dur(prog.get('eta_s'))}",
        f"  started   {prog.get('started', '?')}",
        f"  config    {prog.get('config', '?')}",
        f"  models    {', '.join(prog.get('models', []))}",
    ]
    return "\n".join(lines)


def report():
    prog_path = _latest_progress()
    if prog_path:
        prog = _load_json(prog_path)
        age = time.time() - os.path.getmtime(prog_path)
        if prog and prog.get("status") == "running" and age < STALE_S:
            return _progress_block(prog, prog_path)

    latest = summarize.latest_json()
    if not latest:
        return "No benchmark results yet, and no run in progress."
    header = f"# latest results: {os.path.basename(latest)}"
    if prog_path:
        prog = _load_json(prog_path)
        if prog and prog.get("status") == "running":
            mins = (time.time() - os.path.getmtime(prog_path)) / 60.0
            header += (f"\n# (a run was in progress but its tracker is stale "
                       f"~{mins:.0f}m -- it may have died; showing last results)")
    return header + "\n" + summarize.format_table(summarize.load(latest))


if __name__ == "__main__":
    print(report())
