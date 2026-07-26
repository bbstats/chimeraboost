"""Audit tool for the sealed public suite (issue #37, benchmarks/PUBLIC_PLAN.md).

Two jobs, matching the plan's two gates:

  shortlist  Step 0 + step 1. Sweep the OpenML catalogue for datasets big enough
             to carry a speed axis, drop everything that touches a sealed
             holdout or a decision suite, and rank what survives by regime.

  verify     Step 1's in-session confirmation, for the properties the metadata
             API cannot be trusted for anyway: row count, column names, dtypes,
             per-column cardinality, missingness and whether a column really
             orders as time under the harness's own _time_sort_key. Read from
             the dataset's actual bytes, fetched from data.openml.org.

Usage:
  python benchmarks/public_audit.py shortlist [--min-rows 50000] [--markdown]
  python benchmarks/public_audit.py shortlist --refresh      # re-harvest first
  python benchmarks/public_audit.py verify 41214 45647 ...   # by OpenML id

WHAT THIS CANNOT CONFIRM WITHOUT THE METADATA API: the default target column,
the dataset's current version, and its active/deactivated status. data.openml.org
serves the data files only. `verify` prints an explicit reminder; the frozen list
must not ship until those three are re-checked against a live
www.openml.org/api/v1/json/data/<id> response.

The catalogue read by `shortlist` is benchmarks/data_cache/openml_meta.json,
harvested from the live listing API by benchmarks/synthgen/harvest_metadata.py.
It is a recorded API response, not remembered ids -- but it has a date on it, so
`shortlist` prints that date and `--refresh` re-fetches when the API is up.

SEALED SUITE. Selection is by data properties only. No model is fit here and no
benchmark result may enter the decision, or the suite is born cherry-picked.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import suite_overlap  # noqa: E402

CACHE_PATH = os.path.join(_HERE, "data_cache", "openml_meta.json")
PQ_CACHE = os.environ.get("PUBLIC_AUDIT_CACHE") or r"A:\code\openml_pq"
PQ_URL = "https://data.openml.org/datasets/{pad:04d}/{did}/dataset_{did}.pq"

# Datasets already consumed by an existing suite, spelled out in PUBLIC_PLAN.md.
# Belt-and-braces on top of the registry pools: Grinsztajn and PMLB carry no
# OpenML ids, so a set can hide there under a different name.
CONSUMED = [
    "covertype", "Higgs", "Allstate_Claims_Severity", "road-safety",
    "nyc-taxi-green-dec-2016", "diamonds", "house_sales", "Airlines_DepDelay_1M",
    "MiniBooNE", "jannis", "albert", "delays_zurich_transport", "medical_charges",
    "particulate-matter-ukair-2017", "seattlecrime6", "SGEMM_GPU_kernel_performance",
    "Diabetes130US", "superconduct", "APSFailure", "kddcup09_appetency", "adult",
    "electricity", "letter",
    # Verified aliases the normalised-name matcher cannot join: uci_diabetes_p
    # (42106) is Diabetes130US, same 101,766 rows, and Diabetes130US is in
    # Grinsztajn, in TabArena-51 and above.
    "uci_diabetes_p",
]

# Auto-generated families (criterion 2: real data only) and the streaming
# generators that OpenML hosts thousands of near-identical draws from.
JUNK_PREFIXES = (
    "BNG(", "fri_c", "QSAR-TID-", "autoUniv-", "GAMETES_", "RandomRBF", "SEA(",
    "AGRAWAL", "Agrawal", "LED(", "Hyperplane", "STAGGER", "stagger",
    "simulated_",
)
# Flattened image/text corpora: nominally tabular, actually pixels or embeddings.
# They would make the suite measure something it does not claim to measure.
VECTOR_PREFIXES = (
    "mnist", "MNIST", "CIFAR", "Fashion", "Devnagari", "Kuzushiji", "GTSRB",
    "SVHN", "emnist", "EMNIST", "Meta_Album", "Stylized_Meta_Album", "Afro_",
    "Kannada", "imagenet", "tiny", "dgf_",
)
MAX_FEATURES = 200   # above this it is a vector, not a table
MIN_FEATURES = 5


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
def load_catalogue(refresh=False):
    if refresh:
        sys.path.insert(0, os.path.join(_HERE, "synthgen"))
        import harvest_metadata
        entries = harvest_metadata.fetch_listing()
        tab_ids = harvest_metadata.fetch_study_data_ids(
            harvest_metadata.TABARENA_STUDY_ID)
        blob = {"fetched_utc": "refreshed", "entries": entries,
                "tabarena_ids": sorted(tab_ids), "cc18_ids": []}
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        return blob
    if not os.path.exists(CACHE_PATH):
        raise SystemExit(
            f"no catalogue at {CACHE_PATH}\n"
            "Run with --refresh (needs the OpenML API), or copy the cache in "
            "from the main checkout -- a fresh worktree has an empty data_cache/.")
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _qualities(entry):
    return {q["name"]: q.get("value") for q in entry.get("quality", [])}


def _num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _row(entry):
    """Flatten one catalogue entry to the fields selection cares about."""
    q = _qualities(entry)
    n = _num(q.get("NumberOfInstances"))
    if n is None:
        return None
    cls = int(_num(q.get("NumberOfClasses")) or 0)
    maj, mino = _num(q.get("MajorityClassSize")), _num(q.get("MinorityClassSize"))
    return dict(
        did=int(entry["did"]), name=entry["name"], version=int(entry.get("version", 0)),
        n=int(n),
        d=int(_num(q.get("NumberOfFeatures")) or 0),
        sym=int(_num(q.get("NumberOfSymbolicFeatures")) or 0),
        card=int(_num(q.get("MaxNominalAttDistinctValues")) or 0),
        miss=int(_num(q.get("NumberOfMissingValues")) or 0),
        cls=cls,
        imbalance=(mino / maj if maj and mino else None),
        task=("regression" if cls == 0 else "binary" if cls == 2 else "multiclass"),
    )


def shortlist(cat, min_rows=50000):
    """Candidates surviving every exclusion, plus a per-stage cut tally."""
    pools = suite_overlap.exclusion_pools(
        include_hc=True, extra=[("consumed", set(), CONSUMED)])
    tab_ids = {int(i) for i in cat.get("tabarena_ids", [])}

    cuts = dict(size=0, junk=0, vector=0, width=0, tabarena_id=0, overlap=0)
    kept = []
    for entry in cat["entries"]:
        r = _row(entry)
        if r is None or r["n"] < min_rows:
            cuts["size"] += 1
            continue
        if r["name"].startswith(JUNK_PREFIXES):
            cuts["junk"] += 1
            continue
        if r["name"].startswith(VECTOR_PREFIXES):
            cuts["vector"] += 1
            continue
        if not MIN_FEATURES <= r["d"] <= MAX_FEATURES:
            cuts["width"] += 1
            continue
        if r["did"] in tab_ids:
            cuts["tabarena_id"] += 1
            continue
        fails = suite_overlap.overlap_failures(r["name"], r["did"], pools)
        if fails:
            cuts["overlap"] += 1
            r["excluded_by"] = fails[0].split(": ", 1)[1]
            continue
        kept.append(r)

    # One row per dataset: OpenML lists every version separately.
    best = {}
    for r in kept:
        k = suite_overlap.norm(r["name"])
        if k not in best or r["version"] > best[k]["version"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: -r["n"]), cuts


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
_HDR = f"{'did':>6}  {'name':40} {'n':>9} {'d':>4} {'sym':>4} {'maxcard':>8} {'miss':>10} {'cls':>4}  imb"


def _fmt(r):
    imb = f"{r['imbalance']:.3f}" if r["imbalance"] is not None else "    -"
    return (f"{r['did']:>6}  {r['name'][:40]:40} {r['n']:>9} {r['d']:>4} "
            f"{r['sym']:>4} {r['card']:>8} {r['miss']:>10} {r['cls']:>4}  {imb}")


def report(rows, cuts, cat, min_rows, markdown=False, top=25):
    print(f"catalogue: {len(cat['entries'])} datasets, harvested {cat['fetched_utc']}")
    print(f"cuts: {cuts}")
    print(f"survivors at n >= {min_rows}, {MIN_FEATURES}-{MAX_FEATURES} features: "
          f"{len(rows)}\n")

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    print("task mix (from NumberOfClasses -- UNRELIABLE, see below): "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_task.items())))
    print("  NumberOfClasses is 0 both for genuine regression and for any dataset\n"
          "  with no default target set. A dataset with no default target CRASHES\n"
          "  the builder at ds.target.name -- confirm the target before freezing.\n")

    order = ("binary", "multiclass", "regression")
    for task in order:
        sel = by_task.get(task, [])
        if not sel:
            continue
        # Rank by regime interest: high-cardinality first, then symbolic width,
        # then missingness, then size.
        sel = sorted(sel, key=lambda r: (-r["card"], -r["sym"], -r["miss"], -r["n"]))
        print(f"--- {task}: {len(sel)} candidates (top {min(top, len(sel))} by regime) ---")
        print(_HDR)
        for r in sel[:top]:
            print(_fmt(r))
        print()

    if markdown:
        print("\n--- markdown (paste into PUBLIC_PLAN.md) ---")
        print("| candidate | id | task | n | d | n_cat | max card | missing | imbalance |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows[:top]:
            imb = f"{r['imbalance']:.3f}" if r["imbalance"] is not None else "-"
            print(f"| {r['name']} | {r['did']} | {r['task']} | {r['n']} | {r['d']} "
                  f"| {r['sym']} | {r['card']} | {r['miss']} | {imb} |")


# --------------------------------------------------------------------------
# content verification (reads the real bytes)
# --------------------------------------------------------------------------
def fetch_parquet(did):
    """Download dataset <did>'s parquet into the A: cache; return the path."""
    os.makedirs(PQ_CACHE, exist_ok=True)
    path = os.path.join(PQ_CACHE, f"dataset_{did}.pq")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = PQ_URL.format(pad=did // 10000, did=did)
    req = urllib.request.Request(
        url, headers={"User-Agent": "chimeraboost-benchmarks/public-audit"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


_TIME_NAME = re.compile(
    r"(^|_)(year|yr|date|datetime|time|timestamp|ts|month|day|week|quarter|epoch|"
    r"created|launched|deadline|start|end|period|season)($|_)", re.I)


def _temporal_candidate(col, s, n, rb, time_frac=0.9):
    """(reason, distinct) if `col` could serve as a time_col, else None.

    _time_sort_key alone is far too weak a test: it coerces numerics first, so
    it accepts EVERY numeric column. A column qualifies here only if it is
    datetime-typed, or parses as a date from strings, or is numeric with a
    time-shaped NAME -- and it must still pass _time_sort_key, which is the gate
    the harness actually applies.
    """
    import pandas as pd
    if rb._time_sort_key(s) is None:
        return None
    nun = s.nunique(dropna=True)
    if nun < 2:
        return None
    if pd.api.types.is_datetime64_any_dtype(s):
        return ("datetime dtype", nun)
    named = bool(_TIME_NAME.search(str(col)))
    if pd.api.types.is_numeric_dtype(s):
        return ("numeric, time-shaped name", nun) if named else None
    try:
        dt = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    if float(dt.notna().mean()) >= time_frac:
        return (f"parses as date ({100 * dt.notna().mean():.0f}%)", nun)
    return ("time-shaped name", nun) if named else None


def verify(did):
    """Print what the bytes say about dataset <did>."""
    import pandas as pd
    import run_benchmarks as rb

    path = fetch_parquet(did)
    df = pd.read_parquet(path)
    n = len(df)
    print(f"\n=== did {did} — {n} rows x {df.shape[1]} columns "
          f"({os.path.getsize(path) / 1e6:.1f} MB on disk) ===")
    if n > rb._PUBLIC_MAX_ROWS:
        print(f"    (harness subsamples to {rb._PUBLIC_MAX_ROWS} at random_state=0)")

    print(f"{'column':38} {'dtype':16} {'nunique':>9} {'miss%':>7}  note")
    time_ok, hicard, dropped = [], [], []
    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        notes = []
        is_cat = rb._is_categorical_dtype(s.dtype)
        if is_cat and nun > rb._HIGHCARD_ID_FRAC * n:
            notes.append(f"DROPPED by builder (>{rb._HIGHCARD_ID_FRAC} unique)")
            dropped.append(col)
        elif is_cat and nun >= 50:
            notes.append("high-card cat")
            hicard.append((col, nun))
        cand = _temporal_candidate(col, s, n, rb)
        if cand:
            notes.append(f"time candidate: {cand[0]}")
            time_ok.append((col, cand[0], nun))
        print(f"{str(col)[:38]:38} {str(s.dtype)[:16]:16} {nun:>9} "
              f"{100.0 * s.isna().mean():>6.1f}%  " + "; ".join(notes))

    n_cat = sum(1 for c in df.columns if rb._is_categorical_dtype(df[c].dtype))
    print(f"\n  categorical columns: {n_cat}  "
          f"(high-card >=50 levels: {len(hicard)}"
          + (f" — max {max(hicard, key=lambda t: t[1])}" if hicard else "")
          + f"; builder would drop {len(dropped)})")
    print("  time_col candidates: "
          + (", ".join(f"{c} [{why}]" for c, why, _ in time_ok) if time_ok else "NONE"))
    print("  target column: NOT DETERMINED — data.openml.org serves data only.\n"
          "  Confirm the default target, version and active status against the\n"
          "  metadata API before this id may be frozen into PUBLIC_DATASETS.")


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shortlist", help="sweep the catalogue and apply the overlap gate")
    s.add_argument("--min-rows", type=int, default=50000)
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--markdown", action="store_true")
    s.add_argument("--refresh", action="store_true",
                   help="re-harvest the listing from the OpenML API first")

    v = sub.add_parser("verify", help="content-check finalists from their parquet files")
    v.add_argument("ids", nargs="+", type=int)

    args = ap.parse_args(argv)
    if args.cmd == "shortlist":
        cat = load_catalogue(refresh=args.refresh)
        rows, cuts = shortlist(cat, min_rows=args.min_rows)
        report(rows, cuts, cat, args.min_rows, markdown=args.markdown, top=args.top)
    else:
        for did in args.ids:
            verify(did)


if __name__ == "__main__":
    main()
