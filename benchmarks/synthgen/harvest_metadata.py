"""Harvest public tabular-dataset metadata into the synthgen calibration corpus.

One-time (re-runnable) script. Pages the OpenML listing API for every active
dataset's qualities, cleans out auto-generated pollution, EXCLUDES TabArena's
member datasets (sealed holdout -- its ID list is fetched solely to drop those
rows), tags membership in curated suites (OpenML-CC18 + the in-repo OpenML /
Grinsztajn registries), and distills a compact marginals snapshot that
`synthgen.calibration` bootstraps from at generation time.

Outputs:
  benchmarks/data_cache/openml_meta.json      raw listing cache (gitignored, ~tens of MB)
  benchmarks/synthgen/corpus_marginals.json   checked-in snapshot (<= ~2500 rows)

Usage:
  python benchmarks/synthgen/harvest_metadata.py [--refresh] [--max-rows 2500]

Network etiquette: 1 request/s, 3 retries with backoff. The raw cache makes
re-distillation offline; --refresh forces a refetch.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _BENCH)

API = "https://www.openml.org/api/v1/json"
CACHE_PATH = os.path.join(_BENCH, "data_cache", "openml_meta.json")
SNAPSHOT_PATH = os.path.join(_HERE, "corpus_marginals.json")

# TabArena-v0.1 = OpenML study 457 (task-type suite). Fetched ONLY to exclude
# its member datasets from the calibration corpus (sealed-holdout rule).
TABARENA_STUDY_ID = 457
CC18_STUDY_ID = 99

# Auto-generated dataset families that would swamp the marginals (thousands of
# near-identical synthetic entries on OpenML).
_JUNK_PREFIXES = ("BNG(", "fri_c", "QSAR-TID-", "autoUniv-", "GAMETES_")

_PAGE = 1000


def _get_json(url, tries=3, backoff=2.0):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "chimeraboost-benchmarks/synthgen-harvest"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry everything, report last
            last = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url}: {last}")


def fetch_listing():
    """Page the full active-dataset listing. Returns list of raw entries."""
    out, offset = [], 0
    while True:
        url = f"{API}/data/list/status/active/limit/{_PAGE}/offset/{offset}"
        try:
            page = _get_json(url)
        except RuntimeError:
            if offset == 0:
                raise
            break  # past the end (API errors instead of returning empty)
        entries = page.get("data", {}).get("dataset", [])
        if not entries:
            break
        out.extend(entries)
        print(f"  listing: {len(out)} datasets so far (offset {offset})", flush=True)
        if len(entries) < _PAGE:
            break
        offset += _PAGE
        time.sleep(1.0)
    return out


def fetch_study_data_ids(study_id):
    """Dataset ids of an OpenML study/suite (best-effort; [] on failure)."""
    try:
        study = _get_json(f"{API}/study/{study_id}").get("study", {})
        ids = study.get("data", {}).get("data_id", [])
        return {int(i) for i in ids}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: study {study_id} fetch failed ({exc}); continuing without it",
              flush=True)
        return set()


def _qualities(entry):
    return {q["name"]: q.get("value") for q in entry.get("quality", [])}


def _to_num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def distill(entries, tabarena_ids, cc18_ids, curated_names, curated_ids):
    """Clean + reduce raw listing entries to corpus rows.

    Row: [n, d, task, cat_frac, n_classes, missing_rate, majority_frac,
          max_card, curated]  (task: 0=reg, 1=binary, 2=multiclass)
    """
    by_name = {}
    dropped = {"junk": 0, "tabarena": 0, "qualities": 0, "size": 0, "classes": 0}
    for e in entries:
        name = e.get("name", "")
        did = int(e.get("did", -1))
        if any(name.startswith(p) for p in _JUNK_PREFIXES):
            dropped["junk"] += 1
            continue
        if did in tabarena_ids or name.lower() in _TABARENA_NAME_FALLBACK:
            dropped["tabarena"] += 1
            continue
        q = _qualities(e)
        n = _to_num(q.get("NumberOfInstances"))
        d_total = _to_num(q.get("NumberOfFeatures"))
        if n is None or d_total is None:
            dropped["qualities"] += 1
            continue
        n, d_total = int(n), int(d_total)
        d = d_total - 1  # qualities count the target column
        if not (500 <= n <= 10_000_000 and 3 <= d <= 5000):
            dropped["size"] += 1
            continue
        classes = _to_num(q.get("NumberOfClasses"))
        if classes is None:
            dropped["qualities"] += 1
            continue
        classes = int(classes)
        if classes == 0:
            task = 0
        elif classes == 2:
            task = 1
        elif 3 <= classes <= 50:
            task = 2
        else:
            dropped["classes"] += 1
            continue
        symbolic = _to_num(q.get("NumberOfSymbolicFeatures")) or 0.0
        n_cat = max(0, int(symbolic) - (1 if task else 0))  # target is symbolic for clf
        cat_frac = round(min(1.0, n_cat / max(d, 1)), 4)
        missing = _to_num(q.get("NumberOfMissingValues")) or 0.0
        missing_rate = round(min(0.5, missing / (n * d_total)), 4)
        maj = _to_num(q.get("MajorityClassSize"))
        majority_frac = round(min(1.0, maj / n), 4) if (task and maj) else 0.0
        max_card = int(_to_num(q.get("MaxNominalAttDistinctValues")) or 0)
        curated = int(did in cc18_ids or did in curated_ids
                      or name.lower() in curated_names)
        row = [n, d, task, cat_frac, classes if task else 0,
               missing_rate, majority_frac, max_card, curated]
        # version dedup: keep the largest-n row per lowercased name
        key = name.lower()
        if key not in by_name or by_name[key][0] < n:
            by_name[key] = row
    print(f"  drops: {dropped}; kept {len(by_name)} unique-name rows", flush=True)
    return list(by_name.values())


# Belt-and-braces exclusion if the study fetch fails: known TabArena dataset
# names (public design info; used ONLY to exclude). Non-exhaustive fallback.
_TABARENA_NAME_FALLBACK = {
    "bank-customer-churn", "churn", "coil2000-insurance-policies",
    "taiwanese-bankruptcy-prediction", "credit-g", "blood-transfusion-service-center",
    "diabetes130us", "amazon_employee_access", "otto-group-products",
    "houses", "diamonds", "superconductivity", "wine-quality", "abalone",
}


def _in_repo_curated():
    """Names/ids of the in-repo decision suites (curated tags for the corpus)."""
    names, ids = set(), set()
    try:
        import run_benchmarks as rb
        for folder_names in rb.GRINSZTAJN_DATASETS.values():
            names.update(n.lower() for n in folder_names)
        for spec in rb.OPENML_SUITE.values():
            data_id = spec.get("data_id")
            if isinstance(data_id, int):
                ids.add(data_id)
            elif isinstance(data_id, str):
                names.add(data_id.lower())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: run_benchmarks registries unavailable ({exc})", flush=True)
    return names, ids


def snapshot(rows, max_rows, source_note):
    """Stratified cap: keep all curated rows, subsample broad rows (seed 0)."""
    import numpy as np
    curated = [r for r in rows if r[8] == 1]
    broad = [r for r in rows if r[8] == 0]
    budget = max(0, max_rows - len(curated))
    if len(broad) > budget:
        idx = np.random.default_rng(0).choice(len(broad), size=budget, replace=False)
        broad = [broad[i] for i in sorted(idx)]
    kept = curated + broad
    return {
        "version": 1,
        "source": source_note,
        "columns": ["n", "d", "task", "cat_frac", "n_classes",
                    "missing_rate", "majority_frac", "max_card", "curated"],
        "task_codes": {"0": "regression", "1": "binary", "2": "multiclass"},
        "n_corpus": len(rows),
        "n_curated": len(curated),
        "rows": kept,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch even if cached")
    ap.add_argument("--max-rows", type=int, default=2500)
    args = ap.parse_args()

    if os.path.exists(CACHE_PATH) and not args.refresh:
        print(f"using cached listing {CACHE_PATH}", flush=True)
        cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    else:
        print("fetching OpenML listing (paged)...", flush=True)
        entries = fetch_listing()
        print("fetching TabArena suite (exclusion list only)...", flush=True)
        tabarena_ids = fetch_study_data_ids(TABARENA_STUDY_ID)
        print(f"  {len(tabarena_ids)} TabArena dataset ids to exclude", flush=True)
        print("fetching OpenML-CC18 (curated tag)...", flush=True)
        cc18_ids = fetch_study_data_ids(CC18_STUDY_ID)
        cache = {"fetched_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                 "entries": entries,
                 "tabarena_ids": sorted(tabarena_ids),
                 "cc18_ids": sorted(cc18_ids)}
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp, CACHE_PATH)
        print(f"cached raw listing -> {CACHE_PATH}", flush=True)

    curated_names, curated_ids = _in_repo_curated()
    rows = distill(cache["entries"], set(cache.get("tabarena_ids", [])),
                   set(cache.get("cc18_ids", [])), curated_names, curated_ids)
    note = (f"OpenML data/list {cache.get('fetched_utc', '?')}; "
            f"junk prefixes {list(_JUNK_PREFIXES)} dropped; TabArena study "
            f"{TABARENA_STUDY_ID} excluded; curated = CC18 + in-repo suites")
    snap = snapshot(rows, args.max_rows, note)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(snap, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"snapshot: {snap['n_corpus']} corpus rows ({snap['n_curated']} curated), "
          f"{len(snap['rows'])} kept -> {SNAPSHOT_PATH}", flush=True)

    import numpy as np
    arr = np.array([r[:8] for r in snap["rows"]], dtype=float)
    tasks = arr[:, 2].astype(int)
    print("\ncorpus marginals (kept rows):", flush=True)
    print(f"  task mix: reg {np.mean(tasks == 0):.2f} / bin {np.mean(tasks == 1):.2f} "
          f"/ mc {np.mean(tasks == 2):.2f}", flush=True)
    print(f"  n:    median {np.median(arr[:, 0]):.0f}  p90 {np.percentile(arr[:, 0], 90):.0f}",
          flush=True)
    print(f"  d:    median {np.median(arr[:, 1]):.0f}  p90 {np.percentile(arr[:, 1], 90):.0f}",
          flush=True)
    print(f"  cat_frac>0: {np.mean(arr[:, 3] > 0):.2f}   missing>0: {np.mean(arr[:, 5] > 0):.2f}",
          flush=True)


if __name__ == "__main__":
    main()
