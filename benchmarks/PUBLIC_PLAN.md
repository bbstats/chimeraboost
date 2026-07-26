# The public suite — sealed benchmark behind the published chart

Issue #37. **Status: machinery complete, dataset list NOT yet frozen.** See
"Why the list is empty" at the bottom.

## Why a new suite at all

The published chart must not run on Grinsztajn or HC. We tune against those, so
charting them would be in-sample, and the north star for this work is *true
generalization, never faked from data*. A chart that flatters us because it runs
on the data we tuned on is exactly the failure being ruled out.

TabArena filled this role and no longer fits: it needs a separate AutoGluon
virtualenv, its evaluation is single-threaded and I/O-bound, a tuned full run is
weeks, and its median task is small enough that the figure's own source comments
concede the fit-speed column is "directional only". A benchmark whose speed axis
mostly measures fixed overhead cannot carry a speed claim.

## The vow

**SEALED — report-only.** No result from this suite, aggregate or per-task, may
influence a source change. Decisions keep running on synth → Grinsztajn + HC →
the OpenML one-shot gate. This is the same vow TabArena carries, and it is the
whole reason the chart means anything.

## Selection criteria

1. **Size.** ≥50k rows strongly preferred, cap 200k (`_PUBLIC_MAX_ROWS`). The
   speed axis must reflect work rather than per-fit overhead. This is the
   specific defect that retired TabArena-Lite from this role.
2. **Real data.** No synthetic generators, no `BNG_*` expansions.
3. **Regime spread**, because the north star is *why models commonly fail*:
   high-cardinality categoricals, class imbalance, heavy-tailed regression
   targets, missing data, many irrelevant features, and — via the variant
   families — small n (`@sus`) and distribution shift (`@time`).
4. **Task mix**: regression, binary and multiclass all represented.
5. **Loadable by OpenML id** through the existing builder, cached on `A:`.
6. **A `time_col` where one genuinely exists**, so the public suite carries
   temporal variants too (see `VARIANTS.md` for what qualifies).

## Step 0 — overlap audit (hard gate)

Reuse the procedure proven for HC (`HIGHCARD_PLAN.md:31`). Build the exclusion
lists **before** looking at any candidate; a hit against any of them is an
automatic cut:

| list | source | why |
|---|---|---|
| TabArena-51 | `tests/test_highcard.py::TABARENA_51` (names only) | Nathan may still hand TabArena to its authors; contaminating it would burn that read |
| Grinsztajn 59 | `GRINSZTAJN_DATASETS` | decision suite — overlap makes the public chart in-sample |
| OpenML gate 29 | `OPENML_SUITE` | overlap would un-independent the one-shot gate |
| PMLB 25 | `PMLB_DATASETS` | HP-tuning suite |
| HC 14 | `HC_DATASETS` | decision suite |

Match on OpenML dataset id first, then normalised name (case, underscores and
hyphens stripped — the `_norm` helper in `tests/test_highcard.py`).

**Already known to be consumed** (checked while drafting, so do not re-propose):
covertype, Higgs, Allstate_Claims_Severity, road-safety, nyc-taxi-green-dec-2016,
diamonds, house_sales, Airlines_DepDelay_1M, MiniBooNE, jannis, albert,
delays_zurich_transport, medical_charges, particulate-matter-ukair-2017,
seattlecrime6, SGEMM_GPU_kernel_performance, Diabetes130US, superconduct,
APSFailure, kddcup09_appetency, adult, electricity, letter, and every dataset
already in HC. The well-known large tabular sets are mostly spoken for, so
expect the audit to work harder than HC's did.

## Step 1 — verify every candidate in-session

`HIGHCARD_PLAN.md` is explicit that ids from memory are not trustworthy. For
each candidate confirm, from a live source: id → name, row count, feature count,
task type, target column, class count and imbalance, missing-value count.
`_tmp/openml_check.py`-style metadata queries are enough — no full download
needed for selection.

## Step 2 — freeze

Populate `PUBLIC_DATASETS` in `run_benchmarks.py` (`name -> dict(data_id=..,
task=.., time_col=..)`), write the frozen table into this file, and add a test
mirroring `test_highcard.py::test_frozen_matches_doc` plus
`test_no_suite_overlap` so the list and the doc cannot drift apart and the
overlap gate is re-checked on every test run.

## Step 3 — first run and the chart

```
python benchmarks/run_benchmarks.py --public --seeds 3 --save \
    --models ChimeraBoost ChimeraBoostEns8 CatBoost LightGBM
python benchmarks/make_public_pareto.py <that>.json
```

The chart refuses to present a non-`pub:` run as publishable: it prints a
"NOT A PUBLISHABLE READ" banner and stamps the figure title, because a run on
the decision suites is in-sample. Once the suite is frozen and run, point the
README headline at `images/public_pareto.png` in place of the TabArena figure.

## What the chart does differently

- **Two competitors only**: LightGBM as the speed reference, CatBoost as the
  quality reference. XGBoost tracks LightGBM closely and RandomForest was never
  in the harness — both were chart-only entries on the TabArena figure.
- **Competitor-relative win rate.** Each ChimeraBoost point is scored against
  CatBoost and LightGBM only, never against sibling rungs. The internal chart's
  field-relative rate shifts every row whenever an arm is added, and with
  several of our rungs against two competitors most opponents would be our own
  arms — "wins N% of matchups" would largely be us beating ourselves.
  `tests/test_public_winrate.py` pins both the correctness and the stability
  property (adding a rung leaves every other row untouched).
- **The fast rung is off the chart for now** (Nathan's call on this issue).

## Why the list is empty

The machinery — suite registration, the `--public` flag and its guard, the
larger row cap, temporal support for `pub:` datasets, the chart and its tests —
is complete and exercised. The dataset list is not frozen because the OpenML
API returned HTTP 504 for every request during the session that built this
(single-dataset metadata, qualities, and the filtered list endpoint alike;
general connectivity was fine, so it was an OpenML-side outage). Freezing a list
from remembered ids would violate step 1 above and risk shipping a suite that
overlaps a holdout or points at a dataset that has since changed version.

Filling it in is one line per dataset plus the frozen table, once the audit runs.
