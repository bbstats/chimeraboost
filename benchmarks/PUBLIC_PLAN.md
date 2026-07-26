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

1. **Size.** ≥50k rows, **no upper bound** — `_PUBLIC_MAX_ROWS` (200k) is a
   subsample cap the builder applies at a fixed `random_state=0`, not an
   eligibility ceiling, exactly as HC already treats porto-seguro (595k rows)
   and airlines (539k). The speed axis must reflect work rather than per-fit
   overhead; that is the specific defect that retired TabArena-Lite from this
   role. (This criterion originally read as a 50k–200k *window*, which would
   have excluded most genuinely large real-world data for no reason.)
2. **Real data.** No synthetic generators, no `BNG_*` expansions. Note that the
   prefix filter is not sufficient on its own: OpenML hosts million-row
   *expansions* of small classic datasets under the original name — `autos`
   (id 45080) is 1,000,000 rows carrying the 205-row imports-85 schema. Check
   the row count against the dataset's known size before accepting it.
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
| TabArena-51 | `suite_overlap.TABARENA_51` (names) + the harvested study-457 ids | Nathan may still hand TabArena to its authors; contaminating it would burn that read |
| Grinsztajn 59 | `GRINSZTAJN_DATASETS` | decision suite — overlap makes the public chart in-sample |
| OpenML gate 29 | `OPENML_SUITE` | overlap would un-independent the one-shot gate |
| PMLB 25 | `PMLB_DATASETS` | HP-tuning suite |
| HC 14 | `HC_DATASETS` | decision suite |

Match on OpenML dataset id first, then normalised name (case, underscores and
hyphens stripped). All of this is implemented in `benchmarks/suite_overlap.py`
and applied by `benchmarks/public_audit.py shortlist`; the same module backs
`tests/test_highcard.py` and `tests/test_public.py`, so the gate the audit
applied is the gate the test suite keeps applying.

**Already known to be consumed** (checked while drafting, so do not re-propose):
covertype, Higgs, Allstate_Claims_Severity, road-safety, nyc-taxi-green-dec-2016,
diamonds, house_sales, Airlines_DepDelay_1M, MiniBooNE, jannis, albert,
delays_zurich_transport, medical_charges, particulate-matter-ukair-2017,
seattlecrime6, SGEMM_GPU_kernel_performance, Diabetes130US, superconduct,
APSFailure, kddcup09_appetency, adult, electricity, letter, and every dataset
already in HC. The well-known large tabular sets are mostly spoken for, so
expect the audit to work harder than HC's did.

## Step 1 — verify every candidate in-session

`HIGHCARD_PLAN.md` is explicit that ids from memory are not trustworthy. Every
property below was read in-session from one of two live sources, never from
memory:

- **The catalogue.** `benchmarks/data_cache/openml_meta.json` — 6408 active
  OpenML datasets with full qualities, harvested from the listing API by
  `benchmarks/synthgen/harvest_metadata.py` on 2026-07-14, plus the TabArena
  study-457 dataset ids. `public_audit.py shortlist` reads it; `--refresh`
  re-fetches once the API is reachable.
- **The data itself.** `public_audit.py verify <id>` downloads the dataset's
  parquet from `data.openml.org` (which stayed up throughout the API outage)
  and reports row count, column names, dtypes, per-column cardinality,
  missingness and genuine time-column candidates from the actual bytes. This is
  *stronger* evidence than the qualities table, which is stale or absent for
  many parquet-era uploads — `MaxNominalAttDistinctValues` reads 0 for several
  datasets that in fact carry 18,000-level categoricals.

**Sweep result.** 6408 catalogue entries → 5440 cut for size (<50k rows), 205
synthetic-generator families, 33 flattened image/text corpora, 108 too narrow or
too wide (outside 5–200 features), 6 by TabArena dataset id, 151 by the name and
id overlap gate → **373 survivors**, of which 21 were content-verified against
their parquet files.

### What the qualities table cannot tell you, and the parquet can

Four candidates were cut on evidence that exists only in the data:

| cut | id | why |
|---|---|---|
| nba-shot-logs | 42806 | **target leakage.** `FGM` determines `SHOT_RESULT` exactly (the crosstab is diagonal: 57,905 made ⇔ FGM=1, 70,164 missed ⇔ FGM=0), and `PTS` does the same. Any model scores ~1.0. |
| autos | 45080 | **not real data.** 1,000,000 rows carrying the 205-row imports-85 schema — a generator expansion that the `BNG(` prefix filter does not catch. |
| la_crimes | 42160 | no identifiable target; `Crime_Code` is mirrored by `Crime_Code_Description` and by `Crime_Code_1..4`, so whichever is the target, three columns leak it. Otherwise attractive (1.47M rows, 9 high-card cats, real timestamps) — kept as an alternate if a target can be established. |
| Economic_Census_Delhi | 46096 | ambiguous target; ships both `District` and `DISTRICT`. |

Two more were cut on judgment rather than a hard rule, and are recorded so the
call is visible:

- **uci_diabetes_p (42106)** is `Diabetes130US` under another name — 101,766
  rows, the exact size of the canonical set, which is in Grinsztajn, in
  TabArena-51 and on the consumed list. The normalised-name matcher does not
  join those two strings, so it passed the automated gate. It is now on the
  `CONSUMED` list in `public_audit.py` so it cannot come back.
- **Census-Income (4535) / Census-Income-KDD (42750)** are the KDD extract of
  the same US census source that `adult` comes from. Not the same dataset —
  299k rows against adult's 49k — but close enough to the gate suite that
  charting it would weaken the out-of-sample claim.

### Proposed suite (13 datasets) — NOT YET FROZEN

Everything in the "verified" columns was read from the parquet. The **target
column is a presumption**, not a verification: `data.openml.org` serves data
files only, and OpenML's default-target field is the one thing the audit could
not reach. See "What remains" below.

| candidate | id | task | rows | cats | max card | missing cells | presumed target | time_col | regime it buys |
|---|---|---|---|---|---|---|---|---|---|
| BNP_Paribas_Cardif_Claims_Management | 46856 | binary | 114,321 | 20 | 18,210 (`v22`) | 5.1M | `target` | — | extreme-card cat + pervasive missingness |
| Medical-Appointment-No-Shows | 43439 | binary | 110,527 | 5 | 81 (`Neighbourhood`) | 0 | `No-show` | `ScheduledDay` | real timestamps, 0.25 imbalance |
| kickstarter_projects | 42076 | binary | 331,675 | 7 | 3,102 (`deadline`) | 210 | `state` | `launched` | temporal + mixed-timezone dates |
| SantanderCustomerSatisfaction | 45566 | binary | 200,000 | 0 | — | 0 | `target` | — | wide all-numeric, 0.11 imbalance |
| hcdr | 45071 | binary | 244,280 | 47 | 58 (`ORGANIZATION_TYPE`) | 0 | `class` | — | the most categorical-heavy candidate found |
| internet_firewall | 46978 | multiclass | 65,532 | 5 | 29,152 (`NAT_Source_Port`) | 0 | `Action` (4) | — | high-card multiclass, 0.08% rarest class |
| connect-4 | 40668 | multiclass | 67,557 | 42 | 3 | 0 | `class` (3) | — | all-categorical multiclass |
| Otto-Group-Product-Classification-Challenge | 45548 | multiclass | 61,878 | 0 | — | 0 | `target` (9) | — | 9-class over 93 count features |
| hls4ml_lhc_jets_hlf | 42468 | multiclass | 830,000 | 0 | — | 0 | `class` (5) | — | multiclass at a size the speed axis can read |
| rossmann_store_sales | 45647 | regression | 804,056 | 9 | 1,115 (`Store`) | 0 | `Sales` | `Year` | retail demand; entity cat |
| freMTPL2freq | 41214 | regression | 678,013 | 4 | 22 (`Region`) | 0 | `ClaimNb` | — | heavy-tailed count target (~95% zeros) |
| fps-in-video-games | 42737 | regression | 425,833 | 14 | 446 (`GpuName`) | 1.3M | `FPS` | — | entity cats + missingness |
| federal_election | 42080 | regression | 3,348,209 | 16 | 10 cats ≥50 (`employer`, `occupation`, `city`, `zip_code`) | 10.8M | `transaction_amt` | `transaction_dt` | the entity-cat regime at full scale |

Composition against the targets: 13 datasets; 5 binary / 4 multiclass /
4 regression; 8 with a categorical of ≥50 levels; 3 regression-with-cats;
4 carrying a genuine time column.

**Per-dataset column cuts.** Three of these carry columns that near-uniqueness
cannot catch, so the builder now honours a `drop_cols` key on the spec:
`rossmann_store_sales` ships a `Set` column reading train/test/valid (an
uploader's split marker, not a feature); `freMTPL2freq` ships `IDpol`, a numeric
row id that the categorical-only filter leaves in place; `federal_election`
ships `tran_id`/`sub_id`/`image_num`. Every cut is named in the frozen list so
it is reviewable.

**One trap already found.** `federal_election`'s `transaction_dt` is an integer
in MMDDYYYY form, so a numeric sort orders it by month — the same failure
`employee_salaries` has and the reason `_time_sort_key` coerces numerics first.
It needs the same treatment before it can carry a `@time` variant, so the
temporal slot for that dataset is left open pending a decision.

## Step 2 — freeze (BLOCKED, see "What remains")

Populate `PUBLIC_DATASETS` in `run_benchmarks.py` (`name -> dict(data_id=..,
task=.., time_col=.., drop_cols=..)`), write the frozen table into this file in
the `| pub:<name> | <id> | <task> |` form the test parses, and the tests in
`tests/test_public.py` start enforcing it. Those tests are already written: they
pass vacuously while the list is empty, and the moment it is populated they
require the doc table and the code to agree exactly and re-run the full overlap
gate. `test_freeze_is_all_or_nothing` makes a half-freeze (code populated but
doc not, or the reverse) a failure.

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

## What remains

The machinery — suite registration, the `--public` flag and its guard, the
larger row cap, temporal support for `pub:` datasets, the chart and its tests —
was complete before this audit. The audit itself (step 0 and step 1) has now
run: the overlap gate is implemented and tested, 373 candidates survived it, and
13 finalists are content-verified against their actual data.

**Three facts still block the freeze, and all three need one live metadata call
per dataset — about 13 requests:**

1. **The default target column.** `data.openml.org` serves data files only. The
   targets in the table above are presumptions from the column names. If a
   dataset has no default target set at all, `fetch_openml` returns `ds.target
   = None` and the builder dies at `ds.target.name`.
2. **The current version**, so the frozen id points at the data that was
   audited rather than a later re-upload.
3. **Active status**, so a deactivated dataset does not enter the suite.

The OpenML REST API was returning HTTP 504 for every endpoint — single-dataset
metadata, qualities and the filtered listing alike — throughout both the session
that built the machinery (2026-07-25) and the session that ran the audit
(2026-07-26). `api.openml.org` 301-redirects onto the same dead gateway. General
connectivity was fine and `data.openml.org` was serving normally, so this is an
OpenML-side outage of the metadata tier specifically.

When it comes back:

```
python benchmarks/public_audit.py shortlist --refresh   # re-harvest, re-gate
```

then confirm target/version/status per finalist, populate `PUBLIC_DATASETS`,
write the frozen `| pub:... |` table into this file, and run `pytest
tests/test_public.py`. Nothing else in the path is outstanding.
