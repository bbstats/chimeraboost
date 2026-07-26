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

## Step 2 — the frozen suite (13 datasets, frozen 2026-07-26)

Every row was read from the dataset's real bytes in-session, including the
target column. "rows" is the source size; the builder subsamples anything above
200,000. "loaded" is what the harness actually hands a model, measured by
running the builder.

| pub key | id | task | rows | loaded | cats | max card | target | time_col | regime it buys |
|---|---|---|---|---|---|---|---|---|---|
| pub:BNP_Paribas_Cardif_Claims_Management | 46856 | binary | 114,321 | 114,321 × 132 | 19 | 18,210 (`v22`) | `target` | — | extreme-card cat + pervasive missingness (5.1M cells) |
| pub:Medical-Appointment-No-Shows | 43439 | binary | 110,527 | 110,527 × 12 | 3 | 81 (`Neighbourhood`) | `No-show` | `AppointmentDay` | real timestamps, 0.20 minority |
| pub:kickstarter_projects | 42076 | binary | 331,675 | 200,000 × 12 | 5 | 3,102 (`deadline`) | `state` | `deadline` | temporal, mixed-timezone dates |
| pub:SantanderCustomerSatisfaction | 45566 | binary | 200,000 | 200,000 × 200 | 0 | — | `target` | — | wide all-numeric, 0.10 minority |
| pub:hcdr | 45071 | binary | 244,280 | 200,000 × 69 | 47 | 58 (`ORGANIZATION_TYPE`) | `class` | — | the most categorical-heavy set found; 0.078 minority |
| pub:internet_firewall | 46978 | multiclass | 65,532 | 65,532 × 11 | 4 | 29,152 (`NAT_Source_Port`) | `Action` | — | 4-class, high-card, rarest class 0.0008 |
| pub:connect-4 | 40668 | multiclass | 67,557 | 67,557 × 42 | 42 | 3 | `class` | — | all-categorical multiclass |
| pub:Otto-Group-Product-Classification-Challenge | 45548 | multiclass | 61,878 | 61,878 × 94 | 0 | — | `target` | — | 9 classes over count features |
| pub:hls4ml_lhc_jets_hlf | 42468 | multiclass | 830,000 | 200,000 × 16 | 0 | — | `class` | — | 5-class at a size the speed axis can read |
| pub:rossmann_store_sales | 45647 | regression | 804,056 | 200,000 × 17 | 8 | 1,115 (`Store`) | `Sales` | `Year` | retail demand, entity cat |
| pub:freMTPL2freq | 41214 | regression | 678,013 | 200,000 × 10 | 4 | 22 (`Region`) | `ClaimNb` | — | heavy-tailed count target (~95% zeros) |
| pub:fps-in-video-games | 42737 | regression | 425,833 | 200,000 × 44 | 14 | 446 (`GpuName`) | `FPS` | — | entity cats + missingness |
| pub:federal_election | 42080 | regression | 3,348,209 | 200,000 × 17 | 15 | `employer`/`occupation`/`city`/`zip_code` | `transaction_amt` | — | the entity-cat regime at full scale |

Composition: 13 datasets; 5 binary / 4 multiclass / 4 regression; 8 with a
categorical of ≥50 levels; 3 regression-with-cats; 3 carrying a time column.
Registration yields 20 keys once the `@sus25`/`@sus50`/`@time` twins are added.

**Loaded from data.openml.org, not through `fetch_openml`.** The `pub:` builder
downloads the dataset's parquet directly and takes the target from the frozen
`target` key. `fetch_openml` needs OpenML's metadata API to discover the default
target, and that tier returned HTTP 504 for days while the data host served
normally — but the better argument is reproducibility: a sealed benchmark must
not be able to change underneath us because someone edited a default-target
field upstream. HC still uses `fetch_openml`; it is a decision suite and
changing how it loads could move results.

**Per-dataset column cuts** (`drop_cols`, for what near-uniqueness cannot
catch): `rossmann_store_sales` ships a `Set` column reading train/test/valid,
an uploader's split marker rather than a feature; `freMTPL2freq` ships `IDpol`,
a numeric row id the categorical-only filter leaves in place;
`federal_election` ships `tran_id`, `sub_id` and `image_num`.

**Time columns are the coarse ones on purpose.** A time column must survive the
near-unique filter to remain a feature, so Medical-Appointment uses
`AppointmentDay` rather than the 94%-unique `ScheduledDay`, and kickstarter uses
`deadline` (3,102 distinct) rather than the 99.9%-unique `launched`. Rossmann has
no single date column, so it sorts on `Year` — the same coarse-year arrangement
four HC datasets already use.

**Two properties worth knowing before reading any result.**
`federal_election`'s `transaction_amt` is **already log-transformed** in this
upload: the median exponentiates to $416 and the maximum to exactly $10,000,000.
Skew is 0.94, so it is not a heavy-tailed target — `freMTPL2freq` is the one
carrying that regime. And `federal_election`'s `transaction_dt` is a float in
MMDDYYYY form, so a numeric sort would order it by month; it therefore gets no
`time_col` rather than a wrong one.

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

The suite is frozen and runs. All 13 datasets and all 7 variant twins load
through the harness, with class counts, imbalances and target ranges matching
what the audit measured. What is left is the run itself:

1. **Cost probe.** Nobody has measured what a full `--public` run costs. Time
   one dataset at one seed across the four arms first and extrapolate before
   committing to the whole thing.
2. **The run and the chart** (step 3 below).
3. **Point the README headline** at `images/public_pareto.png`.

**No dependency on OpenML's metadata API remains**, by design — see "Loaded from
data.openml.org" above. `public_audit.py shortlist` still reads the harvested
catalogue if the suite ever needs re-auditing, and `--refresh` would need the
API back, but nothing in the run path touches it.

**Re-auditing.** The frozen list changes only with a re-run of
`public_audit.py shortlist` and a fresh content verification, on the
synthgen-freeze discipline. `tests/test_public.py` keeps the list, this document
and the overlap gate in agreement on every test run, and
`test_freeze_is_all_or_nothing` makes a half-edit fail.
