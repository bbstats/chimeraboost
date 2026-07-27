# The public suite — sealed benchmark behind the published chart

Issue #37. **Status: frozen at 22 datasets and run.** The audit, the frozen
list, the first read and the open items are all below; `images/public_pareto.png`
is the figure it produces.

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
| ~~nba-shot-logs~~ | 42806 | **reinstated** — see below. `FGM` and `PTS` each determine `SHOT_RESULT` exactly, but they are two named columns, which is what `drop_cols` is for. |
| autos | 45080 | **not real data.** 1,000,000 rows carrying the 205-row imports-85 schema — a generator expansion that the `BNG(` prefix filter does not catch. |
| la_crimes | 42160 | no identifiable target; `Crime_Code` is mirrored by `Crime_Code_Description` and by `Crime_Code_1..4`, so whichever is the target, three columns leak it. Otherwise attractive (1.47M rows, 9 high-card cats, real timestamps) — kept as an alternate if a target can be established. |
| Economic_Census_Delhi | 46096 | ambiguous target; ships both `District` and `DISTRICT`. |

**Four aliases the name matcher cannot join**, each caught by reading columns
rather than by the gate, and all now on `CONSUMED` in `public_audit.py`:
`uci_diabetes_p` (42106) is `Diabetes130US` at the same 101,766 rows;
**`Winedata` (43651) is the Kaggle wine-reviews data that is already
`hc:wine-reviews` (41275)** — same country/designation/points/price/taster/
variety/winery schema, and HC is a suite we tune on; `hcdr_main` (45567) and
`rossmann_store_sales_processed` (45646) are re-uploads of frozen `pub:`
members. Two concatenations were cut for smuggling a consumed dataset in as a
column block: `AirlinesCodrnaAdult` (1240) contains `adult`, and `CovPokElec`
(149) contains covertype and electricity.

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
| pub:nba-shot-logs | 42806 | binary | 128,069 | 128,069 × 15 | 6 | 1,808 (`MATCHUP`) | `SHOT_RESULT` | — | high-card cats on a balanced, genuinely hard binary target |
| pub:Dota2-Games-Results | 45563 | binary | 102,944 | 102,944 × 116 | 116 | 3 | `Team_won` | — | every feature categorical; best single column 0.534 against a 0.527 base |
| pub:Cardiovascular-Disease | 45547 | binary | 70,000 | 70,000 × 11 | 6 | low | `cardio` | — | perfectly balanced real medical data |
| pub:BMC_TrainingData | 43066 | binary | 177,640 | 177,640 × 38 | 8 | 11,200 (`geo_level_3_id`) | `category` | — | 0.096 minority; nested geographic codes |
| pub:volkert | 41166 | multiclass | 58,310 | 58,310 × 180 | 0 | — | `class` | — | 10 classes over 180 numeric features |
| pub:helena | 41169 | multiclass | 65,196 | 65,196 × 27 | 0 | — | `class` | — | **100 classes**, rarest at 0.0017 |
| pub:ldpa | 1483 | multiclass | 164,860 | 164,860 × 7 | 2 | — | `Class` | — | 11 classes, rarest 0.0084, few features |
| pub:fars | 45066 | multiclass | 100,959 | 100,959 × 29 | 15 | — | `class` | — | 7 classes with 15 categorical columns |
| pub:criteo-uplift-balanced | 47039 | multiclass | 1,366,544 | 200,000 × 13 | 1 | — | `label` | — | 4 classes at advertising scale |
| pub:internet_firewall | 46978 | multiclass | 65,532 | 65,532 × 11 | 4 | 29,152 (`NAT_Source_Port`) | `Action` | — | 4-class, high-card, rarest class 0.0008 |
| pub:connect-4 | 40668 | multiclass | 67,557 | 67,557 × 42 | 42 | 3 | `class` | — | all-categorical multiclass |
| pub:Otto-Group-Product-Classification-Challenge | 45548 | multiclass | 61,878 | 61,878 × 94 | 0 | — | `target` | — | 9 classes over count features |
| pub:hls4ml_lhc_jets_hlf | 42468 | multiclass | 830,000 | 200,000 × 16 | 0 | — | `class` | — | 5-class at a size the speed axis can read |
| pub:rossmann_store_sales | 45647 | regression | 804,056 | 200,000 × 17 | 8 | 1,115 (`Store`) | `Sales` | `Year` | retail demand, entity cat |
| pub:freMTPL2freq | 41214 | regression | 678,013 | 200,000 × 10 | 4 | 22 (`Region`) | `ClaimNb` | — | heavy-tailed count target (~95% zeros) |
| pub:fps-in-video-games | 42737 | regression | 425,833 | 200,000 × 44 | 14 | 446 (`GpuName`) | `FPS` | — | entity cats + missingness |
| pub:federal_election | 42080 | regression | 3,348,209 | 200,000 × 17 | 15 | `employer`/`occupation`/`city`/`zip_code` | `transaction_amt` | — | the entity-cat regime at full scale |

Composition: **22 datasets**; 9 binary / 9 multiclass / 4 regression; 9 with a
categorical of ≥50 levels; 3 regression-with-cats; 3 carrying a time column.
Registration yields 32 keys once the `@sus25`/`@sus50`/`@time` twins are added.

**Regression is under-represented at 4 of 22, and that is a known gap.** It is
not for want of looking: the expansion round screened nine further regression
candidates and every one failed on the data. `Google-Play-Store-Apps` encodes
unrated apps as `Rating` exactly 0.0 — 45.8% of rows, every one of them with
`Rating_Count` 0, so half the target is a sentinel any model would learn in one
split. `Cinema-Tickets` is an exact identity (`tickets_sold` × `ticket_price` =
`total_sales`). `Bus-Breakdown-and-Delays-NYC` has a zero-inflated target with a
median of 0 and a maximum of 9,007 students on one bus. `30mlday` has no signal
at all — the largest absolute Spearman correlation against its target across
every feature is 0.057. `SoilHydroDB` has no identifiable target and its
water-retention columns leak each other. Padding the count with any of these
would have made the suite look bigger and mean less.

**On reinstating nba-shot-logs.** It was cut in the first pass for target
leakage and then brought back, because the leak is two named columns rather than
a property of the dataset. Measured per-column purity — how well knowing one
column alone pins the label — is 1.0000 for both `FGM` and `PTS`, and at or
barely above the 0.5479 majority baseline for everything else, the one exception
being `SHOT_DIST` at 0.611, which is just basketball. With those two dropped the
target is balanced at 0.452 and genuinely hard, and the set contributes three
real high-cardinality categoricals (`MATCHUP` 1,808, `CLOSEST_DEFENDER` 473,
`player_name` 281) to a suite that was thin on high-card *binary*. `player_id`
and `CLOSEST_DEFENDER_PLAYER_ID` go too: they are numeric twins of the name
columns, and leaving them in hands the model a leak-free numeric surrogate for
the very categorical the dataset is here to exercise.

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

## First read — 2026-07-27 (22 datasets, whole quality ladder)

`--public --seeds 3` over all five `quality` rungs plus both competitors:
**295 minutes** wall on this box (5 parallel jobs, 2 threads each). Base suite
only, no variant families.

| model | win% vs CatBoost + LightGBM | 95% CI | median × | mean × | on frontier |
|---|---|---|---|---|---|
| CatBoost | 66.7% | 48–86 | **52.7×** | 121.1× | yes |
| quality=4 (ensemble) | 64.3% | 50–79 | **15.5×** | 20.1× | yes |
| quality=5 (max) | 64.3% | 50–79 | **22.5×** | 28.1× | no — same strength as rung 4, 45% more cost |
| quality=3 (accurate, default) | 50.0% | 33–67 | **6.9×** | 12.9× | yes |
| quality=2 (balanced) | 40.5% | 26–55 | **4.8×** | 6.4× | yes |
| LightGBM | 33.3% | 14–52 | **1.0×** | 1.0× | yes |
| quality=1 (fast) | 28.6% | 14–43 | **2.3×** | 4.2× | no — weaker than LightGBM *and* 2.3× slower |

**Slowdown: report the median, and say so.** The mean fit-time multiple is not a
representative number on this suite. Ratios are right-skewed by construction — a
model can be 900× slower but never 900× faster — and a handful of datasets own
the average. CatBoost needs **2,883 s against LightGBM's 3 s on `pub:fars`**, a
970× ratio on one dataset; that alone roughly doubles its mean, taking 52.7×
to 121.1×. Our own default rung shows the same shape, 6.9× median against 12.9×
mean. The chart plots the median (the typical dataset) and the table carries
both, because the mean answers a different and still-legitimate question: what
running the whole suite costs.

The worst CatBoost-vs-LightGBM ratios are all wide-categorical or many-class
sets — `fars` 970×, `Dota2-Games-Results` 289×, `connect-4` 276×, `ldpa` 158× —
which is ordered target statistics doing what they do.

21 of 22 datasets scored; one near-solved set excluded by the PR #31 guard.

**CatBoost is no longer dominated. That is a reversal, and it is the headline.**
On 13 datasets our top rung led it 65.4% to 61.5%; on 22 it trails, 64.3%
against 66.7%. The earlier claim was an artifact of a suite too small to
separate them — exactly the failure mode the wide error bars were warning
about, and the reason the suite was grown. Anyone quoting "CatBoost is
dominated" from the earlier read should stop.

What survives is a cost argument rather than a strength argument: CatBoost buys
those 2.4 points for **6× the fit time** (121.1× against rung 4's 20.1×), and
the gap sits well inside both confidence intervals. The defensible claim is
"within noise of CatBoost at a sixth of the cost", not "better than CatBoost".

**Two rungs are off their own frontier**, which is worth knowing and must not be
acted on:

- `quality=5 (max)` scores identically to `quality=4 (ensemble)` — 64.3% each,
  same interval — while costing 40% more. Eight bagged members bought nothing
  over five here.
- `quality=1 (fast)` is beaten by LightGBM on strength *and* is 4.2× slower.
  As a speed rung it has no argument on this suite.

Both are sealed-suite observations. Under the vow they may not motivate a
change to the ladder, the defaults, or anything in `chimeraboost/`. They are
recorded because publishing a chart while staying quiet about two rungs sitting
off the frontier would be dishonest.

**Error bars behaved as predicted.** Going from 13 to 21 scored datasets pulled
the interval on the default rung from 38 points to 34, and rung 4 sits at 29 —
close to the ~29 estimated from the 1/√n scaling. Seeds were never the lever;
the bootstrap resamples datasets.

## What remains

1. **Point the README headline** at `images/public_pareto.png` — Nathan's call,
   given the numbers above differ substantially from the current TabArena-based
   headline.
2. The `@sus` and `@time` variant families, as a second figure, when wanted.

**No dependency on OpenML's metadata API remains**, by design — see "Loaded from
data.openml.org" above. `public_audit.py shortlist` still reads the harvested
catalogue if the suite ever needs re-auditing, and `--refresh` would need the
API back, but nothing in the run path touches it.

**Re-auditing.** The frozen list changes only with a re-run of
`public_audit.py shortlist` and a fresh content verification, on the
synthgen-freeze discipline. `tests/test_public.py` keeps the list, this document
and the overlap gate in agreement on every test run, and
`test_freeze_is_all_or_nothing` makes a half-edit fail.
