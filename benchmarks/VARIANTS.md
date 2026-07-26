# Variant families — supplemental under-sampling (SUS) and temporal splits

Issue #37. A **variant** is a derived view of a parent dataset that probes one
named failure regime. Keys are `<parent>@<variant>`, so every downstream tool
(`summarize`, `compare_runs`, the charts) keeps working unchanged.

North star: *why do models commonly fail.* Each family isolates one cause.

| family | what it changes | regime it probes |
|---|---|---|
| `@sus25` / `@sus50` | trains on 25% / 50% of the training rows; **test set unchanged** | data volume — how the ranking looks earlier in a dataset's life |
| `@time` | rows ordered by a real timestamp, train early / test late | distribution shift — the regime every other split here is blind to |

## Rules that make these honest

**Strata, never pooled.** A variant reuses its parent's rows, so counting both
in one sign test would inflate the effective sample size and quietly weaken
every ship decision. `summarize.stratum_of` puts each family in its own
stratum; `compare_runs.py --by-suite` runs one independent sign test per
stratum. No combined verdict is printed anywhere.

**SUS shrinks training rows only.** The twin is scored on exactly the rows its
parent is scored on for that seed, so the pair reads as a learning curve rather
than as two noisier datasets. Shrinking the test set too would have added
variance precisely where the sample is already smallest.

**Two families, not one.** "Earlier and smaller" is their intersection. Kept
apart, a result is attributable to volume or to shift; blurred together it is
attributable to neither. The combination is cheap to add later once each family
has shown it carries signal.

**Fixed `random_state=0` for the shrink**, matching the house convention for the
suite row caps, so the train/test split stays the only seed-dependent step.

**Rolling origin for `@time`.** A single fixed cut is deterministic, and since
every model runs at `random_state=0`, all seeds would reproduce one number
exactly — replication in name only. Seed *s* takes cut `TEMPORAL_CUTS[s % 3]`
(0.65 / 0.70 / 0.75) and tests the 25% window that follows.

**Unseen classes are dropped from a temporal test window**, and the count is
reported. A model cannot emit a probability for a label it never saw; scoring
such a row would make the one-hot vector all zeros, so Brier would *reward*
assigning low probability to every real class.

**The time column stays a feature.** A model meeting timestamps beyond its
training range is exactly the deployment failure being measured.

## Frozen SUS assignment

Selection is by position in each suite's sorted key order: index `% 5 == 0` gets
a 25% twin (20% of the suite), index `% 10 == 3` gets a 50% twin (10%). The
offset keeps the picks disjoint, and stepping through sorted order spreads them
across the suite instead of clustering. `tests/test_variants.py` locks these
tables to the code.

### Grinsztajn — 59 datasets, 12 at 25% (20.3%), 6 at 50% (10.2%)

| # | dataset | twin |
|---|---|---|
| 0 | `gr:clf_cat/albert` | `@sus25` |
| 3 | `gr:clf_cat/default-of-credit-card-clients` | `@sus50` |
| 5 | `gr:clf_cat/eye_movements` | `@sus25` |
| 10 | `gr:clf_num/MagicTelescope` | `@sus25` |
| 13 | `gr:clf_num/california` | `@sus50` |
| 15 | `gr:clf_num/credit` | `@sus25` |
| 20 | `gr:clf_num/house_16H` | `@sus25` |
| 23 | `gr:reg_cat/Airlines_DepDelay_1M` | `@sus50` |
| 25 | `gr:reg_cat/Bike_Sharing_Demand` | `@sus25` |
| 30 | `gr:reg_cat/analcatdata_supreme` | `@sus25` |
| 33 | `gr:reg_cat/house_sales` | `@sus50` |
| 35 | `gr:reg_cat/nyc-taxi-green-dec-2016` | `@sus25` |
| 40 | `gr:reg_num/Ailerons` | `@sus25` |
| 43 | `gr:reg_num/MiamiHousing2016` | `@sus50` |
| 45 | `gr:reg_num/cpu_act` | `@sus25` |
| 50 | `gr:reg_num/house_sales` | `@sus25` |
| 53 | `gr:reg_num/nyc-taxi-green-dec-2016` | `@sus50` |
| 55 | `gr:reg_num/sulfur` | `@sus25` |

### HC — 14 datasets, 3 at 25% (21.4%), 2 at 50% (14.3%)

| # | dataset | twin |
|---|---|---|
| 0 | `hc:Moneyball` | `@sus25` |
| 3 | `hc:cjs` | `@sus50` |
| 5 | `hc:employee_salaries` | `@sus25` |
| 10 | `hc:okcupid-stem` | `@sus25` |
| 13 | `hc:wine-reviews` | `@sus50` |

The HC percentages run slightly high because 14 datasets cannot be split into
exact fifths and tenths. That is the cost of a fixed, auditable rule, and it is
preferable to a per-suite fudge factor.

### Public — 13 datasets, 3 at 25% (23.1%), 1 at 50% (7.7%)

| # | dataset | twin |
|---|---|---|
| 0 | `pub:BNP_Paribas_Cardif_Claims_Management` | `@sus25` |
| 3 | `pub:SantanderCustomerSatisfaction` | `@sus50` |
| 5 | `pub:federal_election` | `@sus25` |
| 10 | `pub:internet_firewall` | `@sus25` |

Index order is the plain sort of the frozen keys, so the capitalised names sort
ahead of the lowercase ones. Nothing chose these four — the stride did.

## Frozen temporal registry

Audited by hand against the cached frames. A column qualifies only if it is a
real time of record (not merely year-shaped), is not the target, and survives
the near-unique-categorical drop. Verified: present, correct dtype, and enough
distinct values to cut a window.

| dataset | time column | notes |
|---|---|---|
| `hc:kick` | `PurchDate` | epoch seconds, vehicle purchase |
| `hc:sf-police-incidents` | `Year` | **unordered** pandas category — must be coerced, not sorted as a category |
| `hc:Traffic_violations` | `Year` | float; 0.6% unparseable, those rows dropped |
| `hc:house_prices_nominal` | `YrSold` | year the sale closed (Ames) |
| `hc:Moneyball` | `Year` | baseball season |
| `hc:employee_salaries` | `date_first_hired` | **`MM/DD/YYYY` string** — a lexicographic sort would order by month |
| `hc:eucalyptus` | `Year` | measurement year; only 736 rows, read gently |
| `pub:Medical-Appointment-No-Shows` | `AppointmentDay` | ISO date string. Not `ScheduledDay`, which is 94% unique and so is dropped as an identifier |
| `pub:kickstarter_projects` | `deadline` | datetime string, 3,102 distinct. Not `launched`, which is 99.9% unique. **Mixed UTC offsets** — the reason `_time_sort_key` passes `utc=True` |
| `pub:rossmann_store_sales` | `Year` | 2013–2015; no single date column exists, so this is the coarse-year arrangement four HC sets already use |

`pub:federal_election` deliberately has **no** time column: its `transaction_dt`
is a float in `MMDDYYYY` form, which a numeric sort would order by month.

`_time_sort_key` coerces numeric first, then datetime. That order matters: it
gets year columns and epoch seconds right regardless of whether they arrive as
int, float or category, and still parses real date strings.

**Grinsztajn has no temporal coverage and will not get any.** The HuggingFace
mirror ships pre-transformed numeric CSVs with no recoverable timestamp. The
regime has no expression there, and the registry declares that rather than
inventing a proxy. Coverage is therefore 7 datasets, all high-card — a named
regime with a small sample, reported as such and never inflated into a headline.

## Rejected candidates

Columns matching a date-like name that are **not** an observation time:
`colleges` (`mean_earnings_6_years` and friends — outcome horizons),
`black_friday` (`Stay_In_Current_City_Years` — a customer attribute),
`house_prices_nominal` (`YearBuilt`, `GarageYrBlt` — property attributes; only
`YrSold` is the time of record), `kdd_ipums_la_97-small` (`year`, `yrlastwk` —
census survey fields), `kick` (`VehYear` — the vehicle's model year, not the
transaction date).
