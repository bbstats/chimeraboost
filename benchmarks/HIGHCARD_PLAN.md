# HC suite — real high-cardinality datasets for the decision stack

Self-sufficient handoff (the PAYOFF.md convention): everything needed to
implement this without its authoring session.

## Why (state of the world, 2026-07-15)

The Brier-gap program (benchmarks/synthgen/PAYOFF.md, results log inside)
found that the CatBoost Brier gap lives entirely in entity-cat / high-card
regimes: synth entity_strength Q4 = CatBoost 91% winrate at 5.2x
concentration, while cats=none is dead flat. Grinsztajn — the suite that
decides ships — has NO such datasets (its curation removes high-card cats
and has 0 multiclass), so no lever targeting the gap can clear the protocol,
and every ship to date has been selected on a suite blind to this regime.
Ensembles proved the danger concretely: synth screen said 64W-24L, Grinsztajn
said 7W-16L. Separately, decision-selection on one fixed 59-set suite
accumulates composition overfitting (warning sign already on record:
cross_features +1.5% 51W/8L on Grinsztajn, TabArena-Lite flat, OpenML gate
only +0.4%).

Fix: add a REAL high-card suite (`hc:`) to the decision tier. Not synthetic
(the generator's prior must not vote on ships), not a Grinsztajn split (both
halves inherit the same blind spots).

Hard constraints (unchanged): pure Python loaders (sklearn fetch_openml is
fine); TabArena sealed in every form — its DATASETS must not enter this
suite; one benchmark at a time; script files, never `python -c`; C: has
~4 GB free (put sklearn's cache on A:); trust file reads over scrolled
stdout.

## Step 0 — overlap audit (hard gate for every candidate)

Build the exclusion lists BEFORE looking at any candidate's data:

1. **TabArena dataset list** (names + OpenML dataset/task IDs). Source:
   the local TabArena env/caches on `A:\code` (see `/tabarena` skill) — the
   tabarena/tabrepo metadata enumerates tasks offline. Fallback: the task
   list file in the TabArena GitHub repo. Dataset NAMES/IDs only — no
   results of any kind are consulted; using the membership list to AVOID
   contamination is the correct use of it.
2. **Grinsztajn 59**: keys from `_add_grinsztajn_datasets` /
   `benchmarks/data_cache/`.
3. **OpenML one-shot gate panel (29 sets)**: `_add_openml_datasets` in
   run_benchmarks.py. Overlap here would un-independent the gate.
4. **PMLB 25**: the list in run_benchmarks.py.

Match on OpenML dataset ID first, fuzzy name second (case/underscore/hyphen
variants — "Amazon_employee_access" vs "amazon-employee-access").
Deliverable: a candidate x {tabarena, grinsztajn, gate, pmlb} matrix
committed into this file. Any TabArena, gate, or Grinsztajn hit = OUT.

## Step 1 — candidate pool (verify EVERYTHING in-session; IDs from memory)

Selection is by DATA PROPERTIES ONLY — n, d, task, cardinality, missingness
— measured without fitting any model. No model result may influence
inclusion/exclusion (else the suite is born cherry-picked). Degenerate
exclusions allowed: load failure, constant target, exact duplicates.

| Candidate | OpenML id (verify) | Task | Why |
|---|---|---|---|
| KDDCup09_appetency | 1111 | binary | 50k x 230, high-card cats, missing, 98/2 imbalance |
| KDDCup09_churn | 1112 | binary | same family |
| KDDCup09_upselling | 1114 | binary | same family |
| kick (Don't Get Kicked) | 41162 | binary | 72k, vehicle model/trim/color, card ~1k |
| Click_prediction_small | 1220 | binary | ad/site/user IDs (verify the version with raw cats) |
| porto-seguro | 42742 | binary | 595k (subsample), ps_car_11_cat card ~104 |
| airlines | 1169 | binary | 539k (subsample), carrier/airport codes card ~300 |
| sf-police-incidents | 42344 | binary | 2.2M (subsample), address = extreme card |
| open_payments | 42738 | binary | physician/company IDs (TabArena suspect — audit) |
| Amazon_employee_access | 4135 | binary | RESOURCE card ~7k (TabArena suspect — audit) |
| okcupid-stem | 42734 | 3-class | 50k, job/speaks/ethnicity cats |
| Diabetes130US | 4541 | 3-class | 100k, diag codes card ~900 (TabArena suspect — audit) |
| Traffic_violations | 42345 | 3-class | high-card (verify id/version) |
| Allstate_Claims_Severity | 42571 | reg | 188k x 130 cats, card ~300 |
| Mercedes_Benz | 42570 | reg | 4k x 376, X0-X8 cats |
| KDD98 (UCI direct) | — | reg | classic high-card; heavy, optional |

Target frozen suite: **N >= 12** after audit cuts, with >= 8 sets having
max cardinality >= 50 on >= 1 feature, >= 2 regression-with-cats, and (if
the multiclass columns land, see step 2) >= 3 multiclass-with-cats. If the
audit kills too many, expand by property search on OpenML metadata
(cardinality filters) BEFORE any model is fit — never after.

Freeze = a `HC_DATASETS` list (name, openml id, task, subsample cap)
committed to run_benchmarks.py + the rationale table here. After the first
baseline run, the list only changes with a version bump (same discipline as
synthgen freezes).

## Step 2 — harness work

1. `_add_highcard_datasets()` + `hc:<name>` keys + `--highcard` flag,
   mirroring the Grinsztajn registration/run-only convention
   (run_benchmarks.py ~line 1045). Loader: `fetch_openml(data_id=...,
   as_frame=True)` -> cats from dtype, cached. Set `SCIKIT_LEARN_DATA=A:\code\sklearn_data`
   (C: has no room). Subsample cap 100k rows, deterministic
   per dataset (fixed seed, NOT the harness seed — splits must stay the only
   seed-dependent thing; follow the existing builder pattern `(scale, rng)`).
2. Big-set hygiene: KDD09/porto/sf-police are 50-500 MB fetches — first run
   warm-caches serially (Grinsztajn HF-401 precedent); flaky network = just
   relaunch.
3. summarize.py: add `Multi F1%` / `Multi Brier%` columns rendered only when
   multiclass records exist. **Blended-strength formula UNCHANGED** — the
   north star (HarmonicMean of reg/binary columns) is a separate user
   decision; these columns are report-only for now. Ripples: bench_status
   caption, make_pareto ignores the new columns.
4. Guards: near-solved analogs will appear (98/2 imbalance makes F1-macro
   fragile and %-vs-best Brier can explode when best -> 0) — the existing
   `skip_best_below` / NEAR_SOLVED guards apply per-column; verify they
   fire, don't invent new ones preemptively.
5. Tests: registration idempotent, loader smoke on the 2 smallest sets,
   frozen-list-matches-doc assertion, a TabArena-overlap regression test
   (frozen list x audited exclusion list = empty intersection).

## Step 3 — baseline + shakedown (report-only, one session)

1. `python benchmarks/run_benchmarks.py --highcard --seeds 3 --save
   benchmarks/results/hc-baseline.txt` — all available models. Print the
   aggregate table (always, unprompted).
2. Shakedown read: near-solved artifacts, degenerate F1 on the imbalanced
   sets, load flakes, per-model failures (XGB/LGBM native-cat paths on
   card ~7k can be slow or OOM — record, don't fix).
3. **First-read deliverable**: measured CatBoost-vs-ChimeraBoost gap on hc
   (Brier winrate + %-of-best), compared against the synth entity-slice
   prediction (entity Q4 = 91% CatBoost winrate). This doubles as the
   fidelity test of SynthGen v2's entity-cat prior — feed the answer into
   the v3 watch items in benchmarks/synthgen/PAYOFF.md either way.
4. The suite is SHADOW (report-only) for this one session — it becomes a
   co-decider only after the shakedown confirms no metric artifacts.

## Step 4 — wire into the decision protocol

After shakedown passes: /experiment tier 2 becomes "Grinsztajn + hc, with
per-suite sign tests reported separately and a pooled union verdict."
A change that wins only on one suite needs a mechanism story for why.
Update: `.claude/skills/experiment/SKILL.md`, CLAUDE.md benchmark section,
memory (algorithm history + a new hc-suite entry). Exact ship-rule weighting
between the suites = Nathan's call at first live use, not hardcoded now.

## Decision points reserved for Nathan

- Ship-rule weighting Grinsztajn vs hc (step 4).
- Whether multiclass enters the blended north star (step 2.3 keeps it out).
- Whether KDD98 (heavy, license-awkward) is worth including.

## Explicit non-goals

No synthetic data in the decision tier. No Grinsztajn train/test split. No
TabArena dataset reuse (audit is a hard gate). No OpenML-gate rotation here
(separate program). No library changes. No north-star formula change.

## Acceptance

- [x] Audit matrix committed; frozen list has zero TabArena/gate/Grinsztajn overlap
- [x] N >= 12 frozen by properties only; >= 8 sets max-card >= 50; >= 2 reg-with-cats
- [x] `--highcard` runs end-to-end; summarize renders (incl. multiclass columns if present)
- [x] hc-baseline.json on disk + aggregate table printed (14 sets, 3 seeds, 2026-07-15)
- [x] First-read writeup: real hc gap vs synth entity prediction (CONFIRMED: 86–88% real vs 91% synth), logged in PAYOFF.md v3 watch items
- [x] Suite declared co-decider (shadow this session; only artifact is `cjs` near-solved) + skill/CLAUDE.md/memory updated

---

# Audit results & frozen suite (implemented 2026-07-15)

Exclusion lists used (NAMES/IDs only — no results consulted; membership used to
AVOID contamination, the sanctioned use): **TabArena 51** from
`.../tabarena/nips2025_utils/metadata/curated_tabarena_dataset_metadata.csv`
(offline, on A:); **Grinsztajn 59** from `GRINSZTAJN_DATASETS`; **OpenML gate 29**
from `OPENML_SUITE`; **PMLB 25** from `PMLB_DATASETS`. Match = OpenML id first,
normalized/substring name second. Every candidate was verified against OpenML
metadata and its true per-column cardinality measured on the 100k subsample
WITHOUT fitting any model. The self-contained overlap regression test
(`tests/test_highcard.py::test_no_suite_overlap`) re-checks the frozen 14 against
all four lists on every run.

## Step 0 — overlap audit (hard cuts)

| Candidate | id | hit | verdict |
|---|---|---|---|
| KDDCup09_appetency | 1111 | TabArena exact (`kddcup09_appetency`, did 46939) | **OUT** |
| KDDCup09_churn | 1112 | shares appetency's exact 50k×230 feature matrix (incl. the high-card cats) | **OUT** |
| KDDCup09_upselling | 1114 | same identical feature matrix as TabArena appetency | **OUT** |
| Amazon_employee_access | 4135 | TabArena exact **and** gate id 4135 (`Amazon_access`) | **OUT** |
| Diabetes130US | 4541 | TabArena exact **and** Grinsztajn exact | **OUT** |
| Allstate_Claims_Severity | 42571 | Grinsztajn `reg_cat` exact | **OUT** |
| Mercedes_Benz | 42570 | Grinsztajn `reg_cat` exact (`Mercedes_Benz_Greener_Manufacturing`) | **OUT** |
| airlines | 1169 | shares airline-delay domain + carrier/airport high-card cats with Grinsztajn `Airlines_DepDelay_1M` | **OUT** |

The KDDCup09 trio share one identical 50k×230 feature matrix (byte-identical
OpenML qualities: max card 15415, 8,024,152 missing); only the target differs and
TabArena tests the appetency target — so tuning cat-handling on churn/upselling
would leak the appetency mechanism. Name-matching can't see this; the reasoning is
recorded. `airlines`↔`Airlines_DepDelay_1M` is a domain/cat overlap with the
primary decision suite (Grinsztajn), cut to keep the suites independent.

## Step 1 — property-based cuts (degenerate / not high-card / unstable)

| Candidate | id | reason | verdict |
|---|---|---|---|
| Click_prediction_small | 1220/1216/1218/1219 | pre-encoded: 1 symbolic feature, max card 2 (no raw high-card cats in any OpenML version) | **OUT** |
| open_payments | 42738 | empty/all-NaN target on load; 1 symbolic feature | **OUT** |
| kdd_internet_usage | 4133 | target undefined + `who` is a pure ID column (card == n) | **OUT** |
| KDDCup99 | 1113 | 18 classes in the 100k subsample with min_class_count=0 (zero-support) → stratified split / macro-F1 undefined | **OUT** |
| avocado_sales | 41210 | OpenML marks the version INACTIVE (reproducibility) | **OUT** |
| ipums_la_99-small | 378 | "high card" is mis-encoded continuous income (ftotinc/inctot as nominal), not entity cats | **OUT** |
| nfl_games | 42143 | default target undefined; top cat `date` is semi-ID | **OUT** |
| SpeedDating | 40536 | structural leakage: `dec`+`dec_o` functionally determine `match` → near-solved | **OUT** |
| Census-Income (KDD) | 4535 | target undefined + income-prediction domain overlaps gate `adult` | **OUT** |
| socmob | 541 | legit but redundant tiny (1156) low-card (17) reg filler; cut to keep suite tight | **CUT** |

The audit killed BOTH clean reg-with-cats candidates (Allstate, Mercedes →
Grinsztajn), so replacements were found by property search on OpenML metadata
(`MaxNominalAttDistinctValues ≥ 50`, cardinality filters) BEFORE any fit —
yielding `wine-reviews`, `colleges`, `kdd_ipums_la_97-small`, `cjs`. KDD98
(id 23513, 25847-card reg) surfaced in that search but is DEFERRED (heavy +
license-awkward — Nathan's call, per the plan).

## Frozen HC suite — N=14

Composition: **4 binary + 4 multiclass + 6 regression · 10 sets with max-card ≥ 50
(incl. 3 high-card regressions) · 6 reg-with-cats · 4 multiclass-with-cats.** All
fetched by OpenML id with a 100k deterministic subsample (random_state=0). This
table is the source of truth; `tests/test_highcard.py::test_frozen_matches_doc`
asserts it equals `HC_DATASETS`.

| hc key | id | task | n (→100k sub) | n_cat | max card (feature) | note |
|---|---|---|---|---|---|---|
| hc:kick | 41162 | binary | 72,983 | 18 | 1063 (Model) | 88/12 imbalance, missing |
| hc:porto-seguro | 42742 | binary | 595,212→100k | 31 | 104 (ps_car_11_cat) | 96/4 imbalance, missing |
| hc:sf-police-incidents | 42344 | binary | 538,638→100k | 5 | 15165 (Address) | balanced |
| hc:kdd_ipums_la_97-small | 993 | binary | 7,019 | 27 | 191 (occ1950) | occupation/industry codes |
| hc:okcupid-stem | 42734 | multiclass | 50,789 | 17 | 7019 (speaks) | 3-class, missing |
| hc:Traffic_violations | 42345 | multiclass | 70,340 | 19 | 3830 (Model) | 3-class; watch `Description` leak |
| hc:cjs | 473 | multiclass | 2,796 | 2 | 57 (TREE) | 6-class, small |
| hc:eucalyptus | 188 | multiclass | 736 | 5 | 27 (Sp) | 5-class, small low-card buffer |
| hc:wine-reviews | 41275 | regression | 129,971→100k | 9 | 31959 (designation) / 15633 (winery) | points∈[80,100]; free-text `description`/`title` auto-dropped |
| hc:colleges | 42727 | regression | 7,063 | 12 | 6039 (zip) / 59 (state) | pell-grant fraction |
| hc:house_prices_nominal | 42563 | regression | 1,460 | 43 | 25 (Neighborhood) | Ames SalePrice |
| hc:black_friday | 41540 | regression | 166,821→100k | 4 | 7 (Age) | low-card cats |
| hc:employee_salaries | 42125 | regression | 9,228 | 8 | 2264 (date_first_hired) / 694 (division) | Montgomery salaries; `full_name` ID auto-dropped |
| hc:Moneyball | 41021 | regression | 1,232 | 6 | 39 (Team) | small |

Shakedown watch items for the baseline read (step 3): `Traffic_violations`
(`Description`/`Charge` may leak the 3-class target → near-solved); `cjs`
(all models ~1.0 F1 in the dry run → near-solved multiclass); `porto-seguro`
(96/4 → macro-F1 fragile, %-of-best Brier can explode as best→0); competitor
native-cat paths on card 7k–15k (`speaks`, `winery`, `Address`) slow (CatBoost
~2 min/fit on wine-reviews) — record, don't fix.

### Harness robustness fixes surfaced by the shakedown (not library changes)

The first baseline runs exposed three loader/runner issues that the HC regime hits
but the curated suites never did. All are infrastructure fixes (they let the
benchmark COMPLETE and read correctly; they do not touch the library or bias any
decision):

1. **Per-model skip, not run-abort** (`_run_seed_task`): sklearn HGB has a hard
   255-category cap and raises on high-card cats (`kick` card 1028, `sf-police`
   13.8k, etc.); an uncaught raise aborted the whole 42-draw run. A runner that
   raises is now recorded as skipped (`None`, printed `[skip]`), exactly like an
   uninstalled competitor. sklearn is thus absent on the 6 card>255 sets and
   present on the other 8 — an honest record of its structural high-card limit.
2. **pandas-3.0 `str` dtype detected as categorical** (`_is_categorical_dtype`):
   pandas 3.0 returns free-text columns as the new `str` dtype (repr `"str"`),
   which the old `_is_cat` missed → text columns routed to the numeric branch →
   `astype(float)` crash (`wine-reviews`). Also un-hid genuine string cats that
   metadata had missed — `employee_salaries` is actually a **high-card** reg
   (`date_first_hired` 2264, `division` 694), not the low-card (37) the OpenML
   nominal-only metadata implied.
3. **Near-unique cat columns dropped** (`_HIGHCARD_ID_FRAC = 0.9`, HC builder
   only): row identifiers / free text (`wine-reviews` `description` 94% unique &
   `title` 93%; `employee_salaries` `full_name` 99.9%) carry no repeated-level
   signal — the opposite of the entity-cat regime — and only add noise + fit
   cost, so the loader drops them. The threshold sits above every genuine
   high-card cat (highest kept: colleges `zip` 85.5%).
