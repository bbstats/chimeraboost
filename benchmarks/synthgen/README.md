# SynthGen — prior-sampled synthetic decision suite

Tier-1 of the decision pipeline (synthetic → dev panel → Grinsztajn, OpenML
gate). Generates unlimited realistic tabular datasets from an SCM prior
(TabPFN/TabICLv2/Mitra recipe family, numpy-only), frozen into versioned
suites, so feature/HP screens run on hundreds of paired datasets with known
generative factors and exact Bayes floors — and wins/losses can be *attributed*
to data properties instead of guessed.

**TabArena is excluded in every form** (its 51 members are dropped from the
calibration corpus by OpenML suite id 457; nothing here reads its data,
results, or metadata). Sealed holdout stays sealed.

## How data is made

- Observable marginals (n, d, task mix, categorical make-up, cardinality,
  missingness, imbalance) are **bootstrapped jointly from real public dataset
  metadata**: 1,644 unique-name active OpenML datasets after cleaning
  (auto-generated families dropped, TabArena excluded, version-deduped), with
  curated rows (OpenML-CC18 + in-repo Grinsztajn/OpenML suites) upweighted 50/50.
  Snapshot: `corpus_marginals.json` (regenerate with `harvest_metadata.py`).
- Latent DGP factors are literature priors: layered random DAG at a sampled
  interaction depth; per-node functions {linear, neural (8 activations),
  oblivious tree ensemble, product, plateau}; aggregation {sum, product, max,
  logsumexp}; root distributions {normal, uniform, MoG, heavy-tail}.
- Features are *views* of nodes (rescaled, warped, discretized-to-categorical,
  MCAR-masked). Targets are emitted in node space: regression = warped node +
  N(0, σ) (floor = σ); classification = softmax of distances to class
  references, y sampled from the true p (floor = its sum-form Brier, the
  harness convention).
- **Entity categoricals (v2):** ~40% of cat columns are latent entities, not
  discretized views — Zipf-ish level frequencies (levels centered ≥8, ≤64),
  per-level effect ~N(0, σₑ), σₑ ~ loguniform(0.3, 1.0), injected into the
  target readout before noise (floors stay exact); the observed column is the
  label string only, plus a few singleton rare levels (the unseen-at-train
  stress ordered target statistics exist for).
- ~13% of ids are **saturated sets** (kr-vs-kp analogs): y is a deterministic
  cell rule over 2–4 final columns (cat-cross lookup or axis-aligned cells),
  floor 0. Canary status is **earned, not assumed** (v2): freeze fits the
  default baseline on every saturated candidate across the harness's own 3
  seed-splits and only ids verified at the ceiling (mean excess Brier ≤ 0.005,
  worst seed ≤ 0.01 / mean RMSE ≤ 1.1σ) enter `suites.CANARIES` — a
  flag that "wins" there is injecting variance. Unverified cat-cross sets are
  genuinely-hard cat interactions (car analogs), scored as their own slice.
- Freeze-time filters (TabICLv2): degeneracy, ExtraTrees-learnability,
  cat-combination tractability. ~21% of candidates rejected. The screen's
  n-mix is stratified (n<2000 share capped at 35%) so greedy small-n packing
  can't skew the suite tiny.

## Determinism & versioning

Content is a pure function of (VERSION, id) via `SeedSequence([VERSION_SEED,
id])` with per-stage/per-node child streams; the harness seed only moves the
train/test split, and builders ignore `--scale`. Keys carry the version
(`syn:v2/031`) so a generator change can never silently pair against old data
in `compare_runs.py`. Golden hashes (`tests/golden_synthgen.json`) trip on
numpy RNG stream drift → bump VERSION, re-freeze, never re-pin. Canary ids
live in `suites.CANARIES` (freeze-time knowledge; meta stays a pure function
of the key).

## Usage

```
python benchmarks/run_benchmarks.py --synth [--synth-suite smoke|screen|full] \
       [--synth-n N] --seeds 3 --save            # standard results JSON
python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost
python benchmarks/synth_report.py BASE.json NEW.json   # factor attribution
python benchmarks/synth_report.py RUN.json              # excess-vs-floor view
python benchmarks/synth_report.py RUN.json --realism    # cross-model checks
```

Suites (v2, frozen 2026-07-14, `suites.py`): smoke 6 sets (~3 min), screen
136 sets / 401K rows (~30 min wall, all models, jobs 5 — CatBoost is 50–70×
ChimeraBoost on synthetic targets and dominates the wall clock), full 211
sets / 1.61M rows (~1–2 h). smoke ⊂ screen ⊂ full, so pairing stays valid
across tiers.

## Validation (adoption gate)

`backtest.py` re-runs the screen with one known-outcome lever flipped per arm
(cross_features/linear-leaves ablations, cat_combinations, patience, ordered
boosting, min_child_weight, depth×2, lr) and scores sign agreement against the
project ledger. Capacity/lr arms (depth4/depth8/lr03) are judged excluding
saturated sets, which reward low capacity by design. Gate: ≥7/9 agreement AND
the canary slice (CANARIES & cats) non-empty and not positive. Failures
re-weight the meta-distribution into VERSION+1. Suite verdicts never ship
anything alone — Grinsztajn remains the decision suite, OpenML the gate.

**v1 verdict (2026-07-14): PASS, 8/9 arms.** Highlights: `crossfeat_off`
−0.94% overall, **−3.30% on the pre-registered interaction-depth≥2 numeric
slice (n=50)**; `catcombo` mixed on ordinary data (W18-L17) with +27.4% on the
six car-analog cat-interaction sets; `patience300` flat; `lr03` −0.99%.
Sole disagree: `depth4` (+0.31% — wins concentrate on saturated cell-rules and
small/categorical lookups where 4 levels suffice; reality-shaped slices agree
with the ledger). Known v1 biases for the v2 freeze: (1) targets run slightly
shallow — raise interaction-depth/width mix; (2) categorical columns are
discretized latents, not entity effects — CatBoost's high-card moat is absent;
(3) the screen has zero cat-bearing verified-at-ceiling canaries — require ≥3,
and earn canary status by a freeze-time fit check instead of by construction;
(4) mcw large-n slice leans positive (registered ~neutral) — watch.

## Files

`recipe.py` meta-distribution → `scm.py` DAG/functions → `emit.py` dataset +
floors → `api.py` keys/cache/hash · `calibration.py` corpus bootstrap ·
`filters.py` freeze-time QC · `suites.py` frozen ids · `freeze.py` scan/select
· `backtest.py` adoption gate · `harvest_metadata.py` corpus refresh ·
`../synth_report.py` attribution.
