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
- ~10% of ids are **saturated canaries** (kr-vs-kp analogs): y is a
  deterministic cell rule over 2–4 final columns (cat-cross lookup or
  axis-aligned cells), floor 0, baseline fits near the ceiling — a flag that
  "wins" there is injecting variance.
- Freeze-time filters (TabICLv2): degeneracy, ExtraTrees-learnability,
  cat-combination tractability. ~20% of candidates rejected.

## Determinism & versioning

Content is a pure function of (VERSION, id) via `SeedSequence([VERSION_SEED,
id])` with per-stage/per-node child streams; the harness seed only moves the
train/test split, and builders ignore `--scale`. Keys carry the version
(`syn:v1/031`) so a generator change can never silently pair against old data
in `compare_runs.py`. Golden hashes (`tests/golden_synthgen.json`) trip on
numpy RNG stream drift → bump VERSION, re-freeze, never re-pin.

## Usage

```
python benchmarks/run_benchmarks.py --synth [--synth-suite smoke|screen|full] \
       [--synth-n N] --seeds 3 --save            # standard results JSON
python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost
python benchmarks/synth_report.py BASE.json NEW.json   # factor attribution
python benchmarks/synth_report.py RUN.json              # excess-vs-floor view
python benchmarks/synth_report.py RUN.json --realism    # cross-model checks
```

Suites (frozen 2026-07-14, `suites.py`): smoke 6 sets (~3 min), screen 182
sets / 401K rows (~15 min wall, all models, jobs 5), full 242 sets / 1.63M
rows (~45–60 min). smoke ⊂ screen ⊂ full, so pairing stays valid across tiers.

## Validation (adoption gate)

`backtest.py` re-runs the screen with one known-outcome lever flipped per arm
(cross_features, linear_leaves, cat_combinations, patience, ordered boosting,
min_child_weight, depth×2, lr) and scores sign agreement against the project
ledger. Gate: ≥7/9 agreement AND the cat_combinations canary slice
(saturated & cats) not positive. Failures re-weight the meta-distribution into
VERSION+1. Suite verdicts never ship anything alone — Grinsztajn remains the
decision suite, OpenML the independent gate.

## Files

`recipe.py` meta-distribution → `scm.py` DAG/functions → `emit.py` dataset +
floors → `api.py` keys/cache/hash · `calibration.py` corpus bootstrap ·
`filters.py` freeze-time QC · `suites.py` frozen ids · `freeze.py` scan/select
· `backtest.py` adoption gate · `harvest_metadata.py` corpus refresh ·
`../synth_report.py` attribution.
