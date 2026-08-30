# Complexity audit & refactor program

Status: stages 0-3 shipped. Started 2026-08-30.

Goal: every non-frozen function at or below ruff C901 complexity 10, enforced
permanently, with **zero behavior change** — each refactor stage is
bit-identical, gated by `benchmarks/identity_snapshot.py` (exact
`np.array_equal` over ~30 configs) plus the full test suite, never by
benchmarks.

## The ratchet

- `pyproject.toml`: `select = ["E4", "E7", "E9", "F", "C901", "RUF100"]`,
  `max-complexity = 10`. The rule set is pinned explicitly (not
  `extend-select`) and ruff is pinned to `0.16.5` (dev extra + CI), because
  both default rules and McCabe counts shift across ruff releases and would
  silently move the ratchet.
- Baseline: 20 C901 violations at threshold 10, each carrying a `# noqa: C901`
  marker — temporary ones name the stage that removes them; permanent ones say
  "frozen" and point here. RUF100 fails the build if a noqa outlives its
  violation.
- CI: the `lint` job in `.github/workflows/tests.yml`.
- Units: this program's planning audit used a stdlib-AST McCabe counter that
  also counts boolean operands, asserts, ternaries and comprehension clauses —
  it reads roughly 2× ruff's C901 (e.g. the regressor `_fit_single`: 77 by
  audit count, 36 by ruff). All ratchet numbers are ruff units.
- Radon was considered and skipped: different units from the enforcement tool,
  an extra dependency, no gate integration.

## Refactor stages (one PR each, merged before the next starts)

| Stage | Target | Ruff CC | State |
|---|---|---|---|
| 0 | tooling, snapshot coverage, guards | — | shipped (PR #97, 2026-08-30) |
| 1 | `sklearn_api._validate_hyperparams` | 25 | shipped (2026-08-30) |
| 2 | `sklearn_api._validate_fit_input` | 34 | shipped (2026-08-30) |
| 3 | `quantile_api.fit`, `sklearn_api._fit_bagged` | 16, 12 | shipped (2026-08-30) |
| 4 | the three `booster._fit_impl` loops | 16/14/16 | pending |
| 5 | classifier `_fit_single` | 31 | pending |
| 6 | regressor `_fit_single` | 36 | pending |
| 7 | `tree.build_oblivious_tree` + closeout | 13 | pending |

Rules for every stage: move code, never rewrite expressions; helpers receive
rng objects and consume draws in original order; `stacklevel` +1 per frame a
`warnings.warn` moves down; error messages byte-identical, first-fire order
unchanged; shared-object identity preserved (`prep_cache`). The verification
sequence per stage is in the plan file and boils down to: clear the numba
cache, `identity_snapshot.py check` (trust
`benchmarks/results/identity_check.txt`), full pytest, ruff, and for stages 4
and 7 the strict-timing golden run plus `profile_fit.py` before/after.

## Frozen — permanent noqa, never refactored

Numba kernels get no readability payoff from decomposition: njit inlining
changes codegen, `cache=True` invalidation churns every machine, and
`parallel=True` semantics are sensitive to function boundaries. Every one is
already pinned by oracle tests, which provide the safety that CC reduction
buys elsewhere.

| Function (ruff CC) | Why frozen |
|---|---|
| `tree._build_split_descend` (21) | docstring carries a clause-by-clause bit-identity argument vs three oracles |
| `tree._build_split_descend_q` (21) | deliberate int64 twin — numba has no generics; stays textually parallel to the float sibling |
| `tree._build_split_descend_vec` (25) | vector-leaf hot path; no safety payoff for recompile risk |
| `tree._build_and_split` (15) | test oracle (tests/test_tree_kernels.py pins the fused kernel against it) |
| `tree._best_split` (13) | test oracle, paired with `_build_histograms_into` |
| `tree._solve_small` (13) | pinned bit-exact vs numpy (tests/test_small_solver.py) |
| `tree._linear_leaf_fit` (17) | numba, cache=True; churn for zero benefit at threshold 10 |
| `tree._select_kth` (11) | same |
| `tree._shap_forest_linear` (24) | deepest nesting in the repo (6), prange body; extraction risks numba inlining behavior |
| `target_encoding._factorize_numeric` (14) | plain Python but the predict-latency hot path, with an audited (not derived) equivalence to the dict loop; optional post-program candidate |

Not flagged but worth recording: the 8-kernel predict family and the 4-kernel
leaf-quantile family in `tree.py` look like copy-paste to a metric but are
intentional — the `_serial` twins exist because `parallel=True` launch
overhead loses at small n, and the float/quantized/vector variants exist
because numba cannot express them generically.

## Snapshot coverage note (why stage 0 exists)

The original identity snapshot fit every config on 2,000 rows; with
`validation_fraction=0.2` that leaves 1,600 post-split — below
`CROSS_MIN_SAMPLES = 2000` — so no config ever ran the cross-feature race,
the selection-rounds audition, or forced cross, and there was no bagged or
conformalized-quantile config. Stage 0 added n=6000 configs plus `bag` and
`mq3_conf`, with `_EXPECT` assertions that fail the save/check if a pinned
path ever goes dark again.
