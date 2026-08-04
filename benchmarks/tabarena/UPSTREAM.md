# ChimeraBoost on TabArena — upstream status

**TabArena is a sealed holdout.** Everything below is report-only: no number
from it may influence a source change.

## Where things stand

| | |
|---|---|
| Model package | Merged upstream in [PR #358](https://github.com/autogluon/tabarena/pull/358) on 2026-06-18. Lives at `packages/tabarena/src/tabarena/models/chimeraboost/` in `autogluon/tabarena`. |
| Warm-up | Our [PR #436](https://github.com/autogluon/tabarena/pull/436) (numba pre-compile outside the fit timer) was closed in favour of a general warm-up API the maintainers built themselves, then applied in [PR #442](https://github.com/autogluon/tabarena/pull/442), merged 2026-07-13. ChimeraBoost was the first model to use it. |
| Runs of record | `suite="tabarena-2026-06-30"` (superseded — its fit times include our numba JIT) and `suite="tabarena-2026-07-13"` (the warm-started rerun). Both were default-only comparisons. |
| Row variants | A full run produces **three** rows per method that sets `can_hpo=True` and registers a search space — default, tuned, and tuned + ensembled. The current pool is 23 / 19 / 19. We set both, so the tuned rows are automatic; they are generated from `hpo.py`'s space at TabArena's convention of 200 random configs plus the single all-defaults `manual_configs=[{}]`. Nobody has to ask for them. |
| Version pin | `pip_extra=("chimeraboost>=0.14.1",)` — a floating lower bound, so any fresh environment installs the current PyPI release. |
| Who runs it | The maintainers, on their cluster, on their schedule. We do not produce the published numbers. |

`upstream/chimeraboost/` is a **verbatim mirror** of the four upstream files, kept
so we can diff after each release. Do not edit it to fix things; fix them in a PR
upstream and re-sync. Synced 2026-08-04 from `autogluon/tabarena@main`.

## Asking for a rerun on a new release

`(method, suite)` is the unique artifact key, so a rerun needs a **new
`MethodMetadata` entry with a fresh `suite`/`date`** or it collides with the
07-13 artifacts — exactly the pattern the maintainers used for the warm rerun.
The pin should move to the release being claimed (`chimeraboost>=0.30.0`), since
nothing else in the metadata records which version produced a row.

Before asking, re-run these four checks against the mirror and the target
release. All four were green for 0.30.0 on 2026-08-04:

1. **Constructor parameters.** Every key in `hpo.py`'s search space, plus
   `n_estimators` / `early_stopping` / `thread_count` / `random_state`, must
   still be a real parameter of both estimators.
2. **Fit signature.** `fit()` must still accept `cat_features`, `eval_set`, and
   `callbacks`, and a callback taking `(iteration, train_loss, val_loss, model)`
   must still stop the fit when it returns True — that is how the wrapper
   enforces TabArena's time budget.
3. **Warm-up coverage.** With a cold `NUMBA_CACHE_DIR`, `chimeraboost.warmup()`
   followed by two identical fits must show no material gap between the first
   and the second. Any gap is a default-path kernel that warmup misses, and it
   would land back inside their fit timer. (0.30.0: worst gap 0.2 s in fit,
   nothing in predict; warmup itself 18.3 s cold.) `benchmarks/cold_start.py
   --kernels` gives the per-kernel breakdown.
4. **Integration test.** `model_unittest.py`, run against the mirrored
   `model.py`, must pass all three problem types.

## Open: the registered search space is from June and has fallen behind (2026-08-04)

The tuned and tuned + ensembled rows are drawn from `hpo.py`'s space, so it is
the one artifact in this integration that is ours to get right, and it has not
been touched since June — eight releases ago. Three of its thirteen knobs now
have adaptive or validation-selected defaults, and the space samples an explicit
value for each, which turns the rule off:

| Knob | Default today | What the space does |
|---|---|---|
| `learning_rate` | `None` → fades 0.07 at ≤5,000 training rows to 0.1 at ≥15,000 (0.30.0) | `Real(0.03, 0.3, log=True)` — an explicit rate short-circuits the fade entirely |
| `linear_leaves` | `None` → validation-selected (0.14.1) | `Categorical(False, True)` |
| `leaf_estimation_iterations` | `None` → auto (0.19.0) | `Int(1, 5)` |

The single all-defaults config in `manual_configs=[{}]` is in the pool, so the
tuner *can* reach the adaptive rules — but only by taking that one config whole.
It can never combine an adaptive learning rate with any other tuned setting.
Adding `None` to each of the three as a sampled option fixes that and costs
nothing; `Categorical(None, ...)` is legal and all three estimators accept the
sentinel (verified 2026-08-04).

**Decide this before the maintainers run the tuned rows**, since that run is 200
configs across every fold and repeat and there is no cheap do-over. Decide it on
dev-side evidence only — Grinsztajn, high-card, PMLB — never on a TabArena
result. Not yet actioned; no upstream PR opened.

## Known stale text in the upstream files (fix in the next PR there)

* `model.py` says the auto learning rate "is pinned at 0.1 under ES". Since
  0.30.0 it fades from 0.07 at 5,000 training rows or fewer up to 0.1 at 15,000
  or more. The conclusion the comment draws is still correct — the
  early-stopping path never consults `n_estimators`, so the 10,000 cap remains
  learning-rate-neutral — only the description of the rate is out of date.
* `model.py`'s docstring still points at a `../REGISTER.md` that only ever
  existed in this repo, as staging for the original PR.
