"""Full-research-mode cascade engine for closing the CatBoost gap.

An extremely-efficient Bayesian cascade (idea -> handful -> medium -> large) that
benchmarks ChimeraBoost variants without ever touching the sealed TabArena-Lite
holdout. The efficiency levers, in order of impact:

  * One fit = the whole validation curve. ChimeraBoost records the per-iteration
    validation loss in ``validation_history_`` (free on every fit), so a single
    fit yields the metric at *every* iteration count at once -- no refitting at
    multiple ``n_estimators``.
  * Paired same-split deltas. A variant's curve is compared against the baseline's
    on the *same* train/val split -- a far lower-variance signal than comparing
    final test scores across different splits, so fewer datasets/seeds reach
    significance.
  * Persistent dataset cache (download once ever) + shared-baseline cache (only
    the variant refits per idea).
  * Tiered cascade with sequential sign-test early-stop: minimal fits to a verdict.

See the package modules: ``datasets`` (tiers + cache), ``curves`` (paired curve
comparison), ``runner`` (three-way split + fast/promotion fits), ``ideas`` (the
pre-registered queue), ``cascade`` (orchestration + gates), ``report``.

GUARDRAIL: this engine must never load TabArena. OpenML is a one-shot gate, not
an iteration target. See memory feedback_tabarena_lite_is_sealed_holdout.
"""
