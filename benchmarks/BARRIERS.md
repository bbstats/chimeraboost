# Barriers — ideas this project has already closed

Companion to `GATE_ROBUSTNESS.md`. That file is about **readings** that fool us.
This one is about **ideas**: each entry is a closure we paid for, stated so that
the next proposal in its family can be recognised before it spends a run.

The scarce resource is benchmark wall-clock — `--decide` is one run at a time and
takes hours. An idea barred here is not forbidden; it is *known to owe an
argument*. Clearing a barrier means saying which specific finding below is wrong
or does not apply, in the plan file, before the run.

`python benchmarks/barrier_check.py "<idea in a sentence>"` matches an idea
against the tags here and prints what it must clear. Run it before tier 1.

Entries are the machine-readable source for that script: `### B<n> — <title>`
followed by a `tags:` line. Keep that shape when adding one.

---

### B1 — Audition knobs are structurally inert below the audition thresholds
tags: audition, selection_rounds, small-data, sus25, sus50, linear_leaves, cross_features, budget

Below `LINEAR_LEAVES_MIN_SAMPLES=1000` and `CROSS_MIN_SAMPLES=2000` no audition
runs at all, so no audition knob can move anything there. 47 of 48 synth sets
under 2000 rows were bit-identical across the sel25 arms.

Consequence: never read a small-data stratum as evidence about an audition knob,
and never propose an audition change *aimed* at small data.

*Incident*: Sel25, 2026-07-31 (`SELECT_PLAN.md`).

### B2 — The rung-3 refit amplifies a bad audition, so audition changes must be judged at `refit_full=True`
tags: audition, selection_rounds, refit, refit_full, replay, quality, budget, rung

Same knob, same suite, same seeds: at rung 3 sel25 read 6W-20L-10T, p=0.009, mean
−0.478%; at rung 2 (`refit_full=False`) it read 10W-16L-10T, p=0.327, mean
**+0.409%**. Rung 3 replays the audition winner's structure over all rows, so a
wrong pick propagates instead of being diluted.

Consequence: selection **fidelity** is worth more than it was under rung 2. Judge
every audition-cheapening idea at the shipped default, and re-value
fidelity-improving ideas upward.

*Incident*: Sel25, 2026-07-31 (`SELECT_PLAN.md`, results `20260731-164927.json`).

### B3 — Partial CatBoost mechanism ports each regress somewhere
tags: catboost, categorical, cat, encoding, onehot, one-hot, combinations, target-statistics, port, leaf-estimation

Seven CatBoost-inspired levers, seven kills. Each partial port helps some
categorical sets and regresses another badly, while the win it chases is already
banked in the current defaults (the all-categorical `cat_combinations` auto-rule,
the binary `linear_leaves` default, plain boosting with size-adaptive
`min_child_weight`).

The gap looks like an emergent property of CatBoost's *integrated*
ordered-boosting-on-permutations machinery, not any single transplantable part.

*Incident*: research cascade, `benchmarks/research/SUMMARY.md`.

### B4 — Ordered boosting is closed
tags: ordered, ordered-boosting, permutation, catboost, small-data

CatBoost never runs ordered boosting — it is `Plain` at every dataset size. The
hypothesis family this project carried for months is retired, and our own dormant
`ordered_boosting` flag with it. Its one size-dependent default is the learning
rate, which is worth 57% of its small-data edge and which we shipped.

*Incident*: SMALLDATA, 2026-08-01 (`SMALLDATA_PLAN.md`, `probe_catboost_ablation.py`).

### B5 — A shrinkage estimator cannot correct a common in-sample/out-of-sample bias
tags: shrinkage, pooling, pooled, blend, calibration, quantile, coverage, band, leaf-values

Shrinkage only corrects the component of the error that **varies across** the
units being shrunk. P15 shrank each leaf's quantile spread toward the pooled band
to fix under-coverage; train coverage sat at the nominal 0.80 at every pseudo-mass
while test coverage never arrived, because the pooled band is computed from
in-sample residuals too and carries the same bias.

Consequence: diagnose whether the defect is a common or a varying component
before designing any correction that rearranges in-sample quantities.

*Incident*: P15 quantile spread smoothing, 2026-07-31 (PR #63, `LEAFTUNE_PLAN.md`).

### B6 — Broad hyperparameter tuning buys nothing that generalises
tags: tuning, hyperparameter, search, random-search, reg_lambda, depth, grid, sweep, defaults

A PMLB random search over the whole space found essentially nothing that
transferred; `min_child_weight` was the one exception, and it shipped. LEAFTUNE
then reproduced the same result on `reg_lambda` specifically against an exact
cross-validated grid: 3 wins, 3 losses, 2 ties of 8 datasets, median exactly
0.000%, none near-solved.

Consequence: the defaults are Grinsztajn-tuned and near a good optimum. A tuning
proposal needs a mechanism story, not a wider grid.

*Incident*: PMLB random-search study; LEAFTUNE P-series, 2026-07-28.

### B7 — Shadow CV: do not rebuild it
tags: cv, cross-validation, folds, shared-structure, out-of-fold, oof, reuse

Built end to end 2026-07-28 and thrown away. Three independent reasons: shared
structure diverges at a 0.70 median so there is nothing to share; sharing it leaks
(out-of-fold log loss 0.376 on pure noise, below the 0.693 floor); and an exact
lockstep tier costs ×1.04 against the naive loop, so even done right it buys
nothing.

*Source*: the program record. It was built and then deleted, so no code or plan
file survives in the repo to re-check these numbers against.

### B8 — Subsample is a dead axis, on both speed and strength
tags: subsample, bagging-fraction, row-sampling, mvs, regularization, stochastic

Killed 2026-07-17 on speed, re-tested 2026-07-30 after the MVS numba work made
`subsample<1` fit 1.32–1.44× faster, and killed again on strength: 0.8 read
primary 63W-67L median −0.000% and **Brier 31W-55L median −0.730%**; 0.7 was
worse. The implicit-regularisation mechanism was absent, not merely small — the
`noise_level` OLS coefficient came back at t = −0.03.

*Incident*: `SUBSAMPLE_PLAN.md`, both dates.

### B9 — Bagging ships only output-identical engineering or more data per member
tags: bagging, ensemble, n_ensembles, members, oob, quality-ladder

The bagging program is closed with `Ens8` as the blessed mode. Two-member bagging
is net negative (2 members lose to 1). Everything that changed member *behaviour*
without giving members more data was killed; what shipped was engineering that
left predictions identical, plus `refit_members`, which reclaims the per-member
data tax.

*Incident*: `BAGGING_PLAN.md`; `refit_members`, 2026-08-01.

### B10 — The remaining grow-kernel objects are below their ceilings
tags: kernel, numba, grow, histogram, scatter, scan, speed, micro-optimisation, dtype

Phase-0 measured every candidate's ceiling as a share of estimator fit before any
was written: multiclass copy trims 0.9%, int32 leaves ~1–2%, uint8 bins ≤2–3% at
small n and ~0 at n≥50K — all killed at ceiling. The one double-digit object left
is the fused scatter+scan split (36.5–57.6% of fit), and it is **FP-drift class**:
it cannot be made bit-identical, so it cannot pass the numerical-identity goldens
that every kernel change here has had to pass.

The one lever that was implemented anyway (L-ridge, a row-major restructure) came
back at ×0.49–0.67 at n≥37.5K and was reverted — the ridge is accumulator-bound,
not gather-bound.

*Incident*: `GROW_PLAN.md`, Phase 0 and Phase-1 verdicts, 2026-07-18.

### B11 — Isotonic-in-sample is a broken stopping rule
tags: calibration, isotonic, early-stopping, es, stopping, brier

Calibration-aware early stopping was killed as a default. The reason is
structural and is the part worth carrying: an isotonic map fitted on the same
rows the stopping decision then scores cannot report honestly on them, so the
stopping curve it produces is optimistic exactly where it is being read.

*Source*: the program record — the shipped calibration is temperature scaling
(`sklearn_api.py`), and no isotonic stopping path exists in the tree today, so
this entry is history rather than something to inspect. Anyone reopening it
should re-derive the numbers rather than quote them.

### B12 — Config portfolios die on cost, not on headroom
tags: portfolio, config, race, multi-config, audition, selection, oracle

A2 passed every strength bar and failed on cost. The test-set oracle ceiling was
+2.587% and validation selection was real (21W-9L-6T, p=0.043) — but it recovers
only **20% of that ceiling**, and racing four configurations projected to
**1.90×** fit time against a 1.35× bar.

The reason generalises past A2: **a k=100 audition is nearly a whole fit whenever
the full fit is short**, which is common (mean 323–382 rounds; multiclass runs
~100). Any proposal that adds auditions must price them against the *short*-fit
case, not the average.

*Incident*: A2 Phase 0, 2026-07-25 (`A2_PLAN.md`).

### B13 — Replay is exact and cheap, and misreads the axis that matters
tags: replay, leaftune, tune_leaves, refit, structure-transfer, sweep

Replay round-trips bit-identically and costs a median 5.2× less per configuration
than re-growing. But its grid agreed with an exact grid's chosen cell on only 2 of
8 datasets, and on the one dataset where the parameter genuinely mattered
(`gr:reg_cat/Brazilian_houses`, +4.4% for exact tuning) replay picked the wrong
end and gave up **6.8%** on test.

Consequence: replay is a screening instrument, not a selection instrument. Its
out-of-fold regret diagnostic *did* detect its own failure — keep that diagnostic
in any future version.

*Incident*: LEAFTUNE, 2026-07-28.

### B14 — No margin rule separates the const-vs-linear race at k=100
tags: audition, const-vs-linear, linear_leaves, margin, race, selection_rounds, early-exit, budget

The const-versus-linear validation race genuinely crosses late on about a third of
regression selections, and on the step-0 curves the two arms' overlap is total —
no margin or early-exit rule at k=100 separates them. `k_ll=500` restores fidelity
but collapses the 1.50× audition speedup to a projected 1.11×.

Consequence: an early-exit rule on this race owes a measurement against those
curves. A *per-leg* budget is the untested direction, because the cross race and
the const-vs-linear race have different crossing behaviour.

*Incident*: `PARETO_PLAN.md`, "Known residual"; Sel25's re-open condition.

---

## Adding an entry

An entry earns its place when a closure is **paid for and general** — a measured
kill whose reason will recur. A one-off negative goes in its plan file and the
algorithm history, not here. Keep the `### B<n> — title` / `tags:` shape so
`barrier_check.py` keeps working, and state the incident that proved it.
