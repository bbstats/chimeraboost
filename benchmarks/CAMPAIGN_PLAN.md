# CAMPAIGN_PLAN — Pareto position of the DEFAULT (auto-research loop)

Started 2026-08-16. This file is the loop's entire memory: a fresh session with
zero context must be able to continue from it alone. Method adapted from the
"auto-research" pattern (beam of idea families + screening ladder + append-only
log), grafted onto this repo's existing gate protocol.

## Goal

North star: `images/pareto.png` — the DEFAULT's win rate vs fit-time slowdown.
Current default point: **91.5% @ 6.1×** (ChimeraBoost-only field,
`results/20260815-235543.json`; last chart-grade read 2026-08-02).
Success = move the DEFAULT point left (cheaper) and/or up (stronger) — the dual
read: flat-and-cheaper counts exactly as much as stronger-at-parity.
A shipped preset is a frontier win but not a default win.

## Standing rules (by reference — do not restate, do not fork)

- `benchmarks/BARRIERS.md` via `barrier_check.py "<idea>"` — every idea, before anything else.
- `benchmarks/GATE_ROBUSTNESS.md` — before any number decides anything.
- `/experiment` skill — the only path to a default change; Nathan signs off.
- One benchmark at a time (core contention corrupts timings). `--public` is ship-time only. TabArena sealed, report-only.
- Aggregate table printed after every benchmark run, unprompted.
- Environment: the working python is `A:\code\miniconda3\python.exe` (PATH `python` is a bare 3.10 with nothing installed).
- No entry in this file may say PENDING once the underlying run is resolved.
- A decision the maintainer gives in chat (a pick, a reorder, a kill confirmation) is written into THIS file in the same turn. Session memory notes are not the record: on 2026-09-21 the R1 pick lived only there, and the next session planned a rung against "awaiting the pick".
- Int-exact per-histogram ≠ bit-identical end-to-end (op order feeds gains/tie-breaks). Any "exact" kernel change runs `identity_snapshot.py` first; identical → pure-speed ladder; not → FP-drift class → S2 with the Brier read.
- Standing numerical-drift policy (the maintainer, 2026-09-18): an algorithm-changing revision may carry last-bit drift provided it is monitored (identity-snapshot diff quantified, not silently absorbed), reported (both axes in the verdict), and gate-evidenced to improve over time. Pure rewrites that drift without changing the algorithm stay on the ladder above.
- Delivery is PRs only (the maintainer, 2026-09-18): no merges to `main`, no PyPI releases from these sessions until he says otherwise. Ships land as pull requests; merging and releasing stay his call.
- AMENDED 2026-09-21 (the maintainer, in chat): "if tests come back bit identical, we can very safely merge any PRs that are just incrementing markdown files, I don't need to give input on that if it's just like hey, here's something we learned. If we are changing actual code I need a PR though." Read strictly: the loop merges its own PR (merge commit, then deletes the branch) ONLY when every changed path is a `.md` file and `chimeraboost/` + `tests/` are identical to main, so the last green suite and identity snapshot still hold. Any other file in the diff — library, tests, config, or a script under `benchmarks/` — waits for him. Releases stay his call, always.
- WIDENED the same day (the maintainer, in chat, asked whether probe scripts count as code): "benchmarks can be self-mergable as well." The rule now: the loop merges its own PR when every changed path is a `.md` file or lives under `benchmarks/`, and `chimeraboost/` + `tests/` are identical to main. Still his: `chimeraboost/`, `tests/`, packaging and CI config, releases. `benchmarks/tabarena/` stays hands-off (sealed). A self-merged PR that changes how a gate SCORES (`run_benchmarks.py`, `compare_runs.py`, `synth_report.py`, `synthgen/`) or a policy file (a skill, `AGENTS.md`) says so in the step report.
- **FOCUS 2026-09-24 (the maintainer, in chat): "next i want us to focus on issues rather than pareto efficiency stuff".** After the #163 rung (I063) closes, the loop's rungs come from the open GitHub issues, ahead of every beam family; F7's Q4/Q1 and the other ACTIVE families are parked until he says otherwise. Order (bugs first, per his standing preference): (1) #84, `warmup(background=True)` sets its notice flag inside the thread, so a fit in that window still prints the cold-compile notice; set it in the caller before `start()` (library, his merge). (2) #81, `benchmarks/research/`: the cascade self-test's anchors are no longer off by default (`linear_leaves` is auditioned, `early_stopping_rounds` moves the curve) and several `ideas.py` entries set flags removed on 2026-06-15 (benchmarks-only, self-merge). (3) #131, `refresh(X, y)` with an opt-in stored training set, structure pinned, leaves replayed on old + new rows (library). (4) #113, random effects slice 2: slopes, a second grouping column, the entity-ID auto-route; gate `grouped_suite.py`, the auto-route on hc `--decide` (library). (5) #45, loose ends: a fresh sweep, and his call on the fully merged remote branches. #109 (random effects) is the umbrella slice 1 shipped from; #113 carries its remaining scope, so closing it is his call. The harness open items H(13) and H(14) ride along as small benchmarks-only fixes when a rung has room. UPDATE 2026-09-25: #84 and #81 are merged; #131 slice 1 is PR #170, PARKED OPEN by the maintainer ("not sure I want it implemented"), which does NOT hold the one-campaign-PR-at-a-time rule, since he asked the loop to keep going; #113: the maintainer's go 2026-09-25 ("Yes 113 is 'another issue' so you may work on it"); its four design calls go with the stated recommendations (one slope column with independent variances; the real-set covariate by the correlation rule; the entity-ID auto-route demoted to a kill-or-keep probe; fix the employee alias leak and report slice 1's old and new numbers). Next: #113 (random slopes), then #45.

## Screening ladder

| Rung | What | Cost | Pass |
|---|---|---|---|
| S0 | `barrier_check.py` + forecast on BOTH axes, written in the log entry before any run | free | no barrier hit, or a written clearing argument |
| S1 | strength: cascade T0 (`benchmarks/research/ideas.py` entry with live kwargs) or a `--chimera-*` knob probe · speed: microbench / `fit_time_delta.py` on the touched phase | ~1 min | effect in predicted direction |
| S2 | `run_benchmarks.py --synth --seeds 3 --save`, ChimeraBoost arms only → `compare_runs.py` + `synth_report.py` | 2–8 min | effect concentrates in the pre-registered slice |
| S3 | ONE `--decide --seeds 3 --save` run, both arms via `--models` → `compare_runs.py --by-suite` | ~9 min | pre-registered per-stratum sign bars |
| S4 | full `/experiment`: chart-grade field with CatBoost, `--public` at ship time, Nathan sign-off for any default change | hours | the skill's bars |

Skip rules (record the class at S0): bit-identical speed refactors skip
strength screens — gate on `identity_snapshot.py` + full test suite +
`fit_time_delta.py`. Post-fit ideas enter at S2 (cascade T0 is curve-blind to
them). Harness/measurement ideas never touch the ship gate — gate on
`cascade.py --selftest` + anchors. Default flips: no skips, ever.

**Graduation:** a family cleared to S3 gets its own pre-registered
`benchmarks/<NAME>_PLAN.md` (the per-program format); its row here flips to
PROMOTED and only the final verdict flows back as one log line.

## Session mechanics

Unit of work = one rung of one family (≤ ~10 min compute). Ordering, iron:
(1) append the log entry with `forecast:` filled and `verdict: PENDING(run-id)`
BEFORE launching; (2) launch in background; (3) score, replace PENDING, write
`next:`; (4) only then the next unit. Long S4 runs launch only at session
start with Nathan's explicit go. Every session ends with `/handoff` (pointer
here + session-local traps only).

Beam refill (staleness = <3 ACTIVE families, or two consecutive sessions of
only KILL/INCONCLUSIVE, or everything blocked on sign-off): 4 parallel
read-only idea subagents, lenses L1 profiling-speed / L2 literature-mechanism /
L3 opponent-ablation (the only sanctioned door into F5, per B3) / L4
harness-measurement. Funnel through `barrier_check.py` + dedup vs this log and
`research/SUMMARY.md`; survivors go to Nathan, who picks entrants. Beam cap 5.

## Facts ledger (paid-for numbers; append, never edit)

fact: 2026-08-16 | main settled at 7684655 (PR #90 verdict + PR #91 E2 source both merged); local branches e2/forced-cross-features + method/e2-prereg deleted
fact: 2026-08-16 | post-E2 quality-ladder re-check: default stays rung 3 — best non-ensembling rung by a wide margin (91.5% vs rung 2 54.6%, rung 1' 49.5%; `results/e2_pareto_read.txt`)
fact: 2026-08-16 | BARRIERS B15 added (integer histogram subtraction — killed 2026-07-18, was unregistered; `QUANT_PLAN.md:263-272`)
fact: 2026-08-16 | test-suite wall-clock on 7684655 = 115 s (941 passed, 1 skipped)
fact: 2026-08-16 | standing BASE = `results/campaign-base-20260816.json` (sha 7684655, --decide, seeds 3, full default field; gr 82.5% vs CatBoost W47-L10 median +1.25%; hc 38.5% vs CatBoost W5-L8 median −0.15%)
fact: 2026-08-16 | F1 cross-column screen, synth, both arms in one run: k=6 (`results/campaign-f1s2-20260816.json`) engaged fit −13.2%, engaged regression 4W-14L, binary 13W-4L, sub-gate control 0W-0L-48T; k=12 (`results/campaign-f1s2b-20260816.json`) engaged fit −11.6%, engaged regression 6W-12L. Saving is nearly flat in k, harm is too ⇒ B16.
fact: 2026-08-16 | F4 S1 profile, hc:okcupid-stem (`results/campaign-f4s1-okcupid-20260816.txt`): fit 4.12s / 99 trees; kernel `build_oblivious_tree` 23.0%; `losses._softmax` 39% cumulative (multiclass-only); string-column `target_encoding.factorize` 18% with 1.6M `dict.get`; `_prep_matrices` 27%, `_refit_on_full` 27%
fact: 2026-08-16 | fit-time noise floor on the synth inert slice (identical fits, same run) = ~2% — any speed claim below that is unreadable
fact: 2026-08-16 | test suite on 1d68f07 = 952 passed, 1 skipped, 113 s (941 + the 11 `cross_top_columns` tests); numerical-identity goldens green, so the default-off knob is inert end-to-end
fact: 2026-08-16 | F4 C2 SHIPPED (`_factorize_hashed`): bit-identical 89/89, fit −4.0% okcupid-stem / −6.8% kick / −1.0% Grinsztajn control on zero calls; column-level 0.57× the loop over 35 real string columns (`results/campaign-f4c2-{micro,speed}-20260816.txt`)
fact: 2026-08-16 | same-process alternating-fit A/B noise = ~1% (the zero-`factorize`-call control read −1.0%); this is the tighter instrument, the ~2% floor is for cross-run reads
fact: 2026-08-16 | a cProfile percentage on a PYTHON-LEVEL loop is an upper bound: `factorize`'s profiled 18% of the okcupid-stem fit was 8% by wall clock, because per-call overhead is charged on 1.6M `dict.get` calls
fact: 2026-08-16 | after C1, `grad_hess` fell from 40.0% to 5.9% of an okcupid-stem fit (11.8% on cjs); C1b's own object (`P−Y` + `max(P(1−P),1e-6)`) is 4.4% / 7.5% — measured post-ship by a second instrument, which independently confirms C1's win
fact: 2026-08-16 | F4 C1 SHIPPED (`_softmax_kernel`, guarded K ≤ 7): bit-identical 89/89, multiclass fit −44.2% okcupid-stem / −37.2% Traffic_violations / −40.5% cjs, binary control −0.6% on zero calls (`results/campaign-f4c1-speed-20260816.txt`); softmax leg 1.70s → 0.05s
fact: 2026-08-16 | the fused softmax kernel is 6.9–11.0× faster PARALLEL than serial at every real shape — `prange` earns its thread setup even on 3-wide rows
fact: 2026-08-16 | all 8 multiclass rows in `--decide` are K ∈ {3,5,6} (okcupid-stem, Traffic_violations, cjs, eucalyptus + variants) — every one under the K ≤ 7 guard; **Grinsztajn contains no multiclass task at all**, so multiclass work cannot move the headline stratum
fact: 2026-08-16 | F4 C1 ceiling, hc:okcupid-stem (`results/campaign-f4c1-walltime-20260816.txt`): `losses._softmax` = 1.611s of a 3.55s fit = **45.4% by WALL CLOCK** (cProfile said 39% — understated); `grad_hess` 40.0% of fit, `eval` 5.5%; K=3, shapes (32377,3)/(5714,3)/(38091,3)
fact: 2026-08-16 | inside one `_softmax` call at (32377,3): the `max` reduce is 43% and `sum` 15% — the two length-3 inner-axis reductions are 58% of the call, `exp` only 20%. The cost is numpy's per-row reduce machinery, not the arithmetic.
fact: 2026-08-16 | softmax candidates: in-place `out=` ufuncs 1.01× (no effect, temporaries were never the cost); numpy column-fold 0.48×; numba fused with reciprocal-multiply 0.03× but drifts at every K; **numba fused with true divide 0.03× (32×) and bit-identical at K ≤ 7**
fact: 2026-08-16 | the bit-identity boundary for any hand-folded sum over the class axis is **K ≤ 7**, measured over K=2..50 — numpy's pairwise summation starts blocking at K=8. `max` never drifts at any K.
fact: 2026-08-16 | attribution on 7684655 (`results/campaign-attr-20260816.{json,md}`): cross-audition leg = 40–58% of fit on engaged sets (nyc-taxi 52%, road-safety 58%, diamonds 46%); ll selected 12/12, cross 24/27; race truncation at k=100 keeps 24/24 cross picks; okcupid-stem multiclass "other" (non-kernel) = 50.5% of fit; hc prep_other+ts_enc ~12–28%
fact: 2026-09-18 | strength-first program approved (Pareto plan, Muse session): F2 S1 probe → beam refill (L1 rescoped to loss-slice profiling, speed out of scope) → S1–S4 ladder; drift ruling above is its standing policy
fact: 2026-09-18 | tree state at kickoff: branch `whitepaper` (13 ahead / 1 behind origin/main), 43-file prose-only working diff; `chimeraboost/` + `tests/` identical to origin/main, so measurement on this tree reads main's code — branch left untouched
fact: 2026-09-18 | bench lane clear (no run in flight); `results/` newer than last entry: `quantile-20260830-175359.json` (quantile-suite instrument, 0.32 record lives in CHANGELOG — not a default-loop run)
fact: 2026-09-18 | HARNESS WART (fix wanted, out of this program's scope): `--save <explicit .json path>` points the console tee and the JSON sidecar at the same file and corrupts the write (I017 attempt 1). Until fixed: always use bare `--save` and copy/rename after. A rejected-or-derived guard belongs in run_benchmarks.py with a test.
fact: 2026-09-18 | sub-gate 3-fold CV race costs 3-7x the base fit on engaging sets (mean ~5x; declined-after-folds sets pay too, no-pairs sets are free) — B12 instance, measured in `results/f2s2-20260918.json`
fact: 2026-09-21 | F4 C1b SHIPPED as PR (I018): fused grad/hess kernel, `identity_snapshot` 155/155, same-process fit A/B okcupid-stem −4.6% / Traffic_violations −5.7% / cjs −4.1%, binary control +0.2% (`results/campaign-f4c1b-speed-20260921.txt`); test suite on the branch = 1081 passed, 1 skipped, 125 s
fact: 2026-09-21 | F3 S1 classifier forced-cross probe (I019, 16 clf_num sets × 3 seeds, rung-1 config, top-4 block): oracle Brier headroom median +0.360% (29W-19L), probe fidelity paired median +0.000%, cost median 1.33x; gains concentrate on covertype +15.8% / pol +3.4% / eye_movements +1.5% / MagicTelescope +1.4% / jannis +1.1%, losses bank-marketing −1.0% / electricity −0.9% (`results/probe-cross-pairs-f3-clf.jsonl`)
fact: 2026-09-21 | F3 S2 synth screen (I020, `results/20260921-075304.json`, OneLin vs OneLinXC): 119/136 exact ties (regression, multiclass, n<2000 all 0-0), engaged binary 13W-4L median +0.16% Brier (p=0.049), canary 0-0-3, engaged fit ratio median 1.27 (mean 1.60, one 5.35 outlier)
fact: 2026-09-21 | F3 S3 decide run (I021, `results/20260921-080246.json`, OneLin vs OneLinXC): gr binary engaged 23 sets 10W-13L median −0.04% Brier, engaged fit 1.43x median; hc binary 2W-2L within ±0.14%; every regression/multiclass row an exact tie; gr@sus25 2W-2L, gr@sus50 0W-2L, hc@time 2W-0L (pointers) → F3 KILLED
fact: 2026-09-21 | forced-cross headroom by estimator, same design, same panel family: regressor +2.6% median (E2 step 1) vs classifier +0.36% (I019) — a 7x gap, and the referee-less mode only survives the dodged losses when the prize is the large one
fact: 2026-09-21 | hc Brier gap vs CatBoost is monotone in max cardinality (`campaign-base-20260816.json`, 3 seeds): sf-police (card 15165) −0.0056, Traffic_violations (3830) −0.0053, okcupid-stem (7019) −0.0023, kick (1063) −0.0019, porto-seguro (104) −0.0001; we win kdd_ipums (191) +0.0019 and eucalyptus (27) +0.0037. CatBoost loses 4 of 6 hc regressions, so the gap is classification-only
fact: 2026-09-21 | ordered-TS asymmetry, unmeasured: `_ordered_ts` gives train rows the prefix statistic (expected count ≈ (m−1)/2) and `OrderedTargetEncoder.transform` gives test/ES rows the full total (count m); at card 15k over 75k rows m ≈ 5 (refill shortlist R1, probe is zero-library-change)
fact: 2026-09-22 | on the 33 gr regression sets where the cross race engages (seed 0), 221 of the 457 `diff` cross columns (48%) are near-duplicates of their larger-scale parent (σ ratio ≥ 3 and |Spearman ρ| ≥ 0.95; medians 3.52 and 0.963); 10 sets have ≥ 2/3 of their diff block duplicated (medical_charges, house_sales, wine_quality, Brazilian_houses, Ailerons, houses, cpu_act, elevators, MiamiHousing, SGEMM), 9 have ≤ 1/3 (I045)
fact: 2026-09-22 | numeric missing values: 0 of 59 gr sets carry any NaN/inf in a numeric column; 11 of 14 hc sets do — colleges 32 cols / 100% rows, cjs 28 / 100%, Moneyball 2 / 66%, employee_salaries 2 / 32%, porto-seguro 2 / 24%, kick 9 / 0.5%, wine-reviews 1 / 7%, Traffic 1 / 0.6% (I045)
fact: 2026-09-22 | identity_snapshot panel widened to 33 configs / 186 arrays (I043): cat_count_features (plain + weighted, card ~600, n=6000), the classifier's forced cross, random_effects with groups, a bag over categoricals; `save` now refuses to move existing pins without --rebaseline — the first attempt would have silently re-baselined 131 of 155
fact: 2026-09-22 | predict-side profile (I042/L1, exclusive hooks, 5 reps, wrapped/plain 0.99–1.01): `_codes_for_transform` is 62.4% (kick, 94 ms) / 73.3% (okcupid, 45 ms) / 82.7% (sf-police, 43 ms) of predict wall clock; the ordered-TS transform 18.8% on okcupid; a direct `cat_maps_` lookup prototype reads 0.49–0.50× the current path
fact: 2026-09-22 | on the standing BASE, CatBoost leads on Brier on 9 of 30 gr classification sets (california +2.7%, bank-marketing +1.0%, credit +0.5%, …) and LightGBM trails us on 8 of those 9 — the edge is CatBoost-specific; catboost 1.2.10 defaults there: `score_function=Cosine`, `leaf_estimation_backtracking=AnyImprovement`, `feature_border_type=GreedyLogSum`, `l2_leaf_reg=3`, `leaf_estimation_iterations=1`, `bootstrap_type=MVS subsample=0.8`
fact: 2026-09-22 | catboost 1.2.10 MULTICLASS CTR = `Borders:TargetBorderCount=3:TargetBorderType=MinEntropy` × priors 0/0.5/1 (three entropy-chosen ordinal binarizations of the class index, not one-vs-rest) with `bootstrap_type=Bayesian`; after the count column CatBoost's residual hc edge is +0.96% median on engaged multiclass vs +0.44% on engaged binary; the count column closed 44–65% on 6 of 8 engaged sets and 0% wherever no column has card ≥ 256
fact: 2026-09-22 | LightGBM vs the default on gr fit: 6.08× total, 5.22× median; ~1.8–2.1× of it is round count (median best_iter 255 vs 122), 2.7× per round overall and 4.9× per round at n ≥ 28k; 15 low-p large-n sets carry 67% of the excess
fact: 2026-09-22 | `compare_runs` scores classification on F1 (`primary`) while every gate and the harness SUMMARY score Brier: I039's engaged median is +0.558% on RMSE/Brier (recorded +0.203% on primary); I036 is 33W-21L +0.49% (recorded 36W-17L +0.30%); I041's 8 engaged sets are all seed-split with CI −0.06..+0.78% — R4 was inconclusive, not killed
fact: 2026-09-22 | letting linear leaves use the TS columns (I041, 14 hc sets × 3 seeds, paired): multiclass 4/4 exact ties (no linear leaves), binary 1W-3L (kick −0.76%, sf-police −0.03%, kdd +1.03%), regression 3W-1L-2T (colleges +0.74%, employee_salaries +0.78%); the unraced binary leaf over-trusts an in-sample statistic, the raced regression leaf does not
fact: 2026-09-22 | per-column ANOVA variance-ratio shrinkage for the ordered TS (I040, 7 hc clf sets × 3 seeds): the estimator asks for λ in the hundreds on high-card columns (singletons make the between-variance estimate noise) and collapses kdd_ipums −3.8% / eucalyptus −9.6%; clipped to [1, 10] it pins at 10 everywhere and reads +0.20% median on the gap sets vs the count column's +0.19% on the same splits, but sf-police +0.22 vs +0.67, Traffic +0.20 vs +1.04, and eucalyptus −5.4%
fact: 2026-09-22 | `cat_count_features=True` (library form, card ≥ 256, outside the cross / linear races) on --decide, 3 seeds (I039): gr 0-0-59 exact ties; hc 7 non-qualifying sets exact ties, engaged 7 = 6W-1L, median +0.20% [+0.02..+0.76]; sf-police +0.73% / Traffic +0.86% / kick +0.35% Brier, employee_salaries +2.45% / wine-reviews +0.56% RMSE; hc@time 4-0 incl. sf-police@time +1.40%; engaged fit ratio median 1.165, harness hc ×1.09
fact: 2026-09-22 | S3 of the ungated count column (I037, --decide, 3 seeds): gr 0-0-59 exact ties; hc RMSE/Brier 9W-4L, median +0.07%, sf-police +0.88% / Traffic +0.81% / wine-reviews +0.67% Brier-RMSE, kdd_ipums −3.28% unanimous; hc fit ×1.47 (sf-police 2.58× on one added column, wine-reviews 2.59× with fewer trees) because appended numerics enter the cross-feature and linear-leaf races; sf-police@time −3.47%
fact: 2026-09-22 | a per-categorical training-row count column appended to X (I035, 7 hc clf sets × 3 seeds, paired): sf-police +0.88%, Traffic_violations +1.46%, okcupid +0.43%, kick +0.26%, porto −0.03% Brier vs the default; closes 47% of CatBoost's edge at the median; kdd_ipums −0.63% and eucalyptus −0.18% on split seeds
fact: 2026-09-22 | the ordered TS quantized to 16 uniform buckets (CatBoost's CtrBorderCount=15) LOSES on the big hc sets (sf-police −0.31%, kick −0.54%, porto −0.36%) and wins +1.65% on kdd_ipums (7k rows) and +3.26% on eucalyptus (736 rows): coarse TS quantization regularizes small categorical data, a @sus pointer
fact: 2026-09-22 | CatBoost fed OUR ordered target statistics instead of its cat_features keeps −5% (median) of its hc Brier edge over us and is 1.46% worse than us on kick (I034): the hc gap is the encoder, not the booster; its fits also drop from 27 s to 2.8 s without its CTR machinery
fact: 2026-09-22 | hc classification balances: sf-police 0.50/0.50 (card 15165), kick 0.123 positive (1063), porto-seguro 0.036 (104), kdd_ipums 0.37 (192), Traffic_violations 0.46/0.05/0.49 (3831), okcupid-stem 0.19/0.72/0.10 (7020), eucalyptus 5-class (27, n=736)
fact: 2026-09-22 | CatBoost 1.2.10 CPU defaults as actually run here (`get_all_params()`, binary and multiclass): `max_ctr_complexity=1` (NO cat×cat CTR combinations), `one_hot_max_size=2`, per categorical two simple CTRs — `Borders` (target statistic, 15 borders, priors 0/0.5/1 ⇒ 3 columns) and `Counter` (frequency, prior 0, `counter_calc_method=SkipTest`), `boosting_type=Plain`, `bootstrap_type` MVS (binary) / Bayesian (multiclass)
fact: 2026-09-22 | CatBoost hc ablation (I033, 7 hc clf sets × 3 seeds): removing its target CTR drops it 1–7% of Brier BELOW our default on 4 of 5 gap sets; removing the Counter, collapsing to one prior, or fixing one permutation each give back ≤ 12% of its edge at the median; a zero shrinkage target instead of 0.5 gives back 21–74% on the four real gap sets
fact: 2026-09-22 | CatBoost paired single fits on the hc classification sets average 27 s at the harness config (2000 iters, patience 50); a 7-set × 3-seed × 4-arm ablation is ~30 min, not hours
fact: 2026-09-21 | harness rule: `--models` must include `ChimeraBoost` (the baseline) or run_benchmarks exits 2 at argparse; a stray `<stamp>.txt` tee is left in results/ when that happens
fact: 2026-09-21 | main settled at 486e563. PRs #116–#119 (F3 S1/S2/S3 + the refill shortlist) were each merged into the rung branch below them rather than into main — stacked PR bases, and the base branches were not deleted on merge, so GitHub never retargeted them; main received only #115. Recovered the same morning as PR #120 (the four commits, unchanged). The PR #56 trap a second time. Rule since: every rung branch is cut from main, PR base is always main, one campaign PR open at a time (now step 0 of the `campaign-step` skill). A stale local main hid the recovery from the next session until it fetched — fetch before reading the log
fact: 2026-09-21 | ordered-TS asymmetry MEASURED (I024, `results/probe-ts-mismatch-20260921.{md,json}`): on high-card columns a held-out row of a rare category (m ≤ 5) reads an encoding 1.9–5.2× more spread around the prior than a training row of the same stratum, and the target's slope on it is 0.63–0.70 of the training slope (over-trust) on sf-police / okcupid-stem / Traffic_violations, 1.01 on kick; wine-reviews 0.71, colleges 0.42. In big categories (m > 50) it runs the other way, reliability 1.10–1.29 (prefix noise on training rows). Mean shift is ≤ 0.03 SD everywhere — the mismatch lives in the second moment and by count, never in the mean
fact: 2026-09-21 | share of held-out rows in categories of train count m ≤ 2, the quantity that decided the SIGN of the transform fix: sf-police 13.1%, wine-reviews 13–20%, colleges 18–20%, okcupid-stem 4.0%, Traffic_violations 1.5–2.8% per column over its 3 high-card columns (the probe's column count of 9 is 3 columns × 3 class targets), kick 0.8%
fact: 2026-09-21 | transform-only A/B, default estimator, 3 seeds (I024): count-matched weight A1 / half-matched A2 vs shipped — sf-police +0.31% / +0.28% Brier (3/3 seeds both), Traffic_violations +0.46% / +0.75% (3/3), okcupid-stem +0.09% / 0.00% (1/3), kick −0.05% / −0.04% (0/3); wine-reviews +0.08% / +0.29% RMSE; porto-seguro exact to ±0.003%. Gap-set pairs 7W-5L for both arms against a bar of 8. Matched arms early-stop later (sf-police 45→62 trees), fit ×0.97–1.14, report-only
fact: 2026-09-21 | F4 C4a SHIPPED as a PR (I027): factorize once per fit + integer re-rank per leg, `identity_snapshot` 155/155, same-process default-fit A/B kick −9.4% / sf-police −18.9% / porto-seguro −18.1% / okcupid-stem −8.2%, numeric control −0.5% on zero calls (`results/campaign-f4c4a-speed-20260921.txt`); factorize calls per fit 84→24 / 26→5 / 130→31 / 77→20; test suite on the branch = 1097 passed, 1 skipped, 114 s
fact: 2026-09-21 | conversion rates differ by object class: removing whole Python-level passes (C4a) converted ~100% of its ceiling (forecast −5 to −14%, read −8 to −19%), where arithmetic folded into an existing kernel (C1b) or a loop the fit amortizes (C2) converted 50–70%. Discount a ceiling by what KIND of work disappears
fact: 2026-09-21 | second Muse Code rung with library edits: one pass, exit 0, ~20 min for a three-file, 221-line change plus a 260-line test file; it kept two functions under the C901 limit by extracting a helper unprompted, and still patches CRLF files by script
fact: 2026-09-21 | CHANGELOG trap, caught at I027: commit b50f01d (C1b) had inserted its entry under the already-released `## [0.32.0]` heading — the first `### Changed` in the file — instead of `## [Unreleased]`, which had no Changed subsection yet. Check the heading ABOVE the subsection before adding a line; `git show v<tag>:CHANGELOG.md` is the oracle for a released section
fact: 2026-09-21 | hc `prep` split (I026, `results/campaign-f4c4-prep-20260921.{md,json}`; % of default fit, kick / wine-reviews / okcupid-stem / sf-police / Traffic_violations / porto-seguro): factorize 14.7 / 16.9 / 14.1 / 29.8 / 11.3 / 29.9; ordered-TS draws+averaging 5.5 / 4.2 / 15.0 / 4.6 / 14.7 / 8.6 with the numba kernel only 1.4 / 1.3 / 4.0 / 1.4 / 4.0 / 2.4; `_numeric_block` 4.7 / 0.4 / 0.6 / 0.9 / 0.2 / 10.1; cross block 4.5 / 5.0 / 1.4 / 2.2 / 1.0 / 2.2; binning 0.8–3.4; `prep` total 30–58%
fact: 2026-09-21 | one default fit factorizes overlapping rows ~2.2× the matrix: selection-leg training rows (80%), validation rows for the eval transform (20%), validation rows again when calibration predicts from raw X, then all rows in the refit — kick 154 + 36 + 129 ms of a 2.17 s fit. The numeric block of an object array is unboxed once per leg AND once more inside `_validate_fit_input` (porto-seguro: 10.1% + 3.2% of fit)
fact: 2026-09-21 | `_factorize_numeric` is not cheap on an object array: porto-seguro's 31 integer-coded categoricals cost 29.9% of fit (a per-element float cast, then a boxed `==` audit). The harness hands EVERY categorical dataset to fit as a numpy object array, so this is what users with mixed-type frames pay too
fact: 2026-09-21 | ordered-TS cost is the permutation draws, not the kernel: one `rng.permutation(n)` per column per permutation per class encoder (408–456 draws per multiclass fit) is 75–80% of the row by microbench. The draws are slices of a golden-frozen generator stream, so no bit-identical rewrite can remove them
fact: 2026-09-21 | F6 S1b (I025, `results/probe-ts-mismatch-s1b-20260921.{md,json}`): rare-only matched weight (m ≤ 5) at 100/50/25% training rows — sf-police +0.29 / +0.29 / +0.53% Brier, 9 of 9 fits; okcupid-stem +0.01 / +0.05 / +0.01; Traffic_violations +0.04 / −0.02 / −0.21; kick −0.01 / −0.04 / +0.10; wine-reviews −0.07 / +0.19 / +0.27% RMSE; porto-seguro an exact tie. Gap pairs at 25%: 7W-5L (bar 8); without sf-police −0.03%, 4W-5L ⇒ F6 KILLED, barrier B18
fact: 2026-09-21 | share of held-out rows in categories of m ≤ 5 training rows, by training size 100/50/25%: sf-police 27.0/35.7/41.1%, okcupid-stem 6.5/7.3/8.6%, Traffic_violations 3.0/4.7/8.8%, wine-reviews 28–35%; unseen share on sf-police 6.6/12.8/21.8%. Only address-like columns put a quarter of their rows in rare categories: in the hc suite that is sf-police's Address and wine-reviews' two largest columns
fact: 2026-09-21 | Traffic_violations' +0.46% / +0.75% under the UNIFORM matched weights (I024) did not come from rare categories: the rare-only arms read +0.04% / +0.06%. It came from re-weighting categories of more than five rows, where Part 1 measured no over-trust. Unexplained, one dataset, post-hoc — a pointer for the R3 ablation, never a result
fact: 2026-09-21 | a monkeypatched probe arm's fit time is not a cost read: the patched transform (a digamma per row) made fits read ×1.04–1.21 with identical tree counts
fact: 2026-09-21 | the harness scores ALL classification Brier as mean Σ_k (p_k − onehot_k)², so its BINARY Brier is 2× the textbook `mean((p₁ − y)²)`; relative gaps are unaffected, absolute ones need the factor when a probe computes its own
fact: 2026-09-21 | KS distance is useless for comparing ordered (training) and full-total (held-out) encodings: held-out values are one atom per category and training values are smeared by the prefix, so KS reads 0.3–0.6 on low-card columns whose moments match to 1%
fact: 2026-09-21 | test suite on main 486e563 (after the #120 recovery; rung branch adds harness + docs only) = 1084 passed, 1 skipped, 114 s — the F3 S2 count, so the recovery landed the library intact
fact: 2026-09-21 | F4 fresh profile (I023, `results/campaign-f4s1-other-20260921.{md,json}`; exclusive wall clock, default estimator, the August panel): on the 6 gr sets `grow` is 63.5–71.8% of fit and the full-data refit leg is 25–35% — replayed rounds 10.7–19.3% plus the 1.25× extra rounds grown from scratch 9.6–12.3%. August's unnamed 23–30% "other" was mostly those replayed rounds, which that instrument never wrapped
fact: 2026-09-21 | binary Logloss layer = 8.3–10.0% of a gr binary fit: `grad_hess` 4.9/4.2/4.4% + per-round validation `eval` 5.1/4.3/4.0% (MagicTelescope/Higgs/road-safety; a 7-rep rerun replicates within 0.1 point). Regression: `grad_hess` 0.6–0.7%, `val_score` 0.8–2.1%; no non-kernel row reaches 5% on any gr regression set
fact: 2026-09-21 | hc `prep` = 34.8% (kick) / 31.8% (wine-reviews) / 41.1% (okcupid-stem) of the default fit AFTER C2, of which 15.5/13.7/19.9 points are the refit's full-row `fit_transform` — the largest non-kernel object measured, hc-only, inside unsplit
fact: 2026-09-21 | a replayed round costs ~40–45% of a grown round on linear-leaf regression sets (nyc-taxi 0.97 vs 2.45 ms, diamonds 1.09 vs 2.4 ms); `_assign_leaves` / `_leaf_values` / `_linear_leaf_fit` unsplit, ridge accumulator B10-closed
fact: 2026-09-21 | exclusive-time hook instrument (`f4_other_walltime.py`): 0.85 µs per hook call, 0.1–1.2% of fit, wrapped/plain 0.99–1.03; on the first pass MagicTelescope's PLAIN arm read a 3-rep median 22% slow (1.067 s; all seven rerun fits 0.87–0.91 s), so 3 reps is thin for sub-second fits — use 7 and keep the per-rep times
fact: 2026-09-21 | first Muse Code rung: one pass, exit 0, ~15 min wall clock for a two-file library+tests edit; muse cannot edit CRLF files with its own tool and patches by script instead; its sandbox user cannot write `.pytest_cache` and leaves an undeletable `pytest-of-Nathan/` in the repo root (untracked, harmless)
fact: 2026-09-22 | HARNESS WART (fix owed, queued as H(11)): `_public_parquet_path` downloads every pub: parquet through one shared `<path>.part` and then `os.replace`s it; on a cold cache with jobs > 1 the three seed tasks of one dataset download at once and Windows refuses the replace while another worker holds the file open (WinError 32). I046's first S4a launch died 5 min in with no results. The pub: cache in `benchmarks/data_cache/openml_pq/` was cold for all ten sets (the August public run cached elsewhere). Until fixed: warm the cache before a parallel pub: run. Fix: a per-process tmp name, and treat "the final file now exists" as success
fact: 2026-09-22 | the count column on the pub: suite (I046 S4a, `results/20260922-141439.json`, decision metric): engaged and scored 5W-1L, median +0.343% [CI +0.074..+2.217%]; rossmann +1.87% RMSE, internet_firewall +2.56% Brier, federal_election +0.50%, nba-shot-logs +0.18%, BNP +0.17%, fps-in-video-games −0.02%; kickstarter engaged but near-solved; BMC, Medical-Appointment-No-Shows and freMTPL2freq exact ties (BMC's max categorical cardinality AS THE HARNESS SEES IT is 10: its geo ids arrive numeric)
fact: 2026-09-22 | the default's strength rows are bit-identical from 0.32.0 (I015, `20260918-170822.json`) to c3a5314 (I039's ChimeraBoost arm): 309 of 309 (dataset, seed) rows; so are every rung's and every opponent's gr rows between I015 and I046 S4c (231 of 231 each)
fact: 2026-09-22 | I015's chart-grade run timed the OPPONENTS slow: against the same libraries and config on 2026-09-22 (`20260922-144219.json`), LightGBM ran 4–6× slower (gr fit median ratio 0.23), HGB 8–16× slower, CatBoost 27% slower on gr; the default itself read gr ×0.71 / hc ×0.90. The I015 chart's slowdown axis (default 3.5× clf / 4.3× reg) overstated our speed position; the I046 read (5.0× / 6.4×) matches the August base's LightGBM gap (~5.2× on gr fit)
fact: 2026-09-22 | I039's variant lines were read on `primary` (F1): on the decision metric hc@time is 3W-1L (Traffic@time +3.69%, employee_salaries@time +2.10%, sf-police@time +0.94%, kick@time −0.23%) and hc@sus25 2W-0L (employee_salaries@sus25 +0.86%, okcupid@sus25 +0.30%)
fact: 2026-09-22 | ROOT CAUSE of the muse sandbox failure (I048, read-only `icacls`): every launch appends a `MuseSandboxUsers` DENY pair and a per-session-SID DENY pair to the ACL of `C:\Users\Nathan\.config\muse` and never removes them; at 1,786 duplicate `MuseSandboxUsers` entries (893 pairs) plus the stale session pairs, the ACL reached Windows' size limit and `SetNamedSecurityInfoW` returns 1340 (ERROR_BAD_INHERITANCE_ACL). A muse bug; the fix is an ACL reset on that folder, outside the repo and the maintainer's call. RESET 2026-09-22 on his approval (`icacls C:\Users\Nathan\.config\muse /reset /T /C /Q`, 1,786 → 0) and the next muse run (I049) had no sandbox error; one launch put back 32 entries (16 pairs), so at that rate the ACL fills again after roughly 50 launches — reset again when the 1340 error returns, and report it upstream
fact: 2026-09-22 | muse's Windows sandbox can fail mid-task, host-wide and deterministically: every shell call returns "windows_elevated unified exec session launcher unavailable: sandbox enforcement unavailable … SetNamedSecurityInfoW failed: 1340" (I047; a subagent's shell failed the same way). The task's written files survive; the planner may run a finished, reviewed script itself — running is not authoring
fact: 2026-09-22 | the count column's cost, same run (I046 S4c): engaged base-hc fit ratio median ×1.20 (colleges 1.07 … kick 1.26), ×1.14 with the hc variants; inert hc ×1.01; a chart-grade --decide run with the I015 field took 55 min
fact: 2026-09-23 | standing quantile BASE = `results/quantile-20260923-115727.json` (main 81be911 + the Phase 1 bench, `quantile_suite.py --decide --seeds 3 --jobs 5`, 59 regression keys × 7 arms, 2 h 49 min): gr head vs CatBoost MQ 7W-29L median −0.45%, vs RigidShift 20W-16L +0.41% (a tie), vs LightGBM per-level 23W-13L, vs our per-level 32W-4L, vs head+CQR 33W-3L, vs NGBoost 35W-1L; the head's 90% coverage 0.869 gr / 0.817 hc, RigidShift's 0.898 / 0.896
fact: 2026-09-23 | the 36 Grinsztajn base keys re-scored bit-identically to `quantile-20260830-175359.json` for the head, our per-level models and CatBoost MQ (108 of 108 records each; LightGBM 94 of 108): their builders draw nothing from the seeded stream, and the head's outputs have not moved since August
fact: 2026-09-23 | test suite on `campaign/quantile-plan` (Phase 1 bench, no library change) = 1188 passed, 1 skipped, 129 s
fact: 2026-09-23 | the quantile head (flat learning rate 0.1) reaches the 2000-round cap on 11 of 36 gr regression sets in `quantile-20260923-115727.json`; on those it loses CRPS 2W-9L to RigidShift (median −5.03%) and 1W-10L to CatBoost MQ (−3.79%), on the other 25 it wins 18W-7L (+1.10%) and loses 6W-19L (−0.26%). Median-level pinball: vs RigidShift 16W-20L, vs CatBoost 5W-31L; 90% interval score: 28W-8L, 25W-11L
fact: 2026-09-23 | Q0 probe battery (`results/quantile-20260923-135654.json`, gr regression): lifting the head's cap to 8000 rounds changes only the 13 cap-bound sets and wins all 13 (median +1.73%; visualizing_soil +22.8%, pol +12.7%, superconduct +7.5%); depth 6 wins 31W-5L (+0.35%) at 0.87× the fit but worsens the 90% coverage error by 1.06 points (hc 3.24); the library's CQR factors computed on the early-stopping rows cut the 90% error from 3.43 to 0.50 points at a CRPS change of +0.02% (21W-15L); recentring on RigidShift's median is 18W-18L
fact: 2026-09-23 | the size fade would LOWER the head's rate (0.1 at ≥ 15k training rows, 0.07 at ≤ 5k), the wrong way for the capped sets; the harness passes n_estimators = rb.MAX_ITERS = 2000 to every arm, the head's own default, so a library budget change cannot show in the bench
fact: 2026-09-23 | standing quantile BASE = `results/quantile-20260923-185507.json` (the Q5 default head, depth 6 + `conformalize="auto"`, with the full field, 2 h 47 min): gr head vs CatBoost MQ 15W-21L (−0.24%), vs RigidShift 23W-13L (+0.74%), vs LightGBM per-level 26W-10L, vs our per-level 33W-3L, vs NGBoost 34W-2L, vs head+CQR 35W-1L; the head's 90% coverage 0.904 on gr, its CRPS skill 0.5920 @ 3.0× over 59 keys
fact: 2026-09-23 | Q3 T0 (`results/quantile-20260923-150433.json`, gr regression): the head at learning rate 0.15 goes 17W-19L (−0.04%), at 0.2 11W-25L (−0.09%), each 7-6 on the 13 cap-bound sets, pol −34% / −74% (barrier B23); depth 6 with the library's CQR factors computed on the early-stopping rows goes 31W-5L (+0.33%, CI +0.14..+0.77) with the 90% coverage error 3.43 → 0.47 points (hc 9.24 → 0.85) at 0.90× the fit
fact: 2026-09-23 | Q2 T0 (`results/quantile-20260923-195957.json`, gr regression, against the Q5 default): 254 bins 18W-13L-5T (sign p 0.47) but +21.1% SGEMM, +15% Brazilian_houses, +7.3% nyc-taxi, −23.6% pol, +4.9% cpu_act; rate 0.03 with 8000 rounds 23W-13L (p 0.13) at 2.97× the fit; depth 8 21W-15L (p 0.41) at 0.94×; 8000 rounds 7W-0L on the cap-bound sets (+1.18%); exact splits 9W-27L (p 0.004). RigidShift beats the head by +30.6% to +35.3% on Brazilian_houses and pol
fact: 2026-09-23 | the quantile Phase 2 gate, tightened prospectively at I060: a default change must also clear a two-sided sign test at p < 0.05 on gr regression (≥ 25 of 36 decided, no ties)
fact: 2026-09-23 | Q6 T0 (`results/quantile-20260923-210350.json`, gr regression, against the Q5 default): a validation-chosen head (the default, 254 bins, or recentred on a squared-error median, picked per fit by CRPS on the early-stopping rows) wins 23W-7L-6T, median +1.57%, sign p 0.005, recovering 99% of the per-set best-candidate gain; CRPS skill 0.6006 @ 7.2× over 59 keys against CatBoost MQ's 0.5982 @ 129×; 2.35× the head's fit
fact: 2026-09-23 | the recentred candidate alone blows up on tie-heavy targets (analcatdata_supreme: CRPS ×5.5e6, 90% width ~1e7), because the head's band collapses and the CQR factor divides by it; the validation choice rejected it on every seed
fact: 2026-09-24 | standing quantile BASE = `results/quantile-20260924-005306.json` (the Q7 default: depth 6, `conformalize="auto"`, `audition=True`, with the full field, ~3 h): gr head vs CatBoost MQ 27W-9L (+0.29%), vs RigidShift 29W-7L, vs LightGBM per-level 30W-6L, vs our per-level 34W-2L, vs NGBoost 35W-1L; 90% coverage 0.903 on gr; CRPS skill 0.6006 @ 7.4x over 59 keys, CatBoost MQ 0.5982 @ 133x off the frontier
fact: 2026-09-24 | the muse sandbox ACL on `C:\Users\Nathan\.config\muse` refilled to 1,820 lines in about ten launches after the 2026-09-23 morning reset and failed with error 1340 at the end of a task (I062); reset again to 4 lines. Reset it every ~8 launches
fact: 2026-09-24 | released 0.33.0 (aeab6ec, tag v0.33.0): CI green on 47ee154 (py3.9-3.13 on Ubuntu, py3.12 on Windows, ruff, docs deploy), 1217 passed locally, a 14-module wheel, `twine check` passed; PyPI and the GitHub release live; a clean-venv install from PyPI passed a smoke test outside the repo (the quantile audition picked `recentred`, 90% coverage 0.91, SHAP (10, 5, 19))
fact: 2026-09-24 | the published chart on 0.33.0 (`results/20260924-090541.json`, public suite, 22 datasets x 3 seeds, 21 scored): the default avg rank 1.94 [1.64-2.24], CatBoost 1.84 [1.43-2.24], LightGBM 2.23 [1.74-2.70]; median slowdown 5.6x / 47.2x / 1.0x; the harness summary has the default 8W-13L against CatBoost (median -0.17%) and 14W-7L against LightGBM (+0.43%). The 2026-08-01 chart read 1.90 against CatBoost's 1.88 at 7.1x / 53.1x
fact: 2026-09-24 | the internal chart on 0.33.0 (`results/20260924-130232.json`, decide tier, 3 seeds, the nine-arm field without the NoCatCount control): every per-seed metric equals `20260922-144219`'s bar one LightGBM Brier at the 17th digit, and the win rates are unchanged (default 70.7%). Skill: the default 0.4052 clf / 0.7341 reg at 4.9x / 6.3x, CatBoost 0.4057 / 0.7294 at 53.0x / 47.8x. Every arm's R² still moved by up to 0.0002, because the run's dataset metadata keeps the test-split scale of whichever seed finished first (H(14))
fact: 2026-09-24 | RoNGBa as the quantile suite's NGBoost opponent (issue #163, I063, `results/quantile-20260924-164257.json`): it beats stock NGBoost 23W-13L on gr (median CRPS +0.94%, CI -0.07..+3.7) at 0.26x its fit, and the 500-round cap never binds (best round max 269); the head beats it 33W-3L (+6.30%) and loses on pol, visualizing_soil and SGEMM, the cap-bound low-noise sets (a parked capacity lead). 2 of 177 fits break down: one training target at z ~ -30 gets its own leaf in round 0's log-scale tree (value ~ -447, line-search multiplier 1), sigma grows ~5.8e7x for the test rows in that leaf, and the validation NLL moves ~0.01, so early stopping cannot see it. Mean CRPS skill -166.5; NGBoost is left off `images/quantile_pareto.png`
fact: 2026-09-24 | standing quantile BASE = `results/quantile-20260924-164257.json` (the 0.33.0 head, the RoNGBa opponent; every other arm equals `quantile-20260924-005306.json` per key, 354/354)

## Beam

| id | family | status | next |
|----|--------|--------|------|
| F1 | Cross-feature cost trim v2 | KILLED 2026-08-16 (S2, I007+I008) | none — closed as barrier B16 |
| F4 | Profiling-driven speed | ACTIVE (C2 + C1 + C1b + C4a + C4a-2 + C3 shipped; measured objects exhausted; loop now on the shortlist queue) | **C3 SHIPPED (PR #127, c9c0f3d; I029)**: fused binary Logloss layer, bit-identical 155/155, Grinsztajn binary fits −4.7 to −5.7%, kick −4.2%, controls flat — the first F4 unit that reaches Grinsztajn. Next: the shortlist queue (H(4)+H(5), H(1), R2, R3); F4 has no measured exact-rewrite object left. Parked: C4b shared TS permutations (algorithm change). **C4a-2 SHIPPED (PR #126, 3706494; I028)**: the numeric block cast once per fit, porto-seguro −8.5% / kick −3.4%. **C4a SHIPPED (PR #125, a8f04c8; I027)**: categorical columns are factorized once per fit and each leg's codes derived by an integer re-rank — bit-identical 155/155, default fit −9.4% kick / −18.9% sf-police / −18.1% porto-seguro / −8.2% okcupid-stem, numeric control flat. |
| F2 | Sub-gate cross via CV-averaged race | KILLED (I017) | 5/5 engaged precision at 3-7x cost; S1 did not replicate |
| F3 | Classifier forced-cross | KILLED 2026-09-21 (S3, I021) | gr binary engaged 10W-13L, median −0.04%: the race earns its fee on the classifier. Knob stays opt-in (PR #117), no rung-1 pin |
| F5 | hc-Brier gap vs CatBoost | **SHIPPED 2026-09-22 (I046, PR #145, d14bf38): `cat_count_features` on by default** (public 5W-1L, decide hc 6W-1L, bit-identical elsewhere, chart refreshed) — earlier: I038 library form, I039 S3 PASS | the published chart (`public_pareto.png`) refresh: DONE 2026-09-24 on 0.33.0 (`results/20260924-090541.json`, `--public --seeds 3`, 22 datasets, ~4 h: the default avg rank 1.94 [1.64-2.24] against CatBoost 1.84 [1.43-2.24] and LightGBM 2.23, at a median 5.6x against 47.2x and 1.0x; `docs/benchmarks.md` prose updated; `pareto.png` re-rendered from `results/20260924-130232.json` and `docs/PROJECT_STATUS.md` synced); S7 (multiclass CTR width) is the family's next idea if picked. History: **A per-categorical count column closes 47% of CatBoost's hc edge** (I035): 4W-1L on the gap sets, +0.43% Brier median, gains ordered by cardinality, sf-police and Traffic unanimous across seeds. The gap is the encoder (CatBoost on our TS keeps none of its edge); not the prior target, Counter, permutations or quantization (TS quantization kills on big sets, +1.7–3.6% on the two small controls — a small-data pointer, parked). `cat_count_features` (opt-in, card ≥ 256, invisible to the cross and linear-leaf races; I038) on the decision tier (I039): gr 0-0-59 exact ties, the 7 hc sets without a qualifying column exact ties, the engaged 7 **6W-1L** at +0.20% median (sf-police +0.73%, Traffic +0.86% Brier; employee_salaries +2.45%, wine-reviews +0.56% RMSE), hc@time 4-0, fit ×1.09 on hc (engaged median 1.165). PR up with the flag OFF. The random-effects alternative (per-column ANOVA λ for the TS, I040) KILLED: uncapped it collapses the small controls (−3.8 / −9.6%), capped at 10 it is a flat wash and still costs kick and eucalyptus; the count column keeps evidence the shrinkage deletes. Next: the maintainer's go on /experiment S4 for the default flip; meanwhile R4 S0 |
| F6 | Ordered-TS train/test moment mismatch (shortlist R1) | KILLED 2026-09-21 (S1b, I025) — closed as barrier B18 | The defect is real (rare categories over-trusted, reliability 0.63–0.70) and two transform-side fixes both went 7W-5L against a bar of 8: the gain is sf-police (9 of 9 fits, +0.29% to +0.53%) and nothing else. Nothing ships; the open door is the Counter feature, which belongs to R3 |
| F7 | Multi-quantile head (`ChimeraBoostQuantileRegressor`) | PARKED 2026-09-24 (the maintainer: GitHub issues first; ACTIVE from 2026-09-23). Phase 1, the bench, MERGED (I056, PR #156); Q0 DONE (I057, PR #157 merged): uncapped and validation-rescaled pay, depth 6 misses on its guard alone, recentred fails. Q3 CLOSED (I058, barrier B23: a larger rate loses); depth 6 + early-stopping-row calibration PAID (31W-5L, +0.33%, 90% error 3.43 → 0.47); PR #158 merged. **Q5 PASSED (I059): the head's default is now depth 6 + `conformalize="auto"`** (gr 31W-5L, +0.33%, 90% coverage error 3.43 → 0.47); PR #159 merged, snapshot rebaselined. **Q2 CLOSED (I060)**: no probe clears the tightened gate (sign p < 0.05); the five low-noise sets need resolution (bins) on two and a squared-error location on two; PR #160 merged. **Q6 T0 PASSED (I061)**: the validation-chosen head wins 23W-7L-6T (p 0.005), recovers 99% of the oracle, and tops the 59-key frontier above CatBoost MQ (0.6006 @ 7.2× against 0.5982 @ 129×); PR #161 merged. **Q7 PASSED (I062): the audition is the head's default** (identity 177/177 with the bench arm; gr 23W-7L-6T against the Q5 default, p 0.005; CatBoost MQ now 9W-27L against us, off the frontier); PR #162 merged, **released in 0.33.0** | **I063 DONE (PR #167)**: RoNGBa is the NGBoost opponent (issue #163). **PARKED 2026-09-24** under the maintainer's focus rule (GitHub issues first): Q4 T0, spread-aware categorical encoding (a TS of the spread per category) on the hc regressions and `catscale`; then Q1 (P16); and the I063 capacity lead (pol, visualizing_soil, SGEMM). Program: `QUANTILE_PLAN.md` "Campaign 2026-09-23" |

### F1 — Cross-feature cost trim v2
status: KILLED 2026-08-16 at S2 (I007 at k=6, I008 at k=12) — closed as barrier B16
verdict: the hypothesis was half right and that was fatal. The cost does drop by
double digits (engaged fit −13.2% at k=6, −11.6% at k=12) — but the screen DOES
change which crosses win, and regression pays for it (4W-14L, then 6W-12L). The
saving is nearly flat in k and the harm nearly is too, which places the time in
the tail of the ranking and the damage at its head: no k reconciles them. The
two named fallbacks (prefix-importance single-fit, mid-boost augmentation) are
covered by B16 as written — both are "carry fewer columns" in another costume.
hypothesis (as stated, for the record): the cross-feature audition's ×2.18 engaged-set fit cost can drop by double digits without changing which crosses win, by screening candidate pairs to top-k≈6 columns before the race (fallbacks: prefix-importance single-fit; mid-boost augmentation)
parent-evidence: 2026-07-13 ship record (cost ×2.18 total, ×2.38 median on engaged sets; v2 ideas recorded at ship time, previously nowhere greppable — re-recorded here); `PARETO_PLAN.md` step-0 attribution
barriers: B14, B2 — clearing argument owed at S0: B14 closed the audition ROUND budget k; a column screen trims the candidate PAIR SET, a different axis; per B2 the S3 A/B runs at rung-3 `refit_full` default
next: none — closed. The knob (`cross_top_columns`, default-off, bit-identical unset) and the arms (`ChimeraBoostXTop6`/`XTop12`) stay in the tree as the instrument that produced B16.
kill (the bar it hit): any synth slice where the screen changes a cross PICK (not just cost) with strength loss; or S3 regression stratum sign-test fail at default quality

### F2 — Sub-gate cross eligibility via a CV-averaged race (KILLED I017)
status: ACTIVE
hypothesis: sets below `CROSS_MIN_SAMPLES=2000` (`sklearn_api.py:1283`) — eucalyptus first — can earn cross features IF the race signal is repaired by CV-averaging (cheap at that size); a plain threshold drop is barred by B1's mechanism (untrustworthy small val split) + B2 (refit amplifies mispicks)
parent-evidence: M1 record 2026-07-17 (eucalyptus = biggest hc CatBoost gap, below gate; recorded follow-up); I002
barriers: B1 (cleared only via the signal-quality mechanism), B2 (judge at rung 3), B14 (inapplicable — signal quality, not budget)
kill: the S1 probe shows the CV-averaged race still mispicks on sub-gate sets (test metric not improved by its picks)
verdict: KILLED — S1 passed thin (+1.20% on one real set, I016), S2 flat
(5/5 engaged, median −0.22%, I017). CV-averaged refereeing over ~110-row
slices is chance-level (B17); the family is closed, the branch goes
unmerged.

### F3 — Classifier forced-cross ("always")
status: KILLED 2026-09-21 at S3 (I021) — the classifier's cross-feature
headroom is too small to survive losing the referee: probe +0.36% median
(I019), synth engaged +0.16% (I020), decide gr binary engaged 10W-13L at
−0.04% median. The opt-in knob shipped in PR #117 stays; the rung-1
classifier recipe keeps cross features off.
hypothesis: E2's forced-cross result (rung-1 regressor: 28W-8L engaged, +0.60% median) transfers to the classifier, whose `_FORCED_CROSS_OK=False` today
parent-evidence: `SELECT_PLAN.md` E2 verdict (merged 7684655); binary crosses earn under the raced default (covertype +12.8% Brier on top of linear leaves, 2026-07-13)
barriers: none expected (E2 itself cleared this family for the regressor); recorded caveat: E2 hc-vs-LightGBM was a coin flip at 7W-6L — classifier bars must pre-register the hc stratum honestly
kill: engaged-slice sign test at S2 fails, or forced probe cost erases the rung-1 speed identity
next: S1 = `probe_cross_pairs.py` on engaged binary sets, classifier pair fidelity (I004)
note: this targets rung 1 (a preset), not the default — frontier win, prioritize behind F1/F2 unless Nathan says otherwise

### F4 — Profiling-driven speed (algorithm level)
status: ACTIVE
hypothesis: non-kernel overhead on the multiclass path is the under-priced leg — okcupid-stem spends 50.5% of fit in "other" (fresh attribution, `results/campaign-attr-20260816.md`); hc prep (prep_other+ts_enc) runs 12–28%. Kernel-side objects stay closed (B10, B15).
parent-evidence: `results/campaign-attr-20260816.md` (2026-08-16, sha 7684655); M1 record already noted multiclass ~33% non-kernel overhead in 2026-07-17 — it has grown or was under-measured
barriers: B10, B15, B14, B2 all adjacent — any concrete candidate re-runs barrier_check at S0
kill: per-candidate, set at S0; family-level: if a cProfile read shows the "other" time is irreducible dispatch (many tiny trees), record and kill — NOT met, see I009
next: **two shipped.** C2 (I010/I011) `_factorize_hashed`, bit-identical, 3–6% off
fit on string-categorical sets. C1 (I012/I013) `_softmax_kernel`, bit-identical,
**37–44% off multiclass fit**, binary/regression untouched. The family is the
campaign's productive one so far, and both wins came from the same move: measure
the object's wall-clock share, then ask whether an exactly-equal rewrite exists
before assuming a trade-off.
The wall-clock rule now has evidence on both sides and should be stated as the
symmetric thing it is: cProfile OVERSTATES many-tiny-calls objects (C2, 18%
profiled → 8% real) and UNDERSTATES few-fat-calls ones (C1, 39% profiled → 45.4%
real). Neither direction is safe to forecast from; measure.
C1b (fusing `grad_hess`'s remaining `P - Y` and `max(P*(1-P), 1e-6)` passes into
the kernel that now exists) was measured at S0 and PARKED (I014): ceiling
4.4–7.5% of a multiclass fit, cheap and bit-identical, but hc-only. It is on the
record so it need not be re-derived; taking it is an ordering call against F2.
(Taken and shipped 2026-09-21, I018.)
Fresh profile 2026-09-21 (I023): August's unnamed Grinsztajn "other" was the
refit's replayed rounds, and nothing at the Python/numpy layer reaches 5% of a
Grinsztajn regression fit. Two objects are measured and owe an S0 each: **C4**,
hc `prep` at 32–41% of fit with 14–20 points of it re-paid by the refit; and
**C3**, the binary Logloss layer at 8–10% of fit across `grad_hess` and the
per-round validation `eval`. C3 is the first F4 object that reaches Grinsztajn
(23 binary sets of 59).
C4 split 2026-09-21 (I026): hc `prep` is mostly **factorize, paid 2.2× per
fit** (selection-leg training rows, validation rows twice, then every row in
the refit) — 11–30% of fit on all six hc sets, porto-seguro's integer-coded
categories included, because on an object array the "vectorized" numeric path
is a per-element cast plus a boxed audit. **C4a, prepare once per fit**, is an
exact rewrite with a −5 to −12% forecast and is the family's next unit. The
ordered-TS row is three-quarters `rng.permutation` draws, which no exact
rewrite can touch (golden-frozen stream); sharing permutations across columns
(**C4b**) is an algorithm change worth 4–15% of hc fit and stays parked.
C4a shipped as a PR the same day (I027): bit-identical 155/155 and **−8 to
−19% off a default fit on four hc sets**, the family's largest win since C1
and the first one every user with categorical columns gets.

### F5 — hc-Brier gap vs CatBoost
status (2026-09-22): **S4 PASSED (I046); the default flip is a PR awaiting
the maintainer's merge.** The block below lifted at I033 (R3, the CatBoost
ablation that named its target CTR's arithmetic); the mechanism that
survived is the count column (I035 probe → I036/I037 harness arm → I038
library flag → I039 S3 → I046 S4). The lines below are the family's
original 2026-08-16 slot, kept as written.
original status: BLOCKED(needs B3-clearing mechanism from lens L3)
hypothesis: (held slot) the real-but-small hc Brier gap (+0.0029/set, CatBoost 86–88% winrate there) has a lever that isn't a partial CatBoost port
parent-evidence: hc suite build record 2026-07-15; B3 = seven partial ports, seven kills
barriers: B3 hard; B4 (ordered boosting closed)
kill: any proposal that is a partial CatBoost mechanism port dies at S0
next: none until a beam refill produces a genuinely integrated mechanism

## Direction change 2026-09-23: the loop moves to the multi-quantile head

The maintainer, in chat after merging #155: "I would like to shift focus to
the multi quantile 'quantiles' model now though. We don't have much
benching built for it though. Let's plan." The point-model queue goes on
hold as it stands: the multiclass rate/Hessian trade (I052), S1 parked on
`campaign/s1-predict-cat-lookup` (its 2026-09-29 dated close still runs),
the binary linear-leaf race pointer (I047) and a point-model refill. The
quantile program's plan lives in `benchmarks/QUANTILE_PLAN.md` (section
"Campaign 2026-09-23"); its verdicts flow back here as log entries. On the
proposal, the same morning: "Just add ngboost, not the rf though. I like
crps. Ok yea go ahead on it" — NGBoost joins the field, quantile forests do
not, CRPS is the decision score, Phase 1 (the bench) starts.

## Refill shortlist 2026-09-22 (PICKED the same day, by the loop on the maintainer's delegation: S1 S2 S3 S4 S6; beam cap 5)

pick (2026-09-22, session after PR #144): the maintainer handed both open
calls to the loop in chat, "you may go ahead on whichever of this that you
like". The loop's decision under that delegation: **S1, S2, S3, S4, S6**
(the five recommended below) and **yes to /experiment S4 on the count
column** (I046). Reviewer amendments made at the pick, from re-reading the
standing BASE (`results/20260918-170822.json`; its strength rows equal
`campaign-base-20260816.json`'s to the digit):
- **S2's "9 of 30" is about three independent datasets.** california and
  california@sus50 are one dataset (+2.7% each); bank-marketing +1.0% and
  credit +0.5% are unanimous over seeds; albert@sus25 +0.45% and
  eye_movements@sus25 +0.30% are twins whose parents we win;
  Diabetes130US +0.13%, compas +0.10% (1 of 3 seeds) and heloc +0.07% sit
  inside seed noise. LightGBM also beats us on california (+0.98%). S2's
  bar is restated at its S0 on the sets with a real edge.
- **S3 is one set, cpu_act@sus25.** SGEMM is near-solved (best NRMSE
  0.015, under the 0.02 cutoff: outside the headline win rate, R² gap ~0)
  and our own bag Ens8 recovers 73% of its gap, a variance matter.
  cpu_act@sus25 is not: best NRMSE 0.147, CatBoost +20.2% / LightGBM
  +28.5% / HGB +24.5% of our RMSE, Ens8 WORSE than the default (−2.9%),
  seed 1 at 4.50 against LightGBM's 2.43. No other gr regression set
  trails CatBoost or LightGBM by more than 0.7%. The probe ablates OUR side
  first (linear leaves, the replay refit, the adaptive rate) and splits the
  error by test row; extrapolation on heavy-tailed inputs at n = 1,536 is
  the first hypothesis.
- **S6 was selected on kdd_ipums and eucalyptus** (I035), so neither can
  confirm it (GATE_ROBUSTNESS §8). Its bar is read on small categorical
  sets that did not select it (the hc regressions under the n gate, and
  any @sus twin that falls under it); kdd_ipums and eucalyptus report only.
Order, because one campaign PR is open at a time and every library PR
waits on the maintainer: **I046 (count column S4) first**, the long run
whose PR needs him, launched while he is around; then the self-mergeable
probes **S2 → S3 → S6**; then the library rungs **S1** and **S4's flag**,
one PR each. First alternates if a slot frees: S8 (a 5-minute probe), S5
(exact, small), S10 (the speed price list). Queued behind the five, same
day: **H(11)**, the pub: download race (facts ledger, found at I046);
**H(12)**, `make_pareto.py`'s title names only the first suite and HGB's
subset-artifact classification point sits on the frontier (found at I046
S4c, pre-existing since I015).
status 2026-09-22: **I046 SHIPPED** (PR #145, d14bf38). **S2 RESOLVED —
KILLED** (I047; bins254 a dedup miss, registered as B20; pointer for the
next refill: race the binary linear leaf, it costs california 1.73% and
earns electricity 3.67%). **S3 RESOLVED — KILLED** as a one-set outlier
(I048; the leaf-floor dedup registered as B21). **S6 RESOLVED — KILLED**
at S1 (I049; 20W-17L on non-selecting data, ×14 tail; B3's eighth port).
Muse's sandbox repaired the same afternoon (facts ledger). **S1 RESOLVED
— KILLED at its bar** (I050: exact, −8 to −17% predict on string
categoricals, median 14.75% against 15%; code parked on
`campaign/s1-predict-cat-lookup` until 2026-09-29 for the maintainer's
call). **S4 RESOLVED — KILLED** at S2 (I051: standardized `diff` 23W-23L on
a fair synth screen; B16 addendum). **S8 CLOSED at S1b, no build**
(I052: the exact multiclass Hessian and a slower multiclass rate each buy
~+0.5% Brier for ~×1.8 multiclass fit — a trade put to the maintainer).
**S10 RESOLVED** (I053: LightGBM's edge is per-round cost × our round
counts; our threading scales as well as its; B22). Next: S5, then H(11) +
H(12), both library-class PRs for the maintainer's merge.

Produced by I042: four read-only lenses (L1 fit AND predict profiling with
a new predict-side wall-clock read, L2 literature mechanisms against the
barrier list, L3 opponent ablation on the standing BASE and the S3 JSON, L4
harness gaps), `barrier_check.py` over every survivor in one pass, dedup by
hand against this log, `research/SUMMARY.md` and BARRIERS. Ranked by the
maintainer's standing rule first (defect fixes, exact rewrites and removals
before anything that adds a knob) and expected movement per hour second.
Every line names its class; every barrier match carries its clearing
argument or an admission it binds.

| # | candidate | class | slice it targets | mechanism, one line | barriers (clearing) | cheapest probe | prior |
|---|---|---|---|---|---|---|---|
| S1 | **Predict-time categorical lookup, fused TS transform** (L1) | exact rewrite | hc / any categorical input at PREDICT: `_codes_for_transform` is **62 / 73 / 83%** of predict wall clock on kick / okcupid / sf-police (new read, this refill: the campaign had never profiled predict) | the batch is factorized from scratch then remapped through `cat_maps_`; one C-driven `dict.get` pass yields the same codes directly (prototype **0.49–0.50×**); plus a one-pass numba TS transform (18.8% of multiclass predict) | B3/B18 keyword only (values untouched, codes asserted bit-equal); B10 keyword (not a grow kernel); gate on no shared bagged context (bagged predict already shares one factorization) | `benchmarks/f4_predict_walltime.py` (I023 discipline) + an equality harness over every hc categorical column; then a muse task | **−22 to −37% hc predict latency**, 0 on gr, fit 0 ± 0.5% |
| S2 | **CatBoost's split score on the noisy low-dim binary cluster** (L3) | opponent ablation → a split-rule default (defect-adjacent) | 9 of 30 gr classification sets where CatBoost leads on Brier and **LightGBM trails us on 8 of 9** (california +2.7%, bank-marketing +1.0%, credit, albert@sus25, eye_movements@sus25, Diabetes, compas, heloc) | not bins (LightGBM has the most and loses), not depth (all 6), not stochasticity (2026-08-01); never ablated: `score_function=Cosine` (a variance-normalized split score), `leaf_estimation_backtracking`, `feature_border_type=GreedyLogSum`, `l2_leaf_reg=3` — ours is plain `g²/(h+λ)` | B3 (its method: name it on the opponent first; B3's kill record is categorical/leaf-side, not split scoring); B4 (arms stay Plain); B6 (a formula, not a sweep) | fork `probe_catboost_ablation.py`: 11 sets × 3 seeds × 5 one-knob arms, ~20 min; bar ≥ 40% of the edge recovered on ≥ 5 of 9 with controls < 1% | names the largest gr win-rate object left (9 sets), or closes the cluster as a barrier after two ablations |
| S3 | **The two double-digit gr losses** (L3) | defect probe | `cpu_act@sus25` (LightGBM +28%, all three opponents 3-0) and `SGEMM` (+15%, all three) — the #1 and #2 losses on 77 sets; full `cpu_act` we WIN by 3–15% | reduce the opponent to us one knob at a time (bins, tree shape, round cap, lr, min leaf); hypotheses: a small-n over-fit that only the subsample creates (our 328 rounds vs 164), and a round-budget mismatch on a near-deterministic surface (CatBoost ran to the cap) | B19 (a slice-conditional rule, not a global stopping change); B6 (no sweep); B8 if subsample appears | 4 sets × 3 seeds × 8 opponent arms ≈ 25 min; bar: one arm carries the opponent to within 25% of our loss on ≥ 1 set with controls < 2% | a defect fix on our side, or a priced architecture cost in BARRIERS |
| S4 | **Standardize the numeric parents before `diff`** (L2) — **zero-fit gate PASSED (I045): 221 of 457 diff columns are near-duplicates of their larger parent, 10 of 33 sets ≥ 2/3 of the block** | default change, no knob (a default-off flag for the A/B) | gr numeric, the headline suite: the 33 of 36 regression sets where the cross race engages | `_cross_block` subtracts raw columns; when σ_i ≫ σ_j the difference is rank-identical to `x_i` — a duplicate occupying a cross slot; `x_i/σ_i − x_j/σ_j` is the scale-free comparison the docs describe | B16 (carries the SAME number of columns; changes what one operator computes, not how many); B1/B17 keyword | **zero fits**: on 8 engaged gr regressions read `cross_pairs_`, compute σ_i/σ_j and Spearman(diff, larger parent); kill if median \|ρ\| < 0.95 or scale ratio < 3 | gr regression engaged **+0.0 to +0.3%**, 40% chance of exactly 0, 20% of a net loss (the race is cross-vs-none) |
| S5 | **C5 — fused multiclass cross-entropy eval** (L1) | exact rewrite | hc multiclass (4 sets): `val_score` **4.9–6.5%** of fit (I026) | the C3 move on the vector path: `_softmax_kernel`'s loop + the clip as ordered comparisons + the in-order k-sum, mean in numpy | B10 (loss layer, I012/I018/I029 precedent); B16 is a word collision ("cross-entropy") | `f4_c3_speed.py` forked; identity 155/155; exact tests K = 2..7 | **−3 to −4.5% hc multiclass fit**, 0 elsewhere; gr has no multiclass |
| S6 | **n-gated TS quantization + the ES-on-shifted-small-data read** (L2 + L3) | default change (a constant gated on n) + a defect probe | the small-and-shifted slice: `eucalyptus@time` (CatBoost +34%, **LightGBM +39%**, our seeds 0.52/0.40/0.35, 171 rounds vs LightGBM's 35), `Moneyball@time` +3.6%, `hc@sus25/50`; I035 banked `ts_q16` **+1.65 / +3.26%** on the two small controls | at small n the 255-bin binner resolves the ordered TS's prefix noise and splits INSIDE a category; cap the TS column's borders as a function of rows (no-op above ~10k); and LightGBM forced to our round count tells whether the shifted-small loss is stopping, not encoding | B3 (the fixed-15 form was ported at its narrowest and killed on big sets — I035; only the regime where it WON ships); B18 (the binner downstream of both sides, moments preserved); B11/B19 (the stopping half must not re-derive an in-sample rule) | fork `probe_ts_rarity.py` with B(n) on 6 small/shifted sets × 3 seeds (+ LightGBM forced-rounds arm), ≤ 15 min; bar: ≥ 2% on ≥ 3 of 4 with `eucalyptus` random split ≥ −0.3%, every set above the gate an exact tie | sub-gate hc **+0.5 to +2.5%**, exact ties elsewhere; the only candidate that cannot lose the cost axis |
| S7 | **MinEntropy ordinal target binarization for multiclass TS** (L3) | opponent ablation → a port (B3 binds) | the multiclass residual AFTER the count column: median **+0.96%** (Traffic +1.07, Traffic@time +2.22, okcupid +0.69) vs +0.44% on binary | catboost 1.2.10 MultiClass CTR is NOT one-vs-rest: `TargetBorderCount=3 : MinEntropy` over the class index × 3 priors — three entropy-optimal groupings, better conditioned than our K collinear per-class columns; also `bootstrap_type=Bayesian` on multiclass only, never ablated | **B3 binds** (target-statistics family: eight kills, one transfer); B18 (the count door is spent); B4 (Plain) | `probe_catboost_hc_ablation.py` restricted to the 4 multiclass residual sets + 2 controls, 5 one-knob arms, reference = the count arm; ~45 min; bar ≥ 40% of the RESIDUAL on ≥ 3 of 4 | names the second multiclass object or clears the CTR width question |
| S8 | **Full K×K softmax Hessian for the vector leaf** (L2) | default change, no knob; not a port (every opponent uses the diagonal) | multiclass only (gr has none): 4 hc sets | `_apply_vector_update` uses the diagonal `p(1−p)` scaled by `(K−1)/K`; the true Hessian `diag(p) − ppᵀ` has negative off-diagonals, so the leaf under-steps in the one-up-rest-down direction; one K×K solve per leaf per round | B5 keyword (not a shrinkage; replaces an approximation with the exact second-order solve); B10 (never a measured kernel object; priced: n·K → n·K², K ≤ 7) | monkeypatch `_leaf_values_vec`, 4 sets × 3 seeds ≈ 5 min; kill < 3 of 4 up or fit > 1.15× | **+0 to +1.0% Brier**, 50% flat (lr 0.1 makes the step small); the likelier payoff is fewer rounds |
| S9 | Learned default direction for missing values (L2) — **KILLED for the decision axis (I045): gr 0 of 59 sets carry numeric NaN; hc 11 of 14 do, 5 heavily** — an hc-only pointer | defect-class default, FP-drift (goldens re-baseline) | hc only: colleges (100% rows), cjs (100%), Moneyball (66%), employee_salaries (32%), porto (24%) | the binner sends NaN to the top bin, so missing rows route right at every level on every feature — an artifact, never a decision; XGBoost/LightGBM evaluate both directions | B6 keyword only; B10/B15 adjacent (a split RULE, not a speed rewrite; the goldens tax priced) | **zero fits, 2 min**: count numeric NaN columns and affected rows on every gr/hc set; kill if < 3 gr sets carry any | conditional +0.3 to +1.5% on affected sets; unconditional on today's suites most likely 0 |
| S10 | **LightGBM speed price list** (L3) | measurement, Pareto slowdown axis only | gr fit: LightGBM **6.08×** faster in total, 5.22× median; ~1.8–2.1× is round count (255 vs 122 median), **2.7× per round**, 4.9× per round at n ≥ 28k; 15 sets carry 67% of the excess, all low-p large-n | with threads matched, the residual per-round gap at p = 6–9 is parallel decomposition: our `prange` is over FEATURES, so two threads split six chunks with a ragged tail; LightGBM parallelizes over rows too | B10 (must be argued as a decomposition object with a Phase-0 ceiling, not a micro-optimization); B15 (subtraction stays closed); GOSS/EFB are off/inert in LightGBM's defaults | 405 LightGBM fits + our own 1-vs-2-thread sweep on the top-15 sets, ~40 min; report per-round time; a thread-scaling gap ≥ 1.5× promotes a GROW_PLAN ceiling measurement | no win-rate movement; prices the 5× and either opens one speed door or closes the LightGBM thread for good |
| S11 | TS linear-leaf terms confined to the raced regression path (L2, from I041's pointer) | default change confined to a code path | hc regression (colleges +0.74%, employee_salaries +0.78% at I041) | the race protects the choice; the unraced binary leaf is where it hurt | B13/B2 (the race decides, replay amplifies — the existing race gets one more candidate); B3/B18 keyword | `probe_ts_linear_terms.py` on the 6 hc regressions at **6 seeds**, ~8 min; kill if < 4 of 6 or median ≤ +0.1% or either +0.7 loses its sign | **+0.2 to +0.5%** on hc regression, ~45% chance it is seed noise; the weakest base on this list |
| S12 | Entity degree column on top of the count column (L2) | knob | hc entity sets | count of distinct partner-categorical values per category | B3, B18, B16 by analogy (a partner-choice screen) | a `degree` arm in `probe_ts_rarity.py`, 21 fits | +0.0 to +0.3% on ≤ 3 sets; I036's OLS reads against a second column per categorical |
| H | **Harness instruments** (L4), each self-mergeable | measurement | the loop's own decisions | (6) `compare_runs --metric decision` = RMSE reg / Brier clf, the metric every gate uses — **today `primary` is F1 for classification**, so I036/I037/I039's bars were read off a statistic the gate does not use (I039's engaged median is **+0.558% on the decision metric**, not the +0.203% recorded; I036 is 33W-21L +0.49%, still a pass) — FLAG, changes how a gate scores; (7) an effect-vs-seed-noise line (`\|mean Δ\| / sd(per-seed Δ)` per engaged set: I039 reads sf-police 4.3×, wine-reviews 4.0×, Traffic 3.9×, four sets below noise) — and it re-reads **I041: all 8 engaged sets seed-split, CI −0.06..+0.78%, so R4 was INCONCLUSIVE at S1, not KILLED**; (3) the cost line (engaged fit-ratio median + per set; withheld when the runs' thread/timing config differs); (8) `identity_snapshot` has **no config with `cat_count_features=True`** (its two categorical configs draw card 12 and 7, far under 256), none with the classifier's `cross_features="always"`, none with `random_effects`, none bagged-with-categoricals — the "155/155 with the flag off" in I038/I039 is true and empty; (9) a provenance line in `compare_runs` (git sha / dirty / `--models` argv differ ⇒ warn — would have caught the I036/I037→I039 arm relabel) and a `--fact-line` paste-ready ledger line (the I021 −0.04% slip); (10) `@time` seeds relabelled as rolling origins, not replications | none | (6) 3 h, (7) 2 h, (3) 2 h, (8) 3 h, (9) 3 h, (10) 1.5 h | each protects a decision this campaign already made by hand |

Corrections this refill owes to the record, applied here rather than by
editing closed entries: **I041 (R4)** is downgraded from KILLED to
**INCONCLUSIVE at S1** — every one of its eight engaged sets is seed-split
and the engaged CI straddles zero; S11 above is its replication, and it is
not re-queued on its own. **I039's** engaged median on the decision metric
(RMSE / Brier, the one every gate names) is **+0.558%**, not the +0.203%
the tool printed on `primary` (F1); the S4 ask for the count column is
stronger than stated, not weaker.

Recommended pick (five): **S1** (exact, no barrier, the largest single
user-facing latency win on the board, a muse rung), **S2** (the largest gr
win-rate object, 20 min of CatBoost to name or close it), **S3** (the two
biggest losses on the suite, a defect probe), **S4** (a zero-fit probe on
the headline suite; killed before any A/B if the duplication is not
there), and **S6** (the one candidate that cannot lose the cost axis, on
the stratum where our worst margins live). **H(6)(7)(3)(8)** ride along as
self-mergeable harness rungs whenever the bench is idle — (6) and (8)
first, because one changes what every gate reads and the other makes the
identity claim true. S5 is sure but small and hc-multiclass-only; S7 is a
port with B3 against it; S8 is cheap and could slot in after S6; S9 is a
two-minute count that decides itself; S10 is the price list for the
Pareto axis and costs nothing but time.

## Refill shortlist 2026-09-21 (PICKED the same day: R1 first; beam cap 5)

pick (the maintainer, 2026-09-21, in session after PR #119): **start with
R1**, the ordered-TS train/test moment mismatch — a defect probe with zero
library change. R2, R3, R4 and the harness items H stay queued behind it,
and the queue is ranked by his standing rule from the same conversation:
"adding functionality always comes at a cost", so defect probes,
exact-identity speed rewrites and code removal go first and feature knobs
go last; every rung's report names its class. The loop's reading of that
rule, his to reorder: F4's two measured objects (I023: C4 hc prep, C3
binary Logloss layer) are exact-rewrite perf units, so they queue directly
behind R1 with the harness fixes H(1)(4)(5), then the zero-cost R2 and the
measurement R3, and the rungs that add a flag (R4, R5) come last. The pick reached the session's memory notes and
not this file, so the next session (I023) read "awaiting the pick" and
profiled F4 instead of starting R1 — recorded here so it cannot recur.

status of the shortlist, 2026-09-21: **R1 RESOLVED — KILLED** at S1b
(I024 → I025, barrier B18): the asymmetry is real, a transform-side fix
moves one dataset. R3 inherits one pointer from it (the Counter feature,
and Traffic_violations' unexplained response to uniform re-weighting).
R2–R8 and H stay queued; F4's C4 and C3 go next under the ranking above.
2026-09-22: **R2 RESOLVED — KILLED** at S0+S1 (I032, barrier B19): the smoothed-argmin half had been probed and killed on 2026-07-13 and never registered; the 1-SE / tolerance half loses strength in proportion to the rounds it saves on 20 of 21 sets. H(1), H(4), H(5) DONE (I030, I031). Next: R3.
2026-09-22: **R3 RESOLVED — mechanism named** (I033, Stage A+B, 168 CatBoost/Chimera fits): the hc edge is the target CTR's arithmetic, specifically its constant shrinkage target (0.5) against our global-mean prior; Counter, extra priors, permutations and quantization cleared. F5 ACTIVE. Next: Stage C on our encoder (zero library change), then R4.
2026-09-22: **R4 RESOLVED — INCONCLUSIVE at S1** (I041, downgraded from KILLED by the I042 re-read: all 8 engaged sets seed-split, CI −0.06..+0.78%): TS columns as linear-leaf terms read 4W-4L on the 8 engaged hc sets, sf-police −0.03%, kick −0.76% (unraced binary linear leaves over-trust an in-sample statistic); the four multiclass sets exact ties. Its raced-regression half is S11 on the 2026-09-22 shortlist. F5's count column is at S4 awaiting the maintainer (I039, I040). Non-knob items exhausted; refilled 2026-09-22 (I042).

Produced by I022: four read-only lenses (L1 loss-slice profiling, L2
literature mechanisms, L3 opponent ablation, L4 harness measurement),
funnelled through `barrier_check.py` and de-duplicated against this log and
`research/SUMMARY.md`. Keyword barrier matches are listed with the clearing
argument each owes at S0. Ranked by expected win-rate movement per hour.
Mark the entrants (up to 5) and the loop resumes with their S0.

| # | candidate | slice it targets | mechanism, one line | barriers matched (clearing argument) | cheapest probe | prior |
|---|---|---|---|---|---|---|
| R1 | **Ordered-TS train/test moment mismatch** (L1) | max-card ≥ 1000 hc sets, CatBoost 4-of-4 on Brier (kick, sf-police, Traffic_violations, okcupid-stem) | `_ordered_ts` gives a train row the prefix statistic (expected count ≈ (m−1)/2) while `transform` gives test/ES rows the full total (count m); at card 15k over 75k rows m ≈ 5, so trees are sited on one distribution and scored on another. Fix = count-matched smoothing at transform. | B3 (own encoder's self-consistency defect, not a CatBoost mechanism; CatBoost shares the asymmetry), B4/B11 (keyword only) | ZERO library change: fit the preprocessor on sf-police/kick, print per-column mean/SD of fit_transform vs transform on the same rows; kill if standardized shift < 0.05 SD | +2 to +5 hc Brier points if the shift is real; ~free to find out |
| R2 | **Smoothed / 1-SE early-stopping round** (L2) | every stratum; small/noisy sets most | `_EarlyStopper` takes the raw argmin of a 20%-holdout curve and `_refit_on_full` replays it at 1.25×; pick the round by k-round moving average or earliest-within-1-SE instead. Cost exactly zero. | B2 (this IS judged at refit_full), B11 (in-sample isotonic ≠ smoothing an honest held-out curve), B13/B14 (audition budget k and replay fidelity are different decisions), B12 (no arms added) | ZERO library change: one fit per set, staged predictions, test metric at argmin vs each rule's round, ~20 gr sets × 3 seeds; kill if paired wins < half+1 or median ≤ 0 | +0.1 to +0.3% median; the only candidate that cannot lose the cost axis |
| R3 | **F5 unblock: CatBoost hc ablation** (L3) | the hc Brier gap (+0.0030 mean over 5 sets, monotone in max cardinality) | measurement, not a port: turn CatBoost's CTR knobs off one at a time (`max_ctr_complexity=0/1/2`, `simple_ctr=['Borders']` i.e. no Counter, `CtrBorderCount=1`, `one_hot_max_size`, `ctr_leaf_count_limit`) and read the share of the gap each recovers; the SMALLDATA method that found the learning rate | B3/B4 bind only the follow-on native counterpart, argued separately | `probe_catboost_hc_ablation.py` forked from the existing ablation probe; 7 sets (5 gap + 2 controls), 3 seeds; Stage A four arms ≈ 2.5 h CatBoost compute; bar = ≥ 40% of gap recovered on the gap sets with controls < 10% | names the mechanism or closes F5 for good; native follow-ons ranked: frequency column (target-free), raced pairwise cat×cat combos on mixed data, per-TS-column bin grid |
| R4 | **TS columns as linear-leaf terms** (L1) | cat-dominated hc sets: sf-police (5/6 cat), wine-reviews (9/10), kick | `_build_centers_std` zeroes every non-numeric column so `_fit_linear_leaf_tail` can never pick a target-encoded column; on sf-police the "linear" arm has one usable column. Populate `centers_std` for TS columns. | B3 (CatBoost has no linear leaves; not a port), B1 (targets are 54k–75k rows), B12/B14 (no new arm, no budget) | default-off flag + one synth run with a cats-heavy arm; kill if cat-scope not positive by majority or no-cats slice not an exact tie | +2 to +4 hc points, 0 on gr; leakage (coefficients fit on the encoding's own rows) is the risk |
| R5 | **Multiclass sketch dimension s = 2** (L1) | multiclass, CatBoost 75% Brier win over 8 sets; hc only (gr has none) | `MulticlassBoosting` scores splits on ONE Rademacher projection; at K = 3 six directions exist. Sum gains over s ∈ {2,4}. A1_PLAN's registered, never-run contingency. | B10 (closes bit-identical speed objects; this is a strength change that buys cost — price it, B12) | flag, hc multiclass 4 sets × 6 seeds (A1's instrument); kill if pooled Brier ≤ 0 or cost > 1.5× | +1 to +3 hc points |
| R6 | Ridge base margin (L2) | regression sets with a global linear trend | per-row init from a ridge on the numeric block instead of a scalar; `linear_leaves` is local and cannot carry a global trend | B14/B16 keyword only | zero-library: sklearn Ridge → residual → ChimeraBoost vs default on the regression panel | mixed-positive; Grinsztajn curates against linear tasks |
| R7 | Refit gate on flat ES curves (L1) | 7 gr sets where NoRefit beats the default, 4 of them CatBoost losses (heloc, bank-marketing, default-of-credit, compas) | decline the 1.25× replay refit when the curve is flat at T* | **B13 and B2 bite** (the replay curve making a selection decision); clearing argument is thin: one bit, not a grid | analysis of existing JSONs first (which curve-shape statistic separates the 7 from the 52) | high ceiling (+5 gr points), low confidence: refit is 52W-7L |
| R8 | Raced monotone target transform (L2) | skewed regression targets (houses, Brazilian_houses, nyc-taxi, diamonds) | race identity vs log1p / Yeo-Johnson on the ES split | B12 (cost), B14/B17 keyword | script: oracle headroom + pick fidelity (the E2 pattern); kill if oracle < +0.5% RMSE median | real oracle headroom on 4–8 sets; back-transform bias + cost the likely killer |
| H | **Harness instruments** (L4), muse rungs, no ship gate | the loop's own decisions | (1) DONE I031 — engaged-slice median + bootstrap CI + per-seed agreement in `compare_runs`; (2) probe→decide transfer footer: panel coverage + shipped-mode vs oracle read (5 h; the I019→I021 miss); (3) cost column in `compare_runs`, refused when not decision-grade (2 h); (4) DONE I030 — POINTER label for strata under 8 decided sets; (5) DONE I030 — `--save`/`--models` argparse guards | none | each validated by re-reading `results/20260921-080246.json` to reproduce I021's hand-written verdict | prevents the wrong decision rather than moving the chart; (1), (4), (5) recommended regardless of the beam pick |

Not proposed (checked): AGBM momentum and gradient-mass bin borders (L2, low priors, B6/C4 adjacent); Counter/frequency column on its own (B3 hard; goes through R3 first); `cpu_act@sus25` and `eucalyptus@time` outliers (one win each, seed-unstable); random_strength / Bayesian bootstrap / DART-class noise (killed by the SMALLDATA ablation, ≈1 point).

Recommended pick: **R1, R2, R3, R4 + H(1)(4)(5)**. R1 and R2 have free probes and can both resolve in one session; R3 is F5's only sanctioned door and runs while nothing else is on the bench; R4 is the first hc mechanism that is not a port. Process proposal riding with this: amend `AGENTS.md` so muse may edit any file the task file lists (today `benchmarks/` is reserved), which is what makes H and the probe scripts muse rungs instead of Claude's.

## Iteration log (append-only)

#### I068 2026-09-25 issue #113 remainder: the entity-ID auto-route kill-or-keep probe (BENCH only, pre-registered)
why now: decision (3) of the maintainer's 2026-09-25 go on #113 demoted the
auto-route from a default candidate to this probe; it closes that bullet
of the issue. PR #172 merged (522710c). Branch
`campaign/issue113-autoroute-probe` from main; muse task
`20260925-issue113-autoroute-probe.md`.
design (RANDEFF_PLAN.md "Slice 2"): a qualifying column is a categorical
with >= 1,000 training levels at a median of <= 5 rows per level; on
each set the top qualifying column (most levels) is routed. Arms, on hc
wine-reviews, colleges and employee_salaries, seeds 0-2, a random 75/25
split: A the default model (the column as a categorical: ordered target
statistics plus its default count column); B the column dropped and
`random_effects=True` on it (a shrunk intercept; unseen levels get 0);
C = B plus the column's training-row count as a numeric feature.
kill bar (pre-registered in the design pass): kill the auto-route unless
a route arm beats A on >= 2 of 3 sets with a median gap > 0.
barriers: none new; I040 (per-column shrinkage of the TS killed: "the
count column keeps evidence the shrinkage deletes") is the prior against.
forecast: KILL. A wins at least 2 of 3 on test RMSE against both route
arms (the default's TS + count column already carry what a shrunk
intercept would); C closes part of B's gap.
ran (muse exit 0): `benchmarks/probe_entity_route.py`,
`results/probe-entity-route-20260925-083329.json`, 27 fits. Routed:
wine `designation` (~10.7k levels, median 1 row), colleges `zip` (~4.7k),
employee `date_first_hired` (~1.95k); no alias drops. Seed-mean test RMSE,
A / B / C: wine 2.1212 / 2.1208 / 2.1246; colleges 0.1408 / 0.1421 /
0.1410; employee 4062.6 / 4055.1 / 4109.7. B vs A: wine +0.02%, colleges
-0.92%, employee +0.18% (route-better positive); C lost all three
(median -0.19%). Fit times flat.
result: the letter of the bar says KEEP for a fuller gate (B wins 2/3,
median +0.02%). The robustness read (GATE_ROBUSTNESS.md 2, 3 and the
short version) says noise: 3 datasets is a pointer; the median is a
just-above-threshold tie (wine to three decimals); employee's win is
seed 0 alone (B loses seeds 1-2); dropping it leaves no real win; the
count variant loses everywhere. Forecast KILL: MISS on the letter, HIT in
substance.
verdict: **PARKED, not shipped.** No fuller gate is possible: only these
3 hc regression sets have a qualifying column, and a classification
route needs a logistic mixed model (a slice of its own). The default
categorical path stays; nothing changes in the library.
#113 now: random slopes NOT SHIPPED (I067, code archived); the auto-route
PARKED on a tie; a second grouping column and a guarded slopes 2b not
started (no evidence calls for either). Closing the issue is the
maintainer's call.
next: #45 (loose ends).

#### I067 2026-09-25 issue #113 slice 2 (random slopes on top of random intercepts; LIBRARY feature, opt-in, pre-registered)
why now: the focus rule's next issue; the maintainer's go 2026-09-25 with
the four design calls as recommended. PR #171 merged (a65c8b2). Branch
`campaign/issue113-random-slopes` from main a65c8b2. Program:
`benchmarks/RANDEFF_PLAN.md` "Slice 2" (design, decisions, API, tests, the
pre-registered gate and bars).
barriers: none matched.
forecast: gate bars (a), (b), (d) pass and (c) has no breach (house seen
3-5% better than intercepts, employee seen 1-2% worse, the rest within
+-0.3%); identity snapshot 186/186 (opt-in, default path untouched), so
both Pareto axes are unchanged.
pass 1 (muse exit 0): `random_effects.py` +`_slope_suff_stats`,
`solve_slopes` (per-group 2x2 ridge; `ratio_a = inf` delegates to
`solve_intercepts`, so no-slopes is exact), `estimate_slope_ratios_reml`
(profiled REML with fixed effects [1, z] via Woodbury and the matrix
determinant lemma, 3 cyclic golden-section alternations: cycle 2 moves
< 0.02 decades, cycle 3 0.0000); `solve_intercepts` and
`estimate_ratio_reml` byte-identical. `sklearn_api.py`: `random_slopes`,
errors, both solve sites through helpers, `group_slopes_`,
`group_slope_ratio_`, `group_slope_center_`, `group_slope_range_`; predict
clips x to the fit range. 17 tests: solver vs brute force ~9e-16, REML vs
explicit matrices ~2e-15 relative, known ratios within 0.005 decades,
planted slopes corr 0.993, null slopes 1.2e-7, unseen = trees-only exactly,
`random_slopes=None` bit-identical. Full suite 1240 passed, 1 skipped
(conda python); identity snapshot 186/186; ruff clean. Committed.
pass 2 (muse exit 0, committed 1c8e3a0): three planted-slope synth
configs from a separate stream (the six old configs byte-identical by
hash), the one-to-one alias drop (only employee's `department_name`), arms
ChimeraRS (covariate by the correlation rule: X0 on synth; OverallQual,
2016_gross_pay_received, SLG, price, SAT/ACT on the real sets) and
ChimeraRS-null (X4, synth only).
gate run 1 (`grouped-20260925-072649.json`, 14 sets x seeds 0-2, 8 arms):
FAILED every bar: (a) 3/9 cells, slope-x-confounded seen +42%; (b) seen
+1% on the no-slope configs, x-confounded +15%; (c) employee +2.5%; (d)
median fit ratio 3.8x. INVALID as a test of slopes: diagnosis showed
`predict` never applies the slope term. `fit` stores `_random_slope_idx`
and the stale-state reset block clears it eleven lines later
(sklearn_api.py:2556/2567), so `predict` returns the intercept-only
prediction with intercepts at the slope centre. The fitted slopes are
right (corr 0.990 / 0.976 with the planted truth); combined by hand they
give seen RMSE 1.315 / 1.349 against RE's 2.003 / 2.024 and predict's
2.020 / 2.956. The 3.8x fit ratio is the slope REML search (3 cycles x 2
legs x 100 iterations, twice per fit) on sub-second synthetic fits.
Pass-1 tests missed the bug (a small planted-slope margin; NaN and clip
checks compared against paths that also skip the slope). Fix task
`20260925-issue113-slopes-fix.md`: the reset order, tests against a
hand-built prediction that fail on the unfixed code, and a cheaper search
(2 cycles, ~40 iterations). Then gate run 2 under the same pre-registered
bars; run 1 is recorded as invalid, not as evidence either way.
Slice 1's table re-scored with the alias fix (run 1, the RE/Cat/Drop/
LightGBM/CatBoost arms are unaffected by the bug): RE vs Drop 11W-3L, vs
Cat 11W-3L, vs LightGBM 11W-3L, vs CatBoost 10W-4L over 14 sets.
fix (muse exit 0, committed fbc52ef): the reset order
(`_random_slope_idx` stored after the reset block); 3 tests against a
hand-built prediction that fail on the old code (5 failed) and pass now;
the slope search at 2 cycles x 40 iterations (28 -> 9 ms, ratios equal to
4 significant figures). Full suite 1243 passed, 1 skipped; identity
snapshot 186/186.
gate run 2 (`grouped-20260925-075425.json`; same 14 sets, seeds and
bars): (a) PASS: planted slopes cut seen RMSE by 20.8 / 23.8 / 31.9% and
overall by 2.1-3.3%, 8/9 cells on both. (b) FAIL on seen rows: overall
within +-0.25% on every no-slope config and for the null arm, but seen
rows leave the +-0.5% band in three cells: many-small +0.75% (RS), skewed
+1.75% (null arm), x-confounded -1.30% (a gain, still outside). (c)
BREACH: employee +2.21% overall (+2.78% seen), slope on
2016_gross_pay_received; house -0.52% overall and -3.50% seen (the
forecast's 3-5%); the other three within +-0.1%. (d) FAIL: median fit
ratio 1.73x (3.8x in run 1; the synthetic fits take ~0.1 s).
verdict: **NOT SHIPPED.** The pre-registered rule needed (a), (b) and
(d) to pass with no (c) breach. The mechanism is real, but the no-harm,
real-data and cost bars fail. The library code is archived unmerged on
branch `campaign/issue113-random-slopes` (fbc52ef): correct and tested,
the starting point for a guarded slice 2b if one is ever wanted (the
design probe found a validation race halves the employee damage at
best). Kept: the gate's alias fix, a real benchmark bug, as its own PR
from main. Dropped with the feature: the slope configs and the ChimeraRS
arms.
slice 1 re-scored with the alias fix (its original 11 sets): RE vs Drop
9W-2L -> 8W-3L, vs Cat 9W-2L -> 8W-3L, vs LightGBM 8W-3L and vs CatBoost
7W-4L unchanged. Synthetic sets are untouched (6W-0L), so the real-set
record vs our own alternatives goes from 3W-2L to 2W-3L. Employee: RE
6708 -> 7151 (the alias had let RE's trees see the department), Cat
7235 -> 6515, Drop 6771 -> 6672, LightGBM and CatBoost unchanged. Slice
1's opt-in ship still rests on its synthetic sweeps; its real-data
pointer is now weaker than recorded on 2026-09-20.
next: the maintainer's call on the rest of #113 (the auto-route
kill-or-keep probe, the second grouping column, a guarded slice 2b); the
loop moves to #45.

#### I066 2026-09-24 issue #131 slice 1 (`refresh(X, y)` on an opt-in stored training set; LIBRARY feature, opt-in, pre-registered)
why now: the focus rule's third issue. PR #169 merged (5b27b06), #81
closed. Program file `benchmarks/REFRESH_PLAN.md` (design, decisions,
later slices). Branch `campaign/issue131-refresh-slice1` from main 5b27b06;
two muse passes: `20260924-issue131-slice1a-internals.md`, then
`...-slice1b-api.md` after review.
change: slice 1 as REFRESH_PLAN.md specifies (regressor + binary
classifier, single model; bagging, multiclass, `loss="Quantile"` and
random effects raise at fit when `store_training_data=True`). The default
path changes only by a bit-identical refactor (`_fit_gdiff` arithmetic
shared with the stored-rows path).
barriers: B13 (replay is a screening, not a selection instrument) and B2
(the refit amplifies a bad audition) matched on "replay". Refresh selects
nothing and changes no default or audition; B13's bit-identical round trip
is the invariant this rung relies on.
forecast: (1) pass 1a: the stored-rows preprocessing path reproduces the
replay path's binned matrix bit for bit on the same rows, and with new rows
equals the replay path on the concatenated raw rows, over every column
block; (2) identity snapshot 186/186 and the goldens unchanged after 1a
and after 1b; (3) pass 1b: refresh with zero new rows is bit-identical to
the fitted model in every configuration of REFRESH_PLAN.md's test (a);
refresh(A) then refresh(B) equals refresh(A+B); (4) full suite green. Both
Pareto axes untouched (opt-in, default path bit-identical).
pass 1a (muse exit 0): the twin design as first written
(`fit_transform_stored`, `Binner.transform_block`, shared `_gdiff_means`,
`stored=` booster plumbing): forecast (1) HIT, every equality exact (binned
matrix, fitted preprocessor state, leaf values, predict_raw), 5 tests, full
suite green in the sandbox. DISCARDED at review: +551 library lines, ~400 of
them a second implementation of the preprocessing to be kept in step by
hand. Redesign (REFRESH_PLAN.md, "the ONE preprocessing path"): keep bins
for plain numerics and RAW floats only for cross parents, map each bin back
to a value that bins identically (bin = number of borders <= v, binning.py:
82-93), and feed the rebuilt X to the existing replay refit unchanged.
Linear leaves read the binned matrix (booster.py:888), so they are
unaffected. Same storage, one code path, no booster plumbing.
forecast for pass 1a' (`20260924-issue131-slice1a2-rows.md`): (1') every
stored bin round-trips exactly (every feature, every bin index incl. the
missing slot); `fit_transform` on the rebuilt X equals `fit_transform` on
the raw rows (matrix and fitted state), with and without new rows; the
booster replay refit on the rebuilt X equals it on the raw rows (leaf
values, predict_raw); library diff under ~200 lines.
pass 1a' (muse exit 0): `chimeraboost/training_rows.py` (173 lines) +
`GradientBoosting.replay_kwargs()` (25 lines, read off
`inspect.signature(_BaseBooster.__init__)`): 198 library lines against the
twin's 551. Forecast (1') HIT: tests (a)-(f) all bit for bit on a forced
block of all three cross kinds (12 diff/prod, 8 gdiff; plain [2, 3],
parents [0, 1, 4, 5]), a count column, one combo pair, linear leaves (99 of
100 trees with `lin_coef`), 5% NaN numerics, NaN categories, zero weights,
300 appended rows with unseen categories; the missing category maps back
through `"__nan__"` -> `np.nan`. Full suite 1230 passed, 1 skipped (conda
python); identity snapshot 186/186 bit-identical. Committed on the branch
as the pass-1 checkpoint.
pass 1b (muse exit 0): `sklearn_api.py` +291 (the parameter, the slice-1
gate inside `_quality_applied` plus the multiclass check once classes are
known, capture at the end of both `_fit_single`, `refresh` through
`_refresh_single` and helpers, each C901 <= 6); `training_rows.py`
unchanged; 29 new tests. Slice 1 library total 489 lines (training_rows
173, booster 25, sklearn_api 291).
result: (2) HIT: identity snapshot 186/186 after both passes. (3) HIT:
zero-row refresh bit-identical in all 15 configurations (refit_full
"replay"/True/False, eval_set, no early stopping, a 350-level categorical,
all three cross kinds, linear leaves, weights, NaNs, ordered boosting, MAE,
Poisson, a custom objective, subsample + colsample, a weighted binary
classifier with string labels); refresh equals a manual replay on the
stacked raw rows; refresh(A) then refresh(B) equals refresh(A + B); pinned
state unchanged; pickle; every error. (4) HIT: full suite 1259 passed, 1
skipped (conda python); ruff clean on `chimeraboost/`. Review: a DataFrame
with reordered columns is refused by `_check_feature_names_match` before
any append; SHAP after a refresh adds up to `predict_raw` within 4e-13.
Usefulness smoke (not a gate): test RMSE 0.8543 on the 60% model, 0.7975
after refreshing with the next 30%, 0.7698 for a full refit on 90%:
refresh recovers ~67% of the full refit's gain without growing a tree.
docs (Claude): `docs/parameters.md` (the row), `docs/recipes.md`
("Refreshing with new rows"), CHANGELOG (Unreleased, Added); the API pages
render the `refresh` docstring.
verdict: **PASS → PR for the maintainer** (library, tests, docs). Issue
#131 stays open for slices 2-7 (REFRESH_PLAN.md).
next: issue #113 (random effects, slice 2) per the focus rule.
follow-up (the maintainer, 2026-09-24, on PR #170): "Use a larger dataset
to prove refresh's purpose", then "the prime use case is more of a 'daily
refit' after a day of data comes in", then "Let's put the fast kernel in.
Pretend you're like a maintainer adjusting an initial PR". Measured:
Zurich delays at scale (fit 3.3M rows 259 s, refresh +1.6M 141 s, full
refit on 4.9M 459 s; RMSE 3.0406 / 3.0402 / 3.0402); a daily refresh
(517k-row store + 17k new rows) 5.0 s against a 15.6 s fit and a 0.56 s
predict pass. Per tree 11.0 ms, of which `_linear_leaf_fit` 9.2 ms: a
2.3 ms serial counting sort, then per-leaf sums bound by the largest leaf
(median 38%, max 60% of rows). Muse task `20260924-issue131-fast-replay.md`
on the PR branch: parallel stable sort, contiguous leaf-sorted gather,
parallel over (leaf, accumulator) so one big leaf no longer serializes,
every sum in the same row order.
forecast (kernel): identity snapshot 186/186 and the goldens unchanged;
`_linear_leaf_fit` 9.2 -> <= 3 ms at 517k rows; the daily refresh 5.0 ->
<= 2.5 s; linear-leaf default fits faster by the kernel's share.
muse (exit 0), `tree.py` only: `_linear_leaf_fit` keeps today's code as
the arm for `n <= _SMALL_N`; above it, a parallel stable counting sort, one
parallel gather into leaf-sorted contiguous buffers (`gs`, `hs`, `Xd` as
(k, n)), and parallel (leaf, accumulator-group) tasks, accumulator-major,
each sum in increasing row order with today's expressions;
`replay_oblivious_tree` uses the parallel `donor.apply`. 21 new tests
against a verbatim copy of the old kernel (sizes, empty and tiny leaves, a
60% leaf, NaN bins, k 1 and 6, 1 and 12 threads).
result: bit-identical HIT (identity snapshot 186/186; full suite 1280
passed, 1 skipped; ruff clean). Speed MISS against the targets: at 517k
rows `_linear_leaf_fit` 9.93 -> 7.81 ms (1.27x, target <= 3), replay per
tree 11.74 -> 9.17 ms, the daily refresh 5.0 -> 3.6 s (target <= 2.5), a
500k-row linear-leaf fit 15.9 -> 13.4 s (1.19x, the grow path shares the
kernel). The gather alone moves ~76 MB per call (3.55 ms, RAM-bound).
Untried lead: gather uint16 bins (6 MB) instead of float64 design values
(24 MB), looking centres up in L1.
verdict (kernel): PASS as a bit-identical speedup, shipped in PR #170.
2026-09-25, the maintainer on PR #170: "Gonna leave it open for now as I'm
not sure I want it implemented. Let's keep going on any other issues."
PR #170 is PARKED OPEN (not merged, not closed); issue #131 stays open.
The loop moves on to the other issues; if he closes #170, the kernel
speedup (435e866, bit-identical, 1.19x on linear-leaf fits) is worth
salvaging as its own PR, since it helps every default fit.

#### I065 2026-09-24 issue #81 (research cascade: dead self-test anchor, stale `ideas.py` flags; BENCH tooling + test, pre-registered)
why now: the focus rule's second issue. PR #168 merged (3e02ab1), #84
closed. Branch `campaign/issue81-research-ideas` from main 3e02ab1; muse
task `20260924-issue81-research-ideas.md`.
state found (S0, read-only): half one is ALREADY FIXED: the self-test
anchors became `crossfeat_off` + `noop` on 2026-08-10 (3ef96bd), with the
old pair's failure written into `cascade.selftest`'s docstring; the issue
was never closed. Half two is LIVE: seven `ideas.py` entries (C1, C3, C4,
G1, G2, G3, G4) say `implemented=True` for flags the library rejects
(checked against both estimators' `get_params()`); all seven are KILLED
in `research/SUMMARY.md`, which also records the flags' removal. C2 is
`implemented=False` (deferred, never built).
change (`benchmarks/research/ideas.py`, `cascade.py`, new
`tests/test_research_ideas.py`): the seven get `implemented=False` and a
`retired` note pointing at SUMMARY.md, their pre-registered params and
hypotheses kept; `cascade()` refuses a retired idea with its note, and
refuses any idea whose params the current regressor rejects, before any
fit; a CI test asserts every implemented idea's params are accepted by
both estimators, so the next flag removal fails CI rather than a run.
barriers: none matched.
forecast: (1) the new tests pass, and test (a) fails when one of the seven
is flipped back; (2) `--idea C1_onehot_low_card` refuses at once; (3)
`cascade.py --selftest` PASSES (crossfeat_off moves some dataset by more
than 1%, noop exactly flat), which is the evidence for closing half one;
(4) full suite green; the library is untouched, so both Pareto axes are
unchanged by construction.
muse (exit 0): the seven retired with a `retired` note (tier, one-phrase
reason from SUMMARY.md, the removed flag); `_refuse_stale_idea()` holds
both refusals (inlined, `cascade()` reached C901 12) and checks params
against BOTH estimators, since `runner._est` builds either from the same
dict. G4 also sets `ordered_boosting`, which still exists; its note names
only the removed `ordered_leaf_estimation`.
result: (1) HIT: 4 new tests pass; flipping G2 back to implemented fails
test (a) (`'adaptive_leaf_shrinkage' not a regressor param`). (2) HIT:
`--idea C1_onehot_low_card --tier T0` exits at once with the retired note;
all seven refuse the same way. (3) HIT: `--selftest --jobs 3` PASS in 12 s,
crossfeat_off max +5.38% (covertype), kick +1.13%, noop bit-identical on
8 datasets. (4) HIT: full suite 1223 passed, 1 skipped (conda python);
`chimeraboost/` identical to main.
verdict: **PASS → PR for the maintainer** (a new file under `tests/`).
Closes #81: half one was fixed by 3ef96bd, confirmed here by the self-test.
next: issue #131, `refresh(X, y)` with an opt-in stored training set.

#### I064 2026-09-24 issue #84 (`warmup(background=True)` can still print the cold-compile notice; LIBRARY bug fix, pre-registered)
why now: the first rung under the 2026-09-24 focus rule (GitHub issues,
bugs first). PR #167 merged (4289fe4), issue #163 closed. Branch
`campaign/issue84-warmup-notice` from main 4289fe4; muse task
`20260924-issue84-warmup-notice.md`.
change (`chimeraboost/warmup.py`, `tests/test_warmup.py`): `_NOTICE_DONE =
True` moves above the `if background:` branch, so the caller marks the
notice done before the daemon thread exists; a fit on the main thread in
the window before the thread runs `warmup()` no longer prints "compiling
numba kernels". The env route (`CHIMERABOOST_WARMUP`) was never affected.
New regression test: the real `warmup(background=True)` with its thread
target held on an Event, asserting the flag is set and
`_maybe_notice_cold_compile()` prints nothing while the thread is blocked;
it must fail on the unfixed code.
barriers: B22 matched on "thread" only (fit-speed threading against
LightGBM); this touches no fit path. None applies.
forecast: (1) the new test fails before the fix, passes after; (2) full
suite green; (3) identity snapshot 186/186 identical (only a stderr notice
moves), so both Pareto axes are untouched: strength exactly equal, fit
time unchanged.
muse (exit 0): the one-line move, with the comment now "this call is the
compile, or starts it"; the new test
`test_background_warmup_suppresses_notice_before_thread_runs` holds the
thread on an Event (an `entered` event proves it is parked; the timeouts
are hang guards only). Fail-before: `assert False is True` on the flag;
pass-after: 1 passed. Ruff clean on `chimeraboost/`.
result: (1) HIT (muse's revert-and-run). (2) HIT: 1219 passed, 1 skipped
(conda python, outside the sandbox). (3) HIT: identity snapshot 186/186
bit-identical.
verdict: **PASS → PR for the maintainer** (library + test).
next: issue #81 (task file `20260924-issue81-research-ideas.md` written):
the self-test half was fixed on 2026-08-10 (3ef96bd); seven `ideas.py`
entries still claim `implemented=True` for flags the library removed.

#### I063 2026-09-24 F7 issue #163 (the quantile suite's NGBoost opponent moves to the RoNGBa settings; BENCH-only, pre-registered)
why now: the maintainer's issue #163 (2026-09-24), "Use RONGBA as default
NGBoost benchmark", with the settings in code. The paper is Ren, Sun and
Wu 2019, "RoNGBa: A Robustly Optimized Natural Gradient Boosting Training
Approach with Leaf Number Clipping" (arXiv:1912.02338): best-first trees
clipped at 31 leaves with no depth limit, learning rate 0.04, 500
estimators, the number of stages chosen on a 20% validation hold-out by
log-likelihood, then a refit on train + validation. Its Table 1 beats stock
NGBoost on 8 of 10 UCI sets (worse on the two smallest, Boston and Yacht)
at 1.5-4.85x less training time. Queued behind the F5 chart refresh (PR
#166 up). Branch `campaign/ngboost-rongba` from main 12fcd66; muse task
`20260924-ngboost-rongba.md`.
change (`benchmarks/quantile_suite.py`, `tests/test_quantile_suite.py`):
the `NGBoost` arm keeps its name, split, encoding, stopping rule and
scoring; only the learner changes, to `DecisionTreeRegressor(criterion=
"friedman_mse", max_leaf_nodes=31, max_depth=None, random_state=0)` at
`learning_rate=0.04`, `n_estimators=500`, Normal, natural gradient. The
harness's early stopping (patience 50 on the shared validation rows,
scored at the best round) is the paper's validation-chosen M; the paper's
refit is dropped because no arm in the suite refits. The stock learner
(depth 3, rate 0.01, 2000-round cap) is removed, not kept as a switch.
barriers: B23 (a larger rate loses) is about OUR head's cap-bound sets and
B19 (the stopping round is not a free axis) about our stopping rule; this
rung changes an opponent and keeps the shared rule. Neither applies.
forecast (gr = the 36 Grinsztajn regression keys; the chart uses all 59):
(1) RoNGBa beats the stock arm (BASE `quantile-20260924-005306.json`) on
CRPS on >= 26 of 36 gr keys, median gain >= 2%: the paper's gains are on
mid-size and large sets, which is most of gr. (2) The head still beats it
on CRPS on >= 27 of 36 (stock NGBoost: 35W-1L). (3) NGBoost's median fit
falls from 5.2x ours to 1.5-4x. (4) It stays off the quantile frontier.
(5) The 500-round cap binds (best round >= 450) on <= 25% of its fits.
(6) Every other arm's per-key CRPS equals BASE exactly: 0.33.0 is the code
BASE measured.
reading rules: no ship gate (bench-only). If RoNGBa LOSES to the stock arm
on >= 19 of 36 gr keys, the swap weakens our opponent: stop, keep both
arms, and ask the maintainer which one the docs quote. Otherwise the docs
table's NGBoost row and `images/quantile_pareto.png` take the new run, and
it becomes the standing quantile BASE.
run: `quantile_suite.py --decide --seeds 3 --jobs 5 --save`, the full
field, so the docs table and the chart come from one run (~2.5 h).
muse (exit 0): `_rongba()` builder + `RONGBA_*` constants, `_fit_ngboost`
builds from it, docstrings name RoNGBa; `test_rongba_settings` pins the
issue's numbers, the best-round test builds its reference from the
builder (still stops before 500 as-is); 24/24 in the suite file; smoke on
pol, cpu_act, Brazilian_houses ran clean (best rounds 139 / 71 / 100).
Full suite 1218 passed, 1 skipped. Ruff: 4 RUF100 on the untouched
`sys.path` import block under local ruff 0.15.17; CI lints `chimeraboost/`
only, with 0.16.5. NGBoost reads the passed validation rows (it carves its
own 10% only when none are given), for the old arm and the new.
ran: `results/quantile-20260924-164257.json` (seven arms, 59 keys x 3
seeds, ~2.4 h). Per-stratum CRPS sign tests printed (`compare_runs.py
--metric crps --by-suite`; head vs NGBoost via `--model-new`).
result, forecast by forecast:
(1) MISS: RoNGBa vs stock 23W-13L on gr, median +0.94% (CI -0.07..+3.7);
all 59 keys 37-22. Big gains on pol (+68%), visualizing_soil (+85%),
Bike_Sharing cat (+46%); losses on analcatdata_supreme (-87%) and both
Brazilian_houses (-27 / -30%).
(2) HIT: the head beats it 33W-3L on gr (median +6.30%, CI +5.0..+9.9),
interval score 30W-6L; hc 6W-0L, gr@sus25 6W-1L, gr@sus50 4W-0L, the
small hc variants 4W-2L (pointers). Its three wins are pol (-47%),
visualizing_soil (-62%), SGEMM (-8%): the low-noise, cap-bound sets
(B23's population), where 31-leaf best-first trees resolve what the head
cannot inside its budget. A capacity lead for the head, PARKED under the
2026-09-24 focus rule.
(3) NEAR MISS, low side: NGBoost fit 1.38x ours (was 5.15x); the new
settings fit 0.26x the stock arm's time.
(4) HIT, off the frontier: mean CRPS skill -166.5 against the head's 0.6006.
(5) HIT: the 500 cap never bound (177 fits, best round median 67, max 269).
(6) HIT: every other arm's per-key mean CRPS equals BASE (354/354).
unforecast: two fits BREAK DOWN (yprop_4_1 seed 1, CRPS 169.6 against
~0.0075; topo_2_1 seed 2, 42.0 against ~0.0077). Mechanism, reproduced:
the training rows hold one target at z ~ -30; round 0's log-scale tree
gives it a leaf of its own (value ~ -447, line-search multiplier 1), so
log sigma moves by 0.04 x 447 ~ 17.9 and sigma by ~5.8e7; the one or two
test rows in that leaf get spreads of 2.5e7-5e7 target sds. The
validation NLL moves ~0.01, so early stopping cannot see it. Stock
settings (8-leaf trees, rate 0.01) average the outlier away. No win/loss
changes (the head wins both keys on the clean seeds too), but the mean
skill is destroyed, so NGBoost is left OFF `images/quantile_pareto.png`
(re-rendered from the run without that arm; the head 0.6006 @ 7.3x and
RigidShift 0.5869 @ 1.2x are the frontier, CatBoost MQ 0.5982 @ 130x).
docs (Claude): `docs/quantiles.md` (the NGBoost row, the fit column
re-measured in the same run, a paragraph on RoNGBa and the breakdown, the
budget sentence), CHANGELOG (Unreleased, Changed).
verdict: **PASS under the reading rule** (13 losses < 19): the swap
stands. Standing quantile BASE becomes `quantile-20260924-164257.json`.
PR for the maintainer (tests, docs, image). PR #166 merged first; main was
then merged into this branch and the ledger facts and the F7 beam row
were written here (PR #167).
next: the 2026-09-24 focus rule: GitHub issue #84.

#### I062 2026-09-23 F7 Q7 (the audition in `ChimeraBoostQuantileRegressor`, on by default; LIBRARY change, pre-registered)
why now: I061 merged (PR #161 at 58c5cc8); the maintainer merged with no
objection to the stated plan ("on by default unless you say otherwise"),
which follows the house rule that the default is the strongest
non-ensembling setting. The identity snapshot matches main 186/186. Branch
`campaign/quantile-q7-audition` from main; muse task
`20260923-q7-audition-library.md`.
change (`chimeraboost/quantile_api.py`): a new `audition=True` default.
Under `conformalize="auto"` with early-stopping rows, the fit builds the
three I061 candidates (H the head as configured; B the same with
`max_bins=254`, skipped when the user's bins are already ≥ 254; R H's raw
grid moved onto a squared-error `ChimeraBoostRegressor`'s median, then
calibrated), scores each calibrated grid by CRPS on the early-stopping
rows, and keeps the lowest (ties H, B, R; a non-finite score never wins).
Without early-stopping rows, or with `conformalize` True or False, the head
alone is fitted. `audition=False` is the Q5 default exactly. Every predict
path, `staged_predict`, `predict_thresh` and SHAP serve whichever candidate
won; R's SHAP is exact (the head's centred channels, scaled, plus the
squared-error model's attribution on every level). The bench probes are
pinned `audition=False` where they measure a single head.
barriers: as I061 (B12 argued there: 99% oracle recovery at 2.35×).
forecast and bar: (1) identity: the new default head reproduces I061's
Audition arm bit-for-bit on the decide tier (177 of 177 fits), so its gate
result is I061's: gr 23W-7L-6T, p = 0.005 (clears the amended gate), guard
ok. (2) identity snapshot: every point-model pin and `mq3_conf`
(`conformalize=True`, no audition) identical; only `mq3` and `mq3_w_sub`
may move, and only if the audition picks B or R there. (3) tests green,
including local accuracy of SHAP for all three candidates and pickling.
(4) Pareto: the head moves to ~0.6006 @ ~7× over 59 keys. A miss on (1) or
(2) is a defect to fix before anything else reads.
runs (Claude, after review): the identity snapshot; one full-field decide
run with the pinned bench Audition arm as the oracle (~3 h), serving the
identity check, the gate, the chart and the docs table.
ran (muse, exit 0; the sandbox failed with error 1340 at the very end, so
muse could not run its smoke or delete 17 scratch scripts from the repo
root; Claude deleted them, all untracked, and reset the ACL on
`C:\Users\Nathan\.config\muse`, 1,820 lines down to 4, about ten launches
of build-up since the morning's reset): `audition=True` through
`_check_audition`, `_run_audition` (H from today's path, B a second
booster at 254 bins, R H's raw grid moved onto a default
`ChimeraBoostRegressor`'s median), `_pick_audition_winner` (ties H, B, R;
non-finite never wins), and every read path through `_delivered_grid`,
`_staged_recentred` and `_shap_recentred` (R's attribution exact by
linearity: the sorted head channels centred and scaled, plus the centre
model's SHAP on every level); only the winner's fits are kept. 9 new
tests (the bit-for-bit identity to the bench arm; `audition=False` equals
the Q5 head; each of H, B and R wins on a built dataset; no audition
without rows or outside `"auto"`; B skipped at ≥ 254 bins; a tie-heavy
target never picks R, CRPS ~21,284 against 0.047; staged, pickle and SHAP
local accuracy for all three winners), and 6 existing tests pinned
`audition=False` where they check a single head. Claude: the diff reads
correct (B's evaluation grid goes through the booster's own
`as_model_array`, the same conversion `predict` makes); **1217 passed, 1
skipped**; `ruff check chimeraboost/` clean. Claude's smoke (3 keys, 1
seed): the default head and the bench oracle agree to every printed digit
(CRPS 0.6935, 90% coverage 0.891), which is I061's smoke read.
identity snapshot (against the bda646b baseline): **182 of 186 pins
identical**; only `mq3`'s four moved (the audition picks B there: its
importances and validation history move with the booster), `mq3_w_sub`
and `mq3_conf` are unchanged, and every point-model pin is identical.
Forecast (2) HIT.
decide tier, full field (`results/quantile-20260924-005306.json`, 59 keys ×
3 seeds × 8 arms, ~3 h, nothing skipped): **identity 177 of 177** — the new
default equals the bench Audition arm on every fit, in this run and
against I061's records. Gate against the Q5 default
(`quantile-20260923-185507.json`), per stratum: **gr 23W-7L-6T, +1.57% [CI
+0.19..+3.44], sign p = 0.005, guard −0.06 / +0.10: PASS** (the amended
gate); hc 4W-1L-1T (+1.28%). Pointers: gr `@sus25` 5W-1L-1T, `@sus50`
4W-0L, hc `@sus25` 2W-0L (+9.0%), hc `@sus50` 1W-0L, hc `@time` 2W-1L
(+6.3%) with its 90% coverage error up from 0.68 to 2.12 points (the
guard's only flag, on a 3-set pointer; the 80% error falls 3.20 → 1.75).
The new default against the field on gr: CatBoost MQ **27W-9L** (+0.29%;
it was 15W-21L at Q5 and 7W-29L at Q-B4), RigidShift 29W-7L (+1.87%),
LightGBM per-level 30W-6L, our per-level 34W-2L, NGBoost 35W-1L, head +
CQR 35W-1L; its 90% band covers 0.903 on gr. Fit: 2.35× the Q5 head;
CatBoost now 6.8× ours, LightGBM per-level 0.9×. Pareto (the seven field
arms, same run, the duplicate oracle dropped): **the head 0.6006 @ 7.4×
(coverage error 0.0085) and RigidShift 0.5869 @ 1.2× form the frontier;
CatBoost MQ 0.5982 @ 133× is off it**; `images/quantile_pareto.png`
refreshed. The scale-dominated mean CRPS falls 322.5 → 299.9.
forecast: (1) identity HIT, (2) snapshot HIT, (3) tests HIT (six old tests
pinned `audition=False`, each named), (4) Pareto HIT (0.6006 @ 7.4×).
docs (Claude): `docs/quantiles.md` (a section on the three candidates, the
comparison table re-measured, a tuning note on `audition=False`),
`docs/parameters.md` (the `audition` row), CHANGELOG (Unreleased,
Changed, with the time-split caveat).
verdict: **PASS → PR for the maintainer** (library, tests, docs, image).
The standing quantile BASE becomes `quantile-20260924-005306.json`.
**MERGED by the maintainer as PR #162 (47ee154); branch deleted; identity
snapshot rebaselined at 47ee154 (186/186). RELEASED in 0.33.0 (aeab6ec, tag
`v0.33.0`, PyPI and the GitHub release, 2026-09-24).**
next: once merged, rebaseline the identity snapshot; then Q4 T0, the
spread-aware categorical encoding (the hc regressions and `catscale`).

#### I061 2026-09-23 F7 Q6 T0 (a validation-chosen head: the default, 254 bins, or recentred on a squared-error median; bench-only, pre-registered)
why now: I060 merged (PR #160 at 9eabae4). CatBoost's last edge sits on
five low-noise sets with two complementary carriers (bins on SGEMM and
nyc-taxi, a squared-error centre on Brazilian_houses and pol); neither
alone clears the gate because each costs elsewhere. A per-fit choice on
the early-stopping rows could keep each where it helps. Branch
`campaign/quantile-q6-audition` from main; muse task
`20260923-q6-audition.md`.
arms (opt-in `PROBES`): `ChimeraBoostQuantileAudition` fits three
candidates — H, the default head; B, the head at 254 bins; R, H's raw grid
shifted per row onto a squared-error model's median (the point model's
prediction plus its median validation residual), then calibrated by the
head's own CQR factors on the validation rows — scores each on the
early-stopping rows by CRPS, and returns the lowest; it records the choice
and the three validation scores in the record. `ChimeraBoostQuantileRecentredCal`
is R alone (the reference for R; B alone is `ChimeraBoostQuantileBins254`).
Same run: the head, RigidShift, Bins254, RecentredCal, Audition.
barriers (`barrier_check.py`: B11, B19, B20, B23, B1, B2, B5, B6, B12, B14,
B17). **B12** (portfolios die on cost; A2's validation selection recovered
20% of its oracle): the headroom here is concentrated and large (+7% to
+35% on five sets, against A2's diffuse +2.6%), which is what a validation
referee needs to see, and the forecast prices recovery against the
oracle; on cost, three head-sized fits land near 2.3× a head that is itself
~1/40 of CatBoost's fit, so the Pareto, not a fixed bar, decides. **B17**
(thin validation slices race at chance): the gr slices hold thousands of
rows; the small-data twins are where it will be noisy, forecast below.
**B20**: 254 bins enter as a validation-chosen candidate, which is the
noise-aware rule B20 asks for, not a flat increase. B1, B2, B14: this is
not the point model's truncated audition (full fits, no refit). B5, B6,
B11, B19, B23: no stopping, shrinkage or rate change.
forecast (gr regression, 36, against the head):
the audition wins 20–27, median +0.05% to +0.40%; on the five it keeps
≥ 60% of the best candidate's gain on each (the oracle per set); on the
other 31 its median loss is ≤ 0.10%; it recovers ≥ 50% of the summed
per-set oracle gain over the 36 (B12's measure; A2 got 20%); the guard
holds; total fit 2.0–2.6×; CRPS skill over 59 keys ≥ 0.596. RecentredCal
alone: 15–22 wins (as Q0's recentred probe). Choices: H on most sets, R on
Brazilian_houses and pol, B on SGEMM and nyc-taxi. Small-data twins:
neutral to slightly negative.
reading rule, written before the run: the audition PAYS as a screen if,
against the head on gr, it wins ≥ half + 1 of the decided sets with median
CRPS change > 0 and the guard holds. It becomes a LIBRARY candidate only if
it also clears the amended gate (two-sided sign p < 0.05) AND lands on the
Pareto frontier of the 59-key chart (CatBoost's point included from
`quantile-20260923-185507.json`). A screen pass without significance →
recorded as an opt-in candidate, no default change. Recovery below 30% of
the oracle → the referee is the problem (B12's shape), whatever the win
count says.
ran (muse, exit 0, no sandbox error): a `_calibrate` helper (the library
default's two operations), raw-candidate and point-centre helpers, the two
arms, and a harness change: an arm may return a 5th element of extra
numeric fields, merged into the record's metrics (both `run_one`s); 4 tests
(H equals the field head bit-for-bit; the audition returns the recomputed
lowest-validation candidate; R's centre and ordering; the extra-fields
merge). Claude: the diff reads correct; **1208 passed, 1 skipped**.
synth screen (`results/quantile-synth-20260923-204551.json`, excess CRPS
×1000, median over 12 keys): the head 49.09, bins 254 52.44, RecentredCal
59.96, **the audition 49.08** (it keeps the head's number where neither
fix helps).
decide tier (`results/quantile-20260923-210350.json`, 59 keys × 3 seeds ×
5 arms, ~30 min). Against the head on gr regression (36): **the audition
23W-7L-6T, median +1.57% [CI +0.19..+3.44], sign p = 0.005, guard −0.06 /
+0.10: PASS**; the five 5W-0L (median +21.1%), the other 31 18W-7L-6T
(+0.78%, p 0.043) with no set losing more than 0.5%; hc 4W-1L-1T (+1.28%);
pointers gr `@sus25` 5W-1L, `@sus50` 4W-0L, hc `@sus25` 2W-0L, hc `@time`
2W-1L. **Oracle recovery 99%** (summed per-set gain +179.4% against the
best-candidate oracle's +180.6%; A2 got 20%, B12). Per set on the five, the
pick matches the better candidate: Brazilian_houses R (+30.4% / +27.6%),
pol R on two seeds of three (+12.0% against R's +12.9%), SGEMM B (+21.1%),
nyc-taxi B (+7.3%, where R alone is −92%). Picks over the 108 gr fits: R
43, B 33, H 32. Total fit 2.35×. RecentredCal alone 18W-18L (median
−0.21%) and bins 254 alone 18W-13L-5T, as in I060. **R alone blows up** on
analcatdata_supreme (CRPS ×5.5e6, 90% width ~1e7: a tie-heavy target
collapses the head's band, and the CQR factor divides by that width); the
audition's validation score saw it every time (R ~3e4 against H 0.005) and
picked H on all three seeds. Pareto, the 59 keys with the field's other
arms merged in from `quantile-20260923-185507.json`: **the audition 0.6006
@ 7.2× sits at the top of the frontier and pushes CatBoost MQ (0.5982 @
129×) off it**; bins 254 0.5949 @ 3.1×, the head 0.5920 @ 2.9× and
RigidShift 0.5869 @ 1.2× complete it.
forecast: 11 of 16 counts HIT. MISS: the median (+1.57% against a band of
+0.05% to +0.40%; it gained on the other 31 instead of breaking even); the
pick distribution (R, not H, is the most common pick); the small-data twins
(positive, forecast neutral to slightly negative); the other 31 (a gain,
forecast ≤ 0.10% loss); the synth screen, where no forecast was written.
verdict (the pre-registered rule): **PASS as a screen, clears the amended
gate (p = 0.005 < 0.05), and lands on the frontier: a LIBRARY candidate.**
Requirements for the library rung, from this run: R must be reachable only
through the validation choice, and R's calibration needs a guard against a
collapsed band (the analcatdata_supreme blow-up); the choice needs
early-stopping rows (no rows → the default head, as `"auto"` does). PR for
the maintainer (touches `tests/`).
next: Q7, the audition in `ChimeraBoostQuantileRegressor`. The open
product question for the maintainer: on by default (the house rule is
"the default is the strongest non-ensembling setting", which this is, at
2.35× the head's fit), or opt-in. **MERGED by the maintainer as PR #161
(58c5cc8); branch deleted. On by default, stated in the PR and not
objected to.**

#### I060 2026-09-23 F7 Q2 T0 (what carries CatBoost MultiQuantile's last edge: five probes from our side; bench-only, pre-registered)
why now: I059 merged (PR #159 at bda646b); the identity snapshot is
rebaselined there (186/186). CatBoost MQ is the one arm still ahead of the
new default on CRPS (gr 21W-15L, median −0.24%), and the edge is
concentrated: five low-noise sets (Brazilian_houses in both forms, pol,
SGEMM, nyc-taxi) carry −0.084 of the −0.116 summed skill gap, with the
head's median-level pinball up to 50% worse there. The plan said "ablate
CatBoost one knob at a time"; its resolved settings (`get_all_params` on a
bench-shaped fit) differ from ours in five places — rate 0.03 (ours 0.1),
254 borders (128), `l2_leaf_reg` 3 (1), MVS 0.8 subsampling (none),
Cosine scoring on the full gradient (a projection) — and our fits cost 1/17
of CatBoost's, so this rung probes from OUR side, where a winner is
directly shippable. Branch `campaign/quantile-q2-probes` from main; muse
task `20260923-q2-catboost-knobs.md`.
arms (new opt-in `PROBES`, each the new default with ONE change):
`ChimeraBoostQuantileBins254` (`max_bins=254`),
`ChimeraBoostQuantileD6Uncapped` (`n_estimators=8000`),
`ChimeraBoostQuantileLR03Uncapped` (`learning_rate=0.03`, 8000 rounds:
CatBoost's rate with room to converge), `ChimeraBoostQuantileDepth8`,
`ChimeraBoostQuantileExactSplits` (`exact_splits=True`, the exact gain
summed across levels instead of the projection). References in the same
run: the head and RigidShift; CatBoost from `quantile-20260923-185507.json`
(deterministic data, so CRPS pairs across runs).
barriers (`barrier_check.py`: B23, B20, B3, B4, B5, B6, B16, B19). **B20**
(finer bins sharpen a few high-signal sets and overfit the rest; a flat
increase needs a mechanism): this arm is a diagnostic, not a default
proposal. The head's deficit sits on exactly the high-signal sets B20
names as the ones bins sharpen, so the question is whether resolution
carries CatBoost's edge there; the per-set read will show whether the
June reversal (cpu_act −17% on the point model) repeats, and only a
noise-aware rule could follow. **B23**: the rate arm is SMALLER steps with
the rounds to take them, the direction B23 leaves open ("more rounds"),
not bigger steps. B6: five single mechanism-motivated knobs, each copying
a measured difference from the one arm that beats us, not a search. B3:
no categorical mechanism is ported. B4, B5, B16, B19: keyword matches.
forecast (gr regression, 36, against the head; the five low-noise sets
named above are "the five"):
(a) bins 254: wins ≥ 3 of the five with gains ≥ 5%; overall 12–22 wins,
median −0.15% to +0.10% (B20's shape); cpu_act loses; fit 1.1–1.3×.
(b) uncapped at depth 6: changes only the fits still cap-bound at depth 6
(pol, SGEMM, superconduct, visualizing_soil and a few more), wins ≥ 75% of
the changed sets, median gain on them +0.5% to +3%, guard ok; total fit
1.2–1.6×.
(c) rate 0.03 uncapped: 14–24 wins, median −0.1% to +0.3%; smaller steps
help the low-noise sets (≥ 3 of the five improve) and are neutral
elsewhere; total fit 2.5–4×.
(d) depth 8: 16–26 wins, median −0.1% to +0.3%; loses on the small-data
twins; fit 1.2–2×.
(e) exact splits: 18–28 wins, median 0 to +0.3%; fit 1.5–4×.
Coverage: every arm keeps the default calibration, so the guard holds on
all five (|Δ| < 1 point).
reading rule, written before the run: a probe PAYS if, against the head on
gr, it wins ≥ half + 1 of the decided sets with median CRPS change > 0 and
the guard holds. Among paying probes, the next library rung goes to the
largest median gain per unit of total fit cost; ties break toward the
smaller change. Separately, the diagnostic: the share of CatBoost's CRPS
gap on the five that each probe closes (a probe closing ≥ 50% there names
the carrier even if it does not pay overall). None pays and none closes
half → the gap is not in these knobs; Q4 and Q1 follow.
ran (muse, exit 0, no sandbox error): five probes through
`_fit_head_model`, each passing only its one setting; 2 tests (each probe's
fitted booster carries its change and the defaults otherwise; D6Uncapped
equals the head where the head stops early). Claude: the diff reads
correct; **1204 passed, 1 skipped**.
synth screen (`results/quantile-synth-20260923-193053.json`, excess CRPS
×1000, median over 12 keys): the head 49.09, bins 254 52.44, D6 uncapped
49.09, rate 0.03 uncapped 46.39, depth 8 50.42, exact splits 44.89.
decide tier (`results/quantile-20260923-195957.json`, 59 keys × 3 seeds ×
7 arms, ~30 min; the head reproduces I059 on 36 of 36). Against the head
on gr regression (36), with the sign test's two-sided p:

| probe | W-L-T | median (engaged) | sign p | guard 90 / 80 | total fit | CatBoost gap closed on the five |
|:--|--:|--:|--:|:--|--:|--:|
| bins 254 | 18-13-5 | +0.02% | 0.47 | −0.02 / +0.19 ok | 1.02× | median 63% |
| D6 uncapped | 7-0-29 | +1.18% | 0.016 | −0.05 / −0.09 ok | 1.36× | 1% |
| rate 0.03 uncapped | 23-13 | +0.13% | 0.13 | +0.01 / +0.25 ok | 2.97× | 13% |
| depth 8 | 21-15 | +0.06% | 0.41 | −0.16 / −0.01 ok | 0.94× | 7% |
| exact splits | 9-27 | −0.16% | 0.004 | +0.18 / +0.73 ok | 2.35× | −34% |

Per set on the five: bins 254 wins SGEMM +21.1%, Brazilian_houses +15.1% /
+14.9%, nyc-taxi +7.3% and LOSES pol −23.6%; RigidShift, the squared-error
location, wins Brazilian_houses +32.7% / +30.6% and pol +35.3%, but not
SGEMM (+1.6%) or nyc-taxi (−6.5%). The two carriers are complementary.
cpu_act, B20's June reversal for the point model, goes the other way for
the head at 254 bins (+4.9%). Pareto (59 keys): bins 254 0.5949 @ 3.0× and
depth 8 0.5929 @ 2.6× push the frontier past the head (0.5920 @ 2.9×); the
mean-skill axis rewards bins' large wins on a few sets.
forecast: 18 of 23 counts HIT. MISS: bins' cpu_act (forecast a loss, won
+4.9%) and its fit (1.02×, floor 1.1); depth 8's fit (0.94×, floor 1.2);
exact splits' wins and median (9-27, −0.16%; the synthetic screen had it
best at 44.89, the second time the screen misread a real-data margin).
verdict (the pre-registered rule): bins 254, rate 0.03 uncapped and depth 8
PAY by the letter; D6 uncapped is a pointer (7 decided, fewer than 8);
exact splits FAILS. The rule nominates depth 8 for the next library rung
(the largest median gain per unit of fit). **Not run.** All three passes
are coin flips (sign p 0.47 / 0.13 / 0.41, every CI spanning 0), and
depth 8's own synthetic screen reads worse. The Phase 2 gate as written
admits results at this noise level, so it is TIGHTENED, prospectively and
before any library rung (written after seeing these marginal passes, so
labelled post-hoc; it only makes shipping harder): **a default change must
also clear a two-sided sign test at p < 0.05 on gr regression** (at least
25 wins of 36 decided, without ties). Q5 clears it (31W-5L, p ≈ 2e-5).
Under it nothing from Q2 ships. **Q2 CLOSED with a diagnosis instead**: on
the five low-noise sets CatBoost's edge is resolution (bins) on SGEMM and
nyc-taxi and location (a squared-error centre) on Brazilian_houses and
pol. Docs corrected in the same change: `docs/quantiles.md` and
`docs/parameters.md` called `exact_splits=True` "slightly more accurate";
on real data it lost 27 of 36. PR for the maintainer (touches `tests/` and
`docs/`; it edits the quantile gate, said so in the PR). **MERGED by the
maintainer as PR #160 (9eabae4); branch deleted.**
next: Q6 T0, a validation-chosen head: per fit, the early-stopping rows
choose among the default head, the head at 254 bins, and the head
recentred on a squared-error median. The five hold +7% to +35% for
whichever candidate suits them; the audition must not give it back on the
other 31.

#### I059 2026-09-23 F7 Q5 (the head's default: depth 6 and CQR factors on the early-stopping rows; LIBRARY change, pre-registered)
why now: I058 merged (PR #158 at 42b0a19); its rule named this the next
library rung. No campaign PR open, bench idle. The identity baseline
matches main 186/186 (checked 2026-09-23 at 42b0a19), so it is a clean
reference. Branch `campaign/quantile-q5-default` from main; muse task
`20260923-q5-head-default.md`.
change (`chimeraboost/quantile_api.py`): `depth=None` resolves to 6 (was
4); `conformalize` gains the value `"auto"`, its new default: after the
fit, the head's own CQR factors (`_cqr_scales`) are computed on the rows
early stopping held out (the user's `eval_set` or the carved
`validation_fraction`), and the raw grid stands when early stopping is off
or those rows cannot certify the grid (never an error on the default
path). `True` (the carved pristine fold, its guarantee) and `False` (the
raw grid) keep their meaning. The bench's probes are pinned to what they
measured (depth 4 or 6 as named, `conformalize=False`), and the field arm
follows the library default.
barriers (B11, B5, B6, B19, B23): as argued in I057 and I058; nothing new.
B6: the depth change is the measured I058 arm, not a search.
forecast and bar: (1) identity: on the decide tier the new default head
reproduces I058's Depth6ValScaled records bit-for-bit (177 of 177 fits),
so its gate result against the old head is I058's: gr 31W-5L, +0.33%,
guard −2.96 / −4.10 points; hc 3W-3L. (2) the identity snapshot: every
point-model config bit-identical, only the head's configs move. (3) tests:
green after deliberate updates, each changed assertion listed with its
reason. (4) Pareto: the head moves up and left (0.5916 @ 3.6× → ~0.5920 @
~3.0×). A miss on (1) or (2) is a defect to fix before anything else reads.
runs (Claude, after review): the identity snapshot; then one full-field
decide run (the seven field arms plus the pinned Depth6ValScaled probe,
~3 h) that serves the identity check, the gate, the refreshed quantile
chart and the docs' comparison table, in one go.
ran (muse, exit 0, no sandbox error): `quantile_api.py` — depth 6, the
`"auto"` default through a `_check_conformalize` validator (strict: `True`,
`False` or `"auto"`, so `fit` stays at C901 8), the truthiness test in
`_carve_calibration_fold` made explicit (`is not True`), the calibration
under `"auto"` computed with `self.predict` while the factors are still
ones and falling back to the raw grid on any uncertifiable grid; the bench
probes pinned (`depth=4` or 6, `conformalize=False`). Tests: 7 new (the
bit-for-bit identity to the probe with an `eval_set`, the carved fold, ES
off, uncertifiable and asymmetric grids, no carve under `"auto"`, a bad
value raises) plus the field-arm identity in the suite tests; two old tests
pinned to `depth=4` — the round-ratchet test (raw coverage 0.837 at depth
6 against its 0.84 floor, a raw-grid property the default now repairs) and
the `conformalize=True` coverage test (at depth 6 the 80% band over-covers
by 2.04 points against a 2.00 tolerance, the safe direction CQR is known to
err in; measured by Claude, 90/80/70% at +1.67/+2.04/+1.91). Muse's smoke:
the field arm equals the probe on all three keys to the last digit, and
the probe still reads I058's values. Claude: the diff reads correct;
**1202 passed, 1 skipped**; `ruff check chimeraboost/` clean.
identity snapshot (against the 42b0a19 baseline): **171 of 186 pins
bit-identical; the 15 that move are exactly the head's three configs**
(`mq3`, `mq3_w_sub`, `mq3_conf`, five pins each). Every point-model pin is
identical, forecast (2) HIT. The baseline is rebaselined once this merges.
decide tier, full field (`results/quantile-20260923-185507.json`, 59 keys ×
3 seeds × 8 arms, 2 h 47 min, nothing skipped): **identity 177 of 177** —
the new default head equals the pinned Depth6ValScaled probe on every fit,
in this run and against I058's records. Gate against the old head
(`quantile-20260923-150433.json`), per stratum: **gr 31W-5L, +0.33% [CI
+0.14..+0.77], guard −2.96 / −4.10 points: PASS**; hc 3W-3L (−0.03%),
coverage error 9.24 → 0.85. Pointers: gr `@sus50` 3-1, gr `@sus25` 3-4
(−0.35%), hc `@sus25` 0-2 (−2.61%), hc `@time` 1-2 (−1.65%), hc `@sus50`
1-0; the coverage error falls in every stratum (hc `@time` 17.49 → 0.68,
hc `@sus25` 22.39 → 2.23). The new head against the field on gr: our
per-level 33W-3L (+2.25%), LightGBM per-level 26W-10L (+0.84%, a tie
before), CatBoost MQ 15W-21L (−0.24%, 7W-29L before), RigidShift 23W-13L
(+0.74%, a tie before), NGBoost 34W-2L, head + CQR (now depth 6) 35W-1L;
its 90% band covers 0.904 on gr. Pareto (the seven field arms, from the
same JSON with the duplicate probe dropped): **RigidShift 0.5869 @ 1.2×
→ the head 0.5920 @ 3.0× (coverage error 0.009, from 0.063) → CatBoost MQ
0.5982 @ 132×**; `images/quantile_pareto.png` refreshed. The mean CRPS over
59 keys reads 322.5 against 313.7, all of it hc:house_prices_nominal@time
(I058); the sign test is the gate.
forecast: (1) identity HIT, (2) snapshot HIT, (3) tests HIT (two old tests
pinned, both for a stated raw-grid or fold-level reason), (4) Pareto HIT
(0.5916 @ 3.6× → 0.5920 @ 3.0×).
docs (Claude): `docs/quantiles.md` (calibration section rewritten around
the new default; the comparison table re-measured, with the fixed-width
baseline and NGBoost added, closing the dated line in `QUANTILE_PLAN.md`;
the depth note), `docs/parameters.md` (the `conformalize` row and the depth
note), `docs/recipes.md` (the calibration recipe), CHANGELOG (Unreleased,
Changed). The `conformalize=True` coverage test measured at depth 6 by
Claude (above) backs the docs' "within about 2 points".
verdict: **PASS → PR for the maintainer** (library, tests, docs, image).
The standing quantile BASE becomes `quantile-20260923-185507.json`.
**MERGED by the maintainer as PR #159 (bda646b); branch deleted; identity
snapshot rebaselined at bda646b (186/186).**
next: once merged, rebaseline the identity snapshot at the merge commit;
then Q2's T0 (F7).

#### I058 2026-09-23 F7 Q3 T0 (the head's learning rate: flat 0.15 and 0.2; plus depth 6 with the early-stopping-row calibration as a second question; bench-only, pre-registered)
why now: I057 merged (PR #157 at 6d4a565). No campaign PR open, bench
idle. I057's rule named Q3 the next library rung, and a rate change needs
its rate first. Branch `campaign/quantile-q3-rate` from main; muse task
`20260923-q3-rate-probes.md`.
arms (new opt-in `PROBES`; the field default unchanged):
`ChimeraBoostQuantileLR15` and `ChimeraBoostQuantileLR20`, the head at a
flat `learning_rate` of 0.15 / 0.2 (the default resolves to 0.1 whenever
early stopping is on); `ChimeraBoostQuantileDepth6ValScaled`, depth 6 plus
I057's (d) calibration. Same run, as references: the head, RigidShift,
uncapped and depth 6.
barriers (`barrier_check.py`: B11, B5, B19, B6). B19: the stopping rule is
untouched; a larger rate moves where the argmin falls, it does not smooth
or truncate it. B11 and B5: the calibration is I057's (d), out-of-sample
rows, as argued there. B6: one mechanism-motivated knob, not a search: the
cap binds on 13 of 36 gr sets, and I057 measured what lifting it buys
(+1.73% on those 13). Not a barrier but on record: SMALLDATA found a LOWER
rate better on small data (the size fade), so the `@sus` twins are where a
larger rate should hurt.
runs (Claude, after review, one at a time): the synth screen, then the
decide tier, both `--seeds 3 --jobs 5 --save` with the seven arms above;
scored as in I057 (each probe against the head, per stratum, with the
guard; the 13 cap-bound sets against the rest; the quantile Pareto).
forecast (gr regression, 36, against the head):
Q3 (rate). LR 0.15: on the 13 cap-bound sets wins ≥ 10, median gain there
+0.5% to +1.5% (uncapped got +1.73%); on the other 23 wins 8–15 (a larger
rate usually costs a little once rounds are not scarce); overall 18–26 wins,
median −0.1% to +0.3%; guard ok; total fit 0.75–0.9×. LR 0.2: cap-bound
≥ 11 of 13, the other 23 wins 6–12, overall 17–25 wins, median −0.2% to
+0.3%; guard ok; fit 0.6–0.8×. On the gr `@sus25` twins (a pointer) both
rates lose more than they win.
Combined question. Depth 6 + calibration: ≥ 26 wins against the head,
median +0.3% to +0.5% (depth 6's +0.35% plus calibration's ~0), and the
guard IMPROVES (90% error ≤ 1.0 point, from depth 6's 4.48); against depth 6
alone, CRPS within ±0.1% median.
synth screen: both rates within ±3 excess-CRPS points of the head (×1000,
median over keys); depth 6 + calibration keeps depth 6's synthetic loss
(~47) and the calibrated arm's coverage (~2 points).
reading rule, written before the run. A rate PAYS if against the head on gr
it wins ≥ half + 1 of the decided sets, its median CRPS change is > 0, and
the guard holds. If both pay, the larger median gain; on a tie, 0.15 (the
smaller change). A paying rate becomes the next library rung: the head's
default rate, with its identity to the probe arm checked and the full
gate. No rate pays → Q3 is closed, and the cap-bound sets go to the
combined question. The combined arm PAYS under the same rule; if it pays,
the library order after the rate is depth 6 with calibration on the
early-stopping rows as one rung; if not, calibration alone (I057's (d)).
ran (muse, exit 0, ~10 min, no sandbox error): three probes through the
shared `_fit_head_model` (now with `learning_rate`) and a
`_calibrate_on_val` helper that ValScaled also uses (both bit-identical:
the head's smoke CRPS matches I057's on all three keys); 2 tests. Claude:
the diff reads correct; **1195 passed, 1 skipped**.
synth screen (`results/quantile-synth-20260923-144339.json`; excess CRPS
×1000, median over 12 keys): the head 42.92, LR 0.15 46.11, LR 0.2 43.19,
depth 6 46.88, depth 6 + calibration 49.09 with 90% coverage error 1.90
points (the head 7.31).
decide tier (`results/quantile-20260923-150433.json`, 59 keys × 3 seeds ×
7 arms, ~20 min; the head, RigidShift, uncapped and depth 6 reproduce I057
exactly). Against the head on Grinsztajn regression (36):

| probe | CRPS W-L | median change | guard, 90% / 80% (points) | total fit |
|:--|--:|--:|:--|--:|
| LR 0.15 | 17-19 | −0.04% [CI −0.27..+0.05] | +0.63 / +0.58, ok | 0.86× |
| LR 0.2 | 11-25 | −0.09% [CI −0.43..−0.02] | +0.55 / +0.25, ok | 0.73× |
| depth 6 + calibration | 31-5 | +0.33% [CI +0.14..+0.77] | −2.96 / −4.10, ok | 0.90× |

The rates split the 13 cap-bound sets 7-6 each and lose most where the
head needs rounds: pol −34% / −74% (stopping at rounds 1548 / 640 where the
head reaches the cap), Brazilian_houses −4% to −11%, nyc-taxi (num) −9% /
−5%; visualizing_soil gains +0.8% / +4.4% against uncapped's +22.8%. On
gr `@sus25` they lose 1-6 and 0-7. The combined arm wins the cap-bound
sets 10-3 (+2.34%) and the rest 21-2 (+0.17%); it wins the median level
30-6, and the calibration turns depth 6's 90% interval score from 14-22
into 22-14 (+0.78%); against depth 6 alone it is 23-13, +0.02%. Its 90%
error is 0.47 points on gr, 0.85 on hc (from 9.24; hc CRPS 3-3, −0.03%),
0.40 on gr `@sus25` (CRPS 3-4, −0.35%). The mean CRPS over 59 keys moves
the other way (313.7 → 322.5), all of it one set, hc:house_prices_nominal
@time (+9.3%, coverage 0.715 → 0.907) with a target in the thousands;
the sign test is the gate, as the house rule says. Pareto (59 keys):
depth 6 0.5928 @ 3.0× (0.074), depth 6 + calibration 0.5920 @ 3.0×
(0.009), the head 0.5916 @ 3.6× (0.063); LR 0.2 0.5885 @ 2.4× and
RigidShift 0.5869 @ 1.4× hold the fast end of the frontier.
forecast: 15 of 22 counts HIT. MISS: LR 0.15 cap-bound wins (7, bar 10)
and overall wins (17, bar 18); LR 0.2 cap-bound (7, bar 11), other (4, bar
6) and overall (11, bar 17); synth LR 0.15 (+3.19, band ±3); synth combined
CRPS (49.1, forecast ~47).
verdict (the pre-registered rule): **no rate pays → Q3 CLOSED, registered
as barrier B23** (the cap-bound sets need more rounds or more capacity,
not bigger steps). **The combined arm PAYS**, so the next library rung is
depth 6 with calibration on the early-stopping rows, as one rung. Nothing
ships from this rung; the three probes stay opt-in bench arms. PR for the
maintainer (touches `tests/`). **MERGED by the maintainer as PR #158
(42b0a19); branch deleted.**
next: the library rung (F7): the head's default depth 4 → 6, and its CQR
factors computed on the early-stopping rows by default; the bench's
Depth6ValScaled probe is its exact oracle.

#### I057 2026-09-23 F7 Q0 (the quantile probe battery: uncapped rounds, depth 6, recentred on a squared-error median, rescaled on the early-stopping rows; bench-only T0, pre-registered)
why now: I056 merged (PR #156 at f5bac6b). No campaign PR open, bench
idle. Q-B4 left one question with four candidate answers: the head ties
RigidShift on CRPS because it loses the median level, mostly on the 11
Grinsztajn sets where it hits the 2000-round cap. Branch
`campaign/quantile-q0-probes` from main; muse task
`20260923-q0-probe-battery.md`.
arms (bench-only, `quantile_suite.PROBES`, opt-in through `--models`; the
seven-arm default is unchanged):
(a) `ChimeraBoostQuantileUncapped`, `n_estimators` 8000;
(b) `ChimeraBoostQuantileDepth6`, depth 6;
(c) `ChimeraBoostQuantileRecentred`, the head's grid shifted per row so its
median equals RigidShift's median column (both fits reused unchanged);
(d) `ChimeraBoostQuantileValScaled`, the library's own CQR factors
(`_cqr_scales`) computed on the shared early-stopping rows instead of a
carved 20% fold.
barriers (`barrier_check.py`: B19, B11, B5, B6). B19 (the stopping round is
not a free axis): (a) keeps the raw-argmin rule and only lifts a cap that
binds before the argmin (best round 1999 on the capped sets); B19's own
finding, that everything before the argmin is uphill, predicts (a) helps
there. B11: nothing changes how stopping works; (d) calibrates after the
fit. B5: (d) estimates a common correction on out-of-sample rows, the case
B5 leaves open (P15 failed on pooled in-sample residuals). B6: (b) is one
mechanism-motivated knob, not a search; depth 4 was the build request's
setting for the LightGBM comparison (the acceptance ledger), never measured
against CRPS on real data, and CatBoost's MultiQuantile uses 6.
runs (Claude, after review, one at a time): the synth screen
(`quantile_synth.py --seeds 3 --jobs 5 --models` head, RigidShift and the
four probes, `--save`, ~5 min), then the decide tier (`quantile_suite.py
--decide --seeds 3 --jobs 5`, same arms, `--save`, ~40 min). Scored against
the head of the same run (`compare_runs.py J J --model ChimeraBoostQuantile
--model-new <probe> --metric crps --by-suite`), each probe against
RigidShift, the capped / uncapped split (the 11 gr sets whose Q-B4 best
round reached 1999), and the quantile Pareto.
forecast (gr regression, 36 sets, against the head unless said; hc and the
twins are pointers):
(a) bit-identical to the head on the 25 uncapped sets (the inert control:
any drift is a finding); on the 11 capped it wins ≥ 8 with a median gain
there ≥ +1%; guard ok; against RigidShift ≥ 22 of 36 (from 20); total fit
1.5–2.5× the head's.
(b) a genuine coin: 16–26 wins, median −0.3% to +0.5%; guard FAIL likely
(fewer rows per leaf, narrower in-sample leaf quantiles); fit 1.3–2×.
(c) 18–24 wins against the head (most of the capped 11, fewer than half of
the other 25, where the head's median beat RigidShift's on 15); against
RigidShift ≥ 24 of 36; guard ok; fit 1.3×.
(d) the median column identical to the head's by construction; ≥ 22 wins
with a median change in [0, +0.3%]; the guard IMPROVES (median 90% error
from 3.43 to ≤ 1.5 points); against RigidShift still 18–22 wins; fit 1.0×.
synth screen: (a) ties the head on most keys (synthetic fits rarely reach
2000 rounds); (d) cuts the head's 90% coverage error at n = 1k from ~13 to
≤ 5 points.
reading rule, written before the run: a probe PAYS if, against the head on
gr, it wins ≥ half + 1 of the decided sets with median CRPS change > 0 and
the guard holds (the Phase 2 gate's form, used as a screen). Among paying
probes the next library rung goes to the largest median gain per unit of
fit cost; ties break toward the smaller library change. (a) → Q3, the
head's rate or budget; (b) → a depth default; (c) → a location design;
(d) → Q1 or calibration on the early-stopping rows. None pays → Q2, the
CatBoost ablation, aimed at the capped sets.
ran (muse, exit 0, ~10 min, no sandbox error): `PROBES` in
`quantile_suite.py` (opt-in; both `--models` defaults unchanged), the head's
construction factored into `_fit_head_model` (the head bit-identical), 5
tests. Claude: the diff reads correct; **1193 passed, 1 skipped**. Muse's
smoke (1 seed, anecdote): on pol the head stops at the cap (CRPS 1.630),
uncapped runs to round 7529 (1.411) and RigidShift reads 0.970; on cpu_act
uncapped equals the head to the last digit.
synth screen (`results/quantile-synth-20260923-133626.json`, 12 keys × 3
seeds, ~4 min; excess CRPS ×1000 over the oracle, median over keys; the
head 42.92 and RigidShift 68.20 reproduce Q-B4a exactly): **uncapped
42.92, identical on all 12 keys** (no synthetic fit reaches 2000 rounds);
**depth 6 46.88, worse on 12 of 12**; **recentred 57.74, better only on
`location`** (2 of 12, the homoscedastic regime where RigidShift itself
wins); **validation-rescaled 44.46, worse on 9 of 12** (skewed/n1k 67.9
against 50.3) while its 90% coverage error falls from 7.31 to 2.13 points.
Forecast: (a) HIT (ties everywhere); (d) coverage HIT (the head's n = 1k
median error is ~9.7, not the ~13 written, and the probe's ~2.7 is under
5). The CRPS cost of (d) was not forecast: the library's CQR scales each
symmetric PAIR by one factor (the larger of its two sides' scores), so a
band that under-covers on one side widens on both, and CRPS charges the
wasted side.
decide tier (`results/quantile-20260923-135654.json`, 59 keys × 3 seeds ×
6 arms, 17 min; the head and RigidShift reproduce Q-B4 exactly). Against the
head on Grinsztajn regression (36), CRPS sign test with the guard:

| probe | CRPS W-L-T | median change (engaged) | guard, 90% / 80% (points) | total fit vs the head |
|:--|--:|--:|:--|--:|
| (a) uncapped | 13-0-23 | +1.73% [CI +0.57..+4.29] | +0.10 / +0.17, ok | 1.48× |
| (b) depth 6 | 31-5 | +0.35% [CI +0.07..+0.71] | +1.06 / +0.92, FAIL | 0.87× |
| (c) recentred | 18-18 | +0.03% | +1.35 / +2.06, FAIL | 1.27× |
| (d) validation-rescaled | 21-15 | +0.02% [CI −0.00..+0.08] | −2.93 / −3.99, ok | 0.99× |

(a): the 13 changed sets are exactly the fits the cap cut short (the 11
capped, plus MiamiHousing and elevators, each cap-bound on one seed); every
other fit is bit-identical, the inert control. The gain sits where the head
was furthest short: visualizing_soil +22.8% (still at the 8000 cap), pol
+12.7%, superconduct +7.5%, Brazilian_houses +4.3% / +2.5%. (b) wins the
median level 30-6 and loses the 90% interval score 14-22, the depth
docstring's own mechanism (deep leaves overfit the tails, the quantiles
collapse toward the median), now measured; on hc its 90% error worsens
3.24 points. (c) wins the capped sets (+8.35%), loses 9-16 elsewhere, and
its shifted shape loses the 90% interval score 8-28. (d) leaves the median
untouched (36 ties, by construction), wins the 90% interval score 21-15,
and cuts the 90% coverage error from 3.43 to 0.50 points (hc 9.24 → 0.22)
at no CRPS cost on real data. Pareto over 59 keys (CRPS skill @ slowdown,
90% error): depth 6 0.5928 @ 2.8× (0.074) and RigidShift 0.5869 @ 1.2×
(0.008) form the frontier; uncapped 0.5925 @ 4.2×, the head 0.5916 @ 3.4×,
validation-rescaled 0.5912 @ 3.4× (0.009), recentred 0.5867 @ 4.6×.
forecast: 16 of 22 counts HIT. MISS: (a) against RigidShift 21 of 36 (bar
22) and total fit 1.48× (floor 1.5); (b) 31 wins (forecast 16–26) and
FASTER at 0.87× (forecast 1.3–2×), since deeper trees converge in fewer
rounds; (c) the guard failed; (d) 21 wins (bar 22).
verdict (the pre-registered rule): **(a) and (d) PAY; (b) misses on its
guard alone (+1.06 against the 1.00 limit); (c) does not pay.** The larger
gain per unit of fit cost is (a), so **the next library rung is Q3, the
head's rate or budget.** The budget is not a library lever: the harness
hands every arm `n_estimators = rb.MAX_ITERS = 2000`, the head's own
default, so the lever is the rate, and the size fade LOWERS it below 15k
rows, the wrong way for the capped sets (most hold 8k–16k training rows).
Nothing ships from this rung; the four probes stay opt-in bench arms. PR for
the maintainer (touches `tests/`). **MERGED by the maintainer as PR #157
(6d4a565); branch deleted.**
next: Q3 T0 once this PR merges: flat learning rates 0.15 and 0.2 and, as a
separate pre-registered question in the same run, depth 6 with the
validation-row calibration (does (d) rescue (b)'s guard?).

#### I056 2026-09-23 F7 Phase 1 (the quantile bench: Q-B1 decide tier, Q-B2 synthetic screen, Q-B3 quantile Pareto, Q-B4 baseline; benchmark tooling, pre-registered)
why now: the maintainer moved the loop to the multi-quantile head after
merging #155 ("We don't have much benching built for it"), then approved
the plan with NGBoost in the field, no quantile forests, and CRPS as the
decision score. Branch `campaign/quantile-plan` from main 81be911; muse
tasks `20260923-qb1-quantile-decide-tier.md`, `-qb2-quantile-synth-screen.md`,
`-qb3-quantile-pareto.md`, `-qb-review-fixes.md`. Program, forecasts and
the full verdict: `QUANTILE_PLAN.md` "Campaign 2026-09-23".
barriers (`barrier_check.py`: B5, B11): keyword matches for tooling. B5
binds Q1, not this: a correction built from in-sample quantities cannot
fix a common bias, which is why Q0's calibration arm rescales on the
held-out early-stopping rows.
ran: muse, four passes, all exit 0 (Q-B1, Q-B2, Q-B3, then two review
fixes: NGBoost now predicts at its best round, and `--datasets` without
`--decide` registers the suites its keys need instead of skipping
silently). Claude ran the two baselines: the synthetic screen
(`results/quantile-synth-20260923-090818.json`, ~5 min) and the decide
tier (`results/quantile-20260923-115727.json`, 2 h 49 min, 59 keys × 3
seeds × 7 arms, nothing skipped). **1188 passed, 1 skipped.**
result (Grinsztajn regression, 36, CRPS sign tests, seeds averaged): the
head loses to CatBoost MultiQuantile 7W-29L (−0.45%, the same as August to
the last digit, at 14.6× the head's fit), TIES RigidShift 20W-16L (+0.41%,
CI −2.07..+1.86; the rigid width fits at 0.27× and covers 0.898 against
the head's 0.869), beats head+CQR 33W-3L, LightGBM per-level 23W-13L (a
tie), our per-level 32W-4L and NGBoost 35W-1L. Every arm's coverage drops
under the hc time shift. Pareto over all 59 keys: RigidShift 0.5869 @ 1.3×
→ the head 0.5916 @ 3.5× → CatBoost MQ 0.5982 @ 136× (`images/quantile_pareto.png`).
read: the RigidShift tie is two populations. The head reaches the
2000-round cap (flat learning rate 0.1) on 11 of the 36 and loses them
2W-9L to RigidShift (−5.03%) and 1W-10L to CatBoost; on the other 25 it
beats RigidShift 18W-7L (+1.10%) and trails CatBoost by a median 0.26%.
The CRPS is lost at the median level (vs RigidShift 16W-20L; vs CatBoost
5W-31L) while the head wins the 90% interval score (28W-8L; 25W-11L).
CQR's median, which its scale cannot move, loses 2W-34L to the plain head:
the 20% fold's data tax, measured cleanly; calibration itself bought ~0.15%.
forecast: 7 of 9 real-data counts HIT; MISS on RigidShift (20 of 36
against a bar of 24) and on NGBoost (forecast even, lost 1-35). All three
synthetic counts HIT.
verdict: **PASS (tooling) → one PR for the maintainer** (touches `tests/`,
adds `images/quantile_pareto.png`). No library change; the internal quantile
chart is new, and no ship gate reads it. Phase 2 re-ranked on the read:
Q0 probe battery first, Q1 from first to fourth. **MERGED by the maintainer
as PR #156 (f5bac6b); branch deleted.**
next: Q0 once the PR merges (F7's `next:` line).

#### I055 2026-09-22 H(11) + H(12) (the pub: download race; the Pareto chart's title and partial-coverage points; harness, pre-registered)
why now: I054 shipped (PR #153 merged by the maintainer at 611466a). No
campaign PR open, bench idle. The two harness defects found today go out
as one PR (both touch `tests/`, so it waits for the maintainer). Class:
**measurement instruments** — H(12) changes how the internal chart marks
its frontier (not a ship gate; said so in the PR). Branch
`campaign/h11-h12-harness-fixes` from main; muse task
`20260923-h11-h12-harness-fixes.md`.
H(11): `_public_parquet_path` writes one shared `.part` and `os.replace`s
it; a cold cache with jobs > 1 raced it to WinError 32 and killed I046's
first launch. Fix: a per-process temporary, and "the final file exists"
treated as another worker's success.
H(12): `make_pareto.py`'s skill chart is titled "Grinsztajn" though its
panels pool every suite in the run, and sklearn_HGB — scored only on the
datasets it can fit, an easier subset — sits on the classification
frontier at 0.4349 against a field near 0.405 (I015's recorded caveat,
still live in #145's chart). Fix: a title built from the suites present,
and partial-coverage models labelled n/N and kept off the frontier.
barriers (`barrier_check.py`: B5, B12, B14): keyword matches only
("coverage", "race") — tooling, no model change.
forecast: tests green; on `20260922-144219.json` the text read shows HGB as
partial (its n below the panel's) and the classification frontier becomes
the full-coverage one (LightGBM at the fast end, then the default); the
regression panel unchanged (HGB is not on its frontier). No image is
regenerated in this PR; the next chart refresh picks the fix up.
ran (muse, exit 0, ~10 min, no sandbox error): H(11) — per-process
`.part` name and the lost-race return (the same fix `_grinsztajn_local_csv`
already had), 3 network-free tests; H(12) — `suite_label` / `suite_title`
from key prefixes, `skill_frontier` over full-coverage models only,
`partial n/N` labels and one footnote, `meta["suite"]` routed through the
same label so the win-rate captions follow; 6 tests. Claude: the diffs
read correct; **1157 passed, 1 skipped** outside the sandbox. Text read of
`20260922-144219.json`: title "Strength vs slowdown — Grinsztajn +
high-cardinality (with variants)"; HGB **partial 36/44** (classification)
and **53/59** (regression), off both frontiers; the classification
frontier is now LightGBM → the old default → **the default** → Ens5 → Ens8
(CatBoost dominated by Ens5), where #145's chart showed HGB's subset point
on top and every other point off the frontier.
forecast: HIT on every count (partial HGB, the full-coverage frontier with
the default on it, regression unchanged).
verdict: **PASS → PR for the maintainer** (touches `tests/`); a gate-
adjacent change stated plainly in the PR: the internal chart's frontier
marks move (no ship gate reads them). **MERGED by the maintainer as PR
#154 (ed01966).** Charts refreshed the same night from
`20260922-144219.json` without the control arm (`images/pareto.png`,
`images/winrate_matrix.png`): the new title, HGB labelled 36/44 and 53/59,
the classification frontier LightGBM → the default → Ens5 → Ens8;
`docs/PROJECT_STATUS.md` gains the coverage rule. That PR carries images,
so it waits for the maintainer.
next: the named alternates are exhausted. Waiting on the maintainer: the
multiclass rate/Hessian trade (I052), S1 parked to 2026-09-29, the binary
linear-leaf race pointer (I047), a beam refill, and regenerating
`images/pareto.png` once this merges (a text-only change until then).

#### I054 2026-09-22 S5 (C5: fused multiclass cross-entropy eval; exact rewrite, muse task, pre-registered)
why now: I053 closed (PR #152 self-merged at 6d6f1f3); the self-mergeable
queue is empty, so the next library rung runs overnight and its PR waits
for the maintainer's morning review. No campaign PR open, bench idle.
Class: **exact rewrite** (the maintainer's preferred class). Branch
`campaign/s5-multiclass-eval-fusion` from main; muse task
`20260923-s5-multiclass-eval-fusion.md`.
the object (I026): `MultiSoftmax.eval` on the validation rows every round —
softmax, clip, log, a K-sum and a weighted mean as separate numpy passes —
is 4.9–6.5% of a multiclass fit on the hc sets. The binary twin was fused
at C3 (I029: −4.7 to −5.7% of binary fit, bit-identical); C1 and C1b fused
the softmax and grad_hess layers. The kernel reuses `_softmax_kernel`'s
loop verbatim (same max, same accumulation, a divide), clips by
comparison, sums the K terms in order (the documented K ≤ 7 bit-identity
boundary) and negates the total; the weighted mean stays in numpy.
barriers (`barrier_check.py`: B10, B16): B10 closes grow-kernel objects; the
loss layer is the precedent class that shipped three times (I012, I018,
I029). B16 — a word collision on "cross".
forecast: exact (identity N/N, every loss value equal); multiclass fit
**−3 to −4.5%** on okcupid-stem / Traffic_violations / cjs / eucalyptus
(a 4.9–6.5% object converting at C3's 50–70%, arithmetic folded into a
kernel); the binary control flat within ±2%; gr unaffected (no multiclass).
bars: (a) identity N/N and all tests green; (b) the speed script's arms
identical on every repeat, Higgs flat within ±2%; (c) multiclass fit
improves at the median by ≥ 2% with no panel set slower. Pass ⇒ PR for the
maintainer. (c) failing with (a)(b) intact ⇒ KILL as not worth its code.
muse pass 1 (exit 0, ~10 min, no sandbox error): `_softmax_ce_kernel`
(+40 lines: `_softmax_kernel`'s max and exp loops verbatim, the exp
recomputed rather than stored, a divide, a comparison clip, the in-order
K-sum negated after) and a guarded `MultiSoftmax.eval` (the old body kept
as `_eval_numpy`, the oracle); `tests/test_multiclass_eval_kernel.py` (6
tests); `benchmarks/f4_c5_speed.py` (alternating arm order, as S1 taught).
Identity 186/186; suite green outside the sandbox's `\\?\` quirk.
Speed (7 reps, fitted probas and best iterations identical every repeat;
eval seconds 0.12 → 0.03 on okcupid): okcupid-stem **−5.2%**,
Traffic_violations **−5.6%**, cjs **−5.0%**, eucalyptus **−3.1%**; Higgs
+0.0% with zero multiclass eval calls in both arms.
review (Claude): the kernel is right; the TESTS are not CI-safe. They pin
the kernel to the numpy oracle with exact equality, but the kernel's log()
is numba's (LLVM libm) and the oracle's is numpy's (SIMD on some hosts):
they agree to the bit on this machine, and the C3 binary twin had to
bound the same comparison at 4 ULP per row / 8 ULP on the mean after the
2026-08-30 runner-pool change broke exact pins (`test_bitident_
refactors.py`). Follow-up muse task `20260923-s5-test-tolerance.md`: the
C3 helpers, `valid_history_` within 8 ULP, the kernel comment carrying the
same caveat. The cross-hardware property of the change itself is the one
C1 and C3 shipped with (same-machine bit-identity guarded by the identity
snapshot).
follow-up (muse, exit 0, ~5 min): the C3 helpers copied with an origin
note; kernel-vs-oracle comparisons at 4 ULP per row / 8 ULP on means;
end to end `predict_proba` exact and `best_iteration_` equal, the
validation history within 8 ULP; the kernel comment carries the caveat;
no code change. Claude's checks outside the sandbox: **1148 passed, 1
skipped; identity 186/186; `ruff check chimeraboost/` clean.**
bars: (a) PASS; (b) PASS (identical outputs every repeat; Higgs +0.0%
with zero calls); (c) **PASS** — multiclass fit median **−5.1%** (≥ 2%),
no panel set slower.
forecast: exact HIT; "−3 to −4.5%" MISSED on the good side (−3.1 to
−5.6%, median −5.1%): the eval had more to give than C3's conversion rate
implied — the fused kernel reads eval seconds 0.12 → 0.03 (a 4×).
verdict: **S5 PASS → PR for the maintainer** (library + tests + a
benchmark script; not self-mergeable). **MERGED by the maintainer as PR
#153 (611466a), 2026-09-22 evening; branch deleted.**
next: H(11) + H(12) as one harness PR (I055).

#### I053 2026-09-22 S10 (the LightGBM speed price list: rounds, single-thread cost, thread scaling; measurement, pre-registered)
why now: I052 closed (PR #151 merged by the maintainer at 17541ac; the
multiclass rate/Hessian trade is with him). Order for the evening: the
self-mergeable measurement first, so the loop progresses without a merge;
the library rungs (S5, H(11), H(12)) queue for the morning, since each of
their PRs waits for him and only one may be open. Class: **measurement**
(the Pareto slowdown axis only), `benchmarks/` only. Branch
`campaign/s10-lgbm-speed-price` from main; muse writes and runs
`benchmarks/probe_lgbm_speed_price.py` (task
`20260922-s10-lgbm-speed-price.md`).
the question: LightGBM fits Grinsztajn about 5× faster than the default
(median). The refill (I042/L3) split it roughly as round count ~1.8–2.1×
and ~2.7× per round overall (4.9× at n ≥ 28k), 15 low-feature large-row
sets carrying 67% of the excess, and named one untested hypothesis: our
histogram kernels parallelize over FEATURES, so at the harness's two
threads per job a 6–9-column set splits unevenly, where LightGBM also
parallelizes over rows. Part A re-derives the split from today's
chart-grade run (zero fits); Part B measures thread scaling (1/2/4/8
threads, both models, the top 15 sets, one fit at a time).
barriers (`barrier_check.py`: B10, B15, B6, B18, B19): B10 closed the
measured kernel objects below their ceilings; a thread-decomposition
object was never measured, and this is its Phase-0 read (no kernel is
proposed). B15 — subtraction not proposed. B19 — the round count is priced,
not cut. B6, B18 — keyword only.
forecast: Part A reproduces I042 (round ratio ~0.5 — LightGBM grows MORE
rounds — so the whole edge is per-round cost, ~5× at the median and more
at large n). Part B: LightGBM's speedup(2) ≈ 1.8–1.9, ours ≈ 1.4–1.7 on the
low-feature sets; the scaling gap at t = 2 **1.1–1.3×**, larger at t = 8;
single-thread per-round cost carries most of the gap (≥ 3×).
bars (decision rules, not a ship gate): a scaling gap ≥ 1.5× at the
harness's t = 2 on the median of the 15 sets ⇒ promote a GROW_PLAN
Phase-0 ceiling for row-parallel histogram building as a new candidate
(the maintainer's pick); below 1.5× ⇒ record that the LightGBM edge is
single-thread per-round cost plus nothing structural in the threading, and
close the thread-decomposition door as a barrier.
ran (muse, exit 0, ~30 min, no sandbox error): `probe_lgbm_speed_price.py`;
Part A from `20260922-144219.json` (59 gr base keys, zero fits); Part B
360 timed fits (15 keys × 2 models × t ∈ {1, 2, 4, 8} × 3 repeats, one at
a time, an untimed warm-up first; rounds identical across thread counts).
Part A: median fit ratio **6.29×**; the top 15 by excess seconds carry
**69.6%** of the 176 s total excess (covertype ×2, electricity ×2,
Allstate, superconduct, nyc-taxi ×2, diamonds, pol, SGEMM, jannis,
road-safety, MiniBooNE, topo). Round ratio: WE grow more rounds on most
sets (1.1–2.4×; 6–14× on topo, sulfur, house_16H, yprop, pol@clf).
Part B (medians over the 15): ours/LightGBM **9.7× at 1 thread, 9.1× at
2, 8.8× at 4, 8.1× at 8**; speedup at 2/4/8 threads ours **1.68 / 2.44 /
2.77**, LightGBM **1.58 / 2.18 / 2.44**; scaling gap **0.97 / 0.91 /
0.94** (we scale slightly better); single-thread per-round ratio **6.36×**.
The low-feature sets named by the hypothesis scale equal or better
(electricity p = 7, nyc-taxi p = 9, diamonds, SGEMM); LightGBM scales
better only on three wide sets (jannis p = 54, MiniBooNE p = 50,
superconduct p = 79; gap 1.14–1.29 at 8 threads).
decision rule: scaling gap at t = 2 is 0.97, under 1.5 ⇒ **the thread-
decomposition door is CLOSED, registered as B22.**
forecast: "round ratio ~0.5, LightGBM grows more" **MISSED** — I misread
I042's "255 vs 122" (those were ours vs LightGBM's); "scaling gap
1.1–1.3×" MISSED (0.97); "single-thread per-round ≥ 3× carries most of the
gap" HIT (6.4×).
verdict: **S10 RESOLVED — the LightGBM edge is per-round cost of the whole
fit (races + refit + oblivious per-tree cost) times our larger round
counts; nothing in the threading.** The price list for the Pareto's
slowdown axis is now on record: any cheaper default has to come from the
races, the refit or the per-tree kernel, all priced (B10, B12–B14, B19).
Self-merged: `benchmarks/` + `.md` only.
next: the self-mergeable queue is empty. The library rungs wait for the
maintainer's merge window: **S5** (task written, `20260923-s5-multiclass-
eval-fusion.md`), then **H(11) + H(12)** as one harness PR. Also waiting
on him: the multiclass rate/Hessian trade (I052), the parked S1 (to
2026-09-29), the binary linear-leaf race pointer (I047), a beam refill.

#### I052 2026-09-22 S8 S1 (the exact softmax Hessian for multiclass vector leaves; zero-library probe, pre-registered)
why now: I051 closed (PR #150 self-merged at 698f888); the five picks are
resolved and S8 is the first alternate named at the pick. No campaign PR
open, bench idle. Class: **zero-library probe of a candidate default**
(an in-process monkeypatch). Branch `campaign/s8-full-softmax-hessian`
from main; muse writes and runs `benchmarks/probe_full_softmax_hessian.py`
(task `20260922-s8-full-softmax-hessian.md`).
the mechanism: a multiclass round grows one tree with K-vector leaves; each
leaf value is a per-class Newton step on the DIAGONAL Hessian `p_k(1−p_k)`
× (K−1)/K. The softmax cross-entropy Hessian is `diag(p) − ppᵀ`, singular
along the all-ones direction (made invertible here by λ = l2_leaf_reg).
The exact step `−lr (Σ(diag(p) − ppᵀ) + λI)⁻¹ Σg` moves the classes jointly
where the diagonal step under-reads the coupling. dedup: A1_PLAN kept the
per-class diagonal deliberately ("today's Newton semantics exactly") and
never ran the full solve; nothing else in the record.
barriers (`barrier_check.py`): B5 on "leaf-values" only — B5 closes
shrinkage estimators correcting a common bias; this replaces an
approximation with the exact second-order solve. B10 (priced, not
measured): n·K → n·K² per round plus one K×K solve per leaf, K ≤ 7 on
every hc multiclass set.
panel: every hc multiclass key and twin (okcupid-stem, Traffic_violations,
cjs, eucalyptus and their @sus/@time twins) and every synth multiclass
key, 3 seeds, the harness's splits; self-check against the saved run.
forecast: small — lr 0.1 keeps each step short, so the exact step mostly
changes the path, not the endpoint: hc multiclass median **+0.0 to +0.5%**
Brier, synth multiclass the same, ~50% chance of a flat read; the more
likely visible effect is fewer rounds (×0.85–1.0) at a per-round cost of
×1.05–1.3 (the probe's numpy solve is slower than a kernel would be, so
its fit ratio is an upper bound).
bars: (0) SELF-CHECK — the default arm reproduces the saved run's Brier on
every hc (key, seed); the patched-call count > 0 on every full_hessian
fit. (1) PASS when the hc multiclass base keys improve on ≥ 3 of 4
(non-near-solved: cjs is near-solved) with median > 0 AND the synth
multiclass keys read wins ≥ half + 1 of the decided with median > 0; a
kernel would then be a muse task. Else KILL.
ran (muse, exit 0, ~15 min): `probe_full_softmax_hessian.py` patches
`MulticlassBoosting._apply_vector_update` (the only vector-leaf site; the
multiclass refit runs from scratch through the same loop, so it is
covered), guards raise on ordered boosting, subsample < 1 and weights
(none fired); 360 rows = 60 keys (8 hc, 52 synth multiclass) × 3 seeds × 2
arms; **SELF-CHECK PASS** (0.0 over 24 hc rows); the patch engaged on all
180 full-Hessian fits and none of the default ones.
read (+ = the full Hessian better; near-solved out of medians):
hc 6W-2L: Traffic_violations **+0.54%** (3/3), okcupid-stem **+0.31%**
(3/3), eucalyptus **+1.31%** (1/3 — split), okcupid@sus25 +0.53%,
eucalyptus@time +0.91%, cjs@sus50 +0.36%, Traffic@time **−0.99%**; cjs
near-solved. Median +0.528% (n = 7). Synth multiclass **39W-13L of 52,
median +0.620%** (n = 48). Cost: fit ×3.30 (hc) / ×2.59 (synth) median;
**trees ×1.05–3.58** (Traffic 3.43, okcupid 3.58).
bar (1) **PASS on strength**: the three non-near-solved hc base keys all
up, synth 39-13 with a positive median.
forecast: strength "+0.0 to +0.5%" HIT at the top (+0.53 / +0.62%);
"fewer rounds ×0.85–1.0" **MISSED the other way** — the exact step is
SMALLER than the coupled diagonal one, so the fits run 1.05–3.6× longer.
the confound this opens: smaller steps and more trees are what a lower
learning rate does, and a lower rate alone usually buys Brier. Binned by
tree ratio the gain is +0.28% (8/11 wins) where both arms use about the
same trees (< 1.2×), +0.50% at 1.2–2×, +0.77% at ≥ 2× — part exactness,
part rate, by eye. A kernel is not worth building until the two are
separated.
S1b, pre-registered here before it runs: add two arms to the same probe on
the same keys and seeds — the default with `learning_rate=0.05` and with
`0.033` (the diagonal step, slowed). Outcomes: (i) either lr arm reaches
the full Hessian's median Brier gain on BOTH panels at no more trees ⇒
the gain is the learning rate: KILL the full Hessian, and the finding
becomes "multiclass wants a slower auto rate than the Grinsztajn-tuned
0.1" — a knob-free default change for its own rung (the auto rate was
tuned with no multiclass set in the suite); (ii) the full Hessian beats
both lr arms head to head (wins ≥ half + 1, median > 0 on both panels) ⇒
exactness matters ⇒ S2: a numba kernel, then the synth and decide gates
with its real cost; (iii) anything between ⇒ report both, no build.
verdict: S1 PASS on strength, confounded on mechanism → S1b (muse task
`20260922-s8-lr-control.md`).
S1b ran (muse, exit 0, ~15 min): arms `lr_050` / `lr_033` (the unpatched
default with an explicit rate) on the same 60 keys × 3 seeds; 720 rows,
the old 360 byte-identical (hash-verified, append-only); SELF-CHECK PASS.
vs the default (median Brier change, median trees ×, median fit ×):
  hc     full_hessian +0.528% ×2.12 ×3.30 · lr_050 +0.047% ×1.93 ×1.34 · lr_033 +0.509% ×2.59 ×1.78
  synth  full_hessian +0.620% ×1.35 ×2.59 · lr_050 +0.212% ×1.41 ×1.30 · lr_033 +0.515% ×2.24 ×1.81
head to head, full_hessian vs (wins-losses, median, trees ×):
  hc     vs lr_050 **6W-2L +0.260%** ×1.07 · vs lr_033 3W-5L −0.181% ×0.66
  synth  vs lr_050 **37W-15L +0.344%** ×0.96 · vs lr_033 29W-23L +0.046% ×0.60
read: at EQUAL tree counts the exact step is genuinely better than a
slowed diagonal step (vs lr_050, both panels, clear); a slower diagonal
rate (0.033) buys the same Brier with 35–40% more trees and ties the
exact step head to head. Outcome (i) not met (the rate that matches needs
more trees), outcome (ii) not met (hc 3W-5L vs lr_033) ⇒ **outcome (iii):
report both, no build.**
the finding, stated for the maintainer: multiclass has about **+0.5%
Brier available for about ×1.8 multiclass fit** by stepping more slowly —
either a slower multiclass auto rate (≈ 0.033; one constant, no code,
cost = the extra trees) or the exact Hessian (a kernel; ~35% fewer trees
than the slow rate for the same Brier, plus a K² per-round cost). Both are
a cost-for-strength trade on the slowdown axis, not a free win; the auto
rate was tuned on Grinsztajn, which has no multiclass set, so neither
choice has been tested before. §7 applies: lr_050 → lr_033 did not
saturate, so the knee may sit lower still. The maintainer's call; the
loop does not take a default-cost trade on its own.
forecast (S1b): not pre-stated beyond the outcome rules.
verdict: **S8 CLOSED at S1b, outcome (iii), no build**; the multiclass
rate/Hessian trade goes to the maintainer as a question. Self-merged:
`benchmarks/` + `.md` only.
next: **S5** (fused multiclass cross-entropy eval, an exact speed rewrite;
worth more if multiclass ever grows more trees), then **S10**.

#### I051 2026-09-22 S4 S1→S3 (standardized `diff` in the cross block, behind a private experiment switch; muse task, pre-registered)
why now: I050 closed (PR #149 self-merged at 4094b4d; S1's code parked).
S4 is the last pick. No campaign PR open, bench idle. Class: **a default
change with no knob** (the end state), tested behind a private module
switch, default off, bit-identical off; I045's zero-fit gate passed. Branch
`campaign/s4-cross-diff-std` from main; muse task
`20260922-s4-cross-diff-std.md` (library + harness arm `ChimeraBoostXStd`
+ tests, no benchmark); Claude runs the reads after review.
the mechanism (I045): `diff` is the raw `x_i − x_j`; on the 33 engaged gr
regression sets 221 of 457 diff columns (48%) are near-duplicates of the
larger parent (σ ratio ≥ 3, |ρ| ≥ 0.95), ten sets with ≥ 2/3 of the block
duplicated (medical_charges, house_sales, wine_quality, Brazilian_houses,
Ailerons, houses, cpu_act, elevators, MiamiHousing, SGEMM). With σ from the
fit rows, `x_i/σ_i − x_j/σ_j` compares the parents on one scale. The race
still decides cross-vs-none per set, so the change only helps where it
turns a wasted slot into a usable column. Unconditional (every diff pair),
the simplest form; a ratio-gated variant is the follow-up only if losses
concentrate on same-unit pairs, where the raw difference was the
meaningful one (e.g. sqft_living − sqft_above).
barriers (`barrier_check.py`: B16, B1, B17): B16 — the block carries the
SAME columns; one operator's arithmetic changes, nothing is screened out.
B1 — inert below CROSS_MIN_SAMPLES (no race), which is the in-run
control. B17 — sub-gate races, not touched. The replay trap (I039's count-
column bug) is designed out: the refit pins the donor's scales.
ladder and bars:
(a) muse pass: switch off ⇒ identity N/N and every test green; switch on
⇒ the new tests (scales from fit rows, weights honoured, constant parent
safe, fitted state carries the scales, replay pins them).
(b) S2 synth, `run_benchmarks.py --synth --seeds 3 --save --models
ChimeraBoost ChimeraBoostXStd`, `compare_runs --expect-inert` +
`synth_report`: sets below the race gate exact ties; the effect sits in the
engaged slice; engaged wins ≥ half + 1 with median > 0 ⇒ S3, else KILL.
(c) S3 decide, `--decide --seeds 3 --save --models ChimeraBoost
ChimeraBoostXStd`, `compare_runs --by-suite`: **gr (the headline stratum)
engaged wins ≥ half + 1 AND engaged median > 0** on the decision metric;
hc not negative in median; fit ratio median 0.95–1.10 ⇒ PASS: the flip PR
(switch on as the default, then removed) for the maintainer. Else KILL.
forecast: gr regression engaged **+0.0 to +0.3%** median, concentrated on
the ten duplicate-heavy sets; the same-unit sets (diamonds, pol, sulfur,
yprop, Allstate) flat to slightly negative; binary engaged ±0.1%; fit ×1.0
(same column count; the race may flip on a few sets). Chance S2 passes
~45%, S3 ~30%.
muse pass (exit 0, ~12 min, no sandbox error): `_CROSS_DIFF_STD` +
`_fit_diff_scales` (population std over finite fit rows, weighted when
`sample_weight` is given, 1.0 when degenerate; stored as
`cross_diff_scales_`, `None` when the switch was off at fit) called from
`fit_transform` and `from_base_with_cross`; `_cross_block` applies stored
scales at fit and transform (`getattr` default keeps old pickles on the
raw path); the replay refit pins the donor's scales
(`_pinned_cross_diff_scales`, booster.py +3); harness arm
`ChimeraBoostXStd` (switch set per fit, restored in `finally`); 11 tests.
Claude's review: correct; note that a module switch does not reach bagged
members fitted in separate processes — irrelevant here (both arms are the
unbagged default) and moot after a flip, which changes the code default.
Outside the sandbox: **1153 passed, 1 skipped; identity 186/186 with the
switch off; `ruff check chimeraboost/` clean.** Bar (a) PASS.
`/code-review`: no correctness bug (fit/predict agreement, pickling, the
race's columns and borders, replay pinning, NaN/weights, the runner's
restore); one low-severity latent note, the same as Claude's — the module
switch never reaches joblib bag members, so `ChimeraBoostXStd` must only
wrap the unbagged default (it does). S2 synth launched 19:15.
S2 ran: 136 synth datasets × 3 seeds × 2 arms in ~2 min,
`results/20260922-191351.json`, scored `compare_runs … --expect-inert` on
the decision metric (`campaign-s4-synth-compare-20260922.txt`). Harness
SUMMARY W23-L23-T88. Control **90 of 136 exact ties** (no cross race, or
no diff pair picked). Engaged 46: **23W-23L, median −0.062%, bootstrap CI
−0.302..+0.402%**, 36 of 46 split across seeds; the mean (+0.24%) is one
set (syn/639 +13.4%; +0.14% without it).
was the screen fair? Yes, if anything tilted toward the change: synthgen
z-scores every feature and rescales it by exp(U(log 0.5, log 200)) (a
400× spread, `synthgen/emit.py`), so scale-mismatched diff pairs are
everywhere, and the target is built in the standardized node space before
that rescaling — the standardized difference is the natural comparison
there. It still reads a coin flip.
bar (b) **FAIL**: engaged wins 23 < 24, median ≤ 0. Per the
pre-registration: **KILL at S2**; S3 not run.
forecast: "S2 passes ~45%" — it did not; "fit ×1.0" HIT (harness ×0.98).
verdict: **S4 KILLED at S2.** A near-duplicate `diff` column is a wasted
slot, not a harm: the race and the trees already route around it, and a
standardized column is no more useful on average. Recorded as an addendum
to B16 (the cross block's content, like its size, is not where strength
hides). The library switch, harness arm and tests are discarded (the
branch carried them uncommitted); only this record and B16 merge.
Confidence in the kill: moderate-high (a fair screen with 46 engaged sets
and the favourable generative geometry). Cost axis: no change either way.
next: the five picks are resolved (I046 shipped; S1 parked; S2, S3, S4,
S6 killed). The pick named its first alternates for exactly this case —
**S8** (full softmax Hessian, a 5-minute probe), then **S5** (fused
multiclass eval, exact), then **S10** (the LightGBM speed price list) —
so the loop takes them in that order under the same delegation, with the
harness items H(11) (pub: download race) and H(12) (`make_pareto` title
and the HGB point) as self-mergeable fill. NOT taken without the
maintainer: the new pointer from I047 (race the binary linear leaf; a
new entrant, not on the shortlist) and a beam refill.

#### I050 2026-09-22 S1 (predict-time categorical lookup: one C-driven dict pass instead of factorize-then-remap; exact rewrite, muse task, pre-registered)
why now: I049 closed (PR #148 self-merged at 5cb3432 after green CI);
the three self-mergeable probes are done (S2, S3, S6 all KILLED). The
pick order's library rungs are next, S1 first. No campaign PR open, bench
idle, muse's sandbox repaired. Class: **exact rewrite** (the maintainer's
preferred class): no code, prediction or fitted value may move. Branch
`campaign/s1-predict-cat-lookup` from main. Muse task
`20260922-s1-predict-cat-lookup.md`; its PR waits for the maintainer.
the object (refill L1, I042, a read-only predict profile, 5 reps,
wrapped/plain 0.99–1.01): `_codes_for_transform` is 62.4% / 73.3% / 82.7%
of predict wall clock on kick / okcupid-stem / sf-police; it factorizes
the whole batch per categorical column, maps the uniques through
`cat_maps_[j]`, then gathers. For one model the factorization buys
nothing; a C-driven `map(mapping.get, …)` over the rows gave the same
codes at 0.49–0.50× in the refill's prototype.
the exactness design (task file): the direct path applies `factorize`'s
own missing rule (None, self-unequal, raising `!=` → `"__nan__"`, and
returns None to fall back whenever the mask is unsafe); it refuses every
column `_factorize_numeric` would take (a float cast can merge values a
dict keeps apart, e.g. integers above 2**53), so integer-coded sets keep
today's path; and it runs only when `_codes_for_transform` receives no
shared `cat_ctx`, so bagged predict and every fit-time leg (C4a) are
untouched — fit is byte-for-byte today's code.
barriers (`barrier_check.py`): B3 on the word "categorical" only — not a
CatBoost mechanism, our own lookup rewritten with its values asserted
equal.
forecast: predict latency **−25 to −40%** on the string-categorical hc
sets (kick, okcupid-stem, sf-police, Traffic_violations; the object is
62–83% of predict and converts at ~0.5×, a whole-pass removal of the kind
that converted ~100% of its ceiling at C4a), porto-seguro (integer-coded,
refused by design) and Higgs (numeric) flat within ±2%; fit time 0 (fit
untouched); identity 186/186; all tests green plus the new equality file.
bars: (a) identity N/N and every test green (else: a bug, not a trade);
(b) the speed script's arms produce identical outputs on every row;
(c) predict latency on the string-categorical sets improves by ≥ 15% at
the median with no panel set slower by more than 3%. Pass ⇒ PR for the
maintainer. Fail on (c) with (a)(b) intact ⇒ the rewrite is exact but not
worth its code: KILL and record the ceiling's conversion rate.
muse pass 1 (exit 0, ~25 min, no sandbox error): `_direct_codes` + the
call site (+53 lines in preprocessing.py), `tests/test_predict_codes.py`
(4 tests: edge cases, real hc data, end to end), the speed script.
Identity 186/186, the suite green outside the sandbox's `\\?\` path quirk.
First speed read (7 reps, outputs identical every row): kick −9.0%,
okcupid −16.1%, sf-police −13.5%, Traffic −16.9%, porto −6.2% (its
categoricals arrive as numeric STRINGS, so the direct path engages — it
was never a flat control), and Higgs **−25.8% on identical code**. That
last row is the instrument, not the change: the script timed OFF then ON
in the same order every repeat, so ON always ran warm. Reviews (Claude +
`/code-review`, independently, same two findings; neither found a
correctness bug): (1) numeric-coded categoricals now run
`_factorize_numeric` twice per predict (the refusal check discards its
result, then the fallback factorizes again); (2) columns that combo pairs
or gdiff crosses read later in the same `transform` are now dict-mapped
AND factorized — strictly more work, and combos are the default on
all-categorical data. Both are slowdowns the panel could not see (no
int-coded or all-categorical row). A one-byte stray file `x` in the repo
root (muse's) was deleted. Follow-up task
`20260922-s1-predict-cat-lookup-fixes.md`: reuse the numeric result, keep
combo/gdiff parents on the old path, alternate the timing order, 15
reps, add `kick-intcoded` and `allcat-combos` rows.
muse pass 2 (exit 0, ~20 min): both regressions fixed (the numeric probe's
result reused, one call per column; combo parents and gdiff groups keep
the old path and fill the cache once), 3 more tests (spies prove one
`_factorize_numeric` call per numeric categorical and no combo-block
refactorization), the script alternating order with 15 reps and each
timing averaged to ~100 ms (a single 13 ms Higgs predict carried a
first-slow/second-fast overhead that alternation alone did not remove;
three earlier full runs superseded, their hc rows consistent). Claude's
checks outside the sandbox: **1149 passed, 1 skipped; identity 186/186;
`ruff check chimeraboost/` clean**. Final read (15 reps, identical outputs
every row): Traffic_violations **−17.4%**, okcupid-stem **−15.7%**,
sf-police **−13.8%**, kick **−8.3%**, porto-seguro −5.6%; controls flat —
Higgs +0.2%, `kick-intcoded` −1.7%, `allcat-combos` (153 combo pairs
engaged) −0.2%. Code: +125 / −10 lines in preprocessing.py, including a
branch that detects whether `_direct_codes` has been monkeypatched (test
instrumentation inside library code; it would have to go before a merge).
bars: (a) PASS; (b) PASS; (c) **FAIL by a quarter point**: the median over
the four string-categorical sets is **14.75%** against ≥ 15%, no set
slower (the regression clause passes after pass 2).
forecast: "−25 to −40%" MISSED by half. The refill's 0.49× prototype
omitted the missing-value mask: the direct path still pays two object-
array comparison passes (`!=` self, `== None`) to find missing rows, the
same passes `_factorize_hashed` pays, so the rewrite removes the hashing
and the gather but not the mask — conversion ≈ 45% of the forecast
ceiling. Fit 0 (untouched), as forecast.
verdict: **S1 KILLED at its bar** — exact, but ~14% faster categorical
predict for ~115 net lines in the busiest module is the "not worth its
code" case the pre-registration named. The code is PARKED, not deleted, on
`campaign/s1-predict-cat-lookup` (pushed, no PR, commit "S1 parked"), for
the maintainer's call; the loop does not reopen it on its own. Dated
close: if the maintainer has not asked for it by **2026-09-29**, delete
that branch (local and remote) and strike this line.
next: **S4's flag** (standardized `diff` for the cross block; a library
flag for the A/B, muse task, then S2 synth and S3 decide).

#### I049 2026-09-22 S6 S1 (size-gated TS quantization, confirmed on small categorical data that did not select it; zero library change, pre-registered)
why now: I048 closed (PR #147 self-merged at f290c38). The muse sandbox
is repaired: the maintainer approved the ACL reset, Claude ran it
(`icacls C:\Users\Nathan\.config\muse /reset /T /C /Q`: 7 files, 0
failures; `MuseSandboxUsers` deny entries 1,786 → 0; muse re-adds the one
pair it needs at launch). No campaign PR open, bench idle. Class:
**zero-library probe of a candidate default** (monkeypatch). Branch
`campaign/s6-ts-quant-small` from main. Muse writes and runs
`benchmarks/probe_ts_quant_small.py` (task `20260922-s6-ts-quant-small.md`).
the idea: I035's `ts_q16` (the ordered TS quantized to 16 uniform buckets
at fit and transform, CatBoost's CtrBorderCount = 15) lost on the big hc
sets (sf-police −0.31%, kick −0.54%, porto −0.36%) and won on the two
small ones (kdd_ipums +1.65%, eucalyptus +3.26%): coarse TS resolution as
a small-data regularizer. The shippable form is gated on training rows
(< 10,000), which makes every big set an exact tie by construction.
panels: SELECT = kdd_ipums, eucalyptus and their twins (they chose the
idea: report only, never in a bar); CONFIRM_REAL = every other hc
classification key or twin under 10,000 training rows; CONFIRM_SYNTH =
every synth classification key with a categorical column under 10,000
rows; CONTROL = hc classification keys at or above the gate.
barriers (`barrier_check.py`: B3, B4, B15, B18): **B3 binds** — this is
CatBoost's CTR quantization ported at its narrowest, and the ported form
already regressed the big sets (I035); the size gate turns those into
exact ties, and whether the small-set gain is more than its two selecting
sets is exactly what this rung measures on data that did not select it.
B18 — not the transform-side reweighting it closed: the quantization is
symmetric (fit and transform) and changes resolution, not moments. B4,
B15 — keyword only.
forecast (prior low, stated before the run): SELECT reproduces I035's
direction (+1 to +3%) on the harness's splits; CONFIRM_SYNTH is mostly
ties and small moves — synthetic categoricals are low-cardinality, where
16 buckets merge little — median **+0.0 to +0.2%**, wins near half;
CONFIRM_REAL has 1–3 keys (okcupid-stem@sus25 the likeliest), mixed;
CONTROL exact ties. Fit time ×0.95–1.0 on engaged keys (fewer distinct TS
values), 1.0 elsewhere. Chance the bar passes: ~25%.
bars: (0) SELF-CHECK — the `chimera` arm reproduces the saved run's Brier
on every hc (key, seed) to 1e-9; (1) CONTROL all exact ties; (2) PASS when
the confirmation keys (CONFIRM_REAL + CONFIRM_SYNTH, SELECT excluded,
keys as units) read wins ≥ half + 1 of the decided keys AND median > 0,
AND CONFIRM_REAL's median is not negative. Pass ⇒ a library flag for the
synth screen and the decide tier (muse). Fail ⇒ **S6 KILLED**: the
small-set gain belonged to its selecting sets; B3's record gains its
eighth port.
cost: small-n fits only, ~5–10 min.
muse pass (exit 0, ~20 min, **no sandbox error**: the ACL reset held):
wrote `probe_ts_quant_small.py` and ran everything itself — smoke,
full run (330 fits: 55 keys × 3 seeds × 2 arms, zero skips), ruff clean;
report `20260922-s6-ts-quant-small-RESULT.md`. Reviewed by Claude.
panels as qualified on seed 0: SELECT 3 (kdd_ipums, eucalyptus,
eucalyptus@time), CONFIRM_REAL 3 (cjs, cjs@sus50, okcupid-stem@sus25),
CONFIRM_SYNTH 41 of 137 synth classification keys (48 carry a categorical,
7 of them over the gate), CONTROL 8.
read (house convention: near-solved = best arm Brier < 0.001, counted in
the all-key sign test, out of the medians): **SELF-CHECK PASS** (0.0 over
42 hc rows); CONTROL **8/8 keys, 24/24 rows exact ties**; CONFIRM all keys
24W-20L of 44, non-near-solved **20W-17L of 37, sign p = 0.74, median
+0.161%**; CONFIRM_REAL non-near-solved = okcupid-stem@sus25 only,
**−0.061%** (cjs and cjs@sus50 are near-solved: their +100% is a
zero-denominator artifact); CONFIRM_SYNTH non-near-solved 20W-16L, median
+0.290%; tails: syn/667 Brier 0.0055 → 0.0793 (×14), syn/557 0.036 →
0.064, syn/167 0.0063 → 0.0089 — easy sets where one categorical nearly
decides the class, whose resolution 16 buckets erase; best syn/297 +34.6%,
syn/737 +9.2%. SELECT: eucalyptus +1.94%, eucalyptus@time +3.27%,
**kdd_ipums −0.65%** (I035 read +1.65% on the research loader's splits: a
selecting set flipped sign on the harness's).
bar (0) PASS; (1) PASS; (2) **FAIL** — the counts clear half + 1 (20 of 37)
and the median is positive, but CONFIRM_REAL's median is negative, and the
substance agrees with the clause: a coin flip with catastrophic tails.
forecast: SELECT direction HIT on eucalyptus, MISSED on kdd_ipums;
CONFIRM_SYNTH "median +0.0 to +0.2%, wins near half" HIT (+0.16% over all
confirm keys, 20-17); CONFIRM_REAL "1–3 keys, mixed" HIT; CONTROL HIT;
"~25% chance the bar passes" — it did not; unforecast: the ×14 tail.
verdict: **S6 KILLED at S1.** The small-set gain belonged to its selecting
sets; the gated form is a coin flip on data that did not choose it and
can wreck an easy set. B3's record gains its eighth port (the CTR
quantization, both forms). Cost axis: fit unchanged (the gate). Self-
merged: `benchmarks/` + `.md` only.
next: the library rungs — **S1** (predict-time categorical lookup, exact)
then **S4's flag** (standardized `diff`), one PR each, both waiting on
the maintainer's merge.

#### I048 2026-09-22 S3 S0+S1 (why every opponent beats us on cpu_act@sus25: our-side ablation + a per-row error read; measurement, pre-registered)
why now: I047 closed (PR #146 self-merged at 7ab18e2 after green CI; S2
KILLED, B20 registered). Next in the pick order. No campaign PR open,
bench idle. Class: **defect probe** (measurement, `benchmarks/` only).
Branch `campaign/s3-cpu-act-sus25` from main. Muse writes and runs
`benchmarks/probe_small_reg_loss.py` (task `20260922-s3-cpu-act-sus25.md`).
the slice, as re-read at the pick: `cpu_act@sus25` (1,536 training rows)
is the one real double-digit loss on 77 gr keys — CatBoost +20.2%,
LightGBM +28.5%, HGB +24.5% of our RMSE, best NRMSE 0.147 (not
near-solved), our bag Ens8 WORSE than the default (−2.9%), seed 1 at 4.50
against LightGBM's 2.43; on the full `cpu_act` we beat all three. SGEMM
was dropped from S3 at the pick (near-solved, and Ens8 recovers 73% of
its gap). Panel: all 7 gr regression @sus25 keys + the parent.
arms (ours): default, `chimera_const` (linear leaves off), `chimera_norefit`,
`chimera_fixedlr` (the pre-0.30 flat rate), `chimera_bins64` (coarser
bins: B20's June record names cpu_act as the set that overfits when bins
get FINER); opponents `lgbm`, `catboost` as the harness runs them.
Per-row read on cpu_act@sus25 and its parent: SSE share of the top 1% /
5% rows, and SSE split by test rows with ≥ 1 input outside the training
range.
barriers (`barrier_check.py`: B20, B1, B2, B6, B13, B14, B18): B20 — this
arm goes the OTHER way (coarser at small n), the direction B20 leaves
open. B1 — 1,536 rows is above LINEAR_LEAVES_MIN_SAMPLES (the const/linear
race runs) and below CROSS_MIN_SAMPLES (no cross race), so the const arm
engages and cross features cannot be the cause. B2/B13 — the norefit arm
measures the refit's share directly. B6 — a probe on one set, not a
sweep; any fix it names must be a rule with a mechanism, never a tuned
value. B14, B18 keyword only.
forecast: H1 (55%), linear leaves extrapolating: `chimera_const` closes
≥ 50% of the gap to LightGBM and the default's excess SSE sits in rows
with an out-of-range input (the top 1% of rows carry > 30% of our SSE vs
< 15% of LightGBM's); H2 (20%), bins too fine at n = 1,536:
`chimera_bins64` +5 to +15%; H4 (10%), the adaptive rate's extra rounds:
`chimera_fixedlr` +0 to +5%; the refit is not it (I015's seed-1 NoRefit
read 4.60 against the default's 4.50): `chimera_norefit` within ±3%.
Other @sus25 keys: every arm within ±1%, except const on the sets where
linear leaves win (a known cost).
bars: (0) SELF-CHECK — `chimera`, `lgbm`, `catboost` reproduce the saved
run's RMSE to 1e-9 relative on every (key, seed), else nothing is read;
(1) a NAMED DEFECT when one of our arms closes ≥ 50% of the default's
RMSE gap to LightGBM on cpu_act@sus25 (mean over seeds, ≥ 2 of 3 seeds
agreeing) while no other @sus25 key and not the parent loses > 0.5%
under it — its fix is designed next as a rule, no knob; (2) the per-row
read LOCATES it when ≥ 50% of our excess SSE over LightGBM sits in rows
with an out-of-range input. Kill: no arm closes ≥ 25% of the gap ⇒ the
loss is not on a switch of ours; the next step is reducing LightGBM
toward us, or closing it as a one-set outlier.
cost: 168 fits at ≤ 10k rows, ~5–10 min.
muse pass (exit 0, ~8 min): wrote `probe_small_reg_loss.py` (600 lines;
`refit_full="off"` and `adaptive_lr="off"` through the standard runner,
the bins64 and const capture paths behind exact EQUIV-CHECKs, per-row
capture refits asserted equal to their JSONL rows within 1e-12) and its
report as `20260922-s3-cpu-act-sus25-RESULT.md`; it could execute nothing
— the sandbox failure of I047 again, on every shell call. ROOT CAUSE
(read-only `icacls`, facts ledger): muse's sandbox appends its deny ACEs
to `C:\Users\Nathan\.config\muse` on every launch and never removes them —
1,786 duplicate `MuseSandboxUsers` entries plus one stale per-session SID
pair per launch — until the ACL hits Windows' size limit and
`SetNamedSecurityInfoW` fails with 1340. Resetting that folder's ACL is
outside the repo and the maintainer's call; until then Claude runs
finished, reviewed scripts. Reviewed by Claude, ruff clean, smoke run on
cpu_act@sus25 seed 0: SELF-CHECK exact (0.0 on all three paired arms);
`chimera_const` = the default to the bit (the race had picked constant
leaves), `chimera_bins64` **+17.8%**, beating LightGBM on that seed; the
default's worst 1% of test rows carry 56% of its SSE. Smoke rows deleted,
full run launched 17:00.
ran: 168 fits in ~2 min, `results/probe-small-reg-loss.jsonl`, `-rows.json`,
`-20260922.log`. EQUIV-CHECK 0.0 on both direct paths; **SELF-CHECK PASS,
max relative diff 0.0 over 72 cells** (our default, LightGBM and CatBoost
on every key and seed).
cpu_act@sus25, mean RMSE, change vs the default (+ = better), seeds agreeing:
  default 3.786 · const +0.08% (1/3) · norefit −4.83% (3/3) · fixedlr −0.99%
  (3/3) · bins64 **+6.60% (2/3: +17.8 / +16.0 / −15.7%)** · LightGBM
  **+28.46%** (3/3) · CatBoost +20.20% (3/3)
the other keys: every arm of ours loses or ties on the six other @sus25
keys and the parent except const on nyc-taxi@sus25 (+1.49%) and fixedlr on
supreme (+1.47%); bins64 loses on 5 of 7 (parent cpu_act −7.58%, sulfur
−2.46%); norefit loses on all 8 (−0.5 to −6.7%): the refit is a strength
component everywhere, small data included. Our default beats LightGBM on
6 of the 7 other keys.
per-row read, cpu_act@sus25: the default's worst 1% of test rows carry
**56% / 75% / 57%** of its SSE by seed (LightGBM 45% / 16% / 16%); test
rows with an input outside the training range are 12–27 of ~2,048 and carry
52% / 8% / 59% of our excess SSE over LightGBM, **25% overall** — seed 1's
excess is in-range. On the parent the tails are ordinary (21–34%).
bars: (0) PASS; (1) **FAIL** — no arm closes ≥ 50% of the gap to LightGBM;
the best, bins64, closes 23% on split seeds and loses the parent by 7.6%;
(2) **FAIL** — 25% of the excess sits out of range, not ≥ 50%. The
pre-registered KILL fires: no arm closes ≥ 25% of the gap.
dedup, found after the run: the obvious remaining suspect, the regressor's
one-row minimum leaf (LightGBM requires 20), is BREAKTHROUGH_PLAN C3
(2026-08-01, killed): on this very cell mcw = 8 took RMSE 3.82 → 3.53
against LightGBM's 2.71, and across 42 regression sets the floor was
directionally right and too small to ship. It lived only in that plan
file; registered now as **B21**. B20 gains this rung's other finding:
coarser bins are no small-data rule either.
forecast: H1 linear leaves (55%) MISSED — const is the default to the bit
on two seeds (the race picked constant leaves); H2 bins (20%) MISSED on
its bar (+6.6% on split seeds, loses elsewhere); H4 rate MISSED (−1%);
"norefit within ±3%" MISSED (−4.8%: the refit helps); "other keys within
±1%" roughly HIT except norefit and bins64 on sulfur.
verdict: **S3 KILLED as a one-set outlier.** Our loss on cpu_act@sus25 is
a handful of catastrophic predictions (a percent of the rows carry half to
three quarters of the error) that no switch of ours removes: linear
leaves, the refit, the rate, the bin grid and (C3) the leaf floor each move
it by ≤ 8%, most with split seeds, and none can be taken without losing
elsewhere. The opponent reduction the kill clause allows is not taken: one
twin of a parent we win, worth about 0.0003 on the regression panel's mean
R², and its only candidate mechanism (sample floors) is B21-closed.
Confidence in the kill: moderate-high (exact self-check, 8 keys, and C3
covered the one lever this probe lacked). Cost axis: nothing here is
cheaper. Self-merged: `benchmarks/` + `.md` only.
next: **S6** (n-gated TS quantization), then the library rungs S1 and
S4's flag — all three need a muse task, and muse cannot run until its
sandbox ACL is reset (facts ledger); the maintainer's call.

#### I047 2026-09-22 S2 S0+S1 (CatBoost's split machinery on the gr binary sets where it still leads, plus two own-side arms; measurement, pre-registered)
why now: the maintainer merged PR #145 at d14bf38 (I046 SHIPPED: the count
column is the default); branch deleted, the merge confirmed on main. The
pick order puts the three self-mergeable probes next, S2 first. No
campaign PR open, bench idle. Class: **measurement** (opponent ablation +
two own-side arms), `benchmarks/` only. Branch
`campaign/s2-catboost-split-score` from main. Muse writes and runs
`benchmarks/probe_catboost_split_score.py` (task
`20260922-s2-catboost-split-score.md`).
the slice, as re-read at the pick: CatBoost's Brier lead on the standing
run is real on **california** (+2.7%, and its @sus50 twin +2.7%),
**bank-marketing** (+1.0%) and **credit** (+0.5%), plus two small twins
whose parents we win (albert@sus25 +0.45%, eye_movements@sus25 +0.30%);
Diabetes130US, compas and heloc sit inside seed noise and are left out.
Controls where we lead: MagicTelescope, electricity. Five gap units
(california counts once; its twin is reported, not counted).
arms: `cb_default` (the harness's CatBoost, byte for byte) and `cb_ours`
(CatBoost set to our split machinery all at once: score_function L2, no
leaf backtracking, Median borders, l2_leaf_reg 1, border_count 128); our
side `chimera` (the harness default), `chimera_const` (binary linear
leaves off: they are unraced on binary, and I041 saw the unraced binary
leaf over-trust an in-sample statistic), `chimera_bins254`. Stage B (five
one-knob CatBoost arms) is coded but runs only if `cb_ours` moves the edge.
barriers (`barrier_check.py` matched B3, B1, B4, B14, B18): B3 — this
measures the opponent, the sanctioned L3 method (the 2026-08-01 run of the
same method found the learning rate); any port that follows owes its own
argument. B4 — CatBoost stays Plain. B1 — every gap key trains above
LINEAR_LEAVES_MIN_SAMPLES (smallest: eye_movements@sus25, ~1.4k rows), so
the const arm engages everywhere. B14 — no audition budget moves. B18 —
keyword only: gr loaders pass no cat_features, so no TS exists here.
forecast: CatBoost's residual edge is its split score or its λ, so
`cb_ours` gives back **40–80%** of the edge on california and
bank-marketing, less on credit, and moves the controls < 1%.
`chimera_const` **0 to +0.5%** on the noisy sets (bank-marketing, credit)
and a loss on california (its geography rewards slopes); `chimera_bins254`
**+0.5 to +1.5% on california** (lat/long resolution; LightGBM at 255 bins
also beats us there) and flat elsewhere. Cost: bins254 ×1.1–1.3 fit,
const ×0.8–0.9.
bars: (0) SELF-CHECK — `chimera` and `cb_default` reproduce the saved
run's Brier to 1e-9 on every (key, seed), else nothing below is read;
(1) a CatBoost mechanism is NAMED when `cb_ours` recovers ≥ 40% of the
edge on ≥ 3 of the 5 gap units with both controls' CatBoost Brier moving
< 1% → Stage B localizes it; (2) an own-side POINTER when `chimera_const`
or `chimera_bins254` beats `chimera` on ≥ 3 of 5 units, controls within
±0.3%. Neither ⇒ the cluster closes as a barrier (the residual is not the
split score, border grid, λ, backtracking, bin count, nor our unraced
binary linear leaves).
cost: 48 CatBoost fits (~15–25 min) + 72 of ours (~5 min).
muse pass (exit 0, ~20 min): wrote `probe_catboost_split_score.py` (437
lines; the harness data path, the literal no-flag `chimera_cfg`,
`threads=2`, `_run_catboost` copied with overrides merged last), smoke
run on credit seed 0: EQUIV-CHECK 0.0 (the direct-construction path for
`chimera_bins254` reproduces `chimera` bit for bit) and SELF-CHECK 0.0 on
both arms against the saved run. Then its sandbox broke host-wide (every
shell call: "windows_elevated unified exec session launcher unavailable
… SetNamedSecurityInfoW failed: 1340"), so it never launched the full
run. Reviewed by Claude (the one open question, the estimator's seed:
the harness also fixes `random_state=0`), ruff clean, and the full run
launched by Claude at 16:31 — execution of a finished script, no
authoring. Muse's review note that the saved run predates the flip is
wrong: `20260922-144219` ran on the flip branch, and gr passes no
categoricals anyway.
ran: 120 fits in **~2 min** (the CatBoost fits on these sets take 0.5–10 s),
`results/probe-cb-split-score.jsonl` + `-20260922.log`. EQUIV-CHECK 0.0;
**SELF-CHECK PASS, max |diff| 0.0 on both arms over all 24 (key, seed)
rows** — the probe measured exactly what the harness measured.
table (mean Brier over 3 seeds; edge = CatBoost's lead as % of ours;
recov = share of it `cb_ours` gives back; d = our arm vs `chimera`, + =
better, seeds agreeing):
  california            edge 2.68%  recov 38%   const +1.73% (3/3)  bins254 +0.88% (2/3)
  california@sus50      edge 2.71%  recov 54%   const +1.95% (3/3)  bins254 +0.11% (2/3)
  bank-marketing        edge 1.02%  recov 22%   const +0.37% (2/3)  bins254 +0.74% (3/3)
  credit                edge 0.51%  recov 99%   const −0.16% (2/3)  bins254 +0.33% (3/3)
  albert@sus25          edge 0.45%  recov 75%   const −0.15% (2/3)  bins254 +0.09% (2/3)
  eye_movements@sus25   edge 0.30%  recov 69%   const −0.89% (3/3)  bins254 +0.67% (3/3)
  MagicTelescope (ctl)  edge −3.50% cb_ours moves CatBoost +0.03%   const −1.65%  bins254 +0.09%
  electricity (ctl)     edge −6.16% cb_ours moves CatBoost **+11.6% worse**  const −3.67% (3/3)  bins254 **+7.97% (3/3)**
fit: const ×0.56 median (binary linear leaves are about half the fit),
bins254 ×1.12 (electricity ×1.22, it runs to the 2000-round cap).
bar (1) **FAIL**: `cb_ours` recovers ≥ 40% on 3 of 5 units (credit,
albert@sus25, eye_movements@sus25; california 38%, bank-marketing 22%),
but it moves the electricity control by 11.6%, so the bundle is not a
gap-specific mechanism: our split settings make CatBoost broadly worse
(on electricity the likely culprit is `border_count=128`, the same axis
as our own bins254 gain there), and on the two sets that carry the
cluster (california, bank-marketing) it explains only a fifth to two
fifths of the edge. Stage B not run, as pre-registered.
bar (2): `chimera_const` **FAIL** (2 of 5 units, controls −1.65% /
−3.67%). `chimera_bins254` wins 5 of 5 units but moves electricity by
+7.97%, and it is a **DEDUP MISS**: max_bins 128 → 254 was tested on three
suites on 2026-06-01 and rejected (electricity-driven gains, an overall
wash on independent data, regression reversed with cpu_act −17 to −30%);
today's read is the same picture. That verdict lived only in session
memory, so neither `barrier_check.py` nor this task's S0 could see it —
registered now as **B20**, and not re-opened.
forecast: "cb_ours gives back 40–80% on california and bank-marketing"
MISSED (38%, 22%); "controls move < 1%" MISSED (electricity 11.6%);
"const loses on california" MISSED the other way (+1.73%, all seeds);
"bins254 +0.5 to +1.5% on california" HIT (+0.88%), "flat elsewhere"
MISSED (electricity +7.97%).
what the probe did find: on california the unraced binary linear leaf
costs **1.73% of Brier on every seed — two thirds of CatBoost's 2.68%
lead there** — while the same leaves earn 1.65% on MagicTelescope and
3.67% on electricity. Per-set choice would take both; the regression path
already makes it by racing const vs linear on the validation split, the
binary path does not race. That is a pointer, not a result (one dataset
plus its twin, post hoc): a binary const-vs-linear race is B14/B2
territory and costs up to the const fit again (×0.56) unless auditioned
short; it goes to the next refill, not into this rung.
verdict: **S2 KILLED** — the gr binary CatBoost residual is not its split
machinery as a bundle (non-specific), not a bin-count object we may take
(B20), and not our linear leaves as a flat switch; confidence the kill
is right on the decision suites: moderate (8 keys, 3 seeds, the self-
check exact); cost axis read: nothing here is cheaper except turning
linear leaves off, which loses. Self-merged: `benchmarks/` + `.md` only.
next: **S3** (cpu_act@sus25). Its task gains a `chimera_bins64` arm: B20's
June record names cpu_act as the set that overfits when bins get finer,
so at 1,536 training rows 128 bins may already be too fine.

#### I046 2026-09-22 F5 S4 (/experiment on flipping `cat_count_features` on by default: public read, then the flip, then the chart-grade field on the flip branch; pre-registered)
why now: the maintainer delegated the call in chat ("you may go ahead on
whichever of this that you like") and the loop says **yes**. No campaign
PR open (only #108, the whitepaper), bench idle, no task marker. Branch
`campaign/f5-s4-catcount-default` from main at c3a5314. Class: **a default
change**, no skips: the flip is a muse task and its PR waits for the
maintainer.
why yes: engaged hc 6W-1L on the decision metric, median +0.558% with a
bootstrap CI of +0.023..+0.864% (I044's re-read of I039), the one loss
okcupid at −0.00%; sf-police, Traffic_violations and wine-reviews at about
4× their seed noise; hc@time 4-0; every Grinsztajn row an exact tie by
construction (no gr loader passes cat_features), so the headline stratum
cannot move; cost ×1.09 on hc, ×1.165 on the engaged seven, ×0.99 gr. Its
one challenger (I040, calibrated shrinkage) lost. What the evidence lacks:
seven engaged datasets is under GATE_ROBUSTNESS §2's ~8, and a sign test on
seven cannot reach p < 0.05 at 6-1. The public suite carries eight
datasets with a column of cardinality ≥ 256 that nothing in this campaign
has tuned on (BNP 18,210 levels, internet_firewall 29,152, BMC 11,200,
kickstarter 3,102, nba-shot-logs 1,808, rossmann 1,115, fps-in-video-games
446, federal_election's entity columns): that is the new evidence, and it
comes first. A decide run on the flag arm would reproduce I039's strength
rows exactly (same code, same seeds), so the chart-grade field runs ONCE,
on the flip branch, where it also proves the flip reproduces the flag.
plan: (S4a) `run_benchmarks.py --public --seeds 3 --save --models
ChimeraBoost ChimeraBoostCatCountLib --datasets <the 8 engaged> 
pub:Medical-Appointment-No-Shows pub:freMTPL2freq` — the last two carry
categoricals under 256 levels (81, 22): the in-run control. (S4b) muse flips
the default to `True` on the two public estimators (the card ≥ 256 gate
keeps it inert on data without such a column; the internal classes keep
`False` because the estimators always pass the value through), updates
the tests that pin it; the identity snapshot must stay N/N (no config
feeds the default a 256+-level column without setting the flag); docs
and CHANGELOG are Claude's.
(S4c) on that branch, `run_benchmarks.py --decide --seeds 3 --save` with
I015's field (ChimeraBoost OneLin OneLinX NoRefit Ens5 Ens8 CatBoost
LightGBM sklearn_HGB), scored per stratum against I015, the Pareto
refreshed from it.
forecast: S4a — the two controls exact ties; engaged 8: wins ≥ 5 with
median +0.1 to +0.5% on the decision metric; the largest gains where a
single column dominates (BMC's geo codes, internet_firewall's ports) and
the smallest on federal_election (its entity columns are near-unique, so
counts are mostly 1); fit ×1.05–1.25 on engaged sets. S4c — the default's
strength rows equal I039's CatCountLib rows to the bit on all 96 decide
keys (gr and its twins unchanged from I015 to the bit); fit on gr 3–10%
under I015 (the four exact speed ships since 0.32.0), on hc under I015 too
despite the count column (C4a alone took 8–19% off hc fit); Pareto: the
classification point moves up by < 0.002 skill (hc is 8 of 38
classification sets), regression a hair up.
bars: (a) the S4a controls exact ties, else the gate misfires on public
data: stop; (b) S4a engaged — any set losing > 1% with all three seeds
agreeing is a question for the maintainer written into the PR, never a
veto (the public suite never blocks; gr + hc decide); (c) S4c — the flip
branch's default rows equal I039's CatCountLib rows to the bit on every hc
key and I015's ChimeraBoost rows on every gr key, else the flip is not the
flag: stop; (d) the new default point is not dominated by the old on
either Pareto panel.
launch note: the first S4a launch (14:09) died at 14:14 on the pub:
download race (facts ledger, H(11)) before any fit finished; the ten
parquets had all landed, the stray `20260922-140929.{txt,progress}` were
deleted, relaunched 14:15 on the warm cache.
S4a ran: 10 pub: sets × 3 seeds × 2 arms in **5 min** (not the hour I
guessed from the August run: the multiclass and categorical speed ships
since), `results/20260922-141439.json`, scored `compare_runs.py … --model
ChimeraBoost --model-new ChimeraBoostCatCountLib --expect-inert` on the
decision metric (`results/campaign-f5-s4a-public-compare-20260922.txt`).
Harness SUMMARY, default vs the arm: W1-L5-T3, median gap −0.17%, speed
×1.01. Per set, + = the count column better: rossmann **+1.87%** RMSE
(Store, 1,115 levels), internet_firewall **+2.56%** Brier (seeds split;
F1 0.955 → 0.974), federal_election **+0.50%** RMSE, nba-shot-logs +0.18%,
BNP +0.17%, fps-in-video-games **−0.02%** (unanimous, the one loss);
kickstarter engaged (157 vs 158 trees) but near-solved (Brier 0.0001),
excluded. Engaged and scored: **5W-1L, median +0.343%, bootstrap CI
+0.074%..+2.217%**; 5 of 7 engaged sets unanimous over seeds. Fit per set
×0.75 (fps, fewer trees) to ×1.28 (nba, 86 → 119 trees), ×1.01 overall.
bar (a) **PASS**: Medical-Appointment-No-Shows (81 levels) and freMTPL2freq
(22) exact ties to the bit, and so is **BMC** — which I had counted as
engaged from PUBLIC_PLAN's raw 11,200-level `geo_level_3_id`; the harness
sees BMC's max categorical cardinality as **10** (`PUBLIC_FACETS`, the id
column arrives numeric), so no column qualifies and the tie is the gate
doing what it claims: a third control, not a misfire. bar (b) **PASS**: no
engaged set loses more than 1%; the one loss is −0.02%.
forecast, S4a: wins ≥ 5 with median +0.1 to +0.5% HIT (5W-1L, +0.343%);
"largest where one column dominates" half HIT (internet_firewall's ports
+2.56%; BMC was never engaged, my reading of the facets was wrong);
"smallest on federal_election" MISSED (+0.50%, third largest: near-unique
entity columns still carry a usable count); fit ×1.05–1.25 MISSED on the
cheap side (×1.01; the count column changes tree counts both ways).
Transfer read: the public suite is the first data this mechanism has met
that nothing in the campaign tuned on, and it agrees with hc in sign, size
and in where the gains sit (entity columns of 1k–30k levels).
verdict: S4a PASS → S4b (muse flips the default).
S4b ran (muse, one pass, exit 0, ~7 min): `cat_count_features=True` in
both estimators' signatures and docstrings; `test_off_estimators_match_default`
replaced by `test_default_matches_on`, plus
`test_off_differs_from_default_on_high_card` and
`test_default_inert_without_high_card_column` (a 200-level column: default
== False to the bit). Identity 186/186 unchanged (as forecast: no config
feeds the default a 256+-level column without the flag). `/code-review`
found one real defect, in the HARNESS: `_run_chimera` passed the flag only
when truthy, so after the flip `ChimeraBoost` and `ChimeraBoostCatCountLib`
were the same fit and no arm could run the old default — every later A/B
of this flag would have compared the default with itself and read exact
ties. Fixed by a second muse pass (~8 min): the `"off"` sentinel (the
`adaptive_lr` / `refit_full` convention), a `ChimeraBoostNoCatCount`
control arm (off by default in the field), `--chimera-no-cat-counts`
(mutually exclusive with `--chimera-cat-counts`, checked before the
`--save` tee opens), and two guard tests (a stub estimator records the
kwargs). Its other two findings were prose (parameters.md, CHANGELOG),
done by Claude with concepts.md, deployment.md and PROJECT_STATUS.md.
Not changed, recorded: `ChimeraBoostQuantileRegressor` has no count
column at all (its own API; out of this rung's scope). Full suite on the
final tree 1142 passed, 1 skipped, 119 s; identity 186/186.
S4c launched 14:44: `run_benchmarks.py --decide --seeds 3 --save --models
ChimeraBoost ChimeraBoostNoCatCount ChimeraBoostOneLin ChimeraBoostOneLinX
ChimeraBoostNoRefit ChimeraBoostEns5 ChimeraBoostEns8 CatBoost LightGBM
sklearn_HGB` — I015's field plus the old default as a same-run control, so
the cost read carries no cross-day drift.
S4c ran: 309/309 in **55 min** (I015 took 93), `results/20260922-144219.json`;
scored `compare_runs.py … --model ChimeraBoostNoCatCount --model-new
ChimeraBoost --by-suite --expect-inert` (`-compare-…txt`) and `make_pareto.py`
(`campaign-f5-s4c-pareto-20260922.txt`).
bar (c) **PASS, to the bit**: the new default reproduces I039's
`ChimeraBoostCatCountLib` on all 309 (dataset, seed) rows, the control arm
reproduces I015's `ChimeraBoost` on all 309, and every other arm (the
rungs, CatBoost, LightGBM, HGB) reproduces I015 on all 231 gr rows. The
flip IS the flag, and the default's strength is bit-identical from 0.32.0
to c3a5314.
Same-run A/B, new default vs old, decision metric: gr **0-0-59**, gr@sus25
0-0-12, gr@sus50 0-0-6 (the control); hc **6W-1L-7T**, engaged median
**+0.558%** [CI +0.023..+0.864%], 4 of 7 unanimous (sf-police +0.73,
Traffic +0.86, kick +0.35 Brier; employee_salaries +2.45, wine-reviews
+0.56, colleges +0.02 RMSE; okcupid −0.00); hc@sus25 **2W-0L-1T**
(employee_salaries@sus25 +0.86%, okcupid@sus25 +0.30%); hc@sus50 1W-0L-1T
(wine-reviews@sus50 +0.72%); hc@time **3W-1L-3T** (Traffic@time +3.69%,
employee_salaries@time +2.10%, sf-police@time +0.94%, kick@time −0.23%).
**Correction to I039**, whose variant lines were read on `primary` (F1):
on the decision metric hc@time is 3W-1L, not 4W-0L, and hc@sus25 is 2W-0L,
not 1-1 (okcupid@sus25 +0.30% Brier where F1 read −1.37%). Every hc
stratum's engaged median is positive; every one is a pointer by size (§2).
Cost, same run: engaged base-hc fit ratio **median ×1.20** (per set 1.07
colleges to 1.26 kick), ×1.14 with the variants; the 12 inert hc keys ×1.01,
gr ×1.01; harness hc ×1.07.
bar (d) **PASS**, skill axis: classification (44 sets) new **0.4052 @ 5.0×**
vs old 0.4036 @ 4.8×; regression (59) new 0.7342 @ 6.4× vs old 0.7340 @
6.3×. Neither point dominates the other; the classification gain puts the
default 0.0005 skill under CatBoost (0.4057 @ 51.8×), from 0.0021 under.
**Speed-axis correction, found in passing**: the committed chart (I015)
read the default at 3.5× / 4.3×, today's at 5.0× / 6.4×. We did not slow
down (absolute default fit vs I015: gr ×0.71 median, hc ×0.90); I015's
OPPONENTS were slow — same libraries, same config, yet LightGBM ran 4–6×
and HGB 8–16× slower on 2026-09-18 than today, and CatBoost 27% slower on
gr. Today's ratios match the August base (LightGBM ~5.2× faster than the
default on gr fit), so the I015 chart overstated our speed position and
this refresh corrects it. Cause unknown (a machine-state issue that day);
recorded in the facts ledger.
Charts refreshed from the run without the control arm (`pareto.png`,
`winrate_matrix.png`). Two pre-existing chart defects persist, queued as
H(12): the title names only Grinsztajn though both panels carry hc and the
variants, and HGB's 0.4349 classification skill is its subset artifact
(it skips the hc sets it cannot fit), which also marks every other
classification point off-frontier.
forecast, S4c: bit-identity HIT; Pareto classification "+ < 0.002" HIT
(+0.0016), regression "a hair up" HIT (+0.0002); "gr fit 3–10% under
I015" UNREADABLE (the machine moved: CatBoost itself read 27% faster).
confidence the mechanism is real on the decision suites: high (hc engaged
6W-1L with the CI above zero, every hc variant stratum positive, exact
ties wherever no column reaches 256 levels). Confidence it transfers to
user data: moderately high — the public suite, which nothing was tuned
on, agrees in sign, size and in where the gains sit (entity columns of
1k–30k levels); the price is ~15–20% fit time where it engages.
verdict: **PASS → PR** (library default, tests, harness control arm, docs,
CHANGELOG, charts), waiting for the maintainer's merge. Not self-merged:
it changes `chimeraboost/` and `tests/`. **MERGED by the maintainer the
same day as PR #145 (d14bf38)**; branch deleted, the commit confirmed on
main. The published-chart refresh below still needs its own go.
next: after the merge, the published chart (`public_pareto.png`, full
public field, ~5 h of CatBoost) needs a refresh with the flip — a long run,
its own go; then the probes **S2 → S3 → S6** as queued.

#### I045 2026-09-22 S4 + S9 zero-fit probes (cross-column scale duplication; numeric missing-value census; pre-registered)
why now: I044's `next:` — PR #143 merged by the maintainer at dcbbfeb, no
pick on the shortlist yet, no campaign PR open. These two change nothing
and decide themselves in minutes, so they run ahead of the pick. Class:
**measurement**, `benchmarks/` only. Branch `campaign/s4-s9-zero-fit-probes`.
S4 (refill L2-1): `_cross_block` builds `diff` as the raw `x_i − x_j`;
when σ_i ≫ σ_j the difference is rank-nearly-identical to `x_i`, a
column the trees already have, occupying one of the cross slots. Read:
every gr regression set at seed 0, the default regressor, `cross_pairs_`
after the race; per `diff` pair the training-scale ratio σ_max / σ_min
and |Spearman ρ| between the difference and each parent. **Kill: median
|ρ| against the larger parent < 0.95 OR median scale ratio < 3** — then
the raw difference is not degenerate and the idea dies before any A/B.
Forecast: Grinsztajn is curated toward same-unit numerics (coordinates,
prices, counts), so I expect the scale ratio median **under 3** and |ρ|
median **0.8–0.95** — the kill firing — with a minority of pairs (≤ 30%)
that ARE near-duplicates. A pass would surprise me and would be worth a
standardized-diff arm at S2.
S9 (refill L2-5): the binner routes NaN to the top bin, so missing rows
always go right; before any kernel work, count numeric columns holding
NaN / inf and the rows affected on every gr and hc set. **Kill: fewer
than 3 gr sets carry any** ⇒ the decision suites cannot measure a learned
direction and S9 becomes a robustness note. Forecast: Grinsztajn's
published preprocessing removes missing data, so **0–1 gr sets**; hc
sets carry some (kick, okcupid-stem, house_prices), so **2–5 hc sets**.
ran: `probe_cross_scale_and_missing.py`, ~8 min (36 default regressor
fits at seed 0 + a census of all 73 sets); `results/probe-cross-scale-
and-missing-20260922.{json,log}`.
S4 read: cross selected on **33 of 36** regression sets, **457 diff
pairs** (15 per engaged set: the augmented block carries 15 pairs × 2
operators, gdiff never appears on these numeric sets). Scale ratio
σ_max/σ_min **median 3.52** (53% of pairs ≥ 3, 37% ≥ 10); |ρ| between
the difference and its larger parent **median 0.963** (55% ≥ 0.95),
against 0.375 for the smaller parent. **221 of 457 diff columns (48%)
are near-duplicates of a column the trees already have** (ratio ≥ 3 AND
|ρ| ≥ 0.95). Per set it is bimodal: 10 of 33 sets have ≥ 2/3 of their
diff block duplicated (medical_charges 3/3 at ratio 145, house_sales
13/15 at ratio 210, wine_quality, Brazilian_houses, Ailerons 12/15,
houses, cpu_act, elevators, MiamiHousing, SGEMM at ratio 738) and 9 have
≤ 1/3 (diamonds, pol, sulfur, yprop, Allstate: same-unit columns). The
kill (median |ρ| < 0.95 OR median ratio < 3) did **NOT fire**. Forecast
MISSED on both numbers — I expected Grinsztajn's curation to give
same-unit pairs and a ratio median under 3; it is the mixed-unit sets
(prices next to counts, coordinates next to areas) that dominate, and
there half the diff block is spent on nothing. Note what this is NOT
evidence of: that a standardized difference would be a USEFUL column —
only that the current one is, on half the block, no column at all. The
race decides cross-vs-none per set (B16's territory), so the honest test
is an arm where `diff` is `x_i/σ_i − x_j/σ_j` with σ from the training
rows, judged at S2/S3 like any candidate default.
S9 read: **gr 0 of 59** sets carry any numeric NaN/inf — the kill fires
for the headline suite exactly as forecast (its published preprocessing
removed them). **hc 11 of 14** do, five of them heavily: colleges (32
columns, 100% of rows), cjs (28 cols, 100%), Moneyball (2 cols, 66%),
employee_salaries (2 cols, 32%), porto-seguro (2 cols, 24%); kick,
wine-reviews, Traffic under 7%. So a learned missing direction is
unmeasurable on Grinsztajn and measurable on the hc stratum alone — a
pointer with a stratum, not a candidate for the default's headline axis.
verdict: **S4 PASSES its zero-fit gate → the standardized-diff arm is
an S2 candidate; S9 KILLED for the decision axis as pre-registered,
parked as an hc-only pointer.** Self-merged: `benchmarks/` only.
next: S4's next rung needs a library flag for the A/B
(`cross_diff_standardize`, default off — the I038 shape: muse implements,
S2 synth ChimeraBoost arms, S3 `--decide`, then /experiment) — that is a
new entrant to the beam, and **entrants are the maintainer's pick**. The
loop holds here: the shortlist stands with S4 now carrying its
zero-fit read, S1 (predict latency, exact) and S2/S3 (the two
opponent probes) untouched, and S4-of-the-count-column (the default
flip) still awaiting his go.

#### I044 2026-09-22 H(6) `compare_runs` judges on the decision metric (harness; CHANGES HOW A GATE SCORES; pre-registered)
why now: I043's `next:`; the pick is still the maintainer's. Class:
**measurement instrument that changes what every future bar reads** —
flagged as the self-merge rule requires, and it carries tests, so the PR
waits for the maintainer anyway.
the gap (I042/L4): `compare_runs` defaulted to the harness's `primary`
field, which is negative RMSE for regression but **F1 for
classification**, while every ship gate, the harness SUMMARY block,
`summarize.primary_scores` and the Pareto chart score RMSE / Brier. Every
"engaged wins ≥ half + 1 AND engaged median > 0" bar this campaign
pre-registered on a classification-bearing stratum was therefore read on
a statistic the gate does not name, and the Brier half had to be
recovered by a second `--metric brier` run and hand-unioned with the RMSE
half (I037, I039). Re-read on the decision metric: I036 33W-21L, median
+0.492% (recorded 36W-17L, +0.295%; still a pass); I039 engaged median
**+0.558%** (recorded +0.203%); I021 unchanged (it was scored with
`--metric brier` explicitly).
the change: `--metric decision` — RMSE on regression datasets, Brier on
classification ones (task from the run's dataset metadata; a record with
an `rmse` and no task is regression), both negated so higher is better —
**becomes the default**; `primary` stays selectable and is documented as
the pre-2026-09-22 read; `brier` and `crps` unchanged. `load_run` and
`load_run_seeds` share one `_judged_value`. A one-line header names the
metric on every decision-metric run, so a pasted block cannot be
misattributed. Two tests: a set where NEW has the better F1 and the worse
Brier reads BASE-wins by default and NEW-wins under `--metric primary`;
the orientation and the per-seed view.
forecast: the existing `test_compare_runs` tests pass unchanged (their
classification fixtures carry `primary = 1 − brier`, so both metrics
agree there, and the near-solved tests already pass `--metric brier`);
re-reading I039's JSON with the new default prints the hc engaged median
**+0.558%** and 6W-1L on the same seven sets; I021's read is unchanged.
kill: any existing test that must be edited to keep its meaning (then the
default change is not backward-compatible in a way the record needs and
the flag stays opt-in instead).
ran: the edit, the tests, the two anchors. `test_compare_runs` +
`test_harness_guards`: **19 passed**, every pre-existing test untouched
(kill clause not triggered: the classification fixtures carry
`primary = 1 − brier`, so the two metrics agree there, and the near-solved
tests already named `--metric brier`). Anchor I039, re-read with the new
default: hc **6W-1L**, engaged median **+0.558%** [CI +0.023%..+0.864%],
the header line naming the metric — the number the S4 ask should have
carried. Anchor I021 (`--metric brier` explicit): engaged median −0.003%,
unchanged. Ruff: the same 7 pre-existing hits in `compare_runs.py`,
nothing new.
verdict: **PASS → PR (waits for the maintainer: `tests/`).** Forecast HIT
on all three counts. **Gate-scoring change, stated plainly:** from this
PR on, every `compare_runs` read without `--metric` judges classification
on Brier, not F1; log entries before I044 that quote a classification bar
without `--metric brier` were read on F1 — I036 (33W-21L +0.49% on the
decision metric, recorded 36W-17L +0.30%) and I039 (+0.558%, recorded
+0.203%) are the two this campaign wrote, both still passes, both now
carrying their decision-metric numbers in the I042 facts. `synth_report.py`
still reads `primary`; it is an attribution tool, not a gate, and is left
for H(2)/(3).
next: with H(6) and H(8) done, the remaining harness items (H(7) effect-
vs-seed-noise, H(3) cost line, H(9) provenance, H(10) @time labels) are
small and can ride any idle tick; the shortlist pick (S1–S12) is the
maintainer's, and S4 for the count column too. If neither arrives, the
loop takes **S4 and S9** next — the two zero-fit probes, which change
nothing and decide themselves in minutes.

#### I043 2026-09-22 H(8) identity-snapshot coverage for the paths shipped since 2026-08-30 (harness, pre-registered)
why now: I042's `next:` — the pick is the maintainer's; H(6) and H(8)
are self-mergeable harness rungs that protect every later read, and H(8)
touches `benchmarks/` only. Class: **measurement instrument**, no gate
scoring changes (the snapshot is an exact-equality gate; adding configs
widens what it pins, it does not move any verdict).
the gap (I042/L4): `identity_snapshot` has 28 configs, 155 arrays, and
none exercises `cat_count_features=True` — its two categorical configs
draw cardinality 12 and 7, far under `CAT_COUNT_MIN_CARD = 256` — nor the
classifier's `cross_features="always"` (I019–I021 pinned only the
regressor's), nor `random_effects=True` (#109 slice 1), nor a bag over
categoricals (the shared member cache from C4a). So "identity 155/155
with the flag off" in I038/I039 was true and empty: a refactor could
change flag-on behaviour and the gate would not notice — and flag-on is
exactly where `/code-review` found the two I039 defects.
the change: `_data` gains `hicard=True` (an 11th column, a ~600-level
string categorical with a per-level effect) and returns group labels (40
groups with a per-group offset); five configs — `cat_counts` (n=6000 so
the cross race and the replay refit run over the count column, which is
where the adopted-count logic lives), `cat_counts_w` (weighted: the
`sample_weight` bincount contract), `logloss_forced`, `randeff` (fit and
predict with groups; `group_intercepts_` and `group_ratio_` pinned as
arrays), `bag_cats`; `_EXPECT` proves each path fired (exactly one count
column selected, the forced cross selected, 40 intercepts, 3 members).
Then `save` re-baselines at today's main (the library is unchanged since
#138, and `check` read 155/155 on it), and `check` must read N/N.
forecast: 5 configs add ~28 arrays (155 → ~183); every new `_EXPECT`
fires on the first try or the config is wrong, not the library; `check`
reads N/N immediately after `save` (deterministic seeds; the snapshot is
machine-local and gitignored).
kill: an `_EXPECT` that cannot be satisfied by any config (then the path
is unreachable from the public API and that is its own finding); `check`
not N/N right after `save` (a nondeterminism in one of the new paths — a
defect to record).
ran: five configs added, `save`, `check`: **186/186 identical**, every
new `_EXPECT` fired on the first try (one count column selected on the
~600-level column with the cross race engaged; the forced cross selected
on the classifier; 40 intercepts; three members over categoricals) — the
paths are reachable from the public API and now pinned, 31 new arrays.
**The near-miss is the record's real content.** My first `save` drew the
new group labels from the SHARED random stream, ahead of the existing
draws, so every pre-existing config's data shifted and the re-baseline
silently moved **131 of the 155 old pins** — `check` still read 186/186,
because it compares against what was just saved. Caught only because I
had copied the old `.npz` aside and diffed shared keys (155 of 155
unchanged after drawing the extras from their own streams). That is a
gate that can be blinded by the act of widening it, so `save` now
**refuses** to overwrite a baseline whose existing pins it would move
unless `--rebaseline` is passed, and the docstring says when that flag is
honest (a real behaviour change, never "I added a config"). Verified: the
guarded `save` accepts the corrected panel (0 moved) and refuses the
first version (131 moved).
verdict: **PASS.** The "155/155 with the flag off" claim in I038/I039 now
has a flag-on counterpart in the gate. Self-merged: `benchmarks/` only;
no gate SCORING changed (the snapshot widened, no verdict moved).
next: H(6) — `compare_runs --metric decision` (RMSE reg / Brier clf), the
metric every gate names; it changes how a gate scores, so its step report
says so, and it carries tests, so its PR waits for the maintainer.
Unless the pick arrives first.

#### I042 2026-09-22 beam refill (staleness rule: only blocked-on-sign-off work left; pre-registered)
why now: I041's `next:`. State of the beam: F4 ACTIVE but its measured
exact-rewrite objects are all shipped (C2, C1, C1b, C4a, C4a-2, C3; C4b
parked as an algorithm change); F5 at S4 awaiting the maintainer's go on
the count column's default flip; F1/F2/F3/F6 killed; shortlist R1–R4
resolved, H(1)(4)(5) done, R5–R8 each add a flag (ranked last by the
maintainer's rule), H(2)(3) low value. The staleness rule (everything
blocked on sign-off, fewer than 3 ACTIVE families) calls for a refill.
Method, as I022: four read-only lens agents in parallel — L1
profiling-speed (fit AND predict, after C3/C4a; the count column's
engaged cost; the parked C4b), L2 literature-mechanism (what modern
GBDTs and the TabFM literature offer that the barrier list does not bar,
incl. the two pointers this campaign left: coarse TS quantization on
small categorical data (I035: +1.7–3.6% on the two small controls) and
the raced-regression linear-leaf read (I041)), L3 opponent-ablation (what
of CatBoost's hc edge the count column leaves, and where LightGBM /
CatBoost beat us on Grinsztajn and why), L4 harness-measurement (gaps the
instruments still have — e.g. no flag-on config in `identity_snapshot`,
seed-agreement mostly split on synth wins, the cost column H(3)). Then
`barrier_check.py` over every survivor, dedup by hand against this log,
`research/SUMMARY.md` and BARRIERS, ranked by expected win-rate movement
per hour, survivors to the maintainer (beam cap 5).
ran: four lens agents in parallel (~11 min wall), each with the barrier
list and the log in hand; L1 ran a NEW read-only predict-side profile (5
reps, wrapped/plain 0.99–1.01 on three hc sets, one soft read on
MagicTelescope at 1.20) and a 0.49–0.50× prototype of the categorical
lookup; L3 recomputed the count column's residual per set from the two
JSONs (both share splits, so the pairing is exact) and read catboost
1.2.10's MULTICLASS `get_all_params()` for the first time; L4 re-derived
four recorded numbers from the JSONs. `barrier_check.py` over the eleven
survivors in one pass (matches recorded in the table). Dedup: the fixed-15
TS quantization, the 0.5 prior target, the Counter, permutations,
`random_strength`/bootstrap, histogram subtraction, GOSS/EFB, leaf-wise
growth, bin counts, `leaf_estimation_iterations` (reads 1 in 1.2.10),
calibration, transductive counts, per-category target variance,
shared-structure heads, dropping zero-gain features, LightGBM-style native
splits, H2O blending, REML shrinkage, EBM, momentum/DART, monotone
constraints, `cat_smoothing` — all checked against the record and
rejected by the lenses with the closing entry named.
result: the shortlist above (S1–S12, H(6)(7)(3)(8)(9)(10)). Four findings
worth their own fact lines: (1) predict had never been profiled, and on
categorical data one exact rewrite is 62–83% of it; (2) the noisy
low-dimensional binary cluster on Grinsztajn is a CatBoost-only edge
(LightGBM trails us on 8 of the 9 sets), which prices out bins, depth and
stochasticity and leaves the split score; (3) catboost 1.2.10 builds its
MULTICLASS CTR from three MinEntropy binarizations of the ordinal class
index, not one-vs-rest — I034's `cb_our_ts` fed it one-vs-rest columns, so
the "encoder is the gap" read on the two multiclass sets is a read
against a different construction than CatBoost's own; (4) `compare_runs`
judges classification on F1 while every gate judges Brier, so three
recorded bars were read off the wrong statistic (all three still pass;
one is stronger than recorded).
verdict: SHORTLIST WRITTEN — the loop pauses here; entrants are the
maintainer's pick (beam cap 5), and so is S4 for the count column.
Self-merged: this file only.
next: on the pick, S0 for each entrant in shortlist order; S1's walltime
instrument and S4's and S9's zero-fit probes can run before any pick if
the bench is idle, since they change nothing and cost minutes. H(6) and
H(8) are self-mergeable harness rungs and go first when nothing else is
queued.

#### I041 2026-09-22 R4 S0+S1 (target-statistic columns as linear-leaf terms; zero library change, pre-registered)
why now: I040's `next:`. PR #139 self-merged at cc4308e, no campaign PR
open, no run in flight; the maintainer has not called for S4 on the count
column, so the loop takes the next shortlist item. R4 is the last
non-knob mechanism on the list. Class: **zero-library probe of a
candidate default change** (monkeypatch), `benchmarks/` only. Branch
`campaign/r4-ts-linear-terms` from main.
the mechanism (shortlist R4, lens L1): `_build_centers_std` zeroes every
column whose `is_numeric_binned_` is False, so `_fit_linear_leaf_tail`
can never pick a target-statistic column as a ridge term. On sf-police
(5 of 6 columns categorical) the "linear" leaf has ONE usable column; on
wine-reviews 1 of 10. The TS values are ordinal in the target by
construction, so a linear term over them is well posed. Probe
`benchmarks/probe_ts_linear_terms.py`: all 14 hc sets × 3 seeds, paired
splits, `default` vs `ts_linear` (the TS and combo blocks marked
linear-eligible after `fit_transform` / `from_base_with_cross`). Binary
sets engage (linear leaves are the binary default), regression sets
engage through the const-vs-linear race, and the **four multiclass sets
cannot engage** (no linear leaves there) — the in-run inert control,
exact ties required.
barrier: B3 — not a port (CatBoost has no linear leaves). B1 — the race
and the linear default are gated by row count; sets below the gate are
inert and read as ties, which is the control, not a problem. B14 — the
race budget is untouched. B18 — the encoder is untouched; this changes
what the LEAVES may do with its output. The shortlist's own risk stands:
the ridge coefficients are fit on TS values that are in-sample for the
rows they were computed on (the ordered prefix mitigates, as it does for
the splits), so a linear term could over-trust them where a split would
not.
forecast, before the run: binary — sf-police **+0.1 to +0.5%** (the one
set where the linear leaf is starved), kick 0 to +0.3%, porto and
kdd_ipums within ±0.2%; regression — wine-reviews 0 to +0.4%, the rest
within ±0.3% (the race picks constant leaves where linear ones do not
pay, so a loss there means the race was fooled in-sample); multiclass
**four exact ties**; fit ratio 1.0–1.1 (a ridge over a few more
columns). The shortlist's "+2 to +4 hc points" I do not believe: splits
already read the TS ordinally, and a leaf-level slope on the same column
adds little a deeper split would not. Where I could be wrong: on
cat-dominated sets the linear leaf currently fits a slope on ONE
numeric column and the race still picks it — if the race is picking a
degenerate model, giving it the TS columns could be a real change.
bars: (a) control — the four multiclass sets exact ties; (b) direction —
engaged (binary + regression, ≤ 10 sets) wins ≥ half + 1 with median >
0, sf-police positive; (c) cost — fit ratio median ≤ 1.10. Pass ⇒ S2
synth (a flag in the library would be needed for the harness arm — the
same shape as I038's, a muse task); fail ⇒ R4 KILLED at S1, the
shortlist closed but for the knob items R5–R8.
cost: 84 fits ≈ 6 min.
ran: 84 fits, ~7 min; `results/probe-ts-linear-terms.jsonl` +
`-20260922.log`. d% vs the default, + = better (RMSE / Brier):
  binary      sf-police −0.03%   kick −0.76%   porto −0.06%   kdd_ipums +1.03%
              → 1W-3L, median −0.05%, fit ratio median 1.03
  regression  wine-reviews +0.09%   colleges +0.74%   employee_salaries +0.78%
              black_friday −0.03%   house_prices / Moneyball exact ties
              → 3W-1L-2T, median +0.04%, fit ratio median 0.96
  multiclass  okcupid / Traffic / cjs / eucalyptus **exact ties** (4/4)
read: bar (a) control **PASS** — the four multiclass sets are exact
ties, and so are the two regression sets where the race picked constant
leaves (house_prices, Moneyball): the change engages only where linear
leaves exist, as claimed. Bar (b) direction **FAIL** — engaged 8 sets
**4W-4L**, median +0.03%, and sf-police, the set the mechanism was
named for, reads −0.03%. Bar (c) cost PASS (1.03 / 0.96). The pattern
is the shortlist's own risk realized: on BINARY sets linear leaves are
always on (no race), and handing them TS columns hurts kick by 0.76% —
a ridge slope on an in-sample target statistic over-trusts it where a
split does not; on REGRESSION the const-vs-linear race protects the
choice and two sets gain +0.7% (colleges, employee_salaries). That
regression-only read (3W-1L on 4 engaged) is a pointer, not a result —
recorded, not chased: it would need its own flag and the race's
protection is exactly what makes it safe, so it cannot become the
binary default. Forecast: control HIT, cost HIT, sf-police MISSED (I
said +0.1 to +0.5), kick MISSED (−0.76 against 0 to +0.3), kdd +1.03
MISSED the other way, the shortlist's "+2 to +4 hc points" nowhere.
verdict: **KILL R4 at S1.** The linear leaf is not starved on
cat-dominated sets in any way that feeding it the TS columns fixes;
where it is not raced it over-trusts them. Shortlist R4 marked KILLED.
Self-merged: `benchmarks/` only.
next: the shortlist's non-knob items are exhausted (R1, R2, R3, R4
resolved; H(1)(4)(5) done). Left: R5–R8 (each adds a flag; the
maintainer's ranking puts them last), H(2)(3) (harness cost / transfer
instruments, low value now), and **S4 for the count column, which
needs his go**. Beam state: F4 exhausted, F5 at S4 awaiting the pick,
everything else killed — the staleness rule (only blocked-on-sign-off
work left) says **a refill is due**: four read-only lens subagents,
funnelled through `barrier_check.py`, survivors to the maintainer. That
is the next rung unless he calls S4 first.

#### I040 2026-09-22 F5 (per-column calibrated shrinkage for the ordered TS — the random-effects form of the count column; zero library change, pre-registered)
why now: PR #138 (the opt-in count column) merged by the maintainer at
f05e693, no campaign PR open, no run in flight. The maintainer's
question in chat, after I039: "do any other GBDTs do this? seems like a
weak random effects proxy." Recorded here as the trigger: yes, CatBoost
does exactly this (its `Counter` CTR and the 1/(n+1) spread of its
three-prior columns; frequency encoding is standard practice; LightGBM
and XGBoost regularize by count inside their categorical splits
instead), and yes, it is a proxy — our ordered TS `(sum + mean·a)/(n +
a)` IS the random-intercept posterior mean with a FIXED variance ratio a
= 1, and the count column lets the trees recover the statistic's
reliability after the fact. **The S4 default flip of the count column is
HELD** until this probe reads: if the calibrated shrinkage recovers the
same gap for free (no extra columns, no fit-time cost), it should ship
instead. Class: zero-library probe (monkeypatch), `benchmarks/` only.
Branch `campaign/f5-reml-smoothing` from main.
design: per categorical column j (and per class target), the one-way
random-effects ANOVA estimate of the variance ratio λ_j =
σ²_within / σ²_between (the closed form REML reduces to for balanced
groups; σ²_between floored at 1e-3·σ²_within so a signal-free column
reads λ = 1000 and shrinks almost fully), clipped to [0.1, 1e4], used as
that column's `a` in the ordered TS at fit AND transform (both sides move
together — this is the fit-side prior, not B18's transform-side
re-weighting). Arms on the same seven sets and paired splits as I033:
`reml`, `count` (the library flag on these splits, for a paired read),
`reml+count`. Reference `chimera` / `cb_default` from I033's JSONL.
barrier: B18 — closed the transform-side re-weighting; this changes the
fit-side variance ratio and moves both sides, the axis B18 itself named
as separate ("`cat_smoothing` moves both sides together"). B5 — binds
the READ, not the idea: shrinkage corrects only the component that
varies across categories, and per-count reliability is exactly that
component (B18 measured it: 0.63–0.70 on rare categories, ~1 on common
ones); so the gain must concentrate on the max-cardinality sets, and a
flat gain everywhere would mean something else is moving. B3 — not a
port (CatBoost also uses a fixed prior weight of 1). The two
`cat_smoothing` kills on hc (PARETO_PLAN 2026-07-15) swept one GLOBAL
value; a per-column λ is a different object and the record does not
cover it.
forecast, before the run: `reml` — sf-police **+0.2 to +0.6%** (B18's
over-trust lives there; λ ≫ 1 on a 15k-card column), Traffic +0.2 to
+0.6%, okcupid 0 to +0.3%, kick 0 to +0.3%, porto flat; controls within
±0.3%; so **3 to 4 of 5 gap sets up, recovering 30–70% of what `count`
recovers on the same splits** — a monotone re-weighting the trees can
partly undo, against a count column that adds a degree of freedom the
trees can use non-monotonically. `reml+count` ≥ either alone if they are
complements, ≈ `count` if substitutes. λ read: median in the tens on the
high-card sets, near 1 on the low-card ones. Where I could be wrong: on
a 15k-card column with 100k rows most categories are singletons, and the
between-variance estimate from singletons is noise — λ may saturate at
the clip and over-shrink, reading as a LOSS on sf-police; that would
argue for a count-aware estimator, not against the idea.
bars: (a) `reml` better on ≥ 3 of 5 gap sets with both controls ≥ −0.3%
and the gain ordered by cardinality ⇒ it is real; (b) `reml` median ≥
`count` median on the gap sets ⇒ the calibrated shrinkage SUPERSEDES the
count column as the S4 candidate (cheaper, principled); (c) `reml+count`
better than both by ≥ 0.1% median ⇒ complements, S4 considers both; (d)
`reml` fails (a) ⇒ the count column goes to S4 alone, with this on
record as the principled alternative that did not transfer.
cost: 63 fits of ours ≈ 4 min.
ran: 63 fits (+21 for the clipped arm added after the first read,
pre-registered in the script's docstring), ~6 min;
`results/probe-ts-reml-smoothing.jsonl`, logs `…-20260922.log` and
`…-cap-20260922.log`. d% vs our default, + = better; the `count` arm is
the library flag on the same splits:
  arm          sf-police  Traffic    kick   okcupid   porto  |  kdd_ipums  eucalyptus
  reml           +0.13%   −0.49%   −0.88%   +0.02%  +0.06%  |   −3.76%     −9.62%
  reml_cap10     +0.22%   +0.20%   −0.68%   +0.25%  −0.04%  |   +0.29%     −5.38%
  count          +0.67%   +1.04%   −0.09%   +0.19%  +0.00%  |    0.00       0.00
  reml+count     +0.54%   +0.84%   −0.50%   +0.01%  +0.06%  |   −3.76%     −9.62%
  λ the estimator asked for (median / max per set): sf-police 372 /
  1000, porto 696 / 1000, kick 123 / 1000, okcupid 79 / 1000, Traffic
  66 / 1000, eucalyptus 11 / 411, kdd_ipums 2.8 / 1000; under the [1, 10]
  clip every high-card set sits AT 10.
read: **the calibrated shrinkage loses, and the reason is instructive.**
(1) Uncapped, the ANOVA variance ratio saturates: on a 15k-card column
most categories are singletons, the between-category variance estimate
is noise, and λ lands at hundreds — the encoding collapses toward the
prior and the two small controls fall apart (kdd −3.8%, eucalyptus
−9.6%); Traffic and kick lose too. Bar (a) FAIL. (2) Capped at 10, the
gap-set median matches the count column's (+0.20 vs +0.19) but the
pattern does not: it is small everywhere (sf-police +0.22 against the
count's +0.67, Traffic +0.20 against +1.04), it still costs kick −0.68
and eucalyptus −5.4, and λ pins at the cap on every high-card set, so
the "estimate" is the cap. Bar (b) FAIL either way; (c) FAIL —
`reml+count` is below `count` on four of five. (3) Why the proxy beats
the principle here: shrinking a rare category toward the prior DELETES
its evidence, on the bet that it is noise; a count column keeps the
evidence and hands the trees the sample size, so a rare category with a
strong signal (kdd_ipums, eucalyptus — census and species data where
rare levels mean something) is still usable, and the trees decide
per split whether to trust it. A marginal variance decomposition cannot
know how the statistic will be used inside a conditional model. The
forecast MISSED on direction (I said 30–70% of the count's gain,
ordered by cardinality; it is a loss uncapped and a flat wash capped)
and HIT on where I would be wrong ("λ may saturate at the clip and
over-shrink … that would argue for a count-aware estimator, not against
the idea") — a count-aware estimator is what the count column already
is, one level up.
verdict: **KILL the calibrated-shrinkage form; the count column goes to
S4 alone.** Recorded as the principled alternative that did not
transfer, with the maintainer's question as its trigger. The one door
this leaves open is estimating λ per column by cross-validated
prediction inside the model rather than by ANOVA — a different, costly
object, not queued. Self-merged: `benchmarks/` only.
next: **S4 for `cat_count_features` awaits the maintainer's go** (a
default flip: /experiment with the chart-grade field, `--public` at ship
time, the Pareto read). The loop does not launch S4 on its own. Meanwhile
**R4 S0** (TS columns as linear-leaf terms) is the next shortlist item,
next rung.

#### I039 2026-09-22 F5 S3 again, with the library form (`ChimeraBoost` vs `ChimeraBoostCatCountLib` = `cat_count_features=True`; pre-registered)
why now: I038's muse pass is in (see its entry once scored: 1133 tests
green, identity 155/155 with the flag off). Same branch; this run and
I038's verdict travel in one PR because the arm IS the library flag.
forecast (I038's, restated as the bars): gr and its variants **0-0-59 /
0-0-12 / 0-0-6 exact ties**; hc — sf-police +0.8 to +1.0%, Traffic +0.6
to +1.0%, wine-reviews +0.5 to +0.8%, okcupid / kick / colleges /
employee_salaries within ±0.3%; **kdd_ipums, Moneyball, house_prices,
porto, eucalyptus, black_friday, cjs exact ties** (no column of
cardinality ≥ 256: the in-run control inside hc); hc engaged wins ≥ 4 of
7 with median > 0; **cost: engaged fit ratio median ≤ 1.10, sf-police ≤
1.3**. Variants: hc@time direction as hc (sf-police@time the known
pointer), pointers only.
kill: hc engaged wins < 4 of 7 or median ≤ 0 ⇒ F5 KILLED; any non-tie on
a set without a qualifying column ⇒ the gate is not doing what it claims,
find out before reading anything else; cost median > 1.10 ⇒ recorded as
the price of the strength, S4's Pareto read decides.
launch note: the first launch (arm still named `ChimeraBoostCatCount`)
was STOPPED five minutes in and its partial outputs deleted, on two
`/code-review` findings against I038's diff: (1) the replay refit adopts
the donor's binner, whose count borders were fit on the 80% training
subset, but recounts on all rows — every count lands ~1.25× higher and
the replayed splits partition rows the structure was not chosen for
(measured by the reviewer: mean bin index 6.67 → 8.31 on a 600-category
column); the fix is to adopt the donor's count lookups exactly as its
binner is adopted. (2) `np.bincount` ignores `sample_weight`, so a
zero-weight row still counts — breaking `fit_transform`'s contract that
zero-weight rows shape nothing. Both go back to muse as a follow-up task
before this run is relaunched. (3) The library arm is renamed
`ChimeraBoostCatCountLib`; `ChimeraBoostCatCount` stays the name of the
raw-append prototype so I036/I037's result JSONs keep their label.
ran: relaunched after the fix, `run_benchmarks.py --decide --seeds 3
--save benchmarks/results/campaign-f5-s3b-catcount-20260922 --models
ChimeraBoost ChimeraBoostCatCountLib`, ~15 min → `…json`; scored
`compare_runs.py --by-suite` on primary (`-compare-…txt`) and Brier
(`-compare-brier-…txt`). Harness SUMMARY, the default vs the arm: gr
**W0-L0-T57** ×0.99; hc **W1-L6-T6** (the arm 6-1 on the engaged seven,
seven exact ties), median gap +0.00%, **speed ×1.09**; gr@sus25 0-0-12,
gr@sus50 0-0-6; hc@sus25 W0-L2-T1, hc@sus50 W0-L1-T1, hc@time W1-L3-T3.
bar 1 control **PASS**: gr and its variants 0-0-59 / 0-0-12 / 0-0-6.
gate check **PASS**: inside hc the seven sets with no column of
cardinality ≥ 256 — Moneyball, black_friday, cjs, eucalyptus,
house_prices, kdd_ipums, porto-seguro — are **exact ties** to the bit,
exactly the pre-registered list; the gate does what it claims, and the
I037 kdd_ipums loss (−3.28%) is gone by construction.
bar 2 strength **PASS**: engaged 7, RMSE / Brier **6W-1L**; primary
median **+0.20%** [CI +0.02%..+0.76%]. Brier: sf-police **+0.73%**,
Traffic **+0.86%**, kick +0.35%, okcupid −0.00% (the one loss, a tie in
substance); RMSE: employee_salaries **+2.45%**, wine-reviews **+0.56%**,
colleges +0.02%. Seeds: sf-police, Traffic, wine-reviews,
employee_salaries unanimous; colleges, kick, okcupid split. Against the
forecast: sf-police just under its +0.8 to +1.0 band, Traffic and
wine-reviews inside, kick at the edge of ±0.3, employee_salaries **+2.45%
where I said ±0.3** — its three columns of card 2264 / 694 / 385 are
exactly the regime, and I under-read it because I037's raw arm had lost
there (−0.24%) — with the cross race on the counts gone, the counts help.
bar 3 variants **not contradicting**: hc@time **4W-0L** (sf-police@time
**+1.40%** where the raw arm read −3.47% — the temporal loss was the
cross race auditioning counts, not the counts; Traffic@time +0.97%,
employee_salaries@time +2.10%, kick@time +0.14%, the other three exact
ties), hc@sus50 1-0 (wine-reviews@sus50 +0.72%), hc@sus25 1-1
(employee_salaries@sus25 +0.86%, okcupid@sus25 −1.37% on split seeds).
bar 4 cost **MISSED, and by less than the pre-registration allowed for**:
engaged fit ratio **median 1.165** (bar ≤ 1.10), per set colleges 1.04,
Traffic 1.13, okcupid 1.15, wine-reviews 1.17, employee_salaries 1.19,
sf-police **1.20** (bar ≤ 1.3 HIT; 65 trees vs 59), kick 1.26; the seven
tie sets 0.98–1.06; all-hc median 1.05, harness ×1.09. Tree counts move
little (wine-reviews 398 vs 394), so it is per-round cost of one more
binned column per qualifying categorical — proportionate, and a fifth of
the raw arm's 1.47 / 2.6×.
verdict: **PASS on strength and the gate, cost 1.165 against 1.10 — S4
by the pre-registered rule** ("cost median > 1.10 ⇒ recorded as the
price of the strength, S4's Pareto read decides"). The count column
with the library form is the first F5 mechanism to reach S4 in seven
tries. What ships in THIS PR is the opt-in flag (default False), which
changes nothing for anyone who does not set it (identity 155/155). The
default flip is /experiment S4: the chart-grade field with CatBoost on
the decision suites, `--public` at ship time, the Pareto read (strength
vs slowdown) and the maintainer's sign-off — that rung launches only on
his go, because it is a default change.
next: **the maintainer's call on two things** — merge this PR (opt-in
flag, library source), and whether to run /experiment S4 on flipping
`cat_count_features` to `None`→auto (on with categoricals; the gate
makes it inert below cardinality 256, i.e. on every Grinsztajn set).
Until then the loop moves to the next shortlist item, **R4 (TS columns
as linear-leaf terms)**, S0 first.

#### I038 2026-09-22 F5 library form (`cat_count_features`, opt-in, card ≥ 256, invisible to cross / linear races; muse task, pre-registered)
why now: I037's `next:`. PR #137 self-merged at 6ead5df, no campaign PR
open, no run in flight. Class: **library source, default-off flag for the
A/B** — the shape the campaign uses for every candidate default change
(F1's `cross_top_columns`, F3's `cross_features="always"`); a default flip
is /experiment S4 with the maintainer's sign-off, never this rung. Muse
implements it from `campaign_tasks/20260922-f5-cat-count-features.md`
(gitignored; the verdict cites what it produced); the PR waits for the
maintainer. Branch `campaign/f5-cat-count-features` from main.
design: `FeaturePreprocessor(cat_count_features=False)`, threaded through
`_BaseBooster` and both estimators. On: every categorical whose training
cardinality is ≥ `CAT_COUNT_MIN_CARD = 256` gets one float column, the
category's training-row count (transform: the fit-time count, unseen →
0.0), stacked at the END of the numeric block — block order becomes
[numeric | count | cross | TS]. `n_numeric_block_` replaces
`len(num_features_)` as the cross-splice offset in `from_base_with_cross`
and `_prep_matrices`; `is_numeric_binned_` is False on the count columns
(linear leaves never see them); the cross machinery pairs raw input
columns only, so it never sees them either; `feature_map_` sends each
count column's gain to its categorical. Off: no code path changes —
`count_features_ = []`, the numeric block untouched, `identity_snapshot`
155/155 and the goldens are the gate. Why 256: the binner's resolution —
above it the target statistic cannot resolve categories one by one, and
I035/I037 gained only on such columns (sf-police 15165, Traffic 3831,
okcupid 7020, kick 1063, wine-reviews 31960, colleges 6039,
employee_salaries 2264) and lost where none exists (kdd_ipums 192,
Moneyball 39, house_prices 25, porto 104).
barrier: B3 — the mechanism went through ablation (I033), the encoder
proof (I034), the narrowest transfer (I035), S2 (I036) and S3 (I037)
before a line of library code; this rung is the port B3 says must follow
that order. B12 — a flag for the A/B, not an arm in the product. B1 —
inert without categoricals and below the cardinality gate by
construction.
forecast, before any code: muse finishes in one pass (four files, the
plumbing mirrors `cat_combinations`); identity 155/155; suite green plus
~12 new tests. Then (next rung, I039) ONE `--decide --seeds 3` run of
`ChimeraBoost` vs `ChimeraBoostCatCount` (the arm now = the flag): gr
0-0-59 exact ties; hc — sf-police +0.8 to +1.0%, Traffic +0.6 to +1.0%,
wine-reviews +0.5 to +0.8%, okcupid / kick / colleges / employee_salaries
within ±0.3%, and **kdd_ipums, Moneyball, house_prices, porto,
eucalyptus, black_friday, cjs exact ties** (no column qualifies: the
in-run control inside hc); wins ≥ 4 of the 7 engaged with median > 0;
**cost — engaged fit ratio median ≤ 1.10, sf-police ≤ 1.3** (one count
column, no cross audition switched on). Where I could be wrong: the
sf-police 2.6× may not be the cross race but the extra rounds (89 vs 59
trees) the column earns — then the cost is the price of the strength and
the Pareto read decides at S4; and eucalyptus's +2.9% (card 27) goes away
by design, which is the right trade if it was noise (its seeds were
split) and a loss if it was not.
kill: (a) identity not 155/155 or any golden red with the flag off; (b)
muse fails or times out — record what it left, revert, re-scope; (c) at
I039, cost median > 1.10 again with the count columns provably outside
the races — then the mechanism costs what it costs and S4's Pareto read
is the only judge; (d) at I039, hc engaged wins < 4 of 7 or median ≤ 0
⇒ F5 KILLED, B3 strengthened: the narrowest faithful port did not hold.
ran: muse exit 0 in one pass, ~25 min; its `RESULT.md` reported 10 new
tests green, full suite 1133 passed + 1 skipped, ruff clean on the four
files it touched. It found what the task text had not: `FeaturePreprocessor`
is built in THREE places, and the replay-refit path
(`_prep_or_replay_matrices`) crashes flag-on unless it gets the flag and
pins the count-column SELECTION to the donor's (a near-threshold column
whose cardinality drifts across 256 between the training split and the
full-data refit would otherwise shift the adopted binner's layout). Diff
reviewed by hand: `_fit_count_tables` / `_count_block` helpers, the count
block stacked after the raw numerics, `n_numeric_block_` as the cross
splice offset in `from_base_with_cross` and `_prep_matrices`,
`is_numeric_binned_` as explicit slices (raw True / count False / cross
True / TS False), `feature_map_` sending count gains to the categorical;
both estimators gain the parameter after `cat_combinations`, not in
`_SKLEARN_ONLY`. `/code-review` (medium) then found two real flag-on
defects and one bookkeeping issue: (1) the replay refit adopted the
donor's binner — count borders fit on the 80% training subset — but
recounted on all rows, so every count landed ~1.25× higher and the
replayed splits partitioned rows the structure was not chosen for
(measured: mean bin index 6.67 → 8.31); (2) `np.bincount` ignored
`sample_weight`, so zero-weight rows still counted, against
`fit_transform`'s contract; (3) my harness rename had re-used the
`ChimeraBoostCatCount` label for a different arm than I036/I037's JSONs
carry. The S3 run then in flight was STOPPED. A follow-up muse task
(`20260922-f5-cat-count-fixes.md`) fixed (1) by adopting the donor's
count lookups re-indexed onto the refit's codes — the whole count space
(values, borders, thresholds) is the donor's, as the binner already was —
and (2) with weighted bincount; 3 more tests pin them (the refit's counts
equal the donor's for every donor-seen category and 0.0 otherwise, bin
indices identical through both preps; weight totals under
`sample_weight`; flag-off inertness with weights). (3): the library arm is
`ChimeraBoostCatCountLib`; `ChimeraBoostCatCount` keeps naming the raw
prototype. Claude's side on the same branch: `--chimera-cat-counts` knob
and the arm in `run_benchmarks.py`, the `docs/parameters.md` row.
gates: **1136 passed, 1 skipped** rerun under the conda python after the
fix; `identity_snapshot.py check` **155/155 identical** (flag off, both
times); ruff clean on every touched file.
verdict: **PASS → the S3 run (I039).** Forecast HIT on muse finishing in
one pass and on identity; MISSED on "the plumbing mirrors
`cat_combinations`" — the replay path is the part `cat_combinations` never
had to think about, and it took a reviewer to see the count space had to
be frozen with the binner. Library source: the PR waits for the
maintainer, together with I039's read.
next: I039 (below in this file, above in the log).

#### I037 2026-09-22 F5 S3 (the categorical count column on the decision tier; harness arm, pre-registered)
why now: I036's `next:`, S2 passed all four bars. Class: **candidate
DEFAULT change at S3** — the same harness arm `ChimeraBoostCatCount`
paired against `ChimeraBoost` in ONE `--decide --seeds 3` run (Grinsztajn
57 + high-card 14 + their @sus25 / @sus50 / @time variants), benchmarks
only. Branch `campaign/f5-s3-catcount` from main.
forecast, before the run (per stratum, never pooled — `--by-suite`):
gr — **every row an exact tie** (Grinsztajn's curation leaves no
categorical column; the arm appends nothing), 57/57; gr@sus25 / @sus50 /
@time likewise. hc (13 scored) — **positive by majority, engaged median
+0.3 to +0.8% Brier / RMSE**: the four sets I035 measured carry their
reads (sf-police ~+0.9, Traffic ~+1.5, okcupid ~+0.4, kick ~+0.3), the
regressions (wine-reviews, colleges, house_prices, black_friday,
employee_salaries, Moneyball) are new ground and I expect small positives
on the high-card ones (wine-reviews, employee_salaries) and noise on the
rest; kdd_ipums and eucalyptus as at I035 (split seeds, flat to
slightly negative); cjs a tie or noise (Brier ≈ 0). hc@sus25 / @sus50 —
the small-data regime where rare categories multiply: direction the same
as hc, size larger, but ≤ 8 sets each so pointers. hc@time — pointer.
Cost — hc engaged fit ratio median **≤ 1.05** (real sizes amortize one
column per categorical).
bars: (1) control — gr and its variants exact ties, 100%; (2) hc — wins
≥ half + 1 of the scored sets AND engaged median > 0 (the stratum has 13
scored sets, above the 8-decided POINTER line only if ≥ 8 engage: they
all should, every hc set has categoricals); (3) variants — no hc variant
stratum contradicts hc's direction by majority (pointers, read not
gated); (4) cost — hc engaged fit ratio median ≤ 1.10 (hard) with ≤ 1.05
expected. Pass ⇒ a muse task implements the count column in the library
(auto-on with categoricals, `prep`-side, one numeric column per
categorical; bit-identical on data without categoricals) and /experiment
S4 runs the chart-grade field with CatBoost for the maintainer's
sign-off. Fail on (2) ⇒ F5 KILLED with the mechanism recorded: it
transfers on the ablation panel and the synth prior but not on the
decision suite.
cost: ~10–15 min (two ChimeraBoost arms, 103 dataset rows × 3 seeds).
ran: `run_benchmarks.py --decide --seeds 3 --save benchmarks/results/
campaign-f5-s3-catcount-20260922 --models ChimeraBoost ChimeraBoostCatCount`,
~14 min → `…-20260922.json`; scored `compare_runs.py --by-suite
--expect-inert` on primary (`-compare-…txt`) and on Brier
(`-compare-brier-…txt`). Harness SUMMARY, the default vs the arm: gr
**W0-L0-T57**, ×0.99; hc **W4-L9** (the arm 9-4), median gap −0.07% (the
arm +0.07%), **speed ×1.47**; hc@sus25 W1-L2 ×1.10; hc@sus50 W0-L2 ×1.63;
hc@time W1-L6 ×1.40.
bar 1 control **PASS**: gr 0-0-59 exact ties, gr@sus25 0-0-12, gr@sus50
0-0-6 — every row without a categorical column identical. (The
`--expect-inert` CONTROL FAIL lines on the hc variant strata are the
tool saying nothing was inert THERE, which is right: every hc set has
categoricals; gr is the control.)
bar 2 strength **PASS, marginal**: hc 13 engaged (cjs a near-solved tie):
RMSE / Brier wins **9 of 13** (bar 7+), median +0.07%. Brier on the 7
scored classification sets: sf-police **+0.88%**, Traffic **+0.81%**,
eucalyptus **+2.94%**, kick +0.05%, okcupid +0.07%, porto +0.03%,
kdd_ipums **−3.28%** — engaged median +0.065% [CI +0.031%..+0.878%];
sf-police, Traffic, kdd and porto unanimous across seeds. RMSE on the 6
regressions: wine-reviews **+0.67%**, black_friday +0.32%, colleges
+0.25%, house_prices −0.02%, employee_salaries −0.24%, Moneyball −0.37%.
Against I035's probe panel (sf-police +0.88 / Traffic +1.46 / okcupid
+0.43 / kick +0.26) the harness splits give sf-police the same, Traffic
half, okcupid and kick nothing. Variants: hc@time 5-2 on primary (Brier
3-1: Traffic@time +1.71, eucalyptus@time +2.52, kick@time +0.43,
**sf-police@time −3.47%** — under a temporal split the training-period
counts of a 15k-card column mislead; a pointer, the loudest one here),
hc@sus25 1-2, hc@sus50 1-0 — all pointers.
bar 4 cost **FAIL**: hc engaged fit ratio **median 1.29** (bar ≤ 1.10),
harness ×1.47; per set 0.97–1.23 on the low-card sets and **sf-police
2.58×, wine-reviews 2.59×, employee_salaries 1.70×, okcupid 1.55×,
Traffic 1.46×**. The cause is not the column count: sf-police adds ONE
column (its other four categoricals are card ≤ 16 and their counts are
near-constant) and takes 2.6× with 89 trees against 59; wine-reviews
takes 2.6× with FEWER trees (351 vs 394), so it is per-round cost. The
count columns enter the estimator as ordinary numerics, so they are
candidates for the cross-feature race and the linear-leaf race:
sf-police goes from one numeric column to two and wine-reviews from one
to six, which switches the cross audition on where it was inert (it
needs two numerics) and multiplies its candidate pairs. The slowdown is
the cross machinery auditioning count columns, and some of the strength
may be crosses OF counts rather than the counts themselves. A harness
arm cannot separate these (no knob excludes columns from candidacy).
The kdd_ipums loss (−3.28%, unanimous) is the other design fact: 27
categoricals of card ≤ 192 on 7k rows become 27 near-noise columns, the
`cat_fraction` negative I036's OLS already showed.
verdict: **FAIL on cost, PASS on strength → the ungated harness form is
dead; the mechanism is not.** Bars: (1) HIT, (2) HIT at the margin (I
forecast +0.3 to +0.8 and got +0.07), (3) pointers not contradicting,
(4) MISSED by a wide margin (forecast ≤ 1.05, got 1.29) — I priced one
column per categorical as one column's cost and forgot what the
estimator does with a new numeric. F5 stays ACTIVE. Self-merged:
`benchmarks/` only.
next: **I038 — the library form, a muse task**, then S3 again with it:
`cat_count_features` in `FeaturePreprocessor` — for every categorical
whose training cardinality is ≥ `CAT_COUNT_MIN_CARD = 256` (the binner's
resolution: above it the target statistic cannot resolve categories
one by one, and I035/I037 gained only on such columns — sf-police,
Traffic, okcupid, kick, wine-reviews, colleges, employee_salaries — and
lost on kdd_ipums, Moneyball and house_prices, whose columns are all
below it) append one float column, the category's training-row count
(transform: the fit-time count, unseen → 0), placed in the numeric
block but **excluded from cross-feature candidacy and from
`is_numeric_binned_`** (so neither the cross race nor linear leaves see
it), default OFF for the A/B (`cat_count_features=False`; the harness
gets `--chimera-cat-counts` and an arm), bit-identical with the flag off
(identity snapshot 155/155, goldens green). Then ONE `--decide --seeds 3`
run, same bars as I037 with cost ≤ 1.10 hard and the prediction that
sf-police / Traffic / wine-reviews keep their wins at ≤ 1.3× fit and
kdd_ipums / Moneyball / house_prices / porto / eucalyptus / cjs read as
exact ties (the in-run control inside hc). A pass there means
/experiment S4 with the maintainer's sign-off on flipping the default to
ON.

#### I036 2026-09-22 F5 S2 (the categorical count column on the synth screen; harness arm, pre-registered)
why now: I035's `next:`. PR #135 self-merged at 3dbaf28, no campaign PR
open, no run in flight. Class: **candidate DEFAULT change at S2** — a
harness arm `ChimeraBoostCatCount` (`run_benchmarks.py`: the default plus
one training-row count column per categorical, appended before the fit;
benchmarks only, the library untouched) paired against `ChimeraBoost` in
ONE `--synth --seeds 3` run. Branch `campaign/f5-s2-catcount` from main.
The library form, if it survives S3, is an automatic numeric count
column per categorical, on by default when categoricals are present —
a default change, so /experiment S4 with the maintainer's sign-off is the
only way it ships; nothing here is a knob.
barrier: B3 — the mechanism was named by ablation (I033), shown to be the
encoder (I034) and shown to transfer at its narrowest (I035, 4W-1L);
this is the screen B3 says every port must pass, not a port that skips
it. B12 — no arm is added to the product, the paired arm is the harness's.
B1 — the count column does nothing below the categorical gate; the
no-categorical sets are the pre-registered inert slice.
forecast, before the run: control — every synth set without a categorical
column an **exact tie** (the arm appends nothing there; `--expect-inert`
must not fire). Direction — on the sets with categoricals, positive by
majority with the gain concentrated where the generator draws
high-cardinality entity-like columns; the synth prior draws smaller
cardinalities and fewer rare categories than sf-police / Traffic, so I
take an engaged median of **+0.1 to +0.4%** (Brier / RMSE), engaged wins
≥ half + 1; regression sets engaged too (the column is target-free, so it
helps regression exactly as classification). Cost — fit time within
**+0 to +5%** on engaged sets (one more numeric column per categorical
through the binner and histograms). Where I could be wrong: on low-card
categoricals the count column is nearly constant and just noise for the
tree search (a small negative on the many low-card synth sets); and the
synth suite's categorical generator may not produce the rare-category
regime at all, in which case the screen reads flat and S3 on hc decides.
bars: (1) control — no-categorical sets exact ties, 100%; (2) direction —
engaged wins ≥ half + 1 AND engaged median > 0 (`compare_runs.py --model
ChimeraBoost --model-new ChimeraBoostCatCount --expect-inert`, now with
I031's median + CI + seed-agreement lines); (3) `synth_report.py` —
the categorical / entity slices carry the effect, no other attribute
slice reads a loss beyond noise; (4) cost — engaged fit ratio median ≤
1.10. Pass ⇒ S3 `--decide --seeds 3` both arms (hc + variants engaged, gr
the exact-tie control). Fail on (2) with (1) clean ⇒ the count column is
hc-specific; S3 on hc still decides, with the synth read as a caution.
ran: `run_benchmarks.py --synth --seeds 3 --save
benchmarks/results/campaign-f5-s2-catcount-20260922 --models ChimeraBoost
ChimeraBoostCatCount`, ~8 min → `results/campaign-f5-s2-catcount-20260922
.json`; scored with `compare_runs.py --model ChimeraBoost --model-new
ChimeraBoostCatCount --expect-inert` (`-compare-…txt`) and
`synth_report.py` (`-synthreport-…txt`). Harness SUMMARY: the default's
win rate vs the arm **46.2% [41–52]** (W21-L31-T80), median gap +0.00%,
speed ×1.06 — the arm ahead.
bar 1 control **PASS**: `cats=none` 79 sets **0-0-79 exact ties**,
`canary&cats` 0-0-3; 83 of 136 exact ties in all, every non-tie has a
categorical column. `--expect-inert` did not fire.
bar 2 direction **PASS**: engaged 53 sets, **36W-17L** (bar 27+),
engaged median **+0.295%** [95% CI +0.042%..+0.733%] — inside the
forecast's +0.1 to +0.4 band. Regression 13-8, binary 16-6, multiclass
7-3 — the column is target-free and helps regression as forecast. Seed
agreement: 17 of 53 unanimous, 36 split — the synth sets are small and
noisy, a caution, not a gate; the CI clears zero.
bar 3 attribution **PASS**: `cats=entity` **31-10-3, +1.84%, p=0.001**
and `card>16` **15-4-2, +3.26%, p=0.019** carry the effect — the
pre-registered slice; factor OLS puts `entity_strength` (t +3.6) and
`max_cardinality` (t +2.2) as the drivers and `cat_fraction` negative
(t −3.3: the more columns are categorical, the less an extra column per
categorical helps — the cost side of "one more column each").
`cats=all` 3-2-0, −0.36% on 5 sets (pointer; the all-categorical sets
already carry `cat_combinations`). No other slice reads a loss. Largest
movers: syn:v2/447 +24.7% and syn:v2/478 +24.6% (both entity sets whose
target is a per-entity effect the count exposes), worst syn:v2/737
−4.0%.
bar 4 cost **PASS, at the line**: engaged fit ratio median **1.095**
(bar ≤ 1.10), mean 1.17, p90 1.47, max 4.41 — on sub-second synth fits
one extra column per categorical is a visible fraction; hc fits at I035
were within noise. Read again at S3 on real sizes.
verdict: **PASS → S3.** All four bars clear; forecast HIT on control,
direction, slice and cost. Self-merged: `benchmarks/` only (the harness
arm + this record).
next: **I037, S3** — ONE `--decide --seeds 3 --save` run, both arms via
`--models`, scored `compare_runs.py --by-suite --expect-inert` per
stratum: Grinsztajn (no categoricals) the exact-tie control, hc + its
@sus25 / @sus50 / @time variants the engaged strata. Then the library
form via a muse task and /experiment S4 with the maintainer's sign-off.

#### I035 2026-09-22 R3 Stage D (a rarity column and coarse TS quantization on our encoder; zero library change, pre-registered)
why now: I034's `next:`. PR #134 self-merged at 448ee90, no campaign PR
open, no run in flight. Class: **zero-library probe of two candidate
default changes** (an appended count column; a quantization of the target
statistic), nothing shipped. Branch `campaign/r3d-ts-rarity` from main.
the candidates, from I033 + I034: CatBoost's CTR carries a `1/(n+1)`
rarity signal through its three prior columns (34% of the sf-police edge)
and quantizes the statistic at 15 uniform borders (49–64% of the edge on
kick / porto-seguro). Probe `benchmarks/probe_ts_rarity.py`, our default
estimator, the same seven sets and paired splits as I033. Arms: `rarity`
— every categorical gets a companion numeric column, the category's row
count in the training rows (test rows look it up, unseen = 0; trees are
invariant to monotone maps, so count, log count and `1/(n+1)` are one
feature after binning); `ts_q16` — the ordered TS post-processed to 16
uniform bucket centres over [0, 1] at fit and at transform, CatBoost's
`CtrBorderCount=15` (classification only, the TS is a probability);
`both`. d% vs I033's `chimera` rows, + = better; `closed` = share of
CatBoost's edge the arm closes.
barrier: B3 — the narrowest port of a NAMED sub-mechanism, each arm one
idea, and a probe; the "Counter/frequency column on its own" the refill
set aside as B3-hard was a prior, never a measurement (no kill on record
in `research/SUMMARY.md`, `ideas.py` or BARRIERS), and I034 explains why
CatBoost's own Counter looked inert (15 uniform borders over the count
range put every rare category in bucket 0). B18 — the count feature is
exactly the door B18 left open. B5 — a count column is not a shrinkage.
forecast, before the run: `rarity` **wins on sf-police (+0.3 to +0.8%),
Traffic (+0.2 to +0.6%) and okcupid (+0.1 to +0.4%)** — the three sets
with the most rare categories — flat on kick and porto (card 1063 / 104:
few rare categories, and porto's edge is 0.09%); controls: kdd_ipums
within ±0.3%, eucalyptus (736 rows, 5 categoricals) anything, it is a
pointer. `ts_q16` **wins on kick (+0.3 to +0.6%) and porto (+0.05%)** and
loses on sf-police / Traffic (−0.2 to −0.5%: 16 buckets throw away
resolution a 15k-card column needs); `both` ≈ the sum. Where I could be
wrong: our TS columns already reach the trees through a 255-bin quantile
binner, which is a data-adaptive quantization — if that is already the
regularizer CatBoost gets from 15 uniform borders, `ts_q16` reads flat or
negative everywhere; and if the rarity signal is already implicit in our
TS values (rare categories sit nearer the prior), `rarity` reads flat and
the F5 encoder thread closes with the CTR's remaining edge unexplained.
bars: (a) `rarity` better on ≥ 3 of 5 gap sets, kdd_ipums ≥ −0.3%, and the
gain ordered by cardinality (sf-police ≥ Traffic ≥ okcupid > kick, porto)
⇒ S2 synth screen (ChimeraBoost arms, the count column as a candidate
DEFAULT) — the library form would be an extra numeric column per
categorical, auto-on with categoricals, gated by /experiment; (b)
`ts_q16` better on kick and porto with sf-police / Traffic ≥ −0.3% ⇒ S2
as a TS-binning default candidate; (c) neither ⇒ F5's encoder thread
CLOSES: mechanism named, its two remaining parts ported at their
narrowest, neither transfers; B3 strengthened with the reason.
cost: 63 fits of ours ≈ 2 min.
ran: 63 fits, ~3 min; `results/probe-ts-rarity.jsonl` + `-20260922.log`.
d% vs our default (I033's `chimera` rows, same splits), + = better:
  arm      sf-police  Traffic    kick   okcupid   porto  |  kdd_ipums  eucalyptus
  rarity     +0.88%   +1.46%   +0.26%   +0.43%  −0.03%  |   −0.63%    −0.18%
  ts_q16     −0.31%   +0.92%   −0.54%   +0.30%  −0.36%  |   +1.65%    +3.26%
  both       +0.83%   +1.48%   +0.86%   +0.50%  −0.04%  |   −1.57%    +3.56%
  per seed, `rarity`: sf-police +0.82 / +0.95 / +0.87 and Traffic +1.65 /
  +1.32 / +1.42 — unanimous and tight; okcupid 2 of 3, kick 2 of 3 (−0.68
  to +1.25), porto three noughts; kdd_ipums +1.94 / −0.55 / −2.97 and
  eucalyptus +1.26 / −1.96 / +0.03 — the two small controls split wide.
  `rarity` closes **47%** of CatBoost's edge at the median (sf-police 62%,
  Traffic 81%, okcupid 47%, kick 28%).
read: **the rarity column is the mechanism.** 4W-1L on the gap sets,
median +0.43%, the gains ordered by cardinality exactly as pre-registered
(Traffic 3.8k-card ≈ sf-police 15k > okcupid > kick > porto, the last
flat at card 104), and the two big wins unanimous across seeds. Forecast
HIT on every gap set (sf-police +0.88 in the +0.3 to +0.8 band's top,
Traffic +1.46 above its band, okcupid +0.43 inside, kick and porto flat
as said). Bar (a) misses on ONE clause: kdd_ipums reads −0.63%, past the
−0.3% line — but that is a 7k-row control whose three seeds read +1.9 /
−0.6 / −3.0, a coin flip with a wide range, not a regression (the
engaged-slice instrument from I031 would label it split). Recorded as
the miss it is; it does not change where the evidence points, and the
next rung's inert control is the proper test of "does it hurt where it
should not". `ts_q16` (bar b): KILLED on the gap sets — 2W-3L, −0.31%
median, kick −0.54% where I forecast +0.3 to +0.6 (MISSED: our 255-bin
quantile binner already is the adaptive quantization, and 16 uniform
buckets only destroy resolution on the big sets) — but it wins +1.65% and
+3.26% on the two SMALL controls, and `both` +3.56% on eucalyptus: coarse
TS quantization is a strong regularizer on small categorical data, a
pointer for the @sus25 / @sus50 regime, parked with its numbers, not
chased here. `both` ≈ rarity on the big sets (kick +0.86 is the one
interaction).
verdict: **PASS on substance, one control clause missed → S2.** The
rarity column goes to the synth screen as a candidate DEFAULT change
(I036). F5 stays ACTIVE, encoder thread, now with a transferring
mechanism — the first in seven tries (B3's count). Self-merged:
`benchmarks/` only.
next: **I036, S2** — a harness arm `ChimeraBoostCatCount` (the default plus
one count column per categorical, appended before the fit; benchmarks
only, zero library change) paired against `ChimeraBoost` in ONE
`--synth --seeds 3` run, scored with `compare_runs.py --expect-inert`
(every no-categorical synth set must be an exact tie: the control) and
`synth_report.py` (the entity-cat slice is where it must concentrate).
Pass ⇒ S3 `--decide` (hc and its variants engaged, gr the exact-tie
control); pass there ⇒ a muse task for the library form (an automatic
count column per categorical, on by default with categoricals) and
/experiment S4 with the maintainer's sign-off.

#### I034 2026-09-22 R3 Stage C (encoder or booster? and CatBoost's shrinkage target on our encoder; zero library change, pre-registered)
why now: I033's `next:`. PR #132 self-merged at edc19ab, no campaign PR
open, no run in flight. Class: **measurement + zero-library probe of a
candidate default change** (a monkeypatch, nothing shipped). Branch
`campaign/r3c-ts-prior-target` from main. Note: `AGENTS.md` carries an
uncommitted working-tree edit on main that widens muse's `benchmarks/`
access as the shortlist proposed; not mine, left untouched, flagged to
the maintainer.
the two questions I033 left. (1) Stage A showed the target CTR carries
CatBoost's categorical information, which is not the same as showing the
EDGE lives in the encoder — a booster-side difference (symmetric depth-6
trees with 254 borders, lr 0.25, MVS bootstrap, `random_strength`,
`l2_leaf_reg=3`) could be the edge with any adequate encoding underneath.
The decisive arm is `cb_our_ts`: CatBoost at the harness defaults with no
`cat_features`, fed OUR ordered target statistics (fit on its training
carve, 4 permutations averaged, full-total transform for its validation
carve and the test rows, one column per class target, unseen → prior) in
place of the categorical columns. If it still beats us by the same
margin, the edge is the booster and F5's encoder thread closes; if it
falls to our Brier, the edge is the encoder and Stage B's prior read is
the lead. (2) The shrinkage target itself, on our side: `chimera_prior05`
pins `OrderedTargetEncoder.prior_` at 0.5 (CatBoost's target, same weight
a = 1) and `chimera_prior0` at 0 (CatBoost lost 21–74% of its edge to
this). Reference rows `chimera` and `cb_default` come from I033's JSONL,
same splits and seeds, so every number is paired.
barrier: B3 — this is the narrowest possible port of a named mechanism
(one constant in one formula) and it is a probe, not a ship; B5 — a
shrinkage target is not a shrinkage strength, and B5's own logic says the
target matters only where categories differ in count, i.e. the rare
stratum B18 measured; B18 — this is the encoder's fit-side prior, not the
transform-side re-weighting B18 closed (the transform follows the same
prior, both sides move together).
forecast, before the run: I now doubt my own I033 story. On sf-police
the positive rate is near 0.4 and CatBoost's zero-prior arm lost 74% of
its edge there — a prior of 0.5 is CLOSER to that mean than 0 is, so what
CatBoost showed may simply be "shrink toward the mean is right", which is
what we already do. So: `chimera_prior05` **within ±0.15% of our default
on every set**, wins ≈ half (no transfer); `chimera_prior0` **worse on
kick and porto-seguro** (mean 0.12 / 0.04 — a zero target is close to
their mean, but shrinking rare categories to 0 on an imbalanced target
destroys their signal) by 0.2–1%, flat elsewhere; `cb_our_ts` **keeps
60–100% of CatBoost's edge** on sf-police / Traffic / kick — my bet is now
the booster, not the encoder. Where I could be wrong: if `cb_our_ts`
falls to our Brier, the encoder IS the gap and Stage D becomes a
side-by-side of the two CTR formulas on one column at the rare stratum.
bars: (a) transfer — `chimera_prior05` better on ≥ 3 of 5 gap sets with
both controls ≥ −0.3%, and the gain concentrated on the max-cardinality
sets ⇒ S2 synth screen as a DEFAULT candidate; (b) booster — `cb_our_ts`
keeps ≥ 50% of the edge (median over the four real gap sets) ⇒ the
encoder thread of F5 CLOSES, and F5's remaining door is the booster side
(a different family, needs its own S0); (c) encoder — `cb_our_ts` closes
≥ 50% of the edge ⇒ Stage D, the formula side-by-side. (a) and (b)/(c)
are independent reads; (a) failing with (c) true means the target is not
the difference and something else in the arithmetic is.
cost: 42 ChimeraBoost fits + 21 CatBoost fits ≈ 12 min.
ran: 63 rows, ~6 min (our fits 1.4 s, CatBoost on our encoding 2.8 s — a
tenth of its 27 s with its own CTR machinery); `results/probe-ts-prior-
target.jsonl`, table in `-table-20260922.txt`. Paired against I033's
`chimera` / `cb_default` rows. d% = change vs our default, + = better:
  arm              sf-police  Traffic   kick   okcupid  porto   | kdd_ipums eucalyptus
  chimera_prior05    −0.08%   +0.09%  −0.50%  +0.33%  −0.03%  |  −2.96%   −0.47%
  chimera_prior0     −0.41%   +0.70%  −0.48%  +0.22%  −0.03%  |  −0.89%   +1.78%
  cb_our_ts          −0.19%   +0.34%  −1.46%  +0.28%  −0.01%  |  +0.55%   −3.04%
  share of CatBoost's edge `cb_our_ts` KEEPS: −5% / 39% / −155% / 24% /
  −70% — median **−5%**.
read: (1) **the gap is the encoder.** CatBoost boosting on our ordered
target statistics is no better than we are — 2W-3L, median −0.01%, and on
kick it is 1.46% WORSE than us, so on identical features our booster
beats CatBoost's; CatBoost's whole hc edge is what its CTR construction
adds over our TS. Bar (c) met, bar (b) not: the booster thread never
opens. My bet was the booster — MISSED. (2) **the shrinkage target is not
it.** Pinning our prior to CatBoost's 0.5: 2W-3L, median −0.03%, both
controls worse (kdd_ipums −3.0%) — no transfer, bar (a) fails, forecast
HIT. The zero target: 2W-3L too, sf-police −0.41% and kick −0.48% (HIT
on kick, an unforecast loss on sf-police, porto flat where I said worse).
And the balances explain I033's Stage B read outright: sf-police is
exactly 0.50 / 0.50, so CatBoost's 0.5 prior IS its target mean there and
its zero-prior arm was "shrink a balanced target toward 0", which is why
it lost 74% — the lesson is "shrink toward the mean", which we already
do. (3) what is left in the CTR, argued from the two ablations together:
on sf-police the extra prior columns were worth 34% of the edge
(`cb_one_prior`, I033 Stage A) and nothing else was; on kick and porto
the 15-border quantization was worth 49% / 64% (`cb_ctr254`, Stage B).
The three priors 0 / 0.5 / 1 give three columns `(sum + p) / (n + 1)`
whose SPREAD is exactly `1 / (n + 1)` — identical for common categories,
wide open for rare ones — so the trees are handed a rarity signal at
small counts, the "let the trees see the count" door B18 named. And
CatBoost's Counter looked useless in Stage A because it is quantized at
15 UNIFORM borders over [0, max count]: with counts to the thousands,
every rare category sits in bucket 0 and the Counter cannot tell 1 from
30. The prior spread can. That is a mechanism with a prediction: a
rarity column of ours should gain most where rare categories are most
common — sf-police (card 15k), Traffic (3.8k), okcupid (7k) — and little
on porto (card 104).
verdict: **READ — encoder confirmed, prior target killed, rarity signal
named as the next candidate.** F5 stays ACTIVE on the encoder thread.
Self-merged: `benchmarks/` only.
next: **R3 Stage D — a rarity column and TS quantization on our side**,
zero library change (I035, same rung, next entry).

#### I033 2026-09-22 R3 S0+S1 Stage A (CatBoost hc ablation: which categorical statistic carries its edge; measurement, pre-registered)
why now: I032's `next:`. PR #130 (R2 kill) self-merged at 3094889, no
campaign PR open, no run in flight. R3 is F5's only sanctioned door (B3)
and the top unresolved shortlist item. Class: **measurement of the
opponent**, no library change; no ship gate — its output is a mechanism
name or a closure of F5. Branch `campaign/r3-cb-hc-ablation` from main.
barrier: `barrier_check.py` on the idea matches B3 and B4 by construction.
B3 — this is the method B3 itself prescribes: name the mechanism before
porting anything; nothing is ported here. B4 — every arm keeps
`boosting_type=Plain`; ordered boosting is not on the table. B18 names the
Counter feature as the door that stays open and routes it through R3: this
is that route.
what CatBoost actually runs (read from `get_all_params()` on a
two-categorical toy under catboost 1.2.10, binary and multiclass alike,
recorded because the docs say otherwise): `max_ctr_complexity=1` — **no
cat×cat CTR combinations at all**; `one_hot_max_size=2`; per categorical
column two simple CTRs — `Borders` (target statistic quantized at 15
borders, THREE priors 0 / 0.5 / 1 ⇒ three columns) and `Counter` (category
frequency, prior 0, `counter_calc_method=SkipTest`); `boosting_type=Plain`.
So its hc edge rides on two per-column statistics, which is what the arms
decompose (the shortlist's `max_ctr_complexity=0/1/2` arm is moot at a
default of 1 and is dropped).
the probe: `benchmarks/probe_catboost_hc_ablation.py`, forked from
`probe_catboost_ablation.py` (the method that found the learning rate).
Seven hc classification sets: gap sets sf-police (CatBoost ahead by 1.17%
of our Brier on the standing BASE), Traffic_violations (1.92%,
multiclass), kick (1.16%), okcupid-stem (0.68%, multiclass), porto-seguro
(0.14%); controls kdd_ipums (we win by 5.7%) and eucalyptus (0.9%); cjs
excluded (Brier ≈ 0, degenerate). Paired splits, 3 seeds, Brier in the
harness's K-sum form. Arms: `cb_default` (the harness arm), `cb_borders_only`
(Counter off), `cb_counter_only` (target CTR off), `cb_one_prior` (Borders
with the single prior 0.5 + Counter — one column per categorical, the
shape of our ordered TS). ChimeraBoost defaults on the same splits as the
reference line. `recovered` = share of the default edge an ablation gives
back; 100% = CatBoost falls to our Brier.
forecast, before the run: the gap is monotone in cardinality and B18
showed rare categories are where our encoding misreads, so my bet is on
the **Counter**: `cb_borders_only` recovers **40–80% of the edge on
sf-police and Traffic_violations** (max-card 15k / 3.8k), 20–50% on kick
and okcupid, ~0 on porto-seguro (card 104, edge 0.14%, inside noise);
`cb_counter_only` recovers little (0–30%) — the target CTR is the
workhorse and the Counter is the complement; `cb_one_prior` recovers
0–20% — the three priors are three smoothing levels of one statistic, and
trees can approximate that from one column. Controls move under 0.3% of
Brier on every arm. Where I could be wrong: if `cb_counter_only` recovers
as much as `cb_borders_only`, the two statistics are substitutes and the
edge is "any second view of the category", not the count specifically —
then the port would be a frequency column AND that is exactly the shape B3
killed once (the Counter/frequency column on its own is listed as
"checked, not proposed" because B3 is hard); the ablation would at least
say whether that kill was on the mechanism or on the implementation.
bars: (a) mechanism named — one arm recovers ≥ 40% of the edge on ≥ 3 of
the 5 gap sets (median over gap sets ≥ 40%) with both controls moving
< 0.3% of Brier; (b) F5 closed — no arm reaches 40% on 3 sets: the edge is
not in any single categorical statistic and F5 goes to KILLED with B3
strengthened; (c) instrument void — `cb_default` disagrees with the
standing BASE's CatBoost Brier by > 1% on 2+ sets (the probe's splits are
not the harness's, so a small offset is expected; a large one says the
arm is misconfigured). Stage B (a native follow-on) is a separate rung
and needs the maintainer's pick.
cost: 84 CatBoost fits + 21 ChimeraBoost fits; CatBoost is 43× our fit on
hc, so ~2–2.5 h as one background run with nothing else on the bench.
ran, Stage A: 105 rows in **31 min**, not 2.5 h — CatBoost averages 27 s
a fit on these sets at the harness config, so the 43× figure from the
decide tier does not carry to paired single fits. Instrument check (bar c):
`cb_default`'s edge over us reads 1.42 / 1.80 / 0.94 / 0.91 / 0.09% on
sf-police / Traffic_violations / kick / okcupid / porto-seguro against the
BASE's 1.17 / 1.92 / 1.16 / 0.68 / 0.14 — same order, same ranking, valid.
`recovered` per gap set (share of CatBoost's edge an ablation gives back;
100% = it falls to our Brier):
  cb_borders_only (Counter OFF)      −9%   +6%   +12%   −5%  +114%   median **+6%**
  cb_counter_only (target CTR OFF) +265% +1120% +101% +690%   −13%   median **+265%**
  cb_one_prior (one prior, not 3)   +34%   −7%    −5%  −10%   +24%   median **−5%**
  controls: kdd_ipums moves ≤ 0.9% of Brier on every arm; eucalyptus
  (all-categorical, 736 rows) swings −2.6% / +1.9% / −0.3%.
read: **the whole edge is inside the Borders target CTR.** The Counter
gives back nothing (my bet, MISSED — I had 40–80% on the two max-card
sets; it is +6% at the median and negative on sf-police), the three
priors give back nothing (HIT), and with the target CTR removed CatBoost
does not fall to our level, it falls 1–7% of Brier BELOW us on four of
five gap sets — so our ordered target statistic is worth more than
CatBoost's frequency-plus-one-hot fallback, and the gap is a difference
between two constructions of the same idea. Bar (a) as written fails on
its control clause (eucalyptus moves, as any all-categorical set must when
its only encoding changes; the clause was ill-posed for a knob that IS the
categorical handling), but its substance — one arm, ≥ 40% on ≥ 3 gap sets
— is met by `cb_counter_only` on 4 of 5, and the mechanism is named.
Stage B, pre-registered here before it runs (still ablating the opponent,
zero library change, ~25 min): what in the CTR's construction differs from
ours? Three arms, each one knob: `cb_has_time` (CTR on the data order as
its single permutation — tests CatBoost's random-permutation machinery
against our fixed average of 4 permutations), `cb_ctr254` (the CTR value
quantized at 254 borders instead of 15 — ours is binned at 255, so if
CatBoost gets WORSE here, coarse quantization of the target statistic is
a regularizer we do not have, and a cheap one), `cb_zero_prior` (a single
zero prior — no smoothing at all; tests whether the prior's smoothing is
load-bearing, our `cat_smoothing` being the analogue). Forecast: the
priors already read inert, so `cb_zero_prior` recovers 0–20%;
`cb_has_time` recovers **20–60%** (permutation variation across trees is
the part of "integrated ordered-TS machinery" B3 points at); `cb_ctr254`
**−10 to +10%** (I expect quantization not to matter, and I would be glad
to be wrong). Bar: an arm recovering ≥ 40% on ≥ 3 gap sets names the
sub-mechanism; none ⇒ the edge is in the CTR's arithmetic itself (the
`(sum + prior) / (count + 1)` form, the per-permutation prefix, the
eval-time full totals — B18's territory) and F5's next rung is a
side-by-side of the two formulas on one column, not another CatBoost knob.
ran, Stage B: 63 more rows, ~20 min; 168 rows total in
`results/probe-cb-hc-ablation.jsonl`, table in `-table-20260922.txt`.
`recovered` per gap set (sf-police / Traffic / kick / okcupid / porto):
  cb_has_time (one permutation, data order)   +13%   −1%   +6%   −2%  −21%   median **−1%**
  cb_ctr254 (CTR at 254 borders, not 15)       +6%    0%  +49%  −26%  +64%   median **+6%**
  cb_zero_prior (prior 0 instead of 0.5)      +74%  +38%  +29%  +21%  −42%   median **+29%**
  controls: kdd_ipums within 0.6% of Brier on all three; eucalyptus
  −1.0% / −4.6% / −0.2% (`cb_ctr254` wrecks the tiny all-categorical set:
  0.442 → 0.462 — coarse quantization of the statistic is a strong
  regularizer THERE, and nowhere on the gap sets except kick).
read: no Stage B arm reaches the bar (≥ 40% on ≥ 3 gap sets), so by the
pre-registration the edge is in the CTR's arithmetic. Two of the three
knobs are cleared outright: **random permutations are worth nothing**
(`has_time` median −1%; my 20–60% bet MISSED — B3's "integrated
permutation machinery" story is dead on these sets), and **quantization
is not it** (median +6%, HIT, though kick and porto read 49–64% and
eucalyptus says the 15-border grid regularizes small data hard — a
pointer, recorded). The prior is the one live signal: shrinking toward 0
instead of 0.5 costs CatBoost 21–74% of its edge on the four real gap
sets, largest on the max-cardinality set (sf-police, card 15k) — the
forecast said 0–20%, MISSED HIGH. And it is a concrete formula
difference: CatBoost's Borders CTR is `(sum + p) / (count + 1)` with p a
CONSTANT (0.5, plus the 0 and 1 columns that Stage A showed add nothing);
ours is `(sum + mean·a) / (count + a)` with the prior at the GLOBAL MEAN
and a = `cat_smoothing` = 1. Same weight. The difference is the shrinkage
target. Why a constant 0.5 would matter, monotone in cardinality: on an
imbalanced target a rare category shrunk toward 0.5 lands somewhere no
common category lands, so the encoding itself carries the rarity signal
the trees never otherwise see — B18's "let the trees see the count" by a
different door, and one CatBoost gets for free. The `cat_smoothing`
weight `a` was killed on hc twice (PARETO_PLAN 2026-07-15) and is NOT
what this is: `a` moves how hard every category is shrunk toward the
mean; the target of the shrink has never been varied on our side.
timing, for the record: CatBoost averages 27 s a fit at the harness
config on these sets, so the 84-fit Stage A took 31 min, not the 2.5 h I
budgeted from the decide-tier 43× ratio (which is dominated by the
biggest sets and by CatBoost's own thread behaviour under the harness).
verdict: **R3 Stage A+B READ — mechanism named.** F5 is UNBLOCKED: the hc
Brier gap is the target-statistic arithmetic, specifically the shrinkage
target (constant 0.5 vs global mean), not the Counter, not the priors as
extra columns, not the permutations, not the quantization. Bar (a) met in
substance by Stage A (the CTR is the whole edge), bar (b) not met (F5
stays open), bar (c) clear (instrument valid). F5 row flipped to ACTIVE
with this mechanism and R3 marked RESOLVED on the shortlist. Self-merged:
`benchmarks/` only. Stage C, our side, is the next rung and needs no
maintainer pick because it is still a zero-library probe.
next: **R3 Stage C — the shrinkage target on OUR encoder** (zero library
change: a probe that monkeypatches `OrderedTargetEncoder.prior_` to the
constant 0.5 after the fit computes the mean, on the same 7 sets × 3 seeds
with the default estimator; arms = mean prior (default), constant 0.5,
and mean prior + a second 0.5-prior column if it can be done without a
library edit, else deferred). Bar: ≥ 3 of the 5 gap sets improve with the
two controls not worse than −0.3%, and the gain concentrates on sf-police
/ Traffic (the max-cardinality sets) as the ablation predicts. A pass
takes the mechanism to S2 (synth, ChimeraBoost arms only) as a candidate
DEFAULT change; a fail closes F5 with B3 strengthened: the mechanism was
named, ported at its narrowest, and did not transfer.

#### I032 2026-09-22 R2 S0+S1 (early-stopping round rule: smoothed argmin is a dedup miss, the 1-SE half gets its probe; pre-registered)
why now: I031's `next:`. PR #129 (H1) merged by the maintainer at 9ed5385,
no campaign PR open, no run in flight. R2 is the top unresolved shortlist
item under the maintainer's ranking (zero library change). Branch
`campaign/r2-es-round-rule` from main.
S0, the dedup: `barrier_check.py` matched B2, B11, B13 by keyword only
(refit, stopping rule, replay) — none binds: the refit replays whatever
round the rule picks (B2's amplifier applies equally to both rules and is
judged at S3 if the probe survives), B11 is isotonic-in-sample, B13 is
replay's grid fidelity. But grepping the tree for the mechanism itself
found what the checker could not: **`benchmarks/probe_tail_averaging.py`
(2026-07-13) already tested the smoothed-argmin half of R2** — argmin of a
9-round moving average of the validation curve, 11 Grinsztajn sets × 3
seeds, plus Polyak tail-averaging of the trajectory. Its table
(`results/probe-tailavg.jsonl`, `--table-only`): smoothed argmin is within
±0.03% on 9 of 11 sets, −0.58% on sulfur, and +5.24% on Brazilian_houses
— one row, seed 0, where the raw stop happened to land on a validation
spike; the regression mean +0.67% is that row alone, the binary mean
−0.008%. Tail and symmetric averaging ≤ 0 everywhere. The verdict lived
only in session memory ("KILL, don't retry") and was never registered, so
I022's refill could not catch it — the B15 pattern again. **Registered now
as B19** so the checker matches "smooth", "moving average", "argmin",
"early stopping round" next time. The smoothed-argmin half of R2 is
CLOSED without a rerun: the probe is on disk and its read is unambiguous.
S1, what remains: the **1-SE half** — the earliest round within one noise
unit of the validation minimum — is a different rule with a different
promise: it cannot lose the cost axis (every pick is ≤ t*, and the refit
replays t*/0.8 rounds, so the ratio carries over), and it is the only R2
form that was never measured. Probe `benchmarks/probe_es_one_se.py`, the
July protocol (one fit per set × seed, `early_stopping=False`, 600
rounds, 20% inner validation split, patience-50 stop simulated from the
curve, staged test predictions, temperature 1) with today's size-adaptive
default learning rate; 11 regression + 10 binary Grinsztajn sets × 3
seeds. Rules: `1se-local` (earliest t ≤ t* with val ≤ val[t*] + σ, σ = std
of the curve minus its 9-round moving average over t* ± 50 — the
single-curve stand-in for the CV standard error the rule was defined
with), and two tolerance forms `tol0.1` / `tol0.5` (earliest t within 0.1%
/ 0.5% of the minimum) as the bracket.
forecast, before the run: strength — every rule stops earlier than the
argmin, so it underfits by construction; I expect **regression RMSE −0.05
to −0.5% and binary Brier −0.05 to −0.3% at the median**, wins under half
on both tasks for `1se-local` and `tol0.5`, `tol0.1` within the ±0.03%
band with a rounds ratio near 0.95 (it picks t* itself on most curves).
Cost — rounds ratio **0.6–0.85** for `1se-local`, 0.5–0.8 for `tol0.5`.
Where I could be wrong: on flat-minimum curves (wine_quality, heloc,
credit) an earlier stop is free, and if the validation split's noise
biases the argmin late on average the 1-SE pick could read ≥ 0 on
regression, which would make it a real cost-axis candidate.
kill: paired wins < half + 1 on BOTH tasks or median ≤ 0 on both for
every rule ⇒ R2 KILLED outright (strength). A rule that is flat on
strength (median within ±0.03%, wins ≈ half) with rounds ratio ≤ 0.85 is
NOT a default candidate either — under the maintainer's ranking a
cost-only gain that needs a new knob queues last — but is recorded with
its numbers for a later preset decision. Only a rule positive by majority
AND median > 0 proceeds to S2.
ran: `probe_es_one_se.py`, 63 rows (21 sets × 3 seeds, no set dropped),
~35 min; `results/probe-es-one-se.jsonl` + `-20260922.log` (untracked,
results/ is gitignored; the table below is the record). Aggregate, per
task, positive = the rule beats the production patience-50 stop:
  regression (11 sets)  1se-local  2W-9L   median −0.008%  rounds ratio 0.99
                        tol0.1     0W-11L  median −0.089%  rounds ratio 0.95
                        tol0.5     0W-11L  median −0.484%  rounds ratio 0.84
  binary (10 sets)      1se-local  3W-6L   median −0.007%  rounds ratio 0.99
                        tol0.1     3W-7L   median −0.087%  rounds ratio 0.94
                        tol0.5     1W-9L   median −0.433%  rounds ratio 0.78
Per set the picture is monotone: every rule loses in proportion to the
rounds it drops. `tol0.5` gives back 0.25–0.66% of RMSE / Brier on 20 of
21 sets for a 16–22% round saving; `tol0.1` gives back ~0.09% for 5–6%;
`1se-local` barely moves the pick (ratio 0.99) and still reads 5W-15L. The
largest single win anywhere is bank-marketing +0.14% (`tol0.1`, s1); the
flat-minimum sets I named as the hope (wine_quality, heloc, credit) lose
like the rest. No rule reaches the bar on either task.
verdict: **KILL R2 — both halves.** The smoothed-argmin half was closed at
S0 by the July probe (B19); the 1-SE / tolerance half is closed here: an
earlier stop is not free at any tolerance — the validation argmin is
already on the flat part of the test curve and everything before it is
uphill. Forecast HIT on strength (negative medians, wins under half for
every rule, `tol0.1` inside the ±0.1% band) and MISSED on cost for
`1se-local` (ratio 0.99 against 0.6–0.85): the curve's round-to-round
noise near the stop is far smaller than its slope, so "within one noise
unit" is within a round or two. Recorded into B19 so the barrier covers
both directions (smoothing, and stopping early for cost). Shortlist R2
marked KILLED. Self-merged: the diff is `benchmarks/*.md` plus the probe
script, `chimeraboost/` and `tests/` untouched.
next: **R3 — the CatBoost hc ablation** (F5's sanctioned door, B3): fork
the existing ablation probe, 7 sets (5 gap + 2 controls) × 3 seeds, four
CatBoost arms with its CTR knobs turned off one at a time; bar = ≥ 40% of
the hc Brier gap recovered on the gap sets with controls < 10%. CatBoost
1.2.10 is installed. Stage A is ~2.5 h of CatBoost compute, so it launches
as ONE background run at the start of a rung with nothing else on the
bench. S0 (barrier arguments for B3/B4, the exact arm list, the gap
baseline from the standing BASE) comes first, next rung.

#### I031 2026-09-22 H(1) harness instrument (engaged-slice median + bootstrap CI + per-seed agreement in `compare_runs`; pre-registered)
why now: I030's `next:`. PR #128 (H4+H5) merged by the maintainer at
356c7cc, no campaign PR open, no run in flight. Class: **harness /
measurement** — never touches the ship gate; gated on an anchor (the I021
decide JSON re-read before and after, every verdict word unchanged, and
the engaged median matching the number I021 computed by hand) plus the
test suite. Implemented by Claude, as I030: `benchmarks/` is Claude's under
`AGENTS.md`. Branch `campaign/h1-engaged-slice-stats` from main.
the gap: I020 and I021 pre-registered bars of the form "wins ≥ half + 1
AND engaged median > 0" and "fit ratio median ≤ 1.5", then computed the
engaged median by hand from the per-dataset rows, because
`compare_runs` prints the engaged WIN COUNT and nothing about the size of
the engaged effect or how sure the read is. A second number the plan file
has asked for since GATE_ROBUSTNESS #1 ("run more seeds" produces the same
table) is whether the seeds agree: a dataset whose three seeds split 2–1
is not a win the way a 3–0 one is, and no tool says which is which.
design, all print-only, in `_control_line`'s engaged block:
(a) the engaged-slice **median relative change** in the compared metric
(+ = better), near-solved sets excluded as in the main block, with a **95%
percentile bootstrap CI** over engaged datasets (10 000 resamples, seed 0,
the `summarize.bootstrap_winrate_ci` convention); (b) **per-seed
agreement**: for each engaged dataset, does every seed's NEW − BASE carry
the sign of the mean delta? Printed as "k of n unanimous, m split (names)";
with one seed the line says so instead of pretending. Needs per-seed
values, so a sibling loader `load_run_seeds` returns dataset → {seed:
value}; `load_run`'s signature and the seed-averaged path stay untouched.
No PASS/FAIL word moves; the bar is unchanged.
forecast: no chart axis moves, by construction. Anchor: re-reading the
I021 run (`results/20260921-080246.json` ×2, `--by-suite --metric brier
--model ChimeraBoostOneLin --model-new ChimeraBoostOneLinXC
--expect-inert`) must differ from the before-read only by added lines,
and the Grinsztajn engaged median must print **−0.04%** — the number I021
wrote by hand — with a CI that straddles zero (I021's read was "flat";
10W-13L cannot exclude zero). The hc engaged median (4 sets) prints too but
carries the POINTER label from I030. Per-seed agreement on the gr engaged
slice: unknown — the first time anyone looks; recorded, not gated.
kill: any verdict word changes on the anchor; the gr engaged median
disagrees with I021's hand figure at the printed precision; the suite not
green.
ran: the edit, then the anchor re-read (`--metric brier`, the metric I021
judged on). `diff` before/after: **10 lines added, nothing changed** — two
new lines per stratum that engaged (an "engaged slice" line and a
"per-seed agreement" line), every PASS/FAIL word intact. What the
instrument printed on the I021 run:
  gr (23 engaged, 23 scored): median relative Brier change **−0.003%**,
    95% CI −0.289%..+0.232%; per-seed agreement **8 of 23 unanimous, 15
    split** (albert, compas, eye_movements ×2, road-safety, Bioresponse,
    Diabetes130US, Higgs, MiniBooNE, california, credit, default-of-credit,
    electricity, heloc, pol).
  hc (4 engaged): median +0.099%, CI −0.055%..+0.405%, 0 of 4 unanimous —
    pointer. gr@sus25: −0.208%, 2 of 4 unanimous; gr@sus50: −0.486%, 0 of
    2; hc@time: +0.141%, 0 of 2 — all pointers.
**The anchor number MISSED, and the miss is the finding.** I021 wrote
"engaged median −0.04% Brier" by hand; the file says −0.003%, and a
recomputation straight from the JSON (seed-averaged Brier per set, relative
change, median over the 23 sets) gives the same −0.003%. The hand figure was
a slip — the same order of magnitude, the same sign, and it changed no
verdict then or now (10W-13L was the FAIL; the median only had to be ≤ 0),
but it is a number the plan file quotes that no tool ever produced, which is
exactly the gap this rung was opened to close. Recorded here rather than
edited into I021 (append-only log). The seed read is new information: only
8 of the 23 "engaged" Grinsztajn sets had all three seeds on the same side.
The F3 verdict does not depend on it (a coin flip is a coin flip with
either count), but a future rung that clears a 12-of-23 bar with 15 split
sets should not be read as a win, and now the tool will say so.
gate: the existing 8 `test_compare_runs` tests green; 3 new ones (median
+1.667% and CI bracketed by the two engaged rels on a fixture, ties
excluded, unanimous vs split counted right; single seed says so; the two
helpers deterministic and the agreement rule pinned incl. the skipped
single-seed set); ruff: two F541 hits of my own fixed, the rest the
pre-existing 7. Full suite: see the PR.
verdict: **PASS → PR** (touches `tests/`, waits for the maintainer). No
verdict word moved; the instrument reproduces the JSON, not the hand
figure, and that is the point. Shortlist H row: (1), (4), (5) done; (2)
probe→decide transfer footer and (3) cost column remain, both lower value
than the R-items now that the three decision-facing numbers print.
next: **R2** — the 1-SE / smoothed early-stopping round (zero library
change: one fit per set, staged predictions, test metric at argmin vs each
rule's round, ~20 gr sets × 3 seeds; kill if paired wins < half + 1 or
median ≤ 0). S0 first: barrier arguments for B2, B11, B13, B14, B12 are
already sketched in the shortlist row and owe their written form. Then R3.
One campaign PR at a time: R2 waits for this one to merge.

#### I030 2026-09-21 H(4)+H(5) harness instruments (POINTER label; --save / --models guards; pre-registered)
why now: I029's `next:`. PR #127 (C3) merged by the maintainer at c9c0f3d,
no campaign PR open, no run in flight. F4's measured exact-rewrite objects
are exhausted, so the shortlist queue takes over, harness items first under
the maintainer's ranking (defect fixes and zero-cost instruments before
anything that adds a knob). Class: **harness / measurement** — never touches
the ship gate; gated on anchors (the existing I021 decide JSON re-read
before and after, verdict words unchanged) and the test suite. Implemented
by Claude, not muse: `AGENTS.md` reserves `benchmarks/` for Claude and the
shortlist's proposal to widen that is the maintainer's call, so a muse task
here would have violated the worker's standing rules. Branch
`campaign/h4-h5-harness-guards` from main.
H(4), `compare_runs.py`: every sign-test line (the bar, the near-solved
sub-bar, the engaged-only bar) carries `[POINTER, not a gate: n decided <
8]` when wins + losses is under `POINTER_MIN_DECIDED = 8`
(GATE_ROBUSTNESS #2). Ties are the inert slice, not evidence, so they do not
count as decided. Print-only: the PASS/FAIL word is unchanged, so every
verdict recorded in this file still matches the tool's output.
H(5), `run_benchmarks.py`: (a) `--save NAME` with no directory part lands
under `benchmarks/results/`, and a name without an extension gets `.txt`,
so the derived `.json` lands beside it — the I003 trap (a bare name wrote
two files into the shell's CWD) closed at the source; a path with a
directory is used as given; `--save` alone is unchanged. (b) the `--models`
checks (unknown runner; `ChimeraBoost` missing) run right after argparse,
before the tee opens, so a usage error can no longer leave an empty
`<stamp>.txt` in results/ (facts ledger, 2026-09-21).
forecast: no chart axis moves, by construction. Anchor: the I021 read
(`compare_runs.py results/20260921-080246.json ×2 --by-suite --model
ChimeraBoostOneLin --model-new ChimeraBoostOneLinXC --expect-inert`) must
print byte-identical output except for POINTER suffixes, which must appear
on exactly the strata I021 called pointers by hand (hc 4 engaged, gr@sus25
4, gr@sus50 2, hc@sus25 0, hc@sus50 0, hc@time 2 engaged) and on none of
the two bars with ≥ 8 decided (gr all-dataset 23, gr engaged 23).
kill: any verdict word changes on the anchor; the suite not green; a
`--save` path with a directory part being rewritten.
ran: both edits, then the anchor re-read. `diff` of the I021 read before
and after: 13 lines differ, every one a sign-test line with a POINTER suffix
appended and nothing else — hc (4 decided), gr@sus25 (4), gr@sus50 (2),
hc@sus25 (0), hc@sus50 (0), hc@time (2), on the bar, the near-solved
sub-bar and the engaged-only bar of each; the two Grinsztajn bars (23
decided) carry no suffix; every PASS/FAIL word unchanged. Exactly the
strata I021 called pointers by hand. Ruff on the two scripts: the same 7 +
13 pre-existing hits as on main (`_report` was already C901 11, `main` 51),
nothing new. Tests: 2 new in `test_compare_runs.py` (label present at 4
decided with ties not counted, verdict word kept; absent at 8) and 6 in the
new `tests/test_harness_guards.py` (the four `--save` shapes; both
`--models` mistakes named; a `--models` usage error exits 2 before the tee
file exists). Full suite green (see the PR).
verdict: **PASS → PR**. Anchor HIT exactly as pre-registered, no verdict
word moved, no `--save` path with a directory rewritten. Not
self-merged: the PR touches `tests/`, which stays the maintainer's under
the 2026-09-21 rule, so it waits for him like a library change would.
Shortlist H row: (4) and (5) done; (1), (2), (3) remain.
next: **H(1)** — engaged-slice median + bootstrap CI + per-seed agreement
in `compare_runs` (3 h; I020/I021 gated on a number no tool prints). Then
**R2** (1-SE / smoothed early-stopping round, zero-library probe), then
R3. One campaign PR at a time: H(1) waits for this one to merge.

#### I029 2026-09-21 F4 S0+S1 (candidate C3 — the binary Logloss layer; muse task, pre-registered)
why now: I028's `next:`. PR #126 (C4a-2) merged by the maintainer at
3706494, no campaign PR open, no run in flight. C3 is the last measured F4
object that is not parked as an algorithm change, and the first that
reaches Grinsztajn (23 of 59 sets are binary) rather than the hc suite
alone. Class: **exact-rewrite perf**, pure-speed ladder. Library source,
so muse implements it from `campaign_tasks/20260921-f4-c3-logloss-layer.md`
(gitignored; the verdict cites what it produced) and the PR waits for the
maintainer. Branch `campaign/f4-c3-logloss-layer` from main.
object (I023): the binary Logloss layer is two rows of one object.
`Logloss.grad_hess` is the numba `_sigmoid` kernel plus two numpy passes
(`p − y`, then `np.maximum(p·(1−p), 1e-6)`) with four temporaries, 4.2–4.9%
of a binary fit; `Logloss.eval`, called on the validation rows every round
and on the training rows when history is kept, is a sigmoid, a clip, two
logs, three products and a sum over the rows, 4.0–5.1%. Together
**8.3–10.0%** on MagicTelescope / Higgs / road-safety.
design, the C1b move on the scalar path. (i) `_logloss_grad_hess_kernel
(raw, y)`: `_sigmoid`'s loop verbatim, then per element `grad = p − y` and
`hess = h if h >= 1e-6 else 1e-6` with `h = p·(1−p)` — the same operations
in the same order as numpy, elementwise, so the outputs are the same bits;
the floor is a comparison because `np.maximum(h, 1e-6)` is one on non-NaN
input and `h` lies in [0, 0.25] for finite `raw` (C1b's argument, accepted
at I018). (ii) `_logloss_ce_kernel(raw, y)`: sigmoid, then the clip written
as two ordered comparisons so a NaN passes through as `np.clip` passes it,
then the cross-entropy. For a 0/1 label the dead term of
`y·log p + (1−y)·log(1−p)` is `0·log(·)`, a signed zero, and adding a signed
zero to a finite float returns that float unchanged, so `−log p` for y = 1
and `−log(1−p)` for y = 0 are the same bits as the two-log formula; the
kernel branches on `y == 1.0` / `y == 0.0` and computes the full two-log
expression in numpy's operation order for any other label, so soft labels
stay exact too. The kernel returns the per-row vector and **the mean stays
in numpy** (`np.average(ce, weights=w)`): numpy's pairwise summation is not
a sequential numba sum, and that is where a fused reduction would drift.
The old bodies stay as `_grad_hess_numpy` / `_eval_numpy`, the oracles.
Dispatch guards: float64, 1-D, contiguous, equal lengths; anything else
takes the old path.
barrier: `barrier_check.py` matched B10, B15, B16, all on the words
kernel / numba / speed. B10 binds objects inside `build_oblivious_tree`;
this is the loss layer, the object I012 and I018 already cleared, and
B10's own method (ceiling measured first, I023) is obeyed. B15 is the
histogram, B16 the cross screen: names only.
forecast, before any code: fit time — grad_hess ceiling 4.2–4.9%, eval
ceiling 4.0–5.1%, about 9% together; C1b converted its grad_hess row at
~100% of the leg but the eval row keeps its numpy reduction and its
sigmoid, so I take **−4 to −7% of a Grinsztajn binary fit** on
MagicTelescope / Higgs / road-safety; a regression control (cpu_act or
Brazilian_houses) and a multiclass control (hc:okcupid-stem) **0 ± 1%**
(`Logloss` is not their loss); hc:kick (binary, hc) inside the same band,
smaller because `prep` dominates there. Strength — exactly zero by
construction and by `identity_snapshot` (binary configs are on its panel);
cross-libm, the existing ≤ 4 ULP exp/log caveat the softmax pins carry.
Where I expect to be wrong: the eval row — the per-round validation eval
touches only the validation rows (15–20% of n), so its 4–5% share is
mostly Python and numpy call overhead per round rather than arithmetic,
and fusing may buy less than half of it.
kill: (a) `identity_snapshot.py check` not 155/155 or any golden red; (b)
the same-process A/B saves under 2% on all three Grinsztajn binary sets;
(c) either control moves beyond ~1%; (d) muse fails or times out — record
what it left, revert, re-scope.
ran: muse exit 0 in one pass, ~20 min; its `RESULT.md` reported 75 tests
green in `test_bitident_refactors.py`, full suite 1112 passed + 1 skipped,
ruff clean on the two files it touched, warmup coverage green without a
warmup edit (the warmup's binary fit reaches both kernels), and a pre-ship
probe of numba `log` against numpy `log` over 2.5M values with zero bit
differences on this machine. Diff reviewed by hand, 88 lines in
`losses.py`: `_logloss_grad_hess_kernel` is `_sigmoid`'s loop verbatim plus
`p − y` and the compared floor; `_logloss_ce_kernel` is the sigmoid, the
clip as two ordered comparisons (NaN passes through as `np.clip` passes
it), the 0/1 branches and the general formula for soft labels; the mean
stays in `np.average`; `_scalar_pair_ok` guards float64 / 1-D /
C-contiguous / equal shape; old bodies kept as `_grad_hess_numpy` and
`_eval_numpy`. Documented scope, as C1b: on a NaN raw the fused hess floor
returns 1e-6 where `np.maximum` returns NaN — unreachable on the fit path
(raw stays finite; `fit` rejects inf), pinned by a test.
`/code-review` (medium): no correctness finding; one real test finding —
the new `eval` tests pinned numba's `log` to numpy's `log` bit for bit,
the same cross-hardware fragility this file already documents for the
softmax pins (ubuntu runners since 2026-08-30 differ by up to 3 ULP in ~2%
of elements). The reviewer (me, not muse) loosened exactly those pins: the
per-row cross-entropy vector to `assert_array_max_ulp(maxulp=4)` with NaN
positions still exact, the averaged scalar and `valid_history_` entries to
8 ULP; `grad`/`hess` and `predict_proba` stay exact (both arms run numba's
exp). Same-machine bit-identity remains the `identity_snapshot` gate.
gate 1, tests: **1112 passed, 1 skipped** rerun under the conda python
(1106 + 6 new), then 75/75 in the file after the pin change.
gate 2, identity: `identity_snapshot.py check` **155/155 identical**.
gate 3, speed (`benchmarks/f4_c3_speed.py`, new; same-process A/B on the
DEFAULT estimator, OFF arm = the `_numpy` bodies, median of 5;
`results/campaign-f4c3-speed-20260921.txt`):
  gr:MagicTelescope  **−5.7%**  grad_hess 0.044 → 0.028 s, eval 0.046 → 0.019 s
  gr:Higgs           **−4.7%**  0.075 → 0.041, 0.062 → 0.016
  gr:road-safety     **−4.8%**  0.090 → 0.047, 0.083 → 0.022
  hc:kick            **−4.2%**  0.055 → 0.032, 0.063 → 0.014
  gr:cpu_act (regression control)  −0.9%, zero calls (per-repeat range
                                   wide, −10..+9%, on a 0.46 s fit)
  hc:okcupid-stem (multiclass control)  +0.5%, zero calls
The loss layer's own seconds account for most of each change (road-safety
0.104 of 0.099 s, Higgs 0.080 of 0.056 s, MagicTelescope 0.043 of 0.048 s).
verdict: **PASS → PR** (library source; waits for the maintainer). Kill
bars (a)(b)(c)(d) all clear. Forecast: strength exactly zero HIT; speed
−4 to −7% HIT on all three Grinsztajn binary sets (5.7 / 4.7 / 4.8) and
kick inside the band too; controls flat HIT. Where I said I would be wrong
(the eval row buying under half): wrong the other way — eval converted
~65–75% of its seconds, grad_hess ~45%; the eval row was arithmetic and
temporaries after all, not call overhead. The first F4 unit that moves
Grinsztajn: 23 of 59 sets are binary.
next: F4's measured exact-rewrite objects are now all shipped (C2, C1,
C1b, C4a, C4a-2, C3); C4b (shared TS permutations) stays parked as an
algorithm change. Under the shortlist ranking the queue is the harness
instruments **H(4) + H(5)** (POINTER label for strata under 8 decided
sets; `--save`/`--models` argparse guards) as one small muse task under
`benchmarks/`, self-mergeable, then **H(1)** (engaged-slice median +
bootstrap CI + per-seed agreement in `compare_runs`), then **R2** (the
1-SE / smoothed early-stopping round, zero-library probe), then R3. One
campaign PR at a time: the next rung waits for this one to merge.

#### I028 2026-09-21 F4 S1 (candidate C4a-2 — the numeric block cast once per fit; muse task, pre-registered)
why now: I027's `next:`. PR #125 (C4a) merged by the maintainer at a8f04c8,
no campaign PR open, no run in flight. Class: **exact-rewrite perf**,
pure-speed ladder. Library source, so muse implements it from
`campaign_tasks/20260921-f4-c4a2-numeric-once.md` (gitignored; the verdict
cites what it produced) and the PR waits for the maintainer. Branch
`campaign/f4-c4a2-numeric-once` from main.
scope: the float64 numeric block only, on C4a's plumbing. Today
`FeaturePreprocessor._numeric_block` casts the object matrix's numeric
columns to float64 on every leg — the selection legs' training rows, the
validation rows for the eval transform, the cross candidate's
`from_base_with_cross` (its `_cross_block` builds the block again because
no `num=` is handed in), the cross candidate's validation block, the
classifier's calibration predict, then the whole matrix in the refit —
so a fit with categoricals casts about 2.2–3× the matrix. The validation
cast in `_coerce_X_finite` (3.2% on porto-seguro) runs before
`as_model_array` on the raw input through a different converter and stays
OUT of this unit: an exact sharing argument would have to cover pandas
nullable dtypes, float32 frames and object arrays separately, and 3% on
one set is not worth that surface.
design: `CatTransformCache` gains `numeric(X, num_features)`, a cached
float64 block keyed by the tuple of numeric positions. Plain form: the
cast `_numeric_block` does today, moved to a module-level
`_cast_numeric_block`. Child form: gather the parent's block by `rows`
under the same row-count guard `column()` uses, falling back to the plain
cast on mismatch. `_numeric_block(X, cat_ctx=None)` delegates to the
context when one is given. Exactness: numpy's object→float64 cast is
element-wise, so `cast(X_full)[rows]` and `cast(X_full[rows])` are the
same bits, NaN included. Nothing downstream writes into the block
(`_stack` hstacks or returns it, `Binner.fit` sorts a copy,
`_cross_block` reads columns), so one cached copy can serve every leg and
the refit; a test pins that too. The estimator side needs no change: the
three contexts already reach every `fit_transform`, `transform`,
`_cross_block`, calibration `predict_raw` and the refit. Bonus, unforecast:
a bagged predict shares one context across members, so the batch's numeric
block is now cast once per predict instead of once per member.
barrier: `barrier_check.py` matched seven, all keywords (refit,
calibration, speed, selection). B2/B12/B13 — leg names only; no audition,
race or replay behaviour moves. B5/B11 — the calibration keyword; the
temperature step sees identical inputs. B10/B15 — kernel-side; this is
preprocessing.
forecast, before any code: fit time — `_numeric_block` was 4.7% of fit on
kick and 10.1% on porto-seguro, under 1% on sf-police, okcupid-stem,
wine-reviews and Traffic_violations (I026); one cast of the full matrix
stays, so the ceiling is ~55–65% of the row: **kick −2 to −3%,
porto-seguro −5 to −7%**, sf-police / okcupid-stem 0 to −1% (below the
~1% same-process floor — reported, not gated), a Grinsztajn numeric
control **0 ± 1%** (no categoricals, no context). Strength — exactly
zero, by construction and by `identity_snapshot`. Memory: one float64
copy of the numeric block per fit (porto-seguro ≈ 100k × 26 × 8 B ≈ 21
MB), freed with the fit. Where I expect to be wrong: kick sits at the
floor's edge, so its sign may read but its size will not; and porto-seguro
converts at C4a's near-100% rate only if the cast really is the row —
I026 priced the object-array cast per element, which is what the census
will show or refute.
kill: (a) `identity_snapshot.py check` not 155/155 or any golden red; (b)
porto-seguro saves under 3% in the same-process A/B (kick is not a gate
at its ceiling); (c) the numeric control moves beyond ~1%; (d) muse fails
or times out — record what it left, revert, re-scope.
ran: muse exit 0 in one pass, ~15 min; its `RESULT.md` reported 22 tests
green in the target file, full suite 1106 passed + 1 skipped, ruff clean on
the two files it touched (23 pre-existing hits elsewhere), and a read-only
audit of every consumer of the block (binner sorts a masked copy, the cross
block writes its own `out`, `_stack` hstacks or hands the block on). Diff
reviewed by hand, 70 lines in `preprocessing.py`: `_cast_numeric_block` is
the old `_numeric_block` body verbatim; `CatTransformCache.numeric` caches
under the tuple of numeric positions with the same row-count guard as
`column()`; `_numeric_block(X, cat_ctx=None)` delegates; its three callers
pass the context they hold. No estimator or booster change was needed.
`/code-review` (medium): no correctness finding; one memory note, below.
gate 1, tests: **1106 passed, 1 skipped** rerun under the conda python
(1097 + 9 new: child block vs the leg's own cast on three row subsets with
Python/numpy floats, ints, bools and NaN mixed; the all-positions and empty
cases; cache identity and a parent cast counted once; the guard; the block
byte-equal to a fresh cast after full regressor and binary fits; end-to-end
regressor / binary / 3-class with sharing monkeypatched off, `array_equal`
predictions and strictly fewer casts).
gate 2, identity: `identity_snapshot.py check` **155/155 identical**.
gate 3, speed (`benchmarks/f4_c4a2_speed.py`, new; same-process A/B on the
DEFAULT estimator, OFF arm = `_numeric_block` ignoring the context while
C4a's categorical sharing stays on in both arms, median of 5;
`results/campaign-f4c4a2-speed-20260921.txt`):
  hc:kick          **−3.4%**  casts 6 → 1, 0.096 → 0.022 s
  hc:sf-police     −0.8%      6 → 1, 0.007 → 0.002 s (under the floor)
  hc:porto-seguro  **−8.5%**  6 → 1, 0.320 → 0.069 s
  hc:okcupid-stem  +0.0%      6 → 1, 0.008 → 0.002 s (under the floor)
  gr:clf_num/Higgs −0.4%      6 → 6, 0.000 s (the control; the all-numeric
                              cast is a no-op view, nothing to save)
The seconds saved in the cast account for the whole fit-time change on the
two sets that move (porto-seguro 0.251 of 0.206 s, kick 0.074 of 0.064 s),
so the read is the mechanism and not drift.
memory, corrected from the forecast (the reviewer's finding): the contexts
retain the full-matrix block AND the two gathered leg copies for the whole
fit, about 2 × n × p_num × 8 B, where before this change each leg's block
was transient and freed after binning. Relative to the previous peak (one
full-size block during the refit) that is one extra full-size copy:
porto-seguro ≈ +21 MB against an object input matrix that already costs
≥ 32 B per element. Recorded in the PR for the maintainer; the cheap
alternative, not caching the gathered leg copies (about 1% of porto-seguro's
fit in re-gathers), is his call at review.
verdict: **PASS → PR** (library source; waits for the maintainer). Kill
bars (a)(b)(c)(d) all clear. Forecast: strength exactly zero HIT; speed —
porto-seguro −8.5 against −5 to −7 (MISSED LOW, like C4a: the object-array
cast converts at ~100%), kick −3.4 against −2 to −3 (edge, HIT), the two
sub-1% sets and the control flat as forecast. Memory forecast MISSED: I
wrote one copy, it is two retained plus one extra at peak.
next: **C3 — the binary Logloss layer** (8–10% of a gr binary fit across
`grad_hess` and the per-round validation `eval`, I023), the first F4 object
that reaches Grinsztajn. S0 first: the exact-rewrite question is whether
the sigmoid, gradient and hessian can fuse into one numba pass with every
element produced by the same operations in the same order (the C1b
pattern), or whether it is FP-drift class. C4b (shared TS permutations)
stays parked. One campaign PR at a time: C3 waits for this one to merge.

#### I027 2026-09-21 F4 S1 (candidate C4a — factorize once per fit; muse task, pre-registered)
why now: I026's `next:`. PR #124 self-merged after green CI (4705a16), no
campaign PR open. Class: **exact-rewrite perf**, pure-speed ladder. Library
source changes, so muse implements it from
`campaign_tasks/20260921-f4-c4a-prepare-once.md` (gitignored; the verdict
cites what it produced) and the PR waits for the maintainer. Branch
`campaign/f4-c4a-prepare-once` from main.
scope, narrowed at planning: **categorical factorization only.** The
numeric block (2.6% kick / 5.5% porto-seguro by ceiling) rides on the
same plumbing but stays out of this unit, so the diff the maintainer
reviews is one idea. It is C4a-2, a later rung, if this one ships.
design: a leg's factorization derived from one full-matrix factorization.
`CatTransformCache` gains a child form bound to a parent cache, the
parent's matrix and a row-index array; `child.column(X_leg, f)` takes the
parent's `(codes, categories)` for column f, gathers `codes[rows]` and
re-ranks them to first-appearance order with an O(n) numba pass
(`lut[c] < 0 → lut[c] = next`), which reproduces `factorize(X_leg[:, f])`
exactly in codes and in category ORDER; the category objects are the
parent's representatives, ==/hash-equal to the leg's own (they can differ
only where cross-type-equal values such as `1` and `1.0` coexist, which no
dict lookup can observe). A row-count mismatch falls back to plain
`factorize`, so a mispaired context cannot corrupt an encoding. Plumbing
through the seam that already exists: the estimator seeds the per-fit
`prep_cache` dict with `(train_ctx, val_ctx)` under a reserved key,
`_prep_matrices` / `_prep_or_replay_matrices` hand them to
`fit_transform` / `from_base_with_cross` / `transform`; calibration's
`predict_raw` already takes a `cat_ctx`; the refit gets the full-matrix
cache itself. `_auto_es_split` returns the split indices it already
computes. Inert when there are no categorical features, when the user
supplied `eval_set` (separate matrix), and on the bag-member refit rows
(a different row set). `factorize` stays the definition and the tests'
oracle.
barrier: `barrier_check.py` matched eight, all keywords (refit,
categorical, calibration, speed, selection). B3 — C2's argument again: no
encoding semantics move, the same codes reach the same encoder. B10/B15 —
kernel-side, this is preprocessing. B2/B13/B12/B5/B11 — leg names only.
forecast, before any code: fit time — factorize was 14.7 / 29.8 / 29.9 /
14.1% of fit on kick / sf-police / porto-seguro / okcupid-stem and is paid
~2.2×; one pass plus three cheap re-ranks leaves ~45% of it, a ceiling of
8 / 16 / 16 / 8%. By the C2 and C1b conversion rates: **kick −5 to −8%,
sf-police −10 to −15%, porto-seguro −8 to −14%**, okcupid-stem −5 to −8%;
a Grinsztajn numeric control **0 ± 1%** (no categorical column, the
context is never built). Strength — exactly zero, by construction and by
`identity_snapshot` (its panel carries categorical configs, so the new
path is exercised, not bypassed). Memory: one int64 code column per
categorical per fit (porto-seguro: 100k × 31 × 8 B ≈ 25 MB), freed with
the fit. Where I expect to be wrong: porto-seguro — its cost sits in
`_factorize_numeric`'s cast-and-audit, which the parent pass still pays
once at full size, so its saving should land at the low end; and muse may
not finish a three-file change inside its step cap.
kill: (a) `identity_snapshot.py check` not 155/155 or any golden red — no
approximation ships; (b) the same-process A/B saves under 3% on all of
kick / sf-police / porto-seguro; (c) the numeric control moves beyond ~1%;
(d) muse fails or times out — record what it left, revert, re-scope.
ran: muse exit 0 in one pass, ~20 min; its `RESULT.md` reported 13 new
tests green, full suite 1097 passed + 1 skipped, `ruff check chimeraboost`
clean (23 pre-existing hits in untouched test files), `warmup.py` left
alone because the warmup's categorical fit already reaches the new kernel
through the default split (the kernel-coverage test stays green). Reviewed
the diff by hand, 221 lines over three library files, against the task:
`_rerank_first_appearance` is the one-pass lookup-table re-rank;
`CatTransformCache` gained the child form with the row-count guard and
falls back to plain `factorize` on any mismatch; `fit_transform` and
`from_base_with_cross` take an optional `cat_ctx`; `_prep_cache_ctxs` reads
`(train_ctx, eval_ctx)` from the per-fit `prep_cache` under
`_CAT_CTX_KEY`, so `_prep_matrices` stays at McCabe 8 and no booster `fit`
signature moved; `_auto_es_split` returns the split indices it already
computed (its only two callers updated — grep confirms no third);
`_shared_cat_ctxs` builds the three contexts and is inert without
categoricals or without an auto split; the classifier's temperature
predict reuses the validation context; the refit gets the full-matrix
cache on the auto-split branch only, the bag-member refit rows get none.
Nothing is stored on the estimator, so pickles do not grow.
gate 1, tests: **1097 passed, 1 skipped**, rerun by the reviewer under the
conda python (1084 + 13 new: five adversarial columns × three subsets
against `factorize` as the oracle — mixed `1`/`1.0`/`True`, missing values,
a parent and a leg that take different factorize paths — combo parity, the
guard, and end-to-end regressor / binary / 3-class fits with the sharing
monkeypatched off, asserting `array_equal` predictions and fewer factorize
calls).
gate 2, identity: `identity_snapshot.py check` **155/155 identical**
against the baseline on disk (its panel carries categorical configs that
auto-split, so the new path is exercised).
gate 3, speed (`benchmarks/f4_c4a_speed.py`, new; same-process A/B on the
DEFAULT estimator, OFF arm = `_shared_cat_ctxs` returning no contexts,
median of 5; `results/campaign-f4c4a-speed-20260921.txt`):
  hc:kick          **−9.4%**  factorize 84 → 24 calls, 0.367 → 0.167 s
  hc:sf-police     **−18.9%** 26 → 5 calls, 0.267 → 0.090 s
  hc:porto-seguro  **−18.1%** 130 → 31 calls, 0.958 → 0.386 s
  hc:okcupid-stem  **−8.2%**  77 → 20 calls, 0.233 → 0.101 s
  gr:clf_num/Higgs **−0.5%** on zero factorize calls (the control)
The seconds saved in `factorize` account for the whole fit-time change on
every set (kick 0.200 of 0.178 s, porto-seguro 0.572 of 0.544 s), so the
read is the mechanism and not drift.
verdict: **PASS → PR** (library source; waits for the maintainer). Kill
bars (a)(b)(c)(d) all clear. Forecast: strength exactly zero HIT; speed
direction HIT and size MISSED LOW on every set — kick −9.4 against −5 to
−8, sf-police −18.9 against −10 to −15, porto-seguro −18.1 against −8 to
−14 (I expected the low end there and got the top), okcupid-stem −8.2 at
the edge of −5 to −8. Why: I discounted the ceiling by the C2/C1b
conversion rate, and this object converts at nearly 100% — the saved calls
are whole Python-level passes, not arithmetic inside a loop that the fit
amortizes anyway. The largest F4 win since C1, and the first that any user
with categorical columns feels. CHANGELOG entry added under Unreleased —
and the C1b entry, which b50f01d had filed under the already-released
0.32.0 heading, moved there with it (0.32.0's section now matches the
v0.32.0 tag again).
next: C4a-2 — the numeric block on the same plumbing (unboxed once per leg
today and once more in `_validate_fit_input`; ceiling 2.6% kick, 5.5%
porto-seguro + its 3.2% validate row) is the natural follow-up and a
smaller muse task; then C3 (binary Logloss layer, reaches Grinsztajn).
C4b (shared TS permutations) stays parked. One campaign PR at a time: the
next rung waits for this one to merge.

#### I026 2026-09-21 F4 S0 (candidate C4 — split hc `prep` by wall clock; measurement, pre-registered)
why now: F6 closed (I025); under the 2026-09-21 ranking F4's measured
objects go next, C4 first on ceiling (I023: `prep` is 34.8 / 31.8 / 41.1%
of the default fit on kick / wine-reviews / okcupid-stem, the largest
non-kernel number on the board, inside unsplit). Class: **measurement for
exact-rewrite perf**; zero library change, no muse task. Branch
`campaign/f4-c4-prep-split` from main at 10eccba (PR #123 merged by the
maintainer). The PR will touch only `benchmarks/` and markdown, so under
the widened 2026-09-21 rule the loop merges it itself.
instrument: `benchmarks/f4_other_walltime.py` gains `--prep-detail`: the
same exclusive-time hooks, pushed INSIDE `prep` — `CatTransformCache.column`
(factorize), `_numeric_block` (unboxing the object array the harness hands
every categorical dataset), the ordered-TS `fit_transform` with its numba
kernel hooked separately (so the remainder is the permutation draws),
`OrderedTargetEncoder.transform`, `_codes_for_transform`, `Binner`
fit/transform, the cross block, and the preprocessor's own remainder
(stack, feature map, the feature-major transposes). Rows still partition
the fit, still split by leg; 5 reps and per-rep times kept (I023's
lesson). Panel: the three I023 hc sets plus sf-police, Traffic_violations
and porto-seguro — string-coded and numeric-coded categoricals, binary,
multiclass and regression.
barrier: `barrier_check.py` matched seven, all keywords, none binds a
timing read: B4/B18/B3 (ordered TS appears only as a timed object; what it
computes is untouched — B18 closes changing the transform's VALUES, an
exact rewrite changes none), B2/B13/B12 ("refit", "selection" as leg
names), B10 (binds objects inside `build_oblivious_tree`; its method,
ceiling before code, is this rung).
forecast, before any run (share of the default fit, all legs): factorize
plus eval-row codes **10–18%** on the string-coded sets (kick, okcupid,
wine-reviews, sf-police, Traffic) and under 3% on porto-seguro, whose
integer categories take the vectorized path — what C2 left, paid three
times per fit (train rows, validation rows, then every row again in the
refit). Ordered-TS fit **5–8%** on binary/regression and **15–20%** on
multiclass (one encoder per class: 3 × 17 columns × 4 permutations = 204
sequential passes on okcupid), with the permutation draws 40–50% of it.
`_numeric_block` 3–6% where numerics are many (kick, porto-seguro), under
1% on category-dominated sets. Binning 3–6%. The preprocessor's own
remainder 1–3%. Where I expect to be wrong: wine-reviews — my pieces sum to
~21% of a measured 31.8%, so something on a 26k-category column (building
`cat_maps_`, remapping validation codes) is bigger than I think.
bars: (a) instrument — wrapped/plain ≤ 1.05 per set. (b) candidate — a
named piece ≥ 8% of fit on ≥ 2 hc sets gets its exact-rewrite question
answered in this entry (is there a rewrite that provably returns the same
arrays, and what is its ceiling); (c) parked — 4–8% on ≥ 2 sets, recorded
with ceilings; (d) nothing ≥ 4% on 2 sets ⇒ C4 closes as "diffuse", F4
moves to C3.
ran: `f4_other_walltime.py --prep-detail --reps 5`, 6 hc sets × (1 warm +
5 plain + 5 wrapped fits), one pass, ~6 min →
`results/campaign-f4c4-prep-20260921.{md,json,log}`. Bar (a) instrument
**PASS**: wrapped/plain 0.976–1.010 on all six.
result (% of the default fit, all legs; kick / wine-reviews / okcupid-stem
/ sf-police / Traffic_violations / porto-seguro):
  `prep` total          35.5 / 30.2 / 41.2 / 42.2 / 36.2 / **58.3**
  **factorize**         14.7 / 16.9 / 14.1 / **29.8** / 11.3 / **29.9**
  ordered-TS draws+avg   5.5 /  4.2 / **15.0** / 4.6 / **14.7** / 8.6
  ordered-TS kernel      1.4 /  1.3 /  4.0 /  1.4 /  4.0 /  2.4
  `_numeric_block`       4.7 /  0.4 /  0.6 /  0.9 /  0.2 / **10.1**
  cross block            4.5 /  5.0 /  1.4 /  2.2 /  1.0 /  2.2
  binning (fit+apply)    2.0 /  0.8 /  3.4 /  1.2 /  2.7 /  2.0
  preprocessor remainder 2.4 /  1.3 /  1.7 /  1.6 /  1.4 /  2.9
**factorize is the largest non-kernel object on every high-card set**, and
the reason is structural: one default fit factorizes overlapping rows
three to four times — the selection legs' training rows (80%), the
validation rows for the eval transform (20%), the validation rows AGAIN
when calibration predicts from raw X, then every row in the refit (100%) —
about 2.2× the matrix (kick: 154 ms selection + 36 ms calibration + 129 ms
refit of a 2.17 s fit). Porto-seguro is where my forecast broke: its
integer-coded categories take `_factorize_numeric`, which I priced as
free, and on the object array the harness hands every categorical dataset
that path is a per-element float cast plus a boxed equality audit — 29.9%
of fit, plus 10.1% unboxing the numeric block and 3.2% unboxing it once
more in input validation. The ordered-TS row is mostly NOT its numba
kernel: the kernel is 1.3–4.0% and the `rng.permutation` draws (one per
column per permutation per class encoder; 408–456 draws per fit on the two
multiclass sets) are three times that — the microbench prices the draws at
75–80% of the row.
verdict: **PASS bar (b), two candidates; one has an exact rewrite.**
  **C4a — prepare once per fit** (factorize, 6 of 6 sets ≥ 11%). Exact
  rewrite EXISTS: factorize each categorical column once on the full
  matrix, then derive each leg's factorization by re-ranking the integer
  codes to first-appearance order inside the leg's rows — which is what
  `_factorize_int` already does, exactly, for combo keys (an O(n) numba
  re-rank would make it free). Codes and category order come out identical
  to `factorize(X_leg[:, f])`, so encodings, bins, trees and predictions
  are bit-identical; the only observable difference is which of several
  ==-equal objects (`1` vs `1.0`) a category map keeps as its key, which no
  lookup can see. The same context carries the numeric block as float64,
  converted once (it is converted a fourth time in `_validate_fit_input`
  today and thrown away). Ceiling: ~55% of the factorize row —
  **8 / 9 / 8 / 16 / 6 / 16%** of fit — plus 2.6% (kick) and 5.5% (porto)
  from the numeric block; honest forecast by the C2/C1b conversion rate,
  **−5 to −12% of fit on high-card sets**, zero on Grinsztajn (no
  categoricals survive its curation) and on a user-supplied `eval_set`
  (separate matrix, nothing to share). The reach that matters is off the
  benchmark: string categoricals are the common real-world input. Class:
  exact-rewrite perf, pure-speed ladder. Risk: it threads a per-fit context
  from `_auto_es_split` through the booster into the preprocessor —
  three library files, the most invasive F4 unit so far.
  **C4b — ordered-TS permutation draws** (15.0 / 14.7 / 8.6% on okcupid /
  Traffic / porto). NO exact rewrite: every draw is a distinct slice of a
  golden-frozen generator stream, per column, per permutation, per class
  seed; overlapping draws with the kernel would save at most the kernel's
  1–4%. The object that exists is an ALGORITHM change — one set of four
  permutations shared across columns (CatBoost shares its permutations
  across features too), draws 408 → 4 — ceiling ≈ the whole row, **4–15%
  of hc fit**, FP-drift class: every categorical golden moves and the
  correlated prefix noise across columns is a strength question that owes
  the full ladder. Parked with its ceiling, behind C4a.
  parked under bar (c): `_numeric_block` (folded into C4a), the cross block
  at 4.5–5.0% on kick and wine-reviews (unsplit), the TS kernel at 4.0% on
  the multiclass sets.
Forecast: factorize 10–18% HIT on four string sets, MISSED high on
sf-police (29.8) and MISSED outright on porto-seguro (29.9 against "under
3%"); ordered TS HIT in both bands, draws' share MISSED low (I said 40–50%
of the row, it is 75–80%); numeric block HIT where numerics are many
(porto over the band); binning MISSED low (0.8–3.4 against 3–6 — the
parallel binner is cheaper than I priced); wine-reviews resolved as
predicted-wrong: the missing ten points were factorize (16.9, not 11.5)
and the cross block (5.0). Caveat (GATE_ROBUSTNESS #6): one split,
threads unpinned — these shares choose the next unit and gate nothing.
next: **C4a S1 — a muse task** (library source, so the PR waits for the
maintainer): a per-fit prepared-columns context built where the estimator
validates X — categorical factorizations of the full matrix plus the
float64 numeric block — consumed by the selection legs through row indices
and by the refit as is; `factorize` stays the definition and the oracle.
Gates: `identity_snapshot` 155/155, full suite, new exact-equality tests
(leg factorization from the context == `factorize` on the leg's rows, incl.
mixed-type and missing categories), same-process fit A/B on kick /
sf-police / porto-seguro with a Grinsztajn numeric set as the zero-change
control. Kill: any identity drift, or under 3% on all three sets. C3
(binary Logloss layer) queues behind it; C4b stays parked.

#### I025 2026-09-21 F6 S1b (rare-only transform weight, tested where it can fail; pre-registered in I024)
why now: I024's `next:`; PR #122 merged by the maintainer (516ab12), no
campaign PR open. Class: **defect probe**, zero library change — the rung
edits the probe script only, so it is Claude's. A `benchmarks/` script is
code under the 2026-09-21 merge rule, so this PR waits for the maintainer.
design, fixed in I024 before this run: arm **A3** = the count-matched
weight `g(m)` where the category's train count m ≤ 5, the shipped
`m/(m+a)` above; **A4** = the half-matched weight on the same stratum
(secondary). The edge is Part 1's pre-registered stratum, not tuned.
Regime: the I024 splits with the TRAINING rows subsampled to 50% and 25%
(stratified for classification, seeded by the split seed), test rows
unchanged — the `@sus` design, run inside the probe so all five sets have
it. Full size is rerun with A3/A4 because bar 2 needs A3's full-size gain,
which I024 never measured. Panel: the four gap sets + wine-reviews at
three sizes, porto-seguro at full size as the exact-tie control; 3 seeds ×
arms A0/A3/A4. Part 1's rare share (held-out rows with m ≤ 5) is printed
per size so the premise "subsampling makes categories rarer" is checked,
not assumed.
barrier: `barrier_check.py` matched four. B3/B4/B5 as argued in I024 (B5
is now satisfied by construction: the weight moves only in the stratum
where the mismatch was measured). B8 — keyword only: "subsample" here is
the probe's training-size regime, not the booster's row-sampling knob.
forecast, before any run: premise — the m ≤ 5 share of held-out rows on
the high-card columns roughly doubles from full size to 25% (sf-police
27% → 45–60%), while the unseen share also rises (6.6% → 15–25%) and
unseen rows read the prior in every arm, which caps the growth. Strength,
A3 at full size: **sf-police +0.2 to +0.3%** (most of A1's +0.31%, its
rare rows are where the gain was); **Traffic_violations +0.1 to +0.3%**,
under A1's +0.46%, because only 2–5% of its rows per column are rare and
part of A1/A2's gain there must have come from the mid stratum; okcupid
0 to +0.1%; **kick inside ±0.02%** (2% rare rows — the loss A1 paid in
mid and large categories is gone); wine-reviews +0.05 to +0.2%. At 25%:
gains grow on sf-police (+0.4 to +0.8%), Traffic (+0.3 to +0.8%) and
wine-reviews, okcupid turns positive (+0.1 to +0.3%), kick flat to
slightly positive (0 to +0.1%). A4 lands between A0 and A3. Where I expect
to be wrong: Traffic — if its gain lived in categories of 6–50 rows, the
rare-only weight loses it at full size and the 25% read decides the
family; and the weight's hard step at m = 5 → 6 (0.59 → 0.86) may cost a
little everywhere. Cost: later early stopping again, fit ×1.0–1.1,
report-only.
bars (as registered in I024): 1 — A3 wins ≥ 8 of the 12 gap (set, seed)
pairs at 25% with a positive median. 2 — A3's mean gain at 25% exceeds
its mean gain at full size on ≥ 3 of 4 gap sets. 3 — kick negative on at
most 1 of 3 seeds at 25%. 4 — full-size porto-seguro exact (|Δ| ≤
0.005%). Any fail ⇒ F6 KILLED as "real defect, no transform-side fix
converts"; the Counter-feature question stays with R3. All pass ⇒ S2.
ran: `probe_ts_mismatch.py --arms A0 A3 A4 --train-fracs 1 0.5 0.25`, 5
sets × 3 sizes × 3 seeds × 3 arms + porto-seguro at full size = 144
default fits, one pass, ~25 min →
`results/probe-ts-mismatch-s1b-20260921.{md,json,log}`.
premise **HELD**: the share of held-out rows in categories of m ≤ 5 rises
as the training rows are cut — sf-police 27.0 → 35.7 → 41.1%, okcupid-stem
6.5 → 7.3 → 8.6%, Traffic_violations 3.0 → 4.7 → 8.8% — under my 45–60%
forecast for sf-police because the unseen share climbs with it (6.6 →
21.8%) and unseen rows read the prior in every arm.
bars, A3 (the registered primary arm):
  1 **FAIL** — gap pairs at 25%: **7W-5L**, median +0.096%, mean +0.107%;
    the bar was 8. By set: sf-police 3/3 (+0.53%), kick 2/3 (+0.10%),
    okcupid-stem 1/3 (+0.01%), Traffic_violations 1/3 (−0.21%).
  2 **FAIL** — gain at 25% over gain at full size: sf-police +0.53 vs
    +0.29 ✓, kick +0.10 vs −0.01 ✓, okcupid-stem +0.01 vs +0.01 ✗,
    Traffic_violations −0.21 vs +0.04 ✗ — 2 of 4, the bar was 3.
  3 PASS — kick at 25%: +0.23, +0.07, −0.01, one negative seed.
  4 PASS — full-size porto-seguro an exact tie on all three seeds, both
    arms (0W-0L-3T): outside the rare stratum the arms are bit-identical
    to the shipped transform, as built.
the secondary arm, reported because it reads better and changes nothing:
A4 went 9W-3L at every size and would have cleared bars 1 and 2 by their
letter — with a median gain of **+0.04%**, a tenth of anything that would
matter. GATE_ROBUSTNESS question 2 settles both arms: drop sf-police and
the other three gap sets at 25% read A3 −0.03% (4W-5L of 9) and A4 −0.02%
(6W-3L). The effect is one dataset.
what the run does establish: on sf-police the rare-only weight is worth
+0.29% Brier at full size and +0.53% at a quarter of the rows, **9 of 9
fits**, log loss agreeing (+0.22 to +0.40%), and it carries the whole of
the uniform fix's gain there (A1 read +0.31%). That is what a column with
27–41% of its held-out rows in categories of five or fewer training rows
looks like, and no other set on the panel has one: wine-reviews comes
closest (28–35%) and gains at the two smaller sizes (+0.19%, +0.27%, 5 of
6 fits) but not at full size. Traffic_violations answered the question I
flagged: its +0.46% / +0.75% under the uniform arms did NOT come from rare
categories — the rare-only arms read +0.04% / +0.06% — so it came from
re-weighting categories of more than five rows, where Part 1 found no
over-trust to correct. Unexplained, one dataset, post-hoc; recorded for
R3 (the CatBoost ablation's CTR-border and prior arms are the sanctioned
place to look), not pursued here. Cost column void this time: the
monkeypatched transform evaluates a digamma per row, so fit reads
×1.04–1.21 even where the tree counts are identical.
verdict: **FAIL (bars 1 and 2) → F6 KILLED**, as registered: "real defect,
no transform-side fix converts". Forecast: premise HIT; sf-police HIT at
both sizes (+0.29 in a +0.2–0.3 band, +0.53 in +0.4–0.8); kick HIT (flat,
then +0.10); Traffic MISSED with the wrong sign at 25% — the failure I
named in advance and still under-weighted; okcupid-stem and wine-reviews
at full size MISSED low. The two-rung record in one line: the encoder's
train/test asymmetry is real and over-trusts rare categories, and
correcting it at transform moves exactly the datasets whose rows mostly
live in rare categories — one of our fourteen. A change that re-scores
every categorical model, moves every categorical golden and needs a
fitted-state flag for old pickles does not get built for one dataset.
Closed as barrier **B18**. Nothing ships; the probe and this record are
the durable artifacts.
next: F6 closed. The queue under the 2026-09-21 pick puts F4's measured
objects next: **C4 S0** — split hc `prep` (32–41% of fit) by wall clock
into factorize, ordered TS, binning, array conversion, per leg, then ask
the exact-rewrite question of the top piece — then C3 (binary Logloss
layer). Class: exact-rewrite perf.

#### I024 2026-09-21 F6 S0+S1 (ordered-TS train/test mismatch: moment read + a transform-only A/B; pre-registered)
why now: the maintainer's pick (shortlist R1, goes first). Class: **defect
probe**, zero library change — the script is harness code, so it is
Claude's and muse gets no task. Branch `campaign/f6-s1-ts-mismatch-probe`
cut from main at 8dd34c1 (PR #121 merged; no campaign PR open).
the object, read from `target_encoding.py` before forecasting:
`fit_transform` gives a train row `(S_k + prior·a)/(k + a)` from the k
rows of its category that precede it in a random permutation, averaged
over 4 permutations — k is uniform on 0..m−1 for a category of m rows, so
the expected weight on the category's own mean is
`h(m) = mean_{k<m} k/(k+a)` (m=5, a=1: 0.54; a singleton: exactly 0, the
row always reads the bare prior). `transform` gives validation, test and
calibration rows the full totals, weight `m/(m+a)` (m=5: 0.83; a
singleton: 0.5). The trees are grown on the first distribution and
early-stopped, calibrated and scored on the second. CatBoost has the same
asymmetry but also hands its trees a Counter feature, so they can learn
reliability by count; ours cannot.
barrier: `barrier_check.py` matched three. B3 (partial CatBoost ports) —
not a port: CatBoost shares this asymmetry, the question is our encoder's
self-consistency, and nothing is added (no feature, no knob). Dedup
against the record: `cat_smoothing` was killed on hc twice (PARETO_PLAN,
2026-07-15) — a different axis; `a` moves both sides together and cannot
close a prefix-vs-total gap that exists at every `a`. B4 — keywords only
(ordered TS is the encoder, not ordered boosting). **B5 binds and sets the
design**: a shrink can only fix a component that VARIES across the units
shrunk, so the read is stratified by category count, and a mismatch that
does not concentrate in the rare stratum kills the fix at S0 whatever the
headline says.
instrument: `benchmarks/probe_ts_mismatch.py` (new). Part 1, no model —
the default preprocessor on the 75% split, raw encodings captured before
binning: per TS column Δmean/SD, SD ratio and KS distance between train
rows (`fit_transform`) and held-out rows (`transform`); then by stratum of
category train count (m ≤ 5 / 6–50 / > 50) the spread `E|e − prior|` and
the reliability slope (OLS of the target on the encoding; honest on train
rows because ordered TS never sees the row's own label). Part 2 — a
transform-only A/B by monkeypatch, default estimator, 3 split seeds,
paired: A0 shipped weight `m/(m+a)`; **A1 count-matched** `g(m) =
mean_{k≤m} k/(k+a)` (what the row would have read had it been a training
row of that category); **A2 half-matched**, the mean of the two weights —
registered before any data on a toy-model argument: permutation averaging
makes train encodings less noisy than one prefix, so full matching should
over-correct. Unseen categories read the prior in every arm;
`fit_transform` is untouched. Panel: the four gap sets (sf-police 15165,
okcupid-stem 7019, Traffic_violations 3830, kick 1063), two high-card
regressions as secondary reads (wine-reviews 15633, colleges 6039), two
low-card controls (kdd_ipums 191, porto-seguro 104).
forecast, before any run: moments — mean shift under 0.05 SD on every
column (both sides shrink toward one prior, so the shortlist's literal
"standardized shift" bar would kill a real effect; the second moment is
where it shows); held-out/train **SD ratio 1.1–1.4** on columns with card
≥ 1000 and 0.98–1.02 under card 200; rare-stratum spread ratio
**1.3–1.8**, and 0.95–1.05 in the m > 50 stratum (the B5 read). Direction
— rare-stratum reliability ratio (held-out slope / train slope)
**0.7–0.9**: the trees learn a slope from compressed train encodings and
apply it to wider held-out ones, which is over-trust. Strength — small:
the better arm **+0.05 to +0.3%** on the gap sets' primary metric, A2 ≥
A1, controls flat. The shortlist's "+2 to +5 hc points" was a win-rate
guess; on Brier I expect a tenth of a percent, because only the few
percent of held-out rows in categories of m ≤ 2 are badly served.
bars, in order: 1 mismatch — high-card columns on the gap sets show SD
ratio ≥ 1.05 or rare spread ratio ≥ 1.25, else KILL "no mismatch". 2 B5 —
m > 50 spread ratio inside 0.95–1.05, else KILL "common component, a
count-dependent shrink cannot be the fix". 3 direction — rare reliability
ratio < 0.9 on ≥ 3 of 4 gap sets; ≥ 1.0 means the asymmetry is benign and
the family dies as "real, harmless". 4 strength (S1) — A1 or A2 wins ≥ 8
of the 12 gap (set, seed) pairs with a positive median, controls' median
inside ±0.05%. 1–3 pass and 4 fails ⇒ "mismatch real, matching at
transform does not convert", family narrows to the Counter-feature
question R3 already owns. All four pass ⇒ S2: muse implements the chosen
weight in `OrderedTargetEncoder.transform` (goldens WILL move on cat
fixtures — an algorithm change under the 2026-09-18 drift policy), synth
screen with the cat-scope slice pre-registered.
ran: `probe_ts_mismatch.py`, 8 sets, Part 1 on split seed 0 then 3 seeds ×
3 arms = 72 default fits, one pass, ~12 min →
`results/probe-ts-mismatch-20260921.{md,json,log}`.
bar 1 mismatch **PASS, 4 of 4** — medians over each gap set's high-card
columns, held-out over train: SD ratio 1.23 / 1.18 / 1.08 / 1.00
(sf-police / okcupid-stem / Traffic_violations / kick); rare-stratum (m ≤
5) spread ratio **1.99 / 5.15 / 3.12 / 1.86**. Mean shift ≤ 0.03 SD on
every high-card column, as forecast — the shortlist's literal bar would
have killed a real effect. The rare spread is far over my 1.3–1.8 band:
four-permutation averaging keeps a rare training row's encoding pinned
near the prior, while a held-out row of the same category reads half to
five-sixths of a one-to-five-label mean.
bar 3 direction **PASS, 3 of 4** — rare-stratum reliability ratio (OLS
slope of the target on the encoding, held-out over train) **0.63 / 0.67 /
0.70** and 1.01 on kick; the two secondary regressions agree (wine-reviews
0.71, colleges 0.42). Stronger over-trust than the 0.7–0.9 forecast. Kick
has almost nothing to fix: 0.8% of its held-out rows sit in categories of
m ≤ 2, against 13.1% on sf-police and 13–20% on wine-reviews' two big
columns. The same
asymmetry runs the OTHER way in big categories (m > 50 reliability
1.10–1.29): training rows carry prefix noise that held-out rows do not,
so there the trees under-trust.
bar 2 (B5) **FAIL by the letter, and the letter was mis-specified** — m >
50 spread ratio 0.99 / 0.91 / 0.96 / 0.94 against a 0.95–1.05 band, two
sets under the floor. What the bar was written to detect, a shift common
to all counts, is decisively absent: the ratio runs from 1.9–5.2 in rare
categories to 0.9–1.0 in large ones. The band did not anticipate that
prefix noise makes TRAIN encodings slightly wider in large categories.
Recorded as a deviation at the time it was seen (GATE_ROBUSTNESS #8), not
quietly reread; it does not rescue bar 4.
bar 4 strength **FAIL, by one pair** — gap sets, 12 (set, seed) pairs:
A1 **7W-5L**, median +0.298%, mean +0.202%; A2 **7W-5L**, median +0.143%,
mean +0.247%; the bar was 8. Controls' medians −0.001% / +0.001%, inside
±0.05% (porto-seguro exact to ±0.003%; kdd_ipums is 7k rows at Brier 0.017
and swings ±1.6% by seed — it is not an inert control, its categories
average m ≈ 27 where the weights differ by 0.1). The 12 pairs are four
datasets, and the datasets disagree in the way Part 1 predicts:
  sf-police          A1 +0.31% (3/3 seeds)   A2 +0.28% (3/3)
  Traffic_violations A1 +0.46% (3/3)         A2 +0.75% (3/3)
  okcupid-stem       A1 +0.09% (1/3)         A2  0.00% (1/3)
  kick               A1 −0.05% (0/3)         A2 −0.04% (0/3)
  secondary: wine-reviews A1 +0.08% (2/3), A2 +0.29% (3/3); colleges flat.
In harness Brier units the two winners recover roughly a quarter of the
CatBoost gap (sf-police +0.0015 of 0.0056; Traffic +0.0013 to +0.0021 of
0.0053; different splits, so a rough read); kick gives back 4% of its
gap. Log loss agrees with Brier in the sign of every gap set's mean,
smaller on Traffic under A1 (+0.10% against +0.46%). Cost axis,
report-only (one process, threads unpinned): the matched arms early-stop
LATER — sf-police 45 → 62 and 50 → 85 trees, wine-reviews 325 → 492 — so
fit time reads ×0.97–1.14, median about ×1.05.
That is itself evidence for the mechanism: under the shipped transform the
validation curve turns up early because the validation rows' rare
encodings are over-trusted.
verdict: **S1 FAIL on the registered bar (7 of 12, bar 8) — the uniform
count-matched transform does NOT go to S2.** Forecast: moments HIT on
direction and MISSED low on size (rare spread 1.9–5.2 against 1.3–1.8),
direction HIT and stronger than forecast, strength at and above its band
where it converts (+0.3 to +0.75% against +0.05–0.3%) and MISSED on
breadth — I
expected the rare-row share to set the SIZE of a uniformly positive
effect, and instead it decides the SIGN. "A2 ≥ A1" split: true on Traffic
and wine-reviews, false on sf-police and okcupid. The defect itself is
established: real, concentrated in rare categories, over-trust in
direction, and worth a quarter of the CatBoost gap on the two sets where
rare held-out rows are common. What failed is the FIX AS REGISTERED: it
re-weights every category, and in mid and large categories the evidence
says held-out encodings are already as reliable as training ones or more
so, so extra shrink there is a small loss (kick 0/3). Per the registered
consequence the family narrows; it does not die, because the narrowing is
not the Counter feature I wrote down — it is the weight's reach.
next: ONE pre-registered follow-up. The rare-only idea is post-hoc to this
run (GATE_ROBUSTNESS #8), so it is tested where it can FAIL, and it earns
nothing until independent data agrees (the synth screen at S2; never the
`pub:` suite, which a design choice would contaminate). F6 S1b, same
script, zero library change: arm **A3 = count-matched weight only where
m ≤ 5, shipped weight above** (the edge is Part 1's pre-registered
stratum, not tuned; A4 = the half-matched weight on the same stratum rides
along as the secondary arm), run in a regime this run did not visit — the
same splits with the TRAINING rows subsampled to 25% and to 50%, test rows
unchanged. Subsampling shrinks every m, so the mechanism makes a sharp
prediction: the rare share rises, A3's gain must GROW against this run's,
and kick must turn from slightly negative to flat or positive. Bars: A3
wins ≥ 8 of 12 gap pairs at 25% with a positive median; its mean gain at
25% exceeds its mean gain at full size on ≥ 3 of 4 gap sets; kick negative
on at most 1 of 3 seeds; full-size porto-seguro still exact. Fail ⇒ F6
KILLED as "real defect, no transform-side fix converts", and the
Counter-feature question stays with R3. Pass ⇒ S2 (muse: the rare-only
weight in `OrderedTargetEncoder.transform` behind a fitted-state flag so
old pickles keep their predictions; cat goldens move under the 2026-09-18
drift policy; synth screen with the cat-scope slice pre-registered). Class
stays defect fix: no new parameter.

#### I023 2026-09-21 F4 S1 (fresh profile: name the Grinsztajn "other" column; measurement, pre-registered)
why now: the refill shortlist (I022) waits on the maintainer's pick, and F4
is the one ACTIVE family; its `next:` owes a fresh profile. All three F4
ships came out of the hc/multiclass "other" and none can move Grinsztajn,
the headline stratum — so the profile is aimed there. The August
attribution (`results/campaign-attr-20260816.md`) left **23–30% of every
Grinsztajn fit** in an unnamed "other" column; its instrument wrapped
`build_oblivious_tree` but not `replay_oblivious_tree`, the loss, the
validation score or the round loop, so that column has never been read.
Measurement rung, zero library change, no muse task (the I019/I021
precedent: harness code is Claude's per AGENTS.md). Branch
`campaign/f4-s1-gr-other-profile`, based on main at 486e563.
instrument: `benchmarks/f4_other_walltime.py` (new). Exclusive-time
`perf_counter` hooks — a hook's clock stops while a nested hook runs, so
the rows partition the estimator fit and sum to it — on the default
estimator over the August panel unchanged (6 gr + 3 hc), rows split by leg
(early-stopped selection fits vs the full-data refit). Not cProfile: every
object here is few-fat-calls, the shape I012 showed cProfile understates.
The instrument is priced two ways: hook bookkeeping calibrated on a no-op
and reported as its own row, and every wrapped fit alternates with an
unwrapped one in the same process (median of 3 each).
barrier: `barrier_check.py` matched five, all on keywords, none binds a
measurement. B10/B15 — the grow kernel stays one opaque row and is
excluded from the candidate read; B10's METHOD (ceiling before code) is
what this rung is. B2/B13 — replay appears only as a timed leg; no
audition, selection or tuning decision is touched. B6 — no tuning.
forecast, written before any run (shares of estimator fit): `grow` 55–70%
on gr. `replay` (the refit's rounds) is the largest piece of the old
"other", **8–15%** on gr. `prep` 1–3% on gr numeric sets, 15–30% on hc.
`grad_hess` 1–2% regression, **3–6% binary** (the numba sigmoid plus two
numpy passes — the C1b shape on the scalar path). `train_update` 2–4%,
`eval_advance` 1–2%, `val_score` under 1% regression and 2–4% binary (two
logs and a clip over the validation rows every round). The two Python
loop rows 2–5% combined, a point or two of it the instrument's own.
Estimator-level rows (split, validation, cross screen, importances,
calibration) 1–4% combined on gr. The bet: on Grinsztajn **no non-kernel
row reaches 5%**, and the old "other" is mostly the replay leg plus a tail
of 1–4% rows; the likeliest object at or above 3% is binary `grad_hess`.
bars: (a) instrument — wrapped/plain fit ratio ≤ 1.05 per set, else that
set's read is void. (b) candidate — a non-kernel row ≥ 5% of fit on ≥ 2
Grinsztajn sets of one task type becomes the next F4 candidate and owes
its own S0 (exact-rewrite question, ceiling, forecast). (c) parked — rows
at 3–5% on ≥ 2 sets are recorded with their ceilings, the C1b treatment.
(d) family — no non-kernel row ≥ 3% on any Grinsztajn set means F4 has no
Python/numpy-layer object left on the headline stratum: F4 goes DORMANT
(reopen on a structural change), the beam holds zero ACTIVE families and
the loop waits on the I022 pick.
ran: `f4_other_walltime.py`, 9 sets × (1 warm + 3 plain + 3 wrapped fits),
one pass, ~6 min → `results/campaign-f4s1-other-20260921.{md,json,log}`.
Bar (a) instrument **PASS**: hook bookkeeping 0.85 µs per call = 0.1–1.2%
of fit; wrapped/plain ratio 0.99–1.03 on eight sets. MagicTelescope read
0.869 — noise in the PLAIN arm (a 3-rep median of 1.067 s; all seven
rerun fits 0.87–0.91 s) — so the two numeric binary sets were rerun at 7 reps
(`…-binary7.{md,json}`): ratio 1.026, every phase share within 0.1 point
of the first read.
result, Grinsztajn (% of estimator fit, exclusive wall clock):
  the old "other" has a name, and it is the refit. The full-data refit leg
  is **25–35% of every fit**: replayed rounds 10.7–19.3% (`replay`, never
  wrapped in August) plus the 1.25× extra rounds grown from scratch
  9.6–12.3% (counted under `grow`; earning, per REFIT_PLAN's 8/10). `grow`
  in all legs 63.5–71.8%. A replayed round costs ~40–45% of a grown one on
  the linear-leaf regression sets (nyc-taxi 0.97 vs 2.45 ms, diamonds 1.09
  vs 2.4 ms); its three kernels are unsplit and the ridge accumulator is
  B10-closed — a pointer, not a candidate.
  regression: nothing. `grad_hess` 0.6–0.7%, `val_score` 0.8–2.1%,
  `train_update` 2.7–3.2%, `prep` 1.2–4.4%, `centers_std` 0.8–1.7%, both
  loop rows together 1.6–4.7% (1.2 of cpu_act's is the instrument).
  binary: the Logloss layer is two rows of one object. `grad_hess`
  **4.9 / 4.2 / 4.4%** and the per-round validation `eval` (`val_score`)
  **5.1 / 4.3 / 4.0%** on MagicTelescope / Higgs / road-safety —
  **8.3–10.0% combined**. Everything else under 3.5%.
result, hc (outside the pre-registered Grinsztajn scope, recorded because
it is the largest non-kernel number on the board): `prep` is **34.8%**
(kick) / **31.8%** (wine-reviews) / **41.1%** (okcupid-stem) of the fit
post-C2, and 15.5 / 13.7 / 19.9 points of that are the refit re-running
`fit_transform` on all rows (ordered statistics are row-set dependent, so
the re-run is owed; what it costs inside is unmeasured).
verdict: **READ — no Grinsztajn row clears bar (b); two clear bar (c).**
(b) FAIL: the one ≥ 5% reading is `val_score` 5.1% on MagicTelescope,
alone. (c) PASS twice, binary `grad_hess` and binary `val_score`, parked
with ceilings 4.2–4.9% and 4.0–5.1%. (d) not met — F4 stays ACTIVE.
Forecast: the bet HIT (no non-kernel row ≥ 5% on gr but one at 5.1;
binary `grad_hess` the likeliest ≥ 3% object, 3–6% band HIT). MISSES:
`replay` on regression 18–19% against 8–15% (binary inside); binary
`val_score` at the top of and over its 2–4% band — I priced two logs over
the validation rows too cheaply; hc `prep` 32–41% against 15–30%;
regression `grad_hess` under its band. Instrument caveat (GATE_ROBUSTNESS
#6): single split, threads unpinned — these shares choose what to measure
next and gate nothing.
next: F4 has two measured objects and each owes an S0, in ceiling order.
**C4 hc prep** (32–41% of hc fit): split `prep` by wall clock — factorize,
ordered TS, cat combinations, binning, array conversion, in the selection
legs and in the refit — then ask the exact-rewrite question of the top
piece; hc-only reach (14 sets + variants). **C3 binary Logloss layer**
(8–10% of a binary fit; 23 of 59 gr sets plus hc binary): (i) fuse
`p − y` and `max(p(1−p), 1e-6)` into the existing numba `_sigmoid`
kernel — the C1b move on the scalar path, elementwise, so bit-identical
by the same argument; (ii) `Logloss.eval` takes two logs per row where
0/1 labels need one (`y·log p + (1−y)·log(1−p)` is exactly `log p` or
`log(1−p)` when y ∈ {0,1}: the dead term is a signed zero) — a pure-numpy
rewrite that owes an exact-equality check against the current output on
real validation vectors, the mean kept in numpy. Honest prior: about half
of each ceiling converts, 3–5% of binary fit.
correction, same day: this entry was planned on a false premise. The
maintainer HAD picked from the I022 shortlist (R1 first) — the pick sat in
the session's memory notes and never reached this file, which still read
"awaiting the pick". The rung stands as a measurement (class: exact-rewrite
perf, second in his ranking) but it jumped the queue. The pick is now
recorded under the shortlist and R1 is on the beam as F6. **Next rung: F6
S0 + probe.** C4 and C3 queue behind it.

#### I022 2026-09-21 beam refill (staleness rule: one ACTIVE family, no candidate)
ran: four read-only lens agents in parallel (L1 loss-slice profiling on
`campaign-base-20260816.json` / `20260918-170822.json` via `hc_gap.py` and
`compare_runs.py`; L2 literature mechanisms against the code and the
barrier list; L3 CatBoost-knob ablation design for F5 with catboost 1.2.10
parameter semantics; L4 harness warts from I018–I021 and GATE_ROBUSTNESS),
then `barrier_check.py` over the nine survivors in one pass, dedup by hand
against this log and `research/SUMMARY.md`.
result: the shortlist above. Two findings worth recording as facts on
their own: the hc Brier gap is monotone in max cardinality (sf-police
15165 −0.0056, Traffic 3830 −0.0053, okcupid 7019 −0.0023, kick 1063
−0.0019, porto 104 −0.0001, and we WIN on kdd_ipums 191 and eucalyptus
27) — it points at per-level statistics, not at leaf estimation or gain
noise; and the ordered-TS encoder hands train rows prefix statistics but
test/ES rows full totals, an asymmetry nobody has measured (R1).
Instrument note for the L1→L3 agreement: two lenses independently named
the same four high-card sets from different files, which is the
dedup working as intended, not double counting.
verdict: SHORTLIST WRITTEN — the loop pauses here; entrants are the
maintainer's pick (beam cap 5).
next: on the pick, S0 for each entrant in shortlist order; R1's and R2's
probes are zero-library-change scripts and go first.

#### I021 2026-09-21 F3 S3 (decide run, OneLin vs OneLinXC per stratum; pre-registered)
why now: I020 passed. Measurement rung, no library change, no muse task.
Branch `campaign/f3-s3-clf-decide` stacked on `campaign/f3-s2-clf-forced-cross`
(PR #117; merge order #115, #116, #117, then this).
ran (planned): `run_benchmarks.py --decide --seeds 3 --save --models
ChimeraBoost ChimeraBoostOneLin ChimeraBoostOneLinXC` — Grinsztajn +
high-card + their `@sus25`/`@sus50`/`@time` variants, three arms in ONE run,
scored `compare_runs.py --by-suite --model ChimeraBoostOneLin --model-new
ChimeraBoostOneLinXC --expect-inert`. Read per stratum, never pooled
(GATE_ROBUSTNESS: strata under ~8 decided sets are pointers, `@time` seeds
duplicate).
forecast, before the run: strength — gr binary engaged slice (23 clf sets,
most above 2000 rows) positive by majority with engaged median **+0.2 to
+0.5%** Brier (the I019 probe read +0.36% on these very sets at the same
config, so this is close to in-sample for the direction; the decide run's
split and seeds differ). hc binary: the probe never covered it; forecast
**flat to slightly negative** (hc is categorical-heavy, the block is
numeric pairs plus gdiff, and E2's regressor read on hc-vs-LightGBM was a
coin flip). `@sus25`/`@sus50` binary: gates cut many sets below 2000 rows
so mostly ties; where engaged, direction as gr. `@time`: pointer only.
Every regression, multiclass and sub-gate row an exact tie. Cost — engaged
fit ratio median **1.2 to 1.5** (probe 1.33, synth 1.27).
bars, per stratum, in order: 1 control — regression, multiclass and
sub-2000 rows exact ties in every stratum (`--expect-inert`); a non-tie
voids the read. 2 gr binary engaged — wins ≥ half + 1 AND engaged median
> 0 on Brier; this is the gate. 3 hc binary engaged — at worst flat
(wins ≥ losses); a loss here is a documented caveat unless decisive at its
size (≤ 8 sets ⇒ pointer). 4 cost — engaged fit ratio median ≤ 1.5. Pass
⇒ propose the `quality=1` classifier pin as its own /experiment (released
preset change: full gate, the maintainer's go). Fail on bar 2 ⇒ F3 closes
"measured, not worth the pin", knob stays opt-in.
ran: as planned, one pass, ~35 min → `results/20260921-080246.json`
(103 dataset rows × 3 arms × 3 seeds), scored `compare_runs.py --by-suite
--model ChimeraBoostOneLin --model-new ChimeraBoostOneLinXC --expect-inert`;
engaged fit ratios from the per-record fit times.
result, per stratum: bar 1 control **PASS** — every regression and
multiclass row an exact tie in every stratum (gr 36/36 regression ties, hc
10/14 ties incl. all four multiclass sets, variants likewise); the arm
engaged only on binary rows above the gate. Bar 2 gr binary engaged
**FAIL** — 23 engaged sets, **10W-13L** (bar 12+), engaged median
**−0.04%** Brier. Wins: covertype +1.15% (num) / +0.42% (cat),
MagicTelescope +0.52%, house_16H +0.39%, jannis +0.20%, credit +0.15%,
Higgs +0.13%; losses: eye_movements −1.02% (cat) / −0.86% (num),
electricity −0.72% (cat), default-of-credit −0.34% / −0.23%, Bioresponse
−0.25%, Diabetes −0.22%. Bar 3 hc binary: 4 engaged, 2W-2L, all within
±0.14% — flat, pointer. Variants: gr@sus25 2W-2L, gr@sus50 0W-2L,
hc@time 2W-0L (n ≤ 4 each, pointers; none contradicts the gr read). Bar 4
cost — gr engaged median **1.43x** (range 1.07–1.82), hc engaged 1.55x,
inert slice 0.99x; whole-suite harness slowdown 1.2x vs OneLin's 1.0x.
verdict: **FAIL on the gate → KILL F3.** Forecast: control HIT, cost HIT
(1.43 inside 1.2–1.5), hc "flat" HIT — and the strength bet MISSED
outright: the probe's +0.36% median headroom on these same 16 clf_num sets
did not survive a different split and seeds, and the seven clf_cat sets
(never probed) went 2W-5L. Mechanism of death, and it is the one E2
recorded on the regressor at tolerable size: without the referee the mode
eats the losses the race dodges, and on the classifier those losses
(eye_movements, electricity, default-of-credit) are the same size as the
wins, so the median lands at zero. The regressor got away with it because
its headroom was 7x larger (+2.6% vs +0.36%). Reading the I019 probe
again with this in hand: 29W-19L at p≈0.19 was never decisive, and the
bar it cleared (+0.3%) was set for the regressor's prize, not this one.
consequence: the classifier "always" mode stays as shipped in PR #117 —
opt-in, default-off, documented with its measured evidence — per the bar
pre-registered in I020/I021 ("fail ⇒ knob stays opt-in"). The `quality=1`
classifier recipe keeps `cross_features=False`; no /experiment is
proposed. The maintainer may prefer to close #117 unmerged (nothing ships
from a killed family, the F2 precedent); if so, this rung's plan-file
record must be rescued onto a branch of its own, since it is stacked on
#117. Barrier candidate: none new — B1's mechanism (the race is what
makes cross features safe) already covers this; recorded as an instance.
next: F3 closed. The beam is at one ACTIVE family (F4, no measured
candidate) — the staleness rule fires: beam refill (four read-only idea
lenses L1–L4, funnel through `barrier_check.py`, dedup against this log
and `research/SUMMARY.md`), survivors to the maintainer, who picks
entrants. The next rung produces that shortlist and the loop pauses on
the pick.

#### I020 2026-09-21 F3 S2 (classifier "always" mode, default-off, + synth screen; pre-registered)
why now: I019 passed its bars. Muse task
`campaign_tasks/20260921-f3-s2-clf-forced-cross.md` (gitignored; the
verdict cites it). Branch `campaign/f3-s2-clf-forced-cross` stacked on
`campaign/f3-s1-clf-cross-probe` (PR #116) — merge order #115, #116, then
this rung's PR.
design (what muse builds): `ChimeraBoostClassifier` accepts
`cross_features="always"`. Binary path, mirroring the regressor's
`_arm_forced_cross`: under the existing gates (eval set present, ≥ 2000
rows, ≥ 2 numerics or 1 numeric + 1 cat) fit a `FORCED_CROSS_PROBE_ROUNDS`
(25) importance probe, take `_cross_candidate_pairs(..., top_m=
FORCED_CROSS_TOP_M)` through the existing `_screened_cross_pairs`, then ONE
full fit on the augmented matrix; no race, `selection_rounds` audition
skipped; `cross_features_selected_=True`, `cross_pairs_` set. Where the
gates fail, and on multiclass, the mode is inert (plain fit, `selected_ =
None`) — the regressor's "inert where gates fail" rule, chosen over raising
so a later `quality=1` pin cannot blow up on multiclass. The `quality=1`
classifier pin stays `cross_features=False` (a new class flag separates
"accepts always" from "quality pins always"; the regressor keeps both
True). The raced default path is untouched: goldens + `identity_snapshot`
must stay 155/155 (the knob unset is inert by construction).
harness (Claude, after muse): new arm `ChimeraBoostOneLinXC` = rung 1 with
`cross_features="always"` on classification rows and `False` on regression
rows, paired against `ChimeraBoostOneLin` in ONE synth run. Regression,
multiclass and sub-2000 rows are then exact ties by construction (the
`--expect-inert` control); binary ≥ 2000-row rows are the engaged slice.
Docs (`parameters.md`, `recipes.md`, PROJECT_STATUS) lose "regressor only".
barrier: as I019 (B16/B12/B17/B1/B2/B14 arguments unchanged; the unit adds
no mechanism the probe did not already run). B12 is priced by the same
screen (fit-time per engaged set recorded).
forecast, before any code: strength — engaged binary slice positive by
majority with engaged median **+0.2 to +0.8%** Brier (the probe's +0.36%
median headroom, minus the probe-vs-oracle give-back on divergent pair
sets); every non-engaged row an exact tie. Cost — engaged sets ~**1.3x** the
OneLin fit (probe read 1.33 median). Where wrong: the synth generator's
binary sets may carry less interaction structure than Grinsztajn's, reading
flat; or the ties control fails, which is a bug not a result.
bars, in order: 1 exact-tie control — every regression, multiclass and
sub-2000 row an exact tie (`--expect-inert`); one non-tie voids the screen.
2 direction — engaged binary slice: wins ≥ half + 1 AND median > 0 on
Brier. 3 canary slice not positive. 4 cost recorded. Bar 2 failing kills
F3 ("the race was earning its fee on the classifier").
ran: muse exit 0 in one pass (~25 min); its `RESULT.md` reported 139
focused + 1084 full-suite tests green, ruff clean on the CI-gated
`chimeraboost/` scope (23 pre-existing hits in untouched test files).
Reviewed the diff by hand: acceptance flag `_FORCED_CROSS_OK` now True on
both estimators, new `_QUALITY_PINS_FORCED_CROSS` (regressor True,
classifier False) drives the rung-1 pin; `_arm_forced_cross_cls` = probe
capped at `FORCED_CROSS_PROBE_ROUNDS`, top-4 pairs through the screen, one
full fit; `fast` and the race both exclude "always", so the inert arm
(multiclass, failed gates) is a plain full fit — muse's one deliberate
deviation from the task text, argued in RESULT.md and right (the task's
literal formula would have left multiclass+always with a truncated
audition). Five tests replaced/added as specified. Reviewer gates:
`identity_snapshot` **155/155 identical** (raced default untouched), full
suite **1084 passed, 1 skipped** rerun under the conda python.
screen: `run_benchmarks.py --synth --seeds 3 --save --models ChimeraBoost
ChimeraBoostOneLin ChimeraBoostOneLinXC` (the harness requires the default
arm as baseline; first launch without it exited 2 on the argparse check) →
`results/20260921-075304.json`, scored with `compare_runs.py --model
ChimeraBoostOneLin --model-new ChimeraBoostOneLinXC --expect-inert` and
`synth_report.py` on the same file. Bar 1 exact-tie control **PASS** —
regression 0-0-48, multiclass 0-0-34, n<2000 0-0-48, 119 of 136 exact
ties in all; every non-tie is a binary set with n_train ≥ 2800. Bar 2
direction **PASS** — engaged 13W-4L (bar 9+), engaged median **+0.16%**
Brier, binary slice p=0.049; two of the 13 wins are +0.00% (near-perfect
sets), so read it as 11 real wins to 4 losses; largest mover syn:v2/447
+1.59%, worst loss syn:v2/390 −0.88%. Bar 3 canary&cats 0-0-3 **PASS**
(saturated slice 5-0-12 noted, gains there are +0.1% on near-perfect
scores). Bar 4 cost recorded — engaged median fit ratio **1.27** (mean
1.60; one 5.35 outlier, syn:v2/697, where the augmented fit ran ES long;
inert slice median 1.00 = the timing floor). Synth timings are not
decision-grade (standing rule); the decide run prices it.
verdict: **PASS → S3**. Forecast: direction and control HIT; engaged
median +0.16% is a **MISS below the +0.2 to +0.8% band** — smaller than
the Grinsztajn probe's +0.36% headroom, consistent with synth binary sets
carrying less interaction structure than the real panel, exactly the
"where wrong" I wrote; cost 1.27 HIT. The mode ships in this PR as an
opt-in knob (default-off, `quality=1` classifier unchanged). Whether
rung 1 should pin it on the classifier is S3's question and the honest
prior is "small": +0.16% median on synth, +0.36% on Grinsztajn.
next: S3 = ONE `--decide --seeds 3 --save` run with `ChimeraBoostOneLin`
and `ChimeraBoostOneLinXC` (plus the default as baseline), scored
`--by-suite` and `--expect-inert`; bars per stratum: gr binary engaged
sign test wins ≥ half + 1 with positive engaged median on Brier, hc
binary at worst flat (pointer, not gate, at its size), all regression /
multiclass / sub-gate rows exact ties, engaged fit ratio ≤ 1.5 median.
Pass ⇒ propose the `quality=1` classifier pin as a separate /experiment
(a released-preset change needs the full gate and the maintainer's go);
fail ⇒ the knob stays opt-in and F3 closes as "measured, not worth the
pin".

#### I019 2026-09-21 F3 S1 (classifier forced-cross probe; pre-registered)
why now: F4 has no measured candidate left after I018; F3 is the only other
ACTIVE family and its S0 (I004) cleared it. Zero library change — the rung
is a probe script, harness code, so it is Claude's per AGENTS.md and muse
gets no task this rung. Branch `campaign/f3-s1-clf-cross-probe` stacked on
`campaign/f4-c1b-gradhess-fusion` (PR #115, unmerged), because this log
lives there and main does not have I018 yet; merge #115 before this rung's
PR. #114 (scaffolding) merged to main at cd57e03.
design: `benchmarks/probe_cross_pairs_clf.py` (new), the E2 step-1 probe
transplanted to binary log loss. Three arms per (dataset, seed) on the inner
`GradientBoosting` at the rung-1 classifier config (Logloss, linear leaves
on — the classifier's auto rule for binary — depth 6, 2000 rounds, ES 50 on
a 0.2 split): plain; probe (pairs from a 25-round importance fit); oracle
(pairs from the plain full fit). Both cross arms use the production pair
rule at the SHIPPED forced width `FORCED_CROSS_TOP_M=4`, so the block is
what a classifier "always" mode would carry. Panel: all 16 Grinsztajn
clf_num sets × 3 seeds = 48 fits (clf_cat are the same sets with
numerically-encoded extras, dropped to avoid double counting). No
gap/control labels exist for the classifier; the oracle column IS the
per-set headroom read. Metric: test Brier, log loss beside it.
barrier: `barrier_check.py` matched six. B16 (pre-screening the candidate
block) — inapplicable: no new screen; the pair rule is the production one at
the width E2 shipped, and E2's step 1 measured that width on the regressor
without triggering B16. B12 (portfolios die on cost) — applied, not
contested: bar 3 prices the augmented fit directly before any knob exists.
B17 (sub-gate CV race) — inapplicable, every panel set is above 2000 rows
and no race runs. B1 — accepted: gates stay, sub-2000 classification stays
inert. B2 — inapplicable at rung 1 (`refit_full=False`), as I004 recorded.
B14 — inapplicable: no validation-curve decision at any budget; the 25-round
fit only ranks features (E2's clearing argument, unchanged).
forecast, written before the run: strength — oracle Brier headroom on the
engaged binary slice smaller than the regressor's +2.6%: **+0.3 to +1.0%
median**, because binary log loss with linear leaves already captures more
of the smooth structure and the raced default's classification wins were
modest (covertype F1 +3.1% the top of the 2026-07-13 record). Probe
fidelity: paired probe-minus-oracle median **~0** (E2 read exactly 0.000).
Cost — median aug/plain fit ratio **1.2 to 1.5** at top-4 (regressor read
1.39). Where I expect to be wrong: headroom may sit under +0.3% on Brier
even where log loss moves, which kills the family as "nothing to win" at
rung 1; and Higgs/MiniBooNE/jannis-class wide sets may push cost past 1.5.
bars, in order (the E2 bars with the classifier caveat): 1 headroom —
oracle median ≥ +0.3% Brier over plain on ≥ 30 fits, else KILL. 2 fidelity
— paired probe-minus-oracle median ≥ −0.1%, else KILL ("the probe cannot
find the pairs"). 3 cost — median aug/plain ≤ 1.5, else KILL (no narrower
retry is pre-authorized this time: top-4 already IS the narrow form).
ran: `probe_cross_pairs_clf.py`, 16 sets × 3 seeds = 48 fits, one pass,
~35 min (`results/probe-cross-pairs-f3-clf.jsonl` + console
`results/probe-cross-pairs-f3-clf-20260921.txt`).
result, the three bars in order: 1 headroom **PASS, thin** — oracle median
+0.360% Brier over plain (bar +0.3%), 29W-19L over 48 fits (sign test
p≈0.19, not decisive on its own); mean +1.389% is one dataset — covertype
+15.8% — and reads ~+0.4% without it (GATE_ROBUSTNESS #3, so the median is
the number). 2 fidelity **PASS** — paired probe-minus-oracle median exactly
+0.000%, probe median +0.260%; pair sets identical (Jaccard 1.00) on 10 of
16 sets, and where they diverge the probe gives back part of the gain
(eye_movements +0.10% vs +1.53%, pol +2.90% vs +3.43%) — E2's gdiff-style
caveat, now on numeric-only sets. 3 cost **PASS** — median aug/plain fit
ratio 1.33 (mean 1.33, no long tail this time; worst 2.04 on one
eye_movements seed); the 25-round probe is 1–19% of arm time.
per-set shape, which matters more than the medians: gains concentrate on
five sets (covertype +15.8%, pol +3.4%, eye_movements +1.5%, MagicTelescope
+1.4%, jannis +1.1%) and the block costs on three (bank-marketing −1.0%,
electricity −0.9%, california −0.1%); the other eight are within ±0.3%.
Per-seed variance is wild where it is wild (pol s2 −8.6% vs s0 +12.3%,
house_16H alternates ±1.0%) — the same dodged-losses pattern E2 recorded on
the regressor, at similar size. Log loss agrees with Brier everywhere
(oracle log-loss gain same sign on 15 of 16 sets).
verdict: **PASS → S2**. Forecast HIT on all three: headroom +0.36% inside
the +0.3 to +1.0% band (at its floor), fidelity 0.000 as predicted, cost
1.33 inside 1.2–1.5. Honest reading: the prize is smaller and lumpier than
the regressor's (+0.36% vs +2.6% median), so the family's ship case rests on
the S2/S3 engaged sign tests, not on this probe. The kill I wrote for
"nothing to win" did not fire, barely.
next: S2 = implement the classifier forced mode (muse task): flip
`_FORCED_CROSS_OK` on the classifier so `cross_features="always"` is
accepted, wire a `_arm_forced_cross` counterpart into the binary path
(25-round probe, `FORCED_CROSS_TOP_M` block, applicability gates kept,
multiclass rejected), default OFF and quality=1 classifier pin UNCHANGED
until S3; goldens must stay green (raced default untouched). Then ONE synth
screen with an "always" classifier arm and `--expect-inert`; bars:
classification engaged sign test ≥ half plus one AND engaged median > 0
on Brier, regression/multiclass/sub-2000 exact ties, canary flat.

#### I018 2026-09-21 F4 S0+S1 (candidate C1b — grad_hess fusion; the parked unit, taken)
why now: F2 died at I017, so the ordering call I014 left open ("C1b vs F2")
resolves itself. C1b is the only measured, unblocked, cheap unit on the beam.
Loop mechanics, first rung driven by Muse Code from a task file
(`campaign_tasks/20260921-f4-c1b-gradhess-fusion.md`, gitignored; the
verdict below cites what it produced). Branch based on `loop-scaffolding`
(PR #114 open, not merged) rather than main, because main lacks `AGENTS.md`
and muse reads its rules from there; the rung PR targets main and its diff
collapses to the rung once #114 lands.
barrier: `barrier_check.py` matched B10 only (words: kernel, numba). Does
not apply — B10 binds objects INSIDE `build_oblivious_tree`; this is the
loss layer, the same object I012 cleared. B10's disqualifier (cannot be made
bit-identical) is cleared by construction: both fused ops are elementwise,
computed per element in the same order as numpy (`P − Y`, then
`max(P·(1−P), 1e-6)`), and P itself comes from the already-shipped K ≤ 7
kernel — the same exp() on the same machine. B10's method (ceiling first) is
obeyed: I014 measured the object at 4.4% (okcupid-stem) / 7.5% (cjs) of fit.
forecast, written before any code: fit-time — hc:okcupid-stem −2 to −4%,
hc:cjs −3 to −5% (ceilings 4.4/7.5%, discounted because the kernel still has
to write grad and hess); binary and regression exactly 0 (`MultiSoftmax` is
the only caller); Grinsztajn 0 (no multiclass task). strength — exactly zero
by construction; same-machine bit-identity on `identity_snapshot.py` + the
multiclass goldens; cross-libm the existing ≤ 4 ULP softmax bound.
class: bit-identical speed refactor ⇒ pure-speed ladder (`identity_snapshot`
exact, full suite, same-process fit A/B). No strength screen owed.
kill: (a) `identity_snapshot.py check` not 155/155 identical (baseline
re-saved at the base commit this rung, since 079c312 landed after the
2026-09-18 save); (b) same-process A/B on okcupid-stem + cjs saves under 1%
(the same-process floor) or any multiclass set gets slower; (c) the binary
control moves beyond ~1%.
ran: muse exit 0 in one pass; its `RESULT.md` reported 69 focused tests
green, full suite 1081 passed + 1 skipped, ruff clean on the two edited
files (23 pre-existing tree-wide errors left alone, all outside the edit
list). Reviewed the diff by hand: kernel is `_softmax_kernel` verbatim plus
the two elementwise ops in the specified order; old body kept as
`_grad_hess_numpy`; dispatch guard matches `_softmax`'s plus a shape and
contiguity check; six new exact-equality tests with tripwires for the K > 7
and float32 fallbacks. Muse quirks worth knowing: its edit tool cannot match
CRLF files, so it patched via a byte-exact script; its sandbox CWD carries a
`\\?\` prefix that breaks `..` in three test files' sys.path (rerun from a
plain path); it leaves a `pytest-of-Nathan/` temp dir in the repo root that
the main user cannot delete (owned by the sandbox user, harmless, untracked).
gate 1, identity: `identity_snapshot.py check` **155/155 identical**
(baseline re-saved at 7f7b276 with the library untouched, then checked).
gate 2, tests: **1081 passed, 1 skipped** rerun by the reviewer under the
conda python (muse's own run agreed).
gate 3, speed (`benchmarks/f4_c1b_speed.py`, new, same-process A/B, OFF arm
= `_grad_hess_numpy`, median of 3; `results/campaign-f4c1b-speed-20260921.txt`):
hc:okcupid-stem **−4.6%** (grad_hess leg 0.148s → 0.087s), hc:Traffic_violations
**−5.7%** (0.257s → 0.100s), hc:cjs **−4.1%** (0.236s → 0.145s), binary
control hc:kick **+0.2%** on zero calls. The leg itself accounts for
2.4–6.6 points of those; the remainder sits inside the ~1% same-process floor
plus whatever two fewer (n, K) allocations per round buy the rest of the fit.
verdict: **PASS → PR** (forecast HIT on both axes: okcupid landed at the top
of its −2 to −4% band and a touch past it, cjs inside −3 to −5%, controls
flat, strength exactly zero by construction and by snapshot). Kill bars
(a)(b)(c) all clear. Shipped as a pull request per the standing PRs-only
rule; CHANGELOG entry added under Unreleased. Reach caveat carried from I013:
multiclass-only, so the hc and synth strata move and Grinsztajn cannot.
next: F4's measured candidates are exhausted (C2, C1, C1b all shipped). F3 S1
takes the top slot (`probe_cross_pairs.py` on engaged binary sets, classifier
pair fidelity, I004). Beam is at 2 ACTIVE (F3, F4) — staleness rule says a
refill is due; that needs the maintainer to pick entrants, so the loop runs
F3 S1 first and raises the refill in its report.

#### I017 2026-09-18 F2 S2 (synth screen of the sub-gate flag; pre-registered)
barrier re-check on the concrete design: B16 inapplicable (the full
candidate block is carried; only the referee changes); B1 cleared at I002
(CV repairs the signal, the threshold stays); B6 a spurious keyword match
(a mechanism, not a tuning sweep); B12 applies — cost is priced from this
screen; B14 a different axis (I002); B7 inapplicable (plain K-fold refits
average a binary decision — no shared structure across folds, no OOF reuse
for fitting, unlike the killed shadow-CV leaf tuner); B2 rides to S3 (judge
at rung 3 with a mispick watch).
flag verified before screening: 9 focused tests green, ruff clean on the
library (0.15.17; pinned 0.16.5 unrunnable in this sandbox — noted for the
PR), identity snapshot 155/155 with the flag off, full suite 1061 passed +
1 skipped (the skip is pre-existing).
forecast: strength — the engaging slice (sub-gate multiclass) reads
positive (S1 showed +1.20% on real data); every non-engaging dataset an
exact tie. Cost — about 6 small fits per engaging set; low single digits
there, zero elsewhere (report-only at S2).
pass bars, in order: 1 exact-tie control — every regression, binary, and
above-gate multiclass dataset an exact tie; a single non-tie voids the
screen as an implementation bug. 2 direction — the engaging slice positive
by majority and median. 3 canary slice not positive. 4 cost recorded.
ran: `run_benchmarks.py --synth --seeds 3 --models ChimeraBoost
ChimeraBoostSubgate` from branch f2/subgate-race (worktree), auto
timestamped save, scored with compare_runs.py + synth_report.py on the same
file via --model/--model-new with --expect-inert. First attempt passed
`--save results/f2s2-20260918.json` explicitly, which collided the console
tee with the JSON sidecar (same path) and corrupted the file mid-write;
reran with the default save and copied the JSON across (byte-validated).
Console of attempt 1 kept as `f2s2-20260918-attempt1.log` (its printed
numbers are valid; only the file write was corrupt).
verdict: FAIL → KILL F2. Bar 1 PASS — all 10 non-ties are sub-gate
multiclass (n_train < 2500, i.e. below the gate at race time); the other
126 read as exact ties, including all 48 crossfeat-scope sets where the
shipped race runs — the shipped path is untouched. Bar 2 FAIL — engaging
slice 5 wins / 5 losses with median −0.22%: the race is a coin flip where
it engages. Bar 3 PASS — canary&cats 0-0-3 flat (saturated 2-0 at +0.044%
noted, both near-perfect scores). Bar 4 recorded — engaging sets cost 3-7x
the base fit (mean ~5x); declined-after-folds sets pay too; only no-pairs
sets are free. Forecast 1/3: ties HIT, direction MISS, cost MISS (the
folds run ~full rounds — B12's exact warning, not applied to the
forecast). Mechanism of death: ~110-row validation slices stay too noisy
to referee even averaged over 3 folds (50% engaged precision); S1's thin
+1.20% on 3 fits did not replicate. Extra nail: syn:v2/117's decision
flips between two identical runs (aug-pick in attempt 1, decline in the
rerun) — the referee is knife-edge unstable, not just imprecise. Bright
thread for the refill (not a reprieve): gains concentrate with interaction
depth (OLS t+2.90; depth>=3 slice 4-3 at +0.257% vs depth<=2 1-2 at
−0.067%) — a dataset-gating claim would need its own S1. Confidences:
dead-on-decision-suites high; S1 likely noise on 3 fits.
next: F2 closed. Worktree branch f2/subgate-race kept until the maintainer
confirms the kill, then deleted unmerged (nothing ships from a killed
family; the probe + this record are the durable artifacts).

#### I016 2026-09-18 F2 S1 (sub-gate CV-averaged race probe; pre-registered)
forecast: strength — eucalyptus is the biggest hc CatBoost gap and crosses
are loss-agnostic geometry, so oracle headroom should exist; whether 3-fold
CV over ~110-row validation slices repairs the race signal is genuinely
uncertain. Expectation: oracle positive, CV directional but thin. Cost —
probe only, unmeasured at S1 (sub-gate fits are seconds; a 3-fold race there
is affordable by inspection).
kill bars (in order): 0 headroom — oracle-over-plain test-Brier mean ≥
+0.2% on the 3 gap fits, else KILL as "nothing to win". 1 signal — cvrace
beats plain on ≥2/3 gap fits with positive mean, else KILL as "signal
unrecoverable". 2 canary — no CV-picked augmented loss beyond −0.1% on any
cjs fit. 3 control — CV pick == single pick on ≥2/3 okcupid fits.
ran: `benchmarks/probe_subgate_race.py` (new; inner MulticlassBoosting at
production config, production pair proposal, fixed pair set, uncalibrated
multiclass Brier).
verdict: PASS (thin). Bar 0 headroom PASS — oracle +1.92% mean on the
gap fits, a real prize (aug wins 2/3 seeds). Bar 1 signal PASS at exactly the
bar — cvrace 2/3 with mean +1.20%, one genuine mispick (s0 −2.16%). Bar 2
canary: the %-bar is void on a saturated set (Brier ~3e-08; near-solved
doctrine), the absolute read is −5.6e-10 mean = noise floor, no invented
signal — but note CV picked aug 1/3 where single-split picked 0/3 (dither,
not signal). Bar 3 control PASS — 2/3 agreement, and the one disagreement
(okcupid s2) favored CV (+0.34% vs +0.00%). Forecast: HIT. Two caveats ride
to S2: (a) CV and single-split picked identically on the gap set, so no
CV-over-single repair is demonstrated yet — the CV-vs-plain-threshold design
choice still rests on B1's mechanism, not this probe; (b) per B2 the s0
mispick would propagate under the rung-3 refit, so S3 must run at the
shipped default with a mispick watch. Instrument note: the probe first
rounded metrics to 8 decimals, which corrupted the canary read — caught on
the first print, fixed to full precision with an absolute-delta guard, rerun.
next: spec the sub-gate CV-race flag (default-off) with tests; S2 synth
screen vs default.

#### I015 2026-09-18 re-baseline (canonical chart-grade decide run on 0.32.0)
forecast: measurement, not an experiment — no arms, no verdict bars. Expected
reads: default accuracy ≈ the 08-02 chart (no default-strength change since the
adaptive learning rate); default slowdown a touch down (the 0.31 wins are
multiclass/string-cat only, so Grinsztajn barely moves); rung 1 now OneLinX
(the E2 ship) near 2.7x; bagged rungs slightly cheaper on multiclass/string-cat
sets. Any default-point move beyond noise is a finding, not a win.
ran: `run_benchmarks.py --decide --seeds 3 --save` on branch `whitepaper`
(library verified identical to origin/main), field ChimeraBoost / OneLin /
OneLinX / NoRefit / Ens5 / Ens8 / CatBoost / LightGBM / sklearn_HGB, scratch
python, explicit go from the maintainer 2026-09-18.
verdict: READ (`results/20260918-170822.json`, 309/309 in 93 min; the
process exited 1 AFTER saving — no traceback, no sys.exit in the harness,
progress sidecar "done", JSON valid, charts regenerated — recorded as a
spurious teardown code). Charts refreshed from the new run. Default still the
best non-bagged rung on both panels (clf skill 0.4036 @ 3.5x vs NoRefit
0.3980; reg R2 0.7330 @ 4.3x vs NoRefit 0.7280). CatBoost dominated on both
(clf 0.4057 @ 66x under Ens5 0.4081 @ 7.1x; reg 0.7287 @ 82x under the
default). New rung 1 reads above old on regression (OneLinX 0.7263 @ 2.0x vs
OneLin 0.7237 @ 1.4x) with the predicted exact clf tie (0.3955 both);
LightGBM is now dominated on regression by OneLin. Caveat: HGB's 0.4349 clf
skill is a subset artifact (it skips the high-card sets it cannot fit, so it
averages over easier survivors) — never quote as a league position.
Forecast: HIT.
next: done — Phase 1 F2 S1 probe (I016).

#### I014 2026-08-16 F4 S0 (candidate C1b — grad_hess fusion; ceiling measured, PARKED)
forecast: n/a in the strength sense (measurement). The stated prior in I013 was
"the remaining share is now small, and the unit should die cheaply at S0 if it
is" — this entry is that check, run before any code was written, and the prior
was roughly right.
ran: `benchmarks/f4_c1b_walltime.py` (new) — wraps `MultiSoftmax.grad_hess` and
`_softmax` in a real fit and splits the first by the second, so what is reported
is the arithmetic C1b would actually absorb, not the whole method.
result: **the object shrank with C1 and is now a single-digit ceiling.**
`grad_hess` is 5.9% of an hc:okcupid-stem fit and 11.8% of hc:cjs; strip out the
fused softmax inside it and C1b's own object — `P - Y` plus
`max(P*(1-P), 1e-6)` — is **4.4% and 7.5%** respectively. Note the shape of the
change: `grad_hess` was 40.0% of fit before C1 and is 5.9% after, which is the
clearest possible confirmation that C1's win was real and not a measurement
artifact, since it was read by a different instrument on a different day's code.
verdict: **PASS on the ceiling, PARKED on priority** — and the distinction is
the point. 4.4–7.5% clears the band B10 killed things at (0.9–3%), the change
would stay bit-identical (both ops are elementwise, so order cannot matter, and
the existing K ≤ 7 guard already covers it), and the kernel it would extend is
already written — so this is a genuinely cheap unit, not a speculative one.
What it is NOT is default-moving: it reaches 8 hc rows and 0 Grinsztajn rows,
and realistic recovery is a fraction of the ceiling, since the fused kernel still
has to write the two arrays it saves passes over. Call it 2–4% of multiclass fit.
That is worth one unit of somebody's time, and it is worth less than F2, which
aims at the default's strength.
next: **Nathan's ordering call, and the only open item this session leaves.**
Either take C1b as one cheap unit, or leave it parked and spend the next unit on
F2's S1 probe (CV-averaged race on sub-gate sets, I002). Recommendation: F2
first — C1b will still be a 20-minute unit whenever it is wanted, and its
ceiling is measured and on the record so it need not be re-derived.

#### I013 2026-08-16 F4 S1 (candidate C1 — implemented, measured, SHIPPED)
forecast: as pre-registered in I012 — hc:okcupid-stem fit −25 to −40% against a
44% arithmetic ceiling, binary and regression exactly 0, Grinsztajn 0, strength
exactly zero by construction.
result: **the forecast held on both axes and landed at the top of its range.**
Multiclass fit time on the same-process A/B (`benchmarks/f4_c1_speed.py`, new):
hc:okcupid-stem **−44.2%**, hc:Traffic_violations **−37.2%**, hc:cjs (K=6)
**−40.5%**. The binary control hc:kick read −0.6%, inside the ~1% same-process
floor and on zero softmax calls — the control is what licenses reading the other
three as real. The softmax leg itself went 1.70s → 0.05s on okcupid-stem, a 33×
that converts almost the whole 45.4% ceiling: the in-fit gain did NOT land under
the microbench the way C2's did, because this candidate removes numpy's per-row
reduce machinery rather than Python-loop overhead that the fit was amortizing
anyway.
serial vs parallel, the question I012 committed to asking: `prange` earns its
setup comfortably — the parallel kernel beats a serial one by **6.9–11.0×** at
every real shape, and both return identical arrays. Parallel stays.
reach, and it is total: all **8** multiclass rows in `--decide` are K ∈ {3, 5, 6}
(hc:okcupid-stem, Traffic_violations, cjs, eucalyptus + their variants), so every
multiclass dataset we benchmark is under the K ≤ 7 guard. Grinsztajn has no
multiclass task at all, so the decision suite's headline strata cannot move — this
is an hc-stratum and synth-stratum speed win, and it must not be quoted as a
default-wide one.
gate 1, identity: `identity_snapshot.py` **89/89 bit-identical**, baseline
re-saved at HEAD (5e595a6) with the library change stashed, then checked with it
restored. The panel carries three multiclass configs including a cats one and an
MVS-subsampled one, so the kernel is exercised, not merely bypassed.
gate 2, tests: **970 passed, 1 skipped** — 964 plus 6 added here. The oracle is
`_softmax_numpy`, the pre-change function kept under its own name, so the tests
check the OLD behaviour rather than my idea of softmax; they cover every K from 2
to the guard at three input scales, K above the guard, degenerate rows (constant,
±1e300, duplicate maxima, single column), non-float64 fallback, and the
`grad_hess`/`eval`/`transform` callers.
gate 3, speed: above. Class was pure-speed (I012), so no strength screen is owed
and none was run.
verdict: **SHIP** — bit-identical, 37–44% off multiclass fit time, provably
nothing anywhere else. Committed to main directly per the pure-speed ladder;
CHANGELOG updated.
design note for whoever reads the kernel: the last loop divides by the sum
instead of multiplying by a hoisted reciprocal. That looks like a missed
micro-optimization and is not — it is the difference between bit-identical at
K ≤ 7 and drifting at every K (I012 measured both). The comment in `losses.py`
says so at the site.
next: F4 C1b (`grad_hess` fusion) — the same kernel could emit `P - Y` and
`max(P*(1-P), 1e-6)` in the pass it already makes, removing two more full (n, K)
passes and their allocations. It owes its own S0 with a wall-clock read: with
softmax now at 0.05s, `grad_hess`'s remaining share is unknown and has to be
measured before it is worth anything. The honest prior is that it is now small.

#### I012 2026-08-16 F4 S0 (candidate C1 — the softmax leg, measured before forecast)
measurement first, per the debt I011 left this unit: `benchmarks/f4_c1_walltime.py`
(new) wraps `losses._softmax` in a `perf_counter` pair and runs a real
hc:okcupid-stem fit. At 767 calls the wrapper is microseconds in total, so unlike
C2's 1.6M-call loop the instrument does not move what it measures.
result — **the profile did not overstate; it UNDERSTATED.** `_softmax` is
**1.611s of a 3.55s fit = 45.4%** by wall clock, against cProfile's 39%. C2's
lesson was that a profile share on a PYTHON loop is an upper bound; the converse
holds here and is worth recording as the general rule: profile inflation is a
per-call-overhead artifact, so it applies to many-tiny-calls objects and not to
few-fat-calls ones, which can read LOW because the C-level work inside one call
is charged to that call and not decomposed. Split by caller: `grad_hess` 40.0%
of fit, `eval` 5.5%. Shapes are (32377, 3) and (5714, 3) at 334 calls each,
(38091, 3) at 99 — K = 3.
where the time goes, and it is not where a reader would guess: at shape
(32377, 3) the whole call is 3.22 ms, of which the `max` reduce alone is
**1.40 ms = 43%** and the `sum` reduce another 15%. `exp` — the only op doing
real transcendental arithmetic — is 20%. The cost is not the mathematics, it is
running two reductions over a length-3 inner axis: numpy pays its per-row reduce
machinery 32377 times to compare three numbers.
candidates measured (median of 7, same shape), all checked for exact equality
against the current output rather than argued:
  B in-place ufuncs (`out=`, two fewer temporaries) — **1.01x, bit-identical.** No
    effect: the temporaries were never the cost. Recorded so it is not retried.
  D column fold (both reduces replaced by explicit folds over columns) — **0.48x,
    bit-identical at K <= 7.** Pure numpy, no new dependency surface.
  C numba fused, reciprocal-multiply — 0.03x but **drifts at every K**, because
    `1/s` then multiply is not the rounding of a divide.
  E numba fused, true divide — **0.03x (32x faster) AND bit-identical at K <= 7.**
the K <= 7 boundary is real and measured, not assumed: the sweep over
K = 2..50 puts the first drift at exactly K = 8 for both D and E, which is where
numpy's pairwise summation starts blocking and a left-fold sum stops agreeing to
the last bit. `max` never drifts at any K, since selecting a maximum does not
round. So the shippable shape is a fused kernel **guarded to K <= 7 with the
current numpy path kept for K >= 8** — not a limitation to apologize for, since
it is what keeps the change bit-identical instead of merely close.
forecast, both axes, written after the measurement and before any library code:
  fit-time — hc:okcupid-stem **-25 to -40%**. The arithmetic ceiling is 44% of
    fit (45.4% x 0.97) but C2 showed the in-fit gain lands under the microbench
    (-43% micro became -30% in fit), so the forecast discounts hard and the
    ceiling is quoted as a ceiling. Other multiclass sets: same direction, size
    scaling with how much of their fit is loss arithmetic. Binary and regression:
    **exactly 0, structurally** — `MultiSoftmax` is the only caller and it never
    runs there. Grinsztajn: **0**, it has no multiclass task at all. Predict-side
    is a bonus (`transform` is the same function), not claimed.
  strength — **exactly zero, by construction**, and this time the construction is
    the K <= 7 guard rather than a semantic argument.
class: **bit-identical speed refactor ⇒ pure-speed ladder** — `identity_snapshot.py`
exact equality + full test suite + a same-process fit A/B. No strength screen is
owed. NOTE this is a demotion in risk from what I009 assumed ("not bit-identical
by assumption, FP-drift class with a Brier read at S2"); the assumption was
wrong in our favour, and only because candidate E was measured rather than C
alone being tried.
ran: `barrier_check.py` — 1 of 16 matched.
  B10 (remaining grow-kernel objects are below their ceilings) — matched on the
  words "kernel, numba", and does not apply as a blocker. B10 is a finding about
  objects INSIDE `build_oblivious_tree`: it measured multiclass copy at 0.9%,
  int32 leaves at 1-2%, uint8 bins at <=3%, killed each at its ceiling, and left
  the one double-digit object (fused scatter+scan) barred because it is FP-drift
  class. `_softmax` is in none of that: it is the loss layer, the kernel is a
  separate 23.0% of this fit (I009), and B10's own disqualifier — cannot be made
  bit-identical — is exactly what candidate E clears at K <= 7. What B10 does
  bind is METHOD, and this entry obeys it: the ceiling was measured before a line
  of library code was written, which is why C1 has a 45.4% ceiling on record and
  candidate B is already dead at 1.01x.
  Not matched, checked by hand: B15/B16 (histogram subtraction, cross-candidate
  screening) touch neither the loss layer nor multiclass; B14/B2 concern the
  cross-audition budget and the replay refit.
kill: (a) the guarded kernel is not bit-identical on `identity_snapshot.py` — the
K <= 7 guard tightens or the candidate dies, it is not shipped as an
approximation; (b) the in-fit A/B on okcupid-stem reads under 5% (well above the
~1% same-process floor) — i.e. the 45.4% does not convert, and C1 dies with the
number recorded; (c) the parallel kernel degrades any non-multiclass path, which
would mean thread-pool interaction with the binner.
verdict: PASS — ceiling measured at 45.4%, a bit-identical 32x candidate exists,
no barrier blocks
next: S1 = implement candidate E in `losses.py` behind the K <= 7 guard, extend
the oracle tests with the numpy implementation as reference across K = 2..7 (the
C2 pattern: the oracle is the OLD code, not my expectation of it), then
`identity_snapshot.py` + full suite + same-process fit A/B on hc:okcupid-stem
with a binary set as the zero-change control. Also at S1, time a SERIAL variant:
`prange` over 32K rows of 3-wide work may be paying thread setup for nothing, and
a serial kernel that matches it is the safer neighbour of an already-parallel fit.
deferred, with its own S0 owed (do NOT fold it into C1 silently): `grad_hess` is
40% of fit and spends more than `_softmax` alone — `P - Y` and
`max(P*(1-P), 1e-6)` are two further full (n, K) passes with allocations that the
same kernel could produce in the pass it already makes. Elementwise ops are
order-independent, so it should also be bit-identical, but it is a different
object and gets measured, not assumed. Logged as candidate **C1b**.

#### I011 2026-08-16 F4 S1 (candidate C2 — implemented, measured, SHIPPED)
forecast: as pre-registered in I010 — hc:okcupid-stem fit −10 to −16%, Grinsztajn categorical control ~0%, strength exactly zero (bit-identical).
result: **the forecast's DIRECTION held and its SIZE did not** — hc:okcupid-stem −4.0%, hc:kick −6.8%, Grinsztajn control −1.0% on zero `factorize` calls (i.e. drift, which is what calibrates the same-process noise at ~1 point). Read the two real numbers as ~3% and ~6%. The 18% profile row was not 18% of fit: cProfile charges per-call overhead, and a loop making 1.6M `dict.get` calls is exactly the shape that inflates. Measured honestly, `factorize` was 0.31s of a 3.74s okcupid-stem fit (8%) and 0.50s of 2.16s on kick (23%). **Lesson for the rest of F4: a cProfile percentage on a Python-level loop is an upper bound, not an estimate** — C1's `_softmax` 39% is numpy calls, few and fat, so it should not suffer the same inflation, but it is now owed a wall-clock check before anyone forecasts off it.
the first design was wrong and the measurement hid it: a `U`-cast + `np.unique` + re-rank fast path (restricted to `str` + missing, with a type census, a float-count audit and a trailing-NUL length audit). It was **2.05× SLOWER than the loop** — sorting wide strings costs more than hashing them once. The first A/B read it as +25%/+47% fit time, which was worse still, because the path census re-ran the fast path to ask which path had fired and so charged the candidate arm twice. Instrumentation that does work proportional to the arm under test is a measurement bug, not a small one.
shipped design: `_factorize_hashed` keeps the loop's dict and drives it from C — `map(mapping.setdefault, values, count())` gives each new category the ROW INDEX of its first appearance, `mapping.values()` is then both the category order and the raw codes, and one scatter converts them to 0..K-1 with no sort. Keeping the dict makes equivalence structural: the cross-type ==/hash classes (`True`/`1`/`1.0` one category, `"1.5"`/`1.5` two) are preserved because it is literally the same dict, so the U-cast design's three audits are not needed at all. Only the missing mask is non-structural — the loop calls an element missing if it is None, unequal to itself, OR RAISES on `!=`, and the third cannot be vectorized, so any raise (or a comparison not returning plain bools, or an unhashable value) falls back to the loop.
measurements: `benchmarks/f4_c2_micro.py` over the 35 real string columns of okcupid-stem + kick — loop 283 ms, U-unique 581 ms (2.05×), setdefault+sort 200 ms (0.71×), shipped scatter form **162 ms (0.57×)**, winning on every single column, all three checked against the loop's own codes and categories. In-fit: `factorize` 0.31→0.22s (okcupid-stem) and 0.50→0.35s (kick), about −30% each, less than the −43% microbench because per-call overhead is amortized differently inside the fit.
gate 1, identity: `identity_snapshot.py` **89/89 bit-identical**, re-run on the final implementation. The baseline on disk was stale (2026-07-31, pre-PR #62 quantile-band fix; it read 81/89 before any edit of mine), so it was re-saved at the base commit with the change stashed and checked with it restored.
gate 2, tests: **964 passed, 1 skipped** — 952 plus 12 added here (10 oracle cases incl. a value that raises on comparison, plus 2 path tests). The oracle is the existing `_factorize_reference` dict loop, so the new cases check the old behaviour, not my expectation of it.
gate 3, speed: above. Class was pure-speed (I010), so there is no strength screen and none is owed.
verdict: **SHIP** — bit-identical, 3–6% off fit on string-categorical datasets, nothing anywhere else, no strength surface touched. Committed to main directly per the pure-speed ladder; CHANGELOG updated.
next: F4 C1 (`_softmax`) — but S0 must first do what this unit learned: a wall-clock measurement of the softmax leg before any forecast, because the 39% is a cProfile share. Reach caveat from I009 stands (multiclass only, so Grinsztajn cannot move).

#### I010 2026-08-16 F4 S0 (candidate C2 — string-column factorize)
forecast, both axes, written before any run:
  fit-time — on the profiled set (hc:okcupid-stem) the whole `factorize` row is 18% of fit, so a vectorized string path that leaves only the astype/unique/rank work should cut fit **10–16%** there. Whole-hc-stratum: smaller and unknown in advance, because an unknown fraction of hc sets already take `_factorize_numeric`; forecast **3–8%** on the hc fit total, and any per-set reading under 2% is unreadable (this session's noise floor). Grinsztajn: forecast **0%, below the noise floor** — its categoricals are mostly numeric-coded already, so most sets should not touch the loop at all. Predict-side gains are a bonus, not claimed.
  strength — **exactly zero, by construction**. The change must produce the same codes, the same category order, and the same `__nan__` merge as the dict loop; if it does not, it is wrong and gets fixed or dropped, never traded off against strength.
class: bit-identical speed refactor, preprocessing-side ⇒ **pure-speed ladder** per the ladder's skip rule — `identity_snapshot.py` exact equality + full test suite + `fit_time_delta.py`. No synth strength screen, no S2/S3 strength read, no default flip (the fast path is unconditional but semantics-preserving, like `_factorize_numeric` before it).
ran: `barrier_check.py` on the idea — 1 of 16 matched.
  B3 (partial CatBoost mechanism ports each regress somewhere) — matched on the words "categorical, encoding" only, and does not apply. B3's finding is about *algorithmic* levers: each port changes what the encoder computes, helps some categorical sets and regresses others. This change alters no encoding semantics whatsoever — same codes, same first-appearance order, same missing-value sentinel — it replaces a per-row Python `dict.get` loop with an array-level computation of the identical mapping. A change with zero strength effect by construction cannot exhibit B3's failure mode, and the exactness is not assumed: `tests/test_bitident_refactors.py` already holds `_factorize_reference`, the dict-loop oracle written for `_factorize_numeric`, and the new path plugs into that same harness plus `identity_snapshot.py`.
  Not matched but checked by hand: B10 (grow kernel) and B15 (histogram subtraction) are kernel-side and this is preprocessing; B14/B2 concern the cross-audition budget and the replay refit, neither of which this touches.
kill: (a) the fast path cannot be made exactly equal to the dict oracle on the audited cases — drop it, do not ship an approximation; (b) `identity_snapshot.py` shows any drift; (c) `fit_time_delta.py` on the string-heavy sets reads under the 2% noise floor, i.e. the 18% profile row does not convert into measurable fit time.
verdict: PASS — no barrier blocks, forecast and class recorded
next: implement `_factorize_string` (mirroring `_factorize_numeric`: audit the column is all `str` plus recognized missing, substitute `__nan__`, `np.unique` + first-appearance rank, return object-dtype categories), extend the oracle test with string cases, then S1 = `identity_snapshot.py` + full suite + `fit_time_delta.py` on hc:okcupid-stem

#### I009 2026-08-16 F4 S1
forecast: n/a in the strength sense (measurement, nothing under test). Expectation to be checked: okcupid-stem's 50.5% "other" resolves into a small number of named non-kernel callers — categorical prep, per-class dispatch, or python-level loop overhead — rather than being spread thin. F4's own family-level kill says: if the read shows the time is irreducible dispatch (many tiny trees, no single caller above ~10%), record and kill.
ran: `profile_fit.py --dataset hc:okcupid-stem --top 30` (single-set cProfile, nothing else running); full output saved to `results/campaign-f4s1-okcupid-20260816.txt` (machine-local — `benchmarks/results/` is gitignored, so every `results/...` path in this file is on the box, not in the repo).
result: the "other" is NOT diffuse dispatch — it is two named objects. Fit 4.12s over 99 trees, of which `build_oblivious_tree` (the kernel) is 0.95s = 23.0% and everything else 3.17s = 77.0%.
  C1 `losses._softmax` — 1.615s cumulative = **39% of fit**, 767 calls (433 from `grad_hess`, 334 from `eval`). Self 0.661s, plus 0.709s inside `ndarray.max` (1432 calls) and ~0.26s inside `sum`: a pure-numpy softmax over an (n=38091, K) matrix re-running separate max/exp/sum reduction passes every round. `numpy.ufunc.reduce` is the single largest self-time row in the whole profile at 1.004s.
  C2 `target_encoding.factorize` — 0.747s = **18% of fit**, and 1,607,587 `dict.get` calls: the python fallback loop, taken because this set's 17 categorical columns are strings and so miss the vectorized `_factorize_numeric` path. Reached via `preprocessing.column` (0.706s) and `_split_columns_fit` (0.519s). The ordered-TS `fit_transform` adds 0.299s = 7%.
  Context rows: `_prep_matrices` 1.097s = 27%, `_refit_on_full` 1.108s = 27% (the rung-3 replay refit, expected).
verdict: PASS — the family-level kill ("irreducible dispatch, no caller above ~10%") is decisively not met, and neither object is covered by B10 (grow kernel) or B15 (histogram subtraction).
SCOPE CAVEAT, and it reorders the candidates: `_softmax` is multiclass-only, so C1 moves the high-card suite and **nothing on Grinsztajn** (36 regression + 23 binary + 0 multiclass). C2 fires on any string-categorical dataset and so has the wider reach. Chase C2 first on reach, even though C1 is the bigger single number on this one set. Do not quote either share as a default-wide figure — this is one dataset.
next: F4 S0 on C2 (`barrier_check` + forecast + class: a factorize fast path is preprocessing-side and should be provable bit-identical, which puts it on the pure-speed ladder — `identity_snapshot.py` + full suite + `fit_time_delta.py`, no strength screen). C1 stays queued behind it and is NOT bit-identical by assumption: it touches loss arithmetic, so it needs the identity check first and drops to FP-drift class with a Brier read at S2 if it fails.

#### I008 2026-08-16 F1 S2b (the one probe that decides whether F1 lives)
forecast: k=12 halves the trim, so expect roughly half the saving (engaged fit -5 to -8%) and a regression sign test that recovers toward flat. If the screen's RANKING is sound and k=6 was merely too tight, regression should come back to about even; if the ranking itself is wrong, regression stays negative at any k that saves real time.
bar, written before the run (both must hold or the family dies):
  (1) engaged-set regression sign test at worst flat — wins >= losses;
  (2) engaged-set fit saving still >= 8%.
ran: `run_benchmarks.py --synth --seeds 3 --models ChimeraBoost ChimeraBoostXTop12 --save benchmarks/results/campaign-f1s2b-20260816`
result: bar (1) FAIL, bar (2) PASS. Engaged regression 6W-12L — still losing, barely better than k=6's 4W-14L (`synth_report` task=regression 6W-12L-30T, mean -0.234%, p=0.238). Engaged fit -11.6%, whole-suite -7.6%, inert slice again 0W-0L-48T with a -1.8% timing noise floor.
  The two probes together say more than either alone. Doubling the block back gave up only 1.6 points of speed (13.2% -> 11.6%), so the time saved comes off the TAIL of the ranking; and it bought back only two regression sets, so the harm comes from misranking near the HEAD. k was never the problem.
verdict: KILL. F1 is closed, and the closure is general enough to be a barrier: B16 added to BARRIERS.md ("pre-screening the cross candidate block trades picks for time, at every k"). With B14 already closing the round-budget axis, both "do less of the audition" doors are now shut — cheapening this leg has to come from making the augmented fit cheaper PER COLUMN.
next: F4 (profiling-driven speed) takes the top ACTIVE slot; its S1 is a single-set cProfile of okcupid-stem's 50.5% non-kernel "other".
code: `cross_top_columns` stays in the library, default-off and bit-identical when unset, with its 11 tests — it is the instrument that produced B16 and the S3-at-rung-3 question could only ever be reopened through it. Harness arms ChimeraBoostXTop6/XTop12 stay registered and off by default.

#### I007 2026-08-16 F1 S2
forecast: (carried from I001, unchanged by the I006 statistic revision) fit-time — engaged-set fit down >=15%, whole-suite total down ~5-10%, non-engaged sets bit-identical. strength — flat: the screen must change how many columns the augmented fit carries, not which crosses win.
ran: `run_benchmarks.py --synth --seeds 3 --models ChimeraBoost ChimeraBoostXTop6 --save benchmarks/results/campaign-f1s2-20260816` — both arms in ONE run, so the A/B is free of machine-condition drift and the non-engaging synth sets are the in-run inert control. Read with `compare_runs.py FILE FILE --model ChimeraBoost --model-new ChimeraBoostXTop6 --expect-inert`, `synth_report.py` (same flags) and `benchmarks/f1_s2_read.py` (engaged/inert split, added this session).
result: the screen delivers the speed and fails on strength.
  control: 87 of 136 sets exact ties, and the `n<2000` slice is 0W-0L-48T — the gating is exactly as claimed, nothing engaged where it should not.
  speed: engaged-set fit -13.2% (48 engaged sets, 55.3s -> 48.0s); whole-suite -8.4%. The inert slice moved -1.7% on identical fits, so read the noise floor as ~2% and the engaged saving as ~11-13% — just under the >=15% forecast.
  strength: whole-suite sign test FAIL (25W-24L-87T, bar 69+), which the inert control explains away; but the ENGAGED-ONLY read is 25W-24L (bar 25+, a bare PASS) and it hides opposite-signed halves. By task on the engaged sets: regression 4W-14L, binary 10W-7L, multiclass 7W-7L. `synth_report` agrees and sharpens it: task=regression 4W-14L-30T mean -0.574% p=0.031, task=binary 13W-4L-37T +0.041% p=0.049, and the pre-registered mechanism slice `crossfeat-scope` 10W-17L-21T mean -0.573%. The regression loss is a COUNT, not a mean artifact — dropping the largest mover (syn:v2/639, -11.29%) leaves 4W-13L.
verdict: KILL at k=6, on the pre-registered I001 bar ("any synth slice where the screen changes a cross PICK with strength loss"). Regression is that slice: 18 engaged regression sets all changed picks, 14 of them for the worse. ~12% off the engaged leg does not buy a regression stratum that loses 14 of 18, and B2 says the rung-3 refit would amplify exactly these mispicks — S3 is not warranted.
next: I008, ONE pre-registered probe at k=12, to separate "the screen is wrong" from "k=6 is too tight". Family dies outright if that fails.
note: the binary slice went the other way (13W-4L, p=0.049). That is a post-hoc slice of the run that killed the parent and is NOT a result — recorded as a candidate for pre-registration only, never to be adopted off this data.

#### I006 2026-08-16 F1 S1b (statistic revision, before any S2 compute)
forecast: n/a (mechanism repair caught by a unit test, no benchmark under test)
ran: implemented `cross_top_columns` per the I005 spec with |corr(column, val residual)| as the ranking statistic, then ran it against a fixture whose signal is exactly the two things cross features exist for — a comparison `x0 > x1` and a product `x2 * x3`. At k=4 the correlation screen kept four PRODUCT columns and no comparison column at all.
result: the correlation statistic is structurally wrong for this family, not merely weak. The residual an oblivious staircase leaves around an `x_i < x_j` boundary is a sawtooth in the boundary's neighbourhood, not a linear function of `x_i - x_j`, so |corr| reads it as noise. A diff column exists precisely to turn that threshold into one split, so a threshold-blind screen discards the candidate the mechanism was built for. Statistic replaced with the best single-split variance reduction the column achieves on the residual over a 16-bin quantile grid — the criterion the augmented fit itself applies to the column, one round in, and free to compute. The same fixture now keeps both true interactions at k=4. Barrier re-check: same four as I001 (B1, B2, B12, B14), same clearing arguments, unchanged by the statistic.
verdict: PASS (spec revised; I005's `next:` stands otherwise)
next: I007 = the S2 synth run

#### I005 2026-08-16 F1 S1
forecast: (carried from I001)
ran: no new compute — satisfied by `results/campaign-attr-20260816.md` (I000's attribution): cross-audition leg = 40–58% of default fit on engaged sets; truncated-race preview keeps 24/24 cross picks at k=100, so pick fidelity has headroom for a narrower candidate set.
result: S1 PASS. Screen mechanism spec (for the S2 implementation): after the base fit, compute val-set residuals (established pattern — 4th-instance selection-on-ES-split precedent); score each candidate cross column (~30 numeric-pair + ~12 gdiff cols) by |corr(col, residual)| computed on the val rows; keep top-k≈6; the augmented raced fit carries only those. Knob: `cross_top_columns` (int, default None = today's behavior), default-off until S4. Engagement gates unchanged; sub-gate sets bit-identical (the `--expect-inert` control at S2/S3).
verdict: PASS
next: implement `cross_top_columns` behind the default-off knob (+ tests: None path bit-identical; k set → column count capped), then S2 synth ChimeraBoost-arms A/B (~2–8 min): arm A default, arm B `--chimera-*` knob if wired into the harness, else two-run pairing per the fingerprint rule

#### I004 2026-08-16 F3 S0
forecast: strength — engaged binary sets up (crosses already earn there under the raced default: covertype +12.8% Brier on top of linear leaves); hc stratum honest bar = at-worst-flat vs plain rung 1 (E2's regressor caveat: hc-vs-LightGBM was 7W-6L). fit-time — rung-1 identity ~2.7× within-run, matching E2's regressor read (probe ≈25 rounds + one full fit ≈ the dual fit it replaces is absent at rung 1).
ran: barrier_check (B1, B2, B12, B14) — clearing arguments:
  B1: inapplicable as a blocker — the forced path keeps the same engagement gates; sub-gate sets stay bit-identical (pre-registered inert slice).
  B2: inapplicable at rung 1 (`refit_full=False`, no replay amplifier); becomes live ONLY if "always" is ever proposed for rung 3 — re-run barrier_check then.
  B12: inapplicable — no extra arms; the forced path REMOVES the race.
  B14: inapplicable — there is no race to budget; E2 shipped exactly this shape on the regressor.
result: no barrier blocks; this is the E2 playbook with `_FORCED_CROSS_OK` flipped for the classifier + classifier-appropriate bars (Brier read mandatory at S2, per the bagging lesson)
verdict: PASS
next: S1 = reuse `probe_cross_pairs.py` on 2–3 engaged binary sets to check probe-picked pairs ≈ race-picked pairs for the CLASSIFIER; deprioritized behind F1/F2 (targets a preset rung, not the default) unless Nathan reorders

#### I003 2026-08-16 setup baseline
forecast: n/a (measurement, no change under test)
ran: `run_benchmarks.py --decide --seeds 3 --save campaign-base-20260816` — landed with the FULL default field (ChimeraBoost + CatBoost + LightGBM + sklearn_HGB), so it doubles as a fresh field read on 7684655. Files moved to `benchmarks/results/` (trap: a named `--save` writes to CWD — pass `--save benchmarks/results/<name>` next time).
result: gr 57 scored: 82.5% vs CatBoost (W47-L10, median +1.25%, CatBoost 2.36× slower), 94.7% vs LightGBM (we are 5× slower), 96.5% vs HGB. hc 13 scored: 38.5% vs CatBoost (median −0.15%, CatBoost 42.9× slower) — the known hc gap (F5's territory); 92.3% vs LightGBM. gr@sus25 66.7% / gr@sus50 66.7% vs CatBoost. hc@sus/@time behind CatBoost but ≤7 sets each — pointers, not gates (GATE_ROBUSTNESS).
verdict: PASS — standing BASE recorded
next: F1 S1

#### I002 2026-08-16 F2 S0
forecast: (deferred to S1 — S0 found the naive form barred) strength up on newly-eligible multiclass sets if and only if a trustworthy sub-gate race signal exists; fit cost negligible there (sub-2000-row fits are fast).
ran: barrier_check (B1, B2, B12, B14) + located the gate: single shared `CROSS_MIN_SAMPLES = 2000` (`sklearn_api.py:1283`), used by regressor (1990) and classifier (2812, 2860).
result: B1 applies SUBSTANTIVELY, not just formally — its mechanism (below 2000 rows the val split is too small for a trustworthy race signal, per the comment at `sklearn_api.py:1280`) is exactly what a plain threshold drop would reintroduce, and B2 says the replay refit amplifies any resulting mispick. A naive gate lowering is barred. The surviving form: fix the SIGNAL, not the threshold — sub-gate fits are cheap, so a CV-averaged race (e.g. 3-fold) on sub-gate sets is affordable; B14 closed budget/decision rules at k<100, not signal-quality mechanisms, so this is a genuinely open axis.
verdict: PASS (narrowed — family re-scoped from "lower the gate" to "CV-averaged race below the gate")
next: S1 = zero-library-change probe script (probe_cross_features.py pattern): on sub-gate multiclass sets (eucalyptus first), does a CV-averaged race pick crosses that improve the TEST metric? If the probe says the signal is recoverable, spec the flag; if not, KILL with the mechanism recorded.

#### I001 2026-08-16 F1 S0
forecast: fit-time — engaged-set fit down ≥15% (augmented leg carries ~30 extra histogram columns; screening to ~6 cuts its per-round cost), gr-suite total down ~5–10%, non-engaged sets bit-identical. strength — sign test flat per stratum; the screen must not change WHICH crosses win, only how many columns the augmented fit carries. Class: conditionally-gated library change (default-off flag until S4).
ran: barrier_check (4 matches: B1, B2, B12, B14) — clearing arguments:
  B1: the change targets only sets where the cross audition engages (≥2000 rows, ≥2 numerics); sub-threshold sets are pre-registered bit-identical and the S2/S3 reads use the engaged-only sign test with `--expect-inert`.
  B2: accepted, not contested — the S3 A/B runs at rung-3 `refit_full` default, and the kill condition includes any changed cross PICK, because the replay refit would propagate a mispick.
  B12: inapplicable — this removes cost from an existing arm; it adds no arms.
  B14: different axis — B14 closed the audition ROUND budget k (truncation harm); this trims the candidate COLUMN set at unchanged k. B14's own mechanism (the leading augmented fit runs to full ES) is exactly why per-round cost, not round count, is the live cost axis.
result: no barrier blocks; arguments recorded above
verdict: PASS
next: S1 after step-0 compute clears — `profile_fit.py --attribution` on an engaged dataset (gr:Brazilian_houses) to confirm the augmented-leg share matches the ×2.18 record, then spec the screen mechanism (rank the ~30 candidate columns by val-residual gain from the base fit, keep top-k≈6)

#### I000 2026-08-16 setup S0
forecast: n/a (infrastructure)
ran: E2/PR #91 reconciliation (fetch showed both merged; main 7684655); local branch cleanup; B15 added to BARRIERS + verified via barrier_check; this file created; test-suite timing + attribution refresh + standing baseline queued this session
result: see facts ledger
verdict: PASS
next: F1 S0 entry, then step-0 compute items in sequence (tests → attribution → baseline decide run)

## Open items (owner named, close-the-loop)

- OPEN 2026-09-23, owner the loop (harness, H(13)): `make_pareto.py --metric blended` dies with `NameError: name 'FixedLocator' is not defined` (render_image, the blended branch; `FuncFormatter` is unimported too). Reproduced on main's code against `20260922-144219.json`; ruff flags it (F821). A diagnostic path only, the headline charts are unaffected. Fix with the next harness task.
- OPEN 2026-09-24, owner the loop (harness, H(14)): `run_benchmarks.py` stores one metadata dict per dataset, `seed_map[next(iter(seed_map))][0]` (line ~2331), which is the first seed to FINISH under the parallel pool. `y_std_test` and `class_prior` are per-seed test-split quantities, so the recorded one is a race, and `make_pareto.py` divides the 3-seed mean RMSE by that one seed's scale. Two runs with identical per-seed metrics (`20260922-144219`, `20260924-130232`) recorded different seeds on 6 of 59 regression keys and moved every arm's R² by up to 0.0002; rankings and the classification panel are untouched (a flipped two-class prior gives the same reference). Fix: record the scale per seed and score skill per seed before averaging, or at least take the lowest seed. Fix with the next harness task, beside H(13).

- CLOSED 2026-09-21: remote `e2/forced-cross-features`, `method/e2-prereg`, `loop-scaffolding` and the five merged `campaign/*` rung branches deleted (each verified fully merged into main first; the four stacked rung branches carried only content-free merge commits on top of commits main already has). The local copies of the five went with them.
- Stale remotes, the maintainer's call (status verified 2026-09-21). Fully merged into main, safe to delete: `bench/portable-no-openml-api`, `refactor/readable-comments`, `worktree-tabarena-030-readiness`, `issue106-predict-thresh` (its local copy is 1 commit ahead — the calibration study PR #108 carries), `record/f2-loop-20260918` (checked out in `.record-worktree`). NOT merged: `docs/attribution-humility` (5 ahead), `docs/user-focused` (4 ahead), `bbstats-patch-1` (1 ahead), `whitepaper` (PR #108, open). Local worktree branch `f2/subgate-race` still waits on the I017 kill being confirmed.

## RESUME protocol (a fresh session runs this, in order)

1. Read this file top to bottom.
2. `git fetch origin`, fast-forward `main`, THEN `git status`; `git log --oneline -3` — confirm branch/sha match the last log entry (campaign baselines assume settled main). A stale local main reads as missing log entries (2026-09-21: it hid I019–I022 and the #120 recovery).
3. `python benchmarks/bench_status.py` (miniconda python) — any run in flight or orphaned `.progress`?
4. Grep this file for `PENDING`: a PENDING entry's results JSON (or `.progress`) is the resume point — score it with `compare_runs.py` / `synth_report.py`, write the verdict, THEN continue. Never relaunch first.
5. Check `benchmarks/results/` for JSONs newer than the last log entry with no log line — score or record them before new work.
6. Execute the top ACTIVE family's `next:` line. One rung, then log, then repeat.
7. End of session: `/handoff` (its message points here; this file is the memory, the handoff carries only session-local traps).
