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
- Int-exact per-histogram ≠ bit-identical end-to-end (op order feeds gains/tie-breaks). Any "exact" kernel change runs `identity_snapshot.py` first; identical → pure-speed ladder; not → FP-drift class → S2 with the Brier read.
- Standing numerical-drift policy (the maintainer, 2026-09-18): an algorithm-changing revision may carry last-bit drift provided it is monitored (identity-snapshot diff quantified, not silently absorbed), reported (both axes in the verdict), and gate-evidenced to improve over time. Pure rewrites that drift without changing the algorithm stay on the ladder above.
- Delivery is PRs only (the maintainer, 2026-09-18): no merges to `main`, no PyPI releases from these sessions until he says otherwise. Ships land as pull requests; merging and releasing stay his call.

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
fact: 2026-09-21 | harness rule: `--models` must include `ChimeraBoost` (the baseline) or run_benchmarks exits 2 at argparse; a stray `<stamp>.txt` tee is left in results/ when that happens
fact: 2026-09-21 | main settled at 486e563. PRs #116–#119 (F3 S1/S2/S3 + the refill shortlist) were each merged into the rung branch below them rather than into main — stacked PR bases, and the base branches were not deleted on merge, so GitHub never retargeted them; main received only #115. Recovered the same morning as PR #120 (the four commits, unchanged). The PR #56 trap a second time. Rule since: every rung branch is cut from main, PR base is always main, one campaign PR open at a time (now step 0 of the `campaign-step` skill). A stale local main hid the recovery from the next session until it fetched — fetch before reading the log
fact: 2026-09-21 | test suite on main 486e563 (after the #120 recovery; rung branch adds harness + docs only) = 1084 passed, 1 skipped, 114 s — the F3 S2 count, so the recovery landed the library intact
fact: 2026-09-21 | F4 fresh profile (I023, `results/campaign-f4s1-other-20260921.{md,json}`; exclusive wall clock, default estimator, the August panel): on the 6 gr sets `grow` is 63.5–71.8% of fit and the full-data refit leg is 25–35% — replayed rounds 10.7–19.3% plus the 1.25× extra rounds grown from scratch 9.6–12.3%. August's unnamed 23–30% "other" was mostly those replayed rounds, which that instrument never wrapped
fact: 2026-09-21 | binary Logloss layer = 8.3–10.0% of a gr binary fit: `grad_hess` 4.9/4.2/4.4% + per-round validation `eval` 5.1/4.3/4.0% (MagicTelescope/Higgs/road-safety; a 7-rep rerun replicates within 0.1 point). Regression: `grad_hess` 0.6–0.7%, `val_score` 0.8–2.1%; no non-kernel row reaches 5% on any gr regression set
fact: 2026-09-21 | hc `prep` = 34.8% (kick) / 31.8% (wine-reviews) / 41.1% (okcupid-stem) of the default fit AFTER C2, of which 15.5/13.7/19.9 points are the refit's full-row `fit_transform` — the largest non-kernel object measured, hc-only, inside unsplit
fact: 2026-09-21 | a replayed round costs ~40–45% of a grown round on linear-leaf regression sets (nyc-taxi 0.97 vs 2.45 ms, diamonds 1.09 vs 2.4 ms); `_assign_leaves` / `_leaf_values` / `_linear_leaf_fit` unsplit, ridge accumulator B10-closed
fact: 2026-09-21 | exclusive-time hook instrument (`f4_other_walltime.py`): 0.85 µs per hook call, 0.1–1.2% of fit, wrapped/plain 0.99–1.03; on the first pass MagicTelescope's PLAIN arm read a 3-rep median 22% slow (1.067 s; all seven rerun fits 0.87–0.91 s), so 3 reps is thin for sub-second fits — use 7 and keep the per-rep times
fact: 2026-09-21 | first Muse Code rung: one pass, exit 0, ~15 min wall clock for a two-file library+tests edit; muse cannot edit CRLF files with its own tool and patches by script instead; its sandbox user cannot write `.pytest_cache` and leaves an undeletable `pytest-of-Nathan/` in the repo root (untracked, harmless)

## Beam

| id | family | status | next |
|----|--------|--------|------|
| F1 | Cross-feature cost trim v2 | KILLED 2026-08-16 (S2, I007+I008) | none — closed as barrier B16 |
| F4 | Profiling-driven speed | ACTIVE (C2 + C1 + C1b shipped; C3 + C4 measured) | Fresh profile read (I023). Two objects owe an S0, in ceiling order: **C4** hc `prep` (32–41% of hc fit, inside unsplit) then **C3** the binary Logloss layer (`grad_hess` + per-round validation `eval`, 8–10% of a gr binary fit). An ordering call against the I022 entrants; absent a pick the loop takes C4's S0 |
| F2 | Sub-gate cross via CV-averaged race | KILLED (I017) | 5/5 engaged precision at 3-7x cost; S1 did not replicate |
| F3 | Classifier forced-cross | KILLED 2026-09-21 (S3, I021) | gr binary engaged 10W-13L, median −0.04%: the race earns its fee on the classifier. Knob stays opt-in (PR #117), no rung-1 pin |
| F5 | hc-Brier gap vs CatBoost | BLOCKED(needs B3-clearing mechanism from lens L3) | none until refill |

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

### F5 — hc-Brier gap vs CatBoost
status: BLOCKED(needs B3-clearing mechanism from lens L3)
hypothesis: (held slot) the real-but-small hc Brier gap (+0.0029/set, CatBoost 86–88% winrate there) has a lever that isn't a partial CatBoost port
parent-evidence: hc suite build record 2026-07-15; B3 = seven partial ports, seven kills
barriers: B3 hard; B4 (ordered boosting closed)
kill: any proposal that is a partial CatBoost mechanism port dies at S0
next: none until a beam refill produces a genuinely integrated mechanism

## Refill shortlist 2026-09-21 (awaiting the maintainer's pick; beam cap 5)

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
| H | **Harness instruments** (L4), muse rungs, no ship gate | the loop's own decisions | (1) engaged-slice median + bootstrap CI + per-seed agreement in `compare_runs` (3 h; I020/I021 gated on a number no tool prints); (2) probe→decide transfer footer: panel coverage + shipped-mode vs oracle read (5 h; the I019→I021 miss); (3) cost column in `compare_runs`, refused when not decision-grade (2 h); (4) POINTER label for strata under 8 decided sets (1 h); (5) `--save`/`--models` argparse guards (1 h) | none | each validated by re-reading `results/20260921-080246.json` to reproduce I021's hand-written verdict | prevents the wrong decision rather than moving the chart; (1), (4), (5) recommended regardless of the beam pick |

Not proposed (checked): AGBM momentum and gradient-mass bin borders (L2, low priors, B6/C4 adjacent); Counter/frequency column on its own (B3 hard; goes through R3 first); `cpu_act@sus25` and `eucalyptus@time` outliers (one win each, seed-unstable); random_strength / Bayesian bootstrap / DART-class noise (killed by the SMALLDATA ablation, ≈1 point).

Recommended pick: **R1, R2, R3, R4 + H(1)(4)(5)**. R1 and R2 have free probes and can both resolve in one session; R3 is F5's only sanctioned door and runs while nothing else is on the bench; R4 is the first hc mechanism that is not a port. Process proposal riding with this: amend `AGENTS.md` so muse may edit any file the task file lists (today `benchmarks/` is reserved), which is what makes H and the probe scripts muse rungs instead of Claude's.

## Iteration log (append-only)

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
of each ceiling converts, 3–5% of binary fit. Both are an ordering call
against the I022 entrants; absent a pick the loop takes C4's S0 next.

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
