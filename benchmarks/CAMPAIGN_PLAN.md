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
fact: 2026-08-16 | F4 C1 SHIPPED (`_softmax_kernel`, guarded K ≤ 7): bit-identical 89/89, multiclass fit −44.2% okcupid-stem / −37.2% Traffic_violations / −40.5% cjs, binary control −0.6% on zero calls (`results/campaign-f4c1-speed-20260816.txt`); softmax leg 1.70s → 0.05s
fact: 2026-08-16 | the fused softmax kernel is 6.9–11.0× faster PARALLEL than serial at every real shape — `prange` earns its thread setup even on 3-wide rows
fact: 2026-08-16 | all 8 multiclass rows in `--decide` are K ∈ {3,5,6} (okcupid-stem, Traffic_violations, cjs, eucalyptus + variants) — every one under the K ≤ 7 guard; **Grinsztajn contains no multiclass task at all**, so multiclass work cannot move the headline stratum
fact: 2026-08-16 | F4 C1 ceiling, hc:okcupid-stem (`results/campaign-f4c1-walltime-20260816.txt`): `losses._softmax` = 1.611s of a 3.55s fit = **45.4% by WALL CLOCK** (cProfile said 39% — understated); `grad_hess` 40.0% of fit, `eval` 5.5%; K=3, shapes (32377,3)/(5714,3)/(38091,3)
fact: 2026-08-16 | inside one `_softmax` call at (32377,3): the `max` reduce is 43% and `sum` 15% — the two length-3 inner-axis reductions are 58% of the call, `exp` only 20%. The cost is numpy's per-row reduce machinery, not the arithmetic.
fact: 2026-08-16 | softmax candidates: in-place `out=` ufuncs 1.01× (no effect, temporaries were never the cost); numpy column-fold 0.48×; numba fused with reciprocal-multiply 0.03× but drifts at every K; **numba fused with true divide 0.03× (32×) and bit-identical at K ≤ 7**
fact: 2026-08-16 | the bit-identity boundary for any hand-folded sum over the class axis is **K ≤ 7**, measured over K=2..50 — numpy's pairwise summation starts blocking at K=8. `max` never drifts at any K.
fact: 2026-08-16 | attribution on 7684655 (`results/campaign-attr-20260816.{json,md}`): cross-audition leg = 40–58% of fit on engaged sets (nyc-taxi 52%, road-safety 58%, diamonds 46%); ll selected 12/12, cross 24/27; race truncation at k=100 keeps 24/24 cross picks; okcupid-stem multiclass "other" (non-kernel) = 50.5% of fit; hc prep_other+ts_enc ~12–28%

## Beam

| id | family | status | next |
|----|--------|--------|------|
| F1 | Cross-feature cost trim v2 | KILLED 2026-08-16 (S2, I007+I008) | none — closed as barrier B16 |
| F4 | Profiling-driven speed | ACTIVE (C2 + C1 shipped) | C1 SHIPPED (I012/I013): fused softmax kernel, bit-identical, multiclass fit −37 to −44%. Next = C1b (`grad_hess` fusion), S0 owes a fresh wall-clock read now that softmax is 0.05s |
| F2 | Sub-gate cross via CV-averaged race | ACTIVE | S1 probe script (S0 re-scoped it, I002) |
| F3 | Classifier forced-cross | ACTIVE | S1 probe of classifier pair fidelity (S0 done, I004); behind F4/F2 |
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

### F2 — Sub-gate cross eligibility via a CV-averaged race (re-scoped at S0, was: lower the row gate)
status: ACTIVE
hypothesis: sets below `CROSS_MIN_SAMPLES=2000` (`sklearn_api.py:1283`) — eucalyptus first — can earn cross features IF the race signal is repaired by CV-averaging (cheap at that size); a plain threshold drop is barred by B1's mechanism (untrustworthy small val split) + B2 (refit amplifies mispicks)
parent-evidence: M1 record 2026-07-17 (eucalyptus = biggest hc CatBoost gap, below gate; recorded follow-up); I002
barriers: B1 (cleared only via the signal-quality mechanism), B2 (judge at rung 3), B14 (inapplicable — signal quality, not budget)
kill: the S1 probe shows the CV-averaged race still mispicks on sub-gate sets (test metric not improved by its picks)
next: S1 = zero-library-change probe (see I002)

### F3 — Classifier forced-cross ("always")
status: ACTIVE
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
Next unit = **C1b**, fusing `grad_hess`'s remaining `P - Y` and
`max(P*(1-P), 1e-6)` passes into the kernel that now exists. Its S0 owes a fresh
wall-clock read, because C1 changed the very number C1b would be justified by —
the honest prior is that the remaining share is now small, and the unit should
die cheaply at S0 if it is.

### F5 — hc-Brier gap vs CatBoost
status: BLOCKED(needs B3-clearing mechanism from lens L3)
hypothesis: (held slot) the real-but-small hc Brier gap (+0.0029/set, CatBoost 86–88% winrate there) has a lever that isn't a partial CatBoost port
parent-evidence: hc suite build record 2026-07-15; B3 = seven partial ports, seven kills
barriers: B3 hard; B4 (ordered boosting closed)
kill: any proposal that is a partial CatBoost mechanism port dies at S0
next: none until a beam refill produces a genuinely integrated mechanism

## Iteration log (append-only)

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

- Remote branch deletion blocked by tool permissions this session: `git push origin --delete e2/forced-cross-features method/e2-prereg` — Nathan or a session with push-delete permission. Also stale remotes worth a look: `bench/portable-no-openml-api`, `docs/attribution-humility`, `docs/user-focused` (local copy unmerged), `refactor/readable-comments`, `worktree-tabarena-030-readiness`.

## RESUME protocol (a fresh session runs this, in order)

1. Read this file top to bottom.
2. `git status`; `git log --oneline -3` — confirm branch/sha match the last log entry (campaign baselines assume settled main).
3. `python benchmarks/bench_status.py` (miniconda python) — any run in flight or orphaned `.progress`?
4. Grep this file for `PENDING`: a PENDING entry's results JSON (or `.progress`) is the resume point — score it with `compare_runs.py` / `synth_report.py`, write the verdict, THEN continue. Never relaunch first.
5. Check `benchmarks/results/` for JSONs newer than the last log entry with no log line — score or record them before new work.
6. Execute the top ACTIVE family's `next:` line. One rung, then log, then repeat.
7. End of session: `/handoff` (its message points here; this file is the memory, the handoff carries only session-local traps).
