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
fact: 2026-08-16 | attribution on 7684655 (`results/campaign-attr-20260816.{json,md}`): cross-audition leg = 40–58% of fit on engaged sets (nyc-taxi 52%, road-safety 58%, diamonds 46%); ll selected 12/12, cross 24/27; race truncation at k=100 keeps 24/24 cross picks; okcupid-stem multiclass "other" (non-kernel) = 50.5% of fit; hc prep_other+ts_enc ~12–28%

## Beam

| id | family | status | next |
|----|--------|--------|------|
| F1 | Cross-feature cost trim v2 | ACTIVE | implement `cross_top_columns` (default-off) + S2 synth A/B (spec in I005) |
| F2 | Sub-gate cross via CV-averaged race | ACTIVE | S1 probe script (S0 re-scoped it, I002) |
| F3 | Classifier forced-cross | ACTIVE | S1 probe of classifier pair fidelity (S0 done, I004); behind F1/F2 |
| F4 | Profiling-driven speed | ACTIVE | S1 cProfile of okcupid-stem's 50.5% "other" |
| F5 | hc-Brier gap vs CatBoost | BLOCKED(needs B3-clearing mechanism from lens L3) | none until refill |

### F1 — Cross-feature cost trim v2
status: ACTIVE
hypothesis: the cross-feature audition's ×2.18 engaged-set fit cost can drop by double digits without changing which crosses win, by screening candidate pairs to top-k≈6 columns before the race (fallbacks: prefix-importance single-fit; mid-boost augmentation)
parent-evidence: 2026-07-13 ship record (cost ×2.18 total, ×2.38 median on engaged sets; v2 ideas recorded at ship time, previously nowhere greppable — re-recorded here); `PARETO_PLAN.md` step-0 attribution
barriers: B14, B2 — clearing argument owed at S0: B14 closed the audition ROUND budget k; a column screen trims the candidate PAIR SET, a different axis; per B2 the S3 A/B runs at rung-3 `refit_full` default
next: implement `cross_top_columns` (default-off) per the I005 spec, then S2 synth A/B
kill: any synth slice where the screen changes a cross PICK (not just cost) with strength loss; or S3 regression stratum sign-test fail at default quality

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
kill: per-candidate, set at S0; family-level: if a cProfile read shows the "other" time is irreducible dispatch (many tiny trees), record and kill
next: S1 = `profile_fit.py --dataset okcupid-stem --top 30` (single-set cProfile, ~1 min) to name the "other" 50.5%

### F5 — hc-Brier gap vs CatBoost
status: BLOCKED(needs B3-clearing mechanism from lens L3)
hypothesis: (held slot) the real-but-small hc Brier gap (+0.0029/set, CatBoost 86–88% winrate there) has a lever that isn't a partial CatBoost port
parent-evidence: hc suite build record 2026-07-15; B3 = seven partial ports, seven kills
barriers: B3 hard; B4 (ordered boosting closed)
kill: any proposal that is a partial CatBoost mechanism port dies at S0
next: none until a beam refill produces a genuinely integrated mechanism

## Iteration log (append-only)

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
