---
name: handoff
description: Write the next-session context message — current state, queued work, paid-for evidence, traps — as the entire reply, nothing on top
---

Write a handoff message for the next agent's context. **The reply IS the
message**: no preamble ("Here's the handoff…"), no closing offer, no meta
commentary before or after. Nathan pastes it verbatim into a fresh session.

Before writing, run `git status` and `git log --oneline -3` so the state
section reports what IS, not what you remember. If cleanup is owed (merged
branch undeleted, scratch files, an unrecorded verdict), do it first — a
handoff must never hand off a loose end the close-the-loop rule says this
session owns.

## Audience and stance

The reader is an agent with zero conversation context and full repo access.
Every project shorthand unpacked on first use; every claim carries a file
pointer (`SELECT_PLAN.md` E1, `results/xyz.json`) instead of a re-derivation;
relative dates converted to absolute. Plain-English sentences, not stat
fragments — the CLAUDE.md chat rules apply to handoffs too.

Facts over narration. For each thing you record, the test is: what is true
now, what was spent to learn it, and what decision does it feed. Session
war stories only survive if they changed a rule or closed a door.

## Structure (in this order)

1. **Opening line.** `Context: continuing <thread> in ChimeraBoost.` plus the
   one-sentence goal the whole message serves.
2. **State of the tree.** Branch, committed vs uncommitted, tests
   (pass/skip counts), what merged and where, cleanup already done. Include
   drift warnings — e.g. "main absorbed refactors of sklearn_api.py; every
   file:line in the plan files may be stale, verify from source."
3. **The queued work.** ONE next action, precisely specified: the exact
   command or change, what it is NOT (e.g. "the first step is NOT a decide
   run"), the registered kill bar with its threshold, and which plan file
   owns the pre-registration. If a forecast is already registered, say so —
   the next agent must not re-forecast.
4. **Evidence already paid for.** Findings the next agent must not re-derive
   or re-litigate, each with its load-bearing caveat ("argued from source,
   don't re-derive"; "one race carries this — a pointer, not a gate").
   Distinguish pointer-grade from gate-grade evidence explicitly.
5. **Instrument rules and traps.** Anything that will silently corrupt the
   next session's measurements: self-checks that must not move and what it
   means if they do, circularity traps, flags that are mandatory for
   validity (`--uncapped-auditions`, `--expect-inert`), harness gotchas
   found this session.
6. **Leverage and dead ends.** Open directions ranked by expected frontier
   push, each with one line of why. Then the closed doors relevant to those
   directions, pointing at `BARRIERS.md` / `barrier_check.py` rather than
   restating every kill.
7. **Standing footer**, verbatim rules that bit before: one benchmark at a
   time, never two. Run script files, not `python -c`. PowerShell 5.1 has no
   `&&`. TabArena stays sealed in every form. Print the aggregate table
   after every benchmark run, unprompted, and lead every report to Nathan
   with the plain-English takeaway before the numbers. Add any rule THIS
   session learned the hard way.

## Hard constraints

- Nothing that would let TabArena results influence a source change.
- No session URLs.
- Length: whatever density requires, typically 400–700 words. Cut padding,
  never precision — a number the next agent will gate on keeps its exact
  value, its n, and its source file.
- Omit a section only if it is genuinely empty; never invent content to
  fill one.
