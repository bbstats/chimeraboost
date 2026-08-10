"""Replay const-vs-linear audition rules against the recorded step-0 curves.

Zero benchmark cost. `results/pareto-step0.json` holds the FULL validation
history of both the constant-leaf and linear-leaf regression fits (they were run
to their own early stop for the attribution table), so any capped decision rule
can be simulated on them and scored against what the full curves say is actually
better.

What this answers: the shipped rule mispicks 4 of 12 regression selections at
k=100 (PARETO_PLAN.md step-0 pre-registration), and PARETO_PLAN's "Known
residual" reports that no MARGIN rule at k=100 separates the two arms because
their overlap is total. Those two facts leave open whether a differently-SHAPED
rule -- reading the curve's late trend instead of its best-so-far -- does better
at the same cost. That matters more than it used to: the rung-3 refit replays the
audition winner over all rows, so a mispick propagates (BARRIERS.md B2).

Self-check: the shipped rule at k=100 must reproduce 4/12. If it does not, the
replay does not match the estimator and nothing below is worth reading.

Usage:
  python benchmarks/probe_audition_rule.py
  python benchmarks/probe_audition_rule.py --k 100 200 300 500
  python benchmarks/probe_audition_rule.py --corpus results/audition-corpus-e1.json
"""
import argparse
import json
import os
import sys

import numpy as np

_BENCH = os.path.dirname(os.path.abspath(__file__))
STEP0 = os.path.join(_BENCH, "results", "pareto-step0.json")


def load_races(path):
    """[(dataset, seed, const_curve, linear_curve)] for every recorded race."""
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    races = []
    for r in records:
        # First fit per label. The audition fits come first in fit order; a
        # rung-3 replay refit appends a fourth record that REUSES the winner's
        # label with an empty valid_history (it has no validation loop), and
        # last-wins would let it clobber the real audition curve.
        fits = {}
        for f in r["fits"]:
            fits.setdefault(f["label"], f)
        if "const" not in fits or "linear" not in fits:
            continue          # binary/multiclass: no const-vs-linear race
        c = np.asarray(fits["const"]["valid_history"], dtype=float)
        l = np.asarray(fits["linear"]["valid_history"], dtype=float)
        if c.size == 0 or l.size == 0:
            continue
        races.append((r["dataset"], r["seed"], c, l))
    return races


# ---------------------------------------------------------------- rules
# Each rule takes the two capped curves and returns True for "pick linear".
# The shipped rule's tie-break goes to constant leaves, and every rule here
# keeps that convention so the comparison is like for like.

def rule_best(c, l):
    """SHIPPED: best validation loss seen within the cap."""
    return l.min() < c.min()


def rule_last(c, l):
    """Value at the cap, not best-so-far -- rewards a curve still descending."""
    return l[-1] < c[-1]


def rule_tail_mean(c, l, w=20):
    """Mean over the last w rounds: best-so-far with the noise averaged out."""
    return l[-w:].mean() < c[-w:].mean()


def rule_extrapolate(c, l, w=20):
    """Best-so-far, projected forward one window by each curve's recent slope.

    The failure this targets is the recorded one: the races that go wrong are
    the ones that CROSS LATE, so the arm that is behind at the cap but still
    falling faster is the one the cap is stealing from.
    """
    def proj(v):
        if v.size <= w:
            return v.min()
        slope = (v[-1] - v[-w]) / w          # negative while improving
        return min(v.min(), v[-1] + slope * w)
    return proj(l) < proj(c)


RULES = [
    ("best (shipped)", rule_best),
    ("last value", rule_last),
    ("tail mean w=20", rule_tail_mean),
    ("extrapolated w=20", rule_extrapolate),
]


def truth(c, l):
    """What the FULL curves say: is linear genuinely better? Ties to const."""
    return l.min() < c.min()


def regret(c, l, picked_linear):
    """Relative validation loss given up by the pick, vs the better full arm."""
    best = min(c.min(), l.min())
    got = l.min() if picked_linear else c.min()
    return (got - best) / abs(best) if abs(best) > 1e-12 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, nargs="+", default=[100, 200, 300, 500],
                    help="audition caps to simulate")
    ap.add_argument("--corpus", nargs="*", default=[],
                    help="extra --attribution result JSONs to widen the race "
                         "corpus; step-0 always loads and its self-check "
                         "always runs")
    args = ap.parse_args(argv)

    if not os.path.exists(STEP0):
        print(f"missing {STEP0} -- regenerate with "
              f"`python benchmarks/profile_fit.py --attribution --seeds 3 "
              f"--out pareto-step0`")
        return 1

    # Self-check FIRST, on the step-0 races alone, whatever else is loaded:
    # the shipped rule at k=100 must reproduce the 4/12 mispicks the estimator
    # actually recorded. If it moves, the instrument is wrong, not the finding.
    step0 = load_races(STEP0)
    miss = sum(1 for _, _, c, l in step0
               if rule_best(c[:100], l[:100]) != truth(c, l))
    print(f"self-check: shipped rule at k=100 on step-0 races -- "
          f"{miss}/{len(step0)} mispicks (must be 4/12)")
    if (miss, len(step0)) != (4, 12):
        print("SELF-CHECK FAILED: the replay no longer matches the recorded "
              "estimator decisions. Nothing below is worth reading.")
        return 1

    races = list(step0)
    seen = {(d, s) for d, s, *_ in races}
    for path in args.corpus:
        p = path if os.path.exists(path) else os.path.join(_BENCH, path)
        extra = load_races(p)
        dups = [(d, s) for d, s, *_ in extra if (d, s) in seen]
        kept = [(d, s, c, l) for d, s, c, l in extra if (d, s) not in seen]
        seen.update((d, s) for d, s, *_ in kept)
        races += kept
        note = f", {len(dups)} duplicate (dataset, seed) skipped" if dups else ""
        print(f"corpus {path}: +{len(kept)} races{note}")
    print()

    print(f"{len(races)} recorded const-vs-linear races "
          f"({len(set(d for d, *_ in races))} datasets x seeds)\n")
    if not races:
        print("no races in the file -- nothing to replay")
        return 1

    truths = [truth(c, l) for _, _, c, l in races]
    print(f"ground truth (full early stop): linear genuinely better on "
          f"{sum(truths)} of {len(truths)}\n")

    print(f"{'rule':22s} {'k':>5s} {'mispicks':>9s} {'median regret':>14s} "
          f"{'worst regret':>13s}   worst dataset")
    for k in args.k:
        for name, rule in RULES:
            wrong, regrets, worst = 0, [], ("", 0.0)
            for (ds, seed, c, l), want in zip(races, truths):
                ck, lk = c[:k], l[:k]
                if ck.size == 0 or lk.size == 0:
                    continue
                got = rule(ck, lk)
                r = regret(c, l, got)
                regrets.append(r)
                if got != want:
                    wrong += 1
                if r > worst[1]:
                    worst = (f"{ds} s{seed}", r)
            med = float(np.median(regrets)) if regrets else 0.0
            print(f"{name:22s} {k:5d} {wrong:6d}/{len(regrets):<3d} "
                  f"{med:13.3%} {worst[1]:12.3%}   {worst[0]}")
        print()

    print("Read the REGRET column, not the mispick count: a mispick between two")
    print("arms that are within noise of each other costs nothing, and most of")
    print("them are (PARETO_PLAN 'Known residual': the overlap is total).")
    print(f"n is {len(races)} regression races on "
          f"{len(set(d for d, *_ in races))} datasets -- a pointer, never a gate")
    print("(GATE_ROBUSTNESS.md #2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
