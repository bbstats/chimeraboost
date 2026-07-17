"""Selection agreement across bag members (BAGGING_PLAN.md Phase 0 / B1 risk).

Decodes the per-member linear_leaves / cross_features selection outcomes from
the flat booster-fit records in results/bagging-phase0.json (written by
profile_fit.py --bag-attribution) and reports how often members 2..K would
have picked a different variant than member 1 — the strength risk B1's
pinned selection takes on, measured at zero extra compute.

Decode rules mirror sklearn_api's selection_rounds flow exactly:
  regression: [const audition, linear audition][, cross race][, refit]
  binary:     [base audition (label 'linear')][, cross race][, refit]
  multiclass: [multiclass]  (no selection)
ll winner = lower best val loss (tie -> const); cross wins iff its best val
within the first selection_rounds beats the audition winner's best; a refit
follows only when cross lost AND the audition was capped (rounds >= budget).

Run: python benchmarks/bagging_b1_agreement.py [--json results/bagging-phase0.json]
"""
import argparse
import collections
import json
import os

SELECTION_ROUNDS = 100  # the shipped default the phase-0 run used


def decode_members(task, fits):
    """Split one estimator fit's booster-fit list into members (the recorder
    marks each member's start with a ``__member__`` record) and infer each
    member's (ll_selected, cf_selected). Returns a list of dicts."""
    groups, cur = [], None
    for f in fits:
        if f["label"] == "__member__":
            cur = []
            groups.append(cur)
            continue
        if cur is None:  # no leading marker (single fit): one implicit member
            cur = []
            groups.append(cur)
        cur.append(f)

    members = []
    for g in groups:
        if g[0]["label"] == "multiclass":
            members.append({"ll": None, "cf": None})
            continue
        if task == "regression":
            const, lin = g[0], g[1]
            if const["label"] != "const" or lin["label"] != "linear":
                raise ValueError("unexpected member labels: "
                                 f"{[f['label'] for f in g]}")
            ll = min(lin["valid_history"]) < min(const["valid_history"])
            aud = lin if ll else const
        else:  # binary: one capped base audition (linear_leaves auto-on)
            aud = g[0]
            ll = None
        cross = next((f for f in g if f["label"] == "cross"), None)
        cf = None
        if cross is not None:
            ch = cross["valid_history"][:SELECTION_ROUNDS]
            cf = bool(ch) and min(ch) < min(aud["valid_history"])
        members.append({"ll": ll, "cf": cf})
    return members


def fmt_votes(votes):
    return "".join("-" if v is None else "YN"[not v] for v in votes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "bagging-phase0.json"))
    args = ap.parse_args()
    with open(args.json) as f:
        records = json.load(f)

    by_ds = collections.defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)

    lines = ["## Selection agreement across members (B1 risk bound)", "",
             "| dataset | task | ll votes (per seed) | cf votes (per seed) "
             "| disagree vs m1 |", "|---|---|---|---|--:|"]
    tot_votes = tot_flips = 0
    for ds, recs in sorted(by_ds.items()):
        ll_cols, cf_cols, n_votes, n_flips = [], [], 0, 0
        for r in recs:
            members = decode_members(r["task"], r["bag_fits"])
            if len(members) != r["K"]:
                raise ValueError(f"{ds} seed {r['seed']}: decoded "
                                 f"{len(members)} members, expected {r['K']}")
            lls = [m["ll"] for m in members]
            cfs = [m["cf"] for m in members]
            ll_cols.append(fmt_votes(lls))
            cf_cols.append(fmt_votes(cfs))
            for votes in (lls, cfs):
                if votes[0] is None:
                    continue
                n_votes += len(votes) - 1
                n_flips += sum(1 for v in votes[1:] if v != votes[0])
        tot_votes += n_votes
        tot_flips += n_flips
        dis = f"{n_flips}/{n_votes}" if n_votes else "-"
        lines.append(f"| {ds} | {recs[0]['task']} | {' '.join(ll_cols)} "
                     f"| {' '.join(cf_cols)} | {dis} |")
    pct = 100.0 * tot_flips / tot_votes if tot_votes else 0.0
    lines += ["",
              f"Members 2..K disagree with member 1 on {tot_flips}/{tot_votes} "
              f"selection decisions ({pct:.0f}%) — the fraction of members B1 "
              "pins to a variant they would not have picked themselves.", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
