"""F1 S2 read: what the cross-column screen bought and what it cost.

Splits the synth run into the sets where the screen engaged (the arms' scores
differ) and the inert control (identical scores), then reports fit time and the
sign test on each. Judge metric matches the harness summary: RMSE for
regression, Brier for classification. Speed on synth is directional only, never
decision-grade (project_synthgen).

    python benchmarks/f1_s2_read.py benchmarks/results/campaign-f1s2-20260816.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

BASE, NEW = "ChimeraBoost", "ChimeraBoostXTop6"


def main(path):
    blob = json.load(open(path))
    meta = blob["datasets"]

    fit = defaultdict(lambda: defaultdict(list))
    score = defaultdict(lambda: defaultdict(list))
    for r in blob["records"]:
        m = r["model"]
        if m not in (BASE, NEW):
            continue
        ds = r["dataset"]
        if r.get("fit_time") is not None:
            fit[m][ds].append(r["fit_time"])
        met = r["metrics"]
        # Higher = better for both: primary is already negated RMSE.
        v = -met["brier"] if met.get("brier") is not None else met.get("primary")
        if v is not None:
            score[m][ds].append(v)

    engaged, inert = [], []
    for ds in sorted(score[BASE]):
        if ds not in score[NEW]:
            continue
        a, b = np.mean(score[BASE][ds]), np.mean(score[NEW][ds])
        (inert if a == b else engaged).append(ds)

    def total(names, m):
        return sum(np.mean(fit[m][ds]) for ds in names if fit[m][ds])

    print("datasets: %d engaged (arms differ), %d inert control"
          % (len(engaged), len(inert)))
    for label, names in (("engaged", engaged), ("inert", inert),
                         ("all", engaged + inert)):
        ta, tb = total(names, BASE), total(names, NEW)
        print("  %-8s fit  base %7.2fs  screened %7.2fs  change %+6.1f%%"
              % (label, ta, tb, 100.0 * (tb - ta) / ta if ta else 0.0))

    print("\nengaged-set sign test by task (+ = the screen won):")
    for t in ("regression", "binary", "multiclass"):
        names = [d for d in engaged if meta[d]["task"] == t]
        w = sum(np.mean(score[NEW][d]) > np.mean(score[BASE][d]) for d in names)
        print("  %-11s %2dW-%2dL  of %d engaged" % (t, w, len(names) - w,
                                                    len(names)))

    print("\nlargest engaged moves (relative, + = the screen better):")
    deltas = []
    for ds in engaged:
        a, b = np.mean(score[BASE][ds]), np.mean(score[NEW][ds])
        if abs(a) > 1e-12:
            deltas.append((100.0 * (b - a) / abs(a), ds, meta[ds]["task"]))
    deltas.sort()
    for d, ds, t in deltas[:5] + deltas[-3:]:
        print("  %+8.2f%%  %-14s %s" % (d, ds, t))


if __name__ == "__main__":
    main(sys.argv[1])
