"""Study: how correlated are round-i trees across bagged members, vs i?

Fits an 8-member bag, then on a shared holdout computes, per boosting round:
mean pairwise Pearson correlation of the members' tree-contribution vectors,
the mean contribution magnitude, and how many distinct tree structures the
members grew (ordered (feat, thr-bin) tuples — conservative: oblivious levels
are order-invariant and thresholds are member-specific bin indices).

RESULTS (2026-07-16, feeds BAGGING_PLAN.md; full tables in that plan's
authoring session): round-0 trees are 0.93-0.98 correlated across members,
>0.9 through ~round 15-20, ~0 by round 50-100; decay rate tracks signal
strength (cpu_act slow, abalone fast). ZERO exact structural collisions at
any round -> trees are functionally near-identical early but never
bit-identical (no free dedup; compression must be functional, e.g. ISLE
reweighting). Member early-stop round counts varied 133-816 on cpu_act.

Run: python benchmarks/tree_correlation_study.py   (writes tree_corr_results.txt)
"""
import numpy as np
import pandas as pd

import chimeraboost
from chimeraboost import ChimeraBoostRegressor
from chimeraboost.preprocessing import as_model_array

OUT = open("tree_corr_results.txt", "w")


def emit(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


emit("chimeraboost:", chimeraboost.__file__)

K = 8
CACHE = r"benchmarks\data_cache\grinsztajn"


def run(name):
    df = pd.read_csv(rf"{CACHE}\{name}.csv")
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.75 * len(y))
    tr, ho = idx[:n_tr], idx[n_tr:]

    est = ChimeraBoostRegressor(n_ensembles=K, random_state=0)
    est.fit(X[tr], y[tr])

    contribs, structs = [], []
    for m in est.estimators_:
        b = m.model_
        Xa = as_model_array(X[ho], bool(b.prep_.cat_features_))
        Xb = np.ascontiguousarray(b.prep_.transform(Xa).T)
        contribs.append([t.predict(Xb) for t in b.trees_])
        structs.append([(tuple(t.splits_feat), tuple(t.splits_thr))
                        for t in b.trees_])

    lens = [len(c) for c in contribs]
    emit(f"\n=== {name}: n={len(y)}, holdout={len(ho)}, members={K}, "
         f"trees/member min={min(lens)} max={max(lens)}")

    rows = []
    for i in range(max(lens)):
        alive = [k for k in range(K) if lens[k] > i]
        if len(alive) < 2:
            break
        vecs = [contribs[k][i] for k in alive]
        cors = []
        for a in range(len(vecs)):
            for b2 in range(a + 1, len(vecs)):
                if vecs[a].std() == 0 or vecs[b2].std() == 0:
                    continue
                cors.append(np.corrcoef(vecs[a], vecs[b2])[0, 1])
        mc = float(np.mean(cors)) if cors else float("nan")
        mag = float(np.mean([v.std() for v in vecs]))
        st = [structs[k][i] for k in alive]
        rows.append((i, len(alive), mc, mag,
                     len(set(st)), len(set(s[0] for s in st))))

    show = set(range(10)) | {15, 20, 30, 50, 75, 100, 150, 200, 300, 400,
                             500, 750, 1000}
    emit(f"{'round':>5} {'alive':>5} {'meancorr':>9} {'|contrib|':>10} "
         f"{'uniq_struct':>11} {'uniq_featseq':>12}")
    for r in rows:
        if r[0] in show or r[0] == rows[-1][0]:
            emit(f"{r[0]:>5} {r[1]:>5} {r[2]:>9.3f} {r[3]:>10.4f} "
                 f"{r[4]:>11} {r[5]:>12}")

    n = len(rows)

    def band(lo, hi):
        cs = [r[2] for r in rows[lo:hi] if not np.isnan(r[2])]
        return float(np.mean(cs)) if cs else float("nan")

    emit(f"bands: rounds 0-4 corr={band(0, 5):.3f} | "
         f"middle third={band(n // 3, 2 * n // 3):.3f} | "
         f"last 10 alive={band(max(0, n - 10), n):.3f}")


for name in ["reg_num__cpu_act", "reg_num__abalone"]:
    run(name)

OUT.close()
