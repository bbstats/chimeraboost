"""Factor attribution over synthgen results -- the "prove the feature" instrument.

Modes (all read standard run_benchmarks --save JSONs, syn: datasets only):

  python benchmarks/synth_report.py RUN.json
      Excess view: how far above the known Bayes floor each slice sits.
  python benchmarks/synth_report.py BASE.json NEW.json [--model ChimeraBoost]
      A/B attribution: per-dataset primary deltas sliced by recipe factors,
      sign tests per slice, plus an OLS pass ranking factors by |t|.
      --metric brier restricts to classification sets and judges on Brier
      (sign flipped so + still means "arm better").
      --model-new compares different model names across the runs (e.g.
      baseline ChimeraBoost records vs an arm's ChimeraBoostEns2 records).
  python benchmarks/synth_report.py RUN.json --realism
      Cross-model ordering checks (is the suite shaped like real data?).

Notes: floors are generative lower bounds (feature views may quantize signal),
so excess is comparable across arms but not necessarily attainable-zero. The
harness 25% test split adds O(1/sqrt(n_test)) sampling noise vs the full-data
floor stored in meta.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from synthgen.suites import CANARIES, VERSION

EPS = 1e-9


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_run(path):
    data = json.load(open(path, encoding="utf-8"))
    ds_meta = {k: v for k, v in data["datasets"].items()
               if k.startswith("syn:") and "synth" in v}
    per = defaultdict(lambda: defaultdict(list))  # model -> ds -> [metrics dicts]
    for r in data["records"]:
        if r["dataset"] in ds_meta:
            per[r["model"]][r["dataset"]].append(r["metrics"])
    return ds_meta, per


def metric_means(per_model, model, metric="primary"):
    """Per-dataset mean of `metric`, oriented so higher = better.

    "brier" is lower-better, so it is negated; datasets whose records lack the
    metric (regression sets have no Brier) are dropped, which restricts the
    brier view to classification.
    """
    if model not in per_model:
        raise SystemExit(f"model {model!r} not in run; have {sorted(per_model)}")
    sign = -1.0 if metric == "brier" else 1.0
    out = {}
    for ds, v in per_model[model].items():
        vals = [m[metric] for m in v if m.get(metric) is not None]
        if vals:
            out[ds] = sign * float(np.mean(vals))
    return out


def primary_means(per_model, model):
    return metric_means(per_model, model, "primary")


# ---------------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------------
def _bucket_specs(metas):
    """(label, predicate) slices over the synth meta of each dataset."""
    def synth(ds, field):
        return metas[ds]["synth"].get(field)

    specs = [
        ("all", lambda ds: True),
        ("task=regression", lambda ds: metas[ds]["task"] == "regression"),
        ("task=binary", lambda ds: metas[ds]["task"] == "binary"),
        ("task=multiclass", lambda ds: metas[ds]["task"] == "multiclass"),
        ("saturated", lambda ds: synth(ds, "saturated")),
        # canary status is freeze-time knowledge (suites.CANARIES), valid only
        # for keys of the current generator version
        ("canary&cats", lambda ds: synth(ds, "gen_version") == VERSION
         and synth(ds, "recipe_id") in CANARIES and synth(ds, "n_cat") > 0),
        ("car-analog+", lambda ds: synth(ds, "saturated")
         and synth(ds, "rule_kind") == "cat_cross"
         and not (synth(ds, "gen_version") == VERSION
                  and synth(ds, "recipe_id") in CANARIES)),
        ("cats=none", lambda ds: synth(ds, "n_cat") == 0),
        ("cats=mixed", lambda ds: 0 < synth(ds, "cat_fraction") < 1.0),
        ("cats=all", lambda ds: synth(ds, "cat_fraction") >= 1.0),
        ("card>16", lambda ds: synth(ds, "max_cardinality") > 16),
        ("cats=entity", lambda ds: (synth(ds, "n_cat_entity") or 0) > 0),
        ("depth<=2", lambda ds: synth(ds, "interaction_depth") <= 2),
        ("depth>=3", lambda ds: synth(ds, "interaction_depth") >= 3),
        ("n<2000", lambda ds: synth(ds, "n") < 2000),
        ("n>=2000", lambda ds: synth(ds, "n") >= 2000),
        ("irrelevant>0.3", lambda ds: synth(ds, "irrelevant_fraction") > 0.3),
        ("missing>0", lambda ds: synth(ds, "missing_fraction") > 0),
        ("crossfeat-scope", lambda ds: metas[ds]["task"] in ("regression", "binary")
         and synth(ds, "n") >= 2000 and synth(ds, "cat_fraction") < 0.5
         and synth(ds, "interaction_depth") >= 2),
    ]
    for kind in ("linear", "neural", "tree", "product", "plateau"):
        specs.append((f"func={kind}",
                      lambda ds, k=kind: synth(ds, "func_dominant") == k))
    return specs


def _rel_delta(base, new):
    out = {}
    for ds in set(base) & set(new):
        b = base[ds]
        out[ds] = (new[ds] - b) / max(abs(b), 1e-12)
    return out


def ab_report(metas, base, new):
    deltas = _rel_delta(base, new)
    if not deltas:
        raise SystemExit("no shared syn: datasets between the two runs")
    print(f"{'slice':18s} {'n':>4s} {'W-L-T':>9s} {'mean d':>9s} {'p':>7s}")
    rows = []
    for label, pred in _bucket_specs(metas):
        ds_in = [ds for ds in deltas if pred(ds)]
        if not ds_in:
            continue
        d = np.array([deltas[ds] for ds in ds_in])
        w = int((d > EPS).sum()); l = int((d < -EPS).sum()); t = len(d) - w - l
        p = binomtest(min(w, l), w + l, 0.5).pvalue if w + l else 1.0
        rows.append((label, len(d), w, l, t, d.mean(), p))
        print(f"{label:18s} {len(d):4d} {f'{w}-{l}-{t}':>9s} {d.mean():+9.3%} {p:7.3f}")

    # standardized OLS: which factor carries the effect
    fields = [("interaction_depth", "num"), ("cat_fraction", "num"),
              ("max_cardinality", "num"), ("irrelevant_fraction", "num"),
              ("noise_level", "num"), ("missing_fraction", "num"),
              ("imbalance", "num"), ("n", "log"), ("saturated", "bool"),
              ("entity_strength", "num")]
    ds_list = sorted(deltas)
    cols, names = [np.ones(len(ds_list))], ["intercept"]
    for f, kind in fields:
        v = np.array([float(metas[ds]["synth"].get(f) or 0.0) for ds in ds_list])
        if kind == "log":
            v = np.log(np.maximum(v, 1.0))
        s = v.std()
        if s < 1e-12:
            continue
        cols.append((v - v.mean()) / s)
        names.append(f)
    Xm = np.column_stack(cols)
    yv = np.array([deltas[ds] for ds in ds_list])
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    dof = max(1, len(yv) - Xm.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xm.T @ Xm)
    tvals = beta / np.sqrt(np.maximum(np.diag(cov), 1e-18))
    print("\nfactor OLS on relative delta (|t| ranked):")
    order = np.argsort(-np.abs(tvals))
    for i in order:
        if names[i] == "intercept":
            continue
        print(f"  {names[i]:20s} coef {beta[i]:+9.4f}  t {tvals[i]:+6.2f}")
    return rows


# ---------------------------------------------------------------------------
# excess view
# ---------------------------------------------------------------------------
def excess_report(metas, per_model, model):
    recs = per_model[model]
    print(f"{'slice':18s} {'n':>4s} {'xs Brier':>9s} {'RMSE/sig':>9s}")
    for label, pred in _bucket_specs(metas):
        xs_brier, rmse_ratio = [], []
        for ds, mlist in recs.items():
            if not pred(ds):
                continue
            meta = metas[ds]["synth"]
            if metas[ds]["task"] == "regression":
                if meta.get("noise_sigma"):
                    rmse = float(np.mean([m["rmse"] for m in mlist]))
                    rmse_ratio.append(rmse / meta["noise_sigma"])
            elif meta.get("bayes_brier") is not None:
                brier = float(np.mean([m["brier"] for m in mlist]))
                xs_brier.append(brier - meta["bayes_brier"])
        if not xs_brier and not rmse_ratio:
            continue
        xb = f"{np.mean(xs_brier):9.4f}" if xs_brier else "        -"
        rr = f"{np.mean(rmse_ratio):9.3f}" if rmse_ratio else "        -"
        print(f"{label:18s} {max(len(xs_brier), len(rmse_ratio)):4d} {xb} {rr}")


# ---------------------------------------------------------------------------
# realism checks
# ---------------------------------------------------------------------------
def realism_report(metas, per_model):
    models = sorted(per_model)
    print(f"models in run: {models}\n")

    def winrate(model_a, model_b, pred):
        a, b = primary_means(per_model, model_a), primary_means(per_model, model_b)
        shared = [ds for ds in set(a) & set(b) if pred(ds)]
        if not shared:
            return None, 0
        wins = sum(a[ds] > b[ds] + EPS for ds in shared)
        return wins / len(shared), len(shared)

    checks = []
    if "CatBoost" in models and "LightGBM" in models:
        hc = lambda ds: metas[ds]["synth"]["max_cardinality"] > 8
        num = lambda ds: metas[ds]["synth"]["n_cat"] == 0
        w_hc, n_hc = winrate("CatBoost", "LightGBM", hc)
        w_nu, n_nu = winrate("CatBoost", "LightGBM", num)
        ok = w_hc is not None and w_nu is not None and w_hc > w_nu
        checks.append(("CatBoost>LGBM winrate higher on high-card cat slice "
                       f"({w_hc if w_hc is None else round(w_hc, 2)}@{n_hc} vs "
                       f"{w_nu if w_nu is None else round(w_nu, 2)}@{n_nu})", ok))
    # no model dominates everywhere
    prim = {m: primary_means(per_model, m) for m in models}
    shared = set.intersection(*(set(v) for v in prim.values())) if prim else set()
    for m in models:
        wins = sum(all(prim[m][ds] >= prim[o][ds] - EPS for o in models if o != m)
                   for ds in shared)
        rate = wins / max(1, len(shared))
        checks.append((f"{m} best-on {rate:.0%} of datasets (<70% wanted)", rate < 0.70))
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new", nargs="?", default=None)
    ap.add_argument("--model", default="ChimeraBoost")
    ap.add_argument("--model-new", default=None,
                    help="model name for the NEW run's records (default: "
                         "--model). Lets an ensemble arm compare e.g. base "
                         "ChimeraBoost vs new ChimeraBoostEns2.")
    ap.add_argument("--metric", choices=["primary", "brier"], default="primary",
                    help="A/B judge metric. brier = classification sets only, "
                         "delta sign flipped so + means the arm is better.")
    ap.add_argument("--realism", action="store_true")
    args = ap.parse_args()

    metas, per_model = load_run(args.base)
    if not metas:
        raise SystemExit("no syn: datasets with synth meta in this run")
    if args.realism:
        realism_report(metas, per_model)
        return
    if args.new is None:
        print(f"excess view of {args.base} [model={args.model}]\n")
        excess_report(metas, per_model, args.model)
        return
    metas_new, per_new = load_run(args.new)
    metas.update(metas_new)
    model_new = args.model_new or args.model
    base = metric_means(per_model, args.model, args.metric)
    new = metric_means(per_new, model_new, args.metric)
    tag = args.model if model_new == args.model else f"{args.model}->{model_new}"
    print(f"A/B attribution: {args.base} -> {args.new} "
          f"[model={tag}, metric={args.metric}]\n")
    ab_report(metas, base, new)


if __name__ == "__main__":
    main()
