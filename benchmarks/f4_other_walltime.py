"""F4 fresh profile: name the "other" column of the default fit, by wall clock.

The August attribution (`profile_fit.py --attribution`,
`results/campaign-attr-20260816.md`) wrapped tree growth, binning, target
encoding, validation predict, leaf refinement and the linear update -- and left
23-30% of every Grinsztajn fit in a column called "other". F4's three shipped
candidates all came out of the hc/multiclass "other"; none of them can move
Grinsztajn, which is the headline stratum. This script decomposes what is left
on the DEFAULT estimator, so the next candidate is argued from a measured share
rather than a guess.

Method, and why it is not cProfile: F4 learned that a cProfile share overstates
many-tiny-calls objects and understates few-fat-calls ones (I011, I012). Every
object here is few-fat-calls -- a handful per boosting round -- so each is
wrapped in a `perf_counter` pair instead. Times are EXCLUSIVE: a hook's clock
stops while a nested hook runs, so the rows partition the estimator fit and sum
to it exactly. Rows are also split by leg, "es" (the early-stopped selection
fits) against "refit" (the full-data replay refit), since the refit's rounds
run `replay_oblivious_tree`, which the August instrument did not wrap at all.

The instrument is priced two ways so its own cost cannot pass for a finding:
hook overhead is calibrated on a no-op and reported as a row, and every wrapped
fit alternates with an unwrapped one in the same process (median of 3 each).
The overhead lands in the PARENT's exclusive time (the hook's bookkeeping runs
on the caller's clock), i.e. in the two "(self)" loop rows -- discount those by
the instrument row.

`--prep-detail` (CAMPAIGN_PLAN I026) pushes the same hooks INSIDE the `prep`
row, which is a third of the fit on high-cardinality sets: factorize, the
numeric-block unboxing of the object array every categorical dataset arrives
as, the ordered-TS fit with its numba kernel hooked separately (so the
remainder is the permutation draws and the averaging), binning, and the
preprocessor's own remainder (stacking, the feature-major transposes).

Run: python benchmarks/f4_other_walltime.py [--datasets KEY ...] [--out NAME]
         [--prep-detail] [--reps 5]
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb

import chimeraboost.binning as bnmod
import chimeraboost.booster as bmod
import chimeraboost.losses as lmod
import chimeraboost.preprocessing as pmod
import chimeraboost.sklearn_api as skmod
import chimeraboost.target_encoding as temod
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# The August attribution panel, unchanged, so the two reads line up row for row.
PANEL = [
    "gr:reg_num/cpu_act",
    "gr:reg_num/diamonds",
    "gr:reg_cat/nyc-taxi-green-dec-2016",
    "gr:clf_num/MagicTelescope",
    "gr:clf_num/Higgs",
    "gr:clf_cat/road-safety",
    "hc:kick",
    "hc:wine-reviews",
    "hc:okcupid-stem",
]

# Tree-kernel rows: closed to this family by B10/B15 (grow) or already the
# product of the REPLAY program (replay). Everything else is a candidate row.
KERNEL_ROWS = ("grow", "replay")

# Column order of the phase table. "(self)" rows are what is left of a wrapped
# region after its wrapped children are taken out.
PREP_ROWS = ["factorize", "combo", "eval_codes", "numeric_block",
             "ts_fit(self)", "ts_kernel", "ts_transform", "bin_fit",
             "bin_transform", "cross_block", "prep_fit(self)",
             "prep_tf(self)", "prep"]
PHASES = ["grow", "replay", *PREP_ROWS[:-1], "prep", "grad_hess",
          "train_update",
          "eval_advance", "val_score", "epilogue(self)", "loop(self)",
          "centers_std", "setup", "subsample", "fmask", "sketch", "callbacks",
          "as_array", "importances", "cross_screen", "calibrate", "es_split",
          "validate", "est(self)"]

_pc = time.perf_counter
_STACK = []                               # one [child_secs] frame per open hook
_SELF = collections.defaultdict(float)    # (leg, key) -> exclusive seconds
_CALLS = collections.Counter()            # (leg, key) -> calls
_LEG = ["est"]                            # "est" | "es" | "refit"
_FITS = []                                # per-booster-fit leg records


def _hook(key, func):
    """Exclusive-time wall-clock wrapper. The clock is read immediately around
    the call, so the bookkeeping below runs on the PARENT's exclusive time."""
    def wrapped(*a, **kw):
        frame = [0.0]
        _STACK.append(frame)
        t0 = _pc()
        try:
            return func(*a, **kw)
        finally:
            dt = _pc() - t0
            _STACK.pop()
            k = (_LEG[0], key)
            _SELF[k] += dt - frame[0]
            _CALLS[k] += 1
            if _STACK:
                _STACK[-1][0] += dt
    return wrapped


def _fit_hook(func):
    """Booster-fit wrapper: sets the leg, records the leg's inclusive seconds,
    and charges its own exclusive time (the Python round loop) to loop(self)."""
    inner = _hook("loop(self)", func)

    def wrapped(self, X, y, *a, **kw):
        eval_set = kw.get("eval_set", a[1] if len(a) > 1 else None)
        if eval_set is None:
            role = ("refit:replay" if getattr(self, "replay_donor", None)
                    is not None else "refit:scratch")
        else:
            role = (("cross" if self.cross_pairs else "base")
                    + (":lin" if self.linear_leaves else ":const"))
        prev = _LEG[0]
        _LEG[0] = "refit" if eval_set is None else "es"
        t0 = _pc()
        try:
            return inner(self, X, y, *a, **kw)
        finally:
            _LEG[0] = prev
            _FITS.append({"role": role, "secs": _pc() - t0,
                          "trees": len(self.trees_),
                          "rounds": len(self.valid_history_)})
    return wrapped


def _targets():
    """(owner, attribute, key) for every hook. Module globals are patched on
    the module that LOOKS THEM UP (booster/sklearn_api import by name)."""
    B, G, M = bmod._BaseBooster, bmod.GradientBoosting, bmod.MulticlassBoosting
    return [
        (bmod, "build_oblivious_tree", "grow"),
        (bmod, "replay_oblivious_tree", "replay"),
        (bmod, "_eval_advance", "eval_advance"),
        (bmod, "_run_callbacks", "callbacks"),
        (bmod, "as_model_array", "as_array"),
        (skmod, "as_model_array", "as_array"),
        (B, "_round_epilogue", "epilogue(self)"),
        (B, "_val_score", "val_score"),
        (B, "_maybe_subsample", "subsample"),
        (B, "_mvs_row_weights", "subsample"),
        (B, "_feature_mask", "fmask"),
        (B, "_build_centers_std", "centers_std"),
        (B, "_prep_matrices", "prep"),
        (B, "_alloc_hist_buffers", "setup"),
        (B, "_capture_shap_background", "setup"),
        # On the default path predict_raw runs inside fit only to calibrate
        # (temperature scaling, the conformal offset). A `cross_top_columns`
        # screen predicts too and would land in this row, not cross_screen.
        (B, "predict_raw", "calibrate"),
        (G, "_prep_or_replay_matrices", "prep"),
        (G, "_apply_training_update", "train_update"),
        (M, "_apply_vector_update", "train_update"),
        (M, "_sketch_direction", "sketch"),
        (lmod.RMSE, "grad_hess", "grad_hess"),
        (lmod.Logloss, "grad_hess", "grad_hess"),
        (lmod.MultiSoftmax, "grad_hess", "grad_hess"),
        (skmod, "_auto_es_split", "es_split"),
        (skmod, "_validate_fit_input", "validate"),
        (skmod, "_screened_cross_pairs", "cross_screen"),
        (skmod, "_cross_candidate_pairs", "cross_screen"),
        (skmod, "_fit_temperature", "calibrate"),
        (skmod.ChimeraBoostRegressor, "_conformal_quantile_offset",
         "calibrate"),
        (skmod.ChimeraBoostRegressor, "fit", "est(self)"),
        (skmod.ChimeraBoostClassifier, "fit", "est(self)"),
    ]


def _prep_targets():
    """The hooks `--prep-detail` adds inside the `prep` row. With them on,
    `prep` itself shrinks to the booster-level remainder: the feature-major
    transposes and the cache splice."""
    P, E, Bn = (pmod.FeaturePreprocessor, temod.OrderedTargetEncoder,
                bnmod.Binner)
    return [
        (pmod.CatTransformCache, "column", "factorize"),
        (pmod.CatTransformCache, "combo", "combo"),
        (P, "_numeric_block", "numeric_block"),
        (P, "_codes_for_transform", "eval_codes"),
        (P, "_combo_codes_for_transform", "eval_codes"),
        (P, "_cross_block", "cross_block"),
        (P, "_fit_gdiff", "cross_block"),
        (P, "fit_transform", "prep_fit(self)"),
        (P, "transform", "prep_tf(self)"),
        (P, "from_base_with_cross", "prep_fit(self)"),
        (E, "fit_transform", "ts_fit(self)"),
        (E, "transform", "ts_transform"),
        (temod, "_ordered_ts", "ts_kernel"),
        (temod, "_ordered_ts_weighted", "ts_kernel"),
        (Bn, "fit_transform", "bin_fit"),
        (Bn, "transform", "bin_transform"),
    ]


_INSTALLED = []
_PREP_DETAIL = [False]
_DECISION_FLOOR = [5.0]


def _wrap(key, orig):
    """Hook a plain function, or the function inside a class/static method."""
    if isinstance(orig, classmethod):
        return classmethod(_hook(key, orig.__func__))
    if isinstance(orig, staticmethod):
        return staticmethod(_hook(key, orig.__func__))
    return _hook(key, orig)


def _install():
    targets = _targets() + (_prep_targets() if _PREP_DETAIL[0] else [])
    for owner, name, key in targets:
        # Classes: take the raw attribute out of __dict__ so restoring it puts
        # back exactly what was there, descriptors included.
        orig = owner.__dict__.get(name) if isinstance(owner, type) else None
        if orig is None:
            orig = getattr(owner, name)
        setattr(owner, name, _wrap(key, orig))
        _INSTALLED.append((owner, name, orig))
    for cls in (bmod.GradientBoosting, bmod.MulticlassBoosting):
        orig = cls.__dict__["_fit_impl"]
        setattr(cls, "_fit_impl", _fit_hook(orig))
        _INSTALLED.append((cls, "_fit_impl", orig))
    # feature_importances_ is a property evaluated in the CALLER of
    # _cross_candidate_pairs, so it needs its own hook to leave est(self).
    prop = bmod._BaseBooster.__dict__["feature_importances_"]
    bmod._BaseBooster.feature_importances_ = property(
        _hook("importances", prop.fget))
    _INSTALLED.append((bmod._BaseBooster, "feature_importances_", prop))


def _uninstall():
    while _INSTALLED:
        owner, name, orig = _INSTALLED.pop()
        setattr(owner, name, orig)


def _reset():
    _SELF.clear()
    _CALLS.clear()
    del _FITS[:]
    del _STACK[:]
    _LEG[0] = "est"


def _calibrate(n=200_000):
    """Seconds of bookkeeping one hook adds, measured on a no-op."""
    def noop():
        return None
    hooked = _hook("cal", noop)
    t0 = _pc()
    for _ in range(n):
        noop()
    plain = _pc() - t0
    t0 = _pc()
    for _ in range(n):
        hooked()
    cost = (_pc() - t0 - plain) / n
    _reset()
    return max(cost, 0.0)


def _fit_once(Est, Xtr, ytr, cat):
    t0 = _pc()
    Est(random_state=0).fit(Xtr, ytr, cat_features=cat)
    return _pc() - t0


def run_dataset(key, reps, hook_cost):
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(0))
    strat = y if task != "regression" else None
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=0,
                                      stratify=strat)
    Est = ChimeraBoostRegressor if task == "regression" \
        else ChimeraBoostClassifier
    print(f"\n=== {key}  ({task}, n_train={len(Xtr)}, "
          f"feats={Xtr.shape[1]}, cats={len(cat) if cat else 0}) ===",
          flush=True)

    _fit_once(Est, Xtr, ytr, cat)          # untimed: compile + warm caches

    plain, wrapped = [], []
    phase = collections.defaultdict(float)
    calls = collections.Counter()
    fits = None
    for _ in range(reps):
        plain.append(_fit_once(Est, Xtr, ytr, cat))
        _reset()
        _install()
        try:
            wrapped.append(_fit_once(Est, Xtr, ytr, cat))
        finally:
            _uninstall()
        for k, v in _SELF.items():
            phase[k] += v / reps
        for k, v in _CALLS.items():
            calls[k] += v
        fits = list(_FITS)                 # deterministic fit: same every rep
    n_calls = sum(calls.values()) / reps
    fit_s = sum(wrapped) / reps            # the mean the phase means sum to
    rec = {
        "dataset": key, "task": task, "n_train": int(len(Xtr)),
        "fit_s": fit_s,
        "plain_median_s": statistics.median(plain),
        "wrapped_median_s": statistics.median(wrapped),
        "plain_s": plain, "wrapped_s": wrapped,
        "instrument_est_s": n_calls * hook_cost,
        "hook_calls": n_calls,
        "phase_s": {f"{leg}|{k}": v for (leg, k), v in phase.items()},
        "phase_calls": {f"{leg}|{k}": v / reps for (leg, k), v in calls.items()},
        "legs": fits,
    }
    print(f"  plain {rec['plain_median_s']:.3f}s  wrapped "
          f"{rec['wrapped_median_s']:.3f}s  (x"
          f"{rec['wrapped_median_s'] / rec['plain_median_s']:.3f}; "
          f"{n_calls:.0f} hook calls ~ {rec['instrument_est_s'] * 1e3:.1f} ms)",
          flush=True)
    return rec


def _share(rec, key, legs=("est", "es", "refit")):
    s = sum(rec["phase_s"].get(f"{leg}|{key}", 0.0) for leg in legs)
    return 100.0 * s / rec["fit_s"]


def _perm_seconds(n, passes, reps=3):
    """Seconds `passes` draws of `rng.permutation(n)` cost -- the part of
    ts_fit(self) that is not averaging. A microbench, reported beside the
    in-fit number and never instead of it."""
    rng = np.random.default_rng(0)
    best = float("inf")
    for _ in range(reps):
        t0 = _pc()
        for _ in range(passes):
            rng.permutation(n)
        best = min(best, _pc() - t0)
    return best


def _prep_tables(rows):
    out = ["", "## The preprocessing split, % of estimator fit (all legs)", ""]
    used = [p for p in PREP_ROWS if any(_share(r, p) >= 0.05 for r in rows)]
    out.append("| dataset | prep total | " + " | ".join(used) + " |")
    out.append("|---|--:|" + "--:|" * len(used))
    for r in rows:
        tot = sum(_share(r, p) for p in PREP_ROWS)
        cells = " | ".join(f"{_share(r, p):.1f}" for p in used)
        out.append(f"| {r['dataset']} | {tot:.1f} | {cells} |")

    out += ["", "## The same split inside the refit leg only", ""]
    out.append("| dataset | refit prep | " + " | ".join(used) + " |")
    out.append("|---|--:|" + "--:|" * len(used))
    for r in rows:
        tot = sum(_share(r, p, ("refit",)) for p in PREP_ROWS)
        cells = " | ".join(f"{_share(r, p, ('refit',)):.1f}" for p in used)
        out.append(f"| {r['dataset']} | {tot:.1f} | {cells} |")

    out += ["", "## Ordered-TS passes: kernel vs the permutation draws", "",
            "| dataset | kernel passes per fit | ts_kernel % | ts_fit(self) % "
            "| permutation draws, microbench % |",
            "|---|--:|--:|--:|--:|"]
    for r in rows:
        passes = sum(v for k, v in r["phase_calls"].items()
                     if k.endswith("|ts_kernel"))
        if not passes:
            continue
        # The selection legs encode the 80% split and the refit every row, so
        # price the draws at the mean of the two row counts.
        n_mean = int(r["n_train"] * 0.9)
        perm = 100.0 * _perm_seconds(n_mean, int(passes)) / r["fit_s"]
        out.append(f"| {r['dataset']} | {passes:.0f} | "
                   f"{_share(r, 'ts_kernel'):.1f} | "
                   f"{_share(r, 'ts_fit(self)'):.1f} | {perm:.1f} |")
    return out


def report(rows, hook_cost):
    out = ["# F4 fresh profile: the default fit by exclusive wall clock", "",
           f"Hook bookkeeping calibrated at {hook_cost * 1e6:.2f} us per call; "
           "it lands in the two loop rows marked (self).", "",
           "## Phases, % of estimator fit (all legs)", ""]
    used = [p for p in PHASES
            if any(_share(r, p) >= 0.05 for r in rows)]
    out.append("| dataset | fit s | " + " | ".join(used) + " | instr |")
    out.append("|---|--:|" + "--:|" * (len(used) + 1))
    for r in rows:
        cells = " | ".join(f"{_share(r, p):.1f}" for p in used)
        instr = 100.0 * r["instrument_est_s"] / r["fit_s"]
        out.append(f"| {r['dataset']} | {r['fit_s']:.2f} | {cells} "
                   f"| {instr:.1f} |")

    out += ["", "## The same rows inside the refit leg only "
            "(% of estimator fit)", ""]
    used_r = [p for p in PHASES
              if any(_share(r, p, ("refit",)) >= 0.05 for r in rows)]
    out.append("| dataset | refit total | " + " | ".join(used_r) + " |")
    out.append("|---|--:|" + "--:|" * len(used_r))
    for r in rows:
        tot = sum(_share(r, p, ("refit",)) for p in PHASES)
        cells = " | ".join(f"{_share(r, p, ('refit',)):.1f}" for p in used_r)
        out.append(f"| {r['dataset']} | {tot:.1f} | {cells} |")

    if _PREP_DETAIL[0]:
        out += _prep_tables(rows)

    out += ["", "## Legs (booster fits in order; inclusive seconds, trees)", ""]
    for r in rows:
        legs = ", ".join(f"{f['role']} {f['secs']:.2f}s/{f['trees']}t"
                         for f in r["legs"])
        out.append(f"- {r['dataset']}: {legs}")

    out += ["", "## Instrument check (same process, alternating, median of "
            "the reps)", "",
            "| dataset | plain s | wrapped s | ratio | hook calls |",
            "|---|--:|--:|--:|--:|"]
    for r in rows:
        out.append(f"| {r['dataset']} | {r['plain_median_s']:.3f} "
                   f"| {r['wrapped_median_s']:.3f} "
                   f"| {r['wrapped_median_s'] / r['plain_median_s']:.3f} "
                   f"| {r['hook_calls']:.0f} |")

    floor = _DECISION_FLOOR[0]
    out += ["", f"## Decision read: non-kernel rows at or above {floor:g}% "
            "of fit", ""]
    for r in rows:
        big = [(p, _share(r, p)) for p in PHASES
               if p not in KERNEL_ROWS and _share(r, p) >= floor]
        txt = ", ".join(f"{p} {s:.1f}%" for p, s in
                        sorted(big, key=lambda t: -t[1])) or "none"
        out.append(f"- {r['dataset']}: {txt}")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PANEL)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="campaign-f4s1-other")
    ap.add_argument("--prep-detail", action="store_true",
                    help="hook inside the prep row (I026)")
    ap.add_argument("--decision-floor", type=float, default=5.0,
                    help="percent of fit a row needs to be listed")
    args = ap.parse_args()
    _PREP_DETAIL[0] = args.prep_detail
    _DECISION_FLOOR[0] = args.decision_floor

    rb._add_grinsztajn_datasets()
    rb._add_highcard_datasets()
    print(f"chimeraboost from {os.path.dirname(bmod.__file__)}", flush=True)
    print("Warmup (compiling numba kernels)...", flush=True)
    from chimeraboost.warmup import warmup
    warmup()
    hook_cost = _calibrate()
    print(f"hook bookkeeping: {hook_cost * 1e6:.2f} us per call", flush=True)

    rows = [run_dataset(k, args.reps, hook_cost) for k in args.datasets]

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", args.out)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".json", "w") as f:
        json.dump({"hook_cost_s": hook_cost, "rows": rows}, f)
    text = report(rows, hook_cost)
    with open(base + ".md", "w", newline="\n") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved {base}.json and {base}.md")


if __name__ == "__main__":
    main()
