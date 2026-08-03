"""
benchmarks/synthetic.py — Stage-1 synthetic generators + sweep harness.

Probes the oblivious-tree "sharpness tax" (benchmarks/PROJECT_STATUS.md §6, §9) on
PARAMETRIC families, so a candidate change is judged by MECHANISM (Brier vs n,
Brier vs label noise) BEFORE it ever touches a real-dataset suite. This is
Stage 1 of the 4-stage loop: synthetic (fit/inspect) -> Grinsztajn (test) ->
OpenML (gate) -> TabArena-Lite (sealed).

Design note — why this imports run_benchmarks rather than re-specifying models:
every model here is run through ``run_benchmarks.RUNNERS``, i.e. configured
EXACTLY as in the Grinsztajn benchmark (MAX_ITERS=2000, patience 50, the same
internal validation split, the same multiclass-form Brier in _compute_metrics).
A Stage-1 conclusion therefore transfers to Stage 2 with no config confound.

The three families (docs §9.1):
  A  Sparse Local Interaction (asymmetric-friendly) — oblivious SHOULD lose.
  B  Global Additive (symmetric-friendly)           — control; oblivious ties/wins.
  C  Noise sweep of A (the regularization pivot)     — find the tax->dividend crossover.

Run:  python benchmarks/synthetic.py --family all
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless; we only ever savefig
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reuse the exact Grinsztajn model configs (see module docstring).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "images")


# =====================================================================
# 1. PARAMETRIC SYNTHETIC GENERATORS  (all binary classification)
#
# Every generator returns (X, y, true_prob), where true_prob[i] = P(y=1|x_i)
# under the data-generating process. true_prob lets the harness subtract the
# irreducible Bayes-optimal Brier floor and report EXCESS Brier (§9.3), which
# isolates estimation/sharpness error from label noise. Only possible because
# the DGP is known — the whole point of Stage 1.
# =====================================================================

def family_a_sparse_local(n_samples, n_features=10, n_interact=3,
                          threshold=0.0, seed=42):
    """Family A v1 — single Sparse Local Interaction (DEGENERATE, kept for ref).

    y = 1 iff x0..x{k-1} ALL > threshold. NOTE (§9.3 finding): noiseless, this
    is *exactly representable* by an oblivious depth-k tree (the positive octant
    is one leaf), so ALL models hit ~0 Brier and there is NO tax. Retained only
    to demonstrate that the tax is a capacity/noise effect, not a clean-rule one.
    Use family_a_multi_pocket (v2) for an actual large-n discriminator.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    signal = np.all(X[:, :n_interact] > threshold, axis=1)
    y = signal.astype(np.int64)
    return X, y, signal.astype(np.float64)  # noiseless -> true prob is the label


def family_a_multi_pocket(n_samples, n_features=10, n_pockets=4, threshold=0.3,
                          seed=42):
    """Family A v2 — Disjoint Multi-Pocket Interaction (the real discriminator).

    `n_pockets` disjoint coordinate-aligned 2-way AND pockets, each on its OWN
    feature pair (pocket j uses features 2j, 2j+1), with alternating sign so the
    pockets sit in different corners. y = OR of the pockets.

    Why this taxes us where v1 didn't: with n_pockets=4 the target depends on
    **8 informative features**, but a single oblivious tree at our default
    depth=6 can split on at most 6 features -> it physically cannot address all
    pockets in one tree, and each global split it does make fragments the
    unrelated pockets' regions. A leaf-wise tree isolates each pocket in a
    shallow local branch. So the deficit should PERSIST at large n (unlike v1).
    Noiseless -> true prob is the 0/1 signal.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    assert 2 * n_pockets <= n_features, "need 2 features per pocket"
    signal = np.zeros(n_samples, dtype=bool)
    for j in range(n_pockets):
        a, b = X[:, 2 * j], X[:, 2 * j + 1]
        if j % 2 == 0:
            pocket = (a > threshold) & (b > threshold)
        else:
            pocket = (a < -threshold) & (b < -threshold)
        signal |= pocket
    y = signal.astype(np.int64)
    return X, y, signal.astype(np.float64)


def family_b_global_additive(n_samples, n_features=10, n_informative=5, seed=42):
    """Family B — Global Additive (symmetric/oblivious-friendly CONTROL).

    y ~ Bernoulli(sigmoid(X[:, :m] @ beta)). A single global hyperplane, no
    isolated pockets -> oblivious global splits should match/beat leaf-wise.
    Proves a Family-A loss is STRUCTURAL geometry, not a generic Chimera bug.
    Here true_prob is the actual sigmoid (a genuinely probabilistic DGP), so the
    Bayes floor is nonzero and excess-Brier is meaningful.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_samples, n_features))
    beta = np.linspace(1.5, 0.3, n_informative) * np.where(
        np.arange(n_informative) % 2 == 0, 1.0, -1.0)
    logits = X[:, :n_informative] @ beta
    true_prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0.0, 1.0, size=n_samples) < true_prob).astype(np.int64)
    return X, y, true_prob


def family_c_noise_sweep(n_samples, noise_rate, n_features=10, n_pockets=4,
                         threshold=0.3, seed=42):
    """Family C — Family A v2 with symmetric label noise (flip y w.p. p).

    At p=0 leaf-wise wins on sharpness; as p grows the oblivious
    fragmentation+shrinkage becomes defensive. The EXCESS-Brier crossover (not
    the raw Brier, which is swamped by the noise floor) is the number we want.
    Under a symmetric flip at rate p, the post-noise true prob is
        P(y=1|x) = clean_prob*(1-2p) + p.
    """
    X, y_clean, clean_prob = family_a_multi_pocket(
        n_samples, n_features, n_pockets, threshold, seed)
    true_prob = clean_prob * (1.0 - 2.0 * noise_rate) + noise_rate
    if noise_rate <= 0.0:
        return X, y_clean, true_prob
    rng = np.random.default_rng(seed + 7919)  # disjoint stream from features
    flip = rng.uniform(0.0, 1.0, size=n_samples) < noise_rate
    y = y_clean.copy()
    y[flip] = 1 - y[flip]
    return X, y, true_prob


FAMILIES = {
    "a": family_a_multi_pocket,     # v2 — the discriminator (CLI default for "a")
    "a1": family_a_sparse_local,    # v1 — degenerate, reference only
    "b": family_b_global_additive,
    "c": family_c_noise_sweep,
}


def _bayes_brier(y_test, true_prob_test):
    """Bayes-optimal Brier in the SAME multiclass-sum convention as
    run_benchmarks._compute_metrics (binary -> sum over 2 classes = 2*(p-y)^2),
    so it subtracts cleanly from the runner's reported metrics['brier']."""
    return float(2.0 * np.mean((true_prob_test - y_test) ** 2))


# =====================================================================
# 2. STAGE-1 SWEEP HARNESS
# =====================================================================

def _select_models(requested=None):
    """ChimeraBoost + a leaf-wise reference (LightGBM) + an OBLIVIOUS control
    (CatBoost) + sklearn_HGB, filtered to what's installed. The CatBoost control
    is the point: if CatBoost (also oblivious) shows the same Family-A deficit,
    the tax is structural to oblivious trees, not a Chimera defect."""
    if requested:
        return list(requested)
    wanted = ["ChimeraBoost", "LightGBM", "CatBoost", "sklearn_HGB"]
    out = []
    for m in wanted:
        if m in ("ChimeraBoost", "sklearn_HGB"):
            out.append(m)
        elif m == "LightGBM" and rb.HAVE.get("lightgbm"):
            out.append(m)
        elif m == "CatBoost" and rb.HAVE.get("catboost"):
            out.append(m)
    return out


def run_sweep(family_fn, sweep_param, values, models, n_runs=3,
              fixed_kwargs=None, threads=None):
    """Sweep one generator knob; return per-model mean/std EXCESS Brier per value.

    Excess Brier = metrics['brier'] (model, on the held-out test) − Bayes-optimal
    Brier (from the known true_prob, same convention). It subtracts the
    irreducible noise floor so what remains is estimation/sharpness error (§9.3).

    For each (value, seed) we build (X, y, true_prob), hold out 20% as the TEST
    set, and run each model via run_benchmarks.RUNNERS — which carve their OWN
    internal validation split from the remaining train (test never touched).
    Returns {"mean":{m:[...]}, "std":{m:[...]}, "bayes":[...], ...}.
    """
    fixed_kwargs = dict(fixed_kwargs or {})
    mean = {m: [] for m in models}
    std = {m: [] for m in models}
    bayes_floor = []

    for val in values:
        per_model = {m: [] for m in models}
        per_bayes = []
        for run_idx in range(n_runs):
            seed = 1000 + run_idx
            kwargs = dict(fixed_kwargs, seed=seed)
            kwargs[sweep_param] = val
            X, y, true_prob = family_fn(**kwargs)
            Xtr, Xte, ytr, yte, _ptr, pte = train_test_split(
                X, y, true_prob, test_size=0.2, random_state=seed, stratify=y)
            bayes = _bayes_brier(yte, pte)
            per_bayes.append(bayes)
            for m in models:
                try:
                    metrics, _t, _it = rb.RUNNERS[m](
                        "binary", Xtr, ytr, Xte, yte, None, threads)
                    # max(0,·) guards float noise where model ≈ Bayes.
                    per_model[m].append(max(0.0, metrics["brier"] - bayes))
                except Exception as e:  # one model failing shouldn't kill the sweep
                    print(f"  ! {m} failed at {sweep_param}={val} seed={seed}: {e}")
                    per_model[m].append(np.nan)
        bayes_floor.append(float(np.mean(per_bayes)))
        for m in models:
            mean[m].append(float(np.nanmean(per_model[m])))
            std[m].append(float(np.nanstd(per_model[m])))
        cells = "  ".join(f"{m}={mean[m][-1]:.4f}" for m in models)
        print(f"  {sweep_param}={val!s:<7} | bayes={bayes_floor[-1]:.4f} | "
              f"excess: {cells}")
    return {"mean": mean, "std": std, "values": list(values), "bayes": bayes_floor,
            "param": sweep_param, "models": list(models)}


# =====================================================================
# 3. PLOTTING
# =====================================================================

def _baseline_for_delta(models):
    """The leaf-wise reference we measure the tax against: LightGBM if present,
    else sklearn_HGB (also leaf-wise / asymmetric)."""
    for cand in ("LightGBM", "sklearn_HGB"):
        if cand in models:
            return cand
    return None


def plot_sweep(res, title, xlabel, save_path, logx=False):
    """Two panels: (left) absolute Brier curves; (right) ChimeraBoost minus the
    leaf-wise baseline (>0 = ChimeraBoost pays the sharpness tax here)."""
    mean, std, values, models = res["mean"], res["std"], res["values"], res["models"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for m in models:
        mu = np.array(mean[m])
        sd = np.array(std[m])
        axL.plot(values, mu, marker="o", linewidth=2, label=m)
        axL.fill_between(values, mu - sd, mu + sd, alpha=0.12)
    axL.set_xlabel(xlabel)
    axL.set_ylabel("Excess Brier over Bayes (lower = better)")
    axL.set_title(title)
    axL.grid(True, linestyle="--", alpha=0.5)
    axL.legend()
    if logx:
        axL.set_xscale("log")

    base = _baseline_for_delta(models)
    if base and "ChimeraBoost" in models:
        delta = np.array(mean["ChimeraBoost"]) - np.array(mean[base])
        axR.axhline(0.0, color="k", linewidth=1)
        axR.plot(values, delta, marker="s", color="crimson", linewidth=2)
        axR.fill_between(values, np.minimum(delta, 0), 0, color="green", alpha=0.15)
        axR.fill_between(values, np.maximum(delta, 0), 0, color="crimson", alpha=0.15)
        axR.set_xlabel(xlabel)
        axR.set_ylabel(f"ChimeraBoost - {base} Brier")
        axR.set_title(f"Sharpness tax (>0 worse than {base})")
        axR.grid(True, linestyle="--", alpha=0.5)
        if logx:
            axR.set_xscale("log")
    else:
        axR.axis("off")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  -> saved {save_path}")


# =====================================================================
# 4. CLI
# =====================================================================

def _grids(quick=False, full=False):
    if quick:
        return [400, 1000, 2500], [0.0, 0.15, 0.3]
    if full:
        return [400, 800, 1500, 3000, 6000, 12000], [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]
    return [500, 1000, 2000, 4000, 8000], [0.0, 0.1, 0.2, 0.3, 0.4]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["a", "a1", "b", "c", "all"], default="all")
    ap.add_argument("--runs", type=int, default=3, help="seeds per sweep point")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"override model set. Available: {list(rb.RUNNERS)}")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-grid", type=int, nargs="+", default=None,
                    help="explicit sample-size grid (overrides --quick/--full)")
    ap.add_argument("--noise-n", type=int, default=4000,
                    help="fixed sample size for the Family C noise sweep")
    args = ap.parse_args()

    models = _select_models(args.models)
    n_grid, noise_grid = _grids(args.quick, args.full)
    if args.n_grid:
        n_grid = args.n_grid
    print(f"Stage-1 synthetic sweeps | models={models} | runs={args.runs}")

    todo = ["a", "b", "c"] if args.family == "all" else [args.family]

    if "a" in todo:
        print("\n[Family A v2] Disjoint multi-pocket (8 informative feats > depth 6) "
              "— excess Brier vs sample size (oblivious SHOULD trail leaf-wise):")
        res = run_sweep(family_a_multi_pocket, "n_samples", n_grid, models,
                        n_runs=args.runs, threads=args.threads)
        plot_sweep(res, "Family A v2: disjoint multi-pocket interaction",
                   "sample size n", os.path.join(IMAGES_DIR, "stage1_family_a_n.png"),
                   logx=True)

    if "a1" in todo:
        print("\n[Family A v1] Single sparse local (DEGENERATE ref) "
              "— expect ~0 excess for all (oblivious represents it exactly):")
        res = run_sweep(family_a_sparse_local, "n_samples", n_grid, models,
                        n_runs=args.runs, threads=args.threads)
        plot_sweep(res, "Family A v1: single pocket (degenerate, reference)",
                   "sample size n", os.path.join(IMAGES_DIR, "stage1_family_a1_n.png"),
                   logx=True)

    if "b" in todo:
        print("\n[Family B] Global Additive — Brier vs sample size "
              "(CONTROL; oblivious should match/beat leaf-wise):")
        res = run_sweep(family_b_global_additive, "n_samples", n_grid, models,
                        n_runs=args.runs, threads=args.threads)
        plot_sweep(res, "Family B: global additive (control)",
                   "sample size n", os.path.join(IMAGES_DIR, "stage1_family_b_n.png"),
                   logx=True)

    if "c" in todo:
        print(f"\n[Family C] Noise sweep at n={args.noise_n} — Brier vs label noise "
              "(find the tax->dividend crossover):")
        res = run_sweep(family_c_noise_sweep, "noise_rate", noise_grid, models,
                        n_runs=args.runs,
                        fixed_kwargs={"n_samples": args.noise_n},
                        threads=args.threads)
        plot_sweep(res, f"Family C: label-noise sweep (n={args.noise_n})",
                   "label flip probability", os.path.join(IMAGES_DIR, "stage1_family_c_noise.png"))

    print("\nStage-1 sweeps done.")


if __name__ == "__main__":
    main()
