# The payoff program — closing the classification-Brier gap with the v2 suite

Self-sufficient handoff (the V2.md convention): everything needed to run this
program without its authoring session.

## State of the world (2026-07-15)

- SynthGen v2 SHIPPED: branch `synthgen-v2`, PR #16 (open; #15 merged). Gate
  7/9 + canary clean; all V2.md acceptance lines met. Verdicts in
  `benchmarks/synthgen/V2.md` + README.
- Baselines on disk (gitignored, KEEP them — they are the reusable A/B bases):
  `benchmarks/results/synv2-baseline.json` (screen, 136 sets, ~15 min/arm)
  and `synv2-full-baseline.json` (full, 211 sets, ~1 h).
- **The target** (percent-of-best table, `summarize.py synv2-full-baseline.json`):

  | Model | Reg RMSE% | Bin F1% | Bin Brier% | Bin Calib (MCB) | Speed |
  |---|---|---|---|---|---|
  | ChimeraBoost | **97.1** | 98.9 | 92.8 | 6.41m | 2.5× |
  | CatBoost | 94.9 | **99.5** | **96.5** | 6.38m | 73.9× |

  Bin Brier% 92.8 vs 96.5 is the biggest remaining accuracy-column gap
  (v1 read 91.4 vs 98.1; the entity cats narrowed it but did not close it).
- **Free diagnostic already in hand:** MCB (CORP miscalibration) is TIED
  (6.41 vs 6.38). CatBoost's Brier edge is REFINEMENT — sharper probabilities
  — not calibration. Calibration-flavored levers are therefore deprioritized
  until attribution says otherwise (pre-registered below).

Hard constraints (unchanged): pure Python (numpy/numba/sklearn); TabArena
sealed in every form; one benchmark at a time; script files, never
`python -c`; synth never ships anything alone — Grinsztajn decides, OpenML
one-shot gates. Trust file reads over scrolled console output.

## Already ruled out — do NOT re-run (see `benchmarks/research/SUMMARY.md`)

Seven CatBoost-mechanism ports, seven kills (C1 one-hot, C3 selective combos,
C4 cat-aware binning, G1 forest leaf refit, G2 mass-adaptive shrinkage, G3
adaptive leaf-estimation, G4 ordered+leaf); their flags were REMOVED from the
library (2026-06-15). C2 per-tree TS permutation deferred by architecture.
Also dead: broad HP random search (anti-generalizes, PMLB study),
tail-averaging, lr probe. Knob characterization says defaults are excellent on
real data BUT the cat knobs were blind there — Grinsztajn has almost no
high-card cats. **v2's entity cats are the first instrument that can actually
see them.** That is what this program exploits.

## Step 0 — tooling (~30 min, no benchmark)

`synth_report.py` and `backtest.py` judge on `metrics["primary"]` (F1 for
clf). A Brier-targeted sweep needs Brier attribution:

- Add `--metric {primary,brier}` to `synth_report.py` A/B mode: with `brier`,
  restrict to classification sets and use `metrics["brier"]` (lower=better —
  flip the delta sign so + still means "arm better"). Slices/OLS unchanged.
- Add `--model-new` (defaults to `--model`) so the L1 ensemble arm can compare
  baseline `ChimeraBoost` records against arm `ChimeraBoostEns2` records.
- Extend `tests/test_synth_report.py` with a planted Brier-slice effect.
- Optional: same flags on `compare_runs.py` if a sign-test headline is wanted.

## Step 1 — locate the gap (~0 cost, JSONs exist)

Script over `synv2-full-baseline.json`: per-slice CatBoost-vs-ChimeraBoost
Brier winrate + mean excess-Brier-vs-floor for both models, sliced by the
synth meta (cats=entity / entity_strength quartiles / card>8 / n<2000 /
depth / noise_level / imbalance / func_dominant). Output: the 2–3 slices
where CatBoost's Brier edge concentrates. Pre-registration: the lever whose
signature matches the located gap runs first; a lever that wins ONLY outside
those slices is suspect (variance, not mechanism).

## Step 2 — pre-registered lever queue (screen arms, ~15 min each)

Run each as one screen arm vs `synv2-baseline.json`; judge with
`synth_report.py BASE NEW --metric brier` + canary/car-analog sanity; kill
fast. Expected-value order:

- **L1 small probability ensembles** (`n_ensembles=2`, then 5; harness
  runners exist: `--models ChimeraBoost ChimeraBoostEns2 CatBoost ...` —
  the Ens2/Ens5/Ens10 model names in `run_benchmarks.py:744`). Averaging
  probabilities is the classic refinement lever and the library already
  ships it. Judged on Brier AND blended
  strength: ens2 ≈ 2× fit cost → ~5× total slowdown, still ~15× faster than
  CatBoost here. If it closes ≥half the Brier gap at ens2, this is the
  headline candidate; the ship decision is a Pareto call (user decides the
  speed trade, cross_features precedent).
- **L2 `leaf_estimation_iterations` 2 and 3** (default 1): extra Newton steps
  sharpen leaf probabilities — the refinement mechanism. Expected signature:
  clf Brier gains on deep/noisy sets; canary slice must stay flat; watch
  fit-time (should be ~+few %/step).
- **L3 `cat_smoothing` sweep ×{0.25, 4, 16} of default**: per-level TS prior
  strength. First real test now that entity cats + rare levels exist.
  Expected signature: gains concentrated on cats=entity / high
  entity_strength / card>8, ~ties elsewhere. If the entity slice moves >1%
  either direction, follow with a finer sweep before judging.
- **L4 (conditional) calibration flavored** (temperature/isotonic on the
  internal early-stopping split, post-fit): ONLY if step 1 finds an MCB-heavy
  slice (e.g. small-n or high-imbalance) — the aggregate says calibration is
  tied, so opening this without slice evidence is dredging.

Kill criterion per lever: clf-Brier sign test not favorable (or p>0.2) on the
screen, OR canary slice positive, OR reg/F1 collateral regression >0.3%
overall. Promote criterion: favorable sign test with a slice story matching
step 1, canary clean → `/experiment` (Grinsztajn decides, OpenML gates; PMLB
only if the winner is an HP retune). Budget: ~1 h screen time per idea vs ~6 h
Grinsztajn — kill on the screen, never tune the library on synth.

## Bookkeeping

- Branch off `main` AFTER PR #16 merges (levers touch library src; don't
  stack on the synthgen-v2 branch).
- Screen arms: `python benchmarks/run_benchmarks.py --synth --seeds 3
  --save benchmarks/results/brier-<lever>.txt <flags>` — the `--synth`
  default suite is already `screen`; always print the aggregate table.
- Any DEFAULT change that ships: refresh `images/pareto.png` (`/pareto`) and
  get user sign-off on speed trades.

## Results log (2026-07-15, branch brier-refinement)

- **Step 0 shipped**: `synth_report.py`/`compare_runs.py` `--metric brier`
  + `--model-new`; `--chimera-cat-smoothing` harness flag; brier rel-delta
  denominator floored at 0.01 (saturated sets sit at Brier ~0 and exploded
  the means). Tests extended (planted Brier slice, zero-floor case).
- **Step 1 verdict** (`brier_gap.py` on synv2-full-baseline): the gap is
  CATEGORICAL. cats=none dead flat (+0.0001/set, 54% winrate). Concentration:
  entity_strength Q4 conc 5.24 (CatBoost 91% winrate), card>16 3.70,
  saturated/cellrule 3.88, cats=entity 3.03. No MCB-heavy slice -> L4
  calibration LOCKED OUT (refinement, not calibration, everywhere).
  Pre-registration flipped L3 ahead of L1.
- **L3 cat_smoothing KILLED** (x{0.25,2,4,8,16} full dose curve): mechanism
  real (entity_strength top OLS factor every arm, t=+5.1 at cs4) but the
  clf-Brier sign test never clears p<=0.2 (best: all 19W-15L p=0.61; binary
  15W-7L p=0.13 at cs4), the dose curve is non-monotone around the default
  (cs2 reads -0.68%), and cs16 buys Brier with an accuracy breach (F1 entity
  -1.44%, reg -1.40%). Magnitude ceiling ~+0.2% binary Brier = not worth a
  PMLB retune.
- **L2 leaf_estimation_iterations KILLED**: premise was wrong (clf default
  is already 3; 1 is the regressor's). lei=5: Brier 9W-15L, entity slice
  -2.1% (p=0.049 AGAINST); lei=2: 12W-12L noise. Multiclass path ignores lei
  entirely (0-0-34 exact ties, both arms).
- **L1 ensembles PROMOTED off the screen**: Ens2 clf-Brier 64W-24L p<0.001
  (binary +2.89%); Ens5 82W-6L (binary +7.54%), accuracy also up (+0.82%
  p<0.001, reg +2.09%). Canary clean (F1 exact ties; canary Brier worse =
  safe direction; bootstrap thinning hurts near-floor sets, explains the
  negative slice means under floored rel-deltas). Screen percent-of-best:
  Ens2 Bin Brier% 93.4 vs CatBoost 93.5 at 4.4x vs 81.6x (gap CLOSED);
  Ens5 95.9 (BEATS CatBoost) at 11.2x, Reg RMSE% 96.0->98.0.
- **Next**: Grinsztajn A/B (baseline `merged-crossfeat-20260713.json` —
  its ChimeraBoost records match today's cross_features-on default; the
  22:45 sibling is flag-off) -> OpenML one-shot -> Pareto evidence; the
  n_ensembles default flip (1->2?) is the user's speed-trade call.

## Generator v3 watch items (do NOT act now; only at the next re-freeze)

- mcw1 arm disagreed in v2: its pre-registered small-n clf slice shrank to 10
  sets under the 0.35 small-n cap (W5-L5 +0.41%). Either widen the slice
  definition (n<3000?) or seed more small-n clf sets.
- depth4 is a not-win but not yet a strict loss (W61-L57 −0.11%) — if v3
  wants the flip to decisive, push the depth/width prior further.
- Saturated REGRESSION sets sit far from ceiling (rmse_ratio 8–15 in the
  canary re-verification log) — none qualify as canaries; reg canaries need a
  different construction (fewer cells or higher noise floor) if wanted.
- Entity-cat cardinality center is floored at 8 (see `emit._entity_column`);
  revisit against harvested marginals if the card>8 realism margin drifts.
