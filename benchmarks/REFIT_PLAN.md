# REFIT — full-data refit of the ES winner (pre-registered 2026-07-24)

## Mechanism

The single-model auto fit carves `validation_fraction=0.2` for early stopping,
selection and calibration; the shipped model never trains on those rows — a
permanent 20% data tax. Once ES/selection/calibration have done their jobs,
refit the winning configuration on 100% of the training data at the selected
budget. Same family as AutoGluon `refit_full` and sklearn CV-then-refit; the
bagging program's law ("more effective data per member beats sampling
diversity" — B-samp, B2a) is the in-house precedent.

## Design (implemented as `refit_full`, default False for the A/B)

- Triggers ONLY on the auto-split path (early stopping active, no user
  `eval_set`). Explicit eval_set / early_stopping=False / bag members
  (explicit member OOB eval) / too-small-to-split: bit-identical no-op.
- Quantile loss: no-op (the conformal offset's validity needs a genuine
  holdout; refitting under it would silently break coverage).
- Refit config = winner config pinned: selected linear_leaves variant,
  selected cross_pairs, resolved lr (auto-lr would re-resolve at the new n and
  change what T* means). Size-adaptive autos (mcw, cat_combinations) re-resolve
  at full n by design.
- Rounds: `ceil(T* / (1 - validation_fraction))` — the probe's refitX arm
  (rounds scaled by the train-size ratio) beat plain T* on 8/10 sets
  (+3.98% vs +3.43% mean). Fallback pre-registered: if tier 1 shows a loss
  tail attributable to over-extension, drop to plain T* and re-screen.
- Calibration transfer: temperature (clf) and CQR offset (reg quantile — moot,
  no-op) are fit on the ES winner's val scores as today and applied to the
  refit model's raw scores. `validation_history_` keeps the ES fit's curve.
- User callbacks observe the ES fits only, not the refit (documented).

## Probe evidence (benchmarks/probe_fulldata_refit.py, 10 sets x 3 seeds,
config-pinned: cross off, reg ll=False)

10/10 datasets improved; refit +3.43% mean, refitX +3.98%. Concentrated where
learning curves are steep: cpu_act +10.2%, pol +6.7%, covertype +7.7%,
electricity +4.6%, wine_quality +3.4%. Caveat carried forward: shipped-config
deltas will be smaller where crosses/LL already fix the same underfit.

## Pre-registered predictions

- Tier 1 (synth screen): broad positive on the primary metric; effect
  concentrated in small-n and high-signal/complex slices (steep learning
  curves). Canaries exactly flat is NOT expected (this lever changes real
  fits) — but at-ceiling sets should show no OVERFIT regression cluster.
  Kill if: mean negative, or a systematic loss cluster in low-noise slices
  (= T* transfer overfits without the ES safety net).
- Tier 2: Grinsztajn AND hc sign tests, separately; expect broad wins on
  both (mechanism is regime-independent). Fit cost expected ~+40-80% single
  (one extra winner-length fit, no per-round val evals); speed verdict goes
  to the Pareto refresh.
- Gate: --openml one-shot, non-negative required.

## Arms

- BASE: current library, defaults (flag off; library byte-identical to main).
- VARIANT: `--chimera-refit-full` (harness flag) => refit_full=True.

## Results log

- 2026-07-24 tests: 556 green (10 new in tests/test_refit_full.py);
  default-off paths bit-identical by construction + test.
- **Tier 1 (synth screen) PASS**: base 20260724-125452 vs variant
  20260724-125608 (worktree results/): **95W-39L-2T, mean +1.023%**
  (sign bar 69, p≈0). Attribution: reg +2.19% (40-8), binary +0.17%
  (p=.04), multi +0.72%; n<2000 +1.27% vs n>=2000 +0.89% (steep-curve
  concentration as predicted); canaries flat (0-1-2, −0.16%); no
  low-noise overfit cluster (noise_level t=−0.47); saturated slice
  −0.84% p=.61 = the at-ceiling zone, noise-level. Fallback (drop the
  rounds scaling) NOT triggered.
- **Tier 2 Grinsztajn PASS**: certified base 20260723-192007
  (fingerprint 27/27 exact) vs variant 20260724-125955:
  **48W-11L, mean +2.000%** primary; **Brier 18W-5L +2.428%**; worst
  loss −0.29% (credit). The reg CatBoost-gap cluster collapses:
  Brazilian_houses +18.4/+19.8%, sulfur +11.3%, visualizing_soil
  +13.5%, cpu_act +6.6%, pol +5.8%, superconduct +4.4%.
  **Fit cost: ChimeraBoost summed 492→974 s = ×1.98** (the refit is a
  second winner-length fit at 1.25× rows; expected).
- **Tier 2 hc PASS**: certified base 20260720-210906 (fingerprint 12/12
  exact) vs variant 20260724-130543: **10W-3L-1T, mean +0.571%**;
  **Brier 8W-0L sweep** (the CatBoost high-card Brier regime). Losses:
  eucalyptus −2.06%, okcupid −1.16% (both multiclass, small), Moneyball
  −0.62%. Pooled gr+hc: 58W-14L-1T, ≈+1.73% dataset-weighted.
- **OpenML one-shot gate PASS**: certified base 20260720-212313
  (fingerprint 30/30 exact) vs variant 20260724-130721:
  **26W-8L-2T, mean +1.270%** (worst loss diabetes −1.74%, 442-row toy).
- **Headline (spliced 20260723-192007 field, control = fingerprinted
  base): single win rate 55.7 → 73.7% (5-arm incl Ens8); external field
  only 73.7 → 93.0%; CatBoost 50.9 → 43.4. Slowdown approx 4.9× →
  ~9.7× (fit ×1.98) vs CatBoost 12.9× — CatBoost dominated on both
  axes.** Canonical 5-arm chart re-run deferred to the default-flip
  decision.

## Verdict

All pre-registered gates PASS. Feature ships default-OFF in the PR;
the DEFAULT FLIP (accuracy vs ×1.98 single-model fit) is Nathan's
sign-off per the ship rules (precedent: cross_features 7.9× accepted
"as long as we are Pareto and all python"). Ens8/bag members have no
data tax — bagged points unchanged.

**DECISION (Nathan, 2026-07-24): ship as an opt-in accuracy option
(like `n_ensembles`), NOT the default.** Default stays `False`;
documented in docs/parameters.md + recipes.md. Consequently no pareto
or README refresh (the chart measures defaults, which are unchanged)
and no TabArena re-read needed. The spliced 73.7%/93.0% single-model
headline stays a plan-file fact, not a chart claim.
