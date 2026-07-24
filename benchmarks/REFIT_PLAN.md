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

(appended as runs complete)
