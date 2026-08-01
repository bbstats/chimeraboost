# SMALLDATA — closing the single-model small-data gap vs CatBoost (opened 2026-08-01)

Successor to `BREAKTHROUGH_PLAN.md`, which closed with all three of its
candidates resolved and one thread explicitly left open:

> The single-model small-data collapse against CatBoost (81% → 50%/33%/0%) is
> untouched and remains the largest known headroom.

`refit_members` addressed the **bagged** half. This program is the single-model
half.

## What the predecessor already ruled out — do not re-run these

| lever | outcome |
|---|---|
| calibration (temperature, Platt, beta, isotonic) | KILLED — we are the best-calibrated model in the field; the deficit is resolution |
| CatBoost's `random_strength` / `bagging_temperature` | KILLED by opponent ablation — perfect port worth ≈1 win-rate point |
| depth race (6 vs 4) | priced dead — capacity-racing family oracle is +0.12–0.29% before an ~80% validation haircut |
| regressor `min_child_weight` floor | KILLED — direction real (p=0.002), magnitude inside noise |
| linear-leaf per-leaf guard | REFUTED — linear leaves are load-bearing, including on small data |
| semi-oblivious last level | deprioritized — the oblivious tax is worth ~1.4% on 2 of 57 datasets |

**Three capacity levers all came back too small. Capacity is not the
mechanism.** This program is required to start from a different axis.

---

## FINDING 5 — CatBoost's ONLY size-dependent default is the learning rate, and ours is size-blind

A win rate that collapses as rows shrink is the signature of a mechanism that
*switches on* at small n. So rather than guess which of CatBoost's mechanisms
that is, ask it: fit at a range of row counts and diff `get_all_params()`.

Of CatBoost's 43 resolved parameters, **exactly one varies with dataset size**,
and it is the learning rate (`scratchpad/catboost_size_defaults.py`, harness
budget `n_estimators=2000`, `early_stopping_rounds=50`):

| train rows | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 20,000 | 60,000 |
|---|---|---|---|---|---|---|---|---|
| CatBoost regression | 0.0259 | 0.0299 | 0.0334 | 0.0372 | 0.0430 | 0.0479 | 0.0534 | 0.0635 |
| CatBoost binary | 0.0158 | 0.0198 | 0.0234 | 0.0278 | 0.0349 | 0.0414 | 0.0491 | 0.0644 |
| **ChimeraBoost (any size)** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** | **0.100** |

Two things fall out, and the second is the one that matters:

1. **`boosting_type` does not vary — and its value is `Plain` at every size.**
   See the free kill below.
2. **CatBoost's rate is a clean power law in n** — regression
   `0.0259·(n/200)^0.157`, binary `0.0158·(n/200)^0.247`, both reproducing all
   eight measured points to under 1%. It is below our flat 0.1 at *every* size
   in our suites, but the mismatch widens from ~1.6x at 60k rows to **4x
   (regression) and 6.4x (binary) at 200 rows** — which is the shape of the
   collapse curve.

### Free kill: CatBoost is NOT running ordered boosting. It never was.

Asking for the *values* rather than the variation settles a hypothesis this
project has repeated for months. At both 500 and 20,000 rows, under the harness
budget, CatBoost resolves:

```
boosting_type            Plain          <-- NOT Ordered, at any size
bootstrap_type           MVS            <-- not the Bayesian bootstrap
leaf_estimation_method   Newton
grow_policy              SymmetricTree
score_function           Cosine
l2_leaf_reg              3
```

Finding 3 explained the small-data collapse as CatBoost being strong in "the
regime its ordered boosting was designed for". **Ordered boosting is switched
off.** CatBoost only defaults to `Ordered` in configurations we do not run, so
the mechanism cannot be the explanation for anything we have measured, and no
port of it can be justified by our benchmark.

⇒ **KILLED for free**: ordered boosting, and any small-data story that rests on
prediction-shift correction in the opponent. This also retires our own dormant
`ordered_boosting=False` default as a small-data candidate: the opponent we are
chasing does not use it.

A second correction falls out: the predecessor's Finding 2 described
`bagging_temperature=1` as CatBoost's "Bayesian bootstrap per-tree row
weights". The resolved default is **MVS** (Minimal Variance Sampling), so
`bagging_temperature` was inert and that ablation actually tested MVS on/off.
The conclusion is unaffected — the arm was killed either way — but the
mechanism named in the record was wrong.

`_auto_learning_rate` returns a flat 0.1 whenever early stopping is on, and its
docstring justifies it as converging "in ~half the trees of a smaller rate with
no measured accuracy cost". That was measured at full size. **Nobody has tested
it at a quarter size.**

### Why this is a different axis, not another capacity knob

Depth, `min_child_weight` and the leaf guard all restrict what a single tree can
express. The learning rate does not restrict the model class at all — early
stopping is free to buy back any lost capacity by growing more rounds. What it
changes is the **step size along the boosting path, and therefore how finely
early stopping can choose where to stop**. On small data the validation curve is
noisy and shallow; at 0.1 each step is a coarse jump, so the best reachable
point on the path can sit well off the true optimum. That is an optimisation and
model-selection story, and it predicts things the capacity story does not.

### Why it clears the leverage arithmetic

The bar from the predecessor: judge candidates on how **broadly** they move the
metric, never on how many named losses they target — a broad +0.25% is worth
4–5 win-rate points, while sweeping every named CatBoost loss is capped at ~6.
A default learning rate is as broad as a change gets: one default, every
dataset, no per-dataset audition, and therefore none of the ~80% haircut that
sank the capacity-racing family.

### The cost, stated up front

Lower rate ⇒ more rounds ⇒ slower fit, and the north star is strength *vs
slowdown*. The probe must therefore report round and fit-time ratios, not just
strength. The reason this may still be a Pareto move: the rate would only drop
where rows are few, and small datasets are exactly where fits are cheapest in
absolute seconds. A size-adaptive rate concentrates its cost where cost is
least. If the fit-time ratio is bad even there, this dies like the others.

---

## C4 — probe design (pre-registered 2026-08-01, before any results)

`benchmarks/probe_learning_rate.py`. Pilot first: if a quarter-size effect is
not visible on the datasets most likely to show it, the thread closes cheap.

- **Datasets (12)**: the six binary sets where CatBoost beats us
  (bank-marketing, credit, heloc, california, default-of-credit-card-clients,
  Diabetes130US), two binary controls where we win comfortably (pol,
  MagicTelescope), and four regression sets (cpu_act — which has confessed a
  huge capacity preference twice — plus sulfur, houses, superconduct). Decision
  suites only: no `pub:`, no TabArena in any form.
- **Sizes**: train fraction ∈ {1.00, 0.50, 0.25} via the harness's own
  `_subsample_train` (random_state=0, **test set unchanged**) — the `@sus`
  semantics, read as a learning curve. Rows capped at 20,000 for the pilot,
  which is stated rather than hidden: it shrinks the full-size arm on the
  largest sets, so the pilot's frac=1.00 column is *not* the decide regime.
- **Our arms**: `lr ∈ {0.1 (shipped), 0.05, 0.03}` plus `cb`, the CatBoost
  power law above evaluated at each cell's own row count. Everything else is
  out-of-box default (`n_estimators=2000`, `early_stopping_rounds=50`,
  `random_state=0`), fitted with **no explicit eval_set** so the internal split
  and `refit_full` run exactly as a user gets them.
- **Opponent arms**: CatBoost at its default auto rate, and CatBoost **forced to
  our 0.1**. This is the "ablate the opponent" method that earned its keep
  twice: if denying CatBoost its rate schedule erases its small-data edge, the
  mechanism is confirmed from its side at the cost of one benchmark.
- **Primary arm, named before the run**: **`cb` at frac 0.25** — the
  size-adaptive shape that would actually ship. The fixed rates are supporting
  evidence, and every sign test carries a **Holm correction across the four
  arms**, because four arms times three sizes is twelve chances to find a
  majority in noise.
- **Reading**, matched to the house tools: seeds averaged on the metric before
  any ratio; wins/losses/ties on `compare_runs`' ±1e-9 dead band; near-solved
  excluded on the best arm in the cell; per-dataset rows printed before any
  aggregate; rounds and fit seconds printed as ratios.
- **Metric**: RMSE for regression, Brier for binary — the house primary.

### Pre-registered predictions

- **C4 right**: the `cb` arm beats shipped 0.1 at frac 0.25 on a Holm-corrected
  sign test; the advantage grows monotonically as rows shrink; and the CatBoost
  ablation shows its small-data edge shrinking when forced to 0.1. Ship shape: a
  size-adaptive `_auto_learning_rate`, then the standard tier-1 synth and
  tier-2 decide gates.
- **C4 wrong**: flat-to-negative at every size ⇒ the thread closes and the flat
  0.1 is vindicated at small size as well as large. **One caveat pre-registered
  against that kill**: if the round counts do not move either, the arms did not
  bind and the honest finding is "these rates did nothing", not a refutation.
- **Pareto kill, stated before the numbers**: even a strength win dies if the
  fit-time ratio at the sizes where it wins is worse than the strength gain buys
  on the frontier. Strength alone is not the bar; the chart is.

### Verdict: mechanism CONFIRMED from the opponent's side. Ship shape KILLED on cost.

`benchmarks/results/probe-learning-rate.jsonl`, 12 datasets × 3 seeds × 3
sizes. Two things are true at once, and the program only cares about the
second.

**1. The pre-registered primary test FAILS outright.** `cb` at frac 0.25 is
8W-4L, median +0.314%, Holm-adjusted **p=0.388**. No pre-registered claim
passed, and that is the headline for gating purposes.

| arm | frac 1.00 (6k–15k rows) | frac 0.50 (3k–7.5k) | frac 0.25 (1.5k–3.8k) |
|---|---|---|---|
| lr=0.05 | +0.211% (8W-4L, p=0.775) | +0.158% (10W-2L, p=0.077) | **+0.255%** (10W-2L, p=0.077) |
| lr=0.03 | −0.039% (5W-7L, p=0.775) | +0.236% (10W-2L, p=0.077) | +0.209% (11W-1L, p=0.019) |
| `cb` | +0.094% (9W-3L, p=0.438) | +0.257% (11W-1L, p=0.019) | +0.314% (8W-4L, **p=0.388**) ← primary |

**The direction is nonetheless real and broad.** Every small-data cell is
positive, four of six run 10W-2L or 11W-1L, and the full-size column is a wash
— the sign flips at roughly 5,000–8,000 training rows. Two cells clear a
Holm-corrected 0.05. Read honestly, though: Holm was applied **across arms
within a fraction, not across all nine cells**, so a p=0.019 would land near
0.17 under correction across the grid. The population direction is
trustworthy; no individual cell is.

The within-dataset mechanism read is weak on its own — each dataset's own best
fixed rate falls as its rows shrink on 5 datasets, rises on 1, unchanged on 6
(p=0.219). Unlike C3's min-leaf direction (p=0.002), this does not
independently confirm.

**2. Ablating the opponent confirms the mechanism, and it is large.** At
quarter size, CatBoost's mean edge over our shipped arm is +1.033%; forced to
our 0.1 it drops to +0.442%. **Its learning-rate schedule is worth 57% of its
small-data edge** — and it is the only size-dependent default it has. At full
size the same ablation explains 7%, which is the size dependence showing up
exactly where predicted. (The panel is deliberately enriched with the six sets
where CatBoost beats us, so the "leads N/12" counts are not the suite standing;
the share-explained ratio is a within-panel mechanism reading, and is undefined
in the frac 0.50 row where we lead on average.)

So the question "why does CatBoost win on small data?" now has a measured
answer, and it is not ordered boosting, not capacity, and not calibration: **it
is running a learning rate 4–6x lower than ours, and that is over half its
edge.**

**3. And that is exactly why it does not ship.** The pre-registered Pareto kill
fires on two counts.

| | strength won | fit cost | rounds |
|---|---|---|---|
| `refit_members` (shipped) | gr:sus25 **+1.206%**, 12W-0L | **+17%** | — |
| this, best arm (lr=0.05 small-data) | **+0.255%** median, 10W-2L | **+48%** | **2.2x** |
| this, primary arm (`cb`) | +0.314% median, 8W-4L | +85% | 3.1x |

Roughly a quarter of the strength for three to five times the cost — an order
of magnitude worse trade than the thing we shipped yesterday. Worse, the cost
is **2.1–3.2x more trees**, which is not only fit time: it slows *prediction*
by the same factor, and predict speed is a column we lead.

And the north-star chart cannot show a gain even in principle. The strength is
small-data-only; at full size the effect is a wash (+0.094%, p=0.438) that
would cost 1.63x. Applying the schedule only where it wins leaves the base
stratum — which is what the headline Pareto runs — byte-identical, so the
frontier does not move on either axis. A change that cannot move the chart at
full size and costs 1.5–2x where it does win is not a frontier move.

The pre-registered escape hatch does not apply: round counts moved 2.1–3.2x, so
the arms bound hard. This is a real measurement, not a null from arms that were
too small.

⇒ **KILLED without shipping**: the learning rate as a small-data lever, in both
fixed and size-adaptive (CatBoost-schedule) forms. `_auto_learning_rate` keeps
its flat 0.1. Probe: `benchmarks/probe_learning_rate.py` (resumable,
`--table-only` reprints).

### REVISED (same day): the knee is at 0.07, and the kill above was wrong

Nathan's steer — *prediction time matters a little less to me than train time* —
re-weighted the cost axis, and re-pricing on that axis exposed an arithmetic
error in the kill above that has nothing to do with his preference.

**The error: I compared this candidate's MEDIAN against `refit_members`' MEAN.**
The kill table said "a quarter of the strength for three to five times the
cost". On like-for-like medians that is simply false:

| | strength (median) | fit cost |
|---|---|---|
| `refit_members`, gr:sus25 (shipped) | **+0.304%** | **+17%** |
| lr=0.07 at quarter size | **+0.275%** | **+21%** |

Very nearly the same trade — on the single-model path, which is the half that
`refit_members` could not reach.

**The knee run.** The pilot sampled 0.1 / 0.05 / 0.03 and showed strength
saturating by 0.05 while cost kept climbing, so the best point was somewhere it
never looked. `--knee` fills in 0.07, drops the arms not needed to locate it,
and **pins `thread_count`** — the pilot left it at the class default, which is
why its fit seconds carried a "never compare to a harness slowdown" caveat. With
fit time now the deciding axis, the least trustworthy number in the table was
the one being decided on.

| frac | lr=0.07 strength | lr=0.07 cost | lr=0.05 strength | lr=0.05 cost |
|---|---|---|---|---|
| 1.00 | +0.118% (8W-4L) | 1.53x rounds / **1.28x fit** | +0.211% (8W-4L) | 2.11x / 1.62x |
| 0.50 | +0.254% (9W-3L) | 1.32x / **1.20x fit** | +0.158% (10W-2L) | 2.07x / 1.53x |
| 0.25 | **+0.275% (10W-2L)** | 1.48x / **1.21x fit** | +0.255% (10W-2L) | 2.19x / 1.46x |

**0.07 dominates 0.05**: equal or better strength at small size for less than
half the extra fit. The knee is real and it is sharp.

**And by the house bar this passes at every size.** `compare_runs` gates on a
simple majority (`wins >= n//2 + 1`), which is 7 of 12 here — the same bar that
recorded depth-4's 9W-3L as a PASS. The probe was reporting exact binomial
p-values with a Holm correction, a far stricter standard than the project ships
on. Stated both ways: 10W-2L is p=0.039 uncorrected, 0.077 after correcting for
the two arms, and a clear PASS on the house bar.

**Honest status of this evidence.** The 0.07 arm was chosen *after* seeing the
pilot, so the knee run is exploratory, not a pre-registration — it cannot be
read as a confirmatory test. What confirms it is the standard house gate on
data that did not select it: tier-1 synth screen, then tier-2 `--decide` with
per-stratum sign tests.

⇒ **The kill above is WITHDRAWN.** The mechanism was confirmed all along; what
was wrong was the price. Corrected shape: **a size-gated rate that fades from
0.1 toward 0.07 as training rows fall**, applied only where it wins, so the
base stratum and the headline chart stay byte-identical. Full size is left at
0.1 deliberately: +0.118% there is a wash and would cost 1.28x for nothing.

### Ship shape — `adaptive_learning_rate`, opt-in

Implemented following the `refit_full` / `refit_members` precedent: opt-in and
byte-identical when off, so the gate can measure it without moving a shipped
default. `adaptive_learning_rate=True` fades the auto rate from
`_AUTO_LR_SMALL=0.07` at `_AUTO_LR_LO=5,000` training rows up to the unchanged
`_AUTO_LR_LARGE=0.1` at `_AUTO_LR_HI=15,000`.

Three implementation notes worth keeping:

- **Thresholds are POST-split rows** — what the booster actually trains on
  after the estimator's early-stopping split takes `validation_fraction`. The
  probe reported PRE-split counts, so its measured-positive buckets (up to
  ~7,500 pre-split) land at ~6,000 on this axis. Two scales, documented rather
  than left to be confused later.
- **The fade cannot drift into the refit.** `_refit_on_full` already pins
  `rkw["learning_rate"] = float(winner.lr_)`, so the rate that chose the
  early-stopped round budget is the rate the refit replays at — even though the
  refit sees more rows and a naive re-resolve would fade differently. Locked by
  a test.
- **Unknown row count falls back to 0.1**, never to the small rate. Also
  locked.
- **Full size deliberately stays at 0.1.** The +0.118% there is a wash that
  would cost 1.28x.

Harness arm `ChimeraBoostALR` runs both arms inside ONE benchmark (the Sel25 /
`refit_members` precedent), so the A/B pairing carries no machine-condition
drift. 920 tests pass, 1 skipped, numerical-identity goldens included; 14 new
tests in `tests/test_adaptive_learning_rate.py` lock the contract.

### Tier 1 (synth screen) — PASS on both judges

`results/20260801-134051.json`, both arms in one benchmark, 136 datasets,
3 seeds.

| judge | result |
|---|---|
| **primary** | **71W-57L-8T**, bar 69+ → **PASS**, mean +0.128%, median +0.025% |
| **Brier** | **48W-38L-2T**, bar 45+ → **PASS**, median +0.301% |
| head-to-head | ALR wins **60.4%** (81W-53L), median gap 0.35% |
| fit cost | **1.23x**, matching the probe's 1.20–1.28x |

**The Brier MEAN reads −1.171% and that number is an artefact, not a
regression.** It is one dataset: `syn:v2/117` moves Brier 0.0018 → 0.0036, an
absolute delta of 0.0018 that reads as −99.85% on a near-zero denominator. Its
Brier sits just above the 0.001 near-solved cutoff, so the house filter does not
catch it. That single set contributes −1.161% of the −1.171%; excluding it the
Brier mean is **−0.010%**, flat. This is exactly the trap
[[project-compare-runs-near-solved-fix]] records (near-solved sets turned an
88-set mean into −144%), and the standing rule is to read the sign test, which
passes.

Tier 2 (`--decide`) is the confirmatory test on data that did not select 0.07,
run with CatBoost in the same benchmark so the program's actual target — the
small-data win rate against CatBoost from Finding 3 — is read directly rather
than inferred.

### Tier 2 (decide) — the target moved, and one stratum regressed

`results/20260801-134418.json`, 103 datasets, 3 seeds, all three arms in ONE
benchmark. Per stratum, never pooled. The default arm reproduces Finding 3's
standing exactly (81/31/50/33/0/0/43 vs CatBoost), which validates the run.

| stratum | ALR vs default | mean | median | sign bar | fit | **vs CatBoost: default → ALR** |
|---|---|---|---|---|---|---|
| gr:base | 22W-14L-**23T** | +0.217% | +0.000% | FAIL (ties) | 1.09x | 81% → **82%** |
| hc:base | **6W-0L**-8T | +0.813% | +0.000% | FAIL (ties) | 1.14x | 31% → 31% |
| **gr:sus25** | **9W-3L-0T** | +0.247% | +0.309% | **PASS** | 1.22x | **50% → 67%** |
| gr:sus50 | 3W-1L-2T | +0.197% | +0.131% | FAIL (ties) | 1.12x | **33% → 67%** |
| **hc:sus25** | **3W-0L-0T** | +0.677% | +0.144% | **PASS** | 1.31x | 0% → 0% |
| hc:sus50 | 0W-0L-2T | +0.000% | — | n=2, all ties | 1.20x | 0% → 0% |
| **hc:time** | **1W-3L-3T** | **−1.215%** | +0.000% | **FAIL — a real loss** | 1.12x | 43% → 43% |

**The program's target moved.** Finding 3's two Grinsztajn small-data strata go
from 50% to **67%** and from 33% to **67%** against CatBoost, on the
single-model path that `refit_members` could not reach. That is the collapse
partially closed, and it is the first time anything in this program has moved
that number.

**The large ties counts are the design working, not a weak result.** gr:base
carries 23 ties because those datasets sit above the fade's upper threshold,
where ALR is byte-identical to the default by construction. Among the 36
datasets where the fade actually engages it is 22W-14L. hc:base is **6W-0L with
zero losses** among decided datasets; it misses the bar only because the bar
counts ties against the change (the same reading `refit_members` recorded).

**Brier is neutral-to-positive except where primary is**: gr:base 6W-7L-10T
(−0.046%, flat), hc:base 2W-1L-5T (+0.492%), gr:sus25 3W-2L (+0.377%, PASS),
hc:sus25 1W-0L (+0.360%, PASS), hc:time 0W-1L-3T (−1.006%).

**Cost lands where it should**: 1.09x on gr:base, rising to 1.22–1.31x on the
small-data strata — i.e. the bill is largest exactly where fits are cheapest in
absolute seconds, which was the argument for a size-gated rate in the first
place.

#### The regression: hc:time, and it has a mechanism

Four of the seven temporal datasets engage the fade (the other three sit above
the threshold and tie). Of those four, ALR loses three:

| dataset | change |
|---|---|
| hc:eucalyptus@time | **−4.59%** |
| hc:employee_salaries@time | −2.31% |
| hc:Moneyball@time | −1.77% |
| hc:house_prices_nominal@time | +0.17% |

And the gap to CatBoost on this stratum widens from a +1.94% median to
**+3.74%**.

**This is coherent, not noise.** A lower rate buys its gain by fitting more
rounds; under distribution shift those extra rounds fit the *past* more tightly,
which is precisely the wrong thing when the test window has moved. The same
mechanism that helps on a random split hurts on a temporal one. It also lands on
`hc:eucalyptus@time`, already recorded in Finding 3 as our worst matchup
anywhere and as a "confidently wrong" failure under drift — this makes it worse.

Honest weighting: n=4 engaged datasets, and Finding 3 itself cautioned that
eucalyptus has 736 total rows so each temporal window carries real variance —
"a pointer, not a sign test". It is weak evidence. But it is the *only* negative
stratum, its sign is consistent across three of four datasets, and it has a
mechanism that predicts it in advance. That combination should not be waved
away.

### VERDICT (superseded by C5 below — kept for the record)

- **Ship `adaptive_learning_rate=True` as opt-in.** It is byte-identical when
  off (920 tests, numerical-identity goldens included), it passes tier-1 on both
  judges, and it passes tier-2 on the two strata it was built for while moving
  the program's target metric by 17 and 34 points.
- **The default flip is Nathan's call and I do not recommend it today.**
  (RESOLVED 2026-08-01: withdrawn by C5 below, and the flip shipped in PR #76.)
  Not
  because the small-data evidence is weak — it is the strongest result this
  program has produced — but because a default applies to every user including
  those with drifting data, and hc:time says this hurts exactly there. An
  opt-in carries no such obligation.
- **Owned follow-up, not left implied**: the temporal regression is worth one
  probe. If the drift loss is really "extra rounds overfit a stale past", then
  gating the fade on the presence of a temporal split — or simply documenting
  "do not enable this when your data drifts" — resolves it. That question is
  now the live one in this program.

## C5 — is the temporal regression real, and is it fixable? (pre-registered 2026-08-01)

The tier-2 verdict withheld a default flip on the strength of **four datasets**.
Before accepting that, the regression itself gets a probe.

**First, a constraint that kills the obvious response.** "Run more seeds" does
nothing here: `_temporal_split` takes `TEMPORAL_CUTS[seed % 3]` with
`TEMPORAL_CUTS = (0.65, 0.70, 0.75)` and no other seed-dependent randomness, so
seed 3 reproduces seed 0 **exactly**. The hc:time universe is 7 datasets × 3
cuts, full stop, and no `pub:` temporal columns are registered to widen it.
More seeds would have produced identical numbers and the appearance of more
evidence — worth recording, because that is a trap this suite sets.

So the probe attacks the mechanism instead of the sample size.

**The hypothesis to test**: a lower rate buys its gain with more rounds; under
drift those extra rounds fit a stale past more tightly. Crucially **our
early-stopping holdout is a RANDOM slice of the training rows**, so it is drawn
from the past too — early stopping watches it improve and keeps going, blind to
the fact that fitting the past harder has stopped helping. If that is right,
validating on the *most recent* slice instead should see the drift and stop
earlier, removing the regression.

`benchmarks/probe_temporal_lr.py`, 2×3 design:

- **Datasets**: all 7 `hc:*@time`, rows capped at 30,000 (stated: this brings
  the three largest — kick, sf-police-incidents, Traffic_violations — into the
  size range where the rate matters at all; in tier-2 they sat above the fade
  threshold and tied).
- **Cuts**: 6 rolling origins (0.45, 0.55, 0.65, 0.70, 0.75, 0.82) instead of
  the suite's 3, so the "does it reproduce" question gets twice the windows
  from the same sources. Test window is `TEMPORAL_TEST_FRAC` after the cut,
  with the suite's unseen-class filter.
- **Arms (6)**: rate ∈ {0.1 shipped, 0.07 the knee} × early-stopping holdout ∈
  {`auto` (the shipped path: internal random split, `refit_full` on), `rand`
  (explicit random 20%, refit off), `tail` (explicit *last* 20%
  chronologically, refit off)}. Fixed rates rather than the fade, because the
  mechanism question is about the rate; the fade is only a schedule over it.
  The `rand` arm exists so that `tail` vs `rand` isolates the holdout's
  *composition* — both have the refit off, so the refit cannot confound it.
- **Recorded**: test score, rounds, fit seconds per cell, so the
  "loss tracks the round increase" claim is checkable rather than asserted.

### Pre-registered predictions

- **Mechanism right**: with `rand`/`auto` holdouts, 0.07 loses to 0.1 across the
  6 cuts; with the `tail` holdout that loss shrinks or reverses; and 0.07 runs
  fewer rounds under `tail` than under `rand`. ⇒ a temporal-aware ES split is a
  real fix, and the default flip is back on the table.
- **Regression was noise**: `auto@0.07` vs `auto@0.1` is a wash over 42 windows.
  ⇒ the tier-2 hc:time result was three unlucky cuts, and the default flip is
  back on the table for a different reason.
- **Mechanism right but unfixable**: 0.07 loses under every holdout type. ⇒ the
  caveat is real and structural; ship stays opt-in and gets documented as
  "not for drifting data".

All three outcomes are decisive for the shipping question, which is why this is
worth one run.

### Verdict: the regression was NOISE. My mechanism story was WRONG. Blocker removed.

`results/probe-temporal-lr.jsonl`, 42 windows (7 datasets × 6 rolling origins).

**1. The temporal regression does not reproduce.** On the shipped path — the
same configuration tier-2 measured — lr=0.07 vs lr=0.1 is:

| holdout | median | mean | W-L | p |
|---|---|---|---|---|
| **`auto` (shipped)** | **−0.004%** | +0.093% | **21W-21L** | **1.000** |
| `rand` (explicit random, refit off) | −0.103% | +0.272% | 19W-23L | 0.644 |
| `tail` (explicit most-recent, refit off) | +0.137% | +0.316% | 26W-16L | 0.164 |

Twenty-one wins and twenty-one losses. A perfect coin flip. **Tier-2's
hc:time result — 1W-3L, mean −1.215% — was three unlucky cuts out of the three
the suite happens to run.** Per dataset the `auto` medians are −0.44, −0.10,
+0.20, −2.17, +1.28, −0.18, −0.00: three positive, four negative, centred on
nothing. `hc:eucalyptus@time`, the dataset that drove the tier-2 loss, reads
−2.17% under `auto` but **+3.32%** under `rand` and **+1.74%** under `tail` —
unstable in sign across holdouts, which is what 736 rows buys you.

**2. The mechanism I proposed is not supported.** I claimed the loss came from
extra rounds fitting a stale past, with early stopping blind to it because its
holdout is drawn from the past. Both halves fail their own test:

- The correlation between the round increase and the loss is **−0.12 to −0.23**
  across holdouts — the predicted sign, but far too weak to carry the story.
- The tail holdout was supposed to *see* the drift and stop earlier. It stops
  **later**: rounds(0.07)/rounds(0.1) is **1.46x** under `tail` versus 1.33x
  under `rand`. The opposite of the prediction.

I stated that mechanism concisely and confidently when asked. It was a
plausible story fitted to four data points, and it did not survive its first
real test.

**3. A bonus negative, and a useful one.** Comparing the holdout types in
absolute terms at a fixed rate of 0.1:

| comparison | median | W-L of 42 |
|---|---|---|
| `rand` vs `auto` | −1.492% | 11W-31L |
| `tail` vs `auto` | −3.019% | 10W-32L |
| **`tail` vs `rand`** | **−0.812%** | **13W-29L** |

**A temporal-aware early-stopping split is worse than a random one under
drift**, on 29 of 42 windows. That retires the "tail ES split" idea that
[[project-breakthrough-hunt-2026-08-01]] recorded as the unmined temporal lane
("`@time` training rows arrive in ascending timestamp order, so a tail
early-stopping split is feasible"). The likely reason is simple and was not
obvious to me beforehand: validating on the most recent rows means **training
on none of them**, and those are exactly the rows closest to the future test
window. The shipped `auto` path beats both explicit arms anyway, since it keeps
`refit_full`.

**Scope note kept honest**: C5 sweeps FIXED rates (0.1 vs 0.07) where tier-2
ran the fade, and caps rows at 30,000 — which pulls kick, sf-police-incidents
and Traffic_violations into the range where the rate bites, whereas in tier-2
they sat above the fade threshold and tied. So C5 tests the rate that the fade
delivers, on more of the stratum than tier-2 could reach. That is the right
test for "does a lower rate hurt under drift", and the answer is no.

⇒ **The only stratum that argued against a default flip does not survive
contact with six cuts instead of three.** The recommendation to withhold the
flip was built on noise, and I withdraw it.

### FINAL RECOMMENDATION: flip the default

With hc:time resolved as a coin flip, the evidence for
`adaptive_learning_rate=True` as the default is:

- **Tier-1 synth**: PASS on both judges (primary 71W-57L, Brier 48W-38L).
- **Tier-2 decide**: positive mean in **6 of 7 strata**, sign-test PASS on
  gr:sus25 (9W-3L) and hc:sus25 (3W-0L), **zero losses** on hc:base (6W-0L),
  22W-14L on gr:base among the datasets where the fade engages.
- **The seventh stratum is now measured as noise** on six cuts instead of three.
- **The program's target moved**: 50% → 67% and 33% → 67% against CatBoost on
  the two strata Finding 3 named — the first movement on that number.
- **Cost**: 1.09x on gr:base rising to 1.31x on the smallest strata, and
  **exactly 1.00x above 15,000 training rows**, where the fade is a no-op and
  the model is byte-identical.

The honest counterweight, stated rather than buried: the gains are real but
individually small (medians +0.13% to +0.31%), the strongest strata carry
n=3–12 datasets, and users on large data pay nothing and gain nothing. This is
a change that helps a specific regime meaningfully and leaves the rest exactly
where it was.

Per project precedent (`refit_full`, `refit_members`) the flip itself is
Nathan's call. **I now recommend it.** The remaining implementation step is a
one-line default change plus a golden refresh, deliberately NOT in this PR —
flipping a default rewrites the numerical-identity goldens, and that deserves
its own reviewable change rather than being buried in an investigation.

### SHIPPED 2026-08-01 — default flipped to on (PR #76)

Nathan took the recommendation. `adaptive_learning_rate=True` is the default in
`ChimeraBoostRegressor`, `ChimeraBoostClassifier` and the underlying
`GradientBoosting`. Three notes on what the flip actually touched, since two of
them were not what the recommendation predicted:

- **The committed goldens did not move.** `tests/golden_metrics.json` measures
  with `early_stopping: false`, and the fade lives entirely on the
  early-stopping branch, so the golden panel is a no-op under the flip. The
  "golden refresh" the recommendation anticipated turned out to be unnecessary.
- **`ChimeraBoostQuantileRegressor` is pinned to the flat rate**, explicitly,
  in `quantile_api.py`. It builds a `MultiQuantileBoosting` with early stopping
  on and would otherwise have inherited the new booster default — shipping an
  unmeasured default change, since every number in this program is squared
  error or Brier and none is pinball loss. Measuring it is an open question,
  not a pending task: nothing here says the fade would help or hurt there.
- **The harness arm reversed direction.** `ChimeraBoostALR` (fade on, vs a
  default with it off) is now `ChimeraBoostFlatLR` (fade off, vs a default with
  it on), so the A/B control is still expressible. `_run_chimera` takes
  `adaptive_lr="off"` on the same convention `refit_full` already used.

Measured scope of the flip, from `scratchpad/check_flip_scope.py` — the rate the
single-model path resolves to, against the booster's own row count:

| requested rows | booster rows | default (0.30.0) | `adaptive_learning_rate=False` |
|---|---|---|---|
| 1,200 | 960 | 0.0700 | 0.1000 |
| 6,250 | 5,000 | 0.0700 | 0.1000 |
| 12,500 | 10,000 | 0.0850 | 0.1000 |
| 25,000 | 20,000 | **0.1000** | 0.1000 |

Three paths measured as byte-identical either way, and the first was not
predicted: a 3-member bag returns exactly the same predictions with the flag on
or off, because members carry an explicit member learning rate (0.15) and so
never consult the auto rule at all. The bagged rungs of the quality ladder are
therefore untouched by this flip. `early_stopping=False` and the quantile
regressor are the other two.

This closes SMALLDATA. The open thread it leaves is recorded below and is a
question about our own pipeline, not a pending item in this file.

### What this program established

- **Ordered boosting is dead for free** — CatBoost never runs it here. That
  retires a hypothesis family the project had carried for months, and our own
  dormant `ordered_boosting` flag with it.
- **CatBoost's small-data edge is now measured, not guessed**: 57% of it is a
  learning rate 4–6x below ours, the only size-dependent default it has.
- **We cannot buy it at a price worth paying.** We gain about a third of what
  CatBoost loses when denied the same schedule — our pipeline (`refit_full`
  replay, `selection_rounds`) already recovers most of the benefit for free,
  which is *why* the remaining gain is too small to justify 2–3x the trees.

**The open thread is now sharper than when this program opened.** The
single-model small-data gap is not capacity, not calibration, not ordered
boosting, and not affordable via the learning rate. What remains untested is
the one asymmetry this probe exposed rather than resolved: **we gain far less
from a lower rate than CatBoost loses from a higher one.** Whatever is
absorbing that difference on our side is the next place to look, and it is a
question about our own pipeline rather than about a mechanism to port.

---

## Rules for this program
- Inherited unchanged from the predecessor: every candidate gets a cheap
  decisive probe BEFORE library work; negative results are written down here in
  the same change that produces them; the ship gate is tier-1 synth then
  `--decide --seeds 3 --save` with per-stratum sign tests, never pooled.
- TabArena stays sealed and report-only.
