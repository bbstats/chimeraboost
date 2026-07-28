# Shared-tree multi-quantile head

One booster, an arbitrary tau grid, a K-vector in every leaf. Replaces "fit K
independent quantile boosters" for estimating a whole predictive
distribution.

Run: `python benchmarks/quantile_head.py` (alone). Writes
`benchmarks/results/quantile-head.md`.

## What the split search actually is

Pinball loss has hessian 1 in every channel, so for a fixed candidate split
the exact summed-across-tau gain collapses to a norm:

```
gain = ‖G_L‖²/(n_L+l2) + ‖G_R‖²/(n_R+l2) - ‖G_P‖²/(n_P+l2)
```

By Parseval this equals the sum of projected gains over **any** orthonormal
basis. So projecting the K gradient columns onto one direction is the rank-1
truncation of the exact gain, not a different algorithm, and the only question
is which direction to keep. `exact_splits=True` computes the norm form
directly, at K histogram channels per feature; it is the reference arm.

A second structural fact makes the whole thing cheap. Row i's gradient is
`e(r_i) - taus`, where `r_i` is the row's PIT rank (how many of its own K
estimates the target falls above) and `e(r)` is the step vector that turns on
at r. A row's entire gradient is therefore one integer, so projecting onto `c`
is just scoring that rank by `phi(r) = sum(c[r:])`. The fit never builds an
(n, K) gradient at all: one binary search per row into that row's own sorted
score vector, then a table lookup (`tree._project_pinball`).

It also says what the candidate directions ARE. Taking `c` polynomial in tau
of degree 0, 1, 2 makes phi degree 1, 2, 3 in the rank: location, spread,
skew.

## Decisions, each measured

**Which direction (`split_projection`).** Excess pinball over the oracle,
x1000, 3 seeds, four regimes (see the results file for the current run):

| arm | location | scale | mixed | extreme | TOTAL |
|:--|--:|--:|--:|--:|--:|
| rotate (default) | 13.161 | 21.082 | 30.541 | 55.735 | **120.520** |
| sum | 13.056 | 17.222 | 43.150 | 72.966 | 146.394 |
| gram | 12.715 | 17.573 | 43.264 | 72.996 | 146.548 |
| exact | 15.651 | 20.705 | 28.039 | 48.404 | 112.799 |

- **The literal channel sum is dead.** On a symmetric grid the tau and 1-tau
  pushes are equal and opposite, so a region that is centred correctly but too
  narrow sums to exactly zero. It is competitive only where nothing but the
  centre moves.
- **`rotate` lands within 7% of the exact gain** at one histogram per round
  instead of K.
- **Measuring the direction instead of cycling ("gram") did not pay.** Two
  failure modes, both instructive. Raw gradient energy is dominated by
  per-row Bernoulli noise, which is largest at the median, so the top
  eigenvector collapses onto the location contrast and inherits its blindness
  to spread. Whitening by the no-signal (Brownian-bridge) covariance fixes
  that but is ill-conditioned -- the null's eigenvalues fall like 1/k², so its
  inverse amplifies exactly the high-frequency directions carrying only
  sampling noise, and unrestricted whitening measured four times worse than
  cycling on location-driven data. Restricting the ratio to the Legendre
  subspace makes it well-posed and it still only ties. Kept as an option, not
  the default.

**Cycle weighting.** Two rounds of location per round of spread beat uniform
cycling at both K=3 and K=19, and every schedule that also spent rounds on
skew was worse. `_ROTATE_PATTERN = (0, 0, 1)`.

## Why predictions cannot cross

Init is the sorted global quantiles, every committed leaf vector is projected
onto an admissible set, IEEE addition is monotone, and the packed predictor
accumulates trees in identical order for every channel. So
`diff(pred, axis=1) >= 0` holds exactly. Measured crossing rate: **0.0000**
everywhere, against 0.18-0.21 for K independent LightGBM boosters.

**The obvious construction does not work.** "Commit only non-decreasing
increments" is sound but far too strong: a sum of non-decreasing vectors is
non-decreasing, so the predicted interval could never be NARROWER than the
pooled one, and the low-noise half of heteroscedastic data becomes
inexpressible. Forcing it by sorting is worse still -- it reverses a narrowing
update into a widening one, which is self-reinforcing. Measured: pinball
2.9e7 against an oracle of 0.35, i.e. divergence.

What works is a **narrowing budget**. `gap[k]` is a lower bound on
`Q_k(x) - Q_{k-1}(x)` valid for ANY input row, not just training rows: a row's
gap is the initial gap plus, per tree, the increment gap at whichever leaf it
lands in, and each of those is at least the minimum over that tree's leaves.
Each round may give away at most the currently guaranteed gap, and the
realized minimum is subtracted afterwards. The admissible set
`{v : v_k - v_{k-1} >= -b_k}` becomes plain monotonicity under
`u_k = v_k + sum(b_1..b_k)`, so the projection is isotonic regression
(pool-adjacent-violators, which averages violators -- a sort permutes them and
can flip a contrast's sign). A floor at 1e-9 of the initial spread keeps the
guaranteed margin far above float64 accumulation error.

## Conformalization

`conformalize=True` carves a calibration fold **before** the early-stopping
split, so it sees no training, no stopping decision and no model selection.

The correction is a per-level **scale about the predicted median**, not the
usual additive widening. An additive correction that shrinks an interval is a
non-monotone offset vector, and the only way to apply one safely is out of the
narrowing budget -- which the fit has already spent, so it projects away to
nothing (measured: offsets of exactly 0). Scaling has no such problem, and
shrinking is what is actually needed: a shrunk-fit quantile model is
systematically over-dispersed, because every round's step is scaled by the
learning rate so the grid never fully contracts. Measured raw coverage at
nominal 0.70 was 0.844.

Fails loudly when the fold cannot support the requested levels
(`ceil((n+1)(1-alpha)) <= n` needs `n >= (1-alpha)/alpha`).

## Acceptance ledger

Target from the build request, against K = 19 LightGBM 4.6.0 quantile
boosters sharing one Dataset, 300 rounds, depth 4, lr 0.1, n = 20 000,
3 seeds.

| criterion | target | measured | verdict |
|:--|:--|:--|:--|
| pinball vs K LightGBM boosters | match within noise | ratio 1.029 (5 features) to 0.996 (128) | **PASS** |
| fit wall clock | >= K/2 = 9.5x faster | 3.4x (5 features) rising to 7.8x (128) | **MISS** |
| crossing rate, no predict-time patch | exactly 0 | 0.0000 at every width; LightGBM 0.18-0.21 | **PASS** |
| CQR coverage at n=10k | within 2 points of nominal | worst 0.73 points | **PASS** |

**On the speed miss.** The saving is concentrated in the split search, which
runs once per round instead of K times, so the speedup grows with how wide the
data is: 3.4x at 5 features, 4.8x at 32, 6.0x at 64, 7.8x at 128, and still
climbing. It does not reach 9.5x in the tested range because the leaf refit is
irreducibly K quantile selections per leaf and is ~90% of a round at ordinary
widths, while a single LightGBM quantile round is very cheap. Three
optimizations already landed and are in the numbers above: the PIT-rank
projection (no (n, K) gradient, -35% fit), a K-aware serial/parallel dispatch
threshold (the leaf refit does K times the scalar path's per-row work, so the
fork/join break-even arrives at K times fewer rows, +65%), and parallelizing
the refit over (leaf, channel) pairs rather than leaves, since real oblivious
trees are badly unbalanced.

Tried and reverted: gathering leaf residuals channel-major in one pass. It
trades K forward strided reads for a transposing write and measured slower --
the per-channel scan is prefetcher-friendly and the leaf's slice of the score
matrix stays resident across the K passes.
