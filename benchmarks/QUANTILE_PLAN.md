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

**Superseded 2026-07-31 — see `LEAFTUNE_PLAN.md` P12 and P14.** The mechanism
described in this section shipped in 0.27.0 and has since been removed; the
history is kept because two of its dead ends are still live traps.

Today: every delivered row is sorted on the way out of the booster, so
`diff(pred, axis=1) >= 0` holds exactly, at every `staged_predict` stage.
Measured crossing rate **0.0000** everywhere, against 0.18-0.21 for K
independent LightGBM boosters. Rearrangement is free in accuracy terms
(Chernozhukov, Fernández-Val & Galichon 2010), and being per-row it is exact
where any training-time construction has to be a bound over all rows at once.

**Trap 1, still live.** "Commit only non-decreasing increments" is sound but far
too strong: a sum of non-decreasing vectors is non-decreasing, so the predicted
interval could never be NARROWER than the pooled one. Forcing it by sorting the
leaf vector is worse still -- sorted values re-enter the accumulated scores and
self-reinforce. Measured: pinball 2.9e7 against an oracle of 0.35, i.e.
divergence. **This is why sorting is safe only at delivery**, where nothing
feeds back into the fit.

**Trap 2, the one that shipped.** The fix for trap 1 was a **narrowing budget**:
`gap[k]` a lower bound on `Q_k(x) - Q_{k-1}(x)` valid for any input row, spent
down each round by the realized minimum over all leaves, with the admissible set
`{v : v_k - v_{k-1} >= -b_k}` projected by PAVA in shifted coordinates. Sound as
a bound and ruinous in practice: charging at the worst-case leaf let one leaf
spend on behalf of every row, so it saturated in tens of rounds and froze the
interval width, leaving bands 2x to 10x too wide. Passing this section's own
acceptance ledger while doing so is the cautionary part -- the ledger compared
against per-level LightGBM and against nominal coverage, and never against a
trivial fixed-width baseline, which would have caught it immediately.

## Conformalization

`conformalize=True` carves a calibration fold **before** the early-stopping
split, so it sees no training, no stopping decision and no model selection.

The correction is a per-level **scale about the predicted median**, not the
usual additive widening. The scores it is applied to are already rearranged, so
every deviation from the median is sign-correct and a non-negative factor cannot
reorder anything; an additive correction that shrinks an interval is a
non-monotone offset vector and has no such property. Scaling also moves in both
directions, which matters more now than it did: the head used to be
systematically over-dispersed (raw coverage 0.844 at nominal 0.70) and the
factors shrank, whereas since the budget was removed the raw grid runs slightly
narrow and they widen.

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

Re-measured 2026-07-31 after the budget was removed (same protocol, 3 seeds):

| criterion | measured | verdict |
|:--|:--|:--|
| pinball vs K LightGBM boosters | ratio 0.989 (5 features) to 0.969 (128) — now a **win** at every width | **PASS** |
| fit wall clock | 3.0x (5 features) rising to 6.2x (128) | **MISS**, unchanged in kind |
| crossing rate | 0.0000 at every width | **PASS** |
| CQR coverage at n=10k | worst 2.65 points, erring wide | **FAIL** |

The crossing row's original wording ("no predict-time patch") no longer
describes the implementation and is kept only as the historical target. What the
guarantee is worth to a user is unchanged: zero crossings, at every stage.

The CQR row is a real regression against its 2-point target and is recorded as
such. Cause and follow-up in `LEAFTUNE_PLAN.md` P14 — the raw grid now runs
narrow, so the factors widen, and the outer-at-least-inner monotonization pushes
the middle intervals furthest.

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
