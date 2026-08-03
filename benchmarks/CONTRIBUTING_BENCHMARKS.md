# Contributing a benchmark result

Issue #71. This is the protocol the project runs on its own changes, written so
that someone outside the project can run it too. Evidence produced this way is
read exactly the way we read our own; evidence produced any other way has to be
re-run before it can decide anything, which usually means it decides nothing.

Everything here is one repository, two benchmark runs per arm and two
comparisons. If you only read one
section, read "Strata, and why pooling is forbidden".

---

## 1. Install

```
pip install -e ".[bench]"                # harness only
pip install -e ".[bench,competitors]"    # plus CatBoost, LightGBM, XGBoost
```

The library itself must be installed. `run_benchmarks.py` puts only
`benchmarks/` on `sys.path`, so `from chimeraboost import ...` resolves through
the installed package and not through the checkout you happen to be standing
in. An editable install of the repo is the simplest way to get that. If you
keep two checkouts to compare, set `PYTHONPATH` to the one you mean and confirm
it loaded — `benchmarks/print_chimera_path.py` prints the resolved path, and an
editable install silently winning over the checkout you intended has cost real
hours here.

Competitors are auto-detected: scikit-learn's `HistGradientBoosting` is always
present, and CatBoost, LightGBM and XGBoost are used if importable and skipped
in silence if not. XGBoost is off unless you pass `--with-xgboost`, because it
tracks LightGBM closely and roughly doubles competitor runtime.

**The first run is slow and downloads several gigabytes.** Every suite caches
under `benchmarks/data_cache/`, which is gitignored: Grinsztajn CSVs under
`grinsztajn/`, the high-cardinality and public datasets as parquet under
`openml_pq/`. Set `BENCH_DATA_HOME` to move all of it somewhere with more room.
Downloads are flaky against anonymous rate limits; relaunching resumes from the
cache. The first fit of a session also pays numba compilation, roughly a
quarter-minute that is not a property of the model; `chimeraboost-warmup`
compiles the kernels ahead of time so it does not land inside a timed run.

---

## 2. The tiers, and what each one is for

| tier | command | what it answers | can it ship a change? |
|---|---|---|---|
| screen | `--synth` | does the mechanism show up where the idea says it should? | no — it can kill, never ship |
| **decision** | `--decide` | does the change help on real data, per stratum? | **yes, and only this** |
| tuning | `--pmlb` | hyperparameter values | no |
| validation | `--public` | post-hoc sanity, cited in docs | no — never blocks |
| holdout | TabArena | the sealed out-of-sample read | out of scope, see below |

**The synthetic screen (`--synth`) is a filter, not a verdict.** It runs a
frozen suite of prior-sampled generated datasets, and `benchmarks/synth_report.py
BASE.json NEW.json` breaks the effect down by generator attribute. The question
it answers is whether the effect concentrates in the slice your mechanism
predicts. If it does not, stop — the idea is probably not doing what you think.
A clean screen is permission to spend the decision run, nothing more. The
generators have known biases and their speed ratios are not decision-grade.

**The decision run (`--decide`) is the only tier that ships or kills anything.**
It runs the Grinsztajn suite and the high-cardinality suite together, plus their
variant families, and reports every one of them as a separate stratum. Both
suites are needed: Grinsztajn's loaders pass no categorical features at all, so
a categorical lever cannot express itself there, and the high-cardinality suite
is where real entity-like categoricals and multiclass live.

**The public suite (`--public`) is validation, and it never blocks.** It is 22
audited datasets with no overlap against anything we tune on. Read it freely,
cite it, treat a surprise there as a reason to go looking. It cannot settle a
question; the decision run still answers it.

**PMLB (`--pmlb --pmlb-fold tune|holdout`) is for hyperparameter tuning only.**
It is the one suite the defaults were allowed to be fitted against, so a strength
claim on it is in-sample by construction. It is currently dormant.

**TabArena is a sealed holdout and is out of scope for outside submissions.**
The full run is executed by its authors on their own defaults with days of
turnaround, and its results — aggregate or per task — never influence a source
change. Do not submit TabArena numbers as evidence for a change; they will be
set aside on principle rather than on merit.

---

## 3. The command sequence

Run the unmodified code first, then your change, back to back on the same idle
machine with the same flags. `--save` writes a transcript under
`benchmarks/results/` plus a sidecar `.json` with every metric for every
dataset, model and seed. The `.json` is what the comparison tools read.

```
python -m pytest -q

python benchmarks/run_benchmarks.py --decide --list-datasets

python benchmarks/run_benchmarks.py --synth --seeds 3 --save
python benchmarks/synth_report.py BASE.json NEW.json

python benchmarks/run_benchmarks.py --decide --seeds 3 --save

python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost --by-suite
python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost --by-suite --metric brier
```

Each `--save` run writes `benchmarks/results/<timestamp>.txt` and a sidecar
`<timestamp>.json`. Run every benchmark command twice — once on unmodified code
and once with your change — and `BASE.json` is the baseline arm's sidecar while
`NEW.json` is yours. The sidecar also records a provenance block (library
version, git SHA and whether the tree was dirty, platform, CPU count, every
competitor's version, and the exact command line), which is what lets us tell
whether two runs are comparable at all.

The test suite comes first because a change that breaks the numerical-identity
goldens is not a benchmarking question. `--list-datasets` prints what a run
would cover, grouped by stratum, and downloads nothing — use it to confirm both
arms will cover the same ground before spending hours.

`--seeds 3` is the house setting. More seeds are welcome on everything except
the temporal stratum; see the disqualifier list for why.

`python benchmarks/bench_status.py` reports progress from another shell while a
run is going, and prints the aggregate table of the most recent completed run
once it finishes. Include that table in your submission.

The two `compare_runs.py` invocations are both needed whenever classification
datasets are in play. The default `--metric primary` scores regression on RMSE
and classification on macro F1. Classification decisions in this project are
made on Brier, which is what `--metric brier` reports, oriented so that the new
arm winning means lower Brier. A Brier gain ships even at a small F1 cost, so
the Brier read is the one that carries the classification verdict and the F1
read is context.

If the two arms name their model differently — say a baseline `ChimeraBoost`
against an `ChimeraBoostEns5` arm — pass `--model` for the base and
`--model-new` for the new one. Otherwise use the same model name in both arms
and pass it once.

---

## 4. Strata, and why pooling is forbidden

`--by-suite` is mandatory on a decision run. Without it the tool pools every
dataset into one sign test, and that test is not the one the protocol asks for.

A **variant** is a derived view of a parent dataset, not a new dataset.
`gr:reg_num/cpu_act@sus25` trains on 25% of the rows its parent trains on and
is scored on *exactly the parent's test rows*. A pooled sign test counts that
data twice, inflates the effective sample size, and hands back a weaker test
wearing the name of a stronger one. The two base suites also answer different
questions, so a pooled bar averages away the very contrast that makes running
both worthwhile.

A `--decide` run therefore produces **seven strata**:

| stratum | what it probes |
|---|---|
| Grinsztajn, full | the general tabular regime, no categorical features passed |
| high-card, full | real high-cardinality categoricals and multiclass |
| Grinsztajn `@sus25` | small data — 25% of the training rows, test set unchanged |
| Grinsztajn `@sus50` | small data — 50% of the training rows, test set unchanged |
| high-card `@sus25` | small data in the categorical regime |
| high-card `@sus50` | small data in the categorical regime |
| high-card `@time` | distribution shift — train on the past, predict the future |

Grinsztajn has no temporal stratum and will not get one: the mirror we load
ships pre-transformed numeric CSVs with no recoverable timestamp, so the regime
has no expression there and the registry says so rather than inventing a proxy.
Exact dataset counts per stratum come from `--list-datasets`; they move when a
suite grows, so quote the run's own numbers rather than any written here.

A change that wins on one stratum and not the others is not automatically
rejected, but it needs a story for why — a categorical lever that helps the
high-cardinality suite and is inert on Grinsztajn is a coherent result; the same
pattern with no mechanism behind it is a coin flip that landed well.

---

## 5. Reading the output

**The summary statistic is head-to-head win rate with a bootstrap confidence
interval and a median gap.** The run prints a SUMMARY block per stratum:
win rate against each competitor, a 95% bootstrap interval over datasets, the
win/loss/tie counts with ties counting a half, the median gap, and a mean
fit-time ratio. `compare_runs.py` prints the paired view — a row per dataset,
the win/loss/tie counts, the mean and median relative change, and a PASS/FAIL
bar set at more than half the datasets.

**Never gate on a mean of relative gaps.** It is the statistic that has misled
this project three separate times. One near-solved dataset with a practically
zero denominator contributed a five-figure negative percentage by itself and
dragged an 88-dataset mean to −144% while the sign test read 54 wins to 31
losses. Datasets every model solves to nothing are now excluded from the mean
(regression below 2% normalised RMSE, classification below 0.001 Brier) and
always named in the output, and `compare_runs.py` additionally shouts when the
mean exceeds three times its median, naming the single dataset responsible and
printing the mean without it. If you see that warning, the mean is one dataset
talking. Read the sign test and the median.

**Ties count against the change, so read the decided-only count too.** The bar
is wins against half of all shared datasets, which means a change that is
byte-identical on part of the suite fails by construction: one internal result
read 6 wins, 0 losses, 8 ties and scored FAIL. This gets worse the better your
change is gated, since a size-gated or dtype-gated change is deliberately inert
where it does not apply. Report both readings. "6 wins, 0 losses, 8 ties, bar
FAIL" is honest; "FAIL" on its own is not.

**A stratum with fewer than about eight decided datasets is a pointer, not a
gate.** One internal FAIL on a four-dataset stratum was carried into a shipping
recommendation and later measured at 21 wins to 21 losses, p = 1.000. If a
small stratum is the only thing standing between your change and a verdict, say
so and probe that regime directly rather than treating the stratum as an answer.

**State which statistic you are quoting, every time.** A candidate here was
killed by comparing its median against another program's mean and revived when
the comparison was redone like for like. Record both, labelled.

---

## 6. What a complete submission contains

Open an issue or a pull request with all of the following. Anything missing is
something we have to guess at, and a guess is what sends a submission back.

- **Both arms as saved JSON**, produced back to back on the same idle machine,
  the same suite, the same seed count, the same flags. Attach them or link them.
- **The same model name in both arms**, or the `--model` / `--model-new` pair
  that maps one onto the other.
- **The full per-stratum sign test pasted verbatim**, including the strata that
  lost. A submission that shows only the winning strata is read as a submission
  that has strata it is not showing.
- **The aggregate table** from the run, or from `bench_status.py`.
- **Hardware and versions**: CPU, core count, operating system, Python version,
  and the versions of numpy, numba, scikit-learn and every competitor installed.
- **The commit each arm was built from.**
- **An explicit statement of what the change is**, in a sentence, in terms of
  behaviour rather than diff.
- **A mechanism story**: which stratum should move, in which direction, and why,
  written before the decision run rather than fitted to it. If you picked the
  arm after seeing a pilot, that run is exploratory and cannot be cited as a
  passed test — say so and confirm on data that did not select it.

The bar we apply to ourselves is a decisive sign test together with a
non-negative median improvement on the Grinsztajn stratum and on the
high-cardinality stratum, each tested independently. A large speed regression
needs an explicit argument for why it is worth paying.

---

## 7. What disqualifies a submission

Each of these has produced a wrong answer here at least once.

- **A pooled sign test.** The variant rows are already counted in their parents.
- **A mean of relative gaps as the headline number.** See above.
- **One suite presented as a decision.** Grinsztajn alone cannot see categorical
  levers; the high-cardinality suite alone is fourteen datasets.
- **A verdict resting on a tiny stratum**, including a FAIL resting on one.
- **More than three seeds on `@time` offered as more evidence.** The temporal
  split takes its cut from a fixed list of three rolling origins and has no
  other seed-dependent randomness, so seed 3 reproduces seed 0 exactly. The
  harness warns about this. Extra seeds duplicate rows and make the table look
  twice as strong as it is; vary the cuts in a probe if you need more windows.
- **A speed claim from a busy machine.** One benchmark at a time, nothing else
  running.
- **Two arms from different machines, different commits, or different competitor
  versions.** Every one of those moves the numbers on its own.

---

## 8. Timing honesty

`fit_time` is wall clock and excludes prediction and metric computation; runs
are stamped `timing="fit_only"` in their config, and the tools warn loudly if
you compare a run from before that convention against one from after — older
runs charged scoring to fit and their speed columns read low for slow-fitting
models.

**Run one benchmark at a time.** The harness already spreads seeds across
parallel processes and splits the thread budget between them, so a second run
contends for the same cores and corrupts both sets of timings. Nothing else
demanding should be running either.

**Speed numbers are comparable only within one run on one machine.** The
harness reports fit-time multiples relative to another model in the same run
for exactly this reason. Do not compare your absolute seconds against ours, and
do not compare either against a published leaderboard's timings, which measure
that leaderboard's environment.

---

## 9. What happens after you submit

We weigh the strata ourselves and make the ship call. There is no arithmetic
rule that converts seven sign tests into a decision, deliberately: the strata
answer different questions and the right weighting depends on what the change
claims to do. A passing sign test is evidence, and good evidence moves fast
here, but it is not an automatic acceptance.

Expect one of three outcomes. The change ships, in which case the verdict and
the numbers are recorded. The change is killed, which is a real result and gets
recorded the same way — kills are worth as much as wins and cost less. Or we
ask for one more run, usually because a stratum too small to decide anything is
the only thing in the way.

Before writing a verdict, ours or yours, four questions:

1. How many independent things is this actually, as opposed to how many rows
   the table has?
2. What happens to it if the single biggest contributor is dropped?
3. Is the same statistic being compared on both sides?
4. Was this instrument built to be decided on?

Three of the worst readings in this project's history failed question 2.
