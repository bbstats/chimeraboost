# What

<!-- 1-3 sentences: what changed and why. Link issues if any. -->

## Benchmark impact

<!-- Pick one: -->
- [ ] No default behavior changes (benchmarks unaffected)
- [ ] Default/algorithm change — evidence attached, per
      [benchmarks/CONTRIBUTING_BENCHMARKS.md](../benchmarks/CONTRIBUTING_BENCHMARKS.md)

For a default or algorithm change, that means both arms of

```
python benchmarks/run_benchmarks.py --decide --seeds 3 --save
python benchmarks/compare_runs.py BASE.json NEW.json --model ChimeraBoost --by-suite
```

with the saved JSON for each arm, the per-stratum sign test pasted in full
including the strata that lost, and a sentence on which stratum you expected to
move and why. Never a pooled sign test across strata — a variant is a derived
view of its parent dataset, so pooling counts the same rows twice.

## Checklist

- [ ] `pytest` green
- [ ] TabArena results were not consulted (sealed holdout — report-only, never a
      reason for a change)
- [ ] CHANGELOG.md updated (user-facing changes)
- [ ] Docs updated (API/parameter changes)
