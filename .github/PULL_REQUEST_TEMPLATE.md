# What

<!-- 1-3 sentences: what changed and why. Link issues if any. -->

## Benchmark impact

<!-- Pick one: -->
- [ ] No default behavior changes (benchmarks unaffected)
- [ ] Default/algorithm change — benchmark JSON attached

```
python benchmarks/run_benchmarks.py --decide --seeds 3 --save
```

Attach `benchmarks/results/<timestamp>.json`, ideally one with your change and
one without. That is all we need.

## Checklist

- [ ] `pytest` green
- [ ] CHANGELOG.md updated (user-facing changes)
- [ ] Docs updated (API/parameter changes)
