---
name: release
description: Cut a ChimeraBoost release — version bump, CHANGELOG, build, PyPI upload, tag, GitHub release
---

Pre-flight: working tree clean on main, full test suite green, no unmerged feature branches that
belong in this release (check `git branch` — the user forgets housekeeping).

1. **Bump version in BOTH places** (they must match): `pyproject.toml` `version = "X.Y.Z"` and
   `chimeraboost/__init__.py` `__version__`.
2. **CHANGELOG.md**: rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`. Watch for clobbered version
   headers from past merges.
3. Commit, then **build**: `python -m build` → `dist/chimeraboost-X.Y.Z*` (sdist + wheel).
4. **Upload**: `twine upload dist/chimeraboost-X.Y.Z*` (needs the PyPI token; if absent, ask the
   user — do NOT tag until upload succeeds, the TabArena pip_extra pins `chimeraboost>=`).
5. **Tag + push**: `git tag vX.Y.Z && git push origin main vX.Y.Z`. Always `git fetch` first —
   the user pushes README edits directly to origin/main.
6. **GitHub release**: `gh` is unauthenticated here. Get a token via
   `printf 'protocol=https\nhost=github.com\n\n' | git credential fill` → set `GH_TOKEN` for the
   `gh release create vX.Y.Z` call. If that fails, tell the user to create it manually.
7. **Verify**: `pip index versions chimeraboost` (or the PyPI page) shows X.Y.Z.

Known drift traps (both have happened):
- Source drifting past the PyPI version without a bump → published Elo/speed claims stop being
  reproducible from `pip install chimeraboost`. If defaults or speed changed, release before citing numbers.
- Version bump in one file but not the other.
