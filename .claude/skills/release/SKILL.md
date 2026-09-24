---
name: release
description: Cut a ChimeraBoost release — version bump on a release branch, the user's PR merge, build, PyPI upload, tag, GitHub release
---

**Main is PR-only, releases included.** Never `git push origin main`. The user's credentials
have admin bypass, so a direct push to main succeeds with a "Bypassed rule violations" /
"Changes must be made through a pull request" line instead of failing. If any push prints
that, stop and tell the user. (It happened on 2026-07-13, and again with 0.33.0 on 2026-09-24,
when step 5 of this skill still said to push main.)

Pre-flight: an up-to-date main (`git fetch` first; the user pushes README edits directly to
origin/main), working tree clean, CI green on main's head, full test suite green, no unmerged
feature branches that belong in this release (check `git branch` — the user forgets
housekeeping).

1. **Release branch**: `git checkout -b release/X.Y.Z main`.
2. **Bump version in BOTH places** (they must match): `pyproject.toml` `version = "X.Y.Z"` and
   `chimeraboost/__init__.py` `__version__`.
3. **CHANGELOG.md**: rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`. Watch for clobbered version
   headers from past merges.
4. Commit ("Release X.Y.Z"), then **build and check on the branch**: `python -m build` →
   `dist/chimeraboost-X.Y.Z*` (sdist + wheel), `twine check dist/chimeraboost-X.Y.Z*`, and list
   the wheel's modules against `chimeraboost/*.py`. A packaging problem found here costs nothing;
   one found after upload costs a version number, since PyPI never takes the same version twice.
5. **Push the branch and open the PR**: `git push -u origin release/X.Y.Z`, then the PR through
   `gh` (token recipe in step 9) or the compare URL for the user. Then STOP: the user merges it.
   Nothing is uploaded or tagged before the merge.
6. **After the merge**: `git fetch`, `git merge --ff-only origin/main`, confirm the release commit
   is on main, and rebuild from main (`python -m build`, `twine check`) so the uploaded files
   match the tagged commit exactly.
7. **Upload**: `twine upload dist/chimeraboost-X.Y.Z*` (needs the PyPI token in `~/.pypirc`; if
   absent, ask the user — do NOT tag until upload succeeds, the TabArena pip_extra pins
   `chimeraboost>=`).
8. **Tag the merge commit and push only the tag**: `git tag vX.Y.Z <merge-commit>` then
   `git push origin vX.Y.Z`. Never push main.
9. **GitHub release**: `gh` is unauthenticated here. Get a token via
   `printf 'protocol=https\nhost=github.com\n\n' | git credential fill` → set `GH_TOKEN` (or use
   `bash campaign_tasks/gh.sh`) for `gh release create vX.Y.Z --notes-file ...`. If that fails,
   tell the user to create it manually.
10. **Verify**: `pip index versions chimeraboost` shows X.Y.Z. Then install `chimeraboost==X.Y.Z`
    into a throwaway venv on A: (C: is short on space) and run a smoke test from OUTSIDE the repo,
    where the editable checkout cannot shadow the installed package: `__version__`, and one
    regressor, classifier and quantile fit. Delete the venv afterwards.

Known drift traps (all have happened):
- Source drifting past the PyPI version without a bump → published Elo/speed claims stop being
  reproducible from `pip install chimeraboost`. If defaults or speed changed, release before citing numbers.
- Version bump in one file but not the other.
- A direct push to main (see the top).
