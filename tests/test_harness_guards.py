"""run_benchmarks' argument guards (CAMPAIGN_PLAN I030, H(5)).

Two traps this pins closed:
  * ``--save NAME`` with a bare name wrote the tee file AND its derived
    ``.json`` into the shell's CWD (I003) -- a bare name now lands under
    ``benchmarks/results/`` and a name without an extension gets ``.txt``, so
    the JSON derived by swapping the extension lands beside it;
  * a ``--models`` mistake (unknown runner, or a field without the ChimeraBoost
    baseline) was reported only after the ``--save`` tee had opened a file,
    leaving an empty ``<stamp>.txt`` in results/ -- the check now runs first.

Pure helpers, no benchmark runs, no network.
"""
import os
import sys

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
sys.path.insert(0, BENCH)

import run_benchmarks as rb  # noqa: E402

RESULTS = os.path.join("some", "results")


def test_save_auto_is_the_stamped_file_under_results():
    assert rb.resolve_save_path("auto", RESULTS, "20260921-120000") == \
        os.path.join(RESULTS, "20260921-120000.txt")


def test_save_bare_name_lands_under_results_with_txt():
    got = rb.resolve_save_path("campaign-base", RESULTS, "stamp")
    assert got == os.path.join(RESULTS, "campaign-base.txt")
    # The JSON path run_benchmarks derives by swapping the extension therefore
    # lands beside it, not in the CWD.
    assert got.rsplit(".", 1)[0] + ".json" == \
        os.path.join(RESULTS, "campaign-base.json")


def test_save_bare_name_keeps_its_own_extension():
    assert rb.resolve_save_path("run.log", RESULTS, "s") == \
        os.path.join(RESULTS, "run.log")


def test_save_path_with_a_directory_is_used_as_given():
    explicit = os.path.join("elsewhere", "out.txt")
    assert rb.resolve_save_path(explicit, RESULTS, "s") == explicit
    # ...even when it has no extension: only the bare-name case is rewritten
    # to results/, and only a missing extension is filled in.
    noext = os.path.join("elsewhere", "out")
    assert rb.resolve_save_path(noext, RESULTS, "s") == noext + ".txt"


def test_models_guard_names_both_mistakes_and_passes_a_good_field():
    runners = {"ChimeraBoost": None, "CatBoost": None}
    assert rb.check_models_arg(None, runners) is None
    assert rb.check_models_arg(["ChimeraBoost", "CatBoost"], runners) is None
    assert "Unknown models" in rb.check_models_arg(["ChimeraBoost", "Nope"], runners)
    assert "ChimeraBoost must be" in rb.check_models_arg(["CatBoost"], runners)


def test_models_guard_runs_before_the_save_tee_opens(tmp_path, monkeypatch):
    """A usage error must not leave an empty results file behind."""
    import pytest
    out = os.path.join(str(tmp_path), "should-not-exist.txt")
    monkeypatch.setattr(sys, "argv",
                        ["run_benchmarks.py", "--models", "CatBoost", "--save", out])
    with pytest.raises(SystemExit) as e:
        rb.main()
    assert e.value.code == 2
    assert not os.path.exists(out)
