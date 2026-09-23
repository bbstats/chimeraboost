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


def test_run_chimera_catcount_off_forces_false_and_default_leaves_unset(monkeypatch):
    """The "off" sentinel forces cat_count_features=False; the default
    argument leaves the parameter unset so the class default applies."""
    import pytest

    recorded = {}

    class _Sentinel(Exception):
        pass

    class _StubEst:
        def __init__(self, **kw):
            recorded.update(kw)

        def fit(self, X, y, cat_features=None):
            raise _Sentinel()

    monkeypatch.setattr(rb, "ChimeraBoostRegressor", _StubEst)
    X = [[0.0, 1.0]] * 4
    y = [0.0, 1.0, 0.0, 1.0]
    with pytest.raises(_Sentinel):
        rb._run_chimera("regression", X, y, X, y, None, 1,
                        cat_count_features="off")
    assert recorded["cat_count_features"] is False
    recorded.clear()
    with pytest.raises(_Sentinel):
        rb._run_chimera("regression", X, y, X, y, None, 1)
    assert "cat_count_features" not in recorded


def test_no_catcount_arm_registered_and_off_by_default():
    assert "ChimeraBoostNoCatCount" in rb.RUNNERS
    assert "ChimeraBoostNoCatCount" in rb._OFF_BY_DEFAULT

# ---------------------------------------------------------------------------
# H(11): the public-suite download race. _public_parquet_path downloads into a
# per-process temp file; when os.replace loses to a sibling worker (Windows
# WinError 32 while the final file is open elsewhere) and the final file is now
# complete, it drops its own temp copy and returns the path. No network.
# ---------------------------------------------------------------------------

class _FakeResp:
    """Minimal urllib response: a context manager serving fixed bytes."""

    def __init__(self, payload):
        self._data = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=1 << 20):
        out, self._data = self._data[:n], self._data[n:]
        return out


def _patch_public_download(monkeypatch, tmp_path, payload=b"our-bytes"):
    import urllib.request
    monkeypatch.setattr(rb, "_PUBLIC_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=600: _FakeResp(payload))


def test_public_parquet_lost_race_uses_winners_file(tmp_path, monkeypatch):
    """os.replace loses to a sibling worker: the sibling's final file is
    returned intact and no .part file is left behind."""
    _patch_public_download(monkeypatch, tmp_path)
    for i, err in enumerate((PermissionError, FileExistsError)):
        def _lose(src, dst, _err=err):
            with open(dst, "wb") as f:      # the sibling's own download
                f.write(b"sibling-bytes")
            raise _err("locked by sibling")

        monkeypatch.setattr(os, "replace", _lose)
        data_id = 12340 + i
        got = rb._public_parquet_path(data_id)
        want = os.path.join(str(tmp_path), f"dataset_{data_id}.pq")
        assert got == want
        with open(want, "rb") as f:
            assert f.read() == b"sibling-bytes"
        assert not os.path.exists(f"{want}.{os.getpid()}.part")
    left = [p for p in os.listdir(str(tmp_path)) if p.endswith(".part")]
    assert left == []


def test_public_parquet_lost_race_without_winner_reraises(tmp_path, monkeypatch):
    """os.replace fails and no final file appears: the error propagates."""
    import pytest
    _patch_public_download(monkeypatch, tmp_path)
    for i, err in enumerate((PermissionError, FileExistsError)):
        def _boom(src, dst, _err=err):
            raise _err("locked, and no winner")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(err):
            rb._public_parquet_path(99900 + i)


def test_public_parquet_success_leaves_no_part_file(tmp_path, monkeypatch):
    """The uncontended path lands the bytes and leaves no temp file behind."""
    _patch_public_download(monkeypatch, tmp_path)
    got = rb._public_parquet_path(777)
    with open(got, "rb") as f:
        assert f.read() == b"our-bytes"
    assert [p for p in os.listdir(str(tmp_path)) if p.endswith(".part")] == []
