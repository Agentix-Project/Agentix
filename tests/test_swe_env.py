from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SWE_PLUGIN = ROOT / "plugins" / "datasets" / "swebench"
sys.path.insert(0, str(SWE_PLUGIN))

import agentix.plugins.datasets.swe as swe  # noqa: E402

swe_score = importlib.import_module("agentix.plugins.datasets.swe.score")


def test_swe_public_exports() -> None:
    assert callable(swe.prepare_env)
    assert swe.score is swe_score.score


def test_test_patch_paths_preserve_new_files_for_cleanup() -> None:
    modified, touched = swe_score._test_patch_paths(
        "diff --git a/tests/test_new.py b/tests/test_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_new.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_new(): pass\n"
    )

    assert modified == []
    assert touched == ["tests/test_new.py"]


def test_test_patch_paths_include_modified_preimage() -> None:
    modified, touched = swe_score._test_patch_paths(
        "diff --git a/tests/test_existing.py b/tests/test_existing.py\n"
        "--- a/tests/test_existing.py\n"
        "+++ b/tests/test_existing.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    assert modified == ["tests/test_existing.py"]
    assert touched == ["tests/test_existing.py"]


def test_test_patch_paths_include_renamed_target_for_cleanup() -> None:
    modified, touched = swe_score._test_patch_paths(
        "diff --git a/tests/old.py b/tests/new.py\n"
        "similarity index 90%\n"
        "rename from tests/old.py\n"
        "rename to tests/new.py\n"
        "--- a/tests/old.py\n"
        "+++ b/tests/new.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    assert modified == ["tests/old.py"]
    assert touched == ["tests/old.py", "tests/new.py"]


def test_eval_export_commands_update_subprocess_env() -> None:
    env: dict[str, str] = {}

    assert swe_score._apply_export("export LANG=en_US.UTF-8", env) is True
    assert env == {"LANG": "en_US.UTF-8"}
    assert swe_score._apply_export("sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen", env) is False


def test_get_test_command_uses_swebench_specs_and_directives() -> None:
    command = swe_score._get_test_command(
        {
            "repo": "pallets/flask",
            "version": "2.0",
            "test_patch": "diff --git a/tests/test_basic.py b/tests/test_basic.py\n",
        }
    )

    assert command.startswith("pytest")
    assert "tests/test_basic.py" in command


def test_make_report_uses_swebench_grading(monkeypatch) -> None:
    monkeypatch.setattr(
        swe_score,
        "make_test_spec",
        lambda _instance: SimpleNamespace(
            FAIL_TO_PASS=["tests/test_app.py::test_fixed"],
            PASS_TO_PASS=["tests/test_app.py::test_still_ok"],
        ),
    )

    report = swe_score._make_report(
        {"repo": "pallets/flask"},
        {
            "tests/test_app.py::test_fixed": "PASSED",
            "tests/test_app.py::test_still_ok": "PASSED",
        },
    )

    assert report["resolved"] is True
    assert report["test_status"] == {
        "tests/test_app.py::test_fixed": "PASSED",
        "tests/test_app.py::test_still_ok": "PASSED",
    }


_INSTANCE = {"instance_id": "demo__demo-1", "repo": "demo/demo", "version": "1.0", "base_commit": "abc"}


def _wire_score_pipeline(monkeypatch, *, run_output: str, report: dict) -> None:
    """Stub every subprocess-touching stage so `score()` exercises only
    the result-shaping wiring under test."""
    monkeypatch.setattr(swe_score, "_get_env", dict)
    monkeypatch.setattr(swe_score, "prepare_env", _async(SimpleNamespace(ok=True, head="abc", log="")))
    monkeypatch.setattr(swe_score, "_apply_model_patch", _async((True, "")))
    monkeypatch.setattr(swe_score, "_prepare_tests", _async((True, "")))
    monkeypatch.setattr(swe_score, "_get_test_command", lambda _instance: "pytest")
    monkeypatch.setattr(swe_score, "_run", _async((1, run_output, False)))
    monkeypatch.setattr(swe_score, "_cleanup_tests", _async(None))
    monkeypatch.setattr(swe_score, "_get_log_parser", lambda _instance: lambda _log: {})
    monkeypatch.setattr(swe_score, "_make_report", lambda _instance, _status: dict(report))


def _async(value):
    async def stub(*_args, **_kwargs):
        return value

    return stub


async def test_score_unresolved_keeps_stage_exit_code_and_log_tail(monkeypatch) -> None:
    _wire_score_pipeline(
        monkeypatch,
        run_output="x" * (swe_score.LOG_TAIL_CHARS + 100) + "TAIL-MARKER",
        report={"resolved": False, "patch_applied": True, "timed_out": False, "test_status": {}},
    )

    result = await swe_score.score(_INSTANCE, patch="diff")

    assert result["failure_stage"] == "tests"
    assert result["exit_code"] == 1
    assert result["log_tail"].endswith("TAIL-MARKER")
    assert len(result["log_tail"]) == swe_score.LOG_TAIL_CHARS


async def test_score_resolved_carries_no_failure_evidence(monkeypatch) -> None:
    _wire_score_pipeline(
        monkeypatch,
        run_output="1 passed",
        report={"resolved": True, "patch_applied": True, "timed_out": False, "test_status": {"t": "PASSED"}},
    )

    result = await swe_score.score(_INSTANCE, patch="diff")

    assert result["resolved"] is True
    assert "failure_stage" not in result and "log_tail" not in result


async def test_score_prepare_tests_failure_is_attributed(monkeypatch) -> None:
    _wire_score_pipeline(monkeypatch, run_output="", report={})
    monkeypatch.setattr(swe_score, "_prepare_tests", _async((False, "$ tox\nboom")))

    result = await swe_score.score(_INSTANCE, patch="diff")

    assert result["resolved"] is False
    assert result["patch_applied"] is True
    assert result["failure_stage"] == "prepare_tests"
    assert result["log_tail"] == "$ tox\nboom"


async def test_score_apply_failure_is_attributed(monkeypatch) -> None:
    _wire_score_pipeline(monkeypatch, run_output="", report={})
    monkeypatch.setattr(swe_score, "_apply_model_patch", _async((False, "$ git apply\nerror: corrupt")))

    result = await swe_score.score(_INSTANCE, patch="diff")

    assert result["patch_applied"] is False
    assert result["failure_stage"] == "apply_patch"
    assert result["log_tail"] == "$ git apply\nerror: corrupt"


async def test_prepare_tests_reapplies_pre_install_before_install(monkeypatch, tmp_path) -> None:
    # Task images bake pre_install's tracked-file edits in at build time
    # and prepare_env's `git reset --hard` reverts them — _prepare_tests
    # must re-apply them ahead of the (re)install.
    monkeypatch.setattr(
        swe_score,
        "MAP_REPO_VERSION_TO_SPECS",
        {
            "demo/demo": {
                "1.0": {
                    "pre_install": ["sed -i 's/pytest/pytest -rA/' tox.ini"],
                    "eval_commands": ["export LANG=C"],
                    "install": "python -m pip install -e .",
                    "test_cmd": "pytest -rA",
                }
            }
        },
    )
    ran: list[str] = []

    async def record(command, workdir, env, timeout, *, conda=False):
        ran.append(command)
        return 0, "", False

    monkeypatch.setattr(swe_score, "_run", record)
    monkeypatch.setattr(swe_score, "_remove_untracked_paths", _async(None))

    ok, log = await swe_score._prepare_tests(
        {
            "repo": "demo/demo",
            "version": "1.0",
            "base_commit": "abc",
            "test_patch": (
                "diff --git a/tests/t.py b/tests/t.py\n"
                "--- a/tests/t.py\n"
                "+++ b/tests/t.py\n"
                "@@ -1 +1 @@\n"
                "-a\n"
                "+b\n"
            ),
        },
        str(tmp_path),
        {},
        60.0,
    )

    assert ok and log == ""
    sed_at = next(i for i, c in enumerate(ran) if c.startswith("sed "))
    install_at = next(i for i, c in enumerate(ran) if "pip install" in c)
    assert sed_at < install_at


async def test_prepare_tests_pre_install_failure_is_best_effort(monkeypatch, tmp_path) -> None:
    # pre_install is a build-time recipe (matplotlib wgets a qhull
    # tarball into a dir `git clean` removed) — one failing step must
    # not abort preparation; install and the test patch still run.
    monkeypatch.setattr(
        swe_score,
        "MAP_REPO_VERSION_TO_SPECS",
        {
            "demo/demo": {
                "1.0": {
                    "pre_install": ["wget -O /gone/qhull.tgz http://example.com/qhull.tgz"],
                    "install": "python -m pip install -e .",
                    "test_cmd": "pytest -rA",
                }
            }
        },
    )
    ran: list[str] = []

    async def record(command, workdir, env, timeout, *, conda=False):
        ran.append(command)
        return (1, "wget: No such file or directory", False) if command.startswith("wget") else (0, "", False)

    monkeypatch.setattr(swe_score, "_run", record)
    monkeypatch.setattr(swe_score, "_remove_untracked_paths", _async(None))

    ok, log = await swe_score._prepare_tests(
        {"repo": "demo/demo", "version": "1.0", "base_commit": "abc", "test_patch": ""},
        str(tmp_path),
        {},
        60.0,
    )

    assert ok and log == ""
    assert any(c.startswith("wget") for c in ran)
    assert any("pip install" in c for c in ran)
