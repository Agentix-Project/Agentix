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
