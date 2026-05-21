from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agentix.runtime.env import AGENTIX_ADDED_LD_LIBRARY_PATH, AGENTIX_ADDED_PATH

ROOT = Path(__file__).resolve().parents[1]
EVAL_CC_SWE = ROOT / "examples" / "eval-cc-swe"
sys.path.insert(0, str(EVAL_CC_SWE))

import swe  # noqa: E402


@pytest.mark.asyncio
async def test_swe_eval_script_runs_without_agentix_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/nix/runtime/venv/bin", "/usr/bin", "/bin"]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/nix/runtime/venv/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(["/nix/runtime/lib", "/task/lib"]))
    monkeypatch.setenv(AGENTIX_ADDED_LD_LIBRARY_PATH, "/nix/runtime/lib")

    script = tmp_path / "eval.sh"
    script.write_text(
        "\n".join(
            [
                "printf 'PATH=%s\\n' \"$PATH\"",
                "printf 'LD_LIBRARY_PATH=%s\\n' \"${LD_LIBRARY_PATH-}\"",
                "printf 'TRACKING=%s\\n' \"${AGENTIX_ADDED_LD_LIBRARY_PATH-unset}\"",
            ]
        )
    )

    out = await swe._run_script(script, tmp_path / "test.log", timeout=5)

    assert "PATH=/usr/bin:/bin" in out
    assert "LD_LIBRARY_PATH=/task/lib" in out
    assert "TRACKING=unset" in out


def test_new_file_only_test_patch_reset_preserves_setup_commit() -> None:
    script, fixed = swe._fix_swebench_new_file_only_test_patch_reset(
        base_commit="abc123",
        test_patch=(
            "diff --git a/tests/test_new.py b/tests/test_new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_new.py\n"
            "@@ -0,0 +1 @@\n"
            "+def test_new(): pass\n"
        ),
        eval_script=(
            "cd /testbed\n"
            "git checkout abc123 \n"
            "git apply -v - <<'EOF'\n"
            "tox --current-env -epy39 -v -- tests/test_new.py\n"
            "git checkout abc123 \n"
        ),
    )

    assert fixed is True
    assert "git checkout abc123" not in script
    assert "new-file-only test_patch must not reset the whole repo" in script
    assert "tests/test_new.py" in script
    assert "PYTEST_ADDOPTS" not in script


def test_modified_test_patch_reset_is_left_to_swebench() -> None:
    script = "cd /testbed\ngit checkout abc123 tests/test_existing.py\n"
    fixed_script, fixed = swe._fix_swebench_new_file_only_test_patch_reset(
        base_commit="abc123",
        test_patch=(
            "diff --git a/tests/test_existing.py b/tests/test_existing.py\n"
            "--- a/tests/test_existing.py\n"
            "+++ b/tests/test_existing.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
        eval_script=script,
    )

    assert fixed is False
    assert fixed_script == script
