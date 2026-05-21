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


def test_sphinx_tox_eval_emits_pytest_report_sections() -> None:
    script, fixed = swe._fix_sphinx_tox_pytest_report_sections(
        "cd /testbed\ntox --current-env -epy39 -v -- tests/test_ext_autodoc_automodule.py\n"
    )

    assert fixed is True
    assert 'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA"' in script
    assert script.index("PYTEST_ADDOPTS") < script.index("tox --current-env")
