from __future__ import annotations

import asyncio
import importlib.util
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_files_upload_refuses_symlink_escape(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("AGENTIX_UPLOAD_ROOT", str(root))
    files = _load_module(
        "_agentix_test_files_primitive",
        REPO_ROOT / "primitives/files/src/agentix/files/__init__.py",
    )

    link = root / "link"
    link.symlink_to(outside)

    with pytest.raises(PermissionError):
        await files.upload(str(link), b"escape")
    assert not outside.exists()


@pytest.mark.asyncio
async def test_bash_run_drains_stderr_after_output_cap():
    bash = _load_module(
        "_agentix_test_bash_primitive",
        REPO_ROOT / "primitives/bash/src/agentix/bash/__init__.py",
    )
    code = (
        "import sys; "
        "sys.stderr.write('x' * 200000); "
        "sys.stderr.flush(); "
        "print('done')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await asyncio.wait_for(
        bash.run(command, max_output=1024),
        timeout=5,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "done"
    assert "[truncated at 1024 bytes]" in result.stderr
