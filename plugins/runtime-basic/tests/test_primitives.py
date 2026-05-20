from __future__ import annotations

import asyncio
import shlex
import sys

import agentix.bash as bash
import agentix.files as files
import pytest


@pytest.mark.asyncio
async def test_files_upload_refuses_symlink_escape(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("AGENTIX_UPLOAD_ROOT", str(root))
    # files reads UPLOAD_ROOT from env at import time; force a reload so
    # the patched env takes effect for this test.
    import importlib
    importlib.reload(files)

    link = root / "link"
    link.symlink_to(outside)

    with pytest.raises(PermissionError):
        await files.upload(str(link), b"escape")
    assert not outside.exists()


@pytest.mark.asyncio
async def test_bash_run_drains_stderr_after_output_cap():
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
