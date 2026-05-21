from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

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


@pytest.mark.asyncio
async def test_bash_run_honors_bash_env(tmp_path: Path):
    bash_env = tmp_path / "bash_env"
    bash_env.write_text("export FROM_BASH_ENV=loaded\n")

    result = await bash.run(
        "printf '%s' \"$FROM_BASH_ENV\"",
        env={"BASH_ENV": str(bash_env)},
    )

    assert result.exit_code == 0
    assert result.stdout == "loaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("executable", ["bash", "zsh", "fish"])
async def test_bash_run_can_use_explicit_executable_names(tmp_path: Path, executable: str):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    shell_path = fakebin / executable
    shell_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"-c\" ]; then exit 64; fi\n"
        "shift\n"
        "export AGENTIX_TEST_SHELL=\"$(basename \"$0\")\"\n"
        "exec /bin/sh -c \"$1\"\n"
    )
    shell_path.chmod(0o755)

    result = await bash.run(
        "printf '%s' \"$AGENTIX_TEST_SHELL\"",
        env={"PATH": os.pathsep.join([str(fakebin), os.environ.get("PATH", "")])},
        executable=executable,
    )

    assert result.exit_code == 0
    assert result.stdout == executable


@pytest.mark.asyncio
async def test_bash_run_stream_can_use_explicit_executable(tmp_path: Path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    shell_path = fakebin / "zsh"
    shell_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"-c\" ]; then exit 64; fi\n"
        "shift\n"
        "export AGENTIX_TEST_SHELL=\"$(basename \"$0\")\"\n"
        "exec /bin/sh -c \"$1\"\n"
    )
    shell_path.chmod(0o755)

    events = [
        event
        async for event in bash.run_stream(
            "printf '%s' \"$AGENTIX_TEST_SHELL\"",
            env={"PATH": os.pathsep.join([str(fakebin), os.environ.get("PATH", "")])},
            executable="zsh",
        )
    ]

    assert [event.data for event in events if isinstance(event, bash.BashStdout)] == ["zsh"]
    assert [event.exit_code for event in events if isinstance(event, bash.BashExit)] == [0]
