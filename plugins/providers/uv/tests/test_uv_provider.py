"""uv provider: launch the runtime from a venv and drive a real remote() call.

Uses `reuse_venv` pointed at the interpreter running the tests (it already has
`agentixx` + uvicorn), so the test needs no uv materialization. Remote targets
are stdlib functions (`math.*`) — always importable by the worker, so the test
exercises the provider's runtime wiring without packaging a fixture module. (A
user's own rollout module is reached the same way every provider does it:
installed into the venv via `UvProviderConfig.project` / `install`.)
"""

from __future__ import annotations

import math
import sys

import pytest
from agentix.provider.uv import UvProvider, UvProviderConfig

from agentix.provider.base import SandboxConfig, SandboxProvider


def _reuse_venv() -> str:
    # venv root of the interpreter running the tests. Use sys.prefix, NOT a
    # resolved sys.executable: the venv's bin/python is a symlink, and resolving
    # it jumps to the base interpreter (whose env lacks agentixx).
    return sys.prefix


def test_config_requires_project_or_venv():
    with pytest.raises(ValueError):
        UvProviderConfig()


def test_is_sandboxprovider():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    assert isinstance(provider, SandboxProvider)


@pytest.mark.asyncio
async def test_remote_roundtrip():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    try:
        async with provider.session(SandboxConfig(image="uv", bundle="uv")) as sandbox:
            assert (await sandbox.health()).version
            assert await sandbox.remote(math.factorial, 5) == 120
            assert await sandbox.remote(math.gcd, 12, 8) == 4
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_and_delete():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    try:
        sandbox = await provider.create(SandboxConfig(image="uv", bundle="uv"))
        info = await provider.get(sandbox.sandbox_id)
        assert info.status == "running"
        await sandbox.aclose()
        await provider.delete(sandbox.sandbox_id)
        with pytest.raises(KeyError):
            await provider.get(sandbox.sandbox_id)
    finally:
        await provider.aclose()


# A fake "runtime server" for pipe-handling tests: binds, answers /health,
# then floods stdout and exits. Undrained, asyncio's flow control pauses the
# pipe at ~192KiB (2x the StreamReader limit + one pipe buffer) and the child
# BLOCKS mid-write — a server that logs past that wedges every in-flight
# rollout on that sandbox.
_FLOOD_SERVER = '''
import socket, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port)); srv.listen()
conn, _ = srv.accept()
conn.recv(65536)
conn.sendall(b"HTTP/1.0 200 OK\\r\\ncontent-length: 2\\r\\n\\r\\nok")
conn.close()
sys.stdout.write("x" * (4 * 1024 * 1024))   # 4 MiB of "logs"
sys.stdout.write("TAIL-MARKER")
sys.stdout.flush()
'''

_EXITING_SERVER = '''
import sys
sys.stdout.write("boom-marker: refusing to start")
sys.stdout.flush()
sys.exit(3)
'''


def _fake_venv(tmp_path, server_source: str) -> str:
    """A venv whose bin/python launches `server_source` instead of uvicorn."""
    script = tmp_path / "server.py"
    script.write_text(server_source)
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "python"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    shim.chmod(0o755)
    return str(tmp_path / "venv")


@pytest.mark.asyncio
async def test_server_stdout_is_drained_not_retained(tmp_path):
    """The runtime's stdout must be continuously drained for the sandbox's
    lifetime: undrained, a server that logs past the pipe flow-control limit
    (~192KiB) blocks mid-write and wedges every in-flight rollout. The drain
    keeps only a BOUNDED tail in host memory."""
    import asyncio

    provider = UvProvider(
        UvProviderConfig(reuse_venv=_fake_venv(tmp_path, _FLOOD_SERVER), health_timeout=10.0)
    )
    try:
        sandbox = await provider.create(SandboxConfig(image="uv", bundle="uv"))
        running = provider._running[sandbox.sandbox_id]  # noqa: SLF001
        # the fake server floods 4 MiB after health, then exits — it can only
        # finish if someone is draining the pipe
        for _ in range(100):
            if running.proc.returncode is not None:
                break
            await asyncio.sleep(0.1)
        assert running.proc.returncode is not None, (
            "server wedged mid-write: stdout pipe is not being drained"
        )
        # deterministic: the drain ends once the pipe hits EOF
        await asyncio.wait_for(running.drain, timeout=10)

        retained = sum(len(c) for c in running.tail)
        assert retained < 1024 * 1024, (
            f"provider retains {retained} bytes of server stdout in host memory"
        )
        # ... but the LAST output survives in the bounded tail for diagnostics
        assert b"TAIL-MARKER" in b"".join(running.tail)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_early_exit_diagnostics_survive_draining(tmp_path):
    """A server that dies before health must still surface its output in the
    error — the drain task, not a one-shot read, owns the pipe."""
    provider = UvProvider(
        UvProviderConfig(reuse_venv=_fake_venv(tmp_path, _EXITING_SERVER), health_timeout=10.0)
    )
    try:
        with pytest.raises(RuntimeError, match="boom-marker"):
            await provider.create(SandboxConfig(image="uv", bundle="uv"))
    finally:
        await provider.aclose()


# Serves /health forever on the given port (terminates cleanly on SIGTERM).
_HEALTHY_SERVER = '''
import socket, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port)); srv.listen()
while True:
    conn, _ = srv.accept()
    conn.recv(65536)
    conn.sendall(b"HTTP/1.0 200 OK\\r\\ncontent-length: 2\\r\\n\\r\\nok")
    conn.close()
'''

# Dies before health, but first spawns a grandchild that inherits (and holds
# open) the merged stdout pipe — so the drain sees no EOF for several seconds.
_ORPHAN_HOLDER_SERVER = '''
import subprocess, sys
sys.stdout.write("boom-diagnostic: refusing to start")
sys.stdout.flush()
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(6)"])
sys.exit(3)
'''


@pytest.mark.asyncio
async def test_early_exit_diagnostic_survives_pipe_held_by_grandchild(tmp_path):
    """When a grandchild holds the pipe open past the 2s diagnostic wait,
    create() must still raise the diagnostic RuntimeError — never a bare
    CancelledError from re-awaiting a drain task that wait_for() cancelled."""
    provider = UvProvider(
        UvProviderConfig(reuse_venv=_fake_venv(tmp_path, _ORPHAN_HOLDER_SERVER), health_timeout=10.0)
    )
    try:
        with pytest.raises(RuntimeError, match="boom-diagnostic"):
            await provider.create(SandboxConfig(image="uv", bundle="uv"))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_concurrent_creates_never_share_a_port(tmp_path, monkeypatch):
    """The allocated port is only bound by the subprocess seconds later, so it
    must be reserved in-process until the sandbox dies — two creates that each
    bind-and-close can otherwise collide (same guard as DockerProvider)."""
    import agentix.provider.uv as uv_mod

    f1, f2 = uv_mod._free_port(), uv_mod._free_port()
    ports = iter([f1, f1, f2])  # the kernel hands out the same number twice
    monkeypatch.setattr(uv_mod, "_free_port", lambda: next(ports))

    provider = UvProvider(
        UvProviderConfig(reuse_venv=_fake_venv(tmp_path, _HEALTHY_SERVER), health_timeout=10.0)
    )
    try:
        s1 = await provider.create(SandboxConfig(image="uv", bundle="uv"))
        s2 = await provider.create(SandboxConfig(image="uv", bundle="uv"))
        p1 = provider._running[s1.sandbox_id].port  # noqa: SLF001
        p2 = provider._running[s2.sandbox_id].port  # noqa: SLF001
        assert p1 != p2
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_failed_materialization_removes_temp_root(tmp_path, monkeypatch):
    """A failed `uv venv` / `uv pip install` must not orphan the mkdtemp root —
    a retry loop would otherwise leak one partial venv per attempt."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    provider = UvProvider(UvProviderConfig(project=".", uv_bin="/usr/bin/false"))
    with pytest.raises(RuntimeError):
        await provider._ensure_venv()  # noqa: SLF001
    assert list(tmp_path.iterdir()) == []


def test_default_uv_bin_resolves_packaged_binary(monkeypatch):
    """The packaged `uv` dependency must be found even when the venv's bin is
    not on PATH (systemd/cron/absolute-path launches) — that's the whole point
    of depending on the `uv` wheel."""
    from pathlib import Path

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    resolved = UvProviderConfig(project=".").resolved_uv_bin()
    assert resolved != "uv"
    assert Path(resolved).is_file()
