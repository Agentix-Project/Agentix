"""Tests for the runtime server's built-in endpoints: exec/upload/download.

These endpoints live on the runtime directly (no closure load required),
so we drive them through an ASGI transport on the real FastAPI app.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def builtins_app(runtime_module):
    """Alias kept for clarity in tests below: returns (server_module, upload_root)."""
    server, _mnt, upload_root = runtime_module
    return server, upload_root


async def test_exec_buffered(builtins_app):
    server, _root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post("/exec", json={"command": "echo hi && echo err 1>&2"})
        assert r.status_code == 200
        body = r.json()
        assert body["exit_code"] == 0
        assert body["stdout"] == "hi\n"
        assert body["stderr"] == "err\n"


async def test_exec_respects_cwd_and_env(builtins_app):
    server, root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/exec",
            json={"command": "echo $FOO && pwd", "cwd": str(root), "env": {"FOO": "bar"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert "bar" in body["stdout"]
        assert str(root) in body["stdout"]


async def test_exec_validates_missing_command(builtins_app):
    server, _root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post("/exec", json={})
        assert r.status_code in (400, 422)


async def test_exec_scrubs_nix_env(builtins_app, monkeypatch: pytest.MonkeyPatch):
    """Nix-flavoured env vars in the runtime process must not leak into subprocesses."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nix/store/xxx/lib")
    monkeypatch.setenv("NIX_CFLAGS_COMPILE", "-I/nix/store/xxx/include")
    server, _root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/exec",
            json={"command": "echo LD=$LD_LIBRARY_PATH NIX=$NIX_CFLAGS_COMPILE"},
        )
        body = r.json()
        assert body["exit_code"] == 0
        assert "LD=\n" in body["stdout"] or "LD= " in body["stdout"]
        assert "NIX=\n" in body["stdout"] or "NIX= " in body["stdout"]


async def test_upload_download_round_trip(builtins_app):
    server, root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        target = root / "sub" / "hello.txt"
        r = await http.post(
            "/upload",
            files={"file": ("hello.txt", b"hello world")},
            data={"path": str(target)},
        )
        assert r.status_code == 200
        assert r.json() == {"path": str(target), "size": 11}
        assert target.read_bytes() == b"hello world"

        r = await http.get("/download", params={"path": str(target)})
        assert r.status_code == 200
        assert r.content == b"hello world"


async def test_exec_paths_from_prepends_closure_bin(
    runtime_module, mount_echo,
):
    """`paths_from=[<package>]` prepends `<mount>/entry/bin` to PATH."""
    server, mount_root, _ = runtime_module
    mount = mount_echo()
    # Drop a marker script the test can probe via PATH
    bin_dir = mount / "entry" / "bin"
    bin_dir.mkdir(parents=True)
    marker = bin_dir / "my-marker"
    marker.write_text("#!/bin/sh\necho closure-wins\n")
    marker.chmod(0o755)
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/exec",
            json={
                "command": "my-marker",
                "paths_from": ["agentix_closures.echo"],
            },
        )
        assert r.status_code == 200
        assert r.json()["stdout"].strip() == "closure-wins"

        # Without paths_from, the marker is not on PATH.
        r = await http.post("/exec", json={"command": "my-marker"})
        assert r.json()["exit_code"] != 0


async def test_upload_rejects_path_outside_root(builtins_app):
    server, _root = builtins_app
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/upload",
            files={"file": ("x.txt", b"x")},
            data={"path": "/etc/passwd"},
        )
        assert r.status_code == 403
