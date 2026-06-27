"""Reliability coverage for host-side sidecar process supervision."""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import agentix.bridge.sidecar as sidecar_mod
import httpx
import pytest
from agentix.bridge import Sidecar, SidecarError
from agentix.bridge.sidecars import cc_convert_sidecar

NOISY_SERVER = r"""
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

chunk = b"x" * 65536
for _ in range(32):
    os.write(1, chunk)
    os.write(2, chunk)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

BIND_RETRY_SERVER = r"""
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

marker = Path(sys.argv[2])
if not marker.exists():
    marker.write_text("first attempt failed")
    sys.stderr.write("OSError: [Errno 48] Address already in use\n")
    raise SystemExit(1)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

ENV_FACTORY_SERVER = r"""#!/usr/bin/env python3
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

host, raw_port = os.environ["SIDECAR_LISTEN_ADDR"].rsplit(":", 1)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

HTTPServer((host, int(raw_port)), H).serve_forever()
"""
PRESET_SERVER = ENV_FACTORY_SERVER.replace("SIDECAR_LISTEN_ADDR", "CC_CONVERT_LISTEN_ADDR")


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_auto_port_reservations_are_distinct_and_released() -> None:
    first = sidecar_mod._reserve_free_port("127.0.0.1")
    try:
        second = sidecar_mod._reserve_free_port("127.0.0.1")
        try:
            assert first != second
            assert {first, second} <= sidecar_mod._reserved_ports
        finally:
            sidecar_mod._release_port(second)
    finally:
        sidecar_mod._release_port(first)

    assert sidecar_mod._reserved_ports == set()


async def test_noisy_sidecar_pipes_are_drained_and_tasks_are_cleaned(tmp_path: Path) -> None:
    script = tmp_path / "noisy_server.py"
    script.write_text(NOISY_SERVER)
    sidecar = Sidecar(
        command=[sys.executable, str(script), "{port}"],
        ready_timeout=10.0,
        poll_interval=0.02,
    )

    async with asyncio.timeout(15.0):
        async with sidecar as url:
            drain_tasks = tuple(sidecar._drain_tasks)
            async with httpx.AsyncClient(timeout=2.0) as client:
                assert (await client.get(url + "/healthz")).status_code == 200

    assert drain_tasks
    assert all(task.done() for task in drain_tasks)
    assert sidecar._drain_tasks == []


async def test_startup_error_retains_only_bounded_stderr_tail(tmp_path: Path) -> None:
    script = tmp_path / "failed_server.py"
    script.write_text(
        "import os\n"
        "os.write(2, b'BEGIN-MARKER\\n')\n"
        "for _ in range(32): os.write(2, b'x' * 1024)\n"
        "os.write(2, b'\\nEND-MARKER\\n')\n"
        "raise SystemExit(7)\n"
    )

    with pytest.raises(SidecarError) as exc_info:
        async with Sidecar(
            command=[sys.executable, str(script)],
            ready_timeout=3.0,
            poll_interval=0.02,
        ):
            pass

    message = str(exc_info.value)
    assert "code=7" in message
    assert "END-MARKER" in message
    assert "BEGIN-MARKER" not in message
    assert "stderr truncated to last 4096 bytes" in message
    assert len(message) < 4300


async def test_auto_port_retries_an_explicit_bind_conflict(tmp_path: Path) -> None:
    script = tmp_path / "bind_retry_server.py"
    script.write_text(BIND_RETRY_SERVER)
    marker = tmp_path / "attempted"
    ports: list[int] = []

    def command(_host: str, port: int) -> list[str]:
        ports.append(port)
        return [sys.executable, str(script), str(port), str(marker)]

    async with Sidecar(command=command, ready_timeout=5.0, poll_interval=0.02) as url:
        async with httpx.AsyncClient(timeout=2.0) as client:
            assert (await client.get(url + "/healthz")).status_code == 200

    assert len(ports) == 2
    assert len(set(ports)) == 2
    assert sidecar_mod._reserved_ports == set()


async def test_fixed_port_is_not_replaced_after_bind_conflict(tmp_path: Path) -> None:
    script = tmp_path / "bind_retry_server.py"
    script.write_text(BIND_RETRY_SERVER)
    marker = tmp_path / "attempted"
    fixed_port = _available_port()
    ports: list[int] = []

    def command(_host: str, port: int) -> list[str]:
        ports.append(port)
        return [sys.executable, str(script), str(port), str(marker)]

    sidecar = Sidecar(
        command=command,
        port=fixed_port,
        ready_timeout=3.0,
        poll_interval=0.02,
    )
    with pytest.raises(SidecarError, match="Address already in use"):
        async with sidecar:
            pass

    assert ports == [fixed_port]
    assert sidecar.url == f"http://127.0.0.1:{fixed_port}"


async def test_env_factory_receives_allocated_port_on_entry(tmp_path: Path) -> None:
    binary = tmp_path / "env_factory_sidecar"
    binary.write_text(ENV_FACTORY_SERVER)
    binary.chmod(0o755)
    seen: list[tuple[str, int]] = []

    def env(host: str, port: int) -> dict[str, str]:
        seen.append((host, port))
        return {"SIDECAR_LISTEN_ADDR": f"{host}:{port}"}

    sidecar = Sidecar(
        command=[str(binary)],
        env=env,
        ready_timeout=5.0,
    )

    assert sidecar.url == "http://127.0.0.1:0"
    async with sidecar as url:
        _, raw_port = url.rsplit(":", 1)
        assert seen == [("127.0.0.1", int(raw_port))]
        async with httpx.AsyncClient(timeout=2.0) as client:
            assert (await client.get(url + "/healthz")).status_code == 200


async def test_cc_convert_preset_allocates_port_and_env_on_entry(tmp_path: Path) -> None:
    binary = tmp_path / "fake_cc_convert_sidecar"
    binary.write_text(PRESET_SERVER)
    binary.chmod(0o755)
    sidecar = cc_convert_sidecar(
        binary=str(binary),
        upstream_url="http://upstream.invalid/v1/chat/completions",
        ready_timeout=5.0,
    )

    assert sidecar.url == "http://127.0.0.1:0"
    async with sidecar as url:
        assert not url.endswith(":0")
        async with httpx.AsyncClient(timeout=2.0) as client:
            assert (await client.get(url + "/healthz")).status_code == 200


async def test_teardown_cancels_and_awaits_stuck_drain_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    sidecar = Sidecar(command=[sys.executable, "-c", "pass"])
    stuck = asyncio.create_task(asyncio.Event().wait())
    sidecar._drain_tasks = [stuck]
    monkeypatch.setattr(sidecar_mod, "_DRAIN_SHUTDOWN_TIMEOUT", 0.01)

    await sidecar._terminate()

    assert stuck.done()
    assert stuck.cancelled()
    assert sidecar._drain_tasks == []
