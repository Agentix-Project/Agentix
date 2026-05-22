"""Shared fixtures for agentix tests.

Production remote calls execute import-path callables inside a
worker process. Protocol tests can switch the runtime to an in-process
worker so they exercise HTTP/Socket.IO without subprocess stdio.
"""

from __future__ import annotations

import asyncio
import importlib
import socket
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def runtime_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh runtime per test: tmp upload root + reloaded server modules.

    Returns (server_module, tmp_path, upload_root). The tmp_path slot is
    kept for tests that need a scratch directory.
    """
    upload_root = tmp_path / "workspace"
    upload_root.mkdir()
    monkeypatch.setenv("AGENTIX_UPLOAD_ROOT", str(upload_root))

    # Reload server modules so each test gets a fresh worker client (no
    # cross-test target leakage). Order matters: leaves first,
    # package __init__ last.
    for mod in (
        "agentix.runtime.server.sio",
        "agentix.runtime.server.app",
        "agentix.runtime.server",
    ):
        if mod in sys.modules:
            try:
                importlib.reload(sys.modules[mod])
            except ImportError:
                sys.modules.pop(mod, None)

    from agentix.runtime import server

    return server, tmp_path, upload_root


@pytest.fixture
def use_inprocess_worker(runtime_module) -> Callable[[], None]:
    """Route protocol tests through the in-process worker backend."""
    server, _, _ = runtime_module

    def _use() -> None:
        server.app.state.worker._use_inprocess()

    return _use


@pytest.fixture(autouse=True)
def _purge_test_modules():
    """Per-test cleanup: drop any test-injected modules so the next test
    starts with a fresh slate. Real installed modules stay loaded.
    """
    yield
    # Test fixtures may have stashed temporary modules under arbitrary names.
    # Be conservative: only drop modules under `_agentix_test_*` prefixes
    # that test code might create.
    for mod in list(sys.modules):
        if mod.startswith("_agentix_test_"):
            sys.modules.pop(mod, None)


@pytest.fixture
async def live_server(runtime_module):
    """Yields an async `start()` callable that boots uvicorn on a free port
    serving the runtime's combined FastAPI+Socket.IO ASGI app.

    Test order:
        1. optionally use_inprocess_worker()
        2. base_url = await start()     # uvicorn starts
        3. connect via RuntimeClient(base_url) etc.

    The server is torn down in fixture finalisation.
    """
    import contextlib as _ctx
    import socket as _socket

    import httpx as _httpx
    import uvicorn

    server, _, _ = runtime_module
    state: dict = {"task": None, "srv": None}

    async def _start() -> str:
        if state["task"] is not None:
            raise RuntimeError("live_server already started")
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        config = uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            lifespan="on",
        )
        srv = uvicorn.Server(config)
        state["srv"] = srv
        state["task"] = asyncio.create_task(srv.serve())
        base_url = f"http://127.0.0.1:{port}"
        async with _httpx.AsyncClient() as c:
            for _ in range(100):
                try:
                    r = await c.get(f"{base_url}/health")
                    if r.status_code == 200:
                        return base_url
                except (_httpx.ConnectError, _httpx.ReadError):
                    pass
                await asyncio.sleep(0.05)
        raise RuntimeError("live_server did not become healthy in 5s")

    try:
        yield _start
    finally:
        if state["srv"] is not None:
            state["srv"].should_exit = True
            with _ctx.suppress(BaseException):
                await asyncio.wait_for(state["task"], timeout=5)
