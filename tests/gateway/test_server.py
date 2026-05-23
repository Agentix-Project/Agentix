"""Tests for the gateway FastAPI surface.

Wires a real `Dispatcher` to the fake deployment used by
`test_dispatcher.py`, then drives it through HTTP via FastAPI's
TestClient.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentix.deployment.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo
from agentix.gateway.dispatcher import Dispatcher
from agentix.gateway.server import build_app


class FakeDeployment:
    def __init__(self) -> None:
        self._sandboxes: dict[str, Sandbox] = {}

    async def create(self, config: SandboxConfig) -> Sandbox:
        sid = SandboxId(f"sandbox-{uuid4().hex[:6]}")
        sb = Sandbox(sandbox_id=sid, runtime_url=f"http://127.0.0.1:0/{sid}", status="running")
        self._sandboxes[sid] = sb
        return sb

    async def delete(self, sandbox_id: SandboxId) -> None:
        self._sandboxes.pop(sandbox_id, None)

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        sb = self._sandboxes[sandbox_id]
        return SandboxInfo(sandbox_id=sandbox_id, runtime_url=sb.runtime_url, status="running")


class _FakeClient:
    def __init__(self, url: str, *_, **__):
        self._url = url

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def register_namespace(self, _ns: Any) -> None:
        return None

    async def remote(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _add(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    import agentix.gateway.dispatcher as disp

    monkeypatch.setattr(disp, "RuntimeClient", _FakeClient)

    dispatcher = Dispatcher(deployment=FakeDeployment())
    app = build_app(dispatcher, node_id="test-node")
    with TestClient(app) as c:
        yield c, dispatcher


def test_health_reports_node_and_stats(client) -> None:
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["node_id"] == "test-node"
    assert body["paused"] is False


def test_create_session_validation_errors(client) -> None:
    c, _ = client
    r = c.post("/sessions", json={"image": "x", "bundle": "y"})  # missing callable_ref
    assert r.status_code == 400
    assert "callable_ref" in r.json()["detail"]


def test_dispatch_and_poll_session_lifecycle(client) -> None:
    c, dispatcher = client
    spec = {
        "callable_ref": f"{__name__}::_add",
        "image": "task-image",
        "bundle": "test-bundle",
        "args": [2, 3],
        "metadata": {"instance_id": "i-1"},
    }
    r = c.post("/sessions", json=spec)
    assert r.status_code == 202
    body = r.json()
    sid = body["session_id"]
    assert body["status"] in {"queued", "init", "ready", "running", "postrun", "succeeded"}

    # Spin until terminal.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s = c.get(f"/sessions/{sid}").json()
        if s["status"] == "succeeded":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"session never succeeded: {s}")

    result = c.get(f"/sessions/{sid}/result").json()
    assert result["status"] == "succeeded"
    assert result["metadata"] == {"instance_id": "i-1"}


def test_pause_and_resume_endpoints(client) -> None:
    c, dispatcher = client
    assert c.post("/pause").json() == {"paused": True}
    assert dispatcher.paused is True
    assert c.post("/resume").json() == {"paused": False}
    assert dispatcher.paused is False


def test_records_endpoints(client) -> None:
    c, _ = client
    r = c.get("/records").json()
    assert "records" in r and isinstance(r["records"], list)


def test_session_not_found_returns_404(client) -> None:
    c, _ = client
    assert c.get("/sessions/nope").status_code == 404
    assert c.get("/sessions/nope/result").status_code == 404
