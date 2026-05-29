"""Tests for the `agentix.quick_remote` one-call helper.

Drives the helper with stubbed deployment / session / client so it runs
without Docker or a real sandbox.
"""

from __future__ import annotations

import contextlib
from typing import Any

import agentix._quick as quick_mod
from agentix._quick import quick_remote


async def test_quick_remote_runs_fn_in_sandbox(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeBackend:
        pass

    def fake_load(name: str) -> type[FakeBackend]:
        calls["deployment"] = name
        return FakeBackend

    @contextlib.asynccontextmanager
    async def fake_session(backend: Any, config: Any):
        calls["config"] = config

        class _Sandbox:
            runtime_url = "http://localhost:1234"

        yield _Sandbox()

    class FakeClient:
        def __init__(self, url: str, timeout: float = 300) -> None:
            calls["url"] = url
            calls["timeout"] = timeout

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def remote(self, fn: Any, *args: Any, **kwargs: Any) -> str:
            calls["remote"] = (fn, args, kwargs)
            return "RESULT"

    monkeypatch.setattr(quick_mod, "load_deployment", fake_load)
    monkeypatch.setattr(quick_mod, "session", fake_session)
    monkeypatch.setattr(quick_mod, "RuntimeClient", FakeClient)

    def my_fn(x: int) -> int:
        return x

    out = await quick_remote(my_fn, 1, bundle="/b", image="img", deployment="docker", timeout=99, y=2)

    assert out == "RESULT"
    assert calls["deployment"] == "docker"
    assert calls["config"].image == "img"
    assert calls["config"].bundle == "/b"
    assert calls["url"] == "http://localhost:1234"
    assert calls["timeout"] == 99
    assert calls["remote"] == (my_fn, (1,), {"y": 2})
