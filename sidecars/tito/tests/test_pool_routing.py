"""Wiring tests for BackendPool routing in the SessionServer (no model/GPU).

Uses ``hf_checkpoint=None`` so the vendored session server skips tokenizer/route
setup — we drive the pool-aware ``do_proxy`` / forget hook directly.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_sglang_stub() -> None:
    """Stub `sglang...Tool` so the vendored template module imports without the
    sglang runtime (this routing test needs no model). Mirrors tito_experiment."""
    if "sglang" in sys.modules:
        return
    from typing import Any, Optional

    from pydantic import BaseModel

    class _Function(BaseModel):
        name: str
        description: Optional[str] = None
        parameters: Optional[dict[str, Any]] = None

    class Tool(BaseModel):
        type: str = "function"
        function: _Function

    names = ["sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai"]
    mods = {n: types.ModuleType(n) for n in names}
    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Tool = Tool  # type: ignore[attr-defined]
    mods["sglang.srt.entrypoints.openai"].protocol = protocol  # type: ignore[attr-defined]
    mods["sglang.srt.entrypoints"].openai = mods["sglang.srt.entrypoints.openai"]  # type: ignore[attr-defined]
    mods["sglang.srt"].entrypoints = mods["sglang.srt.entrypoints"]  # type: ignore[attr-defined]
    mods["sglang"].srt = mods["sglang.srt"]  # type: ignore[attr-defined]
    for n, m in mods.items():
        sys.modules[n] = m
    sys.modules["sglang.srt.entrypoints.openai.protocol"] = protocol


_install_sglang_stub()

from tito_gateway.pool import BackendPool  # noqa: E402
from tito_gateway.server import SessionServer, _session_id_from_path  # noqa: E402

A = "http://a:8000"
B = "http://b:8000"


def _args():
    return types.SimpleNamespace(hf_checkpoint=None, miles_router_timeout=600.0)


class _URL:
    def __init__(self, path: str, query: str = "") -> None:
        self.path = path
        self.query = query


class _Request:
    def __init__(self, path: str, method: str = "POST", body: bytes = b"{}") -> None:
        self.url = _URL(path)
        self.method = method
        self.headers = {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _Resp:
    def __init__(self, status: int = 200) -> None:
        self.status_code = status
        self.headers = {}

    async def aread(self) -> bytes:
        return b"{}"


def test_session_id_from_path():
    assert _session_id_from_path("/sessions/abc/v1/chat/completions") == "abc"
    assert _session_id_from_path("/sessions/xyz") == "xyz"
    assert _session_id_from_path("/health") is None
    assert _session_id_from_path("/") is None


@pytest.mark.asyncio
async def test_sticky_routing_pins_session(monkeypatch):
    pool = BackendPool([A, B], policy="sticky")
    srv = SessionServer(_args(), pool)
    seen: list[str] = []

    async def fake_request(method, url, content=None, headers=None):
        seen.append(url)
        return _Resp()

    monkeypatch.setattr(srv._impl.client, "request", fake_request)
    for _ in range(3):
        await srv._impl.do_proxy(_Request("/sessions/s1/v1/chat/completions"), "v1/chat/completions")
    # all three turns of one session hit the same backend (prefix-cache locality)
    assert len({u.split("/v1/")[0] for u in seen}) == 1


@pytest.mark.asyncio
async def test_transport_error_reports_backend_down(monkeypatch):
    import httpx

    pool = BackendPool([A, B], policy="sticky")
    srv = SessionServer(_args(), pool)

    async def boom(method, url, content=None, headers=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(srv._impl.client, "request", boom)
    result = await srv._impl.do_proxy(_Request("/sessions/s9/v1/chat/completions"), "v1/chat/completions")
    assert result["status_code"] == 502
    # the picked backend was marked down
    assert pool._down  # noqa: SLF001 - asserting routing side effect


@pytest.mark.asyncio
async def test_forget_on_delete_drops_pin():
    pool = BackendPool([A, B], policy="sticky")
    pool.pick("s2")
    assert "s2" in pool._assigned  # noqa: SLF001
    srv = SessionServer(_args(), pool)

    async def call_next(_req):
        return _Resp(status=204)

    await srv._impl._forget_on_delete(_Request("/sessions/s2", method="DELETE"), call_next)
    assert "s2" not in pool._assigned  # noqa: SLF001
