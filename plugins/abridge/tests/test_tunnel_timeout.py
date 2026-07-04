"""Tunnel request-window configuration + upstream SDK retry policy.

The tunnel answers each agent request within `request_timeout`; the timer
starts before the host's upstream call does, so the window must strictly
cover the host client's worst case (timeout x (1 + max_retries)). These
tests pin the two halves: `Proxy` threads a configurable window into the
sandbox tunnel, and the bundled clients don't retry behind the operator's
back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from agentix.bridge import Proxy, on
from agentix.bridge.clients import AnthropicFromOpenAIClient, OpenAIClient
from agentix.bridge.proxy import TunnelHandle, _make_forwarder, _start_tunnel
from starlette.requests import Request as StarletteRequest


class _EchoClient:
    @on("/v1/echo")
    async def echo(self, request: Any) -> Any:
        raise AssertionError("never dispatched in these tests")


@dataclass
class _FakeSandbox:
    remote_calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)

    def register_namespace(self, ns: Any) -> None:
        pass

    async def remote(self, fn: Any, **kwargs: Any) -> Any:
        self.remote_calls.append((fn, kwargs))
        return TunnelHandle(url="http://127.0.0.1:1", port=1)


@pytest.mark.asyncio
async def test_proxy_start_threads_request_timeout_into_the_tunnel() -> None:
    sandbox = _FakeSandbox()
    proxy = Proxy(_EchoClient(), request_timeout=1234.5)
    await proxy.start(sandbox)  # type: ignore[arg-type]

    fn, kwargs = sandbox.remote_calls[0]
    assert fn is _start_tunnel
    assert kwargs["request_timeout"] == 1234.5


@pytest.mark.asyncio
async def test_proxy_default_window_matches_tunnel_default() -> None:
    sandbox = _FakeSandbox()
    await Proxy(_EchoClient()).start(sandbox)  # type: ignore[arg-type]
    assert sandbox.remote_calls[0][1]["request_timeout"] == 600.0


def test_proxy_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="request_timeout"):
        Proxy(_EchoClient(), request_timeout=0)


@pytest.mark.asyncio
async def test_forwarder_times_out_to_504_after_the_configured_window() -> None:
    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    # The real ns.request raises TimeoutError when the window elapses —
    # emulate that while capturing the window the forwarder threaded in.
    class _TimingOutNs:
        def __init__(self) -> None:
            self.seen_timeout: float | None = None

        async def request(self, path: str, body: Any, *, timeout: float) -> Any:
            self.seen_timeout = timeout
            raise TimeoutError

    ns = _TimingOutNs()
    forward = _make_forwarder(ns=ns, path="/v1/echo", request_timeout=0.05)  # type: ignore[arg-type]
    request = StarletteRequest(
        scope={"type": "http", "method": "POST", "headers": []}, receive=_receive
    )
    response = await forward(request)
    assert response.status_code == 504
    assert ns.seen_timeout == 0.05


@pytest.mark.asyncio
async def test_real_tunnel_returns_504_within_the_configured_window(monkeypatch) -> None:
    """End-to-end through a real in-sandbox tunnel: a host that never replies
    must 504 at ~request_timeout, not the 600s default — a regression that
    re-hardcodes the window during route registration would blow the budget
    here instead of staying green."""
    import time

    import httpx
    from agentix.bridge import proxy as proxy_mod

    monkeypatch.setattr(proxy_mod._get_namespace(), "emit", lambda *a, **k: _never())
    handle = await proxy_mod._start_tunnel(paths=["/v1/echo"], request_timeout=0.3)
    try:
        started = time.monotonic()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{handle.url}/v1/echo", json={}, timeout=5.0)
        elapsed = time.monotonic() - started
        assert resp.status_code == 504
        assert elapsed < 3.0  # bounded by request_timeout, nowhere near 600s
    finally:
        await proxy_mod._stop_tunnel(handle=handle)


async def _never() -> None:
    """An emit that drops the message, so the host reply never arrives."""


def test_bundled_clients_do_not_retry_behind_the_operators_back() -> None:
    """openai SDK default max_retries=2 silently multiplies handler occupancy
    behind the tunnel window; retry policy must be explicit."""
    anthropic = AnthropicFromOpenAIClient(base_url="http://up.example/v1", api_key="sk-test")
    assert anthropic._client.max_retries == 0  # noqa: SLF001

    openai_client = OpenAIClient(base_url="http://up.example/v1", api_key="sk-test")
    assert openai_client._sdk.max_retries == 0  # noqa: SLF001

    pure_anthropic = pytest.importorskip("agentix.bridge.clients.anthropic")
    passthrough = pure_anthropic.AnthropicClient(
        base_url="http://up.example", api_key="sk-ant-test"
    )
    assert passthrough._client.max_retries == 0  # noqa: SLF001

    tuned = AnthropicFromOpenAIClient(
        base_url="http://up.example/v1", api_key="sk-test", max_retries=2
    )
    assert tuned._client.max_retries == 2  # noqa: SLF001
