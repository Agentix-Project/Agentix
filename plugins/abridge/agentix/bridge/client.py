"""Host-side abridge: the `/abridge` consumer + a pluggable upstream `Client`.

The proxy in the sandbox emits `llm_call` requests on `/abridge` carrying an
OpenAI Chat Completions body. `Bridge` ferries that to the host, calls a
user-supplied `Client` to reach the actual model provider, returns the
response, and captures every call into an `InMemoryStore`.

`Bridge` itself knows nothing about endpoints, keys, or models — it just calls
`client.complete(body)`. The default `OpenAIClient` posts to an OpenAI-compatible
endpoint; swap in your own (litellm, an SGLang gateway, a replay client) by
implementing the `Client` protocol.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

import httpx

from agentix import AsyncClientNamespace
from agentix.utils import trace

from .detection import ApiFamily
from .proxy import NAMESPACE, REQUEST_EVENT, ProxyHandle
from .proxy import start_proxy as _sandbox_start_proxy
from .storage import CompletionRecord, InMemoryStore, TokenUsage

logger = logging.getLogger("agentix.bridge.client")

_OPENAI_CHAT_PATH = "/chat/completions"

UpstreamHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]
"""Optional callable that rewrites the body before it leaves the host."""


class UpstreamError(Exception):
    """Raised by a `Client` when the provider call fails. `Bridge` maps it to an
    error reply (carrying `status_code`) that the sandbox agent receives in place
    of a result — so a failed call doesn't crash the bridge."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@runtime_checkable
class Client(Protocol):
    """Sends an OpenAI Chat Completions body to a model provider and returns the
    OpenAI-format response dict. `Bridge` owns transport + capture; the `Client`
    owns the actual provider call. `headers` carry the rollout identity
    (`x-session-id` / `x-request-id`) for gateways that group by it. Raise
    `UpstreamError(msg, status_code=...)` on provider failure."""

    async def complete(
        self, body: dict[str, Any], *, headers: dict[str, str] | None = None
    ) -> dict[str, Any]: ...


class OpenAIClient:
    """Default `Client`: POST the body to an OpenAI-compatible `/chat/completions`.

    `model` rewrites the body's model so different agents share one upstream;
    `extra_body` is merged in; `request_hook` rewrites the body before the wire.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
        request_hook: UpstreamHook | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._model = model
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._request_hook = request_hook

    async def complete(
        self, body: dict[str, Any], *, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        out = dict(body)
        if self._model:
            out["model"] = self._model
        if self._extra_body:
            out.update(self._extra_body)
        if self._request_hook is not None:
            result = self._request_hook(out)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                out = result

        request_headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
            **(headers or {}),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}{_OPENAI_CHAT_PATH}", headers=request_headers, json=out)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"upstream {resp.status_code}: {resp.text[:1000]}"
            raise UpstreamError(msg, status_code=resp.status_code) from exc
        value = resp.json()
        if not isinstance(value, dict):
            raise UpstreamError("openai-compatible upstream returned non-object JSON")
        return value


class Bridge(AsyncClientNamespace):
    """Host-side abridge: the `/abridge` consumer + in-sandbox proxy lifecycle +
    capture. Bridge only ferries and captures — it calls `client.complete(...)`
    to reach the model provider.

    `await bridge.start_proxy(sandbox)` registers + brings up the in-sandbox
    service; point your agent's `base_url` at `bridge.get_base_url()`. Every
    captured call lands in `store`, grouped by `session_id`.
    """

    def __init__(
        self,
        client: Client,
        *,
        store: InMemoryStore | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__(NAMESPACE)
        self._client = client
        self.store: InMemoryStore = store if store is not None else InMemoryStore()
        self.session_id = session_id or uuid.uuid4().hex
        self._proxy: ProxyHandle | None = None
        self._family = "anthropic"

    # ── in-sandbox proxy lifecycle ─────────────────────────────────────

    async def start_proxy(self, sandbox: Any, family: str = "anthropic") -> str:
        """Register this bridge on the sandbox and start the in-sandbox abridge
        proxy; returns the base URL the agent should point at (sandbox loopback).

        Registration happens here (before the proxy's `remote` call — the first
        on this sandbox — so the connection plan still sees it), so the caller
        does not `register_namespace` separately. The proxy lives on the worker's
        event loop for the sandbox's lifetime; calls captured through it are
        grouped under this Bridge's `session_id`.
        """
        sandbox.register_namespace(self)
        # Capture the active span so the in-sandbox proxy can parent every LLM
        # span to it — the whole rollout then shows as one nested trace.
        parent = trace.get_current_span()
        self._proxy = await sandbox.remote(
            _sandbox_start_proxy,
            session_id=self.session_id,
            parent_trace_id=parent.trace_id if parent is not None else None,
            parent_span_id=parent.span_id if parent is not None else None,
        )
        self._family = family
        return self.get_base_url()

    def get_base_url(self) -> str:
        """The sandbox-loopback URL the agent's SDK should use as its
        Anthropic/OpenAI base URL. Requires `start_proxy(...)` first."""
        if self._proxy is None:
            raise RuntimeError("call await bridge.start_proxy(sandbox) first")
        if self._family == "anthropic":
            return self._proxy.anthropic_base_url
        return self._proxy.openai_base_url

    # ── SIO dispatch ───────────────────────────────────────────────────

    async def on_llm_call(self, envelope: dict[str, Any]) -> None:
        """Round-trip handler for `llm_call`: hand the body to the client, reply
        with the response (or an error `value` so the agent can branch without
        the bridge raising)."""
        if not isinstance(envelope, dict):
            return
        sio_request_id = envelope.get("request_id")
        if not isinstance(sio_request_id, str):
            logger.warning("abridge.host: dropped llm_call with no SIO request_id")
            return

        payload = envelope.get("data") or {}
        if not isinstance(payload, dict):
            await self._reply_value(sio_request_id, _err_value("payload was not a JSON object", 400))
            return

        upstream_body = payload.get("upstream_body") or {}
        if not isinstance(upstream_body, dict):
            await self._reply_value(sio_request_id, _err_value("upstream_body must be a JSON object", 400))
            return

        forward_headers = _trace_headers(payload)
        try:
            response = await self._client.complete(upstream_body, headers=forward_headers)
            await self._reply_value(sio_request_id, {"response": response})
        except UpstreamError as exc:
            await self._reply_value(sio_request_id, _err_value(exc.message, exc.status_code))
        except Exception as exc:  # noqa: BLE001 - report any failure back to the agent
            await self._reply_value(sio_request_id, _err_value(f"{type(exc).__name__}: {exc}", 502))

    async def on_completion_record(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget capture sink. `emit()` from the sandbox `Namespace`
        does not wrap the payload in a `request_id/data` envelope; we receive
        the record dict directly."""
        try:
            record = _record_from_payload(payload)
        except Exception:
            logger.exception("abridge.host: failed to parse completion_record")
            return
        self.store.add(record)

    async def _reply_value(self, sio_request_id: str, value: dict[str, Any]) -> None:
        await self.emit(f"{REQUEST_EVENT}:result", {"request_id": sio_request_id, "value": value})


def _err_value(message: str, status: int) -> dict[str, Any]:
    return {"error": {"message": message, "status_code": status}}


def _trace_headers(payload: dict[str, Any]) -> dict[str, str]:
    """Headers that propagate the rollout identity to the client/provider.

    OpenAI/OpenRouter ignore unknown headers; a separate gateway (e.g. an SGLang
    wrapper) can group its own token-level trajectory by `session_id`. abridge
    just forwards the ids it stamped.
    """
    headers: dict[str, str] = {}
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        headers["x-session-id"] = session_id
    record_id = payload.get("record_id")
    if isinstance(record_id, str) and record_id:
        headers["x-request-id"] = record_id
    return headers


def _record_from_payload(payload: dict[str, Any]) -> CompletionRecord:
    """Reverse the `CompletionRecord.to_dict()` round trip."""
    usage_payload = payload.get("usage") or {}
    if not isinstance(usage_payload, dict):
        usage_payload = {}
    usage = TokenUsage(
        prompt_tokens=int(usage_payload.get("prompt_tokens") or 0),
        completion_tokens=int(usage_payload.get("completion_tokens") or 0),
        cached_tokens=int(usage_payload.get("cached_tokens") or 0),
        reasoning_tokens=int(usage_payload.get("reasoning_tokens") or 0),
        total_tokens=int(usage_payload.get("total_tokens") or 0),
    )
    family = ApiFamily(payload.get("family", ApiFamily.UNKNOWN.value))
    request_id = payload.get("request_id") or payload.get("record_id")
    if not request_id:
        raise KeyError("completion_record missing request_id")
    session_id = payload.get("session_id")
    return CompletionRecord(
        request_id=str(request_id),
        family=family,
        started_at=float(payload["started_at"]),
        ended_at=float(payload["ended_at"]),
        request_path=str(payload.get("request_path", "")),
        request_body=dict(payload.get("request_body") or {}),
        upstream_body=dict(payload.get("upstream_body") or {}),
        response_body=payload.get("response_body"),
        status=str(payload.get("status", "ok")),
        error=payload.get("error"),
        usage=usage,
        session_id=str(session_id) if session_id is not None else None,
    )


__all__ = ["Bridge", "Client", "OpenAIClient", "UpstreamError", "UpstreamHook"]
