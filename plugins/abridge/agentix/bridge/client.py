"""Host-side OpenAI-compatible client + SIO consumer.

Counterpart to `proxy.py`. The proxy in the sandbox emits
`llm_call` requests on `/abridge` carrying an OpenAI Chat Completions
body; this module fires that body against a real OpenAI-compatible
endpoint (OpenAI, Azure OpenAI, vLLM, Together, Anyscale, …) and
returns the JSON response. It also receives fire-and-forget
`completion_record` events and pushes them into an `InMemoryStore`
that the user reads after the run.

Only the OpenAI Chat Completions family is wired today. Anthropic,
Gemini, etc. become host clients as they grow up — see ROADMAP.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from agentix import AsyncClientNamespace

from .detection import ApiFamily
from .proxy import NAMESPACE, REQUEST_EVENT
from .storage import CompletionRecord, InMemoryStore, TokenUsage

logger = logging.getLogger("agentix.bridge.client")


_OPENAI_CHAT_PATH = "/chat/completions"

UpstreamHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]
"""Optional async callable that can rewrite the body before it leaves the host."""


class OpenAICompatibleClient(AsyncClientNamespace):
    """Host SIO consumer that fires OpenAI Chat Completions requests.

    Parameters mirror the openai SDK: `base_url` is what the SDK calls
    its `base_url` argument (e.g. `https://api.openai.com/v1`,
    `http://vllm:8000/v1`); `api_key` is the bearer token; `model`
    optionally rewrites whatever model the sandbox sent us so different
    agents can share one upstream without code changes.

    `request_hook` is the one-shot extension point: a coroutine that
    sees the OpenAI body just before we hit the wire (think: inject
    `reasoning_effort`, swap `response_format`, etc.).

    Captured `CompletionRecord`s land in `store` for whatever the
    caller wants to do with them.
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
        store: InMemoryStore | None = None,
    ) -> None:
        super().__init__(NAMESPACE)
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._model = model
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._request_hook = request_hook
        self.store: InMemoryStore = store if store is not None else InMemoryStore()

    # ── SIO dispatch ───────────────────────────────────────────────────

    async def on_llm_call(self, envelope: dict[str, Any]) -> None:
        """Round-trip request handler for `llm_call`.

        The sandbox's `Namespace.request(...)` wraps the application
        payload as `{"request_id": <sio>, "data": <payload>}` and
        expects a reply on `llm_call:result` carrying
        `{"request_id": <sio>, "value": ...}` (or `llm_call:error`).
        We reply with success and put any upstream error inside `value`
        so the caller can branch without raising. This keeps capture
        records consistent in both paths.
        """
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

        upstream_body = await self._apply_overrides(upstream_body)

        try:
            response = await self._post_openai(upstream_body)
            await self._reply_value(sio_request_id, {"response": response})
        except httpx.HTTPStatusError as exc:
            text = exc.response.text[:1000]
            await self._reply_value(
                sio_request_id,
                _err_value(f"upstream {exc.response.status_code}: {text}", exc.response.status_code),
            )
        except Exception as exc:  # noqa: BLE001 - report any failure back to the agent
            await self._reply_value(sio_request_id, _err_value(f"{type(exc).__name__}: {exc}", 502))

    async def on_completion_record(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget capture sink. Stores into `self.store`.

        `emit()` from the sandbox `Namespace` does not wrap the
        payload in a `request_id/data` envelope; we receive the
        record dict directly.
        """
        try:
            record = _record_from_payload(payload)
        except Exception:
            logger.exception("abridge.host: failed to parse completion_record")
            return
        self.store.add(record)

    # ── helpers ────────────────────────────────────────────────────────

    async def _apply_overrides(self, body: dict[str, Any]) -> dict[str, Any]:
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
        return out

    async def _post_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{_OPENAI_CHAT_PATH}"
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        value = resp.json()
        if not isinstance(value, dict):
            raise ValueError("openai-compatible upstream returned non-object JSON")
        return value

    async def _reply_value(self, sio_request_id: str, value: dict[str, Any]) -> None:
        await self.emit(
            f"{REQUEST_EVENT}:result",
            {"request_id": sio_request_id, "value": value},
        )


def _err_value(message: str, status: int) -> dict[str, Any]:
    return {"error": {"message": message, "status_code": status}}


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
    )


__all__ = ["OpenAICompatibleClient", "UpstreamHook"]
