"""Forward — a schema-agnostic JSON POST handler for a sidecar URL.

The sandbox tunnel decodes an inbound JSON object and sends that object to
the host. `Forward` serializes it as a new JSON POST to `target_url + path`,
then buffers the complete sidecar response and returns its body, media type,
and HTTP status. Shape translation, recording, and routing can therefore live
*behind* `target_url` in any JSON service without teaching abridge those
schemas.

This is intentionally not an HTTP-transparent byte ferry: the original
method, query, headers, and JSON encoding are not preserved. An SSE response
keeps its media type and payload for compatibility, but is fully buffered;
there is no chunk streaming or backpressure on this wire contract.

    proxy = Proxy(Forward(sidecar_url, paths=["/v1/messages"]))
    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, env={"ANTHROPIC_BASE_URL": handle.url, ...})

`Forward` carries the rollout identity: it stamps `x-session-id` (the
rollout this forwarder serves) and a per-call `x-request-id` on every
upstream POST, so the sidecar can group a token-level trajectory.

Routing is dynamic — paths are chosen at construction, not via the
class-level `@on` tag — so `Forward` exposes them through
`abridge_routes()`, which `Proxy` merges alongside any `@on` handlers.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping

import httpx

from .proxy import AbridgeError, ClientResponse, Handler, Request

logger = logging.getLogger(__name__)


class Forward:
    """Forward JSON POSTs on `paths` to `target_url` over httpx.

    `session_id` identifies the rollout this forwarder serves (auto-gen if
    not passed); stamped as `x-session-id` on every upstream call. Reusing
    a `Forward` across sequential proxy sessions keeps that identity. A
    `Proxy` closes the forwarder's HTTP pool when its lifecycle stops; the
    pool is recreated lazily if the forwarder is used again. `headers` are
    merged into every upstream request (e.g. an upstream auth token the
    sidecar expects).

    Responses, including 4xx and 5xx responses, remain normal
    `ClientResponse` values so their status and body survive the tunnel.
    `AbridgeError(502)` is reserved for failures to obtain an HTTP response.
    """

    def __init__(
        self,
        target_url: str,
        *,
        paths: list[str],
        timeout: float = 600.0,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not paths:
            raise ValueError("Forward requires at least one path")
        self._target = target_url.rstrip("/")
        self._paths = tuple(paths)
        self.session_id = session_id or uuid.uuid4().hex
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(timeout=timeout)

    def abridge_routes(self) -> dict[str, Handler]:
        return {path: self._make_handler(path) for path in self._paths}

    def _make_handler(self, path: str) -> Handler:
        async def handler(request: Request) -> ClientResponse:
            return await self._forward(path, request)

        return handler

    async def _forward(self, path: str, request: Request) -> ClientResponse:
        record_id = uuid.uuid4().hex
        headers = {
            **self._headers,
            "x-session-id": self.session_id,
            "x-request-id": record_id,
            "content-type": "application/json",
        }
        url = self._target + path
        try:
            resp = await self._get_client().post(url, json=request.body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("abridge forward %s: %s", url, exc)
            raise AbridgeError(f"forward to {url}: {exc}", status_code=502) from exc

        media_type = resp.headers.get("content-type", "application/json").split(";")[0].strip()
        return ClientResponse(
            body=resp.content,
            media_type=media_type or "application/json",
            status_code=resp.status_code,
        )

    def _get_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self._timeout)
            self._client = client
        return client

    async def aclose(self) -> None:
        """Close the HTTP pool. Safe to call repeatedly and reusable later."""
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


__all__ = ["Forward"]
