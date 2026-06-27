"""Forward — a protocol-blind handler that ferries calls to a sidecar URL.

This is the only "client" abridge needs in the redesigned architecture.
The agent's request body arrives over the tunnel; `Forward` POSTs it
verbatim to `target_url + path` and returns the response bytes verbatim.
All shape/translation/pretokenization lives *behind* `target_url` — a
`cc_convert` translator sidecar, a `tito` gateway, or any HTTP service —
so abridge core stays shape- and protocol-blind.

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
    """Forward `paths` to `target_url` over httpx, returning bytes verbatim.

    `session_id` identifies the rollout this forwarder serves (auto-gen if
    not passed); stamped as `x-session-id` on every upstream call. Reusing
    one `Forward` across multiple `Proxy` instances shares the session —
    the intended model. `headers` are merged into every upstream request
    (e.g. an upstream auth token the sidecar expects).
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
        self._client = httpx.AsyncClient(timeout=timeout)

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
            resp = await self._client.post(url, json=request.body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("abridge forward %s: %s", url, exc)
            raise AbridgeError(f"forward to {url}: {exc}", status_code=502) from exc

        media_type = resp.headers.get("content-type", "application/json").split(";")[0].strip()
        if resp.status_code >= 400:
            # Surface the sidecar's error in-band to the agent, preserving
            # its status. The body is the sidecar's error payload verbatim.
            raise AbridgeError(
                resp.text or f"sidecar {url} returned {resp.status_code}",
                status_code=resp.status_code,
            )
        return ClientResponse(
            body=resp.content,
            media_type=media_type or "application/json",
            status_code=resp.status_code,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["Forward"]
