"""Session server wrapper around the vendored Miles implementation.

Adds multi-backend routing on top of the vendored single-backend
``SessionServer`` WITHOUT modifying any vendored file: a thin subclass
overrides ``do_proxy`` to pick a backend from a :class:`BackendPool` per
request (sticky by ``session_id`` for prefix-cache locality), reports a
backend down on a transport error, and forgets a session's pin when the
session is deleted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from tito_gateway.pool import BackendPool

logger = logging.getLogger(__name__)

_HOP_BY_HOP = ("content-length", "transfer-encoding", "host")


def _session_id_from_path(path: str) -> str | None:
    """Extract ``{session_id}`` from ``/sessions/{session_id}[/...]``."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "sessions":
        return parts[1]
    return None


class SessionServer:
    """Wrapper for Miles' standalone FastAPI session server, routing proxied
    inference across a :class:`BackendPool`."""

    def __init__(self, args: Any, pool: BackendPool):
        from tito_gateway.vendor.miles_compat.rollout.session.session_server import (
            SessionServer as MilesSessionServer,
        )

        class _PooledSessionServer(MilesSessionServer):
            def __init__(self, args: Any, pool: BackendPool) -> None:
                self._pool = pool
                # Nominal backend_url for any vendored code that reads it; the
                # per-request route is chosen in `do_proxy` below.
                super().__init__(args, pool.backends[0])
                self.app.middleware("http")(self._forget_on_delete)

            async def do_proxy(self, request, path, body=None, headers=None) -> dict:  # type: ignore[override]
                session_id = _session_id_from_path(request.url.path)
                backend_url = self._pool.pick(session_id)
                url = f"{backend_url}/{path}"
                if request.url.query:
                    url = f"{url}?{request.url.query}"
                if body is None:
                    body = await request.body()
                if headers is None:
                    headers = dict(request.headers)
                headers = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
                try:
                    response = await self.client.request(request.method, url, content=body, headers=headers)
                except httpx.TransportError as exc:
                    # Mark this replica down so the session re-pins on its next
                    # request; surface the error to the agent unchanged.
                    self._pool.report_down(backend_url)
                    logger.warning("pooled proxy transport error %s -> %s: %s", path, backend_url, exc)
                    error_body = json.dumps(
                        {"error": f"backend transport error: {type(exc).__name__}: {exc}"}
                    ).encode()
                    return {
                        "request_body": body,
                        "response_body": error_body,
                        "status_code": 502,
                        "headers": {"content-type": "application/json"},
                    }
                content = await response.aread()
                return {
                    "request_body": body,
                    "response_body": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }

            async def _forget_on_delete(self, request, call_next):
                response = await call_next(request)
                if request.method == "DELETE" and response.status_code < 300:
                    session_id = _session_id_from_path(request.url.path)
                    if session_id is not None:
                        # Drop the sticky pin so `_assigned` doesn't grow without
                        # bound across a long-lived gateway.
                        self._pool.forget(session_id)
                return response

        self._impl = _PooledSessionServer(args, pool)
        self.args = args
        self.pool = pool
        self.backend_url = pool.backends[0]
        self.app = self._impl.app
