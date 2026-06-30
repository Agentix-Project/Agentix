"""Session server — a FastAPI app over the native TITO engine, routing proxied
inference across a multi-backend pool.

The engine's `session_app` owns the routes (sessions + the token-aligned chat
flow); this module supplies the *backend*: a pooled httpx proxy that picks a
replica per request (sticky by ``session_id`` for prefix-cache locality), reports
a replica down on a transport error, and forgets a session's pin on delete.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .engine.session_app import setup_session_routes
from .pool import BackendPool

logger = logging.getLogger(__name__)

_HOP_BY_HOP = ("content-length", "transfer-encoding", "host")
_RESP_STRIP = ("content-length", "transfer-encoding", "content-encoding")


def _session_id_from_path(path: str) -> str | None:
    """Extract ``{session_id}`` from ``/sessions/{session_id}[/...]``."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "sessions":
        return parts[1]
    return None


class _PooledBackend:
    """Backend for the session routes: proxy each request to a pool-picked replica."""

    def __init__(self, args: Any, pool: BackendPool) -> None:
        self._pool = pool
        timeout = getattr(args, "router_timeout", 600.0)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024), timeout=httpx.Timeout(timeout)
        )

    async def do_proxy(self, request: Request, path: str, body: bytes | None = None) -> dict:
        session_id = _session_id_from_path(request.url.path)
        backend_url = self._pool.pick(session_id)
        url = f"{backend_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        if body is None:
            body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            self._pool.report_down(backend_url)
            logger.warning("pooled proxy transport error %s -> %s: %s", path, backend_url, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
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

    def build_proxy_response(self, result: dict) -> Response:
        content = result["response_body"]
        headers = {k: v for k, v in result["headers"].items() if k.lower() not in _RESP_STRIP}
        try:
            return JSONResponse(content=json.loads(content), status_code=result["status_code"], headers=headers)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(
                content=content,
                status_code=result["status_code"],
                headers=headers,
                media_type=headers.get("content-type", ""),
            )

    async def aclose(self) -> None:
        await self.client.aclose()


class SessionServer:
    """FastAPI session server backed by the native TITO engine + a BackendPool."""

    def __init__(self, args: Any, pool: BackendPool) -> None:
        self.args = args
        self.pool = pool
        self.backend_url = pool.backends[0]
        self.app = FastAPI()
        self._backend = _PooledBackend(args, pool)
        self.app.router.on_shutdown.append(self._backend.aclose)
        setup_session_routes(self.app, self._backend, args)
        self.app.middleware("http")(self._forget_on_delete)

    async def _forget_on_delete(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        if request.method == "DELETE" and response.status_code < 300:
            session_id = _session_id_from_path(request.url.path)
            if session_id is not None:
                self.pool.forget(session_id)
        return response
