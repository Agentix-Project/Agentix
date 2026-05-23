"""Sandbox HTTP forwarder: mitmproxy hook → host SIO."""

from __future__ import annotations

import logging

from agentix.bridge.actions import error_action
from agentix.bridge.runtime.namespace import SandboxNamespace
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import agentix

logger = logging.getLogger("agentix.bridge.runtime")


def build_forwarder_app(*, ns: SandboxNamespace, request_timeout: float) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "abridge-mitm-forwarder"}

    @app.post("/hook")
    async def hook(request: Request) -> JSONResponse:
        event = await request.json()
        try:
            value = await ns.request("proxy_event", event, timeout=request_timeout)
        except agentix.RemoteSioError as exc:
            value = error_action(
                {"error": {"type": exc.type, "message": exc.message}},
                status_code=502,
            )
        except Exception as exc:
            logger.exception("abridge mitm forwarder failed")
            value = error_action(
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
                status_code=500,
            )
        if not isinstance(value, dict):
            value = error_action(
                {"error": {"type": "BadProxyResponse", "message": "host returned non-object response"}},
                status_code=502,
            )
        return JSONResponse(content=value)

    return app
