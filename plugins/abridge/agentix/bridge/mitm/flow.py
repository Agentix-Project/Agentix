"""mitmproxy flow helpers."""

from __future__ import annotations

import importlib
from typing import Any

_MITM_HTTP: Any | None = None


def mitm_http() -> Any:
    global _MITM_HTTP
    from agentix.bridge import mitm as mitm_mod

    patched = getattr(mitm_mod, "_MITM_HTTP", None)
    if patched is not None:
        return patched
    if _MITM_HTTP is None:
        _MITM_HTTP = importlib.import_module("mitmproxy.http")
    return _MITM_HTTP


def set_response(flow: Any, status_code: int, body: bytes, *, content_type: str = "text/plain") -> None:
    http = mitm_http()
    flow.response = http.Response.make(status_code, body, {"content-type": content_type})
