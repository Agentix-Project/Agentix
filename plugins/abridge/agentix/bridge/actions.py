"""Proxy action construction and HTTP flow application."""

from __future__ import annotations

import base64
import json
from typing import Any

from agentix.bridge.types import ProxyAction, ResponseEnvelope


def continue_action() -> dict[str, str]:
    return {"action": "continue"}


def respond_action(response: ResponseEnvelope) -> ProxyAction:
    return {"action": "respond", "response": response}


def error_action(body: dict[str, Any], *, status_code: int = 502) -> ProxyAction:
    return respond_action(json_envelope(body, status_code=status_code)) | {"action": "error"}


def response_envelope(
    status_code: int,
    body: bytes,
    *,
    content_type: str = "text/plain",
) -> ResponseEnvelope:
    return {
        "status_code": status_code,
        "headers": {
            "content-type": content_type,
            "content-length": str(len(body)),
        },
        "body_base64": base64.b64encode(body).decode(),
    }


def json_envelope(body: dict[str, Any], *, status_code: int = 200) -> ResponseEnvelope:
    payload = json.dumps(body, separators=(",", ":")).encode()
    return response_envelope(status_code, payload, content_type="application/json")


def apply_response_envelope(flow: Any, envelope: ResponseEnvelope) -> None:
    from agentix.bridge.mitm.flow import set_response

    raw_body = envelope.get("body_base64")
    if isinstance(raw_body, str):
        body = base64.b64decode(raw_body)
    else:
        body = str(envelope.get("body", "")).encode()

    headers = envelope.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "application/octet-stream")
    status_code = int(envelope.get("status_code") or 200)
    set_response(flow, status_code, body, content_type=content_type)
    for key, value in headers.items():
        if str(key).lower() == "content-type":
            continue
        flow.response.headers[str(key)] = str(value)


def apply_http_action(flow: Any, action: ProxyAction) -> bool:
    from agentix.bridge.mitm.flow import set_response

    if not isinstance(action, dict):
        return False
    name = action.get("action")
    if name == "respond":
        response = action.get("response")
        if not isinstance(response, dict):
            set_response(flow, 502, b"bad respond action from abridge host")
            return True
        apply_response_envelope(flow, response)
        return True
    if name == "kill":
        flow.kill()
        return True
    if name == "error":
        response = action.get("response")
        if isinstance(response, dict):
            apply_response_envelope(flow, response)
        else:
            set_response(flow, int(action.get("status_code") or 502), b"abridge hook error")
        return True
    return False
