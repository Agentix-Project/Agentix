"""Flow → protocol-neutral proxy events."""

from __future__ import annotations

import json
from typing import Any

from agentix.bridge.util import content_payload, message_text


def json_request_body(flow: Any) -> dict[str, Any]:
    raw = message_text(flow.request)
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object request body")
    return value


def http_request_event(flow: Any, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "http_request",
        "protocol": "http",
        "hook": "request",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "method": flow.request.method,
            "scheme": flow.request.scheme,
            "host": flow.request.pretty_host,
            "port": flow.request.port,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
            "json": body,
            **content_payload(getattr(flow.request, "content", b"")),
        },
    }


def http_response_event(flow: Any) -> dict[str, Any]:
    return {
        "kind": "http_response",
        "protocol": "http",
        "hook": "response",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "method": flow.request.method,
            "scheme": flow.request.scheme,
            "host": flow.request.pretty_host,
            "port": flow.request.port,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
        },
        "response": {
            "status_code": flow.response.status_code,
            "headers": dict(flow.response.headers),
            **content_payload(getattr(flow.response, "content", b"")),
        },
    }


def websocket_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "websocket_message",
        "protocol": "websocket",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "host": flow.request.pretty_host,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
        },
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **content_payload(getattr(msg, "content", b"")),
        },
    }


def tcp_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "tcp_message",
        "protocol": "tcp",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "client": str(getattr(flow.client_conn, "address", "")),
        "server": str(getattr(flow.server_conn, "address", "")),
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **content_payload(getattr(msg, "content", b"")),
        },
    }


def udp_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "udp_message",
        "protocol": "udp",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "client": str(getattr(flow.client_conn, "address", "")),
        "server": str(getattr(flow.server_conn, "address", "")),
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **content_payload(getattr(msg, "content", b"")),
        },
    }


def dns_request_event(flow: Any, names: list[str]) -> dict[str, Any]:
    return {
        "kind": "dns_request",
        "protocol": "dns",
        "hook": "request",
        "flow_id": str(getattr(flow, "id", "")),
        "questions": names,
    }
