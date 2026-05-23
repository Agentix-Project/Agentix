"""mitmproxy addon hooks."""

from __future__ import annotations

from typing import Any

from agentix.bridge.events import (
    dns_request_event,
    http_response_event,
    tcp_message_event,
    udp_message_event,
    websocket_message_event,
)
from agentix.bridge.mitm.dispatcher import dispatch_http_request
from agentix.bridge.mitm.hook_client import send_hook_event_if_enabled
from agentix.bridge.util import trace


def request(flow: Any) -> None:
    dispatch_http_request(flow)


def response(flow: Any) -> None:
    if flow.response is None:
        return
    trace(
        "http.response",
        host=flow.request.pretty_host,
        path=flow.request.path,
        status_code=flow.response.status_code,
    )
    send_hook_event_if_enabled(http_response_event(flow))


def websocket_message(flow: Any) -> None:
    msg = flow.websocket.messages[-1]
    trace(
        "websocket.message",
        host=flow.request.pretty_host,
        path=flow.request.path,
        bytes=len(msg.content),
    )
    send_hook_event_if_enabled(websocket_message_event(flow, msg))


def tcp_message(flow: Any) -> None:
    msg = flow.messages[-1]
    trace(
        "tcp.message",
        client=str(getattr(flow.client_conn, "address", "")),
        server=str(getattr(flow.server_conn, "address", "")),
        bytes=len(msg.content),
    )
    send_hook_event_if_enabled(tcp_message_event(flow, msg))


def udp_message(flow: Any) -> None:
    msg = flow.messages[-1]
    trace(
        "udp.message",
        client=str(getattr(flow.client_conn, "address", "")),
        server=str(getattr(flow.server_conn, "address", "")),
        bytes=len(msg.content),
    )
    send_hook_event_if_enabled(udp_message_event(flow, msg))


def dns_request(flow: Any) -> None:
    questions = getattr(flow.request, "questions", None) or []
    names = [str(getattr(q, "name", "")) for q in questions]
    trace("dns.request", names=names)
    send_hook_event_if_enabled(dns_request_event(flow, names))
