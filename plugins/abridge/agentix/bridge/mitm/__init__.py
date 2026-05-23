"""mitmproxy bridge package — backward-compatible public surface."""

from __future__ import annotations

from typing import Any

from agentix.bridge.actions import (
    apply_http_action,
    continue_action,
    error_action,
    json_envelope,
    respond_action,
    response_envelope,
)
from agentix.bridge.host.forwarder import HookForwarder
from agentix.bridge.mitm.cli import ADDON_SCRIPT, main, mitmdump_args
from agentix.bridge.mitm.hook_client import _send_hook_event_impl, hook_url, send_hook_event
from agentix.bridge.plugins.anthropic.handler import OpenAIForwarder
from agentix.bridge.runtime.lifecycle import start_proxy, stop_proxy
from agentix.bridge.types import NAMESPACE, ProxyHandle

_MITM_HTTP = None
_mitmdump_args = mitmdump_args
_response_envelope = response_envelope
_send_hook_event = _send_hook_event_impl
_hook_url = hook_url


def __getattr__(name: str) -> Any:
    if name in {
        "dns_request",
        "request",
        "response",
        "tcp_message",
        "udp_message",
        "websocket_message",
    }:
        from agentix.bridge.mitm import addon

        return getattr(addon, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADDON_SCRIPT",
    "HookForwarder",
    "NAMESPACE",
    "OpenAIForwarder",
    "ProxyHandle",
    "_MITM_HTTP",
    "_hook_url",
    "_mitmdump_args",
    "_response_envelope",
    "_send_hook_event",
    "apply_http_action",
    "continue_action",
    "dns_request",
    "error_action",
    "hook_url",
    "json_envelope",
    "main",
    "mitmdump_args",
    "request",
    "respond_action",
    "response",
    "response_envelope",
    "send_hook_event",
    "start_proxy",
    "stop_proxy",
    "tcp_message",
    "udp_message",
    "websocket_message",
]
