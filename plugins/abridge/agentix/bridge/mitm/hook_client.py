"""Sandbox → host hook transport."""

from __future__ import annotations

import os

import httpx
from agentix.bridge.actions import continue_action
from agentix.bridge.types import ProxyAction, ProxyEvent
from agentix.bridge.util import trace


def hook_url() -> str:
    return os.getenv("ABRIDGE_HOOK_URL", "")


def _send_hook_event_impl(event: ProxyEvent, *, hook_url: str) -> ProxyAction:
    timeout = float(os.getenv("ABRIDGE_FORWARD_TIMEOUT", "600"))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(hook_url, json=event)
    resp.raise_for_status()
    action = resp.json()
    if not isinstance(action, dict):
        raise ValueError("sandbox forwarder returned non-object JSON")
    return action


def send_hook_event(event: ProxyEvent, *, hook_url: str) -> ProxyAction:
    from agentix.bridge import mitm

    fn = getattr(mitm, "_send_hook_event", _send_hook_event_impl)
    return fn(event, hook_url=hook_url)


def send_hook_event_if_enabled(event: ProxyEvent) -> ProxyAction:
    url = hook_url()
    if not url:
        return continue_action()
    try:
        return send_hook_event(event, hook_url=url)
    except Exception as exc:
        trace("proxy_event.error", kind=event.get("kind"), error=repr(exc))
        return continue_action()
