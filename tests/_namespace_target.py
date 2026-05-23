"""Importable targets for namespace round-trip tests."""

from __future__ import annotations

import agentix


class _PluginService(agentix.Namespace):
    namespace = "/plugin-test"


_service: _PluginService | None = None


def _get() -> _PluginService:
    global _service
    if _service is None:
        _service = _PluginService()
        agentix.register_namespace(_service)
    return _service


async def echo_via_namespace(payload: dict) -> dict:
    """Send `payload` to the host on `/plugin-test:echo`, wait for the
    host's `:result` reply, and return its value."""
    svc = _get()
    return await svc.request("echo", payload, timeout=10.0)


async def fire_namespace_event(payload: dict) -> str:
    """Emit `slow` on `/plugin-test` and return immediately — does NOT
    wait for the host handler. Used to prove a slow host handler does
    not stall the runtime's other traffic."""
    svc = _get()
    await svc.emit("slow", payload)
    return "fired"


async def emit_log_line(message: str, level: str = "INFO") -> None:
    """Log via stdlib logging; the worker's log bridge ships it to host."""
    import logging

    logger = logging.getLogger("namespace_target")
    logger.log(getattr(logging, level), message)


async def emit_log_with_extra(message: str, **fields) -> None:
    """Log with `extra={...}` — verify extras survive the bridge."""
    import logging

    logger = logging.getLogger("namespace_target")
    logger.info(message, extra=fields)


async def emit_log_with_exception(message: str) -> None:
    """Log inside an except block — verify the traceback comes through."""
    import logging

    logger = logging.getLogger("namespace_target")
    try:
        raise ValueError("kaboom")
    except ValueError:
        logger.exception(message)


async def emit_formatted_log(template: str, *args) -> None:
    """`logger.info(template, *args)` — verify %-format-on-host = %-format-in-sandbox."""
    import logging

    logger = logging.getLogger("namespace_target")
    logger.info(template, *args)


async def emit_log_burst(prefix: str, count: int) -> int:
    """Emit `count` ordered log records on `namespace_target`. Returns
    the number actually emitted, used by reconnect tests to drive a
    bounded burst from the sandbox."""
    import logging

    logger = logging.getLogger("namespace_target")
    for i in range(count):
        logger.info(f"{prefix}-{i:03d}")
    return count
