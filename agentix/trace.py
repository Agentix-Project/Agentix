"""Trace emission with fan-out sinks.

Closure impls call `agentix.trace.emit(kind, payload)` to record one
event in the rollout's trace. Every registered **trace sink** receives
the event. The framework's runtime ships a sink that fans events out
over the Socket.IO `trace` channel; downstream observability sinks
(Sentry, OTel, Logfire, …) can register their own.

Sink registration uses the `agentix.trace_sink` entry-point group.
Each entry points at a `module:install` callable that the framework
invokes once at startup. The callable receives a *registrar* it can
call with its own sink function:

```python
# downstream agentix-trace-otel/agentix_trace_otel/__init__.py
from agentix.trace import register_sink

def install():
    def _sink(kind, payload, call_id, source):
        # emit OTel span / send to collector / etc.
        ...
    register_sink(_sink)
```

```toml
# downstream pyproject.toml
[project.entry-points."agentix.trace_sink"]
otel = "agentix_trace_otel:install"
```

Then `pip install agentix-trace-otel` is the entire wiring — the
runtime imports + calls `install()` at lifespan-startup and the sink
gets every event after that.

`call_id` correlates events to a specific rollout. The dispatcher pins
the active call_id into a contextvar before invoking each impl, so
`emit()` picks it up automatically — closures don't have to thread it
through their code.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Callable
from typing import Any, Final

from agentix._plugin import Registry
from agentix.idents import CallId, PackageName

logger = logging.getLogger("agentix.trace")

SinkFn = Callable[[str, dict[str, Any], CallId | None, PackageName | None], None]
"""A trace sink: `(kind, payload, call_id, source) -> None`. Sinks
should never raise (the framework swallows exceptions to keep tracing
from breaking a rollout), but the framework also defensively wraps
each call."""

InstallFn = Callable[[], None]
"""Entry-point callable. Called once at runtime startup; expected to
invoke `register_sink` zero or more times to hook its own sink(s) up."""

# In-process sink list. `register_sink` appends; emit() fans out across
# every sink. Sinks added via in-process calls (tests, ad-hoc) and via
# `agentix.trace_sink` entry-point installers live side by side.
_sinks: list[SinkFn] = []

# Entry-point registry — used by `_install_entry_point_sinks` at
# runtime startup. We don't store the install callables in `_sinks`
# because that list holds the *sinks*, not their installers.
_installers: Registry[InstallFn] = Registry("agentix.trace_sink")

_current_call_id: contextvars.ContextVar[CallId | None] = contextvars.ContextVar(
    "agentix_trace_call_id", default=None,
)
_current_source: contextvars.ContextVar[PackageName | None] = contextvars.ContextVar(
    "agentix_trace_source", default=None,
)


def register_sink(sink: SinkFn) -> None:
    """Add a trace sink. Receives every event emitted from any closure
    via `agentix.trace.emit(...)`.

    Sink errors are logged + swallowed (tracing must never break a
    rollout). Sinks are called in registration order.
    """
    _sinks.append(sink)


def unregister_sink(sink: SinkFn) -> None:
    """Remove a previously-registered sink. No-op if not present.

    Mostly used by tests to clean up after themselves; production
    sinks live for the runtime's lifetime.
    """
    try:
        _sinks.remove(sink)
    except ValueError:
        pass


def install_entry_point_sinks() -> None:
    """Invoke every `agentix.trace_sink` installer.

    Called once at runtime lifespan startup. Each installer is a
    `module:install` callable that registers its sink(s) via
    `register_sink`. Installer errors are caught and logged; one
    broken sink doesn't block the others or the runtime.
    """
    for name, installer in _installers.all().items():
        try:
            installer()
        except Exception as exc:
            logger.warning("trace sink installer %r failed: %s", name, exc)


def installers() -> Registry[InstallFn]:
    """The underlying registry — for `agentix plugins` and tests."""
    return _installers


def set_call_context(
    call_id: CallId | None,
    source: PackageName | None,
) -> tuple[contextvars.Token, contextvars.Token]:
    """Set the active call_id + source for trace events emitted while this
    context is on the call stack. Returns the contextvar reset tokens.
    """
    return _current_call_id.set(call_id), _current_source.set(source)


def reset_call_context(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    """Restore the call_id + source contextvars to their previous values."""
    cid_token, src_token = tokens
    _current_call_id.reset(cid_token)
    _current_source.reset(src_token)


def current_call_id() -> CallId | None:
    """The call_id pinned by the dispatcher for the current request, if any."""
    return _current_call_id.get()


def current_source() -> PackageName | None:
    """The closure package currently being dispatched, if any."""
    return _current_source.get()


def emit(
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    call_id: CallId | None = None,
    source: PackageName | None = None,
) -> None:
    """Record a single trace event. Fans out to every registered sink.

    `call_id` and `source` default to the dispatcher-set context. Closures
    should normally call `emit("kind", {...})` and let the runtime fill
    in the correlation. Sink errors are logged + swallowed — tracing
    must never break a rollout.
    """
    if not _sinks:
        return
    cid: Final = call_id if call_id is not None else _current_call_id.get()
    src: Final = source if source is not None else _current_source.get()
    pl = payload or {}
    for sink in _sinks:
        try:
            sink(kind, pl, cid, src)
        except Exception as exc:
            logger.warning("trace sink %r raised: %s", getattr(sink, "__name__", sink), exc)


def now() -> float:
    """Helper for callers that want to record their own timestamps."""
    return time.time()
