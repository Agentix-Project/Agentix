"""Tracing — OTel-style Trace + Span + SpanEvent over process-local pub/sub.

Three primitive types, mirroring the universal model used by
OpenTelemetry / Jaeger / Zipkin / OpenAI Agents:

  - `Trace`    — a workflow grouping (just trace_id + metadata)
  - `Span`     — one operation inside a trace; has start/end, parent,
                 attrs, status, error, and an attached list of events
  - `SpanEvent` — a point-in-time record attached to a span (logs,
                  cache misses, "received chunk", etc.)

Events ride with their span — `span.add_event(name, **attrs)` mutates
the span object; processors see the full event list when `on_span_end`
fires. This matches OTel exactly. For streaming visibility, open a
short sub-span instead of attaching an event (see `log(...)` helper).

Cross-process is not in this module. Transports
(worker→server frame pipe, server→host Socket.IO) are concrete
`Processor` implementations that live in `agentix.runtime.*`.
`agentix.utils.trace` itself never imports sockets, FastAPI, frames, or
anything else transport-shaped.

User surface (steady state):

  ```python
  from agentix import trace as t

  with t.trace("eval-cc-swe", split="verified"):
      with t.span("instance", id="django-11099") as s:
          with t.span("clean"):
              t.log("resetting workdir")
          with t.span("llm.request", model="gpt-4o") as llm:
              llm.add_event("first_chunk")
              llm.set_status("ok")
  ```
"""

from __future__ import annotations

import abc
import atexit as _atexit
import contextlib
import contextvars
import logging
import pkgutil
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

# Make `agentix.trace` a namespace-aware regular package so plugins
# can contribute siblings (e.g. `agentix.trace.otel` from
# `agentix-trace-otel`). Mirrors what `agentix/__init__.py` does for
# the top-level namespace.
__path__ = pkgutil.extend_path(__path__, __name__)
from typing import Any, Literal

__all__ = [
    # types
    "Trace",
    "Span",
    "SpanEvent",
    "SpanError",
    "SpanStatus",
    # context managers / helpers
    "trace",
    "span",
    "get_current_span",
    "get_current_trace",
    "current_span_id",
    "current_trace_id",
    # processor / exporter
    "Processor",
    "Exporter",
    "ConsoleProcessor",
    # provider control
    "add_processor",
    "remove_processor",
    "set_processors",
    "set_tracing_disabled",
    "get_processors",
    "force_flush",
    "shutdown",
    # ids
    "gen_trace_id",
    "gen_span_id",
]

SpanStatus = Literal["unset", "ok", "error"]


# ── ids ────────────────────────────────────────────────────────────


def gen_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def gen_span_id() -> str:
    return f"span_{uuid.uuid4().hex[:24]}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time() % 1) * 1e6):06d}Z"


# ── scope (contextvars) ────────────────────────────────────────────

_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "agentix_trace_current_trace",
    default=None,
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "agentix_trace_current_span",
    default=None,
)


def get_current_trace() -> Trace | None:
    return _current_trace.get()


def get_current_span() -> Span | None:
    return _current_span.get()


def current_trace_id() -> str | None:
    t = _current_trace.get()
    return t.trace_id if t is not None else None


def current_span_id() -> str | None:
    s = _current_span.get()
    return s.span_id if s is not None else None


# ── data types ─────────────────────────────────────────────────────


@dataclass
class SpanEvent:
    """A point-in-time record attached to a span. The OTel-standard
    way to record things like "received chunk", "cache miss", or a
    structured log line — anything that doesn't deserve a start+end
    pair of its own.

    Lives in `span.events`; visible to processors when the span ends.
    For streaming visibility, open a short sub-span instead.
    """

    name: str
    timestamp: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanError:
    """Structured error attached to a closing span. Distinct from
    `Span.status="error"` so callers can record both the boolean
    failure and the rich error context."""

    message: str
    data: dict[str, Any] | None = None


@dataclass
class Trace:
    """A workflow grouping. Just an id + metadata — the structure
    lives in the spans that share `trace_id`."""

    trace_id: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None

    def export(self) -> dict[str, Any]:
        return {
            "object": "trace",
            "id": self.trace_id,
            "name": self.name,
            "metadata": dict(self.metadata) if self.metadata else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class Span:
    """One operation inside a trace.

    `add_event(name, **attrs)` records a point-in-time event on this
    span — events ride to processors with `on_span_end`, not as
    separate callbacks. For streaming, open a short sub-span instead.
    """

    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None
    status: SpanStatus = "unset"
    status_description: str | None = None
    error: SpanError | None = None
    events: list[SpanEvent] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def set_attributes(self, **kwargs: Any) -> None:
        self.attrs.update(kwargs)

    def set_status(self, status: SpanStatus, description: str | None = None) -> None:
        self.status = status
        if description is not None:
            self.status_description = description

    def set_error(self, message: str, **data: Any) -> None:
        self.error = SpanError(message=message, data=data or None)
        if self.status == "unset":
            self.status = "error"
            self.status_description = message

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=_now_iso(),
                attributes=dict(attributes),
            )
        )

    def export(self) -> dict[str, Any]:
        return {
            "object": "span",
            "id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "attrs": dict(self.attrs) if self.attrs else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "status_description": self.status_description,
            "error": ({"message": self.error.message, "data": self.error.data} if self.error is not None else None),
            "events": (
                [
                    {
                        "name": e.name,
                        "timestamp": e.timestamp,
                        "attributes": dict(e.attributes) if e.attributes else None,
                    }
                    for e in self.events
                ]
                if self.events
                else None
            ),
        }


# ── processor + exporter ───────────────────────────────────────────


class Processor(abc.ABC):
    """Receives notifications for traces and spans.

    Callbacks fire synchronously from the call site of `trace()` /
    `span()` — keep them quick and thread-safe. Heavy work (network,
    disk) belongs off the calling thread; see `BatchProcessor`.

    SpanEvents are NOT a separate callback. They ride on the span via
    `span.events` and become visible at `on_span_end`. Open a short
    sub-span if you need streaming visibility for individual records.
    """

    def on_trace_start(self, t: Trace) -> None: ...
    def on_trace_end(self, t: Trace) -> None: ...
    def on_span_start(self, s: Span) -> None: ...
    def on_span_end(self, s: Span) -> None: ...

    def force_flush(self) -> None: ...
    def shutdown(self) -> None: ...


class Exporter(abc.ABC):
    """Batch sink for traces/spans. Pair with `BatchProcessor` to make
    one — the processor takes care of buffering + scheduling, the
    exporter takes a list and ships it somewhere."""

    @abc.abstractmethod
    def export(self, items: list[Trace | Span]) -> None: ...

    def shutdown(self) -> None: ...


# ── provider singleton ────────────────────────────────────────────


class _Provider:
    """The global trace dispatcher. Holds the processor list and
    fans out lifecycle callbacks.

    Lock-free dispatch over a snapshot — a misbehaving processor can't
    deadlock the rest. List mutations take the lock briefly."""

    def __init__(self) -> None:
        self._processors: list[Processor] = []
        self._lock = threading.RLock()
        self._disabled = False

    def add(self, p: Processor) -> None:
        with self._lock:
            if p not in self._processors:
                self._processors.append(p)

    def remove(self, p: Processor) -> None:
        with self._lock:
            try:
                self._processors.remove(p)
            except ValueError:
                pass

    def replace(self, ps: list[Processor]) -> None:
        with self._lock:
            self._processors = list(ps)

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = bool(disabled)

    def snapshot(self) -> list[Processor]:
        return list(self._processors)

    def fan_trace_start(self, t: Trace) -> None:
        if self._disabled:
            return
        for p in self.snapshot():
            try:
                p.on_trace_start(t)
            except Exception:
                _logger.exception("processor.on_trace_start raised")

    def fan_trace_end(self, t: Trace) -> None:
        if self._disabled:
            return
        for p in self.snapshot():
            try:
                p.on_trace_end(t)
            except Exception:
                _logger.exception("processor.on_trace_end raised")

    def fan_span_start(self, s: Span) -> None:
        if self._disabled:
            return
        for p in self.snapshot():
            try:
                p.on_span_start(s)
            except Exception:
                _logger.exception("processor.on_span_start raised")

    def fan_span_end(self, s: Span) -> None:
        if self._disabled:
            return
        for p in self.snapshot():
            try:
                p.on_span_end(s)
            except Exception:
                _logger.exception("processor.on_span_end raised")

    def force_flush(self) -> None:
        for p in self.snapshot():
            try:
                p.force_flush()
            except Exception:
                _logger.exception("processor.force_flush raised")

    def shutdown(self) -> None:
        for p in self.snapshot():
            try:
                p.shutdown()
            except Exception:
                _logger.exception("processor.shutdown raised")


_logger = logging.getLogger("agentix.utils.trace")
_provider = _Provider()


def add_processor(p: Processor) -> None:
    _provider.add(p)


def remove_processor(p: Processor) -> None:
    _provider.remove(p)


def set_processors(ps: list[Processor]) -> None:
    _provider.replace(ps)


def set_tracing_disabled(disabled: bool) -> None:
    _provider.set_disabled(disabled)


def get_processors() -> list[Processor]:
    return _provider.snapshot()


def force_flush() -> None:
    _provider.force_flush()


def shutdown() -> None:
    _provider.shutdown()


# ── user-facing context managers ──────────────────────────────────


@contextlib.contextmanager
def trace(name: str, *, trace_id: str | None = None, **metadata: Any) -> Iterator[Trace]:
    """Open a top-level workflow trace. Spans/events inside this block
    auto-attach via contextvar."""
    t = Trace(
        trace_id=trace_id or gen_trace_id(),
        name=name,
        metadata=dict(metadata),
        started_at=_now_iso(),
    )
    tok = _current_trace.set(t)
    _provider.fan_trace_start(t)
    try:
        yield t
    finally:
        t.ended_at = _now_iso()
        _provider.fan_trace_end(t)
        _current_trace.reset(tok)


@contextlib.contextmanager
def span(name: str, *, span_id: str | None = None, **attrs: Any) -> Iterator[Span]:
    """Open an operation span. Auto-parents to the current span via
    contextvar; auto-attached to the current trace.

    If no trace is open the span still works — it gets a synthetic
    `trace_id` and skips the trace_start/end callbacks. Recommended:
    wrap top-level work in `with trace(...)`."""
    t = _current_trace.get()
    parent = _current_span.get()
    s = Span(
        span_id=span_id or gen_span_id(),
        trace_id=t.trace_id if t is not None else "trace_unbound",
        parent_id=parent.span_id if parent is not None else None,
        name=name,
        attrs=dict(attrs),
        started_at=_now_iso(),
    )
    tok = _current_span.set(s)
    _provider.fan_span_start(s)
    try:
        yield s
    except Exception as exc:
        if s.error is None:
            s.set_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        s.ended_at = _now_iso()
        _provider.fan_span_end(s)
        _current_span.reset(tok)


# Built-in Processor implementations live in submodules. Re-exporting
# the most common one here lets `from agentix.utils.trace import ConsoleProcessor`
# work, but the implementation stays out of __init__.py so this file
# remains the core abstractions only.
from agentix.utils.trace.processors import ConsoleProcessor  # noqa: E402

_atexit.register(shutdown)
