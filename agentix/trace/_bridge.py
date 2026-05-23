"""Cross-process trace bridge over the `/trace` SIO namespace.

Worker side: a `Processor` translates Trace/Span lifecycle into events
emitted on the `/trace` namespace (`trace_start`, `trace_end`,
`span_start`, `span_end`).

Host side: `HostTraceNamespace` receives those events and replays them
against the host's local trace provider. `RuntimeClient` auto-registers
it on connect.

Reconnect safety: the bridge rides `agentix.sio.ReliableStream`, so
events carry monotonic `_seq`, the sandbox buffers them, and the host
emits `_resume` on (re)connect to recover anything missed.
"""

from __future__ import annotations

import contextvars
from typing import Any

import socketio

from agentix import sio as _sio
from agentix import trace

NAMESPACE = "/trace"

# Stamped by the runtime worker just before user code runs, so host-side
# consumers can correlate worker-emitted spans back to the originating
# `c.remote(...)` call.
DISPATCH_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentix_dispatch_call_id",
    default=None,
)


# ── worker side: trace.Processor → /trace events ──────────────────


class _WorkerTraceNamespace(_sio.Namespace):
    namespace = NAMESPACE


class _ForwardProcessor(trace.Processor):
    """Translates Trace/Span lifecycle into `/trace` SIO events."""

    def __init__(self, ns: _sio.Namespace) -> None:
        self._ns = ns
        self._stream = _sio.ReliableStream(ns)

    def on_trace_start(self, t: trace.Trace) -> None:
        self._emit(
            "trace_start",
            {
                "trace_id": t.trace_id,
                "call_id": DISPATCH_CALL_ID.get(),
                "name": t.name,
                "metadata": dict(t.metadata) if t.metadata else None,
                "started_at": t.started_at,
            },
        )

    def on_trace_end(self, t: trace.Trace) -> None:
        self._emit(
            "trace_end",
            {
                "trace_id": t.trace_id,
                "call_id": DISPATCH_CALL_ID.get(),
                "ended_at": t.ended_at,
            },
        )

    def on_span_start(self, s: trace.Span) -> None:
        self._emit("span_start", _span_payload(s))

    def on_span_end(self, s: trace.Span) -> None:
        self._emit("span_end", _span_payload(s, full=True))

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if not _sio._is_installed():
            return
        try:
            self._stream.emit_nowait(event, payload)
        except RuntimeError:
            pass


def _span_payload(s: trace.Span, *, full: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trace_id": s.trace_id,
        "call_id": DISPATCH_CALL_ID.get(),
        "span_id": s.span_id,
        "parent_id": s.parent_id,
        "name": s.name,
        "attrs": dict(s.attrs) if s.attrs else None,
        "started_at": s.started_at,
    }
    if full:
        payload["ended_at"] = s.ended_at
        payload["status"] = s.status
        payload["status_description"] = s.status_description
        if s.error is not None:
            payload["error"] = {
                "message": s.error.message,
                "data": dict(s.error.data) if s.error.data else None,
            }
        if s.events:
            payload["events"] = [
                {
                    "name": ev.name,
                    "timestamp": ev.timestamp,
                    "attributes": dict(ev.attributes) if ev.attributes else None,
                }
                for ev in s.events
            ]
    return payload


def install_worker_bridge() -> _ForwardProcessor:
    """Register the worker-side `/trace` namespace + forward processor."""
    ns = _WorkerTraceNamespace()
    _sio.register_namespace(ns)
    proc = _ForwardProcessor(ns)
    trace.add_processor(proc)
    return proc


# ── host side: receive /trace events, replay locally ──────────────


class HostTraceNamespace(socketio.AsyncClientNamespace):
    """Replays inbound `/trace` events into the host's local provider.

    Reconnect-safe: tracks last delivered `_seq`, emits `_ack` after
    each delivery, and emits `_resume` on every (re)connect.
    """

    def __init__(self) -> None:
        super().__init__(NAMESPACE)
        self._last_seq = 0

    async def trigger_event(self, event: str, *args: Any) -> Any:
        if event == "connect":
            await self._emit_resume()
            return
        if event in ("disconnect", "connect_error"):
            return

        # Server forwards events as msgpack-bytes via `pack(...)`.
        from agentix.runtime.client._sio_facade import _decode

        envelope = _decode(args[0]) if args else None
        if not isinstance(envelope, dict):
            return

        seq = envelope.get("_seq")
        payload = envelope.get("data")
        if not isinstance(seq, int) or not isinstance(payload, dict):
            # Legacy / malformed: best-effort dispatch, no ack.
            if isinstance(envelope, dict):
                _dispatch(event, envelope)
            return

        if seq <= self._last_seq:
            await self._emit_ack(seq)
            return
        self._last_seq = seq
        _dispatch(event, payload)
        await self._emit_ack(seq)

    async def _emit_resume(self) -> None:
        with _suppress():
            await self.emit(_sio._STREAM_RESUME_EVENT, _pack({"since_seq": self._last_seq}))

    async def _emit_ack(self, seq: int) -> None:
        with _suppress():
            await self.emit(_sio._STREAM_ACK_EVENT, _pack({"seq": seq}))


def _pack(data: Any) -> bytes:
    from agentix.runtime.shared.codec import pack as _msgpack

    return _msgpack(data)


def _suppress():
    import contextlib

    return contextlib.suppress(BaseException)


def _dispatch(event: str, frame: dict[str, Any]) -> None:
    provider = trace._provider
    if event == "trace_start":
        provider.fan_trace_start(
            trace.Trace(
                trace_id=str(frame.get("trace_id", "")),
                name=str(frame.get("name", "") or ""),
                metadata=dict(frame.get("metadata") or {}),
                started_at=frame.get("started_at"),
            )
        )
    elif event == "trace_end":
        provider.fan_trace_end(
            trace.Trace(
                trace_id=str(frame.get("trace_id", "")),
                name=str(frame.get("name", "") or ""),
                metadata=dict(frame.get("metadata") or {}),
                ended_at=frame.get("ended_at"),
            )
        )
    elif event in ("span_start", "span_end"):
        attrs = dict(frame.get("attrs") or {})
        call_id = frame.get("call_id")
        if call_id is not None:
            attrs.setdefault("call_id", call_id)
        s = trace.Span(
            span_id=str(frame.get("span_id", "") or ""),
            trace_id=str(frame.get("trace_id", "")),
            parent_id=frame.get("parent_id"),
            name=str(frame.get("name", "") or ""),
            attrs=attrs,
            started_at=frame.get("started_at"),
            ended_at=frame.get("ended_at"),
            status=frame.get("status") or "unset",
            status_description=frame.get("status_description"),
        )
        if frame.get("error"):
            err = frame["error"]
            s.error = trace.SpanError(
                message=str(err.get("message", "")),
                data=err.get("data"),
            )
        if frame.get("events"):
            s.events = [
                trace.SpanEvent(
                    name=str(ev.get("name", "")),
                    timestamp=str(ev.get("timestamp", "")),
                    attributes=dict(ev.get("attributes") or {}),
                )
                for ev in frame["events"]
            ]
        if event == "span_start":
            provider.fan_span_start(s)
        else:
            provider.fan_span_end(s)


__all__ = ["DISPATCH_CALL_ID", "HostTraceNamespace", "install_worker_bridge"]
