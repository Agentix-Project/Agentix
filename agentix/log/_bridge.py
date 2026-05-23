"""`/log` SIO namespace — worker handler + host replayer.

`/log` is a reconnect-safe stream: events carry monotonic `_seq`, the
sandbox buffers them until the host acks, and on reconnect the host
emits `_resume` to re-receive everything since its last ack. See
`agentix.sio.ReliableStream` for the wire envelope and contract.
"""

from __future__ import annotations

import logging
from typing import Any

import socketio

from agentix import sio as _sio
from agentix.log._config import LOG_CONTEXT_ATTR

NAMESPACE = "/log"
RECORD_EVENT = "record"


# ── worker side ───────────────────────────────────────────────────


class _WorkerLogNamespace(_sio.Namespace):
    namespace = NAMESPACE


_namespace_singleton: _WorkerLogNamespace | None = None
_stream_singleton: _sio.ReliableStream | None = None


def _get_worker_stream() -> _sio.ReliableStream:
    global _namespace_singleton, _stream_singleton
    if _namespace_singleton is None:
        _namespace_singleton = _WorkerLogNamespace()
        _sio.register_namespace(_namespace_singleton)
    if _stream_singleton is None:
        _stream_singleton = _sio.ReliableStream(_namespace_singleton)
    return _stream_singleton


class WorkerLogHandler(logging.Handler):
    """Translate `LogRecord`s into `/log:record` events.

    Records ride a `ReliableStream` so the host receives every record
    even across SIO disconnects, with FIFO ordering.

    Avoids self-recursion: `agentix.log` is excluded from forwarding to
    prevent feedback if our own debug logs were ever enabled.
    """

    _EXCLUDED_LOGGERS = ("agentix.sio", "agentix.log")

    def emit(self, record: logging.LogRecord) -> None:
        if any(record.name.startswith(prefix) for prefix in self._EXCLUDED_LOGGERS):
            return
        if not _sio._is_installed():
            return
        try:
            payload = _record_payload(record)
            stream = _get_worker_stream()
            stream.emit_nowait(RECORD_EVENT, payload)
        except Exception:
            self.handleError(record)


# Fields LogRecord defines natively; everything else on `record.__dict__`
# is treated as a user-provided `extra={...}` field and forwarded.
_STD_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        # Added to LogRecord in Python 3.12; absent on 3.11. Listed
        # unconditionally so a record produced on 3.12+ doesn't try to
        # smuggle `taskName` through `extra=` into a fresh record.
        "taskName",
        LOG_CONTEXT_ATTR,
    }
)


def _record_payload(record: logging.LogRecord) -> dict[str, Any]:
    extras = {k: v for k, v in record.__dict__.items() if k not in _STD_RECORD_KEYS and not k.startswith("_")}
    return {
        "name": record.name,
        "level": record.levelname,
        "levelno": record.levelno,
        "message": record.getMessage(),
        "created": record.created,
        "pathname": record.pathname,
        "lineno": record.lineno,
        "funcName": record.funcName,
        "module": record.module,
        "exc_text": record.exc_text
        or (logging.Formatter().formatException(record.exc_info) if record.exc_info else None),
        "stack_info": record.stack_info,
        LOG_CONTEXT_ATTR: getattr(record, LOG_CONTEXT_ATTR, None),
        "extras": extras or None,
    }


# ── host side ─────────────────────────────────────────────────────


class HostLogNamespace(socketio.AsyncClientNamespace):
    """Replays inbound `/log:record` events into the host's `logging` tree.

    Each forwarded record is dispatched against the same logger name it
    had in the sandbox, so existing host-side handlers/formatters pick it
    up naturally.

    Reconnect safety: tracks `_last_seq` per stream; on (re)connect emits
    `_resume {since_seq}` so the sandbox replays anything missed; after
    each delivery emits `_ack {seq}` so the sandbox can release its
    buffer.
    """

    def __init__(self) -> None:
        super().__init__(NAMESPACE)
        self._last_seq = 0

    async def trigger_event(self, event: str, *args: Any) -> Any:
        if event == "connect":
            # Initial connect AND every reconnect goes through here.
            await self._emit_resume()
            return
        if event in ("disconnect", "connect_error"):
            return
        if event != RECORD_EVENT:
            return

        from agentix.runtime.client._sio_facade import _decode

        envelope = _decode(args[0]) if args else None
        if not isinstance(envelope, dict):
            return

        seq = envelope.get("_seq")
        payload = envelope.get("data")
        if not isinstance(seq, int) or not isinstance(payload, dict):
            # Legacy / malformed payload — fall through without dedup.
            if isinstance(envelope, dict) and isinstance(payload, dict):
                _replay_record(payload)
            return

        if seq <= self._last_seq:
            # Duplicate from a resume + already-delivered race. Re-ack
            # so the sandbox can move on.
            await self._emit_ack(seq)
            return
        self._last_seq = seq
        _replay_record(payload)
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


def _replay_record(payload: dict[str, Any]) -> None:
    logger = logging.getLogger(str(payload.get("name", "agentix.sandbox")))
    levelno = int(payload.get("levelno", logging.INFO))
    if not logger.isEnabledFor(levelno):
        return
    # `makeRecord` rejects any `extra` key that collides with a standard
    # LogRecord attribute. Sender and receiver may run different Python
    # versions (the sandbox could add a field this version doesn't have,
    # or vice versa), so filter defensively rather than trusting the
    # sender's `_STD_RECORD_KEYS`.
    extras = {
        k: v for k, v in (payload.get("extras") or {}).items()
        if k not in _STD_RECORD_KEYS
    }
    record = logger.makeRecord(
        name=logger.name,
        level=levelno,
        fn=str(payload.get("pathname", "")),
        lno=int(payload.get("lineno", 0)),
        msg=str(payload.get("message", "")),
        args=(),
        exc_info=None,
        extra=extras,
    )
    record.funcName = str(payload.get("funcName", ""))
    record.module = str(payload.get("module", ""))
    if payload.get("exc_text"):
        record.exc_text = str(payload["exc_text"])
    if payload.get("stack_info"):
        record.stack_info = str(payload["stack_info"])
    if payload.get(LOG_CONTEXT_ATTR):
        setattr(record, LOG_CONTEXT_ATTR, str(payload[LOG_CONTEXT_ATTR]))
    logger.handle(record)


__all__ = ["HostLogNamespace", "WorkerLogHandler"]
