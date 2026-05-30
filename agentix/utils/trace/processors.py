"""Built-in `agentix.utils.trace.Processor` implementations.

The abstract `Processor` and `Exporter` live in the package `__init__`
because they're part of the core abstraction. Concrete implementations
(console pretty-printer, future BatchProcessor, future exporters)
live here so the core file stays small.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from agentix.utils.trace import Processor, Span, Trace


class ConsoleProcessor(Processor):
    """Pretty-print every trace/span lifecycle event to stderr,
    indented by parent depth. Useful for local dev / smoke tests."""

    def __init__(self, *, stream: Any = None) -> None:
        self._stream = stream or sys.stderr
        # Track per-trace span_id → depth so children indent.
        self._depth: dict[str, int] = {}

    def _write(self, line: str) -> None:
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            pass

    def on_trace_start(self, t: Trace) -> None:
        self._depth[t.trace_id] = 0
        self._write(f"[trace.start] {t.name} ({t.trace_id[:14]}…) {t.metadata or ''}".rstrip())

    def on_trace_end(self, t: Trace) -> None:
        self._write(f"[trace.end]   {t.name} ({t.trace_id[:14]}…)")
        self._depth.pop(t.trace_id, None)

    def on_span_start(self, s: Span) -> None:
        depth = self._depth.get(s.trace_id, 0)
        indent = "  " * depth
        attrs = " ".join(f"{k}={v!r}" for k, v in s.attrs.items())
        self._write(f"{indent}[span.start] {s.name}  {attrs}".rstrip())
        # Children of this span indent one further.
        self._depth[s.span_id] = depth + 1

    def on_span_end(self, s: Span) -> None:
        self._depth.pop(s.span_id, None)
        depth = self._depth.get(s.trace_id, 0)
        if s.parent_id is not None:
            depth += 1
        indent = "  " * (depth - 1 if depth > 0 else 0)
        status = f" status={s.status}" if s.status != "unset" else ""
        ev = f" events={len(s.events)}" if s.events else ""
        err = f" error={s.error.message!r}" if s.error else ""
        self._write(f"{indent}[span.end]   {s.name}{status}{ev}{err}")


class JsonlProcessor(Processor):
    """Append every span (and trace) to a JSON Lines file — one
    `Span.export()` / `Trace.export()` per line.

    The batteries-included sink for rollout-data collection: with no
    processor registered, `/trace` data is silently dropped, so the marquee
    "collect the rollout" workflow otherwise means hand-rolling this class.
    Prefer the `trace.collect(path)` helper, which constructs one, registers
    it, and returns it.

    Records share `trace_id` (the `trace_id` field on spans, `id` on the
    trace), so a whole rollout's spans group by it. Writes are flushed per
    record and guarded by a lock — `on_span_end` can fire from multiple
    threads (sync remote callables run in `asyncio.to_thread`). The file is
    opened in append mode; pass a fresh `path` per run for isolation.
    """

    def __init__(self, path: str | Path, *, spans: bool = True, traces: bool = True) -> None:
        self._path = Path(path)
        if self._path.parent != Path():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._spans = spans
        self._traces = traces
        self._lock = threading.Lock()
        self._file = self._path.open("a", encoding="utf-8")

    def _write(self, record: dict[str, Any]) -> None:
        # `default=str` keeps a stray non-JSON value in span.attrs from
        # sinking the whole record — best-effort capture beats a crash.
        line = json.dumps(record, default=str)
        with self._lock:
            if self._file.closed:
                return
            self._file.write(line + "\n")
            self._file.flush()

    def on_span_end(self, s: Span) -> None:
        if self._spans:
            self._write(s.export())

    def on_trace_end(self, t: Trace) -> None:
        if self._traces:
            self._write(t.export())

    def force_flush(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    def shutdown(self) -> None:
        """Flush and close the file. Idempotent. Spans that end afterward are
        dropped (the sink is closed); call `trace.remove_processor(self)` too
        if you want to stop receiving them entirely."""
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


__all__ = ["ConsoleProcessor", "JsonlProcessor"]
