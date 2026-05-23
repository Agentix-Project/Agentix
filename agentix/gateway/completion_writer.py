"""Pluggable sink for captured completion records.

Mirrors `polar.gateway.completion_writer`: keep the in-memory
`RecordStore` as the source of truth, then optionally tee every
record into durable storage via a `CompletionWriter`.

The protocol is intentionally minimal — `write(record)` and `close()`.
That covers jsonl, parquet, kafka, ClickHouse, HF datasets, an OTel
log exporter, anything else. Two built-in implementations:

* `NullCompletionWriter` — drops records on the floor (default).
* `JsonlCompletionWriter` — appends one record per line to a file
  with thread-safe IO + atomic close.

Downstream gateways subclass `CompletionWriter` or pass a writer
into `Dispatcher` via the gateway node's `record_writer` argument.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CompletionWriter(Protocol):
    """Sink for `CompletionRecord` dicts.

    Implementations must be thread-safe — gateway nodes can fan
    several sessions' records into one writer concurrently.
    """

    def write(self, record: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullCompletionWriter:
    """Default: don't persist anything; rely on the in-memory store."""

    def write(self, record: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlCompletionWriter:
    """Append-one-record-per-line writer; flush on every write.

    `flush_every` lets callers trade durability for throughput:
    setting it to 100 batches the flush so the OS only fsyncs after
    100 records.
    """

    def __init__(self, path: str | Path, *, flush_every: int = 1) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._flush_every = flush_every
        self._since_flush = 0
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._fh.flush()
                self._since_flush = 0

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()


__all__ = ["CompletionWriter", "JsonlCompletionWriter", "NullCompletionWriter"]
