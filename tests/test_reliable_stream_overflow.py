"""`ReliableStream` surfaces (rather than silently swallows) buffer overflow."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agentix import sio as _sio


@pytest.fixture(autouse=True)
def _isolated_bridge():
    sent: list[dict[str, Any]] = []
    _sio._bridge._namespaces = {}
    _sio._bridge.install(sent.append)
    try:
        yield sent
    finally:
        _sio._bridge._send = None
        _sio._bridge._namespaces = {}


class _Stream(_sio.Namespace):
    namespace = "/cap"


def test_overflow_drops_oldest_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    stream = _sio.ReliableStream(_Stream(), max_buffer=3)
    with caplog.at_level(logging.WARNING, logger="agentix.sio"):
        for i in range(5):
            stream.emit_nowait("record", {"i": i})

    # The deque is capped; the two oldest unacked events were evicted.
    assert len(stream._buffer) == 3
    assert stream._dropped == 2
    assert any("buffer full" in record.getMessage() for record in caplog.records)
