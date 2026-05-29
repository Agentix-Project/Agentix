"""Unit tests for the server↔worker length-prefixed msgpack framing.

The control pipe is the protocol foundation; these exercise the encode/
decode primitives directly — boundaries the e2e tests can't isolate:
empty frame, EOF mid-header vs mid-body, multiple frames in one buffer,
byte-exact payloads (args/results ride as pickle bytes), large frames.
"""

from __future__ import annotations

import asyncio
import os
import struct
from typing import cast

import pytest

import agentix.runtime.shared.framing as framing
from agentix.runtime.shared.framing import FrameTooLarge, pack_frame, read_frame, write_frame

pytestmark = pytest.mark.asyncio


def _reader(data: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    r.feed_data(data)
    r.feed_eof()
    return r


class _CaptureWriter:
    """Minimal StreamWriter stand-in capturing written bytes."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None


async def test_pack_then_read_round_trips():
    payload = {"type": "call", "call_id": "abc", "n": 7, "items": [1, "two", True]}
    frame = pack_frame(payload)
    assert (await read_frame(_reader(frame))) == payload


async def test_write_frame_then_read_round_trips():
    cap = _CaptureWriter()
    payload = {"type": "result", "call_id": "x1"}
    await write_frame(cast(asyncio.StreamWriter, cap), payload)
    assert (await read_frame(_reader(bytes(cap.buf)))) == payload


async def test_bytes_payload_is_byte_exact():
    # `arguments` / `value` ride as raw pickle bytes inside the frame.
    blob = os.urandom(4096)
    payload = {"call_id": "b", "value": blob}
    got = await read_frame(_reader(pack_frame(payload)))
    assert got is not None and got["value"] == blob


async def test_multiple_frames_in_one_buffer_read_in_order():
    a = {"type": "call", "call_id": "1"}
    b = {"type": "result", "call_id": "1"}
    reader = _reader(pack_frame(a) + pack_frame(b))
    assert (await read_frame(reader)) == a
    assert (await read_frame(reader)) == b
    assert (await read_frame(reader)) is None  # EOF after the last frame


async def test_eof_returns_none():
    assert (await read_frame(_reader(b""))) is None


async def test_truncated_header_returns_none():
    # Fewer than the 4 header bytes, then EOF.
    assert (await read_frame(_reader(b"\x01\x02"))) is None


async def test_truncated_body_raises_incomplete_read():
    # Header claims 10 bytes; only 4 are present before EOF.
    data = struct.pack("<I", 10) + b"\x00\x00\x00\x00"
    with pytest.raises(asyncio.IncompleteReadError):
        await read_frame(_reader(data))


async def test_zero_length_frame_decodes_to_empty_dict():
    assert (await read_frame(_reader(struct.pack("<I", 0)))) == {}


async def test_large_frame_round_trips():
    blob = os.urandom(2 * 1024 * 1024)  # 2 MiB — spans many read chunks
    got = await read_frame(_reader(pack_frame({"value": blob})))
    assert got is not None and got["value"] == blob


async def test_read_frame_rejects_oversized_declared_length():
    # A corrupt/desynced header must be rejected on the length alone, before
    # any attempt to read (and buffer) the bogus body.
    data = struct.pack("<I", framing.MAX_FRAME_BYTES + 1)
    with pytest.raises(FrameTooLarge):
        await read_frame(_reader(data))


async def test_pack_frame_rejects_oversized_body(monkeypatch: pytest.MonkeyPatch):
    # Shrink the cap so the guard is exercised without allocating 256 MiB.
    monkeypatch.setattr(framing, "MAX_FRAME_BYTES", 16)
    with pytest.raises(FrameTooLarge):
        pack_frame({"value": b"x" * 64})


async def test_read_frame_under_cap_still_round_trips(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(framing, "MAX_FRAME_BYTES", 4096)
    payload = {"type": "result", "call_id": "ok"}
    assert (await read_frame(_reader(pack_frame(payload)))) == payload
