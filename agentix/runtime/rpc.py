"""RPC framing for worker stdio.

Each frame on a worker's stdin/stdout is a length-prefixed JSON object:

  +--------+---------------------+
  | u32 LE | n bytes UTF-8 JSON  |
  +--------+---------------------+

Frame schemas (`{"type": "...", ...}` — extra fields per type):

  ─── runtime → worker ─────────────────────────────────────
    call         {call_id, method, args, kwargs, kind: "unary"|"stream"|"bidi"}
    bidi_in      {call_id, item}            — push input chunk to a bidi call
    bidi_end_in  {call_id}                   — close input side of a bidi call
    cancel       {call_id}                   — abort an in-flight call
    shutdown     {}                          — graceful exit; worker drains then exits

  ─── worker → runtime ─────────────────────────────────────
    ready        {package}                   — sent once after class binds OK
    boot_error   {error}                     — sent once if class fails to bind
    result       {call_id, value}            — unary success
    error        {call_id, error}            — unary failure or stream/bidi error
    stream_item  {call_id, value}            — one chunk of a streaming response
    stream_end   {call_id}                   — clean end of stream/bidi out
    trace        {kind, payload, call_id?, source?}   — namespace trace.emit()
    log          {level, name, message, timestamp}    — namespace logging records

`call_id` correlates request frames with their response frames.

JSON-encoded args/kwargs flow through unchanged from the wire — the
runtime doesn't deserialize them. Pydantic validation happens on both
ends: caller-side (host) encodes, worker-side decodes + validates with
the namespace's own pydantic + class definition.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any


def pack_frame(payload: dict[str, Any]) -> bytes:
    """Encode one frame: 4-byte LE length + UTF-8 JSON body."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one frame from `reader`. Returns None on EOF.

    Raises if the framing is corrupt — that means the worker died or sent
    malformed bytes, and the multiplexer's caller should treat the worker
    as poisoned.
    """
    header = await reader.readexactly(4) if reader.at_eof() is False else b""
    if not header:
        return None
    try:
        header = header if len(header) == 4 else header + await reader.readexactly(4 - len(header))
    except asyncio.IncompleteReadError:
        return None
    (n,) = struct.unpack("<I", header)
    if n == 0:
        return {}
    body = await reader.readexactly(n)
    return json.loads(body.decode("utf-8"))


async def write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Write one frame and flush. Safe to await from multiple tasks because
    each `write_frame` writes a complete frame in one call — but callers
    should serialize via a lock if interleaving multiple frames is a
    concern (the multiplexer uses a per-worker send lock for exactly that)."""
    writer.write(pack_frame(payload))
    await writer.drain()


__all__ = ["pack_frame", "read_frame", "write_frame"]
