"""Length-prefixed msgpack framing for worker stdio.

Each frame on a worker's stdin/stdout is:

  +--------+-------------------+
  | u32 LE | n bytes msgpack   |
  +--------+-------------------+

The msgpack blob is a dict — see frame schemas below. `agentix.runtime.shared.codec`
handles encode/decode, including ext types for ndarray + pydantic models.

Frame schemas (`{"type": "...", ...}` — extra fields per type):

  ─── runtime → worker ─────────────────────────────────────
    call         {call_id, callable, arguments}   — start a call
    cancel       {call_id}                        — abort an in-flight call
    shutdown     {}                               — graceful exit; worker drains then exits

  ─── worker → runtime ─────────────────────────────────────
    ready        {}                               — sent once after worker startup
    boot_error   {error}                          — sent once if startup fails
    result       {call_id, value}                 — call succeeded (value is pickle bytes)
    error        {call_id, error}                 — call failed
    trace        {frame}                          — opaque side-channel payload (forwarded as-is)

`call_id` correlates request frames with their response frames.

`callable` is an import-path `RemoteCallable` string
(`module::qualname`); `arguments` is pickle.dumps((args, kwargs)); the
worker pickles the return value back into `value`.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from agentix.runtime.shared.codec import pack, unpack


def pack_frame(payload: dict[str, Any]) -> bytes:
    """Encode one frame: 4-byte LE length + msgpack body."""
    body = pack(payload)
    return struct.pack("<I", len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one frame from `reader`. Returns None on EOF."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    (n,) = struct.unpack("<I", header)
    if n == 0:
        return {}
    body = await reader.readexactly(n)
    return unpack(body)


async def write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Write one frame and flush. Callers serialize concurrent writes via
    a lock; each call writes a complete frame in one shot."""
    writer.write(pack_frame(payload))
    await writer.drain()


__all__ = ["pack_frame", "read_frame", "write_frame"]
