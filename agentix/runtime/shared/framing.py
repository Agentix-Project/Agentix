"""Length-prefixed msgpack framing for the worker control pipe.

Each frame on a worker's private runtime pipe is:

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
    sio_open     {namespace}                      — open a side-channel namespace
    sio_emit     {namespace, event, data}         — emit side-channel data

User stdout is intentionally not part of this byte stream. The worker runtime
captures fd 1 separately and forwards those lines as `/log` records so
`print()` cannot corrupt control framing.

`call_id` correlates request frames with their response frames.

`callable` is an import-path `RemoteCallable` string
(`module::qualname`); `arguments` is pickle.dumps((args, kwargs)); the
worker pickles the return value back into `value`.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from agentix.runtime.shared import MAX_MESSAGE_BYTES
from agentix.runtime.shared.codec import pack, unpack

# The control pipe carries the same pickled arguments / return values as the
# Socket.IO transport, which caps payloads at `MAX_MESSAGE_BYTES`. Enforce the
# same ceiling on the pipe so a corrupt or desynchronized 4-byte length prefix
# cannot drive an unbounded `readexactly` allocation (or an indefinite wait for
# bytes that never arrive). The `<I` prefix already tops out at ~4 GiB.
MAX_FRAME_BYTES = MAX_MESSAGE_BYTES


class FrameTooLarge(ValueError):
    """A frame's declared or encoded length exceeds `MAX_FRAME_BYTES`."""


def pack_frame(payload: dict[str, Any]) -> bytes:
    """Encode one frame: 4-byte LE length + msgpack body."""
    body = pack(payload)
    if len(body) > MAX_FRAME_BYTES:
        raise FrameTooLarge(f"frame body of {len(body)} bytes exceeds MAX_FRAME_BYTES={MAX_FRAME_BYTES}")
    return struct.pack("<I", len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one frame from `reader`. Returns None on EOF.

    Raises `FrameTooLarge` if the declared length exceeds `MAX_FRAME_BYTES`,
    and `ValueError` if the body does not decode to a dict — both indicate a
    desynced control pipe and are caught by the read loops (which then fail
    pending calls), never a silent giant allocation or a downstream
    `AttributeError` on `frame.get(...)`.
    """
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    (n,) = struct.unpack("<I", header)
    if n > MAX_FRAME_BYTES:
        raise FrameTooLarge(
            f"frame length {n} exceeds MAX_FRAME_BYTES={MAX_FRAME_BYTES}; control pipe desynchronized"
        )
    if n == 0:
        return {}
    body = await reader.readexactly(n)
    frame = unpack(body)
    if not isinstance(frame, dict):
        raise ValueError(
            f"frame body decoded to {type(frame).__name__}, expected dict; control pipe desynchronized"
        )
    return frame


async def write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Write one frame and flush. Callers serialize concurrent writes via
    a lock; each call writes a complete frame in one shot."""
    writer.write(pack_frame(payload))
    await writer.drain()


__all__ = ["MAX_FRAME_BYTES", "FrameTooLarge", "pack_frame", "read_frame", "write_frame"]
