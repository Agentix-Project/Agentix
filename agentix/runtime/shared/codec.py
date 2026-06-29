"""Wire codec — msgpack.

Every worker frame and Socket.IO event payload flows through `pack(obj)`
/ `unpack(bytes)`. Payloads are plain msgpack-native types: dicts, lists,
strings, numbers, and bytes. RPC args/returns travel as pickle bytes
*inside* frames; pydantic models are `model_dump()`-ed to dicts before
packing. Cross-language, native binary (no base64), small and fast.
"""

from __future__ import annotations

from typing import Any

import msgpack

# Module-level `Packer` reused across `pack()` calls. `autoreset=True`
# means each `.pack()` returns a complete frame and resets internal
# state — safe for the single-threaded asyncio loop.
_PACKER = msgpack.Packer(use_bin_type=True, autoreset=True)


def pack(obj: Any) -> bytes:
    """Serialize a msgpack-native Python object to bytes."""
    return _PACKER.pack(obj)


def unpack(blob: bytes | bytearray | memoryview) -> Any:
    """Deserialize msgpack bytes back to a Python object. Accepts any
    buffer-protocol input — `msgpack.unpackb` handles bytes/bytearray/
    memoryview natively; widening the signature lets callers pass
    Socket.IO payloads (often `bytearray` after framing) through
    without copying."""
    return msgpack.unpackb(blob, raw=False)


__all__ = ["pack", "unpack"]
