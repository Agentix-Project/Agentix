"""Wire codec — msgpack with extension types.

Every RPC frame, HTTP `/_remote` body, Socket.IO event payload, trace
event, and log record flows through `pack(obj)` / `unpack(bytes)`. The
goal is: cross-language wire format, native binary types (no base64),
small + fast, and round-trippable Python types via msgpack extension
types.

Extension types registered:

  * `_EXT_NDARRAY` (1) — numpy arrays. Header (`dtype_str|shape_csv`)
    + null byte + raw `tobytes()`. Cross-language consumers replicate
    the same header format.
  * `_EXT_PYDANTIC` (2) — pydantic `BaseModel` instances. Encoded as
    `(qualname, model_dump(mode="python") packed)`. On the receiving
    side the qualname is informational; the decoded dict is what the
    Dispatcher feeds into `TypeAdapter.validate_python`.

Numpy is optional — if it's not installed, the ndarray hook is just
skipped (the type never appears on the wire). pydantic is a hard dep
because the rest of the framework uses it.
"""

from __future__ import annotations

from typing import Any

import msgpack
from pydantic import BaseModel

try:
    import numpy as np  # type: ignore[reportMissingImports]
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_EXT_NDARRAY = 1
_EXT_PYDANTIC = 2


def _encode_ext(obj: Any) -> msgpack.ExtType:
    if _HAS_NUMPY and isinstance(obj, np.ndarray):
        header = f"{obj.dtype.str}|{','.join(map(str, obj.shape))}".encode()
        return msgpack.ExtType(_EXT_NDARRAY, header + b"\x00" + obj.tobytes())
    if isinstance(obj, BaseModel):
        payload = msgpack.packb(
            obj.model_dump(mode="python"),
            default=_encode_ext, use_bin_type=True,
        )
        return msgpack.ExtType(_EXT_PYDANTIC, payload)
    raise TypeError(f"agentix.codec: cannot encode {type(obj).__name__}")


def _decode_ext(code: int, data: bytes) -> Any:
    if code == _EXT_NDARRAY:
        if not _HAS_NUMPY:
            raise RuntimeError("ndarray ext received but numpy not installed")
        header, raw = data.split(b"\x00", 1)
        dtype_str, shape_str = header.decode().split("|")
        shape = tuple(int(s) for s in shape_str.split(",") if s)
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
    if code == _EXT_PYDANTIC:
        # Decoded as a plain dict; the receiving side's TypeAdapter
        # validates into the concrete model class.
        return msgpack.unpackb(data, ext_hook=_decode_ext, raw=False)
    return msgpack.ExtType(code, data)


# Module-level `Packer` reused across `pack()` calls. `autoreset=True`
# means each `.pack()` returns a complete frame and resets internal
# state — safe for the single-threaded asyncio loop. Re-entrant
# packing (e.g. `_encode_ext` packing a pydantic model) still goes
# through `msgpack.packb`, which creates its own short-lived Packer
# so the module-level one's state is not clobbered.
_PACKER = msgpack.Packer(default=_encode_ext, use_bin_type=True, autoreset=True)


def pack(obj: Any) -> bytes:
    """Serialize an arbitrary Python object to msgpack bytes."""
    return _PACKER.pack(obj)


def unpack(blob: bytes) -> Any:
    """Deserialize msgpack bytes back to a Python object."""
    return msgpack.unpackb(blob, ext_hook=_decode_ext, raw=False)


__all__ = ["pack", "unpack"]
