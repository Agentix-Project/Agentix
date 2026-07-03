"""Wire codec — msgpack with extension types.

Every worker frame and Socket.IO event payload flows
through `pack(obj)` / `unpack(bytes)`. The goal is: cross-language wire
format, native binary types (no base64), small + fast, and
round-trippable Python types via msgpack extension types.

Extension types registered:

  * `_EXT_NDARRAY` (1) — numpy arrays. Header (`dtype_str|shape_csv`)
    + null byte + raw `tobytes()`. Cross-language consumers replicate
    the same header format.
  * `_EXT_PYDANTIC` (2) — pydantic `BaseModel` instances. Encoded as
    `(qualname, model_dump(mode="python") packed)`. On the receiving
    side the qualname is informational; the decoded dict is returned
    as a plain mapping for callers to interpret.

Numpy is optional — if it's not installed, the ndarray hook is just
skipped (the type never appears on the wire). pydantic is a hard dep
because the rest of the framework uses it.

Decode validation (#140): `unpack` runs on payloads the peer shaped —
including host-side unpacks of sandbox-emitted side-channel events — so
ext decoding is validated and bounded. A malformed ext payload raises
`ExtDecodeError` (never an arbitrary numpy/msgpack error from mid-decode),
the ndarray header is checked before the buffer is interpreted (dtype must
parse, carry no objects, and agree with the buffer size), and pydantic ext
nesting is depth-bounded so a payload cannot recurse the unpacker.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import msgpack
from pydantic import BaseModel

# numpy is an optional dep. Importing it eagerly costs ~400 ms (it
# pulls in a sizeable C-extension graph) and the framework's hot path
# never needs it unless an ndarray actually shows up on the wire — so
# we check for the dist via `find_spec` (no heavy work) and defer the
# real import to first ndarray encode/decode.
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_np: Any = None  # populated lazily by `_numpy()`

_EXT_NDARRAY = 1
_EXT_PYDANTIC = 2

# Legitimate ext nesting is shallow: a pydantic payload carrying an ndarray
# or another dumped model. The bound exists so a wire payload cannot drive
# unpacker recursion arbitrarily deep.
_MAX_EXT_DEPTH = 16
_ext_depth = 0  # single-threaded loop assumption — mirrors `_PACKER` below


class ExtDecodeError(ValueError):
    """A msgpack extension payload failed validation during decode."""


def _numpy() -> Any:
    """Lazy numpy import. Cached on the module."""
    global _np
    if _np is None:
        import numpy  # type: ignore[reportMissingImports]  # noqa: PLC0415

        _np = numpy
    return _np


def _encode_ext(obj: Any) -> msgpack.ExtType:
    if _HAS_NUMPY:
        np = _numpy()
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject or obj.dtype.names is not None:
                # `dtype.str` drops field names/types (a structured dtype
                # collapses to '|V<size>'), and object arrays are pointers —
                # neither round-trips. Refuse loudly at the source.
                raise TypeError(f"agentix.codec: cannot encode ndarray of dtype {obj.dtype}")
            header = f"{obj.dtype.str}|{','.join(map(str, obj.shape))}".encode()
            return msgpack.ExtType(_EXT_NDARRAY, header + b"\x00" + obj.tobytes())
    if isinstance(obj, BaseModel):
        payload = msgpack.packb(
            obj.model_dump(mode="python"),
            default=_encode_ext,
            use_bin_type=True,
        )
        return msgpack.ExtType(_EXT_PYDANTIC, payload)
    raise TypeError(f"agentix.codec: cannot encode {type(obj).__name__}")


def _decode_ndarray(data: bytes) -> Any:
    np = _numpy()
    header, sep, raw = data.partition(b"\x00")
    if not sep:
        raise ExtDecodeError("ndarray ext: missing header terminator")
    try:
        text = header.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExtDecodeError("ndarray ext: header is not ASCII") from exc
    # `dtype.str` may itself lead with the separator ("|b1", "|S5"), so the
    # shape is everything after the LAST separator.
    dtype_str, sep2, shape_str = text.rpartition("|")
    if not sep2:
        raise ExtDecodeError("ndarray ext: malformed header (expected 'dtype|shape')")
    try:
        dtype = np.dtype(dtype_str)
    except Exception as exc:
        raise ExtDecodeError(f"ndarray ext: unknown dtype {dtype_str!r}") from exc
    if dtype.hasobject or dtype.itemsize == 0 or dtype.subdtype is not None or dtype.kind == "V":
        # object: pointers; V/void: a structured dtype's `.str` collapses to
        # '|V<size>' with fields stripped; subarray: frombuffer expands extra
        # elements. No real encode produces any of them (see `_encode_ext`).
        raise ExtDecodeError(f"ndarray ext: refusing dtype {dtype_str!r}")
    try:
        shape = tuple(int(s) for s in shape_str.split(",") if s)
    except ValueError as exc:
        raise ExtDecodeError("ndarray ext: non-integer shape entry") from exc
    if any(n < 0 for n in shape):
        raise ExtDecodeError("ndarray ext: negative shape entry")
    if math.prod(shape) * dtype.itemsize != len(raw):
        raise ExtDecodeError("ndarray ext: shape does not match buffer size")
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    except Exception as exc:
        # Belt-and-braces for numpy refusals the checks above don't model
        # (e.g. ndim above numpy's cap) — the decode guarantee is that a bad
        # payload surfaces as ExtDecodeError, never a raw numpy error.
        raise ExtDecodeError(f"ndarray ext: {exc!r}") from exc


def _decode_ext(code: int, data: bytes) -> Any:
    global _ext_depth
    if code == _EXT_NDARRAY:
        if not _HAS_NUMPY:
            raise ExtDecodeError("ndarray ext received but numpy not installed")
        return _decode_ndarray(data)
    if code == _EXT_PYDANTIC:
        # Decoded as a plain dict for callers to interpret.
        if _ext_depth >= _MAX_EXT_DEPTH:
            raise ExtDecodeError("pydantic ext: nesting exceeds the decode depth bound")
        _ext_depth += 1
        try:
            return msgpack.unpackb(data, ext_hook=_decode_ext, raw=False)
        except ExtDecodeError:
            raise
        except Exception as exc:
            # `{exc!r}`: msgpack's FormatError stringifies to "" — keep the
            # cause class visible in log lines that print only str(error).
            raise ExtDecodeError(f"pydantic ext: malformed payload: {exc!r}") from exc
        finally:
            _ext_depth -= 1
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


def unpack(blob: bytes | bytearray | memoryview) -> Any:
    """Deserialize msgpack bytes back to a Python object. Accepts any
    buffer-protocol input — `msgpack.unpackb` handles bytes/bytearray/
    memoryview natively; widening the signature lets callers pass
    Socket.IO payloads (often `bytearray` after framing) through
    without copying."""
    return msgpack.unpackb(blob, ext_hook=_decode_ext, raw=False)


__all__ = ["ExtDecodeError", "pack", "unpack"]
