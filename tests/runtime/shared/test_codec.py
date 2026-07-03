"""Codec ext-type decoding — round-trip fidelity and decode validation.

The wire codec decodes msgpack extension types on every unpack, including
host-side unpacks of sandbox-emitted side-channel payloads (`/trace`, `/log`,
plugin namespaces). A malformed ext payload must surface as one typed error
(`ExtDecodeError`) instead of whatever numpy or msgpack happens to raise
mid-decode, and decoding must stay bounded (#140).
"""

from __future__ import annotations

import msgpack
import numpy as np
import pytest
from pydantic import BaseModel

from agentix.runtime.shared.codec import ExtDecodeError, pack, unpack

_EXT_NDARRAY = 1
_EXT_PYDANTIC = 2


def ndarray_ext(header: bytes, raw: bytes = b"") -> bytes:
    """Wire bytes for an ndarray ext with an arbitrary (possibly bogus) header."""
    return msgpack.packb(msgpack.ExtType(_EXT_NDARRAY, header + b"\x00" + raw))


# ---------------------------------------------------------------- round-trip


@pytest.mark.parametrize("dtype", ["<f8", "<i4", "bool", "int8", "S5", "<U4"])
def test_ndarray_round_trips_across_dtype_byte_orders(dtype: str) -> None:
    # bool/int8/S5 have `dtype.str` forms that LEAD with the header
    # separator ("|b1", "|i1", "|S5") — the decoder must split the header
    # on the LAST separator, not the first.
    array = np.zeros((2, 3), dtype=dtype)
    decoded = unpack(pack(array))
    assert np.array_equal(decoded, array) and decoded.dtype == array.dtype


def test_ndarray_scalar_round_trips() -> None:
    array = np.array(3.5)
    decoded = unpack(pack(array))
    assert decoded.shape == () and decoded == array


def test_pydantic_model_decodes_to_plain_dict() -> None:
    class Point(BaseModel):
        x: int
        y: int

    assert unpack(pack(Point(x=1, y=2))) == {"x": 1, "y": 2}


def test_ndarray_inside_model_payload_decodes() -> None:
    from typing import Any

    class Sample(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str
        data: Any

    decoded = unpack(pack(Sample(name="s", data=np.arange(4))))
    assert decoded["name"] == "s" and np.array_equal(decoded["data"], np.arange(4))


def test_unknown_ext_code_passes_through_inert() -> None:
    blob = msgpack.packb(msgpack.ExtType(42, b"opaque"))
    assert unpack(blob) == msgpack.ExtType(42, b"opaque")


# ---------------------------------------------------------------- validation


def test_ndarray_missing_header_terminator_refused() -> None:
    blob = msgpack.packb(msgpack.ExtType(_EXT_NDARRAY, b"<f8|3"))  # no NUL
    with pytest.raises(ExtDecodeError):
        unpack(blob)


def test_ndarray_non_ascii_header_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"\xff\xfe|1", b"\x00" * 8))


def test_ndarray_header_without_separator_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"f8", b"\x00" * 8))


def test_ndarray_unknown_dtype_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"notadtype|1", b"\x00" * 8))


def test_ndarray_object_dtype_refused() -> None:
    # `np.dtype("O")` parses fine; the decoder must refuse it before the
    # buffer is ever interpreted (frombuffer on object dtype is unsound).
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"O|1", b"\x00" * 8))


def test_ndarray_zero_itemsize_dtype_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"U0|0"))


def test_ndarray_non_integer_shape_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"<f8|a,b", b"\x00" * 8))


def test_ndarray_negative_shape_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"<f8|-1", b"\x00" * 8))


def test_ndarray_shape_buffer_mismatch_refused() -> None:
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"<f8|3", b"\x00" * 8))  # 3 float64 needs 24 bytes


def test_ndarray_excessive_dimensions_refused() -> None:
    # prod((1,)*100) == 1 passes the size check, but numpy caps ndim at 64 —
    # the refusal must be an ExtDecodeError, not numpy's ValueError.
    header = b"<f8|" + b",".join([b"1"] * 100)
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(header, b"\x00" * 8))


def test_ndarray_subarray_dtype_refused() -> None:
    # '(2,2)f8' parses to a subarray dtype (itemsize 32) that frombuffer
    # expands to extra elements — no real encode produces it.
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"(2,2)f8|2", b"\x00" * 64))


def test_ndarray_void_dtype_refused() -> None:
    # '|V12' is what a structured dtype's `.str` collapses to — decoding it
    # would return silently field-stripped data.
    with pytest.raises(ExtDecodeError):
        unpack(ndarray_ext(b"V12|3", b"\x00" * 36))


def test_structured_array_refused_at_encode() -> None:
    # `dtype.str` drops field names/types, so a structured array cannot
    # round-trip — refuse loudly at the source instead of corrupting.
    with pytest.raises(TypeError):
        pack(np.zeros(3, dtype="i4,f8"))


def test_object_array_refused_at_encode() -> None:
    with pytest.raises(TypeError):
        pack(np.array([object()], dtype=object))


def test_pydantic_ext_nesting_is_bounded() -> None:
    # Each ext-2 level re-enters the unpacker; unbounded nesting would
    # recurse toward a RecursionError. Legitimate nesting is shallow
    # (a model payload carrying an array or another dumped model).
    payload = msgpack.packb("x")
    for _ in range(64):
        payload = msgpack.packb(msgpack.ExtType(_EXT_PYDANTIC, payload))
    with pytest.raises(ExtDecodeError):
        unpack(payload)


def test_pydantic_ext_shallow_nesting_decodes() -> None:
    payload = msgpack.packb("x")
    for _ in range(8):
        payload = msgpack.packb(msgpack.ExtType(_EXT_PYDANTIC, payload))
    assert unpack(payload) == "x"


def test_pydantic_ext_malformed_payload_refused() -> None:
    blob = msgpack.packb(msgpack.ExtType(_EXT_PYDANTIC, b"\xc1"))  # 0xc1: never valid
    with pytest.raises(ExtDecodeError, match="FormatError"):
        # msgpack's FormatError stringifies to "" — the wrapped message must
        # still name the cause class.
        unpack(blob)
