"""Tests for `RemoteCallable.validate` (dev-time remote-safety check)."""

from __future__ import annotations

import pytest

from agentix import RemoteCallable


def top_level_fn(x: int) -> int:
    return x


def test_validate_accepts_top_level_function() -> None:
    ref = RemoteCallable.validate(top_level_fn)
    assert str(ref).endswith("::top_level_fn")


def test_validate_rejects_lambda() -> None:
    with pytest.raises(TypeError):
        RemoteCallable.validate(lambda x: x)


def test_validate_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        RemoteCallable.validate(123)  # type: ignore[arg-type]
