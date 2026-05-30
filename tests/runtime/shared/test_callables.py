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


class _Runner:
    def run(self, x: int) -> int:
        return x

    def __call__(self, x: int) -> int:
        return x


def test_rejection_names_the_function_and_the_bound_method_rule() -> None:
    # `remote(self.run, ...)` is the canonical first mistake — the message
    # must name the function and point at the real fix, not just "not importable".
    with pytest.raises(TypeError, match="bound method") as exc:
        RemoteCallable.validate(_Runner().run)
    assert "run" in str(exc.value)


def test_rejection_names_callable_instance() -> None:
    with pytest.raises(TypeError, match="callable instances"):
        RemoteCallable.validate(_Runner())  # type: ignore[arg-type]


def test_rejection_lambda_message_mentions_lambda() -> None:
    with pytest.raises(TypeError, match="lambda"):
        RemoteCallable.validate(lambda x: x)
