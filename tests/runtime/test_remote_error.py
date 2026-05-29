"""`RemoteCallError` surfaces the sandbox traceback in its message."""

from __future__ import annotations

from agentix.runtime.client.client import RemoteCallError
from agentix.runtime.shared.models import RemoteError


def test_remote_call_error_includes_traceback() -> None:
    err = RemoteError(type="ValueError", message="boom", traceback="Traceback (most recent call last):\n...")
    exc = RemoteCallError("app::run", err)
    text = str(exc)
    assert "app::run: ValueError: boom" in text
    assert "Traceback (most recent call last):" in text
    assert exc.error is err


def test_remote_call_error_without_traceback() -> None:
    err = RemoteError(type="ValueError", message="boom")
    text = str(RemoteCallError("app::run", err))
    assert text == "app::run: ValueError: boom"
    assert "traceback" not in text.lower()
