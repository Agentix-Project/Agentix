"""A `result` frame that can't be written (oversized pickled return value
hitting `FrameTooLarge`) must be converted into a small `error` frame for the
same call, so the host future fails fast instead of hanging forever.
"""

from __future__ import annotations

import pytest

from agentix.runtime.server.worker.process import Worker

pytestmark = pytest.mark.asyncio


async def test_recover_failed_result_emits_frametoolarge_error() -> None:
    w = Worker()
    w._recover_failed_frame({"type": "result", "call_id": "c1", "value": b"x"})
    frame = w._outbound_q.get_nowait()
    assert frame["type"] == "error"
    assert frame["call_id"] == "c1"
    assert frame["error"]["type"] == "FrameTooLarge"


async def test_recover_ignores_non_result_frames() -> None:
    w = Worker()
    # An `error` frame that itself failed to write has no safe fallback.
    w._recover_failed_frame({"type": "error", "call_id": "c1", "error": {}})
    # A result with no call_id can't be addressed.
    w._recover_failed_frame({"type": "result", "value": b"x"})
    assert w._outbound_q.empty()
