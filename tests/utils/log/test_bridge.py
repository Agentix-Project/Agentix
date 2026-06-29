"""Tests for the host-side `/log` raw-line replayer."""

from __future__ import annotations

import logging

import pytest

from agentix.runtime.shared.codec import pack
from agentix.utils.log._bridge import LOG_EVENT, HostLogNamespace

pytestmark = pytest.mark.asyncio


async def test_replays_line_into_sandbox_logger(caplog) -> None:
    ns = HostLogNamespace()
    with caplog.at_level(logging.INFO, logger="agentix.sandbox.stdout"):
        await ns.trigger_event(LOG_EVENT, pack({"stream": "stdout", "line": "hello from sandbox"}))
    assert any(
        r.name == "agentix.sandbox.stdout" and r.getMessage() == "hello from sandbox"
        for r in caplog.records
    )


async def test_stderr_lines_go_to_stderr_logger(caplog) -> None:
    ns = HostLogNamespace()
    with caplog.at_level(logging.INFO, logger="agentix.sandbox.stderr"):
        await ns.trigger_event(LOG_EVENT, pack({"stream": "stderr", "line": "boom"}))
    assert any(r.name == "agentix.sandbox.stderr" for r in caplog.records)


async def test_ignores_non_line_and_malformed_events(caplog) -> None:
    ns = HostLogNamespace()
    with caplog.at_level(logging.INFO):
        await ns.trigger_event("connect")
        await ns.trigger_event(LOG_EVENT, pack({"stream": "stdout"}))  # no line
    assert not [r for r in caplog.records if r.name.startswith("agentix.sandbox")]
