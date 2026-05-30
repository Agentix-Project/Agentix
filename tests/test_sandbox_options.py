"""`provider.session(config, call_deadline=...)` makes the anti-hang knob
reachable from the documented `sandbox.remote(...)` handle — it threads into
the lazily-created `RuntimeClient`, instead of being stuck at the hardcoded
`call_deadline=None` the handle used to build.
"""

from __future__ import annotations

from agentix.provider.base import Sandbox, SandboxId


def test_sandbox_threads_call_deadline_into_client() -> None:
    sb = Sandbox(
        sandbox_id=SandboxId("sb-1"),
        runtime_url="http://127.0.0.1:1",
        status="running",
        call_deadline=12.5,
    )
    client = sb._runtime_client()
    assert client._call_deadline == 12.5


def test_sandbox_default_call_deadline_is_unbounded() -> None:
    sb = Sandbox(sandbox_id=SandboxId("sb-2"), runtime_url="http://127.0.0.1:1", status="running")
    assert sb.call_deadline is None
    assert sb._runtime_client()._call_deadline is None
