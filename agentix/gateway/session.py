"""Session state, status accounting, result envelope.

Mirrors `polar.gateway.session`: each `Session` represents one run of
an agent callable inside a freshly-provisioned sandbox. The session
moves through a fixed lifecycle managed by `Dispatcher`; downstream
consumers see the immutable `SessionResult` once it ends.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class SessionStatus(enum.StrEnum):
    """Where a session is in its lifecycle.

    Mirrors `polar.gateway.session.SessionStatus` 1:1 so a coordinator
    written for either system can drive both. Terminal states are
    `succeeded`, `failed`, and `cancelled`; everything else is
    in-flight.
    """

    QUEUED = "queued"
    INIT = "init"
    READY = "ready"
    RUNNING = "running"
    POSTRUN = "postrun"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (
            SessionStatus.SUCCEEDED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
        )


@dataclass(slots=True)
class SessionSpec:
    """What the coordinator hands the gateway to start a session.

    `callable_ref` is an import-path string (`module::qualname`) — the
    same shape `RuntimeClient.remote(fn, ...)` resolves to. `args` and
    `kwargs` ride into the sandbox as-is via the runtime's pickle
    pipeline.

    `image` + `bundle` map onto `SandboxConfig` directly. Any extra
    `metadata` is opaque to the gateway and carried through into the
    `SessionResult` for the coordinator's convenience (e.g.
    `{"instance_id": "django-12345"}` for SWE-bench).
    """

    callable_ref: str
    image: str
    bundle: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    platform: str | None = None
    env: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    upstream_model: str | None = None
    """Override the host upstream model for this session's LLM calls."""


@dataclass(slots=True)
class SessionResult:
    """Terminal envelope handed back to the coordinator.

    `value` is whatever the agent callable returned (pickled across
    the runtime boundary). `records` are the captured LLM calls.
    `error` is non-null on failure paths; it does not block the
    coordinator from inspecting partial records.
    """

    session_id: str
    status: SessionStatus
    started_at: float
    ended_at: float
    value: Any = None
    error: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    trajectory: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.ended_at - self.started_at) * 1000.0)


@dataclass(slots=True)
class Session:
    """In-flight session state.

    Updated in place by the `Dispatcher`; published as an immutable
    `SessionResult` once `status.terminal` becomes true. Tests and
    coordinators can peek at the live session via the gateway's
    `SessionStore`.
    """

    spec: SessionSpec
    status: SessionStatus = SessionStatus.QUEUED
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:20]}")
    sandbox_id: str | None = None
    runtime_url: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    value: Any = None
    error: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    trajectory: dict[str, Any] | None = None
    stage_durations_ms: dict[str, float] = field(default_factory=dict)

    def mark(self, status: SessionStatus, *, error: str | None = None) -> None:
        self.status = status
        if error is not None:
            self.error = error
        if status.terminal and self.ended_at is None:
            self.ended_at = time.time()

    def to_result(self) -> SessionResult:
        return SessionResult(
            session_id=self.session_id,
            status=self.status,
            started_at=self.started_at,
            ended_at=self.ended_at or time.time(),
            value=self.value,
            error=self.error,
            records=list(self.records),
            trajectory=self.trajectory,
            metadata=dict(self.spec.metadata),
        )


__all__ = ["Session", "SessionResult", "SessionSpec", "SessionStatus"]
