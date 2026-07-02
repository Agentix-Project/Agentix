"""`Result[T]` — the typed outcome of a remote call.

`remote()` raises on failure (idiomatic Python, clean happy path).
`try_remote()` returns this `Ok | Failed` sum type instead, for callers
that branch on the outcome at scale (a rollout harness) and want
exhaustive matching:

    match await sandbox.try_remote(solve, task=t):
        case Ok(patch):                   use(patch)
        case Failed(WorkerExited() as e): retry_with_more_memory(e.returncode)
        case Failed(error):               record_failure(error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """A remote call that returned a value."""

    value: T


@dataclass(frozen=True)
class Failed:
    """A remote call that ended in a terminal error — carries the same
    exception `remote()` would have raised (`RemoteCallError` /
    `WorkerExited` / `CallTimeout` / `RuntimeUnreachable`)."""

    error: Exception


# `Ok[T] | Failed`, subscriptable as `Result[R]` (a generic union alias).
Result = Ok[T] | Failed

__all__ = ["Failed", "Ok", "Result"]
