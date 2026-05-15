"""Bash primitive — shell command execution as an Agentix closure.

Stub-only module. Callers import the typed `Bash` Namespace and the
event/result dataclasses; the impl lives in `_impl.py` and runs only
inside the sandbox. `_register.py` composes the two (stub + impl).

Usage:

    from agentix import RuntimeClient
    from agentix_closures import bash

    async with RuntimeClient(sandbox.runtime_url) as c:
        r = await c.remote(bash.Bash.run, command="ls -la", cwd="/workspace")
        print(r.exit_code, r.stdout)

        async for ev in c.remote(bash.Bash.run_stream, command="long-job.sh"):
            match ev:
                case bash.BashStdout(data=chunk): print(chunk, end="")
                case bash.BashStderr(data=chunk): print(chunk, end="")
                case bash.BashExit(exit_code=code): print(f"\\nexit {code}")
                case bash.BashError(message=msg): print(f"\\nerror: {msg}")
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from agentix.namespace import Namespace


@dataclass
class BashResult:
    """Return value of `Bash.run` — full output captured before the call returns."""

    exit_code: int
    stdout: str
    stderr: str


# Algebraic stream events — each variant is its own dataclass so callers
# can `match event: case BashStdout(...)` and pyright tracks the type.
# The `type` field is the wire discriminator; users pattern-match the
# class, not the field.


@dataclass
class BashStdout:
    """A chunk of subprocess stdout."""

    data: str
    type: Literal["stdout"] = "stdout"


@dataclass
class BashStderr:
    """A chunk of subprocess stderr."""

    data: str
    type: Literal["stderr"] = "stderr"


@dataclass
class BashExit:
    """The subprocess finished. `exit_code` is its return status."""

    exit_code: int
    type: Literal["exit"] = "exit"


@dataclass
class BashError:
    """Wire-side problem (e.g. timeout, fork failure). `message` explains."""

    message: str
    type: Literal["error"] = "error"


BashEvent = Annotated[
    BashStdout | BashStderr | BashExit | BashError,
    Field(discriminator="type"),
]
"""One event from `Bash.run_stream`. Discriminated union of the four
variants above — JSON wire form carries a `type` tag, but in Python
the user pattern-matches the class directly."""


class Bash(Namespace):
    """Shell command execution primitive."""

    async def run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        max_output: int = 10 * 1024 * 1024,
        paths_from: list[str] | None = None,
    ) -> BashResult:
        """Run a shell command in the sandbox and return its captured output.

        `paths_from` lists Python package paths of other loaded closures
        whose `entry/bin/` directories should be prepended to PATH for
        this command. Use `["*"]` to include every mounted closure's bins.
        """
        ...

    async def run_stream(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        paths_from: list[str] | None = None,
    ) -> AsyncIterator[BashEvent]:
        """Run a shell command, yielding events as the subprocess emits them.

        Terminates with a single `BashExit` event on normal completion or
        a single `BashError` event on timeout / wire-level failure.
        """
        ...
