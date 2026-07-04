"""Bash primitive — shell command execution as an Agentix namespace.

Usage:

    from agentix import RuntimeClient
    from agentix.bash import run, run_stream, BashStdout, BashStderr, BashExit, BashError

    async with RuntimeClient(sandbox.runtime_url) as c:
        r = await c.remote(run, command="ls -la", cwd="/workspace")
        print(r.exit_code, r.stdout)

        async for ev in c.remote(run_stream, command="long-job.sh"):
            match ev:
                case BashStdout(data=chunk): print(chunk, end="")
                case BashStderr(data=chunk): print(chunk, end="")
                case BashExit(exit_code=code): print(f"\\nexit {code}")
                case BashError(message=msg): print(f"\\nerror: {msg}")

The package IS the namespace — `run` and `run_stream` are top-level
async functions, dataclasses (`BashResult`, `BashStdout`, …) coexist
as types callers can import. The framework's discovery picks the async
functions; types and constants are just regular Python imports.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from agentix.runtime.shared.env import BUNDLE_RUNTIME_BASH, get_env_without_agentix


def _clean_env(extra: dict[str, str] | None, clean: bool = False) -> dict[str, str]:
    """Build a subprocess env, then apply caller overrides last.

    Default: the worker's own environment — right for runtime tools (the
    claude CLI, bundled rg, ...) that live under /nix. ``clean=True``: the
    task IMAGE's environment (bundle path additions subtracted, stripped vars
    restored) — right for task commands that must resolve the image's own
    toolchain instead of the bundle venv.
    """
    env = get_env_without_agentix() if clean else dict(os.environ)
    if extra:
        env.update(extra)
    return env


def _shell_executable(executable: str | None, env: dict[str, str], clean: bool = False) -> str:
    if executable:
        return shutil.which(executable, path=env.get("PATH")) or executable
    if clean:
        # A clean-env command belongs to the image's world entirely — its
        # restored vars (LD_PRELOAD, ...) target the image's own bash, not
        # the Nix one. Fall back to the bundled bash only when the image has
        # none at all.
        image_bash = shutil.which("bash", path=env.get("PATH"))
        if image_bash:
            return image_bash
        if os.access("/bin/bash", os.X_OK):
            return "/bin/bash"
    if os.access(BUNDLE_RUNTIME_BASH, os.X_OK):
        return BUNDLE_RUNTIME_BASH
    return shutil.which("bash", path=env.get("PATH")) or "/bin/bash"


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> str:
    """Drain a subprocess stream, retaining at most `limit` bytes.

    The drain-to-EOF part matters: if one pipe stops being read after its cap
    is reached, the child can block forever writing to that full pipe while the
    parent waits on the process.
    """
    chunks: list[bytes] = []
    total = 0
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = limit - total
        if remaining <= 0:
            truncated = True
            continue
        if len(chunk) >= remaining:
            chunks.append(chunk[:remaining])
            truncated = len(chunk) > remaining
            total = limit
            continue
        chunks.append(chunk)
        total += len(chunk)
    if truncated:
        chunks.append(b"\n[truncated at %d bytes]" % limit)
    return b"".join(chunks).decode(errors="replace")


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


async def run(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    max_output: int = 10 * 1024 * 1024,
    executable: str | None = None,
    clean_env: bool = False,
) -> BashResult:
    """Run a shell command in the sandbox and return its captured output.

    ``clean_env=True`` runs the command in the task image's environment
    (Agentix's recorded path additions subtracted, stripped vars restored)
    instead of the bundle-polluted worker environment; ``env`` overrides win
    last either way.
    """
    sub_env = _clean_env(env, clean=clean_env)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=sub_env,
        executable=_shell_executable(executable, sub_env, clean=clean_env),
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_task = asyncio.create_task(_read_capped(proc.stdout, max_output))
    stderr_task = asyncio.create_task(_read_capped(proc.stderr, max_output))
    wait_task = asyncio.create_task(proc.wait())
    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, wait_task),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return BashResult(
            exit_code=-1, stdout="", stderr=f"Command timed out after {timeout}s",
        )
    stdout = stdout_task.result()
    stderr = stderr_task.result()
    return BashResult(exit_code=proc.returncode or 0, stdout=stdout, stderr=stderr)


async def run_stream(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    executable: str | None = None,
    clean_env: bool = False,
) -> AsyncIterator[BashEvent]:
    """Run a shell command, yielding events as the subprocess emits them.

    Terminates with a single `BashExit` event on normal completion or
    a single `BashError` event on timeout / wire-level failure.
    ``clean_env`` selects the task image's environment, as in `run`.
    """
    sub_env = _clean_env(env, clean=clean_env)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=sub_env,
        executable=_shell_executable(executable, sub_env, clean=clean_env),
    )

    async def _pump(stream, tag, queue):
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            await queue.put((tag, chunk))
        await queue.put((tag, None))

    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump(proc.stdout, "stdout", queue)),
        asyncio.create_task(_pump(proc.stderr, "stderr", queue)),
    ]
    open_streams = {"stdout", "stderr"}

    try:
        deadline = None
        if timeout is not None:
            deadline = asyncio.get_event_loop().time() + timeout
        while open_streams:
            remaining = None
            if deadline is not None:
                remaining = max(deadline - asyncio.get_event_loop().time(), 0)
                if remaining == 0:
                    proc.kill()
                    yield BashError(message=f"Command timed out after {timeout}s")
                    return
            try:
                tag, chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
            except TimeoutError:
                proc.kill()
                yield BashError(message=f"Command timed out after {timeout}s")
                return
            if chunk is None:
                open_streams.discard(tag)
                continue
            text = chunk.decode(errors="replace")
            if tag == "stdout":
                yield BashStdout(data=text)
            else:
                yield BashStderr(data=text)
        await proc.wait()
        yield BashExit(exit_code=proc.returncode or 0)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
