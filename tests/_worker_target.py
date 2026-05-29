"""Real importable target for the subprocess worker tests.

Lives in tests/ so the worker subprocess can import
`tests._worker_target` without a separate package install.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel


class EchoResult(BaseModel):
    msg: str


class Echo:
    @staticmethod
    async def echo(msg: str) -> EchoResult:
        return EchoResult(msg=f"echo:{msg}")


async def echo(msg: str) -> EchoResult:
    return await Echo.echo(msg)


async def boom() -> str:
    raise RuntimeError("kaboom")


def add(a: int, b: int) -> int:
    return a + b


class Prefixer:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    async def __call__(self, msg: str) -> EchoResult:
        return EchoResult(msg=f"{self.prefix}:{msg}")

    async def bound(self, msg: str) -> EchoResult:
        return EchoResult(msg=f"bound:{self.prefix}:{msg}")


prefixer = Prefixer("instance")

_exec_counter = 0


async def count_exec_and_sleep(delay: float) -> int:
    global _exec_counter
    _exec_counter += 1
    await asyncio.sleep(delay)
    return _exec_counter


async def reset_exec_counter() -> None:
    global _exec_counter
    _exec_counter = 0


def print_stdout(message: str) -> str:
    print(message)
    return "printed"


def self_sigkill() -> None:
    """Simulate an OOM-kill: the worker process kills itself with SIGKILL
    (the same signal the kernel OOM-killer uses), so no result ever comes
    back and no Python traceback is produced."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)


def spawn_stdin_reading_child() -> int:
    """Spawn a child that reads all of stdin to EOF — exactly what the
    `claude` CLI does. The worker repoints fd 0 at /dev/null, so the child
    must see immediate EOF and must NOT consume bytes from the
    server↔worker control pipe (which would desync every later call).
    Returns the child's exit code (0)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        timeout=10,
    )
    return proc.returncode
