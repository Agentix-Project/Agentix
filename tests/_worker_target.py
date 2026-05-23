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
