"""Real importable namespace for the subprocess worker tests.

Lives in tests/ so the worker subprocess can `import _worker_target`
after we add tests/ to PYTHONPATH (the worker doesn't need this to be
pip-installed; the import-path injection is enough).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel

from agentix import trace
from agentix.namespace import Namespace


class EchoResult(BaseModel):
    msg: str


class Echo(Namespace):
    @staticmethod
    async def echo(msg: str) -> EchoResult:
        return EchoResult(msg=f"echo:{msg}")

    @staticmethod
    async def counter(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    @staticmethod
    async def trace_then_echo(msg: str) -> EchoResult:
        trace.emit("test_event", {"msg": msg})
        return EchoResult(msg=f"echo:{msg}")
