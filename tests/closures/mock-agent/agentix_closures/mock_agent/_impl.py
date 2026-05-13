"""mock-agent impl — runs inside the sandbox."""

from __future__ import annotations

from . import RunResult


def run(instruction: str, workdir: str = "/") -> RunResult:
    return RunResult(
        exit_code=0,
        patch=f"# mock patch\n# workdir={workdir}\n# instruction={instruction}\n",
    )
