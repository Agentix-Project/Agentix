"""Run the opencode terminal coding agent inside an Agentix sandbox.

opencode is a prebuilt CLI shipped by this plugin's ``default.nix`` — the
binary lands on ``/nix/runtime/bin`` during ``agentix build``. This wrapper
invokes it non-interactively against ``workdir`` and returns a typed
``Result`` (mirrors ``agentix.agents.claude_code``).

opencode is provider-agnostic: point it at whatever backend the caller
configures through ``env`` / ``extra_args`` — typically an OpenAI-compatible
base URL served by the in-sandbox bridge. This module owns no provider
knowledge; invoke it from the host with::

    from agentix.agents.opencode import run

    result = await client.remote(
        run,
        instruction="fix the failing test",
        workdir="/testbed",
        model="openai/gpt-4o",
        env={"OPENAI_BASE_URL": bridge_url, "OPENAI_API_KEY": "sk-..."},
    )
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str


async def run(
    instruction: str,
    *,
    workdir: str = "/testbed",
    timeout: float = 1800,
    model: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Result:
    """Run ``opencode run`` once over ``workdir`` with ``instruction``.

    ``model`` is forwarded as ``--model`` (use the ``provider/model`` form
    opencode expects). ``env`` is layered over the process environment — pass
    the LLM endpoint/credentials here. ``extra_args`` are appended verbatim.
    """
    opencode_bin = shutil.which("opencode") or "opencode"
    cmd: list[str] = [opencode_bin, "run"]
    if model:
        cmd += ["--model", model]
    if extra_args:
        cmd += extra_args
    cmd.append(instruction)

    full_env = {**os.environ, **(env or {})}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=workdir,
        env=full_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return Result(exit_code=-1, stdout="", stderr=f"opencode timed out after {timeout}s")

    return Result(
        exit_code=proc.returncode or 0,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
    )


__all__ = ["Result", "run"]
