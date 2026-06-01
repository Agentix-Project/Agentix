from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ClaudeCodeInput(BaseModel):
    instruction: str
    model: str
    workdir: str
    api_key: str = "sk-ant-api03-claude-code-default-no-real-credentials-set"
    base_url: str = "https://api.anthropic.com"
    timeout: float = 1800
    max_turns: int | None = None
    effort: str | None = None
    append_system_prompt: str | None = None
    allowed_tools: str | None = None
    disallowed_tools: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


@dataclass
class ClaudeCodeResult:
    returncode: int
    stdout: bytes
    stderr: bytes


async def run(args: ClaudeCodeInput) -> ClaudeCodeResult:
    os.makedirs(args.workdir, exist_ok=True)  # the claude CLI's cwd must exist
    cmd: list[str] = [
        "claude",
        "--verbose",
        "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions",
        "--model", args.model,
    ]
    if args.max_turns is not None:
        cmd += ["--max-turns", str(args.max_turns)]
    if args.effort:
        cmd += ["--effort", args.effort]
    if args.append_system_prompt:
        cmd += ["--append-system-prompt", args.append_system_prompt]
    if args.allowed_tools:
        cmd += ["--allowedTools", args.allowed_tools]
    if args.disallowed_tools:
        cmd += ["--disallowedTools", args.disallowed_tools]
    cmd += ["--print", "--", args.instruction]


    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=args.workdir,
        env=_build_env(args),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(_stream_to_log(proc.stdout, logging.INFO))
    stderr_task = asyncio.create_task(_stream_to_log(proc.stderr, logging.WARNING))
    try:
        async with asyncio.timeout(args.timeout):
            await proc.wait()
    except TimeoutError:
        logger.warning("claude-code timed out after %.1fs", args.timeout)
        proc.kill()
        await proc.wait()
    stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    return ClaudeCodeResult(
        returncode=proc.returncode or 0,
        stdout=stdout,
        stderr=stderr,
    )


async def _stream_to_log(stream: asyncio.StreamReader | None, level: int) -> bytes:
    """Drain `stream` in chunks, logging each newline-terminated line; return all bytes."""
    if stream is None:
        return b""
    chunks: list[bytes] = []
    pending = bytearray()
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        chunks.append(chunk)
        pending.extend(chunk)
        while (nl := pending.find(b"\n")) >= 0:
            line, pending[:] = bytes(pending[:nl]), pending[nl + 1 :]
            text = line.rstrip(b"\r").decode(errors="replace")
            if text:
                logger.log(level, "%s", text)
    if pending:  # trailing partial line (no final newline before EOF)
        text = pending.decode(errors="replace")
        if text:
            logger.log(level, "%s", text)
    return b"".join(chunks)


def _build_env(args: ClaudeCodeInput) -> dict[str, str]:
    return {
        **os.environ,
        **args.env,
        "IS_SANDBOX": "1",
        "FORCE_AUTO_BACKGROUND_TASKS": "1",
        "ENABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_BASE_URL": args.base_url,
        "ANTHROPIC_API_KEY": args.api_key,
        "ANTHROPIC_MODEL": args.model,
    }
