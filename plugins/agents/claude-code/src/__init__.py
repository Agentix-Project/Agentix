"""Sandbox-side: invoke the `claude` CLI against a workdir.

The agent talks to whatever `anthropic_base_url` the host supplies —
typically the URL returned by `abridge.start_anthropic_service(...)`,
also running inside this sandbox. cc.py knows nothing about LLM
providers or Anthropic↔OpenAI translation; that lives in abridge.

Env-var contract for the claude subprocess:
- `IS_SANDBOX=1` lets `--permission-mode=bypassPermissions` work as root
  inside containers.
- `FORCE_AUTO_BACKGROUND_TASKS=1` + `ENABLE_BACKGROUND_TASKS=1` keep
  long-running background tools alive in non-interactive mode.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` suppresses telemetry.
- `ANTHROPIC_BASE_URL` points at the abridge service.
- The four model aliases (sonnet/opus/haiku + subagent) are pinned to
  `anthropic_model` so the CLI doesn't try to switch to an alias the
  upstream provider doesn't know about.

Requires `claude` on PATH inside the sandbox — provided by this
plugin's `default.nix`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("agentix.agents.claude_code")


@dataclass
class Result:
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    log_path: str
    cmd: list[str] = field(default_factory=list)


_LOG_ROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/tmp")) / ".cache" / "cc-logs"
_TAIL_BYTES = 8 * 1024


async def run(
    instruction: str,
    *,
    workdir: str = "/testbed",
    timeout: float = 1800,
    anthropic_base_url: str,
    anthropic_model: str,
    max_turns: int | None = None,
    effort: str | None = None,
    append_system_prompt: str | None = None,
    allowed_tools: str | None = None,
    disallowed_tools: str | None = None,
    env: dict[str, str] | None = None,
    log_name: str | None = None,
) -> Result:
    """Run Claude Code against `workdir` with `instruction`.

    `anthropic_base_url` MUST be a fully-formed URL where an Anthropic-
    compatible HTTP API answers — typically the URL returned by
    `abridge.start_anthropic_service(...)`. `anthropic_model` is the
    model id the CLI will use; it must match the abridge service's
    `response_model`.
    """
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_ROOT / (log_name or "claude-code.jsonl")
    if log_path.exists():
        log_path.unlink()

    full_env = _build_env(
        env or {}, model=anthropic_model, base_url=anthropic_base_url,
    )

    # Python's subprocess.Popen does PATH lookup with the *current* process's
    # PATH, not the `env=` we pass. Resolve `claude` explicitly so the lookup
    # honors the runtime worker's PATH (which already includes the bundle's
    # `/nix/runtime/bin`).
    claude_bin = shutil.which("claude") or "claude"

    cmd: list[str] = [
        claude_bin,
        "--verbose",
        "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions",
        "--model", anthropic_model,
    ]
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]
    if effort:
        cmd += ["--effort", effort]
    if append_system_prompt:
        cmd += ["--append-system-prompt", append_system_prompt]
    if allowed_tools:
        cmd += ["--allowedTools", allowed_tools]
    if disallowed_tools:
        cmd += ["--disallowedTools", disallowed_tools]
    cmd += ["--print", "--", instruction]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=workdir,
        env=full_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_task = asyncio.create_task(_drain_to_file(proc.stdout, log_path))
    stderr_task = asyncio.create_task(_read_tail(proc.stderr, _TAIL_BYTES))

    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, proc.wait()),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        for t in (stdout_task, stderr_task):
            t.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return Result(
            exit_code=-1,
            stdout_tail="",
            stderr_tail=f"claude timed out after {timeout}s",
            log_path=str(log_path),
            cmd=cmd,
        )

    return Result(
        exit_code=proc.returncode or 0,
        stdout_tail=_tail_file(log_path, _TAIL_BYTES),
        stderr_tail=stderr_task.result(),
        log_path=str(log_path),
        cmd=cmd,
    )


# ── Env construction ─────────────────────────────────────────────


def _build_env(extra: dict[str, str], *, model: str, base_url: str) -> dict[str, str]:
    """Build the env passed to the claude subprocess."""
    inherit_keys = (
        "PATH", "HOME", "USER",
        "LANG", "LC_ALL",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "MAX_THINKING_TOKENS",
    )
    env = {k: os.environ[k] for k in inherit_keys if os.environ.get(k)}

    env.setdefault("CLAUDE_CONFIG_DIR", str(_LOG_ROOT / ".claude"))
    Path(env["CLAUDE_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    env["IS_SANDBOX"] = "1"
    env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
    env["ENABLE_BACKGROUND_TASKS"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["DISABLE_PROMPT_CACHING"] = "1"
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_API_KEY"] = "sk-abridge-dummy"
    env["ANTHROPIC_MODEL"] = model

    for alias in (
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        env[alias] = model

    env.update(extra)
    return env


# ── Stream helpers ───────────────────────────────────────────────


async def _drain_to_file(stream: asyncio.StreamReader, path: Path) -> None:
    with path.open("wb") as fh:
        async for line in stream:
            fh.write(line)
            fh.flush()


async def _read_tail(stream: asyncio.StreamReader, limit: int) -> str:
    chunks: list[bytes] = []
    total = 0
    async for line in stream:
        chunks.append(line)
        total += len(line)
        if total > limit * 4:
            chunks = chunks[-200:]
            total = sum(len(c) for c in chunks)
    data = b"".join(chunks)
    if len(data) > limit:
        data = data[-limit:]
    return data.decode(errors="replace")


def _tail_file(path: Path, limit: int) -> str:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return ""
    with path.open("rb") as fh:
        if size > limit:
            fh.seek(size - limit)
        return fh.read().decode(errors="replace")


__all__ = ["Result", "run"]
