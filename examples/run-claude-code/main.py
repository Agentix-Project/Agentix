"""Run Claude Code through sandbox-local abridge-mitm.

Flow:

    Claude Code in sandbox
      -> sandbox-local mitmproxy
      -> sandbox-local Agentix forwarder
      -> host OpenAIForwarder
      -> OpenAI-compatible upstream

The real upstream API key stays on the host. The sandbox only receives
an Anthropic-shaped 127.0.0.1 URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import agentix.agents.claude_code as cc
import agentix.bridge.mitm as abridge_mitm
from agentix.bash import run
from agentix.provider.docker import DockerProvider

from agentix.provider.base import Sandbox, SandboxConfig
from agentix.utils.log import configure_logging

DEFAULT_IMAGE = "python:3.13-slim"
DEFAULT_PLATFORM = "linux/amd64"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_WORKDIR = "/workspace/run-claude-code"


async def main() -> None:
    args = parse_args()
    configure_logging(default_context="host")
    forwarder = abridge_mitm.OpenAIForwarder(
        base_url=require_env("OPENAI_BASE_URL"),
        api_key=require_env("OPENAI_API_KEY"),
        model=require_env("OPENAI_MODEL"),
        extra_body=json_object_env("ABRIDGE_OPENAI_EXTRA_BODY"),
        timeout=args.upstream_timeout,
    )
    await run_claude_code(args, forwarder=forwarder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--proxy-port", type=int, default=0)
    parser.add_argument("--proxy-mode", default="reverse:https://api.anthropic.com")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--upstream-timeout", type=float, default=120)
    parser.add_argument(
        "--instruction",
        default=(
            "Edit math_utils.py. Add a function add(a, b) that returns a + b. "
            "Keep subtract unchanged. Run python to verify add(2, 3) == 5. "
            "Do not create unrelated files."
        ),
    )
    return parser.parse_args()


async def run_claude_code(args: argparse.Namespace, *, forwarder: abridge_mitm.OpenAIForwarder) -> None:
    cfg = SandboxConfig(image=args.image, bundle=args.bundle, platform=args.platform)
    async with DockerProvider().session(cfg) as sandbox:
        print(f"runtime_url={sandbox.runtime_url}", flush=True)
        sandbox.register_namespace(forwarder)
        proxy = await sandbox.remote(
            abridge_mitm.start_proxy,
            port=args.proxy_port,
            mode=args.proxy_mode,
        )
        print(f"abridge_proxy_url={proxy.url}", flush=True)
        try:
            await prepare_repo(sandbox, args.workdir)
            result = await sandbox.remote(
                cc.run,
                instruction=args.instruction,
                workdir=args.workdir,
                timeout=args.timeout,
                max_turns=args.max_turns,
                anthropic_base_url=proxy.url,
                anthropic_model=args.anthropic_model,
                log_name="run-claude-code.jsonl",
            )
            print(f"claude_exit={result.exit_code}", flush=True)
            if result.stderr_tail:
                print("claude_stderr_tail:", flush=True)
                print(result.stderr_tail.rstrip(), flush=True)
            if result.stdout_tail:
                print("claude_stdout_tail:", flush=True)
                print(result.stdout_tail.rstrip(), flush=True)
            await print_verification(sandbox, args.workdir)
        finally:
            await sandbox.remote(abridge_mitm.stop_proxy, handle=proxy)


async def prepare_repo(sandbox: Sandbox, workdir: str) -> None:
    command = f"""
set -eu
rm -rf {shell_quote(workdir)}
mkdir -p {shell_quote(workdir)}
cd {shell_quote(workdir)}
git init -q
git config user.email smoke@example.com
git config user.name Smoke
cat > math_utils.py <<'PY'
def subtract(a, b):
    return a - b
PY
git add math_utils.py
git commit -q -m init
"""
    result = await sandbox.remote(run, command=command, timeout=30)
    if result.exit_code != 0:
        raise RuntimeError(f"repo preparation failed:\n{result.stderr}\n{result.stdout}")


async def print_verification(sandbox: Sandbox, workdir: str) -> None:
    command = """
set -eu
git diff -- math_utils.py
python - <<'PY'
from math_utils import add, subtract
print("verify", add(2, 3), subtract(5, 2))
PY
"""
    result = await sandbox.remote(run, command=command, cwd=workdir, timeout=30)
    print("verification_exit", result.exit_code, flush=True)
    print("verification_stdout:", flush=True)
    print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print("verification_stderr:", flush=True)
        print(result.stderr.rstrip(), flush=True)


def json_object_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    asyncio.run(main())
