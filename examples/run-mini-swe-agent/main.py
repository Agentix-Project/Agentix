"""Run mini-swe-agent in a sandbox without abridge."""

from __future__ import annotations

import argparse
import asyncio
import os

import agentix.agents.mini_swe_agent as mini_swe
from agentix.deployment.docker import DockerDeployment

from agentix import RuntimeClient, bash
from agentix.deployment.base import SandboxConfig, session
from agentix.log import configure_logging

DEFAULT_IMAGE = "python:3.13-slim"
DEFAULT_WORKDIR = "/workspace/run-mini-swe-agent"
DEFAULT_MODEL = "openai/gpt-4.1-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="run-mini-swe-agent:0.1.0")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--model", default=os.getenv("MINI_SWE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--task",
        default=(
            "Edit math_utils.py and add a new function add(a, b) that returns a + b. "
            "Keep subtract unchanged. Run python to verify add(2, 3) == 5."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(default_context="host")
    cfg = SandboxConfig(image=args.image, bundle=args.bundle, platform=args.platform)
    async with session(DockerDeployment(), cfg) as sandbox:
        print(f"runtime_url={sandbox.runtime_url}", flush=True)
        async with RuntimeClient(sandbox.runtime_url, timeout=args.timeout + 180) as client:
            await prepare_workspace(client, args.workdir)
            result = await client.remote(
                mini_swe.run,
                task=args.task,
                workdir=args.workdir,
                model=args.model,
                max_steps=args.max_steps,
                timeout=args.timeout,
                env=forward_model_env(),
            )
            print(f"mini_exit_status={result.exit_status}", flush=True)
            print(f"mini_calls={result.n_calls} mini_cost={result.cost}", flush=True)
            if result.submission:
                print("mini_submission:", flush=True)
                print(result.submission.rstrip(), flush=True)
            if result.details.get("error"):
                print("mini_error:", flush=True)
                print(str(result.details["error"]).rstrip(), flush=True)
            await print_verification(client, args.workdir)


async def prepare_workspace(client: RuntimeClient, workdir: str) -> None:
    command = f"""
set -eu
rm -rf {shell_quote(workdir)}
mkdir -p {shell_quote(workdir)}
cat > {shell_quote(workdir)}/math_utils.py <<'PY'
def subtract(a, b):
    return a - b
PY
"""
    result = await client.remote(bash.run, command=command, timeout=30)
    if result.exit_code != 0:
        raise RuntimeError(f"workspace preparation failed:\n{result.stderr}\n{result.stdout}")


async def print_verification(client: RuntimeClient, workdir: str) -> None:
    command = """
set -eu
python - <<'PY'
from math_utils import add, subtract
print("verify", add(2, 3), subtract(5, 2))
PY
"""
    result = await client.remote(bash.run, command=command, cwd=workdir, timeout=30)
    print("verification_exit", result.exit_code, flush=True)
    print("verification_stdout:", flush=True)
    print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print("verification_stderr:", flush=True)
        print(result.stderr.rstrip(), flush=True)


def forward_model_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT"):
        value = os.getenv(key)
        if value:
            env[key] = value
    return env


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    asyncio.run(main())
