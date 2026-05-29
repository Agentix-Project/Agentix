"""Run mini-swe-agent in a sandbox without abridge."""

from __future__ import annotations

import argparse
import asyncio
import os

import agentix.agents.mini_swe_agent as mini_swe
from agentix.deployment.docker import DockerDeployment
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

from agentix import RuntimeClient
from agentix.bash import run
from agentix.deployment.base import SandboxConfig, session
from agentix.utils.log import configure_logging

DEFAULT_IMAGE = "python:3.13-slim"
DEFAULT_WORKDIR = "/workspace/run-mini-swe-agent"
DEFAULT_MODEL = "openai/gpt-4.1-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--model", default=os.getenv("MINI_SWE_MODEL", DEFAULT_MODEL))
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
            try:
                agent = build_agent(
                    model_name=args.model,
                    api_key=os.environ["OPENAI_API_KEY"],
                    api_base=os.environ["OPENAI_BASE_URL"],
                    workdir=args.workdir,
                )
                result = await client.remote(
                    mini_swe.run,
                    task=args.task,
                    workdir=args.workdir,
                    agent=agent,
                )
            except Exception as exc:
                print("mini_run_error:", flush=True)
                print(f"{type(exc).__name__}: {exc}", flush=True)
                await print_verification(client, args.workdir)
                return
            print(f"mini_exit_status={result.get('exit_status', 'unknown')}", flush=True)
            submission = str(result.get("submission", ""))
            if submission:
                print("mini_submission:", flush=True)
                print(submission.rstrip(), flush=True)
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
    result = await client.remote(run, command=command, timeout=30)
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
    result = await client.remote(run, command=command, cwd=workdir, timeout=30)
    print("verification_exit", result.exit_code, flush=True)
    print("verification_stdout:", flush=True)
    print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print("verification_stderr:", flush=True)
        print(result.stderr.rstrip(), flush=True)


def build_agent(
    *,
    model_name: str,
    api_key: str,
    api_base: str,
    workdir: str,
) -> DefaultAgent:
    base_cfg = get_config_from_spec("mini.yaml")

    model_raw = dict(base_cfg.get("model", {}))
    model_raw["model_name"] = model_name
    model_kwargs = dict(model_raw.get("model_kwargs", {}))
    model_kwargs["api_key"] = api_key
    model_kwargs["api_base"] = api_base
    model_raw["model_kwargs"] = model_kwargs
    model_raw["cost_tracking"] = "ignore_errors"
    model_config = LitellmModelConfig.model_validate(model_raw)

    environment_raw = dict(base_cfg.get("environment", {}))
    environment_raw["cwd"] = workdir
    environment_config = LocalEnvironmentConfig.model_validate(environment_raw)

    agent_raw = dict(base_cfg.get("agent", {}))
    agent_raw.pop("mode", None)
    agent_raw.pop("confirm_exit", None)
    agent_config = AgentConfig.model_validate(agent_raw)
    return DefaultAgent(
        LitellmModel(**model_config.model_dump(mode="python")),
        LocalEnvironment(**environment_config.model_dump(mode="python")),
        **agent_config.model_dump(mode="python"),
    )


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    asyncio.run(main())
