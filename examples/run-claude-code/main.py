"""Run Claude Code in an Agentix sandbox, LLM traffic bridged via abridge.

Claude Code runs *inside* the sandbox; abridge tunnels its Anthropic calls to
the host, which forwards to any OpenAI-compatible endpoint (the transform maps
Anthropic <-> OpenAI). The agent is unmodified — you just point its `base_url`
at the bridge's in-sandbox service URL. Every call is captured into
`bridge.store` and emitted as a `/trace` span.

    agentix build .
    OPENAI_API_KEY=... uv run python main.py --bundle <ref> \
        --container-engine podman --network host \
        --run-arg --runtime=crun --run-arg --cgroups=disabled
"""

from __future__ import annotations

import argparse
import asyncio
import os

from agentix.agents.claude_code import ClaudeCodeArgs
from agentix.agents.claude_code import run as claude_code_run
from agentix.bridge import Bridge
from agentix.provider.docker import DockerProvider, DockerProviderConfig

from agentix.provider.base import SandboxConfig
from agentix.utils.log import configure_logging

DEFAULT_INSTRUCTION = (
    "Create calc.py with functions add, sub, mul, div and a main that prints "
    "add(2,3), sub(5,1), mul(4,6), div(10,2); run it with python3 to verify. Then add "
    "factorial(n) and verify factorial(5)==120 by running python3. Iterate with the shell."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="bundle ref from `agentix build`/`agentix deploy`")
    p.add_argument("--image", default="python:3.13-slim")
    p.add_argument("--platform", default="linux/amd64")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="upstream model the host pins calls to")
    # The claude CLI sends this model id; the host overrides it upstream.
    p.add_argument("--anthropic-model", default="claude-sonnet-4-5-20250929")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--workdir", default="/tmp/cc")
    p.add_argument("--container-engine", default="docker")
    p.add_argument("--network", default=None, help="container network mode, e.g. `host`")
    p.add_argument("--run-arg", action="append", default=[], dest="run_args", help="extra run arg (repeatable)")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(default_context="host")

    # The bridge is the /abridge consumer + upstream config + proxy lifecycle.
    bridge = Bridge(base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"], model=args.model, timeout=180)
    provider = DockerProvider(DockerProviderConfig(
        container_engine=args.container_engine, run_args=args.run_args, network=args.network,
    ))
    cfg = SandboxConfig(image=args.image, bundle=args.bundle, platform=args.platform)

    async with provider.session(cfg, call_deadline=1800) as sandbox:
        sandbox.register_namespace(bridge)
        await bridge.start_proxy(sandbox, family="anthropic")
        result = await sandbox.remote(claude_code_run, ClaudeCodeArgs(
            instruction=args.instruction,
            model=args.anthropic_model,
            workdir=args.workdir,
            max_turns=args.max_turns,
            base_url=bridge.get_base_url(),
            api_key="sk-abridge",
        ))

    print(f"\nclaude returncode={result.returncode}")
    traj = bridge.store.trajectory(bridge.session_id)
    print(f"{len(traj)} LLM call(s) captured for session {bridge.session_id}:")
    for i, rec in enumerate(traj):
        u = rec.usage
        print(f"  call[{i}] {rec.family.value} {rec.status} in={u.prompt_tokens} out={u.completion_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
