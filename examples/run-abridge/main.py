"""Run an agent inside an Agentix sandbox with abridge tunnelling its LLM
traffic back to the host.

Contrast with run-mini-swe-agent, which keeps the model host-side and
feeds the sandbox shell commands. Here the *whole agent* runs in the
sandbox; abridge ferries its OpenAI traffic over the runtime's Socket.IO
connection to the host, which holds the real key and forwards to the
configured endpoint (OpenAI / OpenRouter / vLLM / your gateway). The
agent never sees the real key, and its code is unchanged.

    uv run python main.py --bundle <ref> --task "..."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from agent import solve
from agentix.bridge import BridgeConfig, InMemoryStore, OpenAICompatibleClient, bridged
from agentix.provider.docker import DockerProvider

from agentix.provider.base import SandboxConfig
from agentix.utils.log import configure_logging

DEFAULT_IMAGE = "python:3.13-slim"
DEFAULT_MODEL = "gpt-4o-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="bundle ref from `agentix build`/`agentix deploy`")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--platform", default=None)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible endpoint the host forwards to",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="upstream model the host pins all calls to")
    parser.add_argument("--task", default="In one sentence, what is an agent rollout?")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(default_context="host")

    store = InMemoryStore()
    host = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        store=store,
    )

    session_id = uuid.uuid4().hex
    cfg = SandboxConfig(image=args.image, bundle=args.bundle, platform=args.platform)
    async with DockerProvider().session(cfg) as sandbox:
        # Host half of /abridge — must be registered before the first remote call.
        sandbox.register_namespace(host)
        # One remote call: bridged brings the proxy up around `solve`, points
        # the SDK env at it, runs the (sync) agent off the event loop, tears down.
        answer = await sandbox.remote(
            bridged, solve, args.task, _bridge=BridgeConfig(session_id=session_id)
        )

    print(f"\nanswer: {answer}\n")

    # Agent-eye text trajectory for this rollout. Token-level data
    # (ids/logprobs) lives in your gateway, keyed by the same session_id.
    traj = store.trajectory(session_id)
    print(f"{len(traj)} LLM call(s) captured for session {session_id}:")
    for rec in traj:
        print(f"  {rec.family.value}  in={rec.usage.prompt_tokens} out={rec.usage.completion_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
