"""Minimal remote target for the `agentix build` end-to-end test.

A bundle of this project is the smallest thing that still exercises the
real pipeline: a uv venv with the framework + a plugin, the Nix
toolchain, and plugin/project system closures.
"""

from __future__ import annotations

import argparse
import logging
import subprocess

from agentix.provider.base import SandboxConfig, SandboxProvider
from agentix.utils.log import configure_logging as configure_agentix_logging

logger = logging.getLogger(__name__)


def configure_example_logging() -> None:
    configure_agentix_logging(default_context="host")


def hello() -> str:
    """Return the bundled ripgrep version from the runtime PATH."""
    proc = subprocess.run(["rg", "--version"], check=True, capture_output=True, text=True)
    logger.info("proc.stderr: %s", proc.stderr)
    logger.info("proc.returncode: %s", proc.returncode)
    return proc.stdout.splitlines()[0]


def run() -> str:
    return "hello, world"


def ripgrep_version() -> str:
    return hello()


def _provider_for(name: str) -> SandboxProvider:
    """Construct the sandbox backend the user asked for.

    Each branch imports the typed provider class directly — the
    plugin-discovery registry (`agentix.provider.base.providers`)
    exists for the CLI's string-keyed boundaries, not as a typed
    construction API for user code, so example code spells out the
    import here. Imports are lazy so a user who picks `docker`
    doesn't need `agentix-provider-apptainer` installed.
    """
    if name in {"docker", "podman"}:
        from agentix.provider.docker import DockerProvider, PodmanProvider

        return DockerProvider() if name == "docker" else PodmanProvider()
    if name == "apptainer":
        from agentix.provider.apptainer import ApptainerProvider

        return ApptainerProvider()
    raise SystemExit(f"unknown provider {name!r}; expected one of: docker, podman, apptainer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default="docker",
        choices=("docker", "podman", "apptainer"),
        help="Sandbox backend to run against. Default: docker.",
    )
    parser.add_argument(
        "--image",
        default="python:3.13-slim",
        help=(
            "Task base image. For `docker`/`podman`: a Docker image ref. For "
            "`apptainer`: any reference apptainer can pull "
            "(`docker://...`, `library://...`, a local `.sif`, etc.)."
        ),
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help=(
            "Agentix bundle reference. For `docker`/`podman`: cache path returned by "
            "`agentix deploy`. "
            "For `apptainer`: path to a tar bundle produced by "
            "`agentix build`."
        ),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace | None = None) -> None:
    args = args or _parse_args()
    provider = _provider_for(args.provider)
    config = SandboxConfig(image=args.image, bundle=args.bundle)
    logger.info("config: %s", config)
    async with provider.session(config) as sandbox:
        result = hello()
        print(f"Host result: {result}")
        result = await sandbox.remote(hello)
        print(f"Sandbox result: {result}")


if __name__ == "__main__":
    import asyncio

    configure_example_logging()
    asyncio.run(main())
