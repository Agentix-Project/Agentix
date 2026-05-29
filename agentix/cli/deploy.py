"""`agentix deploy` — materialize a portable bundle for a deployment backend."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import click

from agentix.deployment.base import BundleMaterializer, Deployment, load_deployment

_DEPLOY_HELP = """\
Materialize an Agentix bundle tar for a deployment backend.

`agentix build` always writes a portable bundle tar (`manifest.json + nix/`).
`agentix deploy BACKEND bundle.tar` turns that tar into the backend-native
bundle reference that `SandboxConfig.bundle` should use. The runtime still
appears at `/nix` inside every sandbox.

\b
Examples:
    agentix deploy docker dist/hello.bundle.tar
    agentix deploy podman dist/hello.bundle.tar
    agentix deploy podman dist/hello.bundle.tar --run-arg --runtime=crun --run-arg --cgroups=disabled
    agentix deploy docker dist/hello.bundle.tar --format json   # machine-readable bundle ref

Capture the materialized bundle reference programmatically with `--format json`:

\b
    BUNDLE=$(agentix deploy docker dist/hello.bundle.tar --format json | jq -r .bundle)
"""


@click.command(
    name="deploy",
    help=_DEPLOY_HELP,
    short_help="Materialize a bundle tar for a deployment backend.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("backend")
@click.argument("bundle", type=click.Path(path_type=Path))
@click.option("-n", "--name", default=None, metavar="NAME[:TAG]", help="Optional backend bundle label.")
@click.option("--platform", default=None, metavar="PLATFORM", help="Optional bundle runtime platform.")
@click.option("--container-bin", default=None, metavar="BIN", help="Docker-compatible CLI override.")
@click.option(
    "--run-arg",
    "run_args",
    multiple=True,
    metavar="ARG",
    help="Extra argument for Docker-compatible runtime containers; repeat for multiple args.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: `text` (default, `key -> value` lines) or `json`.",
)
def deploy(
    backend: str,
    bundle: Path,
    name: str | None,
    platform: str | None,
    container_bin: str | None,
    run_args: tuple[str, ...],
    output_format: str,
) -> int:
    deployment = _instantiate_deployment(
        backend,
        container_bin=container_bin,
        run_args=run_args,
    )
    if not isinstance(deployment, BundleMaterializer):
        raise SystemExit(f"deployment backend {backend!r} cannot materialize bundle tars")

    result = asyncio.run(deployment.materialize_bundle(bundle, name=name, platform=platform))
    if output_format == "json":
        print(json.dumps({"bundle": result.bundle, "platform": result.platform, "metadata": result.metadata}))
        return 0
    print(f"bundle -> {result.bundle}")
    if result.platform:
        print(f"platform -> {result.platform}")
    for key, value in sorted(result.metadata.items()):
        print(f"{key} -> {value}")
    return 0


def _instantiate_deployment(
    backend: str,
    *,
    container_bin: str | None,
    run_args: tuple[str, ...],
) -> Deployment:
    cls = cast(Any, load_deployment(backend))
    has_container_options = container_bin is not None or bool(run_args)
    if backend in {"docker", "podman"} or has_container_options:
        try:
            docker_module = import_module("agentix.deployment.docker")
        except ImportError as exc:
            raise SystemExit("Docker-compatible deploy options require agentix-deployment-docker") from exc
        if backend not in {"docker", "podman"}:
            raise SystemExit("container deploy options are only supported by docker and podman backends")
        config_cls = cast(Any, getattr(docker_module, "DockerDeploymentConfig"))
        bin_name = container_bin or backend
        config = config_cls(
            container_bin=bin_name,
            run_args=list(run_args),
        )
        return cast(Deployment, cls(config))
    return cast(Deployment, cls())


def main(argv: Sequence[str] | None = None) -> int:
    try:
        deploy.main(args=argv, prog_name="agentix deploy", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


__all__ = ["deploy", "main"]
