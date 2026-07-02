"""Docker provider: sandbox CRUD via a Docker-compatible CLI.

Design:

  `agentix build` produces a portable tar containing `manifest.json`
  and a full `/nix` runtime tree. `agentix deploy docker|podman`
  deploys that tar into a content-addressed host cache directory (the
  local-extract form of the `BundleDeployer` Protocol). `config.bundle`
  is the cache root returned by deploy; its `nix/` child is bind-mounted
  read-only into each sandbox at `/nix`.

  Two artifacts, one container. `config.image` is the task-specific
  base the workload runs against. The cached bundle supplies
  `/nix/runtime/bootstrap.sh`, the Python venv, and every Nix closure.
  No imported runtime artifact or long-lived carrier container is needed.

  Sandbox create:
      docker run [--platform <platform>] -d --name <sid> \\
         -p 127.0.0.1:<port>:<port> \\
         -e AGENTIX_BIND_PORT=<port> \\
         --mount type=bind,source=<cache>/nix,target=/nix,readonly \\
         --entrypoint /nix/runtime/bootstrap.sh \\
         <image>

  The bundle's `/nix/runtime/bootstrap.sh` preps the runtime PATHs and
  launches the runtime server. We pick a free host port, publish the
  same port to loopback, pass it via `AGENTIX_BIND_PORT`, and
  health-check `/health` on it.

  The backend defaults to the `docker` CLI. Podman can be selected with
  `DockerProviderConfig(container_engine="podman", ...)` when it
  provides the Docker-compatible commands this backend needs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shlex
import socket
import tarfile
from pathlib import Path
from uuid import uuid4

import click
import httpx
from pydantic import BaseModel, Field, field_validator

from agentix.cli.deploy import common_options, print_deploy_result
from agentix.provider._extract import extract_nix_tree
from agentix.provider.base import (
    DeployedBundle,
    Sandbox,
    SandboxConfig,
    SandboxId,
    SandboxInfo,
    SandboxProvider,
    SandboxResource,
)
from agentix.runtime import BIND_HOST_ENV, BIND_PORT_ENV, BUNDLE_NIX_ROOT, BUNDLE_RUNTIME_ENTRYPOINT

logger = logging.getLogger("agentix.provider.docker")


def _split_shell_args(value: str, label: str) -> list[str]:
    try:
        return shlex.split(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} must contain shell-style arguments: {exc}") from exc


class DockerProviderConfig(BaseModel):
    """Container engine settings for the provider backend."""

    container_engine: str = Field(
        default="docker",
        description="Container engine (Docker-compatible CLI), e.g. `docker` or `podman`.",
    )
    run_args: list[str] = Field(
        default_factory=list,
        description="Extra arguments inserted before sandbox networking and env args.",
    )
    network: str | None = Field(
        default=None,
        description="Optional container network mode, e.g. `host` or `slirp4netns`.",
    )
    publish_host: str = Field(
        default="127.0.0.1",
        description="Host address for `-p`; empty string emits `<port>:<port>`.",
    )
    gpu_args: list[str] | None = Field(
        default=None,
        description="Optional resource.gpu translation; args may contain `{gpu}`.",
    )
    bundle_cache_dir: Path | None = Field(
        default=None,
        description="Host directory for deployed bundle caches. Default: ~/.cache/agentix/bundles.",
    )

    @field_validator("container_engine")
    @classmethod
    def _validate_container_engine(cls, value: str) -> str:
        if not value:
            raise ValueError("container_engine must not be empty")
        return value

    @field_validator("run_args", mode="before")
    @classmethod
    def _parse_extra_args(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return _split_shell_args(value, "provider extra args")
        return value

    @field_validator("gpu_args", mode="before")
    @classmethod
    def _parse_gpu_args(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return _split_shell_args(value, "provider gpu_args")
        return value

    def to_provider(self) -> DockerProvider:
        """Construct a `DockerProvider` from this config."""
        return DockerProvider(self)


def _default_config(config: DockerProviderConfig | None = None) -> DockerProviderConfig:
    return config or DockerProviderConfig()


def _container_engine(config: DockerProviderConfig | None = None) -> str:
    return _default_config(config).container_engine


def _port_mapping(port: int, config: DockerProviderConfig | None = None) -> str:
    host = _default_config(config).publish_host
    if not host:
        return f"{port}:{port}"
    return f"{host}:{port}:{port}"


def _network_args(config: DockerProviderConfig | None = None) -> list[str]:
    network = _default_config(config).network
    if not network:
        return []
    return ["--network", network]


def _network_uses_host_ports(config: DockerProviderConfig | None = None) -> bool:
    network = _default_config(config).network
    return network == "host" or bool(network and network.startswith("host:"))


def _publish_args(port: int, config: DockerProviderConfig | None = None) -> list[str]:
    if _network_uses_host_ports(config):
        return []
    return ["-p", _port_mapping(port, config)]


def _format_cpu(cpu: float) -> str:
    return str(int(cpu)) if cpu.is_integer() else str(cpu)


def _gpu_args(gpu: int, config: DockerProviderConfig | None = None) -> list[str]:
    template = _default_config(config).gpu_args
    if template is None:
        return ["--gpus", str(gpu)]
    return [arg.format(gpu=gpu) for arg in template]


def _resource_args(resource: SandboxResource | None, config: DockerProviderConfig | None = None) -> list[str]:
    if resource is None:
        return []
    args: list[str] = []
    if resource.cpu is not None:
        args.extend(["--cpus", _format_cpu(resource.cpu)])
    if resource.memory is not None:
        args.extend(["--memory", str(resource.memory)])
    if resource.gpu is not None:
        args.extend(_gpu_args(resource.gpu, config))
    return args


def _bundle_manifest(bundle_tar: Path) -> dict[str, object]:
    try:
        with tarfile.open(bundle_tar, "r:*") as tar:
            member = tar.getmember("manifest.json")
            f = tar.extractfile(member)
            if f is None:
                raise RuntimeError(f"bundle {bundle_tar} has an unreadable manifest.json")
            manifest = json.loads(f.read().decode())
    except (tarfile.TarError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"bundle {bundle_tar} is not an Agentix bundle tar") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "agentix-bundle":
        raise RuntimeError(f"bundle {bundle_tar} manifest is not an Agentix bundle")
    return manifest


def _bundle_display_name(manifest: dict[str, object], name: str | None) -> str:
    if name:
        return name
    bundle_name = manifest.get("name")
    bundle_tag = manifest.get("tag")
    if not isinstance(bundle_name, str) or not bundle_name:
        raise RuntimeError("bundle manifest missing string field `name`")
    if not isinstance(bundle_tag, str) or not bundle_tag:
        return bundle_name
    return f"{bundle_name}:{bundle_tag}"


def _bundle_platform(manifest: dict[str, object], platform: str | None) -> str | None:
    if platform:
        return platform
    manifest_platform = manifest.get("platform")
    return manifest_platform if isinstance(manifest_platform, str) and manifest_platform else None


def _bundle_digest(manifest: dict[str, object], bundle_tar: Path) -> str:
    digest = manifest.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        value = digest.removeprefix("sha256:").lower()
        if value and all(ch in "0123456789abcdef" for ch in value):
            return value
    h = hashlib.sha256()
    with bundle_tar.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_cache_base(config: DockerProviderConfig | None = None) -> Path:
    configured = _default_config(config).bundle_cache_dir
    if configured is not None:
        return configured.expanduser()
    return Path.home() / ".cache" / "agentix" / "bundles"


def _bundle_cache_root(
    manifest: dict[str, object],
    bundle_tar: Path,
    config: DockerProviderConfig | None = None,
) -> Path:
    return _bundle_cache_base(config) / f"sha256-{_bundle_digest(manifest, bundle_tar)}"


def _extract_bundle_to_cache(
    bundle_tar: Path,
    manifest: dict[str, object],
    cache_root: Path,
) -> None:
    extract_nix_tree(bundle_tar, cache_root, manifest=manifest)


def _bundle_nix_path(bundle: str) -> Path:
    root = Path(bundle).expanduser().resolve()
    nix = root / "nix"
    if not nix.is_dir():
        raise RuntimeError(f"deployed bundle {bundle!r} does not contain a nix/ directory")
    return nix


def _nix_mount_args(bundle: str) -> list[str]:
    nix = _bundle_nix_path(bundle)
    return ["--mount", f"type=bind,source={nix},target={BUNDLE_NIX_ROOT},readonly"]


async def _docker(
    *args: str,
    config: DockerProviderConfig | None = None,
    check: bool = True,
    retries: int = 0,
) -> tuple[int, bytes, bytes]:
    attempt = 0
    delay = 2.0
    engine = _container_engine(config)
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                engine,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"container engine {engine!r} not found on PATH. Install Docker "
                f"(https://docs.docker.com/get-docker/) or Podman "
                f"(https://podman.io/docs/installation), or set container_engine."
            ) from exc
        stdout, stderr = await proc.communicate()
        rc = proc.returncode or 0
        if not check or rc == 0:
            return rc, stdout, stderr
        if attempt >= retries or not _is_transient_docker_error(stderr):
            raise RuntimeError(f"{engine} {args[0]} failed: {stderr.decode(errors='replace')}")
        attempt += 1
        logger.warning(
            "%s %s failed with transient error; retrying in %.1fs (%d/%d)",
            engine,
            args[0],
            delay,
            attempt,
            retries,
        )
        await asyncio.sleep(delay)
        delay *= 2


def _is_transient_docker_error(stderr: bytes) -> bool:
    text = stderr.decode(errors="replace").lower()
    return any(
        needle in text
        for needle in (
            "failed to fetch oauth token",
            "unexpected status from post request",
            "tls handshake timeout",
            "connection reset by peer",
            "i/o timeout",
            "temporarily unavailable",
        )
    )


class DockerProvider(SandboxProvider):
    """Sandbox CRUD via local Docker."""

    def __init__(self, config: DockerProviderConfig | None = None):
        self.config = _default_config(config)
        self._ports: dict[SandboxId, int] = {}  # sandbox_id → host port
        # Host ports handed out but not yet released, so concurrent create()s
        # can't be allocated the same port in the window before `docker run`
        # binds it. Reserved in _allocate_port; released in delete / on a
        # failed create.
        self._inflight_ports: set[int] = set()

    async def deploy_bundle(
        self,
        bundle: Path,
        *,
        name: str | None = None,
        platform: str | None = None,
    ) -> DeployedBundle:
        """Local-extract form of `BundleDeployer.deploy_bundle`.

        Materializes the portable tar into a content-addressed host cache
        directory. The returned `bundle` ref is the cache root; subsequent
        `provider.create(SandboxConfig(bundle=<ref>))` bind-mounts its
        `nix/` child into the sandbox at `/nix`.
        """
        bundle_tar = bundle.expanduser().resolve()
        if not bundle_tar.is_file():
            raise FileNotFoundError(f"bundle tar not found: {bundle}")
        manifest = _bundle_manifest(bundle_tar)
        bundle_name = _bundle_display_name(manifest, name)
        deployed_platform = _bundle_platform(manifest, platform)
        cache_root = _bundle_cache_root(manifest, bundle_tar, self.config)
        _extract_bundle_to_cache(bundle_tar, manifest, cache_root)
        cache_str = shlex.quote(str(cache_root))
        return DeployedBundle(
            bundle=str(cache_root),
            platform=deployed_platform,
            metadata={"cache": str(cache_root), "name": bundle_name},
            hints={
                "inspect contents": f"ls -la {cache_str}/nix/",
                "remove from cache": f"rm -rf {cache_str}",
            },
        )

    def _allocate_port(self) -> int:
        # Ask the kernel for a free TCP port, then reserve it in-process so a
        # concurrent create() isn't handed the same number before `docker run`
        # binds it (the kernel won't hand the same port to two live binds, but
        # two creates that each bind-and-close can still collide on it).
        for _ in range(100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            if port not in self._inflight_ports:
                self._inflight_ports.add(port)
                return port
        raise RuntimeError("could not allocate a free host port")

    async def create(self, config: SandboxConfig) -> Sandbox:
        sandbox_id = SandboxId(f"agentix-{uuid4().hex[:8]}")
        port = self._allocate_port()

        env_args: list[str] = ["-e", f"{BIND_PORT_ENV}={port}"]
        if _network_uses_host_ports(self.config) and not (config.env and BIND_HOST_ENV in config.env):
            env_args.extend(["-e", f"{BIND_HOST_ENV}=127.0.0.1"])
        if config.env:
            for k, v in config.env.items():
                env_args.extend(["-e", f"{k}={v}"])

        platform_args = ["--platform", config.platform] if config.platform else []
        resource_args = _resource_args(config.resource, self.config)
        try:
            await _docker(
                "run",
                *platform_args,
                *resource_args,
                *self.config.run_args,
                *_network_args(self.config),
                "-d",
                "--name",
                sandbox_id,
                *_publish_args(port, self.config),
                *env_args,
                *_nix_mount_args(config.bundle),
                "--entrypoint",
                BUNDLE_RUNTIME_ENTRYPOINT,
                config.image,
                config=self.config,
                retries=3,
            )
            self._ports[sandbox_id] = port
            logger.info("Created sandbox %s on port %d", sandbox_id, port)
            await self._wait_healthy(port)
        except BaseException:
            # Failed or cancelled create: remove any container that started and
            # release the reserved port so neither leaks. The container may
            # have started but never become healthy (bad bundle, crash loop);
            # `session()` can't clean up a sandbox it never received.
            await self.delete(sandbox_id)
            self._inflight_ports.discard(port)
            raise
        return Sandbox(
            sandbox_id=sandbox_id,
            runtime_url=f"http://localhost:{port}",
            status="running",
        )

    async def _wait_healthy(self, port: int) -> None:
        base_url = f"http://localhost:{port}"
        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            for _ in range(120):
                try:
                    r = await client.get("/health")
                    if r.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Runtime server not alive at {base_url}")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        port = self._ports.get(sandbox_id)
        if port is None:
            raise KeyError(f"Sandbox not found: {sandbox_id}")
        rc, stdout, _ = await _docker(
            "inspect",
            "-f",
            "{{.State.Status}}",
            sandbox_id,
            config=self.config,
            check=False,
        )
        status = stdout.decode().strip() if rc == 0 else "unknown"
        return SandboxInfo(
            sandbox_id=sandbox_id,
            runtime_url=f"http://localhost:{port}",
            status=status,
        )

    async def delete(self, sandbox_id: SandboxId) -> None:
        await _docker("rm", "-f", sandbox_id, config=self.config, check=False)
        port = self._ports.pop(sandbox_id, None)
        if port is not None:
            self._inflight_ports.discard(port)
        logger.info("Deleted sandbox %s", sandbox_id)


class PodmanProvider(DockerProvider):
    """`DockerProvider` with the Podman CLI as the default engine.

    Exists so `from agentix.provider.docker import PodmanProvider; PodmanProvider()`
    reads as well as the explicit
    `DockerProvider(DockerProviderConfig(container_engine="podman"))`. The
    `podman` entry point in the `agentix.provider` registry resolves
    here; the `deploy podman` CLI subcommand calls into the same class.
    """

    def __init__(self, config: DockerProviderConfig | None = None):
        super().__init__(config or DockerProviderConfig(container_engine="podman"))


# ── deploy CLI subcommands ─────────────────────────────────────────────
#
# The `agentix deploy` group is a thin discovery shell — every provider
# plugin registers its own `click.Command` via the
# `[project.entry-points."agentix.deploy.commands"]` group in its
# `pyproject.toml`. The core CLI knows nothing about docker- or
# podman-specific flags; it only adds whatever subcommands are
# installed.
#
# `common_options` supplies the three flags every backend needs
# (`bundle` positional, `--name`, `--platform`, `--format`).
# `print_deploy_result` renders the returned `DeployedBundle` (text or
# JSON; includes the `hints` shell-comment block).


def _deploy_via_docker_engine(
    bundle: Path,
    *,
    container_engine: str,
    run_args: tuple[str, ...],
    name: str | None,
    platform: str | None,
    output_format: str,
) -> None:
    """Shared body for the `docker` and `podman` deploy subcommands.

    Both subcommands construct a `DockerProvider` whose `container_engine`
    differs only in default — every other knob (`--run-arg`, `--name`,
    `--platform`, `--format`) is identical. Extracting the body keeps the
    two `@click.command` wrappers down to one line of business logic.
    """
    config = DockerProviderConfig(container_engine=container_engine, run_args=list(run_args))
    provider = DockerProvider(config)
    result = asyncio.run(provider.deploy_bundle(bundle, name=name, platform=platform))
    print_deploy_result(result, output_format=output_format)


@click.command(
    "docker",
    short_help="Deploy a bundle tar to docker (local cache extract).",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@common_options
@click.option(
    "--container-engine",
    default="docker",
    metavar="ENGINE",
    help="Docker-compatible CLI to invoke (`docker`, `nerdctl`, …). Default: docker.",
)
@click.option(
    "--run-arg",
    "run_args",
    multiple=True,
    metavar="ARG",
    help="Extra argument for the in-sandbox `<engine> run` invocation; repeat for multiple args.",
)
def deploy_docker_cmd(
    bundle: Path,
    name: str | None,
    platform: str | None,
    output_format: str,
    container_engine: str,
    run_args: tuple[str, ...],
) -> None:
    """Extract a portable bundle tar into a content-addressed host cache
    and print the cache path (the `SandboxConfig.bundle` ref).
    """
    _deploy_via_docker_engine(
        bundle,
        container_engine=container_engine,
        run_args=run_args,
        name=name,
        platform=platform,
        output_format=output_format,
    )


@click.command(
    "podman",
    short_help="Deploy a bundle tar to podman (local cache extract).",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@common_options
@click.option(
    "--container-engine",
    default="podman",
    metavar="ENGINE",
    help="Docker-compatible CLI to invoke. Default: podman.",
)
@click.option(
    "--run-arg",
    "run_args",
    multiple=True,
    metavar="ARG",
    help="Extra argument for the in-sandbox `<engine> run` invocation; repeat for multiple args.",
)
def deploy_podman_cmd(
    bundle: Path,
    name: str | None,
    platform: str | None,
    output_format: str,
    container_engine: str,
    run_args: tuple[str, ...],
) -> None:
    """Extract a portable bundle tar into a content-addressed host cache
    and print the cache path (the `SandboxConfig.bundle` ref).
    """
    _deploy_via_docker_engine(
        bundle,
        container_engine=container_engine,
        run_args=run_args,
        name=name,
        platform=platform,
        output_format=output_format,
    )


__all__ = [
    "DockerProvider",
    "DockerProviderConfig",
    "PodmanProvider",
    "deploy_docker_cmd",
    "deploy_podman_cmd",
]
