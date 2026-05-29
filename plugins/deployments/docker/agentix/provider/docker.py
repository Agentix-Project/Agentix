"""Docker deployment: sandbox CRUD via a Docker-compatible CLI.

Design:

  `agentix build` produces a portable tar containing `manifest.json`
  and a full `/nix` runtime tree. `agentix deploy docker|podman`
  materializes that tar into a content-addressed host cache directory.
  `config.bundle` is the cache root returned by deploy; its `nix/`
  child is bind-mounted read-only into each sandbox at `/nix`.

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
  `DockerProviderConfig(container_bin="podman", ...)` when it
  provides the Docker-compatible commands this backend needs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import posixpath
import shlex
import shutil
import socket
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, field_validator

from agentix.provider.base import (
    MaterializedBundle,
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
    """Docker-compatible CLI settings for the deployment backend."""

    container_bin: str = Field(
        default="docker",
        description="Docker-compatible CLI binary, e.g. `docker` or `podman`.",
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
        description="Host directory for materialized bundle caches. Default: ~/.cache/agentix/bundles.",
    )

    @field_validator("container_bin")
    @classmethod
    def _validate_container_bin(cls, value: str) -> str:
        if not value:
            raise ValueError("container_bin must not be empty")
        return value

    @field_validator("run_args", mode="before")
    @classmethod
    def _parse_extra_args(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return _split_shell_args(value, "deployment extra args")
        if isinstance(value, (list, tuple)):
            # Shlex-split each item so a whole shell-style sub-flag can be passed
            # as one value, e.g. --container-run-arg "--device /dev/fuse".
            return [token for item in value for token in _split_shell_args(str(item), "deployment extra args")]
        return value

    @field_validator("gpu_args", mode="before")
    @classmethod
    def _parse_gpu_args(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return _split_shell_args(value, "deployment gpu_args")
        return value

    def to_provider(self) -> DockerProvider:
        """Construct a `DockerProvider` from this config."""
        return DockerProvider(self)


def _default_config(config: DockerProviderConfig | None = None) -> DockerProviderConfig:
    return config or DockerProviderConfig()


def _container_bin(config: DockerProviderConfig | None = None) -> str:
    return _default_config(config).container_bin


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


def _checked_nix_member_name(name: str) -> str:
    normalized = posixpath.normpath(name)
    if normalized in {"", "."} or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(f"bundle tar produced unsafe member: {name!r}")
    if normalized != "nix" and not normalized.startswith("nix/"):
        raise RuntimeError(f"bundle tar produced non-/nix member: {name!r}")
    return normalized


def _ensure_safe_parent(root: Path, path: Path) -> None:
    current = root
    for part in path.parent.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError(f"bundle tar member parent is not a directory: {current}")
        else:
            current.mkdir()


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _extract_nix_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    root: Path,
) -> None:
    name = _checked_nix_member_name(member.name)
    target = root / name
    _ensure_safe_parent(root, target)

    if member.isdir():
        if os.path.lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError(f"bundle tar cannot replace non-directory with directory: {name}")
        else:
            target.mkdir()
        return

    _remove_existing_path(target)
    if member.issym():
        os.symlink(member.linkname, target)
        return
    if member.islnk():
        link_target = root / _checked_nix_member_name(member.linkname)
        os.link(link_target, target)
        return
    if member.isfile():
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"bundle tar has unreadable file member: {name}")
        with source, target.open("wb") as f:
            shutil.copyfileobj(source, f)
        os.chmod(target, member.mode & 0o7777)
        return

    raise RuntimeError(f"bundle tar contains unsupported member type: {name}")


def _extract_bundle_to_cache(
    bundle_tar: Path,
    manifest: dict[str, object],
    cache_root: Path,
) -> None:
    if (cache_root / "nix" / "runtime" / "bootstrap.sh").is_file():
        return
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{cache_root.name}.", dir=cache_root.parent) as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(bundle_tar, "r:*") as tar:
            for member in tar:
                if member.name == "manifest.json":
                    continue
                _extract_nix_member(tar, member, tmp_root)
        if not (tmp_root / "nix" / "runtime" / "bootstrap.sh").is_file():
            raise RuntimeError(f"bundle {bundle_tar} does not contain nix/runtime/bootstrap.sh")
        (tmp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _remove_existing_path(cache_root)
        tmp_root.replace(cache_root)


def _bundle_nix_path(bundle: str) -> Path:
    root = Path(bundle).expanduser().resolve()
    nix = root / "nix"
    if not nix.is_dir():
        raise RuntimeError(f"materialized bundle {bundle!r} does not contain a nix/ directory")
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
    bin_name = _container_bin(config)
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                bin_name,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"container CLI {bin_name!r} not found on PATH. Install Docker "
                f"(https://docs.docker.com/get-docker/) or Podman "
                f"(https://podman.io/docs/installation), or set container_bin."
            ) from exc
        stdout, stderr = await proc.communicate()
        rc = proc.returncode or 0
        if not check or rc == 0:
            return rc, stdout, stderr
        if attempt >= retries or not _is_transient_docker_error(stderr):
            raise RuntimeError(f"{bin_name} {args[0]} failed: {stderr.decode(errors='replace')}")
        attempt += 1
        logger.warning(
            "%s %s failed with transient error; retrying in %.1fs (%d/%d)",
            bin_name,
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

    async def materialize_bundle(
        self,
        bundle: Path,
        *,
        name: str | None = None,
        platform: str | None = None,
    ) -> MaterializedBundle:
        bundle_tar = bundle.expanduser().resolve()
        if not bundle_tar.is_file():
            raise FileNotFoundError(f"bundle tar not found: {bundle}")
        manifest = _bundle_manifest(bundle_tar)
        bundle_name = _bundle_display_name(manifest, name)
        materialized_platform = _bundle_platform(manifest, platform)
        cache_root = _bundle_cache_root(manifest, bundle_tar, self.config)
        _extract_bundle_to_cache(bundle_tar, manifest, cache_root)
        return MaterializedBundle(
            bundle=str(cache_root),
            platform=materialized_platform,
            metadata={"cache": str(cache_root), "name": bundle_name},
        )

    @staticmethod
    def _allocate_port() -> int:
        # Ask the kernel for any free TCP port. There's still a small
        # TOCTOU window before the container binds, but no worse than a
        # linear probe and without the seed parameter.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

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
        self._ports.pop(sandbox_id, None)
        logger.info("Deleted sandbox %s", sandbox_id)


class PodmanProvider(SandboxProvider):
    """Docker-compatible provider configured to use the Podman CLI."""

    def __init__(self, config: DockerProviderConfig | None = None):
        self._deployment = DockerProvider(config or DockerProviderConfig(container_bin="podman"))

    async def materialize_bundle(
        self,
        bundle: Path,
        *,
        name: str | None = None,
        platform: str | None = None,
    ) -> MaterializedBundle:
        return await self._deployment.materialize_bundle(bundle, name=name, platform=platform)

    async def create(self, config: SandboxConfig) -> Sandbox:
        return await self._deployment.create(config)

    async def delete(self, sandbox_id: SandboxId) -> None:
        await self._deployment.delete(sandbox_id)

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        return await self._deployment.get(sandbox_id)


__all__ = ["DockerProvider", "DockerProviderConfig", "PodmanProvider"]
