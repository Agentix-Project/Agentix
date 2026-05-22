"""Docker deployment: sandbox CRUD via local Docker.

Design:

  Two images, one container. `config.bundle` is the generic
  Agentix bundle from `agentix build` (carries `/nix/runtime/bin/` and
  the full Python closure under `/nix/store/...`). `config.image` is
  the task-specific base the workload runs against. The runtime is
  overlaid onto the task container's `/nix` so the agentix-server
  entrypoint and its store paths resolve regardless of the task
  image's distribution.

  Overlay mechanism: a per-bundle stopped "carrier" container
  declares the runtime's `/nix` as a VOLUME (set in the image's config
  by `agentix build`); sandbox containers re-use it with
  `--volumes-from <carrier>:ro`. One stopped carrier per distinct
  bundle — they cost only metadata.

  (`--mount type=image,subpath=nix` would let us skip the carrier
  entirely with one docker invocation, but `subpath` isn't yet
  supported on image mounts in stable Docker — landing in a future
  release. Switch when available.)

  Sandbox create:
      docker create [--platform <platform>] --name <carrier> <bundle>
      docker run [--platform <platform>] -d --name <sid> \\
         -p 127.0.0.1:<port>:<port> \\
         -e AGENTIX_BIND_PORT=<port> \\
         --volumes-from <carrier>:ro \\
         --entrypoint /bin/sh \\
         <image> -c '<inject runtime env; exec agentix-server>'

  `agentix-server` binds to the port from `AGENTIX_BIND_PORT`. We pick
  a free host port, publish the same port to loopback, and health-check
  `/health` on it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import socket
from uuid import uuid4

import httpx

from agentix.deployment.base import Deployment, Sandbox, SandboxConfig, SandboxId, SandboxInfo

logger = logging.getLogger("agentix.deployment.docker")

_RUNTIME_ENTRYPOINT = "/bin/sh"
_RUNTIME_BOOTSTRAP = r"""
set -eu
agentix_prepend_path() {
  name="$1"
  added="$2"
  tracking="AGENTIX_ADDED_${name}"
  eval "current=\${$name-}"
  eval "tracked=\${$tracking-}"
  if [ -n "$current" ]; then
    export "$name=$added:$current"
  else
    export "$name=$added"
  fi
  if [ -n "$tracked" ]; then
    export "$tracking=$tracked:$added"
  else
    export "$tracking=$added"
  fi
}
agentix_prepend_path PATH "/nix/runtime/venv/bin:/nix/runtime/bin"
agentix_prepend_path LD_LIBRARY_PATH "/nix/runtime/lib"
agentix_prepend_path LIBRARY_PATH "/nix/runtime/lib"
agentix_prepend_path CPATH "/nix/runtime/include"
agentix_prepend_path C_INCLUDE_PATH "/nix/runtime/include"
agentix_prepend_path CPLUS_INCLUDE_PATH "/nix/runtime/include"
agentix_prepend_path PKG_CONFIG_PATH "/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig"
agentix_prepend_path CMAKE_PREFIX_PATH "/nix/runtime"
exec /nix/runtime/venv/bin/agentix-server
""".strip()


async def _docker(*args: str, check: bool = True, retries: int = 0) -> tuple[int, bytes, bytes]:
    attempt = 0
    delay = 2.0
    while True:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        rc = proc.returncode or 0
        if not check or rc == 0:
            return rc, stdout, stderr
        if attempt >= retries or not _is_transient_docker_error(stderr):
            raise RuntimeError(f"docker {args[0]} failed: {stderr.decode(errors='replace')}")
        attempt += 1
        logger.warning(
            "docker %s failed with transient error; retrying in %.1fs (%d/%d)",
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


def _carrier_name(bundle: str, platform: str | None = None) -> str:
    """Stable name for the stopped container that holds a runtime's /nix volume."""
    key = f"{bundle}@{platform}" if platform else bundle
    slug = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"agentix-runtime-{slug}"


class DockerDeployment(Deployment):
    """Sandbox CRUD via local Docker."""

    def __init__(self):
        self._ports: dict[SandboxId, int] = {}  # sandbox_id → host port

    @staticmethod
    def _allocate_port() -> int:
        # Ask the kernel for any free TCP port. There's still a small
        # TOCTOU window before the container binds, but no worse than a
        # linear probe and without the seed parameter.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def _ensure_carrier(self, bundle: str, platform: str | None) -> str:
        """Create (if missing) a stopped container exposing bundle's /nix.

        Stopped containers cost only metadata; one per distinct
        bundle/platform is enough regardless of how many sandboxes share it.
        """
        carrier = _carrier_name(bundle, platform)
        rc, _, _ = await _docker("inspect", carrier, check=False)
        if rc == 0:
            current_image = await _image_id(bundle)
            carrier_image = await _container_image_id(carrier)
            if current_image and carrier_image and current_image != carrier_image:
                await _docker("rm", "-f", carrier, check=False)
            else:
                return carrier
        platform_args = ["--platform", platform] if platform else []
        await _docker("create", *platform_args, "--name", carrier, bundle)
        return carrier

    async def create(self, config: SandboxConfig) -> Sandbox:
        sandbox_id = SandboxId(f"agentix-{uuid4().hex[:8]}")
        port = self._allocate_port()

        env_args: list[str] = ["-e", f"AGENTIX_BIND_PORT={port}"]
        if config.env:
            for k, v in config.env.items():
                env_args.extend(["-e", f"{k}={v}"])

        carrier = await self._ensure_carrier(config.bundle, config.platform)
        platform_args = ["--platform", config.platform] if config.platform else []
        await _docker(
            "run",
            *platform_args,
            "-d",
            "--name",
            sandbox_id,
            "-p",
            f"127.0.0.1:{port}:{port}",
            *env_args,
            "--volumes-from",
            f"{carrier}:ro",
            "--entrypoint",
            _RUNTIME_ENTRYPOINT,
            config.image,
            "-c",
            _RUNTIME_BOOTSTRAP,
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
            check=False,
        )
        status = stdout.decode().strip() if rc == 0 else "unknown"
        return SandboxInfo(
            sandbox_id=sandbox_id,
            runtime_url=f"http://localhost:{port}",
            status=status,
        )

    async def delete(self, sandbox_id: SandboxId) -> None:
        await _docker("rm", "-f", sandbox_id, check=False)
        self._ports.pop(sandbox_id, None)
        logger.info("Deleted sandbox %s", sandbox_id)


async def _image_id(image: str) -> str | None:
    rc, stdout, _ = await _docker("image", "inspect", "-f", "{{.Id}}", image, check=False)
    if rc != 0:
        return None
    return stdout.decode().strip() or None


async def _container_image_id(container: str) -> str | None:
    rc, stdout, _ = await _docker("inspect", "-f", "{{.Image}}", container, check=False)
    if rc != 0:
        return None
    return stdout.decode().strip() or None
