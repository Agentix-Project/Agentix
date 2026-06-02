"""Apptainer provider: sandbox CRUD via local apptainer.

Targets HPC / shared-cluster environments where Docker is absent but
`apptainer` (formerly Singularity) is available and unprivileged-
friendly.

Two artifacts, one container, same contract as `DockerProvider`:

  - `config.bundle` is the portable tar bundle produced by
    `agentix build`. Its `nix/` tree is extracted once per
    bundle digest under a host scratch dir and reused across sandboxes.
  - `config.image` is an apptainer-native task image reference
    (`docker://...`, `library://...`, `oras://...`, or a local `.sif`).
    Converted on first use into a cached SIF.

The runtime is overlaid by bind-mounting the extracted `nix/` tree to
`/nix:ro` inside the apptainer container; the
`/nix/runtime/bootstrap.sh` entry point and its store paths therefore
resolve regardless of the task image's distribution.

Apptainer shares the host network namespace by default, so the
runtime server's port is reachable on `localhost` with no per-sandbox
network setup. We pick a free port, pass it via `AGENTIX_BIND_PORT`,
and `/health`-check it.

Default isolation flags: `--userns --no-init --writable-tmpfs
--cleanenv`. The user-namespace path works on hosts where pid1
doesn't have `CAP_SYS_ADMIN` in the initial mount namespace (e.g.
inside a capability-restricted scheduler runtime); `--cleanenv`
keeps the container's environment clear of host-side noise (most
visibly an `LD_PRELOAD=/usr/lib64/libcuda.so` that GPU schedulers
set, which spams ld.so warnings inside CPU-only task images). On a
fully permissive host you can swap in the stricter `--containall`
family via the `AGENTIX_APPTAINER_FLAGS` env override.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import posixpath
import shutil
import socket
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from agentix.provider.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo, SandboxProvider
from agentix.runtime import BIND_PORT_ENV, BUNDLE_NIX_ROOT, BUNDLE_RUNTIME_ENTRYPOINT

logger = logging.getLogger("agentix.provider.apptainer")

_DEFAULT_CACHE = Path.home() / ".cache" / "agentix" / "apptainer"


def _apptainer_bin() -> str:
    """Return the apptainer binary to invoke (overridable via env)."""
    return os.environ.get("AGENTIX_APPTAINER_BIN") or "apptainer"


def _isolation_args() -> list[str]:
    """Return the isolation flags passed to `apptainer exec`.

    Defaults to `--userns --no-init --writable-tmpfs --cleanenv`.
    This shape works in capability-restricted hosts (e.g. inside a
    Ray worker runtime where pid1 doesn't have `CAP_SYS_ADMIN` in
    the initial mount namespace), at the cost of slightly weaker
    filesystem isolation than `--containall` would give on a
    permissive host. `--cleanenv` keeps the container's env clean of
    host noise (notably `LD_PRELOAD=/usr/lib64/libcuda.so` on GPU
    nodes, which spams `cannot open shared object file` warnings
    when the task image lacks libcuda). `--env K=V` arguments still
    pass through, so `AGENTIX_BIND_PORT` and user `config.env` are
    unaffected.

    Override via `AGENTIX_APPTAINER_FLAGS` (whitespace-separated). To
    add `--containall` back on a permissive host, set:

        AGENTIX_APPTAINER_FLAGS="--containall --no-init --writable-tmpfs"
    """
    override = os.environ.get("AGENTIX_APPTAINER_FLAGS")
    if override:
        return override.split()
    return ["--userns", "--no-init", "--writable-tmpfs", "--cleanenv"]


def _cache_root() -> Path:
    """Process-wide scratch directory for extracted bundles and SIFs."""
    root = Path(os.environ.get("AGENTIX_APPTAINER_CACHE") or _DEFAULT_CACHE)
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundles").mkdir(exist_ok=True)
    (root / "sifs").mkdir(exist_ok=True)
    return root


async def _run(*args: str, check: bool = True) -> tuple[int, bytes, bytes]:
    """Run a subprocess capturing stdout/stderr.

    Centralised so tests can monkeypatch a recorder. Mirrors the docker
    backend's `_docker` helper.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    rc = proc.returncode or 0
    if check and rc != 0:
        raise RuntimeError(
            f"{args[0]} {' '.join(args[1:])} failed (rc={rc}): "
            f"{stderr.decode(errors='replace')}"
        )
    return rc, stdout, stderr


def _bundle_digest(bundle_tar: Path) -> str:
    """Stable id for a bundle tar — drives the cache key.

    Tries `manifest.json` first (deterministic, cheap) and falls back to
    a streaming sha256 of the tar bytes.
    """
    try:
        with tarfile.open(bundle_tar, "r:*") as tar:
            member = tar.getmember("manifest.json")
            f = tar.extractfile(member)
            if f is not None:
                manifest = json.loads(f.read().decode())
                digest = manifest.get("digest")
                if isinstance(digest, str) and digest:
                    return digest.replace(":", "_")
    except (tarfile.TarError, KeyError, json.JSONDecodeError):
        pass
    h = hashlib.sha256()
    with bundle_tar.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_bundle(bundle_tar: Path, target: Path) -> Path:
    """Materialize `bundle.tar` to `<target>/nix/` if not already there.

    The tar's top-level layout is `nix/` (the runtime tree) plus a
    sibling `manifest.json`. We extract only `nix/` because that's what
    the runtime needs at `/nix`.
    """
    nix_root = target / "nix"
    if (nix_root / "runtime" / "bootstrap.sh").exists():
        return nix_root

    def checked_nix_member_name(name: str) -> str:
        normalized = posixpath.normpath(name)
        if normalized in {"", "."} or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise RuntimeError(f"bundle tar produced unsafe member: {name!r}")
        if normalized != "nix" and not normalized.startswith("nix/"):
            raise RuntimeError(f"bundle tar produced non-/nix member: {name!r}")
        return normalized

    def ensure_safe_parent(root: Path, path: Path) -> None:
        try:
            relative_parent = path.parent.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"bundle tar member escaped extraction root: {path}") from exc
        current = root
        for part in relative_parent.parts:
            current = current / part
            if os.path.lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise RuntimeError(f"bundle tar member parent is not a directory: {current}")
            else:
                current.mkdir()

    def remove_existing_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif os.path.lexists(path):
            path.unlink()

    def extract_nix_member(tar: tarfile.TarFile, member: tarfile.TarInfo, root: Path) -> None:
        name = checked_nix_member_name(member.name)
        member_path = root / name
        ensure_safe_parent(root, member_path)

        if member.isdir():
            if os.path.lexists(member_path):
                if member_path.is_symlink() or not member_path.is_dir():
                    raise RuntimeError(
                        f"bundle tar cannot replace non-directory with directory: {name}"
                    )
            else:
                member_path.mkdir()
            return

        remove_existing_path(member_path)
        if member.issym():
            os.symlink(member.linkname, member_path)
            return
        if member.islnk():
            try:
                link_name = checked_nix_member_name(member.linkname)
            except RuntimeError as exc:
                raise RuntimeError(f"bundle tar produced unsafe hard-link target: {member.linkname!r}") from exc
            link_target = root / link_name
            if not os.path.lexists(link_target):
                raise RuntimeError(f"bundle tar hard-link target does not exist: {member.linkname!r}")
            os.link(link_target, member_path)
            return
        if member.isfile():
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"bundle tar has unreadable file member: {name}")
            with source, member_path.open("wb") as f:
                shutil.copyfileobj(source, f)
            os.chmod(member_path, member.mode & 0o7777)
            return

        raise RuntimeError(f"bundle tar contains unsupported member type: {name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(bundle_tar, "r:*") as tar:
            for member in tar:
                name = member.name
                if not (name == "nix" or name.startswith("nix/")):
                    continue
                extract_nix_member(tar, member, tmp_root)
        remove_existing_path(target)
        tmp_root.replace(target)

    if not nix_root.is_dir():
        raise RuntimeError(f"bundle {bundle_tar} did not contain a `nix/` tree")
    return nix_root


def _image_cache_key(image: str) -> str:
    return hashlib.sha256(image.encode()).hexdigest()[:16]


async def _ensure_sif(image: str) -> Path:
    """Convert `image` to a cached SIF if needed; pass-through for `.sif` inputs.

    Apptainer transparently pulls `docker://`, `library://`, `oras://`,
    and `shub://` references via `apptainer pull`. Local `.sif` files
    are used in place.
    """
    if image.endswith(".sif") and Path(image).is_file():
        return Path(image).resolve()
    sif = _cache_root() / "sifs" / f"{_image_cache_key(image)}.sif"
    if sif.is_file():
        return sif
    sif.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Pulling apptainer image %s -> %s", image, sif)
    await _run(_apptainer_bin(), "pull", "--force", str(sif), image)
    return sif


class ApptainerProvider(SandboxProvider):
    """Sandbox CRUD via local apptainer.

    State is in-process: `_procs` maps `SandboxId` to the live
    `apptainer exec` subprocess and the port we asked the runtime
    server to bind. `delete()` terminates the subprocess, which tears
    down the container; `get()` reports liveness by polling
    `/health`.
    """

    def __init__(self) -> None:
        self._procs: dict[SandboxId, asyncio.subprocess.Process] = {}
        self._ports: dict[SandboxId, int] = {}

    @staticmethod
    def _allocate_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def _prepare_bundle(self, bundle_ref: str) -> Path:
        bundle_tar = Path(bundle_ref).expanduser().resolve()
        if not bundle_tar.is_file():
            raise FileNotFoundError(
                f"apptainer backend expects a tar bundle path; got {bundle_ref!r}"
            )
        digest = _bundle_digest(bundle_tar)
        target = _cache_root() / "bundles" / digest
        nix_root = await asyncio.to_thread(_extract_bundle, bundle_tar, target)
        return nix_root

    async def create(self, config: SandboxConfig) -> Sandbox:
        nix_root = await self._prepare_bundle(config.bundle)
        sif = await _ensure_sif(config.image)

        sandbox_id = SandboxId(f"agentix-{uuid4().hex[:8]}")
        port = self._allocate_port()

        env_args: list[str] = ["--env", f"{BIND_PORT_ENV}={port}"]
        if config.env:
            for k, v in config.env.items():
                env_args.extend(["--env", f"{k}={v}"])

        args = [
            _apptainer_bin(),
            "exec",
            *_isolation_args(),
            "--bind",
            f"{nix_root}:{BUNDLE_NIX_ROOT}:ro",
            *env_args,
            str(sif),
            BUNDLE_RUNTIME_ENTRYPOINT,
        ]
        logger.info("apptainer exec %s (port=%d)", sandbox_id, port)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._procs[sandbox_id] = proc
        self._ports[sandbox_id] = port

        try:
            await self._wait_healthy(port, proc)
        except BaseException:
            await self._terminate(sandbox_id)
            raise

        return Sandbox(
            sandbox_id=sandbox_id,
            runtime_url=f"http://localhost:{port}",
            status="running",
        )

    async def _wait_healthy(self, port: int, proc: asyncio.subprocess.Process) -> None:
        base_url = f"http://localhost:{port}"
        # Health probe MUST not go through any HTTP client that
        # consults environment proxy vars — on hosts behind a corp
        # proxy, an `http_proxy=...` leaks into loopback requests and
        # hangs them. Use a raw TCP connect + tiny HTTP request so we
        # only ever talk to `127.0.0.1` directly.
        for _ in range(240):
            if proc.returncode is not None:
                stderr = (await proc.stderr.read()) if proc.stderr else b""
                raise RuntimeError(
                    f"apptainer exec exited rc={proc.returncode} before runtime came up: "
                    f"{stderr.decode(errors='replace')}"
                )
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), timeout=2
                )
            except (TimeoutError, OSError):
                await asyncio.sleep(0.5)
                continue
            try:
                writer.write(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                status_line = await asyncio.wait_for(reader.readline(), timeout=2)
                if status_line.startswith(b"HTTP/1.") and b" 200 " in status_line:
                    return
            except (TimeoutError, OSError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Runtime server not alive at {base_url}")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        port = self._ports.get(sandbox_id)
        if port is None:
            raise KeyError(f"Sandbox not found: {sandbox_id}")
        proc = self._procs.get(sandbox_id)
        if proc is None or proc.returncode is not None:
            status = "exited"
        else:
            status = "running"
        return SandboxInfo(
            sandbox_id=sandbox_id,
            runtime_url=f"http://localhost:{port}",
            status=status,
        )

    async def delete(self, sandbox_id: SandboxId) -> None:
        await self._terminate(sandbox_id)
        self._ports.pop(sandbox_id, None)

    async def _terminate(self, sandbox_id: SandboxId) -> None:
        proc = self._procs.pop(sandbox_id, None)
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            logger.warning("apptainer exec %s did not exit after SIGTERM; SIGKILL", sandbox_id)
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5.0)


def _diagnostics() -> dict[str, str]:
    """Tiny `apptainer --version` probe used by smoke tests."""
    return {
        "apptainer_bin": _apptainer_bin(),
        "cache_root": str(_cache_root()),
        "host": socket.gethostname(),
    }


def cli_probe() -> None:
    """Print diagnostics — useful when shelling into a worker host.

    Exposed as `agentix-apptainer-probe` console entry-point.
    """
    has = shutil.which(_apptainer_bin())
    info = _diagnostics()
    info["apptainer_on_path"] = "yes" if has else "no"
    print(json.dumps(info, indent=2))
