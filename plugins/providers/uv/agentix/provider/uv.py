"""uv SandboxProvider — run the Agentix runtime from a uv-materialized venv.

A lightweight provider that skips the Docker/Nix bundle entirely. `uv`
materializes a virtualenv for the target project (so its importable callables
plus `agentixx` core are present), then the runtime server is launched as a
local subprocess (`python -m uvicorn agentix.runtime.server.app:app`). The
worker subprocess the server spawns inherits that interpreter
(`sys.executable`), so `await sandbox.remote(fn, ...)` runs `fn` against the
project's real dependencies — no container, no rebuild.

Aimed at local dev / eval / CI where Docker is unavailable or too slow. It
trades isolation for speed: the runtime runs on the host, not in a sandboxed
container. For untrusted code or hard resource limits, use a container
provider (`docker` / `apptainer`) or a managed backend instead.

    from agentix.provider.uv import UvProvider, UvProviderConfig

    provider = UvProvider(UvProviderConfig(project="."))   # uv pip install -e .
    try:
        async with provider.session(SandboxConfig(image="uv", bundle="uv")) as sandbox:
            result = await sandbox.remote(my_rollout, task=task)
    finally:
        await provider.aclose()   # removes the venv this provider materialized

`SandboxConfig.image` / `bundle` are unused here (there is no image or bundle);
pass any placeholder. Only `SandboxConfig.env` is honored — merged into the
runtime server's environment. Backend settings live in `UvProviderConfig`,
mirroring how other providers take a backend config object.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from agentix.provider.base import (
    Sandbox,
    SandboxConfig,
    SandboxId,
    SandboxInfo,
    SandboxProvider,
)

logger = logging.getLogger("agentix.provider.uv")

_RUNTIME_APP = "agentix.runtime.server.app:app"


@dataclass
class UvProviderConfig:
    """Backend config for `UvProvider`.

    Either point at a `project` to materialize a fresh venv (`uv venv` +
    `uv pip install -e <project>` — the project must depend on `agentixx`), or
    point `reuse_venv` at an existing interpreter env to skip materialization
    (fast iteration / CI where the env is prebuilt).
    """

    project: str | None = None
    python: str = "3.12"
    index_url: str | None = None
    extra_index_url: tuple[str, ...] = ()
    install: tuple[str, ...] = ()
    reuse_venv: str | None = None
    uv_bin: str = "uv"
    host: str = "127.0.0.1"
    ws: str = "auto"
    health_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.project is None and self.reuse_venv is None:
            raise ValueError("UvProviderConfig needs either `project` or `reuse_venv`")

    def resolved_uv_bin(self) -> str:
        """The uv executable to shell out to. An explicit `uv_bin` wins; the
        bare default prefers the binary shipped by the `uv` wheel (this
        package depends on it), because the venv's bin/ is not on PATH under
        systemd/cron/absolute-path launches — exactly the environments where
        no system-wide uv exists."""
        if self.uv_bin != "uv":
            return self.uv_bin
        try:
            from uv import find_uv_bin

            return find_uv_bin()
        except (ImportError, FileNotFoundError):
            return self.uv_bin


# Bounded log tail retained per sandbox: last N chunks of <= 64KiB each. Big
# enough for a useful crash diagnostic, small enough to never matter.
_TAIL_CHUNKS = 8


@dataclass
class _Running:
    proc: asyncio.subprocess.Process
    port: int
    # The runtime's merged stdout/stderr MUST be drained for the sandbox's
    # lifetime: asyncio's flow control pauses the pipe once ~192KiB is
    # buffered, and a server that logs past that blocks mid-write — wedging
    # every in-flight rollout. The drain keeps only a bounded tail.
    drain: asyncio.Task
    tail: deque[bytes]


async def _drain_stdout(proc: asyncio.subprocess.Process, tail: deque[bytes]) -> None:
    if proc.stdout is None:
        return
    while True:
        chunk = await proc.stdout.read(65536)
        if not chunk:
            return
        tail.append(chunk)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _run(*argv: str, timeout: float = 1800.0) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{argv[0]!r} not found — install uv (`pip install uv`) or point "
            "UvProviderConfig.uv_bin at the executable"
        ) from e
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        raise
    if proc.returncode != 0:
        tail = out.decode(errors="replace")[-2000:] if out else ""
        raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(argv)}\n{tail}")


class UvProvider(SandboxProvider):
    """Provision sandboxes as a runtime server launched from a uv venv."""

    def __init__(self, config: UvProviderConfig | None = None) -> None:
        if config is None:
            config = UvProviderConfig(project=".")
        self.config = config
        self._running: dict[SandboxId, _Running] = {}
        self._venv: Path | None = None
        self._owned_venv_root: Path | None = None
        self._venv_lock = asyncio.Lock()
        self._inflight_ports: set[int] = set()

    def _allocate_port(self) -> int:
        # Ask the kernel for a free TCP port, then reserve it in-process: the
        # subprocess only binds it seconds later (interpreter boot + imports),
        # so two concurrent creates that each bind-and-close could otherwise
        # collide on the same number. Same guard as DockerProvider.
        for _ in range(100):
            port = _free_port()
            if port not in self._inflight_ports:
                self._inflight_ports.add(port)
                return port
        raise RuntimeError("could not allocate a free host port")

    async def _ensure_venv(self) -> Path:
        """Materialize (once) and return the venv whose `python` runs the
        runtime. Reused across every `create()` on this provider."""
        if self.config.reuse_venv is not None:
            return Path(self.config.reuse_venv)
        async with self._venv_lock:
            if self._venv is not None:
                return self._venv
            root = Path(tempfile.mkdtemp(prefix="agentix-uv-"))
            uv_bin = self.config.resolved_uv_bin()
            try:
                venv = root / "venv"
                await _run(uv_bin, "venv", "--python", self.config.python, str(venv))
                py = str(venv / "bin" / "python")
                idx: list[str] = []
                if self.config.index_url:
                    idx += ["--index-url", self.config.index_url]
                for extra in self.config.extra_index_url:
                    idx += ["--extra-index-url", extra]
                targets: list[str] = []
                if self.config.project is not None:
                    targets += ["-e", self.config.project]
                targets += list(self.config.install)
                if targets:
                    await _run(uv_bin, "pip", "install", "--python", py, *idx, *targets)
            except BaseException:
                # A failed materialization must not orphan the temp root — a
                # retry loop would leak one partial venv per attempt.
                shutil.rmtree(root, ignore_errors=True)
                raise
            self._venv = venv
            self._owned_venv_root = root
            return venv

    async def create(self, config: SandboxConfig) -> Sandbox:
        venv = await self._ensure_venv()
        python = str(venv / "bin" / "python")
        port = self._allocate_port()

        env = dict(os.environ)
        env.setdefault("AGENTIX_LOG_CONTEXT", "uv-sandbox-{uname}")
        if config.env:
            env.update(config.env)

        cmd = [
            python, "-m", "uvicorn", _RUNTIME_APP,
            "--host", self.config.host, "--port", str(port),
            "--log-level", "error", "--ws", self.config.ws, "--lifespan", "on",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        except BaseException:
            self._inflight_ports.discard(port)
            raise
        sandbox_id = SandboxId(f"uv-{uuid.uuid4().hex[:12]}")
        tail: deque[bytes] = deque(maxlen=_TAIL_CHUNKS)
        drain = asyncio.create_task(_drain_stdout(proc, tail))
        self._running[sandbox_id] = _Running(proc=proc, port=port, drain=drain, tail=tail)
        try:
            await self._wait_healthy(sandbox_id, port, proc)
        except BaseException:
            await self.delete(sandbox_id)
            raise
        return Sandbox(
            sandbox_id=sandbox_id,
            runtime_url=f"http://{self.config.host}:{port}",
            status="running",
        )

    async def _wait_healthy(self, sandbox_id: SandboxId, port: int, proc: asyncio.subprocess.Process) -> None:
        # Raw TCP GET /health — never via an HTTP client that honors proxy env
        # vars, which would hang a loopback probe behind a corp proxy.
        attempts = max(1, int(self.config.health_timeout / 0.5))
        for _ in range(attempts):
            if proc.returncode is not None:
                # The drain task owns the pipe — give it a moment to see EOF,
                # then report the bounded tail it kept. asyncio.wait (unlike
                # wait_for) does NOT cancel the task on timeout: a grandchild
                # holding the pipe open must not turn this diagnostic into a
                # CancelledError when delete() later awaits the drain.
                out = b""
                running = self._running.get(sandbox_id)
                if running is not None:
                    await asyncio.wait({running.drain}, timeout=2)
                    out = b"".join(running.tail)
                raise RuntimeError(
                    f"runtime server (uv) exited rc={proc.returncode} before health: "
                    f"{out.decode(errors='replace')[-2000:]}"
                )
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, port), timeout=2
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
                with contextlib.suppress(OSError):
                    await writer.wait_closed()
            await asyncio.sleep(0.5)
        raise TimeoutError(f"runtime server (uv) not healthy on :{port}")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        running = self._running.get(sandbox_id)
        if running is None:
            raise KeyError(f"Sandbox not found: {sandbox_id}")
        status = "running" if running.proc.returncode is None else "exited"
        return SandboxInfo(
            sandbox_id=sandbox_id,
            runtime_url=f"http://{self.config.host}:{running.port}",
            status=status,
        )

    async def delete(self, sandbox_id: SandboxId) -> None:
        running = self._running.pop(sandbox_id, None)
        if running is None:
            return
        await self._terminate(running.proc, sandbox_id)
        self._inflight_ports.discard(running.port)
        # After process death the pipe reaches EOF and the drain task ends on
        # its own; cancel only if it doesn't (e.g. an orphaned grandchild
        # still holds the write end open).
        done, _ = await asyncio.wait({running.drain}, timeout=2)
        if not done:
            running.drain.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running.drain

    async def _terminate(self, proc: asyncio.subprocess.Process, sandbox_id: SandboxId) -> None:
        if proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            logger.warning("uv runtime %s did not exit after SIGTERM; SIGKILL", sandbox_id)
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5.0)

    async def aclose(self) -> None:
        """Terminate every running sandbox and remove a venv this provider
        materialized. An externally supplied `reuse_venv` is left untouched."""
        for sandbox_id in list(self._running):
            await self.delete(sandbox_id)
        if self._owned_venv_root is not None:
            shutil.rmtree(self._owned_venv_root, ignore_errors=True)
            self._owned_venv_root = None
            self._venv = None
