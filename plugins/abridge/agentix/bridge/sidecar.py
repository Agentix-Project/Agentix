"""Sidecar — manage a local host-side protocol/gateway process.

abridge keeps JSON-schema-specific and ML logic outside its request router.
That logic can live behind any JSON HTTP sidecar. `Sidecar` owns the process
lifecycle: pick a port, spawn the command, wait for health, hand back the base
URL, and tear the process down on exit.

    async with Sidecar(
        command=lambda host, port: ["my-gateway", "--listen", f"{host}:{port}"],
        health_path="/healthz",
    ) as url:
        proxy = Proxy(Forward(url, paths=["/v1/messages"]))
        async with proxy.session(sandbox) as handle:
            ...

`Forward(url, ...)` is decoupled from who launches the process: point it at
a `Sidecar`-managed URL, or at an externally-run service. `Sidecar` manages
the process but does not install its executable; callers must provide the
binary on the host. To use an externally-run service, skip `Sidecar` and
pass its URL straight to `Forward`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)

# A command is either a fixed argv (with `{host}` / `{port}` / `{url}`
# tokens substituted) or a factory called with the chosen (host, port).
Command = Sequence[str] | Callable[[str, int], Sequence[str]]

_STDERR_TAIL_LIMIT = 4096
_DRAIN_SHUTDOWN_TIMEOUT = 1.0
_AUTO_PORT_ATTEMPTS = 3
_reserved_ports: set[int] = set()
_reserved_ports_lock = threading.Lock()


def _reserve_free_port(host: str, *, exclude: set[int] | None = None) -> int:
    """Choose and process-locally reserve a port until the child binds it.

    The socket must be closed before an arbitrary sidecar can bind, so this
    cannot eliminate races with other processes.  The reservation does stop
    concurrent ``Sidecar`` instances in this process from receiving the same
    bind-and-close result; an explicit bind failure is retried by ``Sidecar``.
    """
    excluded = exclude or set()
    with _reserved_ports_lock:
        for _ in range(100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, 0))
                port = int(sock.getsockname()[1])
            if port not in excluded and port not in _reserved_ports:
                _reserved_ports.add(port)
                return port
    raise SidecarError(f"could not reserve a free sidecar port on {host}")


def _release_port(port: int) -> None:
    with _reserved_ports_lock:
        _reserved_ports.discard(port)


def _reserve_fixed_port(host: str, port: int) -> None:
    """Fail early if an explicit port is unavailable or already in flight."""
    with _reserved_ports_lock:
        if port in _reserved_ports:
            raise SidecarError(f"sidecar port {port} is already reserved in this process")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
        except OSError as error:
            raise SidecarError(f"sidecar port {host}:{port} is unavailable: {error}") from error
        _reserved_ports.add(port)


def _is_bind_conflict(error: BaseException, stderr: str) -> bool:
    text = f"{error}\n{stderr}".lower()
    return any(
        marker in text
        for marker in (
            "address already in use",
            "eaddrinuse",
            "errno 48",
            "errno 98",
            "only one usage of each socket address",
        )
    )


class SidecarError(RuntimeError):
    """The sidecar process failed to start or become healthy."""


class Sidecar:
    """Launch and supervise a local sidecar process for the life of a
    context manager.

    `command` is the argv to run, either a `Sequence[str]` (whose elements
    may contain `{host}` / `{port}` / `{url}` placeholders) or a callable
    `(host, port) -> Sequence[str]`. `port=0` (default) binds a free port.
    The concrete auto-assigned port is chosen on context entry, not during
    construction. `env` may likewise be a mapping or a `(host, port)` factory.
    `health_path` is GET-polled until it answers `< 400` or `ready_timeout`
    elapses; if the process exits first, its stderr tail is raised.
    """

    def __init__(
        self,
        *,
        command: Command,
        host: str = "127.0.0.1",
        port: int = 0,
        health_path: str = "/healthz",
        env: Mapping[str, str] | Callable[[str, int], Mapping[str, str]] | None = None,
        ready_timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        self._command = command
        self._host = host
        self._requested_port = port
        # Keep port=0 unresolved until __aenter__.  Constructing several
        # sidecars long before they are started otherwise widens the race
        # between bind-and-close allocation and the child process's bind.
        self._port = port
        self._health_path = "/" + health_path.lstrip("/")
        self._env = env
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self._proc: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []
        self._stderr_buffer = bytearray()
        self._stderr_truncated = False
        self._starting = False

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _argv(self) -> list[str]:
        if callable(self._command):
            return list(self._command(self._host, self._port))
        subs = {"host": self._host, "port": str(self._port), "url": self.url}
        return [part.format(**subs) for part in self._command]

    def _spawn_env(self) -> dict[str, str]:
        configured = self._env(self._host, self._port) if callable(self._env) else (self._env or {})
        return {**os.environ, **configured}

    async def __aenter__(self) -> str:
        if self._starting or (self._proc is not None and self._proc.returncode is None):
            raise RuntimeError("sidecar is already running")

        self._starting = True
        attempted_ports: set[int] = set()
        try:
            # Finish any pipe readers left by a process that exited between a
            # previous health check and context teardown before re-entering.
            if self._proc is not None:
                await self._terminate()
            attempts = _AUTO_PORT_ATTEMPTS if self._requested_port == 0 else 1
            for attempt in range(1, attempts + 1):
                reserved_port: int | None = None
                if self._requested_port == 0:
                    reserved_port = _reserve_free_port(self._host, exclude=attempted_ports)
                    attempted_ports.add(reserved_port)
                    self._port = reserved_port
                else:
                    # A caller-specified port is an explicit contract.  Never
                    # silently replace it when the bind fails.
                    self._port = self._requested_port
                    _reserve_fixed_port(self._host, self._port)
                    reserved_port = self._port

                try:
                    await self._spawn()
                    await self._wait_healthy()
                except BaseException as error:
                    stderr = self._stderr_tail()
                    await self._terminate()
                    if (
                        self._requested_port == 0
                        and attempt < attempts
                        and isinstance(error, SidecarError)
                        and _is_bind_conflict(error, stderr)
                    ):
                        logger.warning(
                            "abridge sidecar: port %d was claimed before bind; retrying",
                            self._port,
                        )
                        self._proc = None
                        continue
                    raise
                finally:
                    if reserved_port is not None:
                        _release_port(reserved_port)

                logger.info("abridge sidecar: healthy at %s", self.url)
                return self.url
        finally:
            self._starting = False

        raise SidecarError(f"sidecar exhausted port retries on {self._host}")

    async def __aexit__(self, *exc: object) -> None:
        await self._terminate()

    async def _spawn(self) -> None:
        self._stderr_buffer.clear()
        self._stderr_truncated = False
        self._drain_tasks.clear()
        self._proc = None
        argv = self._argv()
        logger.info("abridge sidecar: starting %s", " ".join(argv))
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            env=self._spawn_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._drain_tasks = [
            asyncio.create_task(self._drain(self._proc.stdout), name="abridge-sidecar-stdout"),
            asyncio.create_task(
                self._drain(self._proc.stderr, capture_stderr=True),
                name="abridge-sidecar-stderr",
            ),
        ]

    async def _drain(
        self,
        stream: asyncio.StreamReader,
        *,
        capture_stderr: bool = False,
    ) -> None:
        """Drain a child pipe to EOF, optionally retaining a bounded tail."""
        try:
            while chunk := await stream.read(8192):
                if not capture_stderr:
                    continue
                self._stderr_buffer.extend(chunk)
                overflow = len(self._stderr_buffer) - _STDERR_TAIL_LIMIT
                if overflow > 0:
                    del self._stderr_buffer[:overflow]
                    self._stderr_truncated = True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("abridge sidecar: pipe drain failed", exc_info=True)

    async def _wait_healthy(self) -> None:
        assert self._proc is not None
        deadline = time.monotonic() + self._ready_timeout
        health_url = self.url + self._health_path
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if self._proc.returncode is not None:
                    await self._finish_drainers()
                    raise SidecarError(
                        f"sidecar exited (code={self._proc.returncode}) before healthy:\n"
                        + self._stderr_tail()
                    )
                with contextlib.suppress(httpx.HTTPError):
                    resp = await client.get(health_url)
                    if resp.status_code < 400:
                        return
                if time.monotonic() >= deadline:
                    raise SidecarError(
                        f"sidecar did not become healthy at {health_url} within "
                        f"{self._ready_timeout:.0f}s\n" + self._stderr_tail()
                    )
                await asyncio.sleep(self._poll_interval)

    def _stderr_tail(self) -> str:
        tail = bytes(self._stderr_buffer).decode(errors="replace")
        if self._stderr_truncated:
            return f"[stderr truncated to last {_STDERR_TAIL_LIMIT} bytes]\n{tail}"
        return tail

    async def _finish_drainers(self) -> None:
        tasks = tuple(self._drain_tasks)
        if not tasks:
            return
        try:
            _, pending = await asyncio.wait(tasks, timeout=_DRAIN_SHUTDOWN_TIMEOUT)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._drain_tasks.clear()
            raise
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._drain_tasks.clear()

    async def _terminate(self) -> None:
        proc = self._proc
        if proc is None:
            await self._finish_drainers()
            return
        try:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()
        except asyncio.CancelledError:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
            raise
        finally:
            await self._finish_drainers()


__all__ = ["Command", "Sidecar", "SidecarError"]
