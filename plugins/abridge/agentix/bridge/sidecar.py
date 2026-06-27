"""Sidecar — manage a local host-side protocol/gateway process.

abridge core is a shape-blind transport; all protocol and ML logic lives
*behind* a sidecar — a `cc_convert` translator binary, a `tito`
pretokenize gateway, or any HTTP service. `Sidecar` owns that process's
lifecycle: pick a port, spawn the command, wait for health, hand back the
base URL, and tear the process down on exit.

    async with Sidecar(
        command=lambda host, port: ["cc_convert_sidecar", "--listen", f"{host}:{port}"],
        health_path="/healthz",
    ) as url:
        proxy = Proxy(Forward(url, paths=["/v1/messages"]))
        async with proxy.session(sandbox) as handle:
            ...

`Forward(url, ...)` is decoupled from who launches the process: point it
at a `Sidecar`-managed URL, or at an externally-run service. The default
is abridge-managed (self-contained); the externally-run mode is just
"don't open a `Sidecar`, pass the URL straight to `Forward`".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)

# A command is either a fixed argv (with `{host}` / `{port}` / `{url}`
# tokens substituted) or a factory called with the chosen (host, port).
Command = Sequence[str] | Callable[[str, int], Sequence[str]]


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class SidecarError(RuntimeError):
    """The sidecar process failed to start or become healthy."""


class Sidecar:
    """Launch and supervise a local sidecar process for the life of a
    context manager.

    `command` is the argv to run, either a `Sequence[str]` (whose elements
    may contain `{host}` / `{port}` / `{url}` placeholders) or a callable
    `(host, port) -> Sequence[str]`. `port=0` (default) binds a free port.
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
        env: Mapping[str, str] | None = None,
        ready_timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        self._command = command
        self._host = host
        self._port = port or _free_port(host)
        self._health_path = "/" + health_path.lstrip("/")
        self._env = {**os.environ, **(env or {})}
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _argv(self) -> list[str]:
        if callable(self._command):
            return list(self._command(self._host, self._port))
        subs = {"host": self._host, "port": str(self._port), "url": self.url}
        return [part.format(**subs) for part in self._command]

    async def __aenter__(self) -> str:
        argv = self._argv()
        logger.info("abridge sidecar: starting %s", " ".join(argv))
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            env=self._env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._wait_healthy()
        except BaseException:
            await self._terminate()
            raise
        logger.info("abridge sidecar: healthy at %s", self.url)
        return self.url

    async def __aexit__(self, *exc: object) -> None:
        await self._terminate()

    async def _wait_healthy(self) -> None:
        assert self._proc is not None
        deadline = time.monotonic() + self._ready_timeout
        health_url = self.url + self._health_path
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if self._proc.returncode is not None:
                    raise SidecarError(
                        f"sidecar exited (code={self._proc.returncode}) before healthy:\n"
                        + await self._stderr_tail()
                    )
                with contextlib.suppress(httpx.HTTPError):
                    resp = await client.get(health_url)
                    if resp.status_code < 400:
                        return
                if time.monotonic() >= deadline:
                    raise SidecarError(
                        f"sidecar did not become healthy at {health_url} within "
                        f"{self._ready_timeout:.0f}s\n" + await self._stderr_tail()
                    )
                await asyncio.sleep(self._poll_interval)

    async def _stderr_tail(self, limit: int = 4096) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        with contextlib.suppress(Exception):
            data = await asyncio.wait_for(self._proc.stderr.read(limit), timeout=1.0)
            return data.decode(errors="replace")
        return ""

    async def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()


__all__ = ["Command", "Sidecar", "SidecarError"]
