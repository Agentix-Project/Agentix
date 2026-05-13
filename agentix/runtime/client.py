"""Async HTTP client for the agentix runtime server.

Wraps:
  - typed remote-call dispatch: `RuntimeClient.remote(fn, *args, **kwargs)`,
    where `fn` is a stub function imported from a closure's Python package.
    Routing key is `fn.__module__`; result is decoded into `fn`'s return type.
  - built-in `/exec`, `/upload`, `/download`, plus `/closures` introspection.

There is no longer a generic HTTP reverse-proxy to closures — `remote` is
the single typed entry point.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import httpx
from pydantic import TypeAdapter

from agentix.models import (
    ClosureInfo,
    ExecRequest,
    ExecResponse,
    HealthResponse,
    RemoteError,
    RemoteRequest,
    RemoteResponse,
    UploadResponse,
)

P = ParamSpec("P")
R = TypeVar("R")


class RemoteCallError(RuntimeError):
    """Raised when a remote closure impl returns a non-ok RemoteResponse."""

    def __init__(self, package: str, method: str, error: RemoteError):
        super().__init__(f"{package}.{method}: {error.type}: {error.message}")
        self.package = package
        self.method = method
        self.error = error


class RuntimeClient:
    """Async client for the agentix runtime server."""

    def __init__(self, base_url: str, timeout: float = 300):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    # ── lifecycle ────────────────────────────────────────────────

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── runtime server endpoints ─────────────────────────────────

    async def health(self) -> HealthResponse:
        r = await self._client.get("/health")
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())

    async def closures(self) -> list[ClosureInfo]:
        r = await self._client.get("/closures")
        r.raise_for_status()
        return [ClosureInfo.model_validate(x) for x in r.json()]

    # ── typed remote call ────────────────────────────────────────

    async def remote(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute `fn` in the sandbox and return its typed result.

        `fn` must be a stub function exported by a closure's Python package
        (e.g. `from agentix_closures.claude_code import run`). Routing uses
        `fn.__module__`; method name is `fn.__name__`; the return type is
        decoded via `inspect.signature(fn).return_annotation`.
        """
        package = fn.__module__
        method = fn.__name__
        body = RemoteRequest(package=package, method=method, args=list(args), kwargs=dict(kwargs))
        r = await self._client.post("/_remote", json=body.model_dump())
        r.raise_for_status()
        resp = RemoteResponse.model_validate(r.json())
        if not resp.ok:
            assert resp.error is not None
            raise RemoteCallError(package=package, method=method, error=resp.error)
        return_ann = inspect.signature(fn).return_annotation
        if return_ann is inspect.Signature.empty:
            return resp.value  # type: ignore[return-value]
        return TypeAdapter(return_ann).validate_python(resp.value)

    # ── runtime I/O primitives (exec / upload / download) ───────

    @staticmethod
    def _exec_body(
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float | None,
        max_output: int | None = None,
        paths_from: list[str] | None = None,
    ) -> dict[str, Any]:
        return ExecRequest(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            max_output=max_output,
            paths_from=paths_from,
        ).model_dump(exclude_none=True)

    async def run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        max_output: int | None = None,
        paths_from: list[str] | None = None,
    ) -> ExecResponse:
        """Buffered shell exec: run `command` and return the full captured output.

        `paths_from` prepends the `bin/` of the listed closures (by Python
        package path) to PATH for this command only. Pass `["*"]` to include
        every mounted closure.
        """
        body = self._exec_body(command, cwd, env, timeout, max_output, paths_from)
        r = await self._client.post("/exec", json=body)
        r.raise_for_status()
        return ExecResponse.model_validate(r.json())

    async def run_stream(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        paths_from: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream exec output as SSE events.

        Yields decoded event dicts like:
            {"event": "stdout", "stream": "stdout", "data": "..."}
            {"event": "exit",   "exit_code": 0}
        """
        body = self._exec_body(command, cwd, env, timeout, paths_from=paths_from)
        buf = b""
        async with self._client.stream(
            "POST", "/exec", json=body, headers={"accept": "text/event-stream"}
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    event_bytes, buf = buf.split(b"\n\n", 1)
                    event = _parse_sse_event(event_bytes)
                    if event is not None:
                        yield event

    async def upload(self, local_path: str | Path, dest: str) -> UploadResponse:
        """Upload a local file to `dest` inside the sandbox."""
        p = Path(local_path)
        with open(p, "rb") as f:
            r = await self._client.post(
                "/upload",
                files={"file": (p.name, f)},
                data={"path": dest},
            )
        r.raise_for_status()
        return UploadResponse.model_validate(r.json())

    async def download(self, path: str, local_path: str | Path) -> int:
        """Stream a sandbox file down to `local_path`."""
        r = await self._client.get("/download", params={"path": path})
        r.raise_for_status()
        lp = Path(local_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_bytes(r.content)
        return len(r.content)


def _parse_sse_event(raw: bytes) -> dict[str, Any] | None:
    """Parse a single SSE event block into a dict. Returns None for keepalives."""
    event: str | None = None
    data_lines: list[str] = []
    for line in raw.decode(errors="replace").splitlines():
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    payload = "\n".join(data_lines)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"data": payload}
    if event:
        parsed.setdefault("event", event)
    return parsed
