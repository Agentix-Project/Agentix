"""Runtime built-ins — exec / upload / download.

Mounted at the runtime server's root. These are the minimum set of
operations an orchestrator needs to drive a sandbox (run commands, place
files, fetch results) independent of any closure that happens to be
mounted. Directory listing and any other file inspection is done via
`/exec` (e.g. `ls -la`, `find`, `stat`).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from agentix.models import ExecRequest, ExecResponse, UploadResponse

UPLOAD_ROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/workspace")).resolve()
MAX_OUTPUT_BYTES = int(os.environ.get("AGENTIX_MAX_OUTPUT_BYTES", str(10 * 1024 * 1024)))

# Env vars stripped before we fork a user-space subprocess.
# Our runtime is Nix-built, so os.environ is pre-loaded with Nix paths
# (LD_LIBRARY_PATH pointing at Nix store libs, PYTHONPATH / FONTCONFIG / NIX_*
# set by the wrapper). Leaking these into a subprocess run by a host-image
# binary (e.g. the target image's /bin/bash) causes glibc ABI mismatches and
# silent library override bugs. Scrub at the boundary.
_RUNTIME_ONLY_ENV = {
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "PYTHONHOME",
    "LOCALE_ARCHIVE",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
    "SSL_CERT_FILE",
    "NIX_SSL_CERT_FILE",
}


router = APIRouter()


def _clean_env(
    extra: dict[str, str] | None,
    prepend_path: list[str] | None = None,
) -> dict[str, str]:
    """Env for a user subprocess: scrubbed base + optional PATH prefixes and
    caller-supplied overrides.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _RUNTIME_ONLY_ENV and not k.startswith("NIX_")
    }
    if prepend_path:
        base_path = env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["PATH"] = ":".join([*prepend_path, base_path])
    if extra:
        env.update(extra)
    return env


CLOSURE_MOUNT_ROOT = os.environ.get("AGENTIX_CLOSURE_MOUNT_ROOT", "/mnt")


def _resolve_closure_bins(packages: list[str]) -> list[str]:
    """Turn closure package paths into their `entry/bin` paths.
    `["*"]` expands to every currently-registered closure.

    Uses agentix.runtime.server._mount_paths as the package → mount map;
    unknown packages are silently dropped.
    """
    from agentix.runtime.server import _mount_paths, registry

    if packages == ["*"]:
        return [str(_mount_paths[p] / "entry" / "bin") for p in registry.packages()]
    out: list[str] = []
    for pkg in packages:
        mount = _mount_paths.get(pkg)
        if mount is not None:
            out.append(str(mount / "entry" / "bin"))
    return out


# ── exec ─────────────────────────────────────────────────────────


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> str:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = limit - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            chunks.append(b"\n[truncated at %d bytes]" % limit)
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks).decode(errors="replace")


@router.post("/exec")
async def exec_endpoint(req: ExecRequest, request: Request):
    """Run a shell command. SSE when `Accept: text/event-stream`; else buffered JSON."""
    prepend = None
    if req.paths_from:
        prepend = _resolve_closure_bins(req.paths_from)
    env = _clean_env(req.env, prepend_path=prepend)
    max_output = req.max_output or MAX_OUTPUT_BYTES

    if "text/event-stream" in request.headers.get("accept", ""):
        return StreamingResponse(
            _exec_sse(req.command, req.cwd, env, req.timeout),
            media_type="text/event-stream",
        )
    result = await _exec_buffered(req.command, req.cwd, env, req.timeout, max_output)
    return JSONResponse(result.model_dump())


async def _exec_buffered(
    command: str,
    cwd: str | None,
    env: dict[str, str],
    timeout: float | None,
    max_output: int,
) -> ExecResponse:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    try:
        async def _collect():
            stdout = await _read_capped(proc.stdout, max_output)
            stderr = await _read_capped(proc.stderr, max_output)
            await proc.wait()
            return stdout, stderr

        stdout, stderr = await asyncio.wait_for(_collect(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return ExecResponse(exit_code=-1, stdout="", stderr=f"Command timed out after {timeout}s")
    return ExecResponse(exit_code=proc.returncode or 0, stdout=stdout, stderr=stderr)


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


async def _exec_sse(
    command: str,
    cwd: str | None,
    env: dict[str, str],
    timeout: float | None,
) -> AsyncIterator[bytes]:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    async def _pump(stream: asyncio.StreamReader, tag: str, queue: asyncio.Queue):
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            await queue.put((tag, chunk))
        await queue.put((tag, None))

    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump(proc.stdout, "stdout", queue)),
        asyncio.create_task(_pump(proc.stderr, "stderr", queue)),
    ]
    open_streams = {"stdout", "stderr"}

    try:
        deadline = None
        if timeout is not None:
            deadline = asyncio.get_event_loop().time() + timeout
        while open_streams:
            remaining = None
            if deadline is not None:
                remaining = max(deadline - asyncio.get_event_loop().time(), 0)
                if remaining == 0:
                    proc.kill()
                    yield _sse("error", {"message": f"Command timed out after {timeout}s"})
                    break
            try:
                tag, chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
            except TimeoutError:
                proc.kill()
                yield _sse("error", {"message": f"Command timed out after {timeout}s"})
                break
            if chunk is None:
                open_streams.discard(tag)
                continue
            yield _sse(tag, {"stream": tag, "data": chunk.decode(errors="replace")})
        await proc.wait()
        yield _sse("exit", {"exit_code": proc.returncode or 0})
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


# ── upload / download / ls ───────────────────────────────────────


def _resolve_within(path: str) -> Path:
    p = Path(path).resolve()
    if not p.is_relative_to(UPLOAD_ROOT):
        raise HTTPException(
            status_code=403, detail=f"Path {p} outside allowed root {UPLOAD_ROOT}"
        )
    return p


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), path: str = Form(...)) -> UploadResponse:
    p = _resolve_within(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    p.write_bytes(data)
    return UploadResponse(path=str(p), size=len(data))


@router.get("/download")
async def download(path: str):
    p = _resolve_within(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    if p.is_dir():
        raise HTTPException(status_code=400, detail=f"Is a directory: {path}")

    def _iter():
        with open(p, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iter(), media_type="application/octet-stream")
