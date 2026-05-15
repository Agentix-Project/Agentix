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

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import agentix.trace as _trace
from agentix.runtime.models import ExecRequest, ExecResponse, UploadResponse

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
    `["*"]` expands to every currently-registered closure. Unknown packages
    are silently dropped.
    """
    from agentix.runtime.server.app import registry

    pkg_list = registry.packages() if packages == ["*"] else packages
    out: list[str] = []
    for pkg in pkg_list:
        mount = registry.mount_for(pkg)
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


# ── LLM proxy ────────────────────────────────────────────────────
#
# Closure SDKs are pointed at `http://127.0.0.1:8000/_llm/<provider>/...`
# instead of the provider's real base URL. The runtime forwards every
# request to the upstream and emits two trace events per call:
#
#   {"kind": "llm_request",  "payload": {provider, method, path, body}}
#   {"kind": "llm_response", "payload": {provider, status, body}}
#
# For streaming responses (SSE) the response body in the trace is the full
# concatenated payload, recorded after the stream finishes. Headers from
# the caller are forwarded verbatim except hop-by-hop hops + `host` /
# `content-length`. Auth lives in those headers — the caller's API key
# never enters this code, just passes through.

_LLM_UPSTREAMS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai":    "https://api.openai.com",
}

_LLM_PROXY_BODY_LIMIT = int(os.environ.get("AGENTIX_LLM_PROXY_TRACE_LIMIT", str(64 * 1024)))


def _trace_body(raw: bytes) -> object:
    """Decode a request/response body for inclusion in a trace event. JSON
    is preserved as a dict; everything else becomes a (truncated) string."""
    try:
        return json.loads(raw)
    except Exception:
        return raw[:_LLM_PROXY_BODY_LIMIT].decode(errors="replace")


@router.api_route(
    "/_llm/{provider}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
)
async def llm_proxy(provider: str, path: str, request: Request) -> Response:
    upstream = _LLM_UPSTREAMS.get(provider)
    if upstream is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown LLM provider {provider!r}; known: {sorted(_LLM_UPSTREAMS)}",
        )
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }
    url = f"{upstream}/{path}"

    _trace.emit("llm_request", {
        "provider": provider,
        "method": request.method,
        "path": "/" + path,
        "body": _trace_body(body) if body else None,
    })

    client = httpx.AsyncClient(timeout=None)
    upstream_req = client.build_request(
        request.method, url,
        headers=headers, content=body, params=request.query_params,
    )
    upstream_resp = await client.send(upstream_req, stream=True)
    out_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
    }

    async def _stream_and_trace():
        collected = bytearray()
        try:
            async for chunk in upstream_resp.aiter_raw():
                if len(collected) < _LLM_PROXY_BODY_LIMIT:
                    collected.extend(chunk[:_LLM_PROXY_BODY_LIMIT - len(collected)])
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()
            _trace.emit("llm_response", {
                "provider": provider,
                "status": upstream_resp.status_code,
                "body": _trace_body(bytes(collected)),
            })

    return StreamingResponse(
        _stream_and_trace(),
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )
