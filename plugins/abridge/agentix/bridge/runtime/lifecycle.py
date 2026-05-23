"""Sandbox proxy lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import sys
import uuid
from dataclasses import dataclass

import uvicorn
from agentix.bridge.runtime.forwarder import build_forwarder_app
from agentix.bridge.runtime.namespace import get_namespace
from agentix.bridge.types import ProxyHandle
from agentix.bridge.util import join_url_path

logger = logging.getLogger("agentix.bridge.runtime")


@dataclass
class _RunningProxy:
    handle: ProxyHandle
    forwarder_server: uvicorn.Server
    forwarder_task: asyncio.Task
    process: asyncio.subprocess.Process
    log_task: asyncio.Task


_running: dict[str, _RunningProxy] = {}


async def start_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    mode: str = "reverse:https://api.anthropic.com",
    forwarder_host: str = "127.0.0.1",
    forwarder_port: int = 0,
    request_timeout: float = 600.0,
    extra_mitm_args: list[str] | None = None,
) -> ProxyHandle:
    """Start sandbox-local mitmproxy plus a local SIO forwarder."""
    ns = get_namespace()
    app = build_forwarder_app(ns=ns, request_timeout=request_timeout)
    f_config = uvicorn.Config(app, host=forwarder_host, port=forwarder_port, log_level="warning")
    f_server = uvicorn.Server(f_config)
    f_task = asyncio.create_task(f_server.serve())
    bound_forwarder_port = await _wait_uvicorn_started(f_server, forwarder_port)
    forwarder_url = f"http://{forwarder_host}:{bound_forwarder_port}"

    bound_proxy_port = port or _free_tcp_port(host)
    args = [
        "--mode",
        mode,
        "--listen-host",
        host,
        "--listen-port",
        str(bound_proxy_port),
        "--set",
        "block_global=false",
        "--quiet",
    ]
    if extra_mitm_args:
        args.extend(extra_mitm_args)

    env = dict(os.environ)
    env["ABRIDGE_HOOK_URL"] = join_url_path(forwarder_url, "/hook")
    env.setdefault("ABRIDGE_TRACE", "1")
    env.setdefault("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com,localhost,127.0.0.1")

    code = f"from agentix.bridge.mitm.cli import main; raise SystemExit(main({args!r}))"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    log_task = asyncio.create_task(_drain_mitm_logs(process.stdout))
    try:
        await _wait_port(host, bound_proxy_port, process)
    except Exception:
        process.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        f_server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(f_task, timeout=5)
        raise

    sid = uuid.uuid4().hex
    handle = ProxyHandle(
        id=sid,
        url=f"http://{host}:{bound_proxy_port}",
        port=bound_proxy_port,
        forwarder_url=forwarder_url,
        forwarder_port=bound_forwarder_port,
        mode=mode,
    )
    _running[sid] = _RunningProxy(
        handle=handle,
        forwarder_server=f_server,
        forwarder_task=f_task,
        process=process,
        log_task=log_task,
    )
    return handle


async def stop_proxy(handle: ProxyHandle) -> None:
    rec = _running.pop(handle.id, None)
    if rec is None:
        return
    rec.process.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(rec.process.wait(), timeout=5)
    if rec.process.returncode is None:
        rec.process.kill()
        await rec.process.wait()
    rec.log_task.cancel()
    await asyncio.gather(rec.log_task, return_exceptions=True)

    rec.forwarder_server.should_exit = True
    with contextlib.suppress(Exception):
        await asyncio.wait_for(rec.forwarder_task, timeout=5)


async def _wait_uvicorn_started(server: uvicorn.Server, configured_port: int) -> int:
    for _ in range(200):
        if server.started and server.servers:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("abridge mitm forwarder did not start within 10s")

    bound_port = configured_port
    for srv in server.servers:
        for sock in srv.sockets:
            bound_port = int(sock.getsockname()[1])
            break
        if bound_port:
            break
    return bound_port


async def _wait_port(host: str, port: int, process: asyncio.subprocess.Process) -> None:
    for _ in range(200):
        if process.returncode is not None:
            raise RuntimeError(f"mitmproxy exited early with code {process.returncode}")
        with contextlib.suppress(OSError):
            with socket.create_connection((host, port), timeout=0.2):
                return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"mitmproxy did not listen on {host}:{port} within 10s")


async def _drain_mitm_logs(stream: asyncio.StreamReader) -> None:
    async for raw in stream:
        text = raw.decode(errors="replace").rstrip()
        if text:
            logger.info("mitmproxy: %s", text)


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
