"""Runtime worker client — one worker subprocess for remote callables.

Bridges the runtime server's Socket.IO handlers to the worker process.
Owns one worker subprocess per server process, routes calls by `call_id`,
shuts the worker down with the server. Also forwards extension SIO
traffic via generic `sio_*` frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import sys
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from agentix.runtime.env import AGENTIX_ADDED_PATH
from agentix.runtime.server.worker.invoker import CallableInvoker
from agentix.runtime.shared.framing import read_frame, write_frame
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse

logger = logging.getLogger("agentix.runtime.server.worker.client")


class WorkerExited(RuntimeError):
    """The worker subprocess died (crash, OOM-kill, exit) mid-call.

    `returncode` is the process exit status when known: negative means
    killed by that signal (e.g. -9 = SIGKILL, the OOM-killer's signature).
    """

    def __init__(self, message: str, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


def _exit_detail(returncode: int | None) -> str:
    if returncode is None:
        return "stdout closed"
    if returncode < 0:
        sig = -returncode
        hint = " (likely OOM-killed)" if sig == 9 else ""
        return f"killed by signal {sig}{hint}"
    return f"exit code {returncode}"


_WORKER_START_TIMEOUT = 15.0
_WORKER_BOOTSTRAP = """
import os
import sys

_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
_import_root = os.environ.pop("AGENTIX_WORKER_IMPORT_ROOT", "")
if _import_root and _import_root not in sys.path:
    sys.path.insert(0, _import_root)
from agentix.runtime.server.worker.process import main

main()
"""
_WORKER_IMPORT_ROOT = Path(__file__).resolve().parents[4]
# Plugin `default.nix` derivations are symlink-joined into this path
# inside the bundle runtime tree (see `agentix/builder/flake.nix`). Worker code
# (`subprocess.run("claude", ...)`, `c.remote(cc.run, ...)`, ...) must
# be able to find those binaries by bare name.
_RUNTIME_BIN_PATH = "/nix/runtime/bin"
_RUNTIME_LIB_PATH = "/nix/runtime/lib"
_RUNTIME_INCLUDE_PATH = "/nix/runtime/include"
_STRIPPED_ENV = {
    "LD_PRELOAD",
    "PYTHONPATH",
    "PYTHONHOME",
    "LOCALE_ARCHIVE",
    "SSL_CERT_FILE",
}
_STRIPPED_ENV_PREFIXES = ("NIX_", "FONTCONFIG_")
_RUNTIME_PATH_ADDITIONS = {
    "LD_LIBRARY_PATH": (_RUNTIME_LIB_PATH,),
    "LIBRARY_PATH": (_RUNTIME_LIB_PATH,),
    "CPATH": (_RUNTIME_INCLUDE_PATH,),
    "C_INCLUDE_PATH": (_RUNTIME_INCLUDE_PATH,),
    "CPLUS_INCLUDE_PATH": (_RUNTIME_INCLUDE_PATH,),
    "PKG_CONFIG_PATH": (
        "/nix/runtime/lib/pkgconfig",
        "/nix/runtime/share/pkgconfig",
    ),
    "CMAKE_PREFIX_PATH": ("/nix/runtime",),
}


def _join_path_entries(entries: Iterable[str]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry or entry in seen:
            continue
        parts.append(entry)
        seen.add(entry)
    return os.pathsep.join(parts)


def _tracking_var(name: str) -> str:
    return f"AGENTIX_ADDED_{name}"


def _prepend_recorded_path_entries(env: dict[str, str], name: str, entries: Iterable[str]) -> None:
    added = _join_path_entries(entries)
    env[name] = _join_path_entries([*added.split(os.pathsep), *env.get(name, "").split(os.pathsep)])
    tracking_name = _tracking_var(name)
    env[tracking_name] = _join_path_entries([*env.get(tracking_name, "").split(os.pathsep), *added.split(os.pathsep)])


def _clean_worker_env(runtime_bin_dir: Path | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _STRIPPED_ENV and not any(key.startswith(prefix) for prefix in _STRIPPED_ENV_PREFIXES)
    }
    # Build PATH from: the venv's bin (`runtime_bin_dir`), the bundle's
    # symlink-join (`/nix/runtime/bin`), then the parent environment's
    # PATH. Inside the bundle runtime tree the first two are siblings and both
    # must be searchable; outside the bundle, only the first one exists.
    parts: list[str] = []
    if runtime_bin_dir is not None:
        parts.append(str(runtime_bin_dir))
    parts.append(_RUNTIME_BIN_PATH)
    parts.extend(env.get("PATH", "").split(os.pathsep))
    env["PATH"] = _join_path_entries(parts)

    added_path = []
    added_path.extend(env.get(AGENTIX_ADDED_PATH, "").split(os.pathsep))
    if runtime_bin_dir is not None:
        added_path.append(str(runtime_bin_dir))
    added_path.append(_RUNTIME_BIN_PATH)
    env[AGENTIX_ADDED_PATH] = _join_path_entries(added_path)

    for name, entries in _RUNTIME_PATH_ADDITIONS.items():
        _prepend_recorded_path_entries(env, name, entries)
    return env


class WorkerBackend(Protocol):
    """Internal execution backend boundary."""

    async def call(self, request: RemoteRequest) -> RemoteResponse: ...
    async def send_inbound(self, namespace: str, event: str, data: Any) -> None: ...
    async def shutdown(self) -> None: ...


SioFrameHandler = Callable[[dict[str, Any]], Any]
"""Called from the worker read loop for each `sio_emit` or
`sio_subscribe` frame the worker produces. The SIO server layer
installs one; the in-process backend has no transport hop."""


class _InProcessWorker:
    """In-process worker: resolves and calls fn in the server's own loop.
    Test fixture only — production routes through `_SubprocessWorker`."""

    def __init__(self) -> None:
        self._invoker = CallableInvoker()

    def _resolve_or_error(self, request: RemoteRequest) -> tuple[Any | None, RemoteError | None]:
        try:
            return request.callable.resolve(), None
        except Exception as exc:
            return None, RemoteError(type=type(exc).__name__, message=str(exc))

    async def call(self, request: RemoteRequest) -> RemoteResponse:
        fn, err = self._resolve_or_error(request)
        if err is not None:
            return RemoteResponse(ok=False, error=err)
        return await self._invoker.call(fn, request)

    async def send_inbound(self, namespace: str, event: str, data: Any) -> None:
        # In-process backend has no extensions running in a separate
        # process; inbound forwarding is a no-op.
        return

    async def shutdown(self) -> None:
        return


class _SubprocessWorker:
    """Single subprocess worker."""

    def __init__(
        self,
        python: str,
        runtime_bin_dir: Path | None = None,
        sio_handler: SioFrameHandler | None = None,
    ) -> None:
        self._python = python
        self._runtime_bin_dir = runtime_bin_dir
        self._worker_id = _new_id()[:8]

        self._proc: asyncio.subprocess.Process | None = None
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._boot_error: dict[str, Any] | None = None
        self._read_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

        self._pending: dict[str, asyncio.Future] = {}
        self._cancel_tasks: set[asyncio.Task] = set()
        self._sio_handler = sio_handler

    async def start(self) -> None:
        env = _clean_worker_env(self._runtime_bin_dir)
        env["AGENTIX_WORKER_IMPORT_ROOT"] = str(_WORKER_IMPORT_ROOT)
        env["AGENTIX_WORKER_ID"] = self._worker_id
        env["AGENTIX_LOG_CONTEXT"] = env.get(
            "AGENTIX_WORKER_LOG_CONTEXT",
            "sandbox-{uname}-worker-{id}",
        )
        if worker_log_format := env.get("AGENTIX_WORKER_LOG_FORMAT"):
            env["AGENTIX_LOG_FORMAT"] = worker_log_format
        self._proc = await asyncio.create_subprocess_exec(
            self._python,
            "-c",
            _WORKER_BOOTSTRAP,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        ready_task = asyncio.create_task(self._ready.wait())
        closed_task = asyncio.create_task(self._closed.wait())
        assert self._proc is not None
        proc_task = asyncio.create_task(self._proc.wait())
        try:
            done, pending = await asyncio.wait(
                {ready_task, closed_task, proc_task},
                timeout=_WORKER_START_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if not done:
                await self.shutdown()
                raise TimeoutError(f"runtime worker did not become ready within {_WORKER_START_TIMEOUT:.0f}s")
            if ready_task not in done:
                rc = self._proc.returncode
                await self.shutdown()
                detail = f"exit code {rc}" if rc is not None else "stdout closed"
                raise RuntimeError(f"runtime worker exited before ready ({detail})")
        finally:
            for task in (ready_task, closed_task, proc_task):
                if not task.done():
                    task.cancel()
        if self._boot_error is not None:
            await self.shutdown()
            raise RuntimeError(
                f"runtime worker failed to boot: {self._boot_error.get('type')}: {self._boot_error.get('message')}"
            )

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                frame = await read_frame(self._proc.stdout)
                if frame is None:
                    break
                await self._on_frame(frame)
        except Exception:
            logger.exception("runtime worker read loop crashed")
        finally:
            self._closed.set()
            returncode = self._proc.returncode if self._proc is not None else None
            if returncode is None and self._proc is not None:
                # EOF can beat the child reaper, leaving returncode unset;
                # wait briefly so we can report the real exit signal
                # (e.g. -9 = SIGKILL, the OOM-killer's signature).
                with contextlib.suppress(Exception):
                    returncode = await asyncio.wait_for(self._proc.wait(), 2.0)
            detail = _exit_detail(returncode)
            # An OOM/crash gives no Python traceback (the process is gone),
            # so this log line is the only place the cause surfaces.
            logger.error("runtime worker exited: %s", detail)
            exc = WorkerExited(f"runtime worker exited: {detail}", returncode)
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(exc)
            self._pending.clear()

    async def _on_frame(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "ready":
            self._ready.set()
        elif kind == "boot_error":
            self._boot_error = frame.get("error") or {"type": "Unknown", "message": ""}
            self._ready.set()
        elif kind == "result":
            cid = frame.get("call_id", "")
            fut = self._pending.pop(cid, None)
            if fut and not fut.done():
                fut.set_result(RemoteResponse(ok=True, value=frame.get("value")))
        elif kind == "error":
            cid = frame.get("call_id", "")
            err_payload = frame.get("error") or {"type": "Unknown", "message": ""}
            err = RemoteError.model_validate(err_payload)
            fut = self._pending.pop(cid, None)
            if fut and not fut.done():
                fut.set_result(RemoteResponse(ok=False, error=err))
        elif kind in ("sio_emit", "sio_open"):
            if self._sio_handler is not None:
                try:
                    result = self._sio_handler(frame)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.debug("sio frame handler raised; dropping", exc_info=True)
        else:
            logger.warning("runtime worker: unknown frame %r", kind)

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        async with self._send_lock:
            await write_frame(self._proc.stdin, payload)

    def _call_frame(self, cid: str, request: RemoteRequest) -> dict[str, Any]:
        return {
            "type": "call",
            "call_id": cid,
            "callable": str(request.callable),
            "arguments": request.arguments,
        }

    def _schedule_cancel(self, cid: str) -> None:
        t = asyncio.create_task(self._send_cancel(cid))
        self._cancel_tasks.add(t)
        t.add_done_callback(self._cancel_tasks.discard)

    async def _send_cancel(self, cid: str) -> None:
        try:
            await self._send({"type": "cancel", "call_id": cid})
        except Exception:
            logger.debug("cancel send failed for call %r", cid)

    async def call(self, request: RemoteRequest) -> RemoteResponse:
        # If the worker already died, the read loop has exited and will
        # never resolve a future — fail fast instead of hanging forever.
        if self._closed.is_set():
            returncode = self._proc.returncode if self._proc is not None else None
            raise WorkerExited(f"runtime worker exited: {_exit_detail(returncode)}", returncode)
        cid = request.call_id or _new_id()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = fut
        try:
            await self._send(self._call_frame(cid, request))
            return await fut
        finally:
            self._pending.pop(cid, None)
            if not fut.done():
                self._schedule_cancel(cid)

    async def send_inbound(self, namespace: str, event: str, data: Any) -> None:
        """Forward a host-emitted SIO event to the worker."""
        await self._send(
            {
                "type": "sio_inbound",
                "namespace": namespace,
                "event": event,
                "data": data,
            }
        )

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            await self._send({"type": "shutdown"})
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(BaseException):
                await self._read_task


def _new_id() -> str:
    return uuid.uuid4().hex


class RuntimeWorkerClient:
    """Owns one worker process and routes all calls through it."""

    def __init__(self) -> None:
        self._python: str = sys.executable
        self._runtime_bin_dir: Path = Path(sys.executable).parent
        self._worker: WorkerBackend | None = None
        self._spawn_lock = asyncio.Lock()
        self._inprocess = _InProcessWorker()
        # Set by the SIO layer; invoked for each `sio_emit` /
        # `sio_subscribe` frame the worker emits.
        self._sio_handler: SioFrameHandler | None = None

    def set_sio_handler(self, handler: SioFrameHandler | None) -> None:
        self._sio_handler = handler

    def _use_inprocess(self) -> None:
        self._worker = self._inprocess

    async def _get_worker(self) -> WorkerBackend:
        if self._worker is not None:
            return self._worker
        async with self._spawn_lock:
            if self._worker is not None:
                return self._worker
            worker = _SubprocessWorker(
                self._python,
                runtime_bin_dir=self._runtime_bin_dir,
                sio_handler=self._sio_handler,
            )
            await worker.start()
            self._worker = worker
            return worker

    async def shutdown(self) -> None:
        if self._worker is not None:
            await self._worker.shutdown()

    async def call(self, request: RemoteRequest) -> RemoteResponse:
        worker = await self._get_worker()
        return await worker.call(request)

    async def send_inbound(self, namespace: str, event: str, data: Any) -> None:
        """Forward a host-side SIO event into the worker process."""
        worker = await self._get_worker()
        await worker.send_inbound(namespace, event, data)


__all__ = ["RuntimeWorkerClient", "WorkerBackend"]
