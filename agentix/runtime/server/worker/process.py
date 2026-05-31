"""Runtime worker subprocess.

Receives CALL frames from the parent server over stdin, executes the
resolved callable, writes RESULT (or ERROR) frames to stdout. Also
hosts the sandbox-side `agentix.sio` channel: extensions inside the
worker can emit / subscribe / request across the SIO connection via
generic `sio_*` frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
import traceback
from typing import Any

from agentix import sio as _sio
from agentix.runtime.server.worker.invoker import CallableInvoker
from agentix.runtime.shared import MAX_MESSAGE_BYTES
from agentix.runtime.shared.callables import RemoteCallable
from agentix.runtime.shared.framing import FrameTooLarge, read_frame, write_frame
from agentix.runtime.shared.idents import CallId
from agentix.runtime.shared.models import RemoteError, RemoteRequest
from agentix.utils import log as _log
from agentix.utils.log._bridge import emit_worker_record
from agentix.utils.log._config import LOG_CONTEXT_ATTR, get_log_context
from agentix.utils.trace._bridge import install_worker_bridge

logger = logging.getLogger("agentix.runtime.server.worker.process")


def _err(exc: BaseException) -> dict[str, Any]:
    return RemoteError(
        type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    ).model_dump()


class Worker:
    """One process serving remote callable invocations."""

    def __init__(self) -> None:
        self._invoker = CallableInvoker()
        self._calls: dict[str, asyncio.Task] = {}
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._shutdown = asyncio.Event()
        self._outbound_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._drainer: asyncio.Task | None = None
        self._stdio_tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        # The server↔worker frame pipe arrives on fd 0 (stdin) / fd 1
        # (stdout). User code inside a remote call routinely spawns
        # subprocesses (claude, git, ...) that INHERIT fd 0/1 — and a
        # child reading stdin (claude does) would steal frame bytes,
        # desyncing the protocol and hanging every later call.
        #
        # Move the framing onto private fds and point fd 0 at /dev/null, so
        # inherited stdin is harmless. fd 1 becomes a user-output pipe:
        # `print()` and child-process stdout are drained separately and
        # forwarded through the `/log` side channel instead of corrupting
        # the control frame stream.
        frame_in_fd = os.dup(0)
        frame_out_fd = os.dup(1)
        stdout_read_fd, stdout_write_fd = os.pipe()
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(stdout_write_fd, 1)
        os.close(stdout_write_fd)
        os.close(devnull)
        _make_stdout_eager()

        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(frame_in_fd, "rb", buffering=0),
        )
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            os.fdopen(frame_out_fd, "wb", buffering=0),
        )
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
        self._reader, self._writer = reader, writer

        self._drainer = loop.create_task(self._drain_outbound())
        # Generic SIO channel: extensions inside the worker use
        # `agentix.sio.emit/on/request`; the bridge ferries frames over
        # the pipe to the server, which puts them on the real SIO.
        _sio._install(self._enqueue_frame)
        # Built-in /trace and /log namespaces — both are agentix-core
        # extensions registered on top of agentix.sio.
        install_worker_bridge()
        _log.install_worker_bridge()
        self._stdio_tasks.append(loop.create_task(self._drain_stdout(stdout_read_fd)))
        await self._send({"type": "ready"})

        while not self._shutdown.is_set():
            try:
                frame = await read_frame(reader)
            except asyncio.IncompleteReadError:
                break
            except (FrameTooLarge, ValueError):
                # Control stream desynced (oversized/garbled header). Nothing
                # downstream is trustworthy — log and shut down gracefully
                # rather than crash mid-loop or allocate a giant buffer.
                logger.exception("worker: control stream desynced; shutting down")
                break
            if frame is None:
                break
            await self._handle(frame)

        for task in list(self._calls.values()):
            task.cancel()
        if self._calls:
            await asyncio.gather(*self._calls.values(), return_exceptions=True)
        if self._stdio_tasks:
            _close_stdout_pipe()
            _, pending = await asyncio.wait(self._stdio_tasks, timeout=1.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await self._outbound_q.join()
        if self._drainer is not None:
            self._drainer.cancel()

    async def _drain_outbound(self) -> None:
        assert self._writer is not None
        try:
            while True:
                frame = await self._outbound_q.get()
                try:
                    await write_frame(self._writer, frame)
                except Exception:
                    logger.exception("outbound frame write failed")
                    self._recover_failed_frame(frame)
                finally:
                    self._outbound_q.task_done()
        except asyncio.CancelledError:
            pass

    def _recover_failed_frame(self, frame: dict[str, Any]) -> None:
        """A `result` frame that couldn't be written (typically an oversized
        pickled return value hitting `FrameTooLarge`) must not vanish — the
        host's future would hang forever. Replace it with a small, writable
        `error` frame for the same call so the caller fails fast. An `error`
        frame that itself fails to write has nothing left to fall back to."""
        if frame.get("type") != "result":
            return
        call_id = frame.get("call_id")
        if not call_id:
            return
        err = RemoteError(
            type="FrameTooLarge",
            message=(
                "remote call result could not be delivered: the pickled return value "
                f"exceeds the {MAX_MESSAGE_BYTES}-byte frame limit. Return a smaller "
                "value, or write large artifacts to a file/volume and return a reference."
            ),
        ).model_dump()
        try:
            self._outbound_q.put_nowait({"type": "error", "call_id": call_id, "error": err})
        except Exception:
            logger.exception("failed to enqueue FrameTooLarge error for call %r", call_id)

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._outbound_q.put(payload)

    async def _drain_stdout(self, fd: int) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(fd, "rb", buffering=0),
        )
        # Read fixed-size chunks and split into lines ourselves. `readline()`
        # raises on a line longer than the StreamReader limit (64 KiB); that
        # error was swallowed and KILLED this loop, so fd 1 stopped draining
        # and the next `print()` blocked on a full pipe — deadlocking the
        # in-flight call. Chunked reads can never overflow, so the pipe is
        # always drained regardless of line length.
        buf = bytearray()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                *lines, buf_rest = bytes(buf).split(b"\n")
                for line in lines:
                    _emit_stdio_line("stdout", line)
                buf = bytearray(buf_rest)
                # A newline-less spew (e.g. a binary blob) must not grow `buf`
                # without bound — flush it as a partial line.
                if len(buf) >= 65536:
                    _emit_stdio_line("stdout", bytes(buf))
                    buf.clear()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("stdout drain failed", exc_info=True)
        finally:
            if buf:
                _emit_stdio_line("stdout", bytes(buf))

    def _enqueue_frame(self, frame: dict[str, Any]) -> None:
        """Sync put for the agentix.sio bridge — must never block."""
        try:
            self._outbound_q.put_nowait(frame)
        except Exception:
            logger.debug("failed to enqueue sio frame", exc_info=True)

    async def _handle(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if not isinstance(kind, str):
            logger.warning("worker: missing frame type")
            return
        if kind == "call":
            await self._on_call(frame)
        elif kind == "cancel":
            self._cancel(frame.get("call_id", ""))
        elif kind == "shutdown":
            self._shutdown.set()
        elif kind == "sio_inbound":
            namespace = frame.get("namespace")
            event = frame.get("event")
            if isinstance(namespace, str) and isinstance(event, str):
                _sio._dispatch_inbound(namespace, event, frame.get("data"))
        else:
            logger.warning("worker: unknown frame type %r", kind)

    async def _on_call(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        try:
            request = RemoteRequest(
                callable=RemoteCallable(frame["callable"]),
                arguments=frame["arguments"],
                call_id=CallId(call_id) if call_id else None,
                context=frame.get("context"),
            )
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
            return
        task = asyncio.create_task(self._run(call_id, request))
        self._calls[call_id] = task
        task.add_done_callback(lambda _t: self._calls.pop(call_id, None))

    async def _run(self, call_id: str, request: RemoteRequest) -> None:
        try:
            fn = request.callable.resolve()
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
            return
        try:
            # The invoker establishes the per-call dispatch scope
            # (DISPATCH_CALL_ID + propagated context.attach) around fn.
            resp = await self._invoker.call(fn, request)
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
            return
        if resp.ok:
            await self._send({"type": "result", "call_id": call_id, "value": resp.value})
        else:
            err = (resp.error or RemoteError(type="Unknown", message="")).model_dump()
            await self._send({"type": "error", "call_id": call_id, "error": err})

    def _cancel(self, call_id: str) -> None:
        task = self._calls.get(call_id)
        if task is not None:
            task.cancel()
            asyncio.create_task(
                self._send(
                    {
                        "type": "error",
                        "call_id": call_id,
                        "error": RemoteError(
                            type="Cancelled",
                            message="remote call cancelled",
                            cancelled=True,
                        ).model_dump(),
                    }
                )
            )


async def _amain() -> None:
    worker = Worker()
    await worker.run()


def _make_stdout_eager() -> None:
    """Make regular `print()` visible without requiring `flush=True`."""
    with contextlib.suppress(Exception):
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _close_stdout_pipe() -> None:
    """Flush fd 1 and detach it from the capture pipe so the drainer reaches EOF."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
        finally:
            os.close(devnull)


def _emit_stdio_line(stream: str, raw: bytes) -> None:
    text = raw.decode("utf-8", "replace").rstrip("\r\n")
    emit_worker_record(
        {
            "name": f"agentix.sandbox.{stream}",
            "level": "INFO",
            "levelno": logging.INFO,
            "message": text,
            "created": time.time(),
            "pathname": "",
            "lineno": 0,
            "funcName": "",
            "module": "stdio",
            "exc_text": None,
            "stack_info": None,
            LOG_CONTEXT_ATTR: get_log_context(),
            "extras": {
                "agentix_stream": stream,
                "worker_id": os.environ.get("AGENTIX_WORKER_ID"),
            },
        }
    )


def main() -> None:
    _log.configure_logging(
        default_context="sandbox-{uname}-worker-{id}",
        stream=sys.stderr,
    )
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
