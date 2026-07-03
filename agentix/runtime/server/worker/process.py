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
from logging.handlers import RotatingFileHandler
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
from agentix.utils.log._config import DEFAULT_LOG_FORMAT, LOG_CONTEXT_ATTR, get_log_context
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
        self._outbound_write_failed = False

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        # The server↔worker frame pipe arrives on fd 0 (stdin) / fd 1
        # (stdout). User code inside a remote call routinely spawns
        # subprocesses (claude, git, ...) that INHERIT fd 0/1 — and a
        # child reading stdin (claude does) would steal frame bytes,
        # desyncing the protocol and hanging every later call.
        #
        # Move the framing onto private fds and point fd 0 at /dev/null, so
        # inherited stdin is harmless. fd 1 / fd 2 become user-output pipes:
        # `print()`, child-process output, and C-extension writes are
        # drained separately and forwarded through the `/log` side channel
        # instead of corrupting the control frame stream (fd 1) or vanishing
        # into the container log (fd 2).
        frame_in_fd = os.dup(0)
        frame_out_fd = os.dup(1)
        # Save the real stderr before fd 2 becomes a capture pipe — stdlib
        # logging's console output is repointed there (see
        # `_redirect_internal_logging`), and `main()` restores fd 2 from it
        # on the way out so interpreter-level output (crash tracebacks,
        # finalization errors) lands in the container log instead of a
        # reader-less pipe.
        global _real_stderr_fd
        real_stderr_fd = _real_stderr_fd = os.dup(2)
        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(stdout_write_fd, 1)
        os.dup2(stderr_write_fd, 2)
        os.close(stdout_write_fd)
        os.close(stderr_write_fd)
        os.close(devnull)
        _redirect_internal_logging(real_stderr_fd)
        _attach_sandbox_log_file()
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
        self._stdio_tasks.append(loop.create_task(self._drain_stream(stdout_read_fd, "stdout")))
        self._stdio_tasks.append(loop.create_task(self._drain_stream(stderr_read_fd, "stderr")))
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
            _close_stdio_pipes()
            _, pending = await asyncio.wait(self._stdio_tasks, timeout=1.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        # Bound the drain: a wedged outbound pipe (writer.drain() blocked on a
        # full OS pipe) would hang join() forever — task_done() never fires for
        # the stuck frame. Mirror the server-side bounded join.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._outbound_q.join(), timeout=2.0)
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
                    # Log the FIRST failure at exception level only: each
                    # logged traceback becomes a /log frame on this same
                    # queue, so per-failure logging self-sustains against a
                    # broken pipe and floods the durable file until shutdown.
                    if not self._outbound_write_failed:
                        self._outbound_write_failed = True
                        logger.exception("outbound frame write failed")
                    else:
                        logger.debug("outbound frame write failed", exc_info=True)
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

    async def _drain_stream(self, fd: int, stream: str) -> None:
        # The drainer shares the event loop: a remote `async def` that BLOCKS
        # the loop (subprocess.run, ...) while its child spews ≥64 KiB into
        # this fd wedges both sides — the same standing constraint the fd 1
        # capture has always had. Async callables must not block the loop;
        # sync callables run in threads and are fine.
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(fd, "rb", buffering=0),
        )
        # Read fixed-size chunks and split into lines ourselves. `readline()`
        # raises on a line longer than the StreamReader limit (64 KiB); that
        # error was swallowed and KILLED this loop, so the fd stopped draining
        # and the next write blocked on a full pipe — deadlocking the
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
                    _emit_stdio_line(stream, line)
                buf = bytearray(buf_rest)
                # A newline-less spew (e.g. a binary blob) must not grow `buf`
                # without bound — flush it as a partial line.
                if len(buf) >= 65536:
                    _emit_stdio_line(stream, bytes(buf))
                    buf.clear()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("%s drain failed", stream, exc_info=True)
        finally:
            if buf:
                _emit_stdio_line(stream, bytes(buf))

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
            # Enqueue synchronously (the outbound queue is unbounded) instead of
            # spawning an untracked `create_task`, which the loop only weakly
            # references and could GC before it runs — dropping the Cancelled
            # frame.
            self._enqueue_frame(
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


async def _amain() -> None:
    worker = Worker()
    await worker.run()


def _redirect_internal_logging(real_stderr_fd: int) -> None:
    """Point stdlib logging's console output at the REAL stderr.

    fd 2 is now a capture pipe whose lines replay on the host under
    `agentix.sandbox.stderr`. stdlib records already reach the host
    STRUCTURED (with ack/replay) via the `/log` bridge, so a console handler
    left on fd 2 would deliver every record twice — and a worker diagnostic
    emitted while the outbound pipe is broken would self-amplify (write
    fails → logged to stderr → captured → enqueued → fails → …). Raw fd-2
    capture is for the writers stdlib logging cannot see: child-process
    stderr, C extensions, direct `sys.stderr` prints."""
    with contextlib.suppress(Exception):
        real_stderr = os.fdopen(real_stderr_fd, "w", buffering=1)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
                handler.setStream(real_stderr)


class _SandboxLogFileHandler(RotatingFileHandler):
    """Best-effort durable log: any write failure detaches the handler for
    good. It must never report its own errors — stdlib's `handleError`
    prints to `sys.stderr`, which is the capture pipe, so a failing write
    per captured line would amplify into a loop."""

    def handleError(self, record: logging.LogRecord) -> None:
        _detach_sandbox_log_file(self)


_sandbox_log_handler: logging.Handler | None = None
_real_stderr_fd: int | None = None


def _restore_real_stderr() -> None:
    """Point fd 2 back at the real stderr saved in `run()`.

    Called on the way out of `main()`: after the loop closes, nothing drains
    the capture pipe, so interpreter output written to fd 2 — the crash
    traceback of an exception escaping `asyncio.run`, 'Exception ignored'
    finalization messages — would either block, break the pipe, or be
    discarded with the buffer. Restoring fd 2 sends it to the container log,
    as on master."""
    if _real_stderr_fd is not None:
        with contextlib.suppress(Exception):
            os.dup2(_real_stderr_fd, 2)


def _attach_sandbox_log_file() -> None:
    """Durable in-sandbox log at `$AGENTIX_LOG_DIR/sandbox-<worker>.log` (#139).

    The `/log` stream buffer is bounded — output that outlives a long
    disconnect (or the host itself) is otherwise unrecoverable. Attached to
    the root logger so stdlib records land in the file; captured
    stdout/stderr lines are written via a direct `emit()` from
    `_emit_stdio_line`. Size-bounded with one rotation — a post-mortem
    artifact, not an archive. Set `AGENTIX_LOG_DIR=` (empty) to disable."""
    global _sandbox_log_handler
    log_dir = os.environ.get("AGENTIX_LOG_DIR", "/tmp/agentix")
    if not log_dir:
        return
    # Per-worker filename: the default dir is machine-shared, and two
    # processes rotating ONE file race in `doRollover` (the loser's rename
    # fails → the handler detaches). Every spawn gets a fresh worker id, so
    # respawns never share a file either.
    worker_id = os.environ.get("AGENTIX_WORKER_ID", str(os.getpid()))
    try:
        os.makedirs(log_dir, exist_ok=True)
        handler = _SandboxLogFileHandler(
            os.path.join(log_dir, f"sandbox-{worker_id}.log"),
            maxBytes=64 * 1024 * 1024,
            backupCount=1,
            encoding="utf-8",
            delay=True,
        )
    except Exception:
        return
    handler.setFormatter(logging.Formatter(os.environ.get("AGENTIX_LOG_FORMAT", DEFAULT_LOG_FORMAT)))
    _sandbox_log_handler = handler
    logging.getLogger().addHandler(handler)


def _detach_sandbox_log_file(handler: logging.Handler) -> None:
    global _sandbox_log_handler
    _sandbox_log_handler = None
    with contextlib.suppress(Exception):
        logging.getLogger().removeHandler(handler)
    with contextlib.suppress(Exception):
        handler.close()
    # Announce the loss on `/log` so a truncated post-mortem file is
    # distinguishable from a sandbox that went quiet. The wire path is safe
    # here — it is the FILE we can no longer write, not the stream.
    with contextlib.suppress(Exception):
        _emit_stdio_line_wire(
            "stderr",
            "agentix: durable sandbox log detached after a write failure; later output is stream-only",
        )


def _make_stdout_eager() -> None:
    """Make regular `print()` visible without requiring `flush=True`."""
    with contextlib.suppress(Exception):
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _close_stdio_pipes() -> None:
    """Flush fd 1 / fd 2 and detach them from the capture pipes so the
    drainers reach EOF."""
    for stream, fd in ((sys.stdout, 1), (sys.stderr, 2)):
        with contextlib.suppress(Exception):
            stream.flush()
        with contextlib.suppress(Exception):
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, fd)
            finally:
                os.close(devnull)


def _emit_stdio_line(stream: str, raw: bytes) -> None:
    text = raw.decode("utf-8", "replace").rstrip("\r\n")
    handler = _sandbox_log_handler
    if handler is not None:
        # handler.handle(), not a logger call: routing through a logger would
        # multiply the line into the console/bridge handlers. handle() (unlike
        # a bare emit()) takes the handler lock — sync remote fns run in
        # threads, so their stdlib records reach this same handler locked, and
        # an unlocked emit racing a rollover kills the file for good.
        with contextlib.suppress(Exception):
            handler.handle(
                logging.makeLogRecord(
                    {
                        "name": f"agentix.sandbox.{stream}",
                        "msg": text,
                        "levelno": logging.INFO,
                        "levelname": "INFO",
                        LOG_CONTEXT_ATTR: get_log_context(),
                    }
                )
            )
    _emit_stdio_line_wire(stream, text)


def _emit_stdio_line_wire(stream: str, text: str) -> None:
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
    finally:
        _restore_real_stderr()


if __name__ == "__main__":
    main()
