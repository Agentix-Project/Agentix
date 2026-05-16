"""Namespace worker — `python -m agentix.runtime.worker --target <pkg>`.

A worker is a single-namespace dispatch process that the runtime
multiplexer spawns lazily on first call. It loads ONE namespace target
(typically a Python package, but a `module:attr` form is also accepted
for class-style or partial targets), binds it via `Dispatcher`, and
serves dispatch over stdin/stdout using the RPC frame protocol in
`agentix.runtime.rpc`.

The worker holds:

  - one `Dispatcher` (the loaded namespace's bound methods)
  - one asyncio task per in-flight call, keyed by `call_id`
  - one input queue per in-flight bidi call (for `bidi_in` chunks)

It forwards trace events via a subscriber that wraps each
`trace.emit()` into a frame on stdout. Logs go through Python's
logging to stderr, which the multiplexer captures separately.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import sys
import traceback
from typing import Any

from agentix import trace
from agentix.dispatch import Dispatcher
from agentix.idents import CallId, PackageName
from agentix.runtime import _pump
from agentix.runtime import frames as F
from agentix.runtime.models import RemoteError, RemoteRequest
from agentix.runtime.rpc import read_frame, write_frame

logger = logging.getLogger("agentix.runtime.worker")


def _load_target(target: str) -> Any:
    """Resolve `target` to a Python object.

    Two forms (setuptools entry-point grammar):
      * `module.path`        → the module itself
      * `module.path:attr`   → `getattr(module, attr)` — a class or any obj

    The dispatcher duck-types whatever comes back; the bare-module form
    is the recommended namespace shape.
    """
    if ":" in target:
        mod_name, attr_name = target.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    return importlib.import_module(target)


def _err(exc: BaseException) -> dict[str, Any]:
    return RemoteError(
        type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    ).model_dump()


class Worker:
    """One worker, one namespace. Owns stdio + a Dispatcher.

    All outbound frames (result, stream item/end, trace, error, ready)
    funnel through `_outbound_q` and are serialized by a single drainer
    task. This gives us:
      - **FIFO ordering** across trace + result frames (observers see
        traces in the order the impl emitted them)
      - **No GC of fire-and-forget tasks** for sync `trace.emit` paths
      - **No send lock** — one writer, no contention
    """

    def __init__(self, dispatcher: Dispatcher, package: str) -> None:
        self._dispatcher = dispatcher
        self._package = package
        self._calls: dict[str, asyncio.Task] = {}
        # Bidi inbound path is two-tier so the main read loop never blocks
        # on a slow impl: `_bidi_intakes[cid]` is unbounded (drained from
        # the main loop with `put_nowait`); a per-call `_bidi_pumps[cid]`
        # task moves items from intake into the bounded `_bidi_queues[cid]`
        # that the impl reads from. The bounded user queue is what gives
        # backpressure — when full, the pump's `await put` blocks, the
        # intake grows briefly, OS pipe fills, multiplexer's stdin write
        # blocks, Socket.IO emit backs up, ultimately the caller's
        # `Channel.send()` awaits. CANCEL frames bypass all this via
        # the main read loop.
        self._bidi_queues: dict[str, asyncio.Queue] = {}
        self._bidi_intakes: dict[str, asyncio.Queue] = {}
        self._bidi_pumps: dict[str, asyncio.Task] = {}
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._shutdown = asyncio.Event()
        # Single FIFO that every outbound frame passes through. Unbounded
        # by default — back-pressure on the writer is sufficient; we
        # don't want sync trace.emit() to block on a full queue.
        self._outbound_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._drainer: asyncio.Task | None = None

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer,
        )
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer,
        )
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
        self._reader, self._writer = reader, writer

        # Spin up the single outbound drainer + subscribe trace forwarder
        # before saying "ready" — any boot-time trace events get captured.
        self._drainer = loop.create_task(self._drain_outbound())
        trace.subscribe(self._trace_handler)

        await self._send({"type": F.READY, "package": self._package})

        while not self._shutdown.is_set():
            try:
                frame = await read_frame(reader)
            except asyncio.IncompleteReadError:
                break
            if frame is None:
                break
            await self._handle(frame)

        # Drain in-flight calls on shutdown.
        for task in list(self._calls.values()):
            task.cancel()
        if self._calls:
            await asyncio.gather(*self._calls.values(), return_exceptions=True)
        # Flush any remaining outbound frames, then stop the drainer.
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
                finally:
                    self._outbound_q.task_done()
        except asyncio.CancelledError:
            pass

    async def _send(self, payload: dict[str, Any]) -> None:
        """Enqueue a frame for the drainer. Effectively non-blocking
        on an unbounded queue."""
        await self._outbound_q.put(payload)

    def _trace_handler(self, kind: str, payload: dict, call_id, source) -> None:
        # Sync handler — enqueue via put_nowait (queue is unbounded,
        # so this never raises in practice). The drainer preserves
        # FIFO order between traces and the result frame that follows.
        frame: dict[str, Any] = {"type": F.TRACE, "kind": kind, "payload": payload}
        if call_id is not None:
            frame["call_id"] = call_id
        if source is not None:
            frame["source"] = source
        try:
            self._outbound_q.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # impossible on an unbounded queue, but defensive

    _RUNTIME_FRAME_HANDLERS: dict[str, str] = {
        F.CALL: "_on_call",
        F.BIDI_IN: "_on_bidi_in",
        F.BIDI_END_IN: "_on_bidi_end_in",
    }

    async def _handle(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if not isinstance(kind, str):
            logger.warning("worker: missing frame type")
            return
        handler_name = self._RUNTIME_FRAME_HANDLERS.get(kind)
        if handler_name is not None:
            await getattr(self, handler_name)(frame)
        elif kind == F.CANCEL:
            self._cancel(frame.get("call_id", ""))
        elif kind == F.SHUTDOWN:
            self._shutdown.set()
        else:
            logger.warning("worker: unknown frame type %r", kind)

    async def _on_call(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        kind = frame.get("kind", F.KIND_UNARY)
        request = RemoteRequest(
            package=PackageName(self._package),
            method=frame["method"],
            args=frame.get("args") or [],
            kwargs=frame.get("kwargs") or {},
            call_id=CallId(call_id) if call_id else None,
        )
        if kind == F.KIND_UNARY:
            task = asyncio.create_task(self._run_unary(call_id, request))
        elif kind == F.KIND_STREAM:
            task = asyncio.create_task(self._run_stream(call_id, request))
        elif kind == F.KIND_BIDI:
            user_q: asyncio.Queue = asyncio.Queue(maxsize=_pump.DEFAULT_BIDI_BUFFER)
            intake_q: asyncio.Queue = asyncio.Queue()
            self._bidi_queues[call_id] = user_q
            self._bidi_intakes[call_id] = intake_q
            self._bidi_pumps[call_id] = asyncio.create_task(
                _pump.drain(intake_q, user_q, _END_SENTINEL)
            )
            task = asyncio.create_task(self._run_bidi(call_id, request, user_q))
        else:
            await self._send({
                "type": F.ERROR, "call_id": call_id,
                "error": RemoteError(type="BadFrame", message=f"unknown call kind {kind!r}").model_dump(),
            })
            return
        self._calls[call_id] = task
        task.add_done_callback(lambda _t: self._calls.pop(call_id, None))
        task.add_done_callback(lambda _t, cid=call_id: self._cleanup_bidi(cid))

    async def _forward_stream_event(self, call_id: str, event: dict[str, Any]) -> bool:
        """Map a Dispatcher stream/bidi event to its wire frame; return True
        if the event was terminal (caller should stop iterating)."""
        kind = event.get("type")
        if kind == "item":
            await self._send({"type": F.STREAM_ITEM, "call_id": call_id, "value": event["value"]})
            return False
        if kind == "error":
            await self._send({"type": F.ERROR, "call_id": call_id, "error": event["error"]})
            return True
        if kind == "end":
            await self._send({"type": F.STREAM_END, "call_id": call_id})
            return True
        return False

    async def _run_unary(self, call_id: str, request: RemoteRequest) -> None:
        try:
            resp = await self._dispatcher.dispatch(request)
        except Exception as exc:
            await self._send({"type": F.ERROR, "call_id": call_id, "error": _err(exc)})
            return
        if resp.ok:
            await self._send({"type": F.RESULT, "call_id": call_id, "value": resp.value})
        else:
            await self._send({"type": F.ERROR, "call_id": call_id,
                              "error": (resp.error or RemoteError(type="Unknown", message="")).model_dump()})

    async def _run_stream(self, call_id: str, request: RemoteRequest) -> None:
        try:
            async for event in self._dispatcher.dispatch_stream(request):
                if await self._forward_stream_event(call_id, event):
                    return
            await self._send({"type": F.STREAM_END, "call_id": call_id})
        except Exception as exc:
            await self._send({"type": F.ERROR, "call_id": call_id, "error": _err(exc)})

    async def _run_bidi(self, call_id: str, request: RemoteRequest, in_q: asyncio.Queue) -> None:
        adapter = self._dispatcher.input_adapter_for(request.method)

        async def _input_iter():
            while True:
                item = await in_q.get()
                if item is _END_SENTINEL:
                    return
                if adapter is not None:
                    try:
                        item = adapter.validate_python(item)
                    except Exception as exc:
                        await self._send({"type": F.ERROR, "call_id": call_id, "error": _err(exc)})
                        return
                yield item

        try:
            async for event in self._dispatcher.dispatch_bidi(request, _input_iter()):
                if await self._forward_stream_event(call_id, event):
                    return
            await self._send({"type": F.STREAM_END, "call_id": call_id})
        except Exception as exc:
            await self._send({"type": F.ERROR, "call_id": call_id, "error": _err(exc)})
        finally:
            # Unblock the input iterator if the impl exited early.
            try:
                in_q.put_nowait(_END_SENTINEL)
            except asyncio.QueueFull:
                pass

    async def _on_bidi_in(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        intake = self._bidi_intakes.get(call_id)
        if intake is None:
            # Late frame after terminal — invariant #2 (terminal-then-quiet).
            return
        # Intake is unbounded; `put_nowait` never raises. Backpressure
        # happens at `_bidi_pump`'s `await user_q.put(...)` which blocks
        # on the bounded user queue. The main read loop is never stalled.
        intake.put_nowait(frame.get("item"))

    async def _on_bidi_end_in(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        intake = self._bidi_intakes.get(call_id)
        if intake is None:
            return
        intake.put_nowait(_END_SENTINEL)

    def _cleanup_bidi(self, call_id: str) -> None:
        self._bidi_queues.pop(call_id, None)
        self._bidi_intakes.pop(call_id, None)
        _pump.cancel_if_running(self._bidi_pumps.pop(call_id, None))

    def _cancel(self, call_id: str) -> None:
        task = self._calls.get(call_id)
        if task is not None:
            task.cancel()


# Module-level singleton used to signal "end of bidi input" through the
# input queue. Compared with `is`; the same object reference must be
# visible from `_on_bidi_end_in` (pusher), `_pump.drain` (forwarder), and
# `_run_bidi` (consumer).
_END_SENTINEL: Any = object()


def _make_dispatcher(target: str) -> tuple[Dispatcher, str]:
    obj = _load_target(target)
    dispatcher = Dispatcher().bind_namespace(obj)
    # Routing key: the module the target lives in. For a bare-module
    # target the module IS the object; for `module:attr` we take the
    # module path so caller-side `fn.__module__` matches.
    package = obj.__name__ if inspect.ismodule(obj) else obj.__module__
    return dispatcher, package


async def _amain(target: str) -> None:
    try:
        dispatcher, package = _make_dispatcher(target)
    except Exception as exc:
        # Worker hasn't initialized stdio framing yet; bootstrap a minimal
        # writer so the multiplexer learns why we're exiting.
        from agentix.runtime.rpc import pack_frame
        sys.stdout.buffer.write(pack_frame({"type": F.BOOT_ERROR, "error": _err(exc)}))
        sys.stdout.buffer.flush()
        sys.exit(1)
    worker = Worker(dispatcher, package)
    await worker.run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(prog="agentix.runtime.worker")
    parser.add_argument(
        "--target", required=True,
        help="namespace to load — `module.path` (recommended, package-as-namespace) "
             "or `module.path:attr` (class-style or partial target)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_amain(args.target))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
