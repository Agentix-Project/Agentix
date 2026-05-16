"""Namespace worker — `python -m agentix.runtime.worker --target module:Class`.

A worker is a single-namespace dispatch process that the runtime
multiplexer spawns lazily on first call. It loads ONE namespace class,
binds it via `Dispatcher`, and serves dispatch over stdin/stdout using
the RPC frame protocol in `agentix.runtime.rpc`.

The worker holds:

  - one `Dispatcher` (the loaded namespace's bound methods)
  - one asyncio task per in-flight call, keyed by `call_id`
  - one input queue per in-flight bidi call (for `bidi_in` chunks)

It forwards trace events via a sink that wraps each `trace.emit()` into
a frame on stdout. Logs go through Python's logging to stderr, which
the multiplexer captures separately.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
import traceback
from typing import Any

from agentix import trace
from agentix.dispatch import Dispatcher
from agentix.runtime.models import RemoteError, RemoteRequest
from agentix.runtime.rpc import read_frame, write_frame

logger = logging.getLogger("agentix.runtime.worker")


def _load_class(target: str) -> type:
    """`module:Class` → class object. Same shape as setuptools entry points."""
    if ":" not in target:
        raise ValueError(f"--target must be 'module:Class', got {target!r}")
    mod_name, cls_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _err(exc: BaseException) -> dict[str, Any]:
    return RemoteError(
        type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    ).model_dump()


class Worker:
    """One worker, one namespace. Owns stdio + a Dispatcher."""

    def __init__(self, dispatcher: Dispatcher, package: str) -> None:
        self._dispatcher = dispatcher
        self._package = package
        self._send_lock = asyncio.Lock()
        self._calls: dict[str, asyncio.Task] = {}
        self._bidi_queues: dict[str, asyncio.Queue] = {}
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._shutdown = asyncio.Event()

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

        # Install trace forwarder before we say "ready" — any boot-time
        # trace events get captured.
        trace.register_sink(self._trace_sink)

        await self._send({"type": "ready", "package": self._package})

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

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._writer is not None
        async with self._send_lock:
            await write_frame(self._writer, payload)

    def _trace_sink(self, kind: str, payload: dict, call_id, source) -> None:
        # Sync sink; schedule the actual frame write on the loop. Per the
        # `agentix.trace` contract, sink errors are caught upstream — we
        # only need to avoid raising.
        frame = {"type": "trace", "kind": kind, "payload": payload}
        if call_id is not None:
            frame["call_id"] = call_id
        if source is not None:
            frame["source"] = source
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._send(frame))

    async def _handle(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "call":
            await self._on_call(frame)
        elif kind == "bidi_in":
            await self._on_bidi_in(frame)
        elif kind == "bidi_end_in":
            await self._on_bidi_end_in(frame)
        elif kind == "cancel":
            self._cancel(frame.get("call_id", ""))
        elif kind == "shutdown":
            self._shutdown.set()
        else:
            logger.warning("worker: unknown frame type %r", kind)

    async def _on_call(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        kind = frame.get("kind", "unary")
        request = RemoteRequest(
            package=self._package,
            method=frame["method"],
            args=frame.get("args") or [],
            kwargs=frame.get("kwargs") or {},
            call_id=call_id,
        )
        if kind == "unary":
            task = asyncio.create_task(self._run_unary(call_id, request))
        elif kind == "stream":
            task = asyncio.create_task(self._run_stream(call_id, request))
        elif kind == "bidi":
            in_q: asyncio.Queue = asyncio.Queue(maxsize=64)
            self._bidi_queues[call_id] = in_q
            task = asyncio.create_task(self._run_bidi(call_id, request, in_q))
        else:
            await self._send({
                "type": "error", "call_id": call_id,
                "error": RemoteError(type="BadFrame", message=f"unknown call kind {kind!r}").model_dump(),
            })
            return
        self._calls[call_id] = task
        task.add_done_callback(lambda _t: self._calls.pop(call_id, None))
        task.add_done_callback(lambda _t: self._bidi_queues.pop(call_id, None))

    async def _run_unary(self, call_id: str, request: RemoteRequest) -> None:
        try:
            resp = await self._dispatcher.dispatch(request)
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
            return
        if resp.ok:
            await self._send({"type": "result", "call_id": call_id, "value": resp.value})
        else:
            await self._send({"type": "error", "call_id": call_id,
                              "error": (resp.error or RemoteError(type="Unknown", message="")).model_dump()})

    async def _run_stream(self, call_id: str, request: RemoteRequest) -> None:
        try:
            async for event in self._dispatcher.dispatch_stream(request):
                if "item" in event:
                    await self._send({"type": "stream_item", "call_id": call_id, "value": event["item"]})
                elif "error" in event:
                    await self._send({"type": "error", "call_id": call_id, "error": event["error"]})
                    return
                elif "end" in event:
                    await self._send({"type": "stream_end", "call_id": call_id})
                    return
            await self._send({"type": "stream_end", "call_id": call_id})
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})

    async def _run_bidi(self, call_id: str, request: RemoteRequest, in_q: asyncio.Queue) -> None:
        sentinel = object()
        adapter = self._dispatcher.input_adapter_for(request.method)

        async def _input_iter():
            while True:
                item = await in_q.get()
                if item is sentinel:
                    return
                yield item

        # Pre-validate items as they arrive in the input queue by wrapping
        # the queue's iterator with the dispatcher's input adapter. The
        # dispatcher itself feeds raw items to the impl; we coerce here.
        async def _coerced_iter():
            async for raw in _input_iter():
                if adapter is not None:
                    try:
                        raw = adapter.validate_python(raw)
                    except Exception as exc:
                        await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
                        return
                yield raw

        try:
            async for event in self._dispatcher.dispatch_bidi(request, _coerced_iter()):
                if "item" in event:
                    await self._send({"type": "stream_item", "call_id": call_id, "value": event["item"]})
                elif "error" in event:
                    await self._send({"type": "error", "call_id": call_id, "error": event["error"]})
                    return
                elif "end" in event:
                    await self._send({"type": "stream_end", "call_id": call_id})
                    return
            await self._send({"type": "stream_end", "call_id": call_id})
        except Exception as exc:
            await self._send({"type": "error", "call_id": call_id, "error": _err(exc)})
        finally:
            # Make sure the input iterator unblocks if the impl exited early.
            try:
                in_q.put_nowait(sentinel)
            except asyncio.QueueFull:
                pass

    async def _on_bidi_in(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        q = self._bidi_queues.get(call_id)
        if q is None:
            return
        await q.put(frame.get("item"))

    async def _on_bidi_end_in(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("call_id", "")
        q = self._bidi_queues.get(call_id)
        if q is None:
            return
        # Sentinel: an explicit "_end_" marker. Use a tuple to disambiguate
        # from legitimate user-supplied None / sentinel-like values. The
        # input iterator in _run_bidi compares with `is sentinel`, but we
        # don't have that object reference here — instead push a special
        # frame and let _run_bidi recognise via `_END_SENTINEL`.
        await q.put(_END_SENTINEL)

    def _cancel(self, call_id: str) -> None:
        task = self._calls.get(call_id)
        if task is not None:
            task.cancel()


# Singleton sentinel for "end of bidi input" pushed through the input queue.
# `_run_bidi` uses a per-call object created at task start; cross-method
# coordination here needs a stable reference. Using an object() at module
# scope works because Worker compares with `is`.
_END_SENTINEL: Any = object()


def _make_dispatcher(target: str) -> tuple[Dispatcher, str]:
    cls = _load_class(target)
    dispatcher = Dispatcher().bind_namespace(cls)
    return dispatcher, cls.__module__


async def _amain(target: str) -> None:
    try:
        dispatcher, package = _make_dispatcher(target)
    except Exception as exc:
        # Worker hasn't initialized stdio framing yet; bootstrap a minimal
        # writer so the multiplexer learns why we're exiting.
        sys.stdout.buffer.write(b"")  # ensure stdout is flushed binary
        from agentix.runtime.rpc import pack_frame
        sys.stdout.buffer.write(pack_frame({"type": "boot_error", "error": _err(exc)}))
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
        help="namespace class to load, in `module:Class` form",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_amain(args.target))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
