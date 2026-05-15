"""In-process closure dispatch.

A `Dispatcher` binds typed stub signatures to their impl callables. Closures
ship a `_register.register()` function that returns a populated Dispatcher.
The runtime imports each mounted closure's package, collects Dispatchers
into a `Registry`, and serves `POST /{ns}/_remote` by calling
`registry.get(ns).dispatch(request)` directly — no subprocess, no UDS,
no reverse proxy.

Serialization is driven by the stub's `inspect.signature`: each parameter's
annotation becomes a pydantic `TypeAdapter`, same for the return type.
Stubs use plain `def`/`async def` with `...` (Ellipsis) bodies — no
decorators, no base classes.
"""

from __future__ import annotations

import asyncio
import collections.abc as cabc
import importlib
import inspect
import logging
import sys
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

import agentix.trace as trace
from agentix.models import ClosureManifest
from agentix.runtime.models import RemoteError, RemoteRequest, RemoteResponse

#: Type origins that indicate a streaming param or return annotation
#: (`AsyncIterator[T]` / `AsyncGenerator[T, ...]`). Shared between the
#: server-side bind detection here and the client-side route selection in
#: `agentix.runtime.client` — single source of truth for what "streams".
STREAM_ORIGINS: tuple[type, ...] = (cabc.AsyncIterator, cabc.AsyncGenerator)

logger = logging.getLogger("agentix.dispatch")

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _BoundMethod(Generic[P, R]):
    name: str
    stub: Callable[P, R]
    impl: Callable[..., Any]
    signature: inspect.Signature
    param_adapters: dict[str, TypeAdapter[Any]]
    return_adapter: TypeAdapter[Any]
    is_stream: bool = False                       # output is AsyncIterator[T]
    item_adapter: TypeAdapter[Any] | None = None  # output item adapter
    is_bidi: bool = False                         # input AND output are AsyncIterator[T]
    input_stream_param: str | None = None
    input_item_adapter: TypeAdapter[Any] | None = None


class Dispatcher:
    """A namespace's collection of bound (stub, impl) pairs.

    Closures construct one of these in their `_register.register()`:

        from agentix.dispatch import Dispatcher
        from . import run               # the stub (Ellipsis body)
        from ._impl import run as _run  # the real impl

        def register() -> Dispatcher:
            d = Dispatcher()
            d.bind(run, _run)
            return d
    """

    def __init__(self) -> None:
        self._methods: dict[str, _BoundMethod[Any, Any]] = {}

    def bind(
        self,
        stub: Callable[P, R],
        impl: Callable[..., R | Awaitable[R]],
    ) -> None:
        """Register `impl` as the implementation of `stub`.

        Both must share the same signature (the stub is just the typed
        contract; impl carries the body). The wire request's `method`
        field is `stub.__name__`.
        """
        sig = inspect.signature(stub)
        name = stub.__name__
        if name in self._methods:
            raise ValueError(f"method '{name}' already bound on this dispatcher")

        param_adapters: dict[str, TypeAdapter[Any]] = {}
        stream_params: list[tuple[str, type]] = []
        for pname, param in sig.parameters.items():
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
            if get_origin(ann) in STREAM_ORIGINS:
                # AsyncIterator[T] params: adapter validates items, not the iterator
                args = get_args(ann)
                item_type = args[0] if args else Any
                stream_params.append((pname, item_type))
                param_adapters[pname] = TypeAdapter(item_type)
            else:
                param_adapters[pname] = TypeAdapter(ann)

        return_ann = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else Any
        is_stream = get_origin(return_ann) in STREAM_ORIGINS
        item_adapter: TypeAdapter[Any] | None = None
        if is_stream:
            args = get_args(return_ann)
            item_type = args[0] if args else Any
            item_adapter = TypeAdapter(item_type)
            return_adapter = TypeAdapter(Any)  # unused on streaming path
        else:
            return_adapter = TypeAdapter(return_ann)

        is_bidi = is_stream and len(stream_params) > 0
        input_stream_param: str | None = None
        input_item_adapter: TypeAdapter[Any] | None = None
        if is_bidi:
            if len(stream_params) > 1:
                raise TypeError(
                    f"method '{name}' has multiple AsyncIterator parameters "
                    f"({[p for p, _ in stream_params]}); only one input stream "
                    f"is supported"
                )
            input_stream_param, input_item_type = stream_params[0]
            input_item_adapter = TypeAdapter(input_item_type)
        elif stream_params and not is_stream:
            raise TypeError(
                f"method '{name}' has AsyncIterator parameter(s) "
                f"{[p for p, _ in stream_params]} but a non-streaming return "
                f"type — client-only streaming is not supported; declare the "
                f"return type as AsyncIterator[T] to use the bidi path"
            )

        self._methods[name] = _BoundMethod(
            name=name,
            stub=stub,
            impl=impl,
            signature=sig,
            param_adapters=param_adapters,
            return_adapter=return_adapter,
            is_stream=is_stream,
            item_adapter=item_adapter,
            is_bidi=is_bidi,
            input_stream_param=input_stream_param,
            input_item_adapter=input_item_adapter,
        )

    def methods(self) -> list[str]:
        return list(self._methods)

    def is_streaming(self, method: str) -> bool:
        m = self._methods.get(method)
        return m is not None and m.is_stream

    def is_bidi(self, method: str) -> bool:
        m = self._methods.get(method)
        return m is not None and m.is_bidi

    def input_adapter_for(self, method: str) -> TypeAdapter[Any] | None:
        m = self._methods.get(method)
        return m.input_item_adapter if m else None

    async def dispatch(self, request: RemoteRequest) -> RemoteResponse:
        """Route a RemoteRequest to its bound impl, returning the wire response.

        Validates kwargs against the stub's signature, awaits async impls,
        serializes the return via the stub's return-type adapter, and
        traps exceptions into a RemoteError so the wire stays 200.
        """
        m = self._methods.get(request.method)
        if m is None:
            return RemoteResponse(
                ok=False,
                error=RemoteError(
                    type="MethodNotFound",
                    message=f"method '{request.method}' is not bound on this dispatcher; "
                    f"available: {sorted(self._methods)}",
                ),
            )
        try:
            args, kwargs = self._coerce(m, request.args, request.kwargs)
        except ValidationError as exc:
            return RemoteResponse(
                ok=False,
                error=RemoteError(type="ValidationError", message=str(exc)),
            )
        tokens = trace.set_call_context(request.call_id, _source_for(m.impl))
        try:
            try:
                result = m.impl(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                logger.exception("closure impl '%s' raised", m.name)
                return RemoteResponse(
                    ok=False,
                    error=RemoteError(
                        type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                    ),
                )
        finally:
            trace.reset_call_context(tokens)
        try:
            value = m.return_adapter.dump_python(result, mode="json")
        except Exception as exc:
            return RemoteResponse(
                ok=False,
                error=RemoteError(
                    type="SerializationError",
                    message=f"failed to serialize return value: {exc}",
                ),
            )
        return RemoteResponse(ok=True, value=value)

    async def dispatch_stream(self, request: RemoteRequest) -> AsyncIterator[dict[str, Any]]:
        """Run a server-streaming impl, yielding event dicts to the transport.

        Event shapes:
            {"item": <serialized>}      — per yielded value
            {"error": {...}}            — impl raised, validation failed, etc.
            {"end": true}               — normal completion sentinel

        The transport (Socket.IO server / HTTP NDJSON) encodes the dicts to
        the wire. The dispatcher only deals with semantic events.
        """
        m = self._methods.get(request.method)
        if m is None:
            yield {"error": RemoteError(
                type="MethodNotFound",
                message=f"method '{request.method}' is not bound on this dispatcher; "
                f"available: {sorted(self._methods)}",
            ).model_dump()}
            return
        if not m.is_stream or m.is_bidi:
            yield {"error": RemoteError(
                type="NotAStreamingMethod",
                message=f"method '{request.method}' is not a (non-bidi) streaming method",
            ).model_dump()}
            return
        try:
            args, kwargs = self._coerce(m, request.args, request.kwargs)
        except ValidationError as exc:
            yield {"error": RemoteError(type="ValidationError", message=str(exc)).model_dump()}
            return
        tokens = trace.set_call_context(request.call_id, _source_for(m.impl))
        try:
            try:
                result = m.impl(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                assert m.item_adapter is not None
                async for item in result:
                    try:
                        value = m.item_adapter.dump_python(item, mode="json")
                    except Exception as exc:
                        yield {"error": RemoteError(
                            type="SerializationError",
                            message=f"failed to serialize item: {exc}",
                        ).model_dump()}
                        return
                    yield {"item": value}
            except Exception as exc:
                logger.exception("closure stream impl '%s' raised mid-stream", m.name)
                yield {"error": RemoteError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                ).model_dump()}
                return
        finally:
            trace.reset_call_context(tokens)
        yield {"end": True}

    async def dispatch_bidi(
        self,
        request: RemoteRequest,
        input_iter: AsyncIterator[Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a bidi impl. `input_iter` yields items already coerced to the
        stub's input item type (transport pre-validates via `input_item_adapter`).

        Event shapes match `dispatch_stream` — same vocab of `item` / `end`
        / `error` — so the transport handles them uniformly.
        """
        m = self._methods.get(request.method)
        if m is None:
            yield {"error": RemoteError(
                type="MethodNotFound",
                message=f"method '{request.method}' is not bound on this dispatcher; "
                f"available: {sorted(self._methods)}",
            ).model_dump()}
            return
        if not m.is_bidi:
            yield {"error": RemoteError(
                type="NotABidiMethod",
                message=f"method '{request.method}' is not bidirectional",
            ).model_dump()}
            return
        assert m.input_stream_param is not None
        # Bind non-stream args/kwargs; inject input_iter as the stream param.
        non_stream_kwargs = dict(request.kwargs)
        non_stream_kwargs.pop(m.input_stream_param, None)
        try:
            bound = m.signature.bind_partial(*request.args, **non_stream_kwargs)
            bound.apply_defaults()
            coerced: dict[str, Any] = {}
            for pname, raw in bound.arguments.items():
                if pname == m.input_stream_param:
                    continue
                adapter = m.param_adapters.get(pname)
                coerced[pname] = adapter.validate_python(raw) if adapter is not None else raw
            coerced[m.input_stream_param] = input_iter
        except (TypeError, ValidationError) as exc:
            yield {"error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump()}
            return
        tokens = trace.set_call_context(request.call_id, _source_for(m.impl))
        try:
            try:
                result = m.impl(**coerced)
                if inspect.isawaitable(result):
                    result = await result
                assert m.item_adapter is not None
                async for item in result:
                    try:
                        value = m.item_adapter.dump_python(item, mode="json")
                    except Exception as exc:
                        yield {"error": RemoteError(
                            type="SerializationError",
                            message=f"failed to serialize item: {exc}",
                        ).model_dump()}
                        return
                    yield {"item": value}
            except Exception as exc:
                logger.exception("closure bidi impl '%s' raised mid-stream", m.name)
                yield {"error": RemoteError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                ).model_dump()}
                return
        finally:
            trace.reset_call_context(tokens)
        yield {"end": True}

    @staticmethod
    def _coerce(
        m: _BoundMethod[Any, Any],
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Bind args/kwargs against the stub signature, coercing each through
        its parameter's TypeAdapter (pydantic does dataclass/BaseModel/JSON
        round-tripping). Defaults are filled from the stub.
        """
        bound = m.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        coerced: dict[str, Any] = {}
        for pname, raw in bound.arguments.items():
            adapter = m.param_adapters.get(pname)
            coerced[pname] = adapter.validate_python(raw) if adapter is not None else raw
        # Re-split into args / kwargs in original order for the impl call.
        out_args: list[Any] = []
        out_kwargs: dict[str, Any] = {}
        for pname, param in m.signature.parameters.items():
            if pname not in coerced:
                continue
            v = coerced[pname]
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                out_args.append(v)
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                out_args.extend(v)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                out_kwargs.update(v)
            else:  # KEYWORD_ONLY
                out_kwargs[pname] = v
        return out_args, out_kwargs


def _source_for(impl: Callable[..., Any]) -> str | None:
    """Derive a closure package path from an impl function for trace events.

    Closure impls live at `agentix_closures.<name>._impl`; strip the `._impl`
    so the trace source reads as the public package the caller imported.
    """
    mod = getattr(impl, "__module__", None)
    if mod is None:
        return None
    return mod[:-6] if mod.endswith("._impl") else mod


@dataclass
class _Entry:
    """One registered closure. `dispatcher` is built lazily on first use."""

    manifest: ClosureManifest
    mount: Path
    dispatcher: Dispatcher | None = None
    error: Exception | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Registry:
    """Per-runtime collection of package-path → closure entry.

    Closures are registered at sandbox-startup mount discovery, but their
    Python packages are not imported and their Dispatchers are not built
    until the first call to `get_or_load(package)`. This keeps sandbox
    boot cheap and isolates per-closure import failures so they surface
    on call rather than blocking startup.

    The closure's Python import path (e.g. 'agentix_closures.claude_code')
    is the routing key — there are no caller-chosen namespaces.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, package: str, manifest: ClosureManifest, mount: Path) -> None:
        """Mark a closure as known but not yet loaded. Adds the closure's
        `entry/python` to sys.path so the stub module becomes importable —
        the Dispatcher is still deferred.
        """
        if package in self._entries:
            raise ValueError(f"package '{package}' already registered")
        py_root = mount / "entry" / "python"
        if py_root.is_dir():
            py_str = str(py_root)
            if py_str not in sys.path:
                sys.path.insert(0, py_str)
        self._entries[package] = _Entry(manifest=manifest, mount=mount)

    async def get_or_load(self, package: str) -> Dispatcher | None:
        """Return the dispatcher for `package`, importing + registering it
        on first call. Returns None for unknown packages. Re-raises the
        original exception on every call if the load has previously failed.

        Concurrent first calls to the same package serialise on a
        per-entry lock so the import + `_register.register()` runs once.
        """
        entry = self._entries.get(package)
        if entry is None:
            return None
        if entry.dispatcher is not None:
            return entry.dispatcher
        if entry.error is not None:
            raise entry.error
        async with entry.lock:
            if entry.dispatcher is not None:
                return entry.dispatcher
            if entry.error is not None:
                raise entry.error
            try:
                entry.dispatcher = _import_and_register(entry.manifest)
            except Exception as exc:
                logger.exception("lazy-load failed for closure '%s'", package)
                entry.error = exc
                raise
            return entry.dispatcher

    def packages(self) -> list[str]:
        """All known packages — registered, regardless of load state."""
        return list(self._entries)

    def loaded_packages(self) -> list[str]:
        """Packages whose dispatcher has been built (post first-use)."""
        return [pkg for pkg, e in self._entries.items() if e.dispatcher is not None]

    def manifest_for(self, package: str) -> ClosureManifest | None:
        e = self._entries.get(package)
        return e.manifest if e else None

    def mount_for(self, package: str) -> Path | None:
        e = self._entries.get(package)
        return e.mount if e else None

    def __contains__(self, package: str) -> bool:
        return package in self._entries


def _import_and_register(manifest: ClosureManifest) -> Dispatcher:
    """Import the closure's package and call `<pkg>._register.register()`.

    `entry/python` is already on sys.path (added at `Registry.register` time).
    The caller wraps any exception into the entry's `.error`.
    """
    importlib.import_module(manifest.package)  # validate the stub module exists
    register_mod = importlib.import_module(f"{manifest.package}._register")
    if not hasattr(register_mod, "register"):
        raise AttributeError(f"{manifest.package}._register has no register()")
    dispatcher = register_mod.register()
    if not isinstance(dispatcher, Dispatcher):
        raise TypeError(
            f"{manifest.package}._register.register() returned "
            f"{type(dispatcher).__name__}, expected Dispatcher"
        )
    return dispatcher
