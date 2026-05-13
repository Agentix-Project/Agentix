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

import inspect
import logging
import traceback
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from pydantic import TypeAdapter, ValidationError

from agentix.models import RemoteError, RemoteRequest, RemoteResponse

logger = logging.getLogger("agentix.dispatch")

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _BoundMethod(Generic[P, R]):
    name: str
    stub: Callable[P, R]
    impl: Callable[..., R | Awaitable[R]]
    signature: inspect.Signature
    param_adapters: dict[str, TypeAdapter[Any]]
    return_adapter: TypeAdapter[Any]


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
        for pname, param in sig.parameters.items():
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
            param_adapters[pname] = TypeAdapter(ann)
        return_ann = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else Any
        self._methods[name] = _BoundMethod(
            name=name,
            stub=stub,
            impl=impl,
            signature=sig,
            param_adapters=param_adapters,
            return_adapter=TypeAdapter(return_ann),
        )

    def methods(self) -> list[str]:
        return list(self._methods)

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


class Registry:
    """Per-runtime collection of package-path → Dispatcher mappings.

    The closure's Python import path (e.g. 'agentix_closures.claude_code')
    is the routing key — there are no caller-chosen namespaces.
    """

    def __init__(self) -> None:
        self._d: dict[str, Dispatcher] = {}

    def add(self, package: str, dispatcher: Dispatcher) -> None:
        if package in self._d:
            raise ValueError(f"package '{package}' already registered")
        self._d[package] = dispatcher

    def get(self, package: str) -> Dispatcher | None:
        return self._d.get(package)

    def packages(self) -> list[str]:
        return list(self._d)

    def __contains__(self, package: str) -> bool:
        return package in self._d
