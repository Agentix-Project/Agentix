"""`_BoundMethod` record + the kwargs coercion helper.

Internal to `agentix.dispatch` — the public surface is `Dispatcher`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar

from pydantic import TypeAdapter

from agentix.dispatch.shape import Shape
from agentix.idents import PackageName

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _BoundMethod(Generic[P, R]):
    name: str
    stub: Callable[P, R]
    impl: Callable[..., Any]
    signature: inspect.Signature
    shape: Shape
    param_adapters: dict[str, TypeAdapter[Any]]
    return_adapter: TypeAdapter[Any]
    item_adapter: TypeAdapter[Any] | None = None  # output item adapter (stream/bidi only)
    input_channel_param: str | None = None        # bidi: name of the Channel[T] param
    input_item_adapter: TypeAdapter[Any] | None = None  # bidi: input item adapter

    @property
    def is_stream(self) -> bool:
        """True for stream and bidi — anything that emits a sequence of items."""
        return self.shape in ("stream", "bidi")

    @property
    def is_bidi(self) -> bool:
        return self.shape == "bidi"


def coerce_args(
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


def source_for(impl: Callable[..., Any]) -> PackageName | None:
    """Derive a namespace package path from an impl function for trace events."""
    mod = getattr(impl, "__module__", None)
    if mod is None:
        return None
    return PackageName(mod)


# Internal — nothing here is part of the public dispatch API.
# `Dispatcher` (in dispatcher.py) is the only consumer.
