"""Wire patterns — pluggable call-shape protocols.

A `WirePattern` owns one call shape (unary / server-streaming / bidi /
… third-party) end to end: how to detect it from a stub's signature,
how the server runs it, and how the client invokes it. Built-in
patterns ship in this module; third parties can `register_pattern(cls)`
to add their own.

```python
class PubSubPattern(WirePattern):
    name = "pubsub"

    @classmethod
    def matches(cls, sig: inspect.Signature) -> bool:
        # e.g. detect a Topic[T] marker in the return annotation
        ...

    async def server_invoke(self, bound, request, transport):
        ...

    async def client_invoke(self, client, package, method, sig, args, kwargs):
        ...

register_pattern(PubSubPattern)
```

Pattern selection is a priority list. `register_pattern(cls)` prepends,
so user patterns outrank the built-ins. The built-ins ship in
specific-to-general order — bidi before stream before unary — so the
fallback is `UnaryPattern`.

The Dispatcher caches the matched pattern instance on each bound
method; pattern lookup happens at `bind` time, not per call.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar, get_args, get_origin

from agentix.runtime.models import STREAM_ORIGINS

if TYPE_CHECKING:
    from agentix.runtime.client.client import RuntimeClient


class WirePattern(ABC):
    """Strategy object for one call shape.

    Concrete subclasses must implement `matches`, `server_invoke`, and
    `client_invoke`. They may carry per-method state on the instance
    (e.g. pre-built `TypeAdapter`s) — one pattern instance per bound
    method.

    The `name` is the wire-protocol tag: the event-name prefix on the
    Socket.IO side (e.g. `stream`, `bidi`), or a future protocol tag.
    Two patterns must not share a name.
    """

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def matches(cls, sig: inspect.Signature) -> bool:
        """Return True if a stub with this signature uses this pattern."""

    @abstractmethod
    def bind(self, sig: inspect.Signature) -> None:
        """Pre-compute per-method state (type adapters, stream params, …).

        Called once at `Dispatcher.bind` time. Subsequent calls reuse
        the cached state.
        """

    @abstractmethod
    def client_invoke(
        self,
        client: RuntimeClient,
        fn: Callable[..., Any],
        sig: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Awaitable[Any] | AsyncIterator[Any]:
        """Invoke `fn` over the wire from the client side.

        Returns either:
          * an `Awaitable[R]` for unary patterns — caller `await`s it
          * an `AsyncIterator[T]` for streaming-style patterns — caller
            iterates with `async for`

        The pattern owns the wire framing: it picks the transport
        (HTTP `/_remote`, Socket.IO `stream`, custom event names) and
        handles correlation, type coercion, and error mapping.
        """


class UnaryPattern(WirePattern):
    """Request/response. The default — used when no other pattern matches.

    Wire: `POST /_remote` with JSON body, JSON response.
    """

    name = "unary"

    @classmethod
    def matches(cls, sig: inspect.Signature) -> bool:
        return True  # fallback — always last in the registry

    def bind(self, sig: inspect.Signature) -> None:
        return  # adapters are computed in Dispatcher._BoundMethod for now

    def client_invoke(
        self,
        client: RuntimeClient,
        fn: Callable[..., Any],
        sig: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Awaitable[Any]:
        return client._remote_unary(fn, sig.return_annotation, *args, **kwargs)


class StreamPattern(WirePattern):
    """Server-streaming. Stub returns `AsyncIterator[T]`, no streaming params.

    Wire: Socket.IO `stream` event → `stream:item` × N + `stream:end`
    (or `stream:error`).
    """

    name = "stream"

    @classmethod
    def matches(cls, sig: inspect.Signature) -> bool:
        if get_origin(sig.return_annotation) not in STREAM_ORIGINS:
            return False
        # no AsyncIterator parameters — that's bidi
        for p in sig.parameters.values():
            if get_origin(p.annotation) in STREAM_ORIGINS:
                return False
        return True

    def bind(self, sig: inspect.Signature) -> None:
        return

    def client_invoke(
        self,
        client: RuntimeClient,
        fn: Callable[..., Any],
        sig: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        return client._remote_stream(fn, sig, *args, **kwargs)


class BidiPattern(WirePattern):
    """Bidirectional streaming. Stub returns `AsyncIterator[U]` and takes
    exactly one `AsyncIterator[T]` parameter.

    Wire: Socket.IO `bidi:start` → `bidi:in` × N (client→server)
    interleaved with `bidi:out` × M (server→client) → `bidi:end`
    (or `bidi:error`).
    """

    name = "bidi"

    @classmethod
    def matches(cls, sig: inspect.Signature) -> bool:
        if get_origin(sig.return_annotation) not in STREAM_ORIGINS:
            return False
        stream_params = [
            p for p in sig.parameters.values()
            if get_origin(p.annotation) in STREAM_ORIGINS
        ]
        return len(stream_params) == 1

    def bind(self, sig: inspect.Signature) -> None:
        return

    def client_invoke(
        self,
        client: RuntimeClient,
        fn: Callable[..., Any],
        sig: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        return client._remote_bidi(fn, sig, *args, **kwargs)


# ── Registry ────────────────────────────────────────────────────────
#
# Pattern lookup walks two sources:
#
#   1. The `agentix.wire_pattern` entry-point group — production third
#      parties install their patterns via `pip install`.
#   2. An in-process list seeded with the three built-ins and extended
#      by `register_pattern(...)` for tests / dynamic registration.
#
# Both are merged at lookup time; the in-process list takes precedence
# (so tests can override an entry-point pattern), then user-installed
# entry-points come ahead of built-ins. Order matters because more
# specific patterns shadow general ones (Bidi before Stream before
# Unary).

from agentix._plugin import Registry  # noqa: E402

_patterns: list[type[WirePattern]] = [BidiPattern, StreamPattern, UnaryPattern]
_pattern_plugins: Registry[type[WirePattern]] = Registry("agentix.wire_pattern")


def register_pattern(pattern_cls: type[WirePattern]) -> None:
    """Register a wire pattern imperatively.

    Patterns are checked in registration order, most recently registered
    first; built-ins are at the tail. Production third parties should
    declare a `[project.entry-points."agentix.wire_pattern"]` instead
    — pip install + nothing else.

    Pattern names must be unique. Re-registering a name overwrites
    the existing entry — useful for tests, dangerous in production.
    """
    name = pattern_cls.name
    for i, p in enumerate(_patterns):
        if p.name == name:
            _patterns[i] = pattern_cls
            return
    _patterns.insert(0, pattern_cls)


def wire_patterns() -> Registry[type[WirePattern]]:
    """The entry-point registry — for `agentix plugins` and tests."""
    return _pattern_plugins


def _ordered_patterns() -> list[type[WirePattern]]:
    """Resolved pattern list: in-process registrations first (override
    everything), then entry-point patterns (loaded lazily, in entry-
    point-name order), then the built-in fallbacks.
    """
    in_process_names = {p.name for p in _patterns}
    out: list[type[WirePattern]] = list(_patterns[:-3])  # skip the built-in trio at the tail
    # Entry-point patterns whose names aren't already shadowed.
    for name, cls in _pattern_plugins.all().items():
        if name in in_process_names:
            continue
        out.append(cls)
    # The three built-ins, at the tail, most-general last.
    out.extend(_patterns[-3:])
    return out


def select_pattern(sig: inspect.Signature) -> type[WirePattern]:
    """Return the first pattern class whose `matches(sig)` is True.

    `UnaryPattern.matches` returns True unconditionally, so this never
    raises — every signature has a pattern.
    """
    for p in _ordered_patterns():
        if p.matches(sig):
            return p
    raise TypeError(f"no WirePattern matches signature {sig!r}")


def registered_patterns() -> list[type[WirePattern]]:
    """Snapshot of the current pattern list, highest priority first."""
    return _ordered_patterns()


def _reset_patterns() -> None:
    """Test-only — restore built-in defaults and clear plugin registry."""
    global _patterns
    _patterns = [BidiPattern, StreamPattern, UnaryPattern]
    _pattern_plugins.reset()


# ── Helpers shared across patterns ──────────────────────────────────


def stream_item_type(ann: object) -> object:
    """Return `T` from `AsyncIterator[T]` / `AsyncGenerator[T, ...]`.

    Raises TypeError if `ann` isn't a stream origin. Returns `Any` when
    the generic is unparameterised.
    """
    if get_origin(ann) not in STREAM_ORIGINS:
        raise TypeError(f"not a stream type: {ann!r}")
    args = get_args(ann)
    return args[0] if args else Any


__all__ = [
    "AsyncIterator",
    "BidiPattern",
    "StreamPattern",
    "UnaryPattern",
    "WirePattern",
    "register_pattern",
    "registered_patterns",
    "select_pattern",
    "stream_item_type",
]
