"""Unified ambient context, propagated across `c.remote(fn, ...)`.

The context is *a set of contextvars*. Any var created with
`context.var(name, default=...)` is automatically carried across the
sandbox boundary: `remote()` snapshots every such var into one structure
(`capture()`), and the worker re-applies it under a single `with`
(`attach()`) before `fn` runs — then resets on exit. There is no
per-subsystem propagation logic; tracing simply declares its two
contextvars through this factory and they ride along like any other.

`var()` returns a genuine `contextvars.ContextVar`, so `get` / `set` /
`reset` keep their native, independently-nestable semantics — which is
exactly why each scope stays its own ContextVar rather than being merged
into one (a shared var's whole-state `reset` would clobber unrelated
nested scopes). The single structure exists only on the wire.

`baggage` is the built-in user-facing slice — a plain key/value map that
host and sandbox share:

    from agentix.utils import context

    context.set(tenant="acme", request_id="r-42")
    with context.scope(attempt=2):
        await client.remote(run, task=task)   # context rides along

    # inside the remote `run`, in the sandbox worker:
    context.get("tenant")     # -> "acme"
    context.get("attempt")    # -> 2

Captured values travel by pickle (like the call's arguments), so they
may be arbitrary Python objects — but every propagated var's value must
be picklable. Keep them small.
"""

from __future__ import annotations

import contextlib
import contextvars
import pickle
from collections.abc import Iterator, Mapping
from typing import Any

# Every contextvar created via `var()`, paired with its default so an
# unchanged var can be omitted from the carrier (empty context is free).
_propagated: list[tuple[contextvars.ContextVar[Any], Any]] = []


def var(name: str, *, default: Any) -> contextvars.ContextVar[Any]:
    """Create a `ContextVar` that is part of the propagated ambient
    context. Identical to `contextvars.ContextVar(name, default=...)`
    except the carrier captures and re-applies it across `remote()`.

    Use this instead of `contextvars.ContextVar` for any state that
    should follow a remote call (the active trace scope, request ids,
    tenancy, deadlines, ...). The return value is a real ContextVar.

    Idempotent on `name`: a repeat call returns the already-registered
    var rather than registering a duplicate (the carrier keys by name).
    """
    for cv, _default in _propagated:
        if cv.name == name:
            return cv
    cv: contextvars.ContextVar[Any] = contextvars.ContextVar(name, default=default)
    _propagated.append((cv, default))
    return cv


# ── baggage: the built-in user-facing key/value slice ─────────────

_baggage: contextvars.ContextVar[Mapping[str, Any]] = var(
    "agentix_context_baggage",
    default={},
)


def get(key: str, default: Any = None) -> Any:
    """Read one key from the ambient baggage."""
    return _baggage.get().get(key, default)


def snapshot() -> dict[str, Any]:
    """A read-only copy of the full ambient baggage."""
    return dict(_baggage.get())


def set(**values: Any) -> None:
    """Merge `values` into the ambient baggage (sticky for this
    task/thread). For a temporary overlay that auto-reverts, use
    `scope(...)`."""
    if not values:
        return
    merged = dict(_baggage.get())
    merged.update(values)
    _baggage.set(merged)


@contextlib.contextmanager
def scope(**values: Any) -> Iterator[dict[str, Any]]:
    """Overlay `values` onto the ambient baggage for the duration of
    the block, then restore the prior state."""
    merged = dict(_baggage.get())
    merged.update(values)
    tok = _baggage.set(merged)
    try:
        yield merged
    finally:
        _baggage.reset(tok)


def clear() -> None:
    """Drop all ambient baggage (sticky). Mostly for tests."""
    _baggage.set({})


# ── capture / re-apply across the remote() boundary ───────────────


def capture() -> dict[str, Any]:
    """Snapshot every propagated contextvar into one serializable
    structure. Vars still at their default are omitted, so an unused
    context captures to `{}`."""
    carrier: dict[str, Any] = {}
    for cv, default in _propagated:
        value = cv.get()
        if value is default or value == default:
            continue
        carrier[cv.name] = value
    return carrier


@contextlib.contextmanager
def attach(carrier: Mapping[str, Any] | None) -> Iterator[None]:
    """Re-apply a captured `carrier` for the duration of the block: set
    each propagated var to its carried value, then reset on exit. This
    is the `with` the worker runs `fn` inside. A falsy carrier is a
    no-op."""
    if not carrier:
        yield
        return
    resets: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    for cv, _default in _propagated:
        if cv.name in carrier:
            resets.append((cv, cv.set(carrier[cv.name])))
    try:
        yield
    finally:
        for cv, tok in reversed(resets):
            cv.reset(tok)


def encode() -> bytes | None:
    """Capture the ambient context and serialize it for the wire.
    Returns `None` when there is nothing to carry."""
    carrier = capture()
    if not carrier:
        return None
    return pickle.dumps(carrier)


def decode(blob: bytes | None) -> dict[str, Any]:
    """Inverse of `encode()` — `None`/empty decodes to `{}`."""
    if not blob:
        return {}
    value = pickle.loads(blob)
    return value if isinstance(value, dict) else {}


# User-facing surface only. `capture` / `attach` / `encode` / `decode`
# are the wire plumbing used by `remote()` + the worker invoker; they
# stay importable but are deliberately left out of `__all__`.
__all__ = [
    "clear",
    "get",
    "scope",
    "set",
    "snapshot",
    "var",
]
