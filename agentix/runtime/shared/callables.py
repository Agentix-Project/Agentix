"""Callable identification.

`RemoteCallable` is the wire type for a remote callable: a `str`
subclass that stores `module::qualname`. Args, kwargs, and return
values are pickled separately; function identity stays import-path
based.

Top-level functions defined in a `python script.py` entrypoint are
encoded as an import reference to the script module, so users do not
have to special-case `__main__`. Lambdas, local closures, bound
methods, and callable instances are intentionally outside the remote
call boundary; put remote code behind an importable top-level function.

`display_name_for(fn)` is a host/worker-local helper for log lines and
error messages. It is not shipped on the wire — both ends recompute it
from their own fn reference.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def display_name_for(fn: Any) -> str:
    """Best-effort name for logs, error messages, and span attrs."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if isinstance(module, str) and module and isinstance(qualname, str) and qualname:
        return f"{module}::{qualname}"
    name = getattr(fn, "__name__", None)
    if isinstance(module, str) and module and isinstance(name, str) and name:
        return f"{module}::{name}"
    cls = type(fn)
    cls_module = getattr(cls, "__module__", "")
    cls_qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{cls_module}::{cls_qualname}" if cls_module else cls_qualname


def _main_module_name() -> str | None:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    spec_name = getattr(spec, "name", None)
    if isinstance(spec_name, str) and spec_name and spec_name != "__main__":
        return spec_name

    main_file = getattr(main, "__file__", None)
    if not isinstance(main_file, str) or not main_file:
        return None

    path = Path(main_file).resolve()
    for raw_entry in sys.path:
        root = Path(raw_entry or ".").resolve()
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.suffix != ".py":
            continue
        parts = rel.with_suffix("").parts
        if parts and all(part.isidentifier() for part in parts):
            return ".".join(parts)
    return None


def _callable_ref(fn: Callable[..., Any]) -> tuple[str, str] | None:
    if not (inspect.isfunction(fn) or inspect.isbuiltin(fn)):
        return None

    name = getattr(fn, "__name__", "")
    qualname = getattr(fn, "__qualname__", "")
    if not name or name == "<lambda>" or not qualname or "<locals>" in qualname:
        return None

    module = getattr(fn, "__module__", None)
    if module == "__main__":
        module = _main_module_name()
    if not isinstance(module, str) or not module:
        return None
    return module, qualname


class RemoteCallable(str):
    """Wire form of a remote callable: `module::qualname`.

    Subclasses `str` so it's directly serializable via msgpack / json /
    any text protocol with no special handling. Use the classmethod to
    construct one from a local fn, and `resolve()` to recover the fn
    on the receiving end.
    """

    __slots__ = ()

    @classmethod
    def _resolve(cls, fn: Callable[..., Any]) -> RemoteCallable:
        """Encode a Python callable as a `RemoteCallable` string."""
        if not callable(fn):
            raise TypeError(f"remote value must be callable (got {type(fn).__name__})")
        ref = _callable_ref(fn)
        if ref is None:
            raise TypeError(
                "remote callable must be an importable top-level function",
            )
        module, qualname = ref
        return cls(f"{module}::{qualname}")

    @classmethod
    def validate(cls, fn: Callable[..., Any]) -> RemoteCallable:
        """Check that `fn` is a remote-safe target and return its wire ref.

        Use at development time to fail fast: raises `TypeError` if `fn`
        is not an importable top-level function (lambdas, local closures,
        bound methods, and callable instances are rejected). This is the
        same encoding `client.remote(fn, ...)` performs internally before
        dispatch, exposed so callers can validate ahead of a sandbox run.
        """
        return cls._resolve(fn)

    def resolve(self) -> Callable[..., Any]:
        """Decode this string back into a Python callable."""
        try:
            module, qualname = self.split("::", 1)
        except ValueError as exc:
            raise ValueError(f"invalid remote callable reference {self!r}") from exc
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        fn = obj
        if not callable(fn):
            raise TypeError(f"resolved value is not callable (got {type(fn).__name__})")
        return fn


__all__ = ["RemoteCallable", "display_name_for"]
