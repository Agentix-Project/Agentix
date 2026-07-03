"""Restricted unpickling for the sandbox→host return boundary (#116).

Return values travel host-ward as `pickle.dumps(result)` and the host decodes
them. `pickle.loads` reconstructs arbitrary objects by invoking whatever
callables a stream names, so decoding a value influenced by sandbox-side code
is a trust-boundary concern: the sandbox may run less-trusted workloads (a
cloned repository, a generated patch, a benchmark task) whose returned object
can direct reconstruction on the host.

`restricted_loads` keeps pickle as the wire format but decodes through a strict
**allowlist**: `find_class` permits only an explicit, individually reviewed set
of data types and inert reconstruction helpers, and refuses everything else. A
refused global's module is never imported (the refusal is decided from the name),
so a stream cannot force import-time code to run on the host. Because the
permitted set contains only value-shaped types (their construction has no
external side effects) and helpers that merely rebuild those values, a decoded
stream cannot reach a callable that acts on the host.

Why an allowlist and not a denylist of "dangerous" names:

  * A stream can reference a global's C-accelerator module (`_socket` vs
    `socket`, `_operator` vs `operator`), so name-based blocking is porous.
  * A callable produced by one reconstruction step sits on pickle's stack and
    is invoked by the next step *without* going through `find_class`, so
    admitting attribute-access helpers (`getattr`, `operator.attrgetter`, …)
    lets a stream walk from any admitted object to an arbitrary callable.
  * Many ordinary constructors have side effects (opening files, binding
    sockets, importing modules), so "admit any class" is not safe either.

Only a closed allowlist of value types closes all three.

Scope: this guards the host-side decode of sandbox return values only
(`RuntimeClient._unpickle_value`). The sandbox-side decode of host-supplied
arguments and context stays plain pickle — that is the trusted host→sandbox
direction.

First-party types (`agentix.*`) are trusted by default: the framework and its
plugins build the bundle and run the sandbox, so their own return types
(`TunnelHandle`, `BashResult`, agent results, …) are part of the trusted
computing base. The boundary defends against a *workload's* return value, and a
workload's own types are not `agentix.*` — they stay opt-in.

Extending / relaxing:

  * `allow_module(prefix)` / `allow_callable(module, name)` add return types the
    default set does not cover (e.g. a project's own dataclasses / pydantic
    models). Prefer these over the full bypass.
  * `AGENTIX_PICKLE_TRUST=1` restores plain `pickle.loads` for deployments where
    the entire sandbox — including any workload it runs — is trusted.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

# Inert reconstruction helpers: functions that only rebuild a value from
# following (also-gated) arguments. Each is reviewed to have no external effect.
SAFE_CALLABLES: set[tuple[str, str]] = {
    ("copyreg", "_reconstructor"),
    ("copyreg", "__newobj__"),
    ("copyreg", "__newobj_ex__"),
    ("numpy._core.multiarray", "_reconstruct"),  # numpy >= 2.0
    ("numpy.core.multiarray", "_reconstruct"),  # numpy < 2.0
}

# Value types whose construction has no external side effect. Reviewed one by
# one; modules here are stdlib/data packages that are inert to import.
SAFE_TYPES: set[tuple[str, str]] = {
    ("datetime", "date"),
    ("datetime", "time"),
    ("datetime", "datetime"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("decimal", "Decimal"),
    ("fractions", "Fraction"),
    ("uuid", "UUID"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("collections", "Counter"),
    ("collections", "deque"),
    ("pathlib", "PurePath"),
    ("pathlib", "PurePosixPath"),
    ("pathlib", "PureWindowsPath"),
    ("pathlib", "Path"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}

# Builtin value types admitted by identity (see `_SAFE_BUILTIN_TYPES`) plus every
# builtin exception subclass. Builtins are always imported, so resolving one has
# no import side effect. Builtin *functions* (getattr/eval/exec/compile/open/
# __import__/…) are types-of `builtin_function_or_method`, not `type`, so the
# "must be an allowed type" check below excludes them.
_SAFE_BUILTIN_TYPES: frozenset[type] = frozenset(
    {
        complex, range, slice, bytearray, frozenset,
        set, list, dict, tuple, bytes, str, int, float, bool,
    }
)

# First-party namespace, trusted by default. The #116 boundary defends against
# an *untrusted workload's* return value; the framework and its plugins are part
# of the trusted computing base that builds the bundle and runs the sandbox, and
# their return types (`TunnelHandle`, `BashResult`, agent results, …) are inert
# dataclasses/models. Trusting `agentix.*` keeps the framework's own paths
# working; a workload's own return types stay opt-in, and gadget callables
# (subprocess/os/eval/attribute-access helpers) are never first-party so remain
# refused.
_FIRST_PARTY_PREFIXES: tuple[str, ...] = ("agentix",)

# Module prefixes the caller has explicitly opted to trust for return types the
# default set does not cover.
_ALLOWED_MODULE_PREFIXES: set[str] = set()


class RestrictedUnpickleError(pickle.UnpicklingError):
    """A global in the stream was not on the host allowlist and was refused."""


def allow_module(prefix: str) -> None:
    """Trust every global whose module equals or starts with `prefix` (dotted).
    Use for a package whose return types the default allowlist does not cover."""
    _ALLOWED_MODULE_PREFIXES.add(prefix)


def allow_callable(module: str, name: str) -> None:
    """Trust one specific `module.name` type/helper by exact identity."""
    SAFE_CALLABLES.add((module, name))


def _trust_enabled() -> bool:
    return os.environ.get("AGENTIX_PICKLE_TRUST", "").strip().lower() in ("1", "true", "yes")


def _module_allowed(module: str) -> bool:
    prefixes = (*_FIRST_PARTY_PREFIXES, *_ALLOWED_MODULE_PREFIXES)
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _is_safe_type(module: str, name: str) -> bool:
    """`(module, name)` is an allowlisted value type, tolerating a private
    implementation submodule of an allowlisted public module — e.g. Python 3.13
    pickles `pathlib.PurePosixPath` as `pathlib._local.PurePosixPath`. Only
    underscore-prefixed submodules of the public root are accepted, so a public
    sibling module (`pathlib.evil`) is not."""
    if (module, name) in SAFE_TYPES:
        return True
    head, _, rest = module.partition(".")
    if rest and all(part.startswith("_") for part in rest.split(".")):
        return (head, name) in SAFE_TYPES
    return False


class RestrictedUnpickler(pickle.Unpickler):
    """`pickle.Unpickler` whose `find_class` enforces the allowlist above."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in SAFE_CALLABLES or _is_safe_type(module, name) or _module_allowed(module):
            return super().find_class(module, name)

        if module == "builtins":
            # Builtins are already imported — resolving has no import effect.
            obj = super().find_class(module, name)
            if isinstance(obj, type) and (obj in _SAFE_BUILTIN_TYPES or issubclass(obj, BaseException)):
                return obj
            raise RestrictedUnpickleError(
                f"refusing builtins.{name}: only builtin value types and exceptions may "
                f"cross the sandbox→host boundary. Set AGENTIX_PICKLE_TRUST=1 to trust the "
                f"sandbox fully."
            )

        # Not on the allowlist — refuse WITHOUT importing `module` (importing an
        # arbitrary module would itself run its top-level code on the host).
        raise RestrictedUnpickleError(
            f"refusing to reconstruct {module}.{name}: it is not on the host allowlist for "
            f"sandbox return values. If this is a return type you trust, call "
            f"agentix.runtime.shared.safepickle.allow_module({module!r}) (or allow_callable) "
            f"before the call; or set AGENTIX_PICKLE_TRUST=1 to trust the sandbox fully."
        )


def restricted_loads(data: bytes) -> Any:
    """Decode sandbox-supplied pickle bytes through the host allowlist.

    Honors `AGENTIX_PICKLE_TRUST=1` as a full-trust bypass (plain `pickle.loads`)."""
    if _trust_enabled():
        return pickle.loads(data)
    return RestrictedUnpickler(io.BytesIO(data)).load()


__all__ = [
    "RestrictedUnpickleError",
    "RestrictedUnpickler",
    "SAFE_CALLABLES",
    "SAFE_TYPES",
    "allow_callable",
    "allow_module",
    "restricted_loads",
]
