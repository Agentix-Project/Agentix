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

Only the exact first-party value types listed in `SAFE_TYPES` are trusted by
default. The framework and its plugins build the bundle and run the sandbox,
so these reviewed return types (`TunnelHandle`, `BashResult`, agent results,
…) are part of the trusted computing base. Other first-party globals and a
workload's own types stay opt-in.

Extending / relaxing:

  * `allow_type(Class)` adds one exact runtime class the default set does not
    cover (e.g. a project's own dataclass / pydantic model). Prefer this over
    the full bypass.
  * `AGENTIX_PICKLE_TRUST=1` restores plain `pickle.loads` for deployments where
    the entire sandbox — including any workload it runs — is trusted.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

GlobalName = tuple[str, str]
_ALLOWED_TYPES: dict[GlobalName, type[Any]] = {}

# Inert reconstruction helpers: functions that only rebuild a value from
# following (also-gated) arguments. Each is reviewed to have no external effect.
_SAFE_CALLABLES: frozenset[GlobalName] = frozenset(
    {
        ("copyreg", "_reconstructor"),
        ("copyreg", "__newobj__"),
        ("copyreg", "__newobj_ex__"),
        ("numpy._core.multiarray", "_reconstruct"),  # numpy >= 2.0
        ("numpy.core.multiarray", "_reconstruct"),  # numpy < 2.0
    }
)

# Value types whose construction has no external side effect. Reviewed one by
# one; modules here are stdlib/data packages or exact first-party value modules.
SAFE_TYPES: frozenset[GlobalName] = frozenset(
    {
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
        ("pathlib._local", "PurePath"),
        ("pathlib._local", "PurePosixPath"),
        ("pathlib._local", "PureWindowsPath"),
        ("pathlib._local", "Path"),
        ("pathlib._local", "PosixPath"),
        ("pathlib._local", "WindowsPath"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("agentix.bridge.proxy", "TunnelHandle"),
        ("agentix.bash", "BashResult"),
        ("agentix.files", "UploadResult"),
        ("agentix.agents.claude_code.agent", "ClaudeCodeResult"),
        ("agentix.agents.qwen_code", "Result"),
        ("agentix.plugins.datasets.swe.env", "PrepareEnvResult"),
    }
)

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


class RestrictedUnpickleError(pickle.UnpicklingError):
    """A global in the stream was not on the host allowlist and was refused."""


def allow_type(cls: type[Any]) -> None:
    """Trust one exact runtime class for sandbox return reconstruction."""
    if not isinstance(cls, type):
        raise TypeError("allow_type() requires a class")
    _ALLOWED_TYPES[(cls.__module__, cls.__qualname__)] = cls


def _trust_enabled() -> bool:
    return os.environ.get("AGENTIX_PICKLE_TRUST", "").strip().lower() in ("1", "true", "yes")


class RestrictedUnpickler(pickle.Unpickler):
    """`pickle.Unpickler` whose `find_class` enforces the allowlist above."""

    def find_class(self, module: str, name: str) -> Any:
        global_name = (module, name)
        allowed_type = _ALLOWED_TYPES.get(global_name)
        if allowed_type is not None:
            return allowed_type

        if global_name in SAFE_TYPES:
            obj = super().find_class(module, name)
            if not isinstance(obj, type):
                raise RestrictedUnpickleError(
                    f"refusing to reconstruct {module}.{name}: the allowlisted value "
                    f"global did not resolve to a type"
                )
            return obj

        if global_name in _SAFE_CALLABLES:
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
            f"sandbox return values. If this is a return type you trust, import its class "
            f"and call agentix.runtime.shared.safepickle.allow_type(Class) before the call; "
            f"or set AGENTIX_PICKLE_TRUST=1 to trust the sandbox fully."
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
    "SAFE_TYPES",
    "allow_type",
    "restricted_loads",
]
