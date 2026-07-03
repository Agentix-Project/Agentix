"""The host restricts what it will reconstruct from a sandbox return value (#116).

`pickle.loads` reconstructs objects by invoking whatever callables a stream
names, so decoding a sandbox-influenced value is a trust boundary. `restricted_loads`
decodes through a strict allowlist: only reviewed value types and inert
reconstruction helpers are permitted; everything else is refused without
importing it. These tests verify both halves — permitted values round-trip, and
reconstruction of non-allowlisted callables/types is refused.
"""

from __future__ import annotations

import collections
import datetime
import decimal
import fractions
import pathlib
import pickle
import uuid
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from agentix.runtime.shared import safepickle
from agentix.runtime.shared.safepickle import (
    RestrictedUnpickleError,
    restricted_loads,
)


def _global_reference(module: str, name: str) -> bytes:
    assert "\n" not in module and "\n" not in name
    return f"c{module}\n{name}\n.".encode("ascii")


_POLICY_CALLS: list[str] = []


def _policy_recorder(value: str) -> str:
    _POLICY_CALLS.append(value)
    return value


@pytest.fixture(autouse=True)
def _restore_allowlist():
    """Restore exact type opt-ins so tests do not leak policy state."""
    allowed_types = dict(safepickle._ALLOWED_TYPES)
    policy_calls = list(_POLICY_CALLS)
    try:
        yield
    finally:
        safepickle._ALLOWED_TYPES.clear()
        safepickle._ALLOWED_TYPES.update(allowed_types)
        _POLICY_CALLS.clear()
        _POLICY_CALLS.extend(policy_calls)


# ── objects that direct reconstruction at non-allowlisted callables ──────────
# Each `__reduce__` names a callable that a restricted host decode must refuse.
# The referenced callables are the ones named in issue #116; the arguments are
# benign so nothing runs even if a regression let one through.


class _ReducesToSubprocessCheckOutput:
    def __reduce__(self):
        import subprocess

        return (subprocess.check_output, (["true"],))


class _ReducesToSubprocessPopen:
    def __reduce__(self):
        import subprocess

        return (subprocess.Popen, (["true"],))


class _ReducesToEval:
    def __reduce__(self):
        return (eval, ("1 + 1",))


class _ReducesToOsSystem:
    def __reduce__(self):
        import os

        return (os.system, ("true",))


@pytest.mark.parametrize(
    "obj_cls",
    [
        _ReducesToSubprocessCheckOutput,
        _ReducesToSubprocessPopen,
        _ReducesToEval,
        _ReducesToOsSystem,
    ],
)
def test_non_allowlisted_callable_is_refused(obj_cls) -> None:
    blob = pickle.dumps(obj_cls())
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(blob)


def test_unregistered_first_party_function_is_refused() -> None:
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(_global_reference("agentix.runtime.shared.safepickle", "_trust_enabled"))


@pytest.mark.parametrize("name", ["allow_module", "allow_callable", "allow_type"])
def test_policy_functions_are_refused(name: str) -> None:
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(_global_reference("agentix.runtime.shared.safepickle", name))


def test_one_load_cannot_modify_policy_then_invoke_new_global() -> None:
    _POLICY_CALLS.clear()
    before_types = dict(safepickle._ALLOWED_TYPES)
    payload = (
        b"\x80\x04"
        b"cagentix.runtime.shared.safepickle\nallow_callable\n"
        b"(Vtests.runtime.test_safepickle\nV_policy_recorder\ntR0"
        b"ctests.runtime.test_safepickle\n_policy_recorder\n"
        b"(Vmarker\ntR."
    )
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(payload)
    assert _POLICY_CALLS == []
    assert safepickle._ALLOWED_TYPES == before_types


def test_refusal_does_not_import_the_named_module() -> None:
    """A refusal is decided from the module name — the restricted decoder must
    not import a non-allowlisted module (importing runs its top-level code on
    the host)."""
    import sys

    modname = "xml.dom.minidom"  # importable, stdlib, not on the allowlist
    sys.modules.pop(modname, None)
    crafted = b"\x80\x04c" + modname.encode() + b"\nDocument\n."
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(crafted)
    assert modname not in sys.modules, "restricted decode must not import a non-allowlisted module"


@pytest.mark.parametrize(
    "crafted",
    [
        # Attribute-access helpers — the technique that would let a stream walk
        # from an admitted object to an arbitrary callable. Neither the pure-
        # python nor the C-accelerator module may be on the allowlist.
        b"\x80\x04coperator\nattrgetter\n.",
        b"\x80\x04c_operator\nattrgetter\n.",
        b"\x80\x04c_operator\nitemgetter\n.",
        b"\x80\x04c_operator\nmethodcaller\n.",
        b"\x80\x04cbuiltins\ngetattr\n.",
    ],
)
def test_attribute_access_helpers_are_refused(crafted) -> None:
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(crafted)


# ── permitted values round-trip ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        42, "hi", 3.5, True, None, b"bytes", bytearray(b"ba"),
        complex(1, 2), range(0, 10, 2), slice(1, 9, 2),
        [1, 2, 3], {"a": 1}, (1, 2), {1, 2}, frozenset([1]),
        {"nested": [1, {"b": (2, 3)}]},
        datetime.datetime(2026, 1, 1, 12, 30),
        datetime.date(2026, 1, 1),
        datetime.timedelta(days=2),
        datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        decimal.Decimal("1.5"),
        fractions.Fraction(3, 4),
        uuid.UUID("12345678123456781234567812345678"),
        collections.OrderedDict(a=1, b=2),
        collections.Counter("aabbc"),
        collections.deque([1, 2, 3]),
        collections.defaultdict(int, {"a": 1}),
        pathlib.PurePosixPath("/tmp/x"),
    ],
)
def test_permitted_values_round_trip(value) -> None:
    assert restricted_loads(pickle.dumps(value)) == value


@pytest.mark.parametrize("exc", [ValueError("boom"), KeyError("k"), RuntimeError("r")])
def test_builtin_exceptions_round_trip(exc) -> None:
    out = restricted_loads(pickle.dumps(exc))
    assert type(out) is type(exc)
    assert out.args == exc.args


def test_numpy_array_round_trips() -> None:
    np = pytest.importorskip("numpy")
    arr = np.arange(6).reshape(2, 3)
    out = restricted_loads(pickle.dumps(arr))
    assert np.array_equal(out, arr)


def test_numpy_object_array_gates_nested_globals() -> None:
    """An object-dtype array pickles its elements in the same stream, so a
    non-allowlisted callable referenced by an element is still refused."""
    np = pytest.importorskip("numpy")
    arr = np.array([_ReducesToOsSystem()], dtype=object)
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(pickle.dumps(arr))


def test_none_returns_none() -> None:
    assert restricted_loads(pickle.dumps(None)) is None


# ── custom return types: refused by default, permitted by opt-in ─────────────


class _ProjectResult(BaseModel):
    patch: str
    score: int


@dataclass
class _ProjectPoint:
    x: int
    y: int


def test_first_party_return_types_round_trip_by_default() -> None:
    """The framework's own shipped return types are trusted (`agentix.*`) and
    must cross the boundary with no opt-in — otherwise the restriction breaks
    the framework's own main paths (Proxy.start, bash.run, agent adapters)."""
    from agentix.bridge.proxy import TunnelHandle

    handle = TunnelHandle(url="http://127.0.0.1:9", port=9)
    assert restricted_loads(pickle.dumps(handle)) == handle

    bash = pytest.importorskip("agentix.bash")
    result = bash.BashResult(stdout="o", stderr="", exit_code=0)
    assert restricted_loads(pickle.dumps(result)) == result


def test_custom_type_refused_by_default() -> None:
    # This module is not first-party (`agentix.*`) and not on the opt-in list.
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(pickle.dumps(_ProjectResult(patch="d", score=1)))


def test_allow_type_opt_in_is_exact() -> None:
    point = _ProjectPoint(1, 2)
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(pickle.dumps(point))

    allow_type = getattr(safepickle, "allow_type", None)
    assert allow_type is not None
    allow_type(_ProjectPoint)
    assert restricted_loads(pickle.dumps(point)) == point

    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(pickle.dumps(_ProjectResult(patch="d", score=1)))


def test_trust_env_disables_restriction(monkeypatch) -> None:
    """The escape hatch fully trusts the sandbox. Verified with a benign
    reducer (json.dumps) that the restricted path refuses but is harmless."""
    monkeypatch.setenv("AGENTIX_PICKLE_TRUST", "1")

    class _ReducesToJsonDumps:
        def __reduce__(self):
            import json

            return (json.dumps, ([1, 2, 3],))

    assert restricted_loads(pickle.dumps(_ReducesToJsonDumps())) == "[1, 2, 3]"


def test_refusal_message_points_to_opt_in() -> None:
    with pytest.raises(RestrictedUnpickleError) as ei:
        restricted_loads(pickle.dumps(_ProjectPoint(1, 2)))
    msg = str(ei.value)
    assert "allow_module" in msg or "AGENTIX_PICKLE_TRUST" in msg
