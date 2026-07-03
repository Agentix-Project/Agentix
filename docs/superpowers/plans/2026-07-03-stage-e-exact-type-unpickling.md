# Stage E Exact-Type Restricted Unpickling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace module-prefix pickle trust with an exact value-type registry while preserving all shipped unary Agentix return types.

**Architecture:** Default value classes and inert pickle helpers live in immutable exact-name tables. Host code can opt in one additional class with `allow_type(cls)`, which stores the approved class object under its exact module and qualname; all module-prefix and callable opt-ins are deleted. Tests first demonstrate the missing API and the unsafe prefix behavior using only pure in-memory functions.

**Tech Stack:** Python 3.11–3.13, stdlib `pickle`, pytest, Pydantic, uv, pyright.

---

## File map

- Modify `tests/runtime/test_safepickle.py`: test exact class opt-in, reject unregistered first-party functions, and verify one load cannot mutate decoder policy.
- Modify `agentix/runtime/shared/safepickle.py`: immutable exact registries, six first-party value identities, `allow_type`, and exact `find_class` enforcement.
- Modify `tests/conftest.py`: opt in only `tests._worker_target.EchoResult`, not its whole module.
- Leave `agentix/runtime/PROTOCOL.md` for the later sanitized Fable compatibility/documentation pass.

### Task 1: Define exact class opt-in behavior

**Files:**
- Modify: `tests/runtime/test_safepickle.py`
- Modify: `agentix/runtime/shared/safepickle.py`

- [ ] **Step 1: Add a failing exact-opt-in test and registry restoration**

Update the autouse fixture so it snapshots `safepickle._ALLOWED_TYPES` when that
attribute exists. Replace the module-wide custom-type test with:

```python
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
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest -q tests/runtime/test_safepickle.py::test_allow_type_opt_in_is_exact
```

Expected: FAIL at `assert allow_type is not None` because Stage E has no exact
type-registration API.

- [ ] **Step 3: Implement the minimal exact class registry**

In `agentix/runtime/shared/safepickle.py`, add:

```python
GlobalName = tuple[str, str]
_ALLOWED_TYPES: dict[GlobalName, type[Any]] = {}


def allow_type(cls: type[Any]) -> None:
    if not isinstance(cls, type):
        raise TypeError("allow_type() requires a class")
    _ALLOWED_TYPES[(cls.__module__, cls.__qualname__)] = cls
```

Before other global checks in `find_class()`, return the stored class on an
exact `_ALLOWED_TYPES` match. Add `allow_type` to `__all__`. Do not remove the
legacy prefix branch yet; that is the later GREEN step for security regressions.

- [ ] **Step 4: Verify GREEN for exact opt-in**

Run the same pytest node. Expected: PASS, while the same-module Pydantic sibling
remains rejected.

### Task 2: Prove prefix and policy functions are rejected

**Files:**
- Modify: `tests/runtime/test_safepickle.py`

- [ ] **Step 1: Add harmless global-resolution helpers and tests**

Add a helper that creates a protocol-0 `GLOBAL` lookup without invoking the
resolved object:

```python
def _global_reference(module: str, name: str) -> bytes:
    assert "\n" not in module and "\n" not in name
    return f"c{module}\n{name}\n.".encode("ascii")
```

Add tests for one ordinary internal function and all Stage E policy functions:

```python
def test_unregistered_first_party_function_is_refused() -> None:
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(
            _global_reference(
                "agentix.runtime.shared.safepickle", "_trust_enabled"
            )
        )


@pytest.mark.parametrize("name", ["allow_module", "allow_callable", "allow_type"])
def test_decoder_policy_function_is_refused(name: str) -> None:
    with pytest.raises(RestrictedUnpickleError):
        restricted_loads(
            _global_reference("agentix.runtime.shared.safepickle", name)
        )
```

`GLOBAL + STOP` only resolves the named object. It never invokes a process,
network operation, file operation, or the function itself.

- [ ] **Step 2: Add an in-memory one-load policy-mutation regression**

Use a pure recorder in the test module:

```python
_POLICY_CALLS: list[str] = []


def _policy_recorder(value: str) -> str:
    _POLICY_CALLS.append(value)
    return value


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
```

This uses only an in-memory list and string. On the vulnerable implementation,
the first reduction expands the callable table and the second reaches the pure
recorder.

- [ ] **Step 3: Run all three security tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/runtime/test_safepickle.py::test_unregistered_first_party_function_is_refused \
  tests/runtime/test_safepickle.py::test_decoder_policy_function_is_refused \
  tests/runtime/test_safepickle.py::test_one_load_cannot_modify_policy_then_invoke_new_global
```

Expected: failures with `DID NOT RAISE`; the policy-mutation case also records
`marker` on the current prefix-trusting implementation.

### Task 3: Replace prefix trust with immutable exact registries

**Files:**
- Modify: `agentix/runtime/shared/safepickle.py`
- Modify: `tests/runtime/test_safepickle.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Make helper and type tables immutable and exact**

Rename the helper table to internal `_SAFE_CALLABLES` and make both tables
`frozenset[GlobalName]`. Add the six reviewed unary Agentix return identities:

```python
("agentix.bridge.proxy", "TunnelHandle")
("agentix.bash", "BashResult")
("agentix.files", "UploadResult")
("agentix.agents.claude_code.agent", "ClaudeCodeResult")
("agentix.agents.qwen_code", "Result")
("agentix.plugins.datasets.swe.env", "PrepareEnvResult")
```

Keep the existing stdlib/numpy types and add both explicit Python identities for
each pathlib type:

```python
("pathlib", "PurePosixPath")
("pathlib._local", "PurePosixPath")
```

Repeat the pair for `PurePath`, `PureWindowsPath`, `Path`, `PosixPath`, and
`WindowsPath`. Delete `_is_safe_type()` and its fuzzy private-submodule rule.

- [ ] **Step 2: Delete every broad or executable opt-in surface**

Delete `_FIRST_PARTY_PREFIXES`, `_ALLOWED_MODULE_PREFIXES`, `_module_allowed()`,
`allow_module()`, and `allow_callable()`. Remove them from `__all__` and update
the module/error prose to direct callers to:

```python
from agentix.runtime.shared.safepickle import allow_type
allow_type(ProjectResult)
```

- [ ] **Step 3: Enforce exact resolution in `find_class()`**

Use this decision order:

```python
key = (module, name)
registered = _ALLOWED_TYPES.get(key)
if registered is not None:
    return registered
if key in SAFE_TYPES:
    obj = super().find_class(module, name)
    if not isinstance(obj, type):
        raise RestrictedUnpickleError(
            f"refusing {module}.{name}: an allowlisted value type resolved "
            f"to non-type {type(obj).__name__}"
        )
    return obj
if key in _SAFE_CALLABLES:
    return super().find_class(module, name)
```

Then retain the current identity-based builtin value/exception check. Reject
everything else before importing its module.

- [ ] **Step 4: Run the Task 2 nodes and verify GREEN**

Before running, update the test fixture to snapshot only `_ALLOWED_TYPES` and
`_POLICY_CALLS`, because callable/default tables are now immutable and legacy
prefix state no longer exists. Remove the one-load test's `SAFE_CALLABLES`
snapshot/assert; `_ALLOWED_TYPES` is the only mutable policy registry in the
final design and remains asserted unchanged. Also replace the session-level
`allow_module("tests._worker_target")` in `tests/conftest.py` with exact
`allow_type(EchoResult)` registration so pytest can collect after the legacy API
is deleted.

Expected: all cases raise `RestrictedUnpickleError`, the pure recorder remains
untouched, and `_ALLOWED_TYPES` retains its pre-load state.

### Task 4: Restore test and first-party compatibility precisely

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/runtime/test_safepickle.py`

- [ ] **Step 1: Replace the test module opt-in**

In `tests/conftest.py`, replace `allow_module("tests._worker_target")` with:

```python
from tests._worker_target import EchoResult

safepickle.allow_type(EchoResult)
```

Update the nearby comment to say that only the exact Pydantic result class is
approved.

- [ ] **Step 2: Restore registries without legacy prefixes**

The safepickle test fixture snapshots `_ALLOWED_TYPES` and restores that dict in
`finally`. Remove all references to `_ALLOWED_MODULE_PREFIXES` and mutable
callable tables.

- [ ] **Step 3: Expand first-party value round trips**

Construct and round-trip these inert values in
`test_first_party_return_types_round_trip_by_default()`:

```python
TunnelHandle(url="http://127.0.0.1:9", port=9)
BashResult(exit_code=0, stdout="o", stderr="")
UploadResult(path="/workspace/a", size=1)
ClaudeCodeResult(returncode=0, stdout=b"o", stderr=b"")
QwenResult(exit_code=0, stdout="o", stderr="")
PrepareEnvResult(ok=True, head="abc", log="")
```

Use the defining modules for imports when needed, matching their pickle
identities.

- [ ] **Step 4: Add API-type validation**

Add:

```python
def test_allow_type_requires_a_class() -> None:
    not_a_class: Any = lambda: None
    with pytest.raises(TypeError, match="requires a class"):
        safepickle.allow_type(not_a_class)
```

Import `Any` from `typing` in the test module; no source-level type suppression
is needed.

- [ ] **Step 5: Run the complete focused suite**

Run:

```bash
uv run pytest -q tests/runtime/test_safepickle.py tests/runtime/test_protocol.py tests/test_public_exports.py
```

Expected: all focused tests pass, including HTTP remote round-trip using the
exact `EchoResult` opt-in.

### Task 5: Type-check and commit the Codex-owned security core

**Files:**
- Modify: `agentix/runtime/shared/safepickle.py`
- Modify: `tests/conftest.py`
- Modify: `tests/runtime/test_safepickle.py`

- [ ] **Step 1: Run formatting and type checks for touched code**

Run:

```bash
uv run ruff format --check agentix/runtime/shared/safepickle.py tests/conftest.py tests/runtime/test_safepickle.py
uv run ruff check agentix/runtime/shared/safepickle.py tests/conftest.py tests/runtime/test_safepickle.py
uv run pyright agentix/runtime/shared/safepickle.py tests/conftest.py tests/runtime/test_safepickle.py
```

Expected: zero formatting, lint, and type errors. Fix root types rather than
adding `type: ignore` comments.

- [ ] **Step 2: Confirm scope and working tree**

Run `git diff --check` and inspect `git diff --stat`. Only the three Codex-owned
code/test files plus this plan should be uncommitted.

- [ ] **Step 3: Create a local implementation commit**

```bash
git add agentix/runtime/shared/safepickle.py tests/conftest.py \
  tests/runtime/test_safepickle.py \
  docs/superpowers/plans/2026-07-03-stage-e-exact-type-unpickling.md
git commit -m "security: restrict pickle globals to exact value types"
```

Do not push, open a pull request, or modify unrelated Stage E/master files.
