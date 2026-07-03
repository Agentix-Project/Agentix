# Stage E Exact-Type Restricted Unpickling Design

## Status and scope

This design replaces the unsafe default trust of the complete `agentix.*`
namespace in the Stage E sandbox-to-host return decoder. It is intentionally
limited to `agentix.runtime.shared.safepickle`, its public API and protocol
documentation, and directly relevant unit/integration tests. Existing issues in
providers, streaming, runner loading, packaging, documentation maturity, and
release process remain outside this change.

The work is split sequentially. Codex owns the security-sensitive decoder
change and harmless regression tests. A later Fable pass receives only a
sanitized compatibility task: check the first-party type inventory, update
ordinary documentation, and run the full verification suite.

## Security invariant

Sandbox-controlled pickle data may resolve only:

1. exact value-type identities that were individually reviewed; and
2. exact reconstruction-helper identities that were individually reviewed.

Package provenance is not a safety property. A trusted package can contain
effectful functions, policy-mutating functions, and classes with effectful
construction hooks. Therefore neither `agentix.*` nor a workload package may be
admitted by module prefix.

## Decoder and registration design

`RestrictedUnpickler.find_class()` uses two closed exact registries:

- `SAFE_TYPES: frozenset[tuple[str, str]]` for value-shaped classes;
- an internal immutable helper table for inert reconstruction helpers.

The `_FIRST_PARTY_PREFIXES`, `_ALLOWED_MODULE_PREFIXES`, `_module_allowed()`,
and `allow_module()` surfaces are deleted. There is no compatibility alias or
deprecation shim because the repository explicitly permits breaking design
changes.

A new `allow_type(cls: type[Any]) -> None` host-side opt-in stores the exact
`(cls.__module__, cls.__qualname__)` identity and the class object in a private
registry. It accepts only actual classes. On a match, `find_class()` returns the
registered class object rather than importing the module again, so later module
rebinding cannot change what the operator approved.

The public `allow_callable()` surface is also deleted. It is new in Stage E,
has no repository caller, and conflicts with the value-type-only policy. The
small fixed set of pickle reconstruction helpers stays internal and immutable.

Required first-party return classes are listed as exact string identities in
`SAFE_TYPES`. String identities preserve workspace dependency separation: core
does not import abridge or runtime-basic packages while importing safepickle.
The class module is imported only if a return pickle actually references that
exact reviewed identity. Python-version aliases such as `pathlib.*` and
`pathlib._local.*` are enumerated explicitly; no private-submodule fuzzy match
is retained.

## Data flow and failure behavior

The worker still serializes a result with stdlib pickle. On the host, both HTTP
and Socket.IO result paths call `restricted_loads()`.

For each referenced global:

1. exact safe helper/type identity is checked before importing its module;
2. approved builtin types and builtin exceptions retain their identity-based
   check;
3. every other global is rejected with `RestrictedUnpickleError` without
   importing the named module.

A failed decode must not alter either registry. `AGENTIX_PICKLE_TRUST=1` remains
the explicit full-trust escape hatch and is outside the protected default.

## Harmless regression strategy

Security regression tests contain no real shell command, process launch, file
write, or network request. They use pickle `GLOBAL` lookups, pure functions,
monkeypatched recorders, and registry snapshots.

The test-first cases are:

- an unregistered function from an `agentix.*` module is rejected before it can
  be invoked;
- decoder configuration functions are themselves rejected;
- one pickle load cannot change policy and then resolve a previously forbidden
  pure callable;
- registry state is unchanged after rejection;
- a workload class is rejected by default and round-trips only after
  `allow_type(TheClass)`;
- every exact first-party value type required by Agentix main paths round-trips
  by default;
- direct non-first-party callable and attribute-access gadget regressions remain
  green;
- protocol tests cover both HTTP fast-path and Socket.IO fallback decoding.

Each new behavioral test is run against the current implementation first and
must fail for the intended missing control before production code changes.

## Verification and handoff

Codex runs the focused safepickle and protocol tests while implementing the
security core. After the focused suite passes, the sanitized Fable task is to:

1. compare the exact first-party type inventory with actual public return
   surfaces;
2. add any missing compatibility-only cases using inert values;
3. update protocol/public API prose without exploit descriptions;
4. run the complete pytest suite and whole-workspace pyright; and
5. create a local commit without pushing or opening a pull request.

The final merge gate is that all intended values cross the boundary, every
unregistered global remains rejected regardless of package prefix, failed loads
cannot mutate policy, and both transport paths use the same protected decoder.
