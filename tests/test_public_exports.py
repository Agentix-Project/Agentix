"""The first-failure vocabulary and `configure_logging` are reachable from
the top-level `agentix` surface — a caller writing `except <Timeout>:` after
a failed run should not have to reach into private `agentix.runtime.client.*`.
"""

from __future__ import annotations

import agentix


def test_failure_vocabulary_importable_from_agentix() -> None:
    from agentix import (
        CallCancelled,
        CallTimeout,
        Failed,
        Ok,
        RemoteCallError,
        RestrictedUnpickleError,
        Result,
        RuntimeUnreachable,
        WorkerExited,
        configure_logging,
    )

    # WorkerExited / CallCancelled are RemoteCallError subclasses, so
    # `except RemoteCallError` still catches them while the narrower names
    # allow branching.
    assert issubclass(WorkerExited, RemoteCallError)
    assert issubclass(CallCancelled, RemoteCallError)
    assert issubclass(RestrictedUnpickleError, Exception)
    assert Ok(1).value == 1 and Failed(ValueError()).error is not None
    assert Result[int]  # generic union alias is subscriptable
    assert callable(configure_logging)
    for name in (CallTimeout, RuntimeUnreachable):
        assert issubclass(name, Exception)


def test_failure_vocabulary_in_dunder_all() -> None:
    for name in (
        "CallCancelled", "CallTimeout", "Failed", "Ok", "Result",
        "RestrictedUnpickleError", "RuntimeUnreachable", "WorkerExited", "configure_logging",
    ):
        assert name in agentix.__all__


def test_providers_importable_from_provider_package() -> None:
    # The documented `from agentix.provider import providers` must work
    # (previously only `agentix.provider.base` exported it -> ImportError).
    from agentix.provider import SandboxConfig, providers, register_provider

    reg = providers()
    assert hasattr(reg, "get") and callable(register_provider) and SandboxConfig is not None
