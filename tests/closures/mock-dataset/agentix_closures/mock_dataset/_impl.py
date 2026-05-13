"""mock-dataset impl — runs inside the sandbox."""

from __future__ import annotations

from . import SetupResult, VerifyResult


def setup(instance_id: str) -> SetupResult:
    return SetupResult(
        instruction=f"Solve instance {instance_id}",
        workdir="/workspace",
        instance_id=instance_id,
    )


def verify(patch: str) -> VerifyResult:
    return VerifyResult(passed=True, reason=f"mock verify (patch was {len(patch)} chars)")
