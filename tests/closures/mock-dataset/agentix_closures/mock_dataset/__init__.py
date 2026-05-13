"""mock-dataset — reference closure for Agentix integration tests."""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.1.0"


@dataclass
class SetupResult:
    instruction: str
    workdir: str
    instance_id: str


@dataclass
class VerifyResult:
    passed: bool
    reason: str


def setup(instance_id: str) -> SetupResult:
    """Return agent_input for the given instance."""
    raise NotImplementedError("call via RuntimeClient.remote(mock_dataset.setup, ...)")


def verify(patch: str) -> VerifyResult:
    """Return whether the patch passes."""
    raise NotImplementedError("call via RuntimeClient.remote(mock_dataset.verify, ...)")
