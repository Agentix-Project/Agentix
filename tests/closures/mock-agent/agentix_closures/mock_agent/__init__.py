"""mock-agent — reference closure for Agentix integration tests.

Stubs only. The runtime imports `agentix_closures.mock_agent._register`
to get a Dispatcher bound to `_impl`.
"""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.1.0"


@dataclass
class RunResult:
    exit_code: int
    patch: str


def run(instruction: str, workdir: str = "/") -> RunResult:
    """Run against an instruction; returns a fake patch echoing the input."""
    raise NotImplementedError("call via RuntimeClient.remote(mock_agent.run, ...)")
