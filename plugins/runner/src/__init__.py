"""Library-first batch rollout runner for Agentix.

Run an agent over a dataset of instances — each inside its own sandbox —
and collect typed `Rollout` records. See `agentix.runner.core` for the
implementation and `agentix.runner.cli` for the `agentix-run` wrapper.
"""

from __future__ import annotations

from .core import (
    Agent,
    AgentResult,
    Dataset,
    Provider,
    Rollout,
    rollout_one,
    run_rollouts,
)

__all__ = [
    "Agent",
    "AgentResult",
    "Dataset",
    "Provider",
    "Rollout",
    "rollout_one",
    "run_rollouts",
]
