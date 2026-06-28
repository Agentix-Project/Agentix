"""Optional Miles command helpers used by session verifier e2e jobs."""

from __future__ import annotations

from dataclasses import dataclass


class MissingMilesTrainingStackError(RuntimeError):
    """Raised when full Miles training helpers are required but unavailable."""


@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = 1
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"


def exec_command(*args, **kwargs):
    raise MissingMilesTrainingStackError(
        "Miles command execution helpers are not bundled with tito-gateway. "
        "Install/provide the optional Miles training stack before running full "
        "session verifier e2e jobs."
    )


def execute_train(*args, **kwargs):
    raise MissingMilesTrainingStackError(
        "Miles execute_train helper is not bundled with tito-gateway. "
        "Install/provide the optional Miles training stack before running full "
        "session verifier e2e jobs."
    )
