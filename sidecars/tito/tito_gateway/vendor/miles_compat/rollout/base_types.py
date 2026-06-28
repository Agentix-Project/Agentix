"""Lightweight Miles rollout type shapes used by verifier imports.

The full Miles training stack owns the production rollout implementation.  This
module preserves the small dataclass surface that `session_verify_agent` needs
for import and CPU-fast wrapper tests.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerateFnInput:
    state: Any
    sample: Any
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    samples: Any
