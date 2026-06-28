"""Optional Miles agentic tool-call generate bridge.

TITO Gateway vendors the session verifier wrapper, but the full Miles rollout
engine remains an optional dependency.  The callable exists so verifier helper
paths are importable and testable; real e2e execution must install/provide the
Miles rollout stack or monkeypatch this bridge in CPU-fast tests.
"""

from __future__ import annotations

import argparse
from typing import Any


async def generate(input: Any) -> Any:
    raise RuntimeError(
        "Miles agentic tool-call rollout generation is not bundled with "
        "tito-gateway. Install the optional Miles/SGLang training stack before "
        "running full session verifier e2e jobs."
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)
    parser.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")


generate.add_arguments = _add_arguments
