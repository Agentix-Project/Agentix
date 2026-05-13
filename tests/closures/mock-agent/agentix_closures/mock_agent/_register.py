"""mock-agent Dispatcher registration. Runtime calls register() at import."""

from __future__ import annotations

from agentix.dispatch import Dispatcher

from . import run
from ._impl import run as _run_impl


def register() -> Dispatcher:
    d = Dispatcher()
    d.bind(run, _run_impl)
    return d
