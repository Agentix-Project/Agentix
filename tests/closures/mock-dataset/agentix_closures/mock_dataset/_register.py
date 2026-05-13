"""mock-dataset Dispatcher registration."""

from __future__ import annotations

from agentix.dispatch import Dispatcher

from . import setup, verify
from ._impl import setup as _setup_impl, verify as _verify_impl


def register() -> Dispatcher:
    d = Dispatcher()
    d.bind(setup, _setup_impl)
    d.bind(verify, _verify_impl)
    return d
