"""Reserved-namespace protection and handler introspection on `agentix.Namespace`."""

from __future__ import annotations

import pytest

from agentix import sio as _sio


def test_reserved_namespace_rejected_for_subclass() -> None:
    # A user subclass must not be able to shadow a core side channel.
    class Bad(_sio.Namespace):
        namespace = "/trace"

    with pytest.raises(ValueError, match="reserved"):
        Bad()


def test_user_namespace_allowed() -> None:
    class Good(_sio.Namespace):
        namespace = "/my-plugin"

    assert Good().namespace == "/my-plugin"


def test_core_subclass_may_opt_into_reserved() -> None:
    class CoreLog(_sio.Namespace):
        namespace = "/log"
        _allow_reserved = True

    assert CoreLog().namespace == "/log"


def test_registered_handlers_lists_auto_registered() -> None:
    class Svc(_sio.Namespace):
        namespace = "/svc"

        async def on_ping(self, data: object) -> None: ...

    handlers = Svc().registered_handlers()
    assert "ping" in handlers
    assert len(handlers["ping"]) == 1
