"""Session server wrapper around the vendored Miles implementation."""

from __future__ import annotations

from typing import Any


class SessionServer:
    """Lazy wrapper for Miles' standalone FastAPI session server."""

    def __init__(self, args: Any, backend_url: str):
        from tito_gateway.vendor.miles_compat.rollout.session.session_server import (
            SessionServer as MilesSessionServer,
        )

        self._impl = MilesSessionServer(args, backend_url)
        self.args = args
        self.backend_url = backend_url
        self.app = self._impl.app
