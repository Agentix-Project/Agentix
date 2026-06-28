"""Python wrapper API for launching TITO Gateway beside a backend server."""

from __future__ import annotations

from dataclasses import replace

from tito_gateway.config import TITOGatewayConfig
from tito_gateway.discovery import discover_backend_url
from tito_gateway.server import SessionServer


class TITOGateway:
    """Small wrapper that resolves a backend and owns a session server app."""

    def __init__(self, config: TITOGatewayConfig):
        backend_url = discover_backend_url(
            config.backend_url,
            probe_candidates=config.backend_probe_candidates,
            probe_timeout=config.backend_probe_timeout,
        )
        self.config = replace(config, backend_url=backend_url)
        self.server = SessionServer(self.config.as_miles_namespace(), backend_url)

    @classmethod
    def from_server(cls, *, hf_checkpoint: str, backend_url: str | None = None, **kwargs) -> "TITOGateway":
        return cls(TITOGatewayConfig(hf_checkpoint=hf_checkpoint, backend_url=backend_url, **kwargs))

    @property
    def app(self):
        return self.server.app

    def run(self) -> None:
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.session_server_ip,
            port=self.config.session_server_port,
            log_level="info",
        )
