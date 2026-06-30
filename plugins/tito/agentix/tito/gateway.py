"""Python wrapper API for launching TITO Gateway beside a backend server."""

from __future__ import annotations

from dataclasses import replace

from .config import TITOGatewayConfig
from .discovery import discover_backend_url, normalize_backend_url
from .pool import BackendPool
from .server import SessionServer


class TITOGateway:
    """Small wrapper that resolves backend(s) and owns a session server app.

    Routes inference across a :class:`BackendPool` — a single backend (resolved
    by discovery) by default, or several when ``config.backend_urls`` is set.
    """

    def __init__(self, config: TITOGatewayConfig):
        if config.backend_urls:
            urls = [normalize_backend_url(u) for u in config.backend_urls]
            self.config = replace(config, backend_url=urls[0])
        else:
            backend_url = discover_backend_url(
                config.backend_url,
                probe_candidates=config.backend_probe_candidates,
                probe_timeout=config.backend_probe_timeout,
            )
            self.config = replace(config, backend_url=backend_url)
            urls = [backend_url]
        self.pool = BackendPool(urls, policy=config.routing_policy)
        self.server = SessionServer(self.config.as_session_args(), self.pool)
        self._register_health_alias()

    @classmethod
    def from_server(cls, *, hf_checkpoint: str, backend_url: str | None = None, **kwargs) -> TITOGateway:
        return cls(TITOGatewayConfig(hf_checkpoint=hf_checkpoint, backend_url=backend_url, **kwargs))

    def _register_health_alias(self) -> None:
        # abridge's Sidecar probes `/healthz` by default; the engine session
        # routes only expose `/health`. Add a thin alias so a default Sidecar
        # wiring works without overriding `health_path`.
        async def healthz() -> dict[str, str]:
            return {"status": "ok"}

        self.app.add_api_route("/healthz", healthz, methods=["GET"])

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
