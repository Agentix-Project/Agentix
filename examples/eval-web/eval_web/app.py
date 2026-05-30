"""FastAPI app: serves the dashboard, the catalog API, and the run WebSocket."""

from __future__ import annotations

from pathlib import Path

from eval_tui.views.catalog import discover_catalog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from eval_web.run import stream_demo_run

# The dashboard is static — read it once at import, not per request.
_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def create_app() -> FastAPI:
    app = FastAPI(title="Agentix eval-web", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/api/catalog")
    async def catalog() -> list[dict[str, str]]:
        """The installed Agentix ecosystem (same introspection the TUI shows)."""
        return [
            {"name": name, "kind": kind, "version": version, "detail": detail}
            for name, kind, version, detail in discover_catalog()
        ]

    @app.websocket("/ws/run")
    async def ws_run(ws: WebSocket) -> None:
        await ws.accept()
        try:
            n = _int(ws.query_params.get("n"), default=24, lo=1, hi=500)
            concurrency = _int(ws.query_params.get("concurrency"), default=4, lo=1, hi=64)
            await stream_demo_run(ws.send_json, n=n, n_concurrent=concurrency)
        except WebSocketDisconnect:
            return  # client navigated away mid-run
        finally:
            with _suppress():
                await ws.close()

    return app


def _int(value: str | None, *, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value))) if value is not None else default
    except (TypeError, ValueError):
        return default


def _suppress():
    import contextlib

    return contextlib.suppress(Exception)


app = create_app()
