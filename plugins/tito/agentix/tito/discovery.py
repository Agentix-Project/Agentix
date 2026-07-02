"""Backend URL discovery for wrapping an OpenAI-compatible server."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BACKEND_ENV_VARS = ("TITO_BACKEND_URL", "OPENAI_BASE_URL", "SGLANG_BASE_URL")
DEFAULT_BACKEND_PROBE_CANDIDATES = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:30000",
    "http://localhost:30000",
)
BACKEND_PROBE_PATHS = ("/health", "/v1/models")

logger = logging.getLogger(__name__)


def normalize_backend_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("backend URL is empty")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return normalized


def _probe_endpoint(url: str, timeout: float) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except HTTPError as exc:
        return 200 <= exc.code < 300
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def probe_backend_url(
    candidate_url: str,
    *,
    timeout: float = 0.25,
    endpoint_probe: Callable[[str, float], bool] | None = None,
) -> str | None:
    """Return the successful probe path for a backend candidate, if live."""
    backend_url = normalize_backend_url(candidate_url)
    probe = _probe_endpoint if endpoint_probe is None else endpoint_probe
    for path in BACKEND_PROBE_PATHS:
        if probe(f"{backend_url}{path}", timeout):
            return path
    return None


def discover_backend_url(
    explicit_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    env_vars: Iterable[str] = DEFAULT_BACKEND_ENV_VARS,
    probe_candidates: Iterable[str] | None = DEFAULT_BACKEND_PROBE_CANDIDATES,
    probe_timeout: float = 0.25,
) -> str:
    """Resolve the backend URL with deterministic precedence.

    Explicit config wins, followed by environment variables in
    ``DEFAULT_BACKEND_ENV_VARS`` order, followed by configured local probe
    candidates in the supplied order.
    """
    if explicit_url:
        backend_url = normalize_backend_url(explicit_url)
        logger.info("Selected backend URL from explicit config: %s", backend_url)
        return backend_url

    source = os.environ if env is None else env
    for key in env_vars:
        value = source.get(key)
        if value:
            backend_url = normalize_backend_url(value)
            logger.info("Selected backend URL from %s: %s", key, backend_url)
            return backend_url

    candidates = tuple(probe_candidates or ())
    for candidate in candidates:
        backend_url = normalize_backend_url(candidate)
        live_path = probe_backend_url(backend_url, timeout=probe_timeout)
        if live_path:
            logger.info("Selected backend URL from probe %s via %s", backend_url, live_path)
            return backend_url

    names = ", ".join(env_vars)
    candidate_text = ", ".join(candidates) if candidates else "none configured"
    raise RuntimeError(
        "backend URL not found; pass --backend-url, "
        f"set one of: {names}, or start a live backend on one of: {candidate_text}"
    )
