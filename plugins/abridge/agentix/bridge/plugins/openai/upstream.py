"""OpenAI-compatible upstream client."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx
from agentix.bridge.util import join_url_path


def openai_chat_completions_url(base_url: str | None = None) -> str:
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid OPENAI_BASE_URL: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}{join_url_path(parsed.path, '/chat/completions')}"


def openai_api_key() -> str:
    return os.getenv("ABRIDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"


def call_openai_compatible(body: dict[str, Any], *, base_url: str | None = None) -> dict[str, Any]:
    headers = {
        "authorization": f"Bearer {openai_api_key()}",
        "content-type": "application/json",
        "accept": "application/json",
    }
    timeout = float(os.getenv("ABRIDGE_OPENAI_TIMEOUT", "120"))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(openai_chat_completions_url(base_url), headers=headers, json=body)
    resp.raise_for_status()
    value = resp.json()
    if not isinstance(value, dict):
        raise ValueError("openai-compatible upstream returned non-object JSON")
    return value


async def post_openai_compatible(
    body: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(openai_chat_completions_url(base_url), headers=headers, json=body)
    resp.raise_for_status()
    value = resp.json()
    if not isinstance(value, dict):
        raise ValueError("openai-compatible upstream returned non-object JSON")
    return value
