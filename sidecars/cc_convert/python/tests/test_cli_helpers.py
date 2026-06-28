"""Unit tests for cc_convert.cli helper functions."""

from __future__ import annotations

import pytest

from cc_convert.cli import _normalize_upstream_url, _path_is_anthropic_messages


# ---------- _normalize_upstream_url ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Bare host → assume /v1/chat/completions
        ("https://api.openai.com", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/", "https://api.openai.com/v1/chat/completions"),
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        # Already has /v1 → just append /chat/completions
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
        ("http://vllm:8000/v1", "http://vllm:8000/v1/chat/completions"),
        # Different version
        ("http://x/v2", "http://x/v2/chat/completions"),
        # Already complete → verbatim (modulo trailing slash strip)
        (
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "https://api.openai.com/v1/chat/completions/",
            "https://api.openai.com/v1/chat/completions",
        ),
        # Proxy-prefix / vendor-prefix paths: pass through assumption that the
        # user knew what they were doing.
        (
            "https://my-proxy.example/openai/v1/chat/completions",
            "https://my-proxy.example/openai/v1/chat/completions",
        ),
    ],
)
def test_normalize_upstream_url(raw: str, expected: str) -> None:
    assert _normalize_upstream_url(raw) == expected


# ---------- _path_is_anthropic_messages ----------


@pytest.mark.parametrize(
    "path,expected",
    [
        # Canonical
        ("/v1/messages", True),
        ("/messages", True),
        # Trailing slash
        ("/v1/messages/", True),
        ("/messages/", True),
        # With query string
        ("/v1/messages?stream=true", True),
        # Behind a vendor / load-balancer prefix
        ("/anthropic/v1/messages", True),
        ("/api/v1/messages", True),
        ("/some/deep/prefix/v1/messages?x=1", True),
        # Negatives
        ("/v1/messages/foo", False),       # trailing extra segment
        ("/healthz", False),
        ("/version", False),
        ("/v1/messages.json", False),      # not a path boundary
        ("/translate/cc-to-oai", False),
        ("/", False),
        ("", False),
    ],
)
def test_path_is_anthropic_messages(path: str, expected: bool) -> None:
    assert _path_is_anthropic_messages(path) is expected
