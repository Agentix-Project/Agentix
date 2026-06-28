"""Tests for the playground's /v1/models probe — it has to accept several
non-standard response shapes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make playground importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "playground"))

import run_roundtrip as _online  # noqa: E402

probe_models = _online.probe_models


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def _patched_probe(response_payload: dict | list):
    """Return a callable that mocks the URL opener to return `response_payload`."""

    def fake_open(req, timeout=5):  # noqa: ARG001
        return _FakeResp(json.dumps(response_payload).encode())

    class _O:
        def open(self, req, timeout=5):
            return fake_open(req, timeout)

    return _O


@pytest.mark.parametrize(
    "payload,expected",
    [
        # Standard OpenAI shape
        ({"data": [{"id": "gpt-4"}, {"id": "gpt-3.5"}]}, ["gpt-4", "gpt-3.5"]),
        # OpenAI shape with extras
        (
            {"data": [{"id": "gpt-4", "object": "model", "owned_by": "openai"}], "object": "list"},
            ["gpt-4"],
        ),
        # SGLang style: {"models": ["string", ...]}
        ({"models": ["/model"]}, ["/model"]),
        ({"models": ["Qwen2.5-7B", "Llama-3"]}, ["Qwen2.5-7B", "Llama-3"]),
        # SGLang variant: {"models": [{"id": "..."}]}
        ({"models": [{"id": "deepseek-r1"}]}, ["deepseek-r1"]),
        # Plain list of strings
        (["gpt-4", "claude"], ["gpt-4", "claude"]),
        # Plain list of dicts
        ([{"id": "gpt-4"}], ["gpt-4"]),
    ],
)
def test_probe_models_recognizes_various_shapes(payload, expected):
    with patch.object(_online, "_opener_no_proxy", lambda: _patched_probe(payload)()):
        assert probe_models("http://x") == expected


def test_probe_models_returns_empty_on_unrecognized_shape():
    with patch.object(_online, "_opener_no_proxy", lambda: _patched_probe({"weird": "shape"})()):
        assert probe_models("http://x") == []
