"""SandboxProvider plugin axis — the entry-point-discovered surface."""

from __future__ import annotations

import pytest


def test_provider_register_and_resolve(monkeypatch):
    from agentix.provider.base import SandboxProvider, providers, register_provider

    class FakeProvider(SandboxProvider):
        async def create(self, config): ...  # noqa: ARG002
        async def delete(self, sandbox_id): ...  # noqa: ARG002
        async def get(self, sandbox_id): ...  # noqa: ARG002

    monkeypatch.setattr(providers(), "_walk_entry_points", lambda: [])
    providers().reset()
    register_provider("fake", FakeProvider)
    cls = providers().get("fake")
    assert cls is FakeProvider
    assert isinstance(FakeProvider(), SandboxProvider)


def test_provider_unknown_name_raises(monkeypatch):
    from agentix.provider.base import providers

    monkeypatch.setattr(providers(), "_walk_entry_points", lambda: [])
    providers().reset()
    with pytest.raises(KeyError, match="agentix.provider"):
        providers().get("never-registered")
