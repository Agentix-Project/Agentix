"""Deployment plugin axis — the one remaining entry-point-discovered surface."""

from __future__ import annotations

import pytest


def test_deployment_register_and_load(monkeypatch):
    from agentix.deployment.base import (
        Deployment,
        deployments,
        load_deployment,
        register_deployment,
    )

    class FakeDep:
        async def create(self, config): ...  # noqa: ARG002
        async def delete(self, sandbox_id): ...  # noqa: ARG002
        async def get(self, sandbox_id): ...  # noqa: ARG002

    monkeypatch.setattr(deployments(), "_walk_entry_points", lambda: [])
    deployments().reset()
    register_deployment("fake", FakeDep)
    cls = load_deployment("fake")
    assert cls is FakeDep
    assert isinstance(FakeDep(), Deployment)


def test_deployment_unknown_name_raises(monkeypatch):
    from agentix.deployment.base import deployments, load_deployment

    monkeypatch.setattr(deployments(), "_walk_entry_points", lambda: [])
    deployments().reset()
    with pytest.raises(KeyError, match="agentix.deployment"):
        load_deployment("never-registered")
